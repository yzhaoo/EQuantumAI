import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def jackson_kernel(num_moments: int) -> np.ndarray:
    """
    Jackson damping kernel for KPM.

    Parameters
    ----------
    num_moments : int
        Number of Chebyshev moments.

    Returns
    -------
    g : ndarray
        Kernel coefficients of length num_moments.
    """
    if num_moments < 1:
        raise ValueError("num_moments must be >= 1")

    n = np.arange(num_moments, dtype=float)
    N = float(num_moments)
    g = ((N - n + 1.0) * np.cos(np.pi * n / (N + 1.0)) +
         np.sin(np.pi * n / (N + 1.0)) / np.tan(np.pi / (N + 1.0))) / (N + 1.0)
    return g


def rescale_hamiltonian(H, bounds=None, eps=0.05):
    H = H.tocsr()

    if bounds is None:
        try:
            emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False, maxiter=10000)[0]
            emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False, maxiter=10000)[0]

        except Exception:
            # fallback bound estimate (Gershgorin-like)
            abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
            bound = abs_rowsum.max()

            emin = -bound
            emax = bound

    else:
        emin, emax = bounds

    a = (emax - emin) / (2.0 - eps)
    b = (emax + emin) / 2.0

    I = sp.identity(H.shape[0], format="csr", dtype=H.dtype)
    Hs = (H - b * I) / a

    return Hs, a, b, (emin, emax)

def chebyshev_moments_local(Hs, site_index: int, num_moments: int) -> np.ndarray:
    """
    Compute local Chebyshev moments:
        mu_n = <i | T_n(Hs) | i>

    Parameters
    ----------
    Hs : sparse matrix
        Rescaled Hamiltonian.
    site_index : int
        Matrix index of target site.
    num_moments : int
        Number of moments.

    Returns
    -------
    mu : ndarray
        Local moments, shape (num_moments,)
    """
    N = Hs.shape[0]
    if not (0 <= site_index < N):
        raise IndexError(f"site_index {site_index} out of bounds for size {N}")
    if num_moments < 1:
        raise ValueError("num_moments must be >= 1")

    e = np.zeros(N, dtype=np.complex128)
    e[site_index] = 1.0

    mu = np.zeros(num_moments, dtype=np.float64)

    v0 = e.copy()
    mu[0] = np.vdot(e, v0).real

    if num_moments == 1:
        return mu

    v1 = Hs @ e
    mu[1] = np.vdot(e, v1).real

    for n in range(2, num_moments):
        v2 = 2.0 * (Hs @ v1) - v0
        mu[n] = np.vdot(e, v2).real
        v0, v1 = v1, v2

    return mu


def chebyshev_moments_random_trace(
    Hs,
    num_moments: int,
    num_vectors: int = 10,
    rng=None,
) -> np.ndarray:
    """
    Stochastic trace moments for total DOS:
        mu_n ~ Tr[T_n(Hs)]

    Uses random phase vectors, matching the standard robust KPM choice.

    Parameters
    ----------
    Hs : sparse matrix
        Rescaled Hamiltonian.
    num_moments : int
        Number of moments.
    num_vectors : int
        Number of random vectors.
    rng : None, int, np.random.Generator
        Random seed or generator.

    Returns
    -------
    mu : ndarray
        Estimated trace moments.
    """
    if num_moments < 1:
        raise ValueError("num_moments must be >= 1")
    if num_vectors < 1:
        raise ValueError("num_vectors must be >= 1")

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    N = Hs.shape[0]
    mu = np.zeros(num_moments, dtype=np.float64)

    for _ in range(num_vectors):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
        r = np.exp(1j * theta) / np.sqrt(N)

        v0 = r.copy()
        mu[0] += np.vdot(r, v0).real

        if num_moments == 1:
            continue

        v1 = Hs @ r
        mu[1] += np.vdot(r, v1).real

        for n in range(2, num_moments):
            v2 = 2.0 * (Hs @ v1) - v0
            mu[n] += np.vdot(r, v2).real
            v0, v1 = v1, v2

    mu /= num_vectors
    mu *= N  # convert normalized-trace estimate to full trace
    return mu


def reconstruct_density(
    mu: np.ndarray,
    energies,
    a: float,
    b: float,
    kernel: str = "jackson",
    clip_x: bool = True,
) -> np.ndarray:
    """
    Reconstruct spectral density from Chebyshev moments.

    Parameters
    ----------
    mu : ndarray
        Chebyshev moments.
    energies : ndarray
        Energies in original units.
    a, b : float
        Rescaling parameters from Hs = (H - b I)/a
    kernel : {"jackson", None}
        Damping kernel.
    clip_x : bool
        If True, clip scaled energies away from +-1 to avoid edge blowups.

    Returns
    -------
    rho : ndarray
        Reconstructed spectral density.
    """
    mu = np.asarray(mu, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)

    num_moments = len(mu)
    if num_moments < 1:
        raise ValueError("mu must have length >= 1")

    if kernel == "jackson":
        mu_eff = jackson_kernel(num_moments) * mu
    elif kernel is None:
        mu_eff = mu
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    rho = np.zeros_like(energies, dtype=np.float64)

    for k, E in enumerate(energies):
        x = (E - b) / a

        if abs(x) >= 1.0:
            continue   # leave rho[k] = 0

        # only stabilize valid region
        if clip_x:
            x = np.clip(x, -1 + 1e-12, 1 - 1e-12)

        T0 = 1.0
        s = mu_eff[0]

        if num_moments > 1:
            T1 = x
            s += 2.0 * mu_eff[1] * T1
        else:
            T1 = None

        for n in range(2, num_moments):
            T2 = 2.0 * x * T1 - T0
            s += 2.0 * mu_eff[n] * T2
            T0, T1 = T1, T2

        rho[k] = s / (np.pi * np.sqrt(1.0 - x * x) * a)

    return rho


def default_energy_grid(bounds, num_moments: int) -> np.ndarray:
    """
    Kwant-like default energy grid: length 2 * num_moments over [emin, emax].
    """
    emin, emax = bounds
    return np.linspace(emin, emax, 2 * num_moments)


def kpm_ldos(
    H,
    site_index: int,
    energies,
    num_moments: int = 512,
    bounds=None,
    eps: float = 0.05,
    kernel: str = "jackson",
    clip_x: bool = True,
):
    """
    LDOS at one site from deterministic KPM.
    """
    Hs, a, b, bounds = rescale_hamiltonian(H, bounds=bounds, eps=eps)
    mu = chebyshev_moments_local(Hs, int(site_index), num_moments)
    rho = reconstruct_density(mu, energies, a, b, kernel=kernel, clip_x=clip_x)
    return np.asarray(energies, dtype=np.float64), rho


def kpm_ldos_many(
    H,
    site_indices,
    energies,
    num_moments: int = 512,
    bounds=None,
    eps: float = 0.05,
    kernel: str = "jackson",
    clip_x: bool = True,
):
    """
    LDOS for many sites using one shared rescaling.
    """
    Hs, a, b, bounds = rescale_hamiltonian(H, bounds=bounds, eps=eps)

    results = []
    for site_index in site_indices:
        mu = chebyshev_moments_local(Hs, int(site_index), num_moments)
        rho = reconstruct_density(mu, energies, a, b, kernel=kernel, clip_x=clip_x)
        results.append((np.asarray(energies, dtype=np.float64), rho))
    return results


def kpm_dos_many(
    H,
    site_indices,
    energies,
    num_moments: int = 512,
    bounds=None,
    eps: float = 0.05,
    kernel: str = "jackson",
    clip_x: bool = True,
):
    """
    Backward-compatible alias for many-site LDOS calculations.

    This returns a list:
        [(energies, rho_site0), (energies, rho_site1), ...]
    """
    return kpm_ldos_many(
        H=H,
        site_indices=site_indices,
        energies=energies,
        num_moments=num_moments,
        bounds=bounds,
        eps=eps,
        kernel=kernel,
        clip_x=clip_x,
    )


def kpm_dos_from_ldos(
    H,
    energies,
    num_moments: int = 512,
    bounds=None,
    eps: float = 0.05,
    kernel: str = "jackson",
    clip_x: bool = True,
):
    """
    Total DOS obtained deterministically by summing LDOS over all sites.

    Very useful for debugging against exact diagonalization or Kwant on small systems.
    """
    Hs, a, b, bounds = rescale_hamiltonian(H, bounds=bounds, eps=eps)
    N = Hs.shape[0]
    rho = np.zeros(len(energies), dtype=np.float64)

    for site_index in range(N):
        mu = chebyshev_moments_local(Hs, site_index, num_moments)
        rho += reconstruct_density(mu, energies, a, b, kernel=kernel, clip_x=clip_x)

    return np.asarray(energies, dtype=np.float64), rho


def kpm_dos(
    H,
    energies=None,
    num_moments: int = 512,
    num_vectors: int = 10,
    bounds=None,
    eps: float = 0.05,
    kernel: str = "jackson",
    rng=None,
    clip_x: bool = True,
):
    """
    Total DOS from stochastic-trace KPM.

    Parameters
    ----------
    H : sparse matrix
        Hamiltonian.
    energies : ndarray or None
        Energy grid. If None, use a Kwant-like default grid of length 2*num_moments.
    num_moments : int
        Number of Chebyshev moments.
    num_vectors : int
        Number of random phase vectors.
    bounds : tuple or None
        Optional (emin, emax).
    eps : float
        Rescaling safety margin. Kwant-like default is 0.05.
    kernel : {"jackson", None}
        Damping kernel.
    rng : None, int, np.random.Generator
        Random seed or generator.
    clip_x : bool
        Clip scaled energies away from +-1.

    Returns
    -------
    energies, rho : ndarray, ndarray
        Energy grid and total DOS.
    """
    Hs, a, b, bounds = rescale_hamiltonian(H, bounds=bounds, eps=eps)
    mu = chebyshev_moments_random_trace(
        Hs, num_moments=num_moments, num_vectors=num_vectors, rng=rng
    )

    if energies is None:
        energies = default_energy_grid(bounds, num_moments)

    rho = reconstruct_density(mu, energies, a, b, kernel=kernel, clip_x=clip_x)
    return np.asarray(energies, dtype=np.float64), rho


def exact_dos_gaussian(H, energies, eta: float = 0.02):
    """
    Exact DOS from full diagonalization with Gaussian broadening.
    Useful for small-system debugging.
    """
    if sp.issparse(H):
        H = H.toarray()

    evals = np.linalg.eigvalsh(H)
    energies = np.asarray(energies, dtype=float)

    rho = np.zeros_like(energies, dtype=float)
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * eta)

    for ev in evals:
        rho += pref * np.exp(-0.5 * ((energies - ev) / eta) ** 2)

    return energies, rho


def exact_ldos_gaussian(H, site_index: int, energies, eta: float = 0.02):
    """
    Exact LDOS from full diagonalization with Gaussian broadening.
    Useful for small-system debugging.
    """
    if sp.issparse(H):
        H = H.toarray()

    evals, evecs = np.linalg.eigh(H)
    weights = np.abs(evecs[site_index, :]) ** 2
    energies = np.asarray(energies, dtype=float)

    rho = np.zeros_like(energies, dtype=float)
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * eta)

    for n, ev in enumerate(evals):
        rho += weights[n] * pref * np.exp(-0.5 * ((energies - ev) / eta) ** 2)

    return energies, rho