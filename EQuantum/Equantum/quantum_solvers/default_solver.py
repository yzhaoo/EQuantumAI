import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from joblib import Parallel, delayed
import scipy.linalg as sla
from tqdm import tqdm
from .kpm_solver import kpm_dos, kpm_ldos, kpm_dos_many, kpm_dos_from_ldos


class QuantumSystem:

    def __init__(self, syst, qparams={'Ufunc': lambda x: 0, 'phi': 0.}):

        # --- global system ---
        self.all_sites = syst.sites
        self.all_Qsites = np.array(syst.Qsites, dtype=int)

        # --- mapping: site_id → full index ---
        self.full_map = {
            sid: i for i, sid in enumerate(self.all_Qsites)
        }

        # --- active region (initially full) ---
        self.Qsite_ids = self.all_Qsites.copy()
        self.q_to_Q_map = {
            sid: i for i, sid in enumerate(self.Qsite_ids)
        }

        self.qparams = qparams.copy()

        self.lattice_type = syst.geometry_params['lattice_type']
        self.lat_spacing = syst.lat_spacing
        self.t = syst.t

        # --- build base Hamiltonian ---
        self.build_H0()

        # --- active Hamiltonian ---
        self.H = None
        self.build_active_H()
        

    # -------------------------------------------------
    # Build base Hamiltonian (ONLY hopping)
    # -------------------------------------------------
    def build_H0(self):
        phi = self.qparams['phi']
        lat_spacing=self.lat_spacing

        N = len(self.all_Qsites)

        self.H0 = lil_matrix((N, N), dtype=np.complex128)

        hop_func = mag_hop_square if self.lattice_type == "square" else mag_hop_honeycomb

        for i, site_id in enumerate(self.all_Qsites):
            site = self.all_sites[site_id]

            for j in site.neighbors:
                if j not in self.full_map:
                    continue

                neighbor = self.all_sites[j]
                j_idx = self.full_map[j]

                hop_val = hop_func(self.t, site, neighbor,
                                   phi,
                                   lat_spacing)

                self.H0[i, j_idx] = hop_val
                self.H0[j_idx, i] = np.conjugate(hop_val)

        self.H0 = self.H0.tocsr()

    # -------------------------------------------------
    # Build active Hamiltonian from H0
    # -------------------------------------------------
    def build_active_H(self):

        Ufunc = self.qparams['Ufunc']

        # --- indices in full system ---
        idx = np.array([self.full_map[sid] for sid in self.Qsite_ids], dtype=int)

        # --- slice H0 ---
        H = self.H0[idx][:, idx].tolil()

        # --- add onsite ---
        for i, sid in enumerate(self.Qsite_ids):
            site = self.all_sites[sid]
            H[i, i] += Ufunc(site)

        self.H = H.tocsr()

        # --- update mapping ---
        self.q_to_Q_map = {
            sid: i for i, sid in enumerate(self.Qsite_ids)
        }

        self.N = len(self.Qsite_ids)

    # -------------------------------------------------
    # Update active region (Qprime)
    # -------------------------------------------------
    def update_active_sites(self, qsite_ids):

        self.Qsite_ids = np.array(qsite_ids, dtype=int)

        self.build_active_H()

    # -------------------------------------------------
    # Update parameters (ONLY rebuild onsite)
    # -------------------------------------------------
    def update_qparams(self, new_qparams):
        """
        General parameter update.
        Always rebuild everything to guarantee correctness.
        """

        self.qparams = {**self.qparams, **new_qparams}

        print("🔄 Full Hamiltonian rebuild (qparams change)")

        # rebuild hopping (depends on phi etc.)
        self.build_H0()

        # rebuild active Hamiltonian
        self.build_active_H()

    # -------------------------------------------------
    def update_U(self, fsc):
        """
        Fast update: only onsite potential changes.
        """

        def Ufunc(site):
            return -fsc.Ui[site.id]

        # update locally without touching H0
        self.qparams["Ufunc"] = Ufunc

        # only rebuild active Hamiltonian (cheap)
        self.build_active_H()

        # keep FSC in sync
        fsc.qparams = dict(self.qparams)

    # -------------------------------------------------
    def get_hamiltonian(self):
        return self.H

    # -------------------------------------------------
    def get_energy_bounds(self, margin=0.05):
        H = self.get_hamiltonian().tocsr()

        diag = H.diagonal().real
        off_abs_sum = np.abs(H).sum(axis=1).A.ravel() - np.abs(diag)

        emin = np.min(diag - off_abs_sum)
        emax = np.max(diag + off_abs_sum)

        width = emax - emin
        pad = margin * width
        return emin - pad, emax + pad

        # H = self.get_hamiltonian()

        # try:
        #     emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
        #     emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        # except Exception:
        #     abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
        #     bound = abs_rowsum.max()
        #     emin, emax = -bound, bound

        # width = emax - emin
        # pad = margin * width

        # return emin - pad, emax + pad
    
    # --------------------------------------------------

    def get_hamiltonian(self):
        return self.H.tocsr()

    def get_dos(self, qparams=None, i=None, w=None, M=512, n_random=10, **kwargs):
        """
        DOS / LDOS from KPM.

        Parameters
        ----------
        qparams : dict or None
            Optional updated Hamiltonian parameters.
        i : None, int, or sequence of ints
            - None: total DOS via stochastic KPM
            - int: LDOS at one site
            - sequence: average LDOS over selected sites
        w : array or None
            Energy grid
        M : int
            Number of Chebyshev moments
        n_random : int
            Number of random vectors for total DOS
        kwargs :
            Forwarded to kpm_solver, e.g. eps=0.05, kernel="jackson", rng=1234
        """
        if qparams is not None:
            self.update_qparams(qparams)

        H = self.get_hamiltonian()

        if i is None:
            return kpm_dos(
                H,
                energies=w,
                num_moments=M,
                num_vectors=n_random,
                **kwargs,
            )

        if np.isscalar(i):
            return kpm_ldos(
                H,
                site_index=int(i),
                energies=w,
                num_moments=M,
                **kwargs,
            )

        site_indices = [int(ii) for ii in i]
        results = kpm_dos_many(
            H,
            site_indices=site_indices,
            energies=w,
            num_moments=M,
            **kwargs,
        )
        avg = np.mean([rho for _, rho in results], axis=0)
        return np.asarray(w, dtype=float), avg

    def get_ldos(self, fsc, qparams=None, approx="TF", Ncore=0, M=512, n_random=8, **kwargs):
        bounds = self.get_energy_bounds(margin=0)
        nE_local = max(256, int(self.N / 2))
        Erange = np.linspace(bounds[0],bounds[1], nE_local)

        if qparams is not None:
            self.update_qparams(qparams)

        if approx == "TF":
            print("LDOS is calculated through kpm as a whole system.")
            bulk_E, bulk_rho = self.get_dos(w=Erange, M=M, n_random=n_random, bounds=bounds,**kwargs)
            bulk_proj = project_ldos_to_global_grid(
                bulk_E, bulk_rho, fsc.E_global, fsc.max_fill
            )
            dataall = np.repeat(
                bulk_proj[None, :, :],
                self.N,
                axis=0
            )

        elif approx == "kmeanssample":
            print("LDOS is caculated through kmeans sampling.")
            # already normalized per cluster inside sample_ldos
            dataall = self.sample_ldos(fsc, Ncore=Ncore, M=M, bounds=bounds, **kwargs)
        elif approx == "ED":
            print("LDOS is calculated with ED method.")
            H = self.get_hamiltonian()

            eta = kwargs.pop("eta", 0.00015)
            broadening = kwargs.pop("broadening", "gaussian")

            E_ed, rho_all = ed_ldos(
                H,
                energies=Erange,
                eta=eta,
                broadening=broadening
            )

            dataall = np.stack(
                [
                    project_ldos_to_global_grid(E_ed, rho_all[ii], fsc.E_global, fsc.max_fill)
                    for ii in range(self.N)
                ],
                axis=0
            )
        else:
            print("No approximation method has been selected, use exact kpm by default.")
            H = self.get_hamiltonian()

            if Ncore > 1:
                raw = Parallel(n_jobs=Ncore)(
                    delayed(kpm_ldos)(
                        H,
                        site_index=ii,
                        energies=Erange,
                        num_moments=M,
                        bounds=bounds,
                        **kwargs
                    )
                    for ii in range(self.N)
                )
            else:
                raw = []
                for ii in tqdm(range(self.N)):
                    raw.append(
                        kpm_ldos(
                            H,
                            site_index=ii,
                            energies=Erange,
                            num_moments=M,
                            bounds=bounds,
                            **kwargs
                        )
                    )

            dataall = np.stack(
                [
                    project_ldos_to_global_grid(E, rho, fsc.E_global, fsc.max_fill)
                    for E, rho in raw
                ],
                axis=0
            )

        return dataall

    def sample_ldos(
        self,
        fsc,
        num_sample=20,
        Ncore=1,
        M=512,
        u_weight=2.0,
        return_groups=False,
        **kwargs
    ):
        # check if Kmeans is installed
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "some_function requires scikit-learn (KMeans). "
                "Install it with: pip install scikit-learn"
            )

        # -----------------------------
        # Adaptive KPM energy window
        # -----------------------------
        emin, emax = self.get_energy_bounds()
        nE_local = max(256, int(self.N / 2))
        Erange = np.linspace(emin, emax, nE_local)

        # -----------------------------
        # Active region (Qprime)
        # -----------------------------
        qprime_ids = np.array(fsc.Qprime, dtype=int)

        coords = np.array(
            [fsc.sites[idx].coordinates[:2] for idx in qprime_ids],
            dtype=float
        )

        Uvals = np.array([fsc.Ui[idx] for idx in qprime_ids], dtype=float)

        num_sample = min(num_sample, len(qprime_ids))

        # -----------------------------
        # Feature construction
        # -----------------------------
        def safe_standardize(arr):
            s = arr.std()
            if s < 1e-14:
                return np.zeros_like(arr)
            return (arr - arr.mean()) / s

        xz = safe_standardize(coords[:, 0])
        yz = safe_standardize(coords[:, 1])
        uz = safe_standardize(Uvals) * u_weight

        features = np.column_stack([xz, yz, uz])

        # -----------------------------
        # KMeans clustering
        # -----------------------------
        kmeans = KMeans(n_clusters=num_sample, n_init=10)
        labels = kmeans.fit_predict(features)

        site_in_b = []
        for k in range(num_sample):
            idx = np.where(labels == k)[0]
            if len(idx) > 0:
                site_in_b.append(idx)

        H = self.get_hamiltonian()

        # -----------------------------
        # Helper: project to global grid
        # -----------------------------
        def project(E_local, rho_local):
            rho_global = np.interp(
                fsc.E_global,
                E_local,
                rho_local,
                left=0.0,
                right=0.0
            )

            norm = np.trapz(rho_global, fsc.E_global)
            if norm > 0:
                rho_global = rho_global / norm * fsc.max_fill

            return np.array([fsc.E_global, rho_global], dtype=float)

        # -----------------------------
        # Compute LDOS per cluster
        # -----------------------------
        def calculate_ldos(indices):

            # choose representative site (closest to cluster center)
            cluster_features = features[indices]
            center = cluster_features.mean(axis=0)

            local_idx = np.argmin(
                np.linalg.norm(cluster_features - center, axis=1)
            )

            # 🔥 IMPORTANT: convert index → site_id
            rep_site = qprime_ids[indices[local_idx]]

            # 🔥 map site_id → local Hamiltonian index
            qidx = self.q_to_Q_map[rep_site]

            # KPM
            E, rho = kpm_ldos(
                H,
                site_index=qidx,
                energies=Erange,
                num_moments=M,
                **kwargs
            )

            return project(E, rho)

        # -----------------------------
        # Parallel / serial execution
        # -----------------------------
        if Ncore > 1:
            bin_ldos = Parallel(n_jobs=Ncore)(
                delayed(calculate_ldos)(indices)
                for indices in site_in_b
            )
        else:
            bin_ldos = [
                calculate_ldos(indices)
                for indices in site_in_b
            ]

        # -----------------------------
        # Broadcast cluster LDOS
        # -----------------------------
        dataall = []
        for bidx, indices in enumerate(site_in_b):
            n = len(indices)
            dataall.append(
                np.repeat(bin_ldos[bidx][None, :, :], n, axis=0)
            )

        # -----------------------------
        # Restore Qprime ordering
        # -----------------------------
        flat_indices = np.concatenate(site_in_b)
        sortidx = np.argsort(flat_indices)

        ldos_out = np.concatenate(dataall, axis=0)[sortidx]

        # -----------------------------
        # Optional debug info
        # -----------------------------
        if return_groups:
            rep_sites = []
            for indices in site_in_b:
                cluster_features = features[indices]
                center = cluster_features.mean(axis=0)
                local_idx = np.argmin(
                    np.linalg.norm(cluster_features - center, axis=1)
                )
                rep_sites.append(qprime_ids[indices[local_idx]])

            group_info = {
                "labels": labels,
                "site_in_b": site_in_b,
                "rep_sites": np.array(rep_sites, dtype=int),
                "coords": coords,
            }

            return ldos_out, group_info

        return ldos_out
    


        

def project_ldos_to_global_grid(E_local, rho_local, E_global, max_fill):
        E_local = np.asarray(E_local, dtype=float)
        rho_local = np.asarray(rho_local, dtype=float)
        E_global = np.asarray(E_global, dtype=float)

        rho_global = np.interp(
            E_global,
            E_local,
            rho_local,
            left=0.0,
            right=0.0
        )

        norm = np.trapz(rho_global, E_global)
        if norm > 0:
            rho_global = rho_global / norm * max_fill

        return np.array([E_global, rho_global], dtype=float)
        

        
        

        

def mag_hop_square(t,to_site,from_site,phi,lat_spacing):
    coord_i = np.array(from_site.coordinates) / lat_spacing
    coord_f = np.array(to_site.coordinates) / lat_spacing

    dx = coord_f[0] - coord_i[0]
    dy = coord_f[1] - coord_i[1]
    x = coord_i[0]

    return t * np.exp(1j * 2 * np.pi * phi * x * dy)
    #coord_i=np.array(from_site.coordinates)/lat_spacing
    #coord_f=np.array(to_site.coordinates)/lat_spacing
    #dx=(coord_f[0]- coord_i[0])
    #ydirection=np.sign(coord_f[1]-coord_i[1])
    #nx=coord_i[0]
    #return t*np.exp(1j*2*np.pi*phi*nx*ydirection)

import numpy as np

def mag_hop_honeycomb(t, to_site, from_site, phi, lat_spacing):
    ri = np.asarray(from_site.coordinates[:2], dtype=float) / lat_spacing
    rf = np.asarray(to_site.coordinates[:2], dtype=float) / lat_spacing

    xi, yi = ri
    xf, yf = rf
    dy = yf - yi

    return t * np.exp(1j * 2 * np.pi * phi * xi * dy)



def onsite_pot(site,Ufunc):
        return Ufunc(site)


def ed_ldos(H, energies, eta=0.02, broadening="gaussian"):
    """
    Exact-diagonalization LDOS for all sites.

    Parameters
    ----------
    H : sparse or dense matrix
        Hamiltonian of shape (N, N)
    energies : 1D array
        Energy grid
    eta : float
        Broadening width
    broadening : {"gaussian", "lorentzian"}
        Type of spectral broadening

    Returns
    -------
    E : ndarray, shape (nE,)
    rho_all : ndarray, shape (N, nE)
        LDOS for every site on the energy grid
    """
    if hasattr(H, "toarray"):
        H = H.toarray()
    else:
        H = np.asarray(H)

    evals, evecs = sla.eigh(H)
    energies = np.asarray(energies, dtype=float)

    weights = np.abs(evecs) ** 2   # shape (N, Nstates)

    dE = energies[None, :] - evals[:, None]   # shape (Nstates, nE)

    if broadening == "gaussian":
        kernel = np.exp(-0.5 * (dE / eta) ** 2) / (np.sqrt(2 * np.pi) * eta)
    elif broadening == "lorentzian":
        kernel = (eta / np.pi) / (dE**2 + eta**2)
    else:
        raise ValueError("broadening must be 'gaussian' or 'lorentzian'")

    # rho_all[i, E] = sum_n |psi_n(i)|^2 * kernel_n(E)
    rho_all = weights @ kernel   # shape (N, nE)

    return energies, rho_all