import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brentq
from joblib import Parallel, delayed


def local_solver_i(idx, ildos, Ci, ni, Ui, limits=None, alpha=1):
    """
    Solve the local electrostatic consistency condition for one site.

    Parameters
    ----------
    idx : int
        Local index in Qprime.
    ildos : array-like
        Expected shape [2, nE], where
        ildos[0] = energy grid
        ildos[1] = LDOS / integrated LDOS-like quantity on that grid
    Ci : float
        Local capacitance coefficient.
    ni : float
        Current density.
    Ui : float
        Current local potential.
    limits : tuple or None
        Optional bounds for dU minimization. If None, infer from ildos energy axis.

    Returns
    -------
    dUsol, dnsol : float, float
    """
    # local energy axis relative to Ui
    x_dis = np.asarray(ildos[0], dtype=float) - Ui
    y_dis = np.asarray(ildos[1], dtype=float)

    # cumulative integrated LDOS
    ildos_dis = np.zeros_like(y_dis, dtype=float)
    ildos_dis[1:] = np.cumsum(0.5 * (y_dis[1:] + y_dis[:-1]) * np.diff(x_dis))

    ildos_interp = interp1d(
        x_dis,
        ildos_dis,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    def F(dU):
        return ni + Ci * dU - ildos_interp(dU)

    if limits is None:
        limits = (float(np.min(x_dis)), float(np.max(x_dis)))

    a, b = limits

    try:
        Fa, Fb = F(a), F(b)
        if np.isfinite(Fa) and np.isfinite(Fb) and Fa * Fb <= 0:
            dUsol = brentq(F, a, b)
        else:
            res = minimize_scalar(lambda u: abs(F(u)), bounds=limits, method="bounded")
            dUsol = res.x
    except Exception:
        res = minimize_scalar(lambda u: abs(F(u)), bounds=limits, method="bounded")
        dUsol = res.x


    # mixed update
    dUsol *= alpha
    dnsol = Ci * dUsol

    # def dn_for_Ci(dU):
    #     return dU * Ci + ni

    # def diff(dU):
    #     return np.abs(dn_for_Ci(dU) - ildos_interp(dU))

    # if limits is None:
    #     xmin = float(np.min(x_dis))
    #     xmax = float(np.max(x_dis))
    #     if xmin == xmax:
    #         return 0.0, 0.0
    #     limits = (xmin, xmax)

    # try:
    #     result = minimize_scalar(diff, bounds=limits, method="bounded")
    # except ValueError:
    #     print(f"local_solver_i failed at idx={idx}")
    #     raise

    # dUsol = result.x
    # dnsol = dn_for_Ci(dUsol) - ni

    return dUsol, dnsol


def local_solver(fsc, **kwarg):
    dUs = np.zeros(len(fsc.Qprime))
    dns = np.zeros(len(fsc.Qprime))

    if fsc.Ncore == 1:
        for ii in range(len(fsc.Qprime)):
            Uii = fsc.Ui[fsc.Qprime][ii]
            ildos_i = fsc.ildos[fsc.Qp_in_Q[ii]]

            # infer bounds directly from the LDOS energy axis
            x_dis = np.asarray(ildos_i[0], dtype=float) - Uii
            elimit = (float(np.min(x_dis)), float(np.max(x_dis)))

            dU, dn = local_solver_i(
                ii,
                ildos_i,
                fsc.Ci[ii],
                fsc.ni[fsc.Qprime][ii],
                Uii,
                elimit,
                **kwarg
            )
            dUs[ii] = dU
            dns[ii] = dn

    else:
        Uis = fsc.Ui[fsc.Qprime]
        Qp_in_Q_map = fsc.Qp_in_Q.copy()
        ildos_old = fsc.ildos.copy()
        Cis = fsc.Ci.copy()
        nis = fsc.ni[fsc.Qprime]

        def get_ldos(ii):
            Uii = Uis[ii]
            ildos_i = ildos_old[Qp_in_Q_map[ii]]

            x_dis = np.asarray(ildos_i[0], dtype=float) - Uii
            elimit = (float(np.min(x_dis)), float(np.max(x_dis)))

            dU, dn = local_solver_i(
                ii,
                ildos_i,
                Cis[ii],
                nis[ii],
                Uii,
                elimit,
                **kwarg
            )
            return dU, dn

        dataall = np.array(
            Parallel(n_jobs=fsc.Ncore)(
                delayed(get_ldos)(ii) for ii in range(len(fsc.Qprime))
            )
        )
        dUs = dataall[:, 0]
        dns = dataall[:, 1]

    return [dUs, dns]


def update_Qprime(fsc, tol=0):
    Qprime = fsc.Qprime.copy()

    remove_idx = []
    for idx, qsite in enumerate(fsc.Qprime):
        if fsc.ni[qsite] < tol or fsc.ni[qsite] > 0.95 * fsc.max_fill:
            remove_idx.append(idx)

    if remove_idx != []:
        fsc.N_indices = np.array(
            sorted(set(np.append(fsc.N_indices, np.array(Qprime)[remove_idx]))),
            dtype=int
        )
        fsc.D_indices = np.array(
            sorted(set(range(fsc.num_sites)) - set(fsc.N_indices)),
            dtype=int
        )

    return np.delete(Qprime, remove_idx)


def Fermi_level_pinning(fsc):
    Qprime = fsc.Qprime.copy()

    remove_idx = []
    for qidx, idx in enumerate(fsc.Qprime):
        for neighbor in fsc.sites[idx].neighbors:
            if fsc.sites[neighbor].material == 'gate':
                remove_idx.append(qidx)
                fsc.ni[idx] = fsc.max_fill
                break

    if remove_idx != []:
        fsc.N_indices = np.array(list(set(np.append(fsc.N_indices, np.array(Qprime)[remove_idx]))))
        fsc.D_indices = np.array(list(set(range(fsc.num_sites)) - set(fsc.N_indices)))

    fsc.Qprime = np.delete(Qprime, remove_idx)