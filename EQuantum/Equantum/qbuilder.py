import numpy as np
from quantum_solvers import kwant_solver as ksolver
from quantum_solvers.default_solver import QuantumSystem


def build_system(syst, builder="kwant", **kwargs):
    """
    Build the quantum system using the requested backend.

    Parameters
    ----------
    syst : object
        System object.
    builder : str
        "kwant" or "default"
    kwargs :
        Passed through to the underlying builder.

    Returns
    -------
    qsystem
        Built quantum system object.
    """
    if builder == "kwant":
        qsystem = ksolver.kwant_builder(syst, **kwargs)
        print("The quantum system has been built using the kwant solver.")
    elif builder == "default":
        qsystem = QuantumSystem(syst, **kwargs)
        print("The quantum system has been built using the default solver.")
    else:
        raise ValueError("Please provide a valid quantum builder: 'kwant' or 'default'.")
    return qsystem


def update_qparams(fsc, qparams):
    builder = fsc.quantum_builder

    fsc.qparams = {**getattr(fsc, "qparams", {}), **qparams}

    if builder == "kwant":
        pass
    elif builder == "default":
        fsc.qsystem.update_qparams(fsc.qparams)
    else:
        raise ValueError(f"Unknown quantum builder: {builder}")


def site_map(fsc, syst):
    """
    Build mapping between Qsites and Hamiltonian indices.
    """

    builder = syst.quantum_builder

    if builder == "kwant":
        ksolver.kwant_site_map_from_Qsites(fsc, syst)

    elif builder == "default":

        # index → site_id
        fsc.idx_to_qsite = {
            i: site_id for i, site_id in enumerate(fsc.Qsites)
        }

        # site_id → index (IMPORTANT)
        fsc.qsite_to_idx = {
            site_id: i for i, site_id in enumerate(fsc.Qsites)
        }

        # keep legacy name if other code uses it
        fsc.Qsites_map = fsc.idx_to_qsite

    else:
        raise ValueError(f"Unknown quantum builder: {builder}")


def update_U(fsc, syst):
    """
    Update onsite potential in the quantum backend.
    """
    fsc.local_update_count +=1
    builder = syst.quantum_builder

    if builder == "kwant":
        ksolver.kwant_update_Ufunc(fsc, syst)
    elif builder == "default":
        fsc.qsystem.update_U(fsc)
    else:
        raise ValueError(f"Unknown quantum builder: {builder}")


def update_n(fsc, syst, **kwargs):
    """
    Return DOS / density-related quantity from the backend.
    """
    builder = syst.quantum_builder

    if builder == "kwant":
        return ksolver.kwant_density_ED(fsc, **kwargs)
    elif builder == "default":
        return fsc.qsystem.get_dos(**kwargs)
    else:
        raise ValueError(f"Unknown quantum builder: {builder}")


def update_ildos(fsc, syst, **kwargs):
    """
    Compute site-resolved LDOS/IDOS-like data.
    """
    builder = syst.quantum_builder

    if builder == "kwant":
        return ksolver.kwant_ildos_kpm(fsc, **kwargs)

    elif builder == "default":
        dataall = fsc.qsystem.get_ldos(fsc, **kwargs)

        return dataall

    else:
        raise ValueError(f"Unknown quantum builder: {builder}")


def get_n_from_ildos(fsc, edos_data, sample="energy"):
    """
    Compute density from integrated LDOS/IDOS data.

    Parameters
    ----------
    fsc : object
        Self-consistent system object.
    edos_data : ndarray
        Expected shape [site, (energy,dos), energy_grid] or [site, 2, nE].
    sample : str
        Currently only "energy" is supported.

    Returns
    -------
    nden : ndarray
        Carrier density on Qprime sites.
    """
    nden = np.zeros(len(fsc.Qprime))

    if sample == "energy":
        charge_cnp = 0
        pinned_idx = np.where(fsc.Ui[fsc.Qprime] <= 0)[0]

        for ii in range(len(fsc.Qprime)):
            if ii in pinned_idx:
                continue

            site_idx = fsc.Qp_in_Q[ii]
            Ei = edos_data[site_idx, 0, :]
            rhoi = edos_data[site_idx, 1, :]

            filled = np.where(Ei <= fsc.Ui[fsc.Qprime][ii])[0]
            if len(filled) == 0:
                nden[ii] = 0.0
            else:
                filled_idx = filled[-1]
                nden[ii] = np.sum(rhoi[:filled_idx])

        return nden + charge_cnp

    raise ValueError(f"Unknown sampling mode: {sample}")