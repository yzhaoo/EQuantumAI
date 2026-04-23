import numpy as np
from tqdm import tqdm
import scipy.linalg as sl
from scipy.spatial import KDTree

try:
    import kwant
except ImportError:
    print(
        "⚠️ Kwant solver is not available. "
        "Install it with: conda install -c conda-forge kwant"
    )
    kwant = None


def build_system(syst, builder="kwant", **kwargs):
    if builder == "kwant":
        return kwant_builder(syst, **kwargs)
    raise ValueError("Only builder='kwant' is supported here.")


def kwant_builder(syst):
    """
    Build a Kwant system from the custom system object.

    Each Qsite is treated as one sublattice basis site in a giant unit cell.
    """

    def qshape(box):
        def ifinshape(pos):
            x, y = pos
            return box[0][0] <= x <= box[0][1] and box[1][0] <= y <= box[1][1]
        return ifinshape

    def mag_hop(to_site, from_site, phi):
        """
        Landau-gauge Peierls phase using actual Kwant positions.
        """
        xi, yi = from_site.pos
        xf, yf = to_site.pos
        dy = yf - yi
        return syst.t * np.exp(1j * 2 * np.pi * phi * xi * dy)

    # coordinates of quantum sites
    qsite_ids = list(syst.Qsites)
    qcoor = np.array([syst.sites[idx].coordinates for idx in qsite_ids])
    primi = [(coor[0], coor[1]) for coor in qcoor]

    # one giant unit-cell lattice
    lat = kwant.lattice.general([(-100, 0), (0, 100)], primi, norbs=1)
    latsites = lat.sublattices

    qsyst = kwant.Builder()

    # bounding box for shape
    qsitebox = [
        [np.min(qcoor[:, 0]), np.max(qcoor[:, 0])],
        [np.min(qcoor[:, 1]), np.max(qcoor[:, 1])],
    ]

    # map:
    # original system site id -> local q index
    qsite_idx_map = {site_idx: q_idx for q_idx, site_idx in enumerate(qsite_ids)}

    # map:
    # kwant sublattice family -> original system site id
    family_to_siteid = {
        latsites[q_idx]: qsite_ids[q_idx]
        for q_idx in range(len(qsite_ids))
    }

    def onsite_pot(site, Ufunc):
        """
        Convert Kwant Site back to original site object before calling Ufunc.
        """
        site_id = family_to_siteid[site.family]
        original_site = syst.sites[site_id]
        return Ufunc(original_site)

    # add all sites in the finite region
    qsyst[lat.shape(qshape(qsitebox), (0, 0))] = onsite_pot

    # add hoppings
    for idx in tqdm(qsite_ids):
        site = syst.sites[idx]
        qidx = qsite_idx_map[idx]

        for idxn in site.neighbors.keys():
            if idxn not in qsite_idx_map:
                continue

            neighbor = syst.sites[idxn]

            # keep this condition consistent with default solver
            if abs(neighbor.coordinates[2]) < 1e-4 and neighbor.material == "Qsystem":
                qidxn = qsite_idx_map[idxn]
                qsyst[kwant.builder.HoppingKind((0, 0), latsites[qidx], latsites[qidxn])] = mag_hop

    fsyst = qsyst.finalized()

    # attach useful metadata for later mapping/debugging
    fsyst._family_to_siteid = family_to_siteid
    fsyst._qsite_ids = qsite_ids
    fsyst._qsite_idx_map = qsite_idx_map

    return fsyst


def kwant_site_map_from_Qsites(fsc, syst):
    """
    Build mapping:
        local index in Qsites -> index in finalized Kwant sites
    """
    kcoord = np.array([site.pos for site in syst.qsystem.sites])
    qcoord = np.array([syst.sites[idx].coordinates[:2] for idx in syst.Qsites])

    tree = KDTree(kcoord)
    threshold = 1e-8

    mapping = {}
    for i, coord in enumerate(qcoord):
        dist, idx = tree.query(coord)
        if dist < threshold:
            mapping[i] = idx
        else:
            mapping[i] = idx

    fsc.Qsites_map = mapping


def kwant_update_Ufunc(fsc, syst):
    """
    Build a Kwant-compatible Ufunc by mapping finalized Kwant Site -> original site.
    """
    family_to_siteid = getattr(syst.qsystem, "_family_to_siteid", None)

    if family_to_siteid is None:
        raise RuntimeError("Kwant system is missing family_to_siteid mapping.")

    def Ufunc(site):
        site_id = family_to_siteid[site.family]
        return -fsc.Ui[site_id]

    fsc.qparams["Ufunc"] = Ufunc


def kwant_density_ED(fsc):
    """
    Density from exact diagonalization of finalized Kwant Hamiltonian.
    """
    k_to_q_map = np.argsort(list(fsc.Qsites_map.values()))
    ham_mat = fsc.qsystem.hamiltonian_submatrix(params=fsc.qparams)
    ew, ev = sl.eigh(ham_mat)
    sort_idx = np.argsort(ew)

    qnden = np.zeros(len(fsc.Qsites_map))

    for eidx, ee in enumerate(ew):
        if ee < 0:
            kdos = np.abs(ev[sort_idx[eidx], :]) ** 2
            qnden += kdos[k_to_q_map]

    return qnden


def kwant_ildos_kpm(fsc, **kwargs):
    """
    Total spectral density from Kwant KPM.
    """
    spectrum = kwant.kpm.SpectralDensity(
        fsc.qsystem,
        params=fsc.qparams,
        energy_resolution=0.02,
        **kwargs,
    )
    return spectrum()