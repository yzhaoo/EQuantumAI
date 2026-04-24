import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar


def _pick_bulk_qsite(fsc, rng=None):
    qprime = np.asarray(fsc.Qprime, dtype=int)
    if qprime.size == 0:
        raise ValueError("No active quantum sites are available for the manual boundary check.")

    coords = np.array([fsc.sites[int(site_id)].coordinates[:2] for site_id in qprime], dtype=float)
    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    bulk_cutoff = np.quantile(distances, 0.35) if distances.size > 1 else distances[0]
    candidate_indices = np.where(distances <= bulk_cutoff)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.arange(qprime.size)

    generator = rng if rng is not None else np.random.default_rng(0)
    qidx = int(generator.choice(candidate_indices))
    return qidx, int(qprime[qidx])


def generate_manual_boundary_check(fsc, artifact_dir, rng=None):
    qidx, site_id = _pick_bulk_qsite(fsc, rng=rng)
    sidx_candidates = np.where(np.asarray(fsc.Qsites, dtype=int) == site_id)[0]
    if sidx_candidates.size == 0:
        raise ValueError(f"Could not locate site {site_id} inside fsc.Qsites.")

    sidx = int(sidx_candidates[0])
    ni = float(fsc.ni[site_id])
    Ci = float(fsc.Ci[qidx])
    ildos = fsc.ildos[sidx]
    Ui = float(fsc.Ui[site_id])

    x_dis = np.asarray(ildos[0], dtype=float) - Ui
    y_dis = np.asarray(ildos[1], dtype=float)
    order = np.argsort(x_dis)
    x_dis = x_dis[order]
    y_dis = y_dis[order]

    ildos_dis = np.zeros_like(y_dis, dtype=float)
    ildos_dis[1:] = np.cumsum(0.5 * (y_dis[1:] + y_dis[:-1]) * np.diff(x_dis))

    ildos_interp = interp1d(
        x_dis,
        ildos_dis,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    def dn_for_Ci(dU):
        return dU * Ci + ni

    def F(dU):
        return ni + Ci * dU - float(ildos_interp(dU))

    limits = (float(np.min(x_dis)), float(np.max(x_dis)))
    a, b = limits

    try:
        Fa, Fb = F(a), F(b)
        if np.isfinite(Fa) and np.isfinite(Fb) and Fa * Fb <= 0:
            dUsol = float(brentq(F, a, b))
        else:
            res = minimize_scalar(lambda u: abs(F(u)), bounds=limits, method="bounded")
            dUsol = float(res.x)
    except Exception:
        res = minimize_scalar(lambda u: abs(F(u)), bounds=limits, method="bounded")
        dUsol = float(res.x)

    mismatch = abs(F(dUsol))
    ci_shift = Ci * dUsol

    plot_path = os.path.join(artifact_dir, "manual_boundary_check.png")
    meta_path = os.path.join(artifact_dir, "manual_boundary_check.json")

    plt.figure(figsize=(8, 5))
    plt.plot(x_dis, [dn_for_Ci(dU) for dU in x_dis], label="Poisson prediction")
    plt.plot(x_dis, ildos_dis, label="Integrated LDOS")
    plt.axvline(dUsol, color="#2563eb", linestyle="--", linewidth=1.0, label=f"dU solution = {dUsol:.4g}")
    plt.xlabel(r"$\Delta U$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    payload = {
        "plot_path": plot_path,
        "meta_path": meta_path,
        "site_id": site_id,
        "qidx": qidx,
        "sidx": sidx,
        "dU_solution": dUsol,
        "ci_times_dU": ci_shift,
        "ni": ni,
        "Ci": Ci,
        "Ui": Ui,
        "mismatch": mismatch,
        "limits": [a, b],
        "message": (
            f"Manual boundary check for site {site_id}: "
            f"dU={dUsol:.4g}, Ci*dU={ci_shift:.4g}, mismatch={mismatch:.4g}."
        ),
    }

    with open(meta_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    return payload
