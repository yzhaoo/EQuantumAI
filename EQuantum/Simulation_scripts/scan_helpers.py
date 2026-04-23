import os
import json
import csv
import time
import inspect
from datetime import datetime

import numpy as np


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if callable(obj):
        return getattr(obj, "__name__", "<callable>")
    return obj


def get_function_source(func):
    try:
        return inspect.getsource(func)
    except Exception:
        return f"<source unavailable: {getattr(func, '__name__', 'anonymous')}>"


def write_scan_manifest(
    out_folder,
    density_function,
    geoparams,
    phis,
    Vbgs,
    config_file,
    fixed_params,
    script_path=None,
):
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": script_path or "<interactive>",
        "output_folder": os.path.abspath(out_folder),
        "config_file": os.path.abspath(config_file),
        "density_function_name": getattr(density_function, "__name__", "anonymous"),
        "density_function_source": get_function_source(density_function),
        "geoparams": sanitize_for_json(geoparams),
        "scan_ranges": {
            "phis": sanitize_for_json(phis),
            "Vbgs": sanitize_for_json(Vbgs),
        },
        "fixed_params": sanitize_for_json(fixed_params),
    }

    with open(os.path.join(out_folder, "scan_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def append_progress_row(out_folder, row):
    csv_path = os.path.join(out_folder, "scan_progress.csv")
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "phi",
        "Vbg",
        "outfile",
        "converged",
        "runtime_sec",
        "iter_qprime",
        "iter_poisson",
        "iter_quantum",
        "qprime_len",
        "ui_maxdiff",
        "ni_maxdiff",
        "ildos_maxdiff",
        "status",
        "restart_mode",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def reset_fsc_log(fsc):
    fsc.log = {
        "Qprime_len": [len(fsc.Qprime) if hasattr(fsc, "Qprime") else 0],
        "Ui_maxdiff": [],
        "Ui_l2diff": [],
        "ni_maxdiff": [],
        "ni_l2diff": [],
        "ildos_maxdiff": [],
        "timing_poisson": [],
        "timing_quantum": [],
        "timing_total": [],
    }


def build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params):
    qparams = {"Ufunc": lambda x: 0, "phi": phi}

    fsc = FSC_cls(syst, ifinitial=False, qparams=qparams)

    fsc.update_BC(syst, "gate", "potential", fixed_params["gate_potential"])
    fsc.update_BC(syst, "dielectric", "dielectric_constant", fixed_params["dielectric_constant"])
    fsc.update_BC(syst, "backgate", "potential", Vbg, ifinitial=True)

    fsc.Ncore = fixed_params["Ncore"]
    fsc.convergence_tol = fixed_params["convergence_tol"]

    reset_fsc_log(fsc)
    return fsc


def update_existing_fsc(fsc, syst, phi, Vbg):
    fsc.Qprime = fsc.Qsites.copy()
    fsc.qsystem.update_active_sites(fsc.Qprime)
    fsc.Qp_in_Q = {
        ii: fsc.qsite_id_to_idx[sid]
        for ii, sid in enumerate(fsc.Qprime)
    }
    fsc.N_indices=(syst.N_indices).copy()
    fsc.D_indices=(syst.D_indices).copy()
    #fsc.initial_Poisson()
    fsc.update_qparams(syst, {"phi": phi}, ifinitial=False)
    fsc.update_BC(syst, "backgate", "potential", Vbg, ifinitial=True)
    reset_fsc_log(fsc)


def run_scan(
    FSC_cls,
    syst,
    phis,
    Vbgs,
    out_folder,
    fixed_params,
):
    os.makedirs(out_folder, exist_ok=True)

    fsc = None
    need_fresh_fsc = True

    for phi in phis:
        for Vbg in Vbgs:
            phi = float(phi)
            Vbg = float(Vbg)

            print(f"\\n=== Running phi={phi:.4f}, Vbg={Vbg:.4f} ===")
            t0 = time.perf_counter()

            status = "ok"
            converged = False
            outfile = f"phi_{phi:.4f}_Vbg_{Vbg:.4f}.npz"
            restart_mode = "fresh" if need_fresh_fsc or fsc is None else "warm"

            before_files = {
                f for f in os.listdir(out_folder)
                if f.endswith(".npz") and f != "run_static.npz"
            }

            try:
                if need_fresh_fsc or fsc is None:
                    fsc = build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params)
                    need_fresh_fsc = False
                    restart_mode = "fresh"
                else:
                    update_existing_fsc(fsc, syst, phi, Vbg)
                    restart_mode = "warm"

                fsc.solve(
                    syst,
                    save=True,
                    snapshot_mode="final_only",
                    snapshot_folder=out_folder,
                    save_ildos=fixed_params["save_ildos"],
                    ldos_method=fixed_params["ldos_method"],
                    eta=fixed_params["eta"],
                    M=fixed_params["M"],
                    eps=fixed_params["eps"],
                    kernel=fixed_params["kernel"],
                )

                after_files = {
                    f for f in os.listdir(out_folder)
                    if f.endswith(".npz") and f != "run_static.npz"
                }

                new_files = sorted(after_files - before_files)

                if len(new_files) != 1:
                    raise RuntimeError(f"Expected exactly 1 new snapshot, got {new_files}")

                latest_file = new_files[0]
                src = os.path.join(out_folder, latest_file)
                dst = os.path.join(out_folder, outfile)

                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)

                converged = "_conv" in latest_file

            except RuntimeError as e:
                status = f"error: RuntimeError: {e}"
                print(status)
                need_fresh_fsc = True

            except Exception as e:
                status = f"error: {type(e).__name__}: {e}"
                print(status)
                need_fresh_fsc = True

            runtime_sec = time.perf_counter() - t0

            row = {
                "timestamp": datetime.now().isoformat(),
                "phi": phi,
                "Vbg": Vbg,
                "outfile": outfile if status == "ok" else "",
                "converged": converged,
                "runtime_sec": runtime_sec,
                "iter_qprime": len(fsc.log["Qprime_len"]) if fsc is not None else "",
                "iter_poisson": len(fsc.log["timing_poisson"]) if fsc is not None else "",
                "iter_quantum": len(fsc.log["timing_quantum"]) if fsc is not None else "",
                "qprime_len": len(fsc.Qprime) if (fsc is not None and hasattr(fsc, "Qprime")) else "",
                "ui_maxdiff": fsc.log["Ui_maxdiff"][-1] if (fsc is not None and len(fsc.log["Ui_maxdiff"])) else "",
                "ni_maxdiff": fsc.log["ni_maxdiff"][-1] if (fsc is not None and len(fsc.log["ni_maxdiff"])) else "",
                "ildos_maxdiff": fsc.log["ildos_maxdiff"][-1] if (fsc is not None and len(fsc.log["ildos_maxdiff"])) else "",
                "status": status,
                "restart_mode": restart_mode,
            }

            append_progress_row(out_folder, row)
            print(f"✔ Logged phi={phi:.4f}, Vbg={Vbg:.4f}")