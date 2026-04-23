import os
import json
import csv
import time
import math
import inspect
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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


PROGRESS_FIELDNAMES = [
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
    "ni_reldiff",
    "ildos_meanreldiff",
    "status",
    "restart_mode",
    "worker_pid",
    "chunk_id",
    "chunk_index",
    "chunk_size",
]


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
    nworkers = int(fixed_params["Ncore"])
    total_points = int(len(phis) * len(Vbgs))
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
        "parallelization_mode": "hybrid_chunked_warm_start",
        "max_parallel_jobs": nworkers,
        "solver_cores_per_job": int(fixed_params.get("solver_Ncore", 1)),
        "expected_total_points": total_points,
        "expected_chunk_count": nworkers,
        "expected_points_per_chunk": math.ceil(total_points / nworkers) if nworkers > 0 else total_points,
    }

    with open(os.path.join(out_folder, "scan_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def append_progress_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def merge_chunk_progress_files(out_folder, chunk_ids):
    merged_csv = os.path.join(out_folder, "scan_progress.csv")
    rows = []

    for chunk_id in sorted(chunk_ids):
        chunk_csv = os.path.join(
            out_folder,
            f"_chunk_progress_{int(chunk_id):03d}.csv",
        )
        if not os.path.exists(chunk_csv):
            continue

        with open(chunk_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    def _sort_key(row):
        try:
            return (
                int(row["chunk_id"]),
                int(row["chunk_index"]),
            )
        except Exception:
            return (10**9, 10**9)

    rows.sort(key=_sort_key)

    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def reset_fsc_log(fsc):
    fsc.log = {
        "Qprime_len": [len(fsc.Qprime)],
        "Ui_maxdiff": [],
        "Ui_l2diff": [],
        "ni_maxdiff": [],
        "ni_l2diff": [],
        "ni_reldiff": [],
        "ildos_maxdiff": [],
        "ildos_l2diff":[],
        "ildos_meanreldiff":[],
        "timing_poisson": [],
        "timing_quantum": [],
        "timing_total": [],
    }


def build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params):
    qparams = {"Ufunc": lambda x: 0, "phi": phi}

    fsc = FSC_cls(syst, ifinitial=False, qparams=qparams,approx="TF")

    boundary_condition = {
        "gate": {"potential": fixed_params["gate_potential"]},
        "dielectric": {"dielectric_constant": fixed_params["dielectric_constant"]},
        "backgate": {"potential": Vbg},
    }
    fsc.update_BC(updates=boundary_condition, ifinitial=True)

    fsc.Ncore = int(fixed_params.get("solver_Ncore", 1))
    fsc.convergence_tol = fixed_params["convergence_tol"]
    reset_fsc_log(fsc)
    return fsc


def update_existing_fsc(fsc, syst, phi, Vbg):
    fsc.Qprime = fsc.Qsites.copy()
    fsc.qsystem.update_active_sites(fsc.Qprime)
    fsc.Qp_in_Q = {ii: fsc.qsite_id_to_idx[sid] for ii, sid in enumerate(fsc.Qprime)}
    fsc.N_indices = syst.N_indices.copy()
    fsc.D_indices = syst.D_indices.copy()

    fsc.update_qparams(syst, {"phi": phi}, ifinitial=False)

    boundary_condition = {
        "backgate": {"potential": Vbg},
    }
    fsc.update_BC(updates=boundary_condition, ifinitial=True)
    reset_fsc_log(fsc)


def _cleanup_point_snapshots(snapshot_folder):
    if not os.path.isdir(snapshot_folder):
        return
    for fname in os.listdir(snapshot_folder):
        if fname.endswith(".npz") and fname != "run_static.npz":
            try:
                os.remove(os.path.join(snapshot_folder, fname))
            except OSError:
                pass


def _solve_current_point(fsc, syst, snapshot_folder, fixed_params):
    os.makedirs(snapshot_folder, exist_ok=True)

    _cleanup_point_snapshots(snapshot_folder)

    before = {
        f for f in os.listdir(snapshot_folder)
        if f.endswith(".npz") and f != "run_static.npz"
    }

    fsc.solve(
        syst,
        save=True,
        snapshot_mode="final_only",
        snapshot_folder=snapshot_folder,
        save_ildos=fixed_params["save_ildos"],
        ldos_method=fixed_params["ldos_method"],
        eta=fixed_params["eta"],
        M=fixed_params["M"],
        eps=fixed_params["eps"],
        kernel=fixed_params["kernel"],
    )

    after = {
        f for f in os.listdir(snapshot_folder)
        if f.endswith(".npz") and f != "run_static.npz"
    }
    new_files = sorted(after - before)

    if len(new_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 new point snapshot in {snapshot_folder}, got {new_files}"
        )

    latest_file = new_files[0]
    src = os.path.join(snapshot_folder, latest_file)
    return src, latest_file


def _run_chunk(
    FSC_cls,
    System_cls,
    geoparams,
    config_file,
    out_folder,
    fixed_params,
    chunk_id,
    jobs,
):
    worker_pid = os.getpid()

    snapshot_folder = os.path.join(
        out_folder,
        f"_worker_tmp_chunk_{chunk_id:03d}_pid_{worker_pid}",
    )
    os.makedirs(snapshot_folder, exist_ok=True)

    chunk_csv = os.path.join(out_folder, f"_chunk_progress_{chunk_id:03d}.csv")
    if os.path.exists(chunk_csv):
        os.remove(chunk_csv)

    syst = System_cls(
        geoparams,
        config_file=config_file,
        ifqsystem=True,
        quantum_builder="default",
    )

    fsc = None
    need_fresh_fsc = True
    chunk_size = len(jobs)
    rows = []

    print(
        f"[WORKER chunk={chunk_id} pid={worker_pid}] start with {chunk_size} points, tmp={snapshot_folder}",
        flush=True,
    )

    try:
        for chunk_index, (phi, Vbg) in enumerate(jobs, start=1):
            phi = float(phi)
            Vbg = float(Vbg)
            t0 = time.perf_counter()
            status = "ok"
            converged = False
            outfile = f"phi_{phi:.4f}_Vbg_{Vbg:.4f}.npz"
            restart_mode = "fresh" if need_fresh_fsc or fsc is None else "warm"

            try:
                if need_fresh_fsc or fsc is None:
                    fsc = build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params)
                    need_fresh_fsc = False
                    restart_mode = "fresh"
                else:
                    update_existing_fsc(fsc, syst, phi, Vbg)
                    restart_mode = "warm"

                src, latest_file = _solve_current_point(
                    fsc,
                    syst,
                    snapshot_folder,
                    fixed_params,
                )

                dst = os.path.join(out_folder, outfile)
                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)

                converged = "_conv" in latest_file

            except Exception as e:
                status = f"error: {type(e).__name__}: {e}"
                need_fresh_fsc = True
                print(
                    f"[WORKER chunk={chunk_id} pid={worker_pid}] "
                    f"phi={phi:.4f}, Vbg={Vbg:.4f} FAILED: {status}",
                    flush=True,
                )

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
                "ni_reldiff": fsc.log["ni_reldiff"][-1] if (fsc is not None and len(fsc.log["ni_reldiff"])) else "",
                "ildos_meanreldiff": fsc.log["ildos_meanreldiff"][-1] if (fsc is not None and len(fsc.log["ildos_meanreldiff"])) else "",
                "status": status,
                "restart_mode": restart_mode,
                "worker_pid": worker_pid,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_size": chunk_size,
            }

            append_progress_row(chunk_csv, row)
            rows.append(row)

            print(
                f"[WORKER chunk={chunk_id} pid={worker_pid}] "
                f"[{chunk_index}/{chunk_size}] phi={phi:.4f}, Vbg={Vbg:.4f}, "
                f"restart={restart_mode}, status={status}",
                flush=True,
            )

    finally:
        shutil.rmtree(snapshot_folder, ignore_errors=True)
        print(
            f"[WORKER chunk={chunk_id} pid={worker_pid}] cleaned tmp={snapshot_folder}",
            flush=True,
        )

    return rows


def _partition_jobs(phis, Vbgs, nchunks):
    jobs = [(float(phi), float(Vbg)) for phi in phis for Vbg in Vbgs]
    raw_chunks = np.array_split(np.array(jobs, dtype=float), nchunks)
    chunks = []
    for i, arr in enumerate(raw_chunks):
        if len(arr) == 0:
            continue
        chunks.append((i, [(float(phi), float(Vbg)) for phi, Vbg in arr.tolist()]))
    return chunks


def run_scan(
    FSC_cls,
    System_cls,
    geoparams,
    config_file,
    phis,
    Vbgs,
    out_folder,
    fixed_params,
):
    os.makedirs(out_folder, exist_ok=True)

    max_workers = int(fixed_params["Ncore"])
    chunks = _partition_jobs(phis, Vbgs, max_workers)
    total_points = sum(len(jobs) for _, jobs in chunks)

    print(
        f"Launching hybrid parameter scan with {len(chunks)} workers over {total_points} points.",
        flush=True,
    )
    print(
        "Each worker gets one chunk: first point is cold-started, later points in the same chunk use warm start.",
        flush=True,
    )
    print(
        f"Internal solver cores per point: {int(fixed_params.get('solver_Ncore', 1))}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _run_chunk,
                FSC_cls,
                System_cls,
                geoparams,
                config_file,
                out_folder,
                fixed_params,
                chunk_id,
                jobs,
            ): (chunk_id, len(jobs))
            for chunk_id, jobs in chunks
        }

        completed_points = 0
        finished_chunk_ids = []

        for future in as_completed(futures):
            chunk_id, chunk_size = futures[future]
            print(f"\n=== Chunk {chunk_id} finished ({chunk_size} points) ===", flush=True)
            finished_chunk_ids.append(chunk_id)

            try:
                rows = future.result()
            except Exception as e:
                print(
                    f"chunk {chunk_id} status=error: ChunkFailure: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue

            completed_points += len([r for r in rows if r["phi"] != ""])
            print(
                f"Completed points so far: {completed_points}/{total_points}",
                flush=True,
            )

    merge_chunk_progress_files(out_folder, [chunk_id for chunk_id, _ in chunks])
    print(
        f"Merged per-chunk progress files into {os.path.join(out_folder, 'scan_progress.csv')}",
        flush=True,
    )