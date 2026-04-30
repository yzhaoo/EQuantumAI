from __future__ import annotations

import json
import os
from datetime import datetime

MPL_CACHE_DIR = os.path.join("/tmp", "equantum_mpl_cache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)

import matplotlib

matplotlib.use(os.environ.get("EQUANTUM_MPL_BACKEND", "Agg"))

import matplotlib.pyplot as plt
import numpy as np


def make_artifact_dir(profile: dict, base_output_dir: str | None = None) -> str:
    root = base_output_dir or os.path.join(profile["setup_root"], "agent_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(root, timestamp)
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def save_query_spec(artifact_dir: str, spec: dict) -> str:
    path = os.path.join(artifact_dir, "query_spec.json")
    with open(path, "w") as handle:
        json.dump(spec, handle, indent=2)
    return path


def save_run_summary(artifact_dir: str, result: dict) -> str:
    path = os.path.join(artifact_dir, "run_summary.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    return path


def summarize_run(fsc, spec: dict, phi: float, artifact_dir: str) -> dict:
    return {
        "query": spec["raw_query"],
        "task": spec["task"],
        "profile": spec["profile"],
        "lattice_type": spec["lattice_type"],
        "device_shape": spec.get("device_shape"),
        "backgate_voltage_V": spec["backgate_voltage"],
        "magnetic_field_T": spec["magnetic_field_T"],
        "phi": phi,
        "phi_to_B_roundtrip_T": fsc.phi_to_B(),
        "dielectric_constant": spec.get("dielectric_constant"),
        "gate_potential": spec.get("gate_potential"),
        "convergence_tol": spec.get("convergence_tol"),
        "Ncore": spec.get("Ncore"),
        "eta": spec.get("eta"),
        "ldos_method": spec.get("ldos_method"),
        "qsites": int(len(fsc.Qsites)),
        "qprime": int(len(fsc.Qprime)),
        "artifact_dir": artifact_dir,
        "timestamp": datetime.now().isoformat(),
    }


def save_dos_artifacts(artifact_dir: str, energy, rho) -> dict[str, str]:
    np.savez(
        os.path.join(artifact_dir, "dos_data.npz"),
        energy=np.asarray(energy, dtype=float),
        dos=np.asarray(rho, dtype=float),
    )

    plt.figure(figsize=(7, 4))
    plt.plot(energy, rho, lw=1.5)
    plt.xlabel("Energy")
    plt.ylabel("DOS")
    plt.title("Density of States")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "dos.png"), dpi=180)
    plt.close()

    return {
        "plot": os.path.join(artifact_dir, "dos.png"),
        "data": os.path.join(artifact_dir, "dos_data.npz"),
    }


def save_ldos_artifacts(artifact_dir: str, fsc, site_mode: str = "center") -> tuple[int, dict[str, str]]:
    coords = np.array([fsc.sites[idx].coordinates[:2] for idx in fsc.Qsites], dtype=float)
    if site_mode == "center":
        center_idx = int(np.argmin(np.linalg.norm(coords, axis=1)))
    else:
        center_idx = 0

    site_id = int(fsc.Qsites[center_idx])
    energy = np.asarray(fsc.ildos[center_idx, 0, :], dtype=float)
    rho = np.asarray(fsc.ildos[center_idx, 1, :], dtype=float)

    np.savez(
        os.path.join(artifact_dir, "ldos_data.npz"),
        site_id=site_id,
        energy=energy,
        ldos=rho,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(energy, rho, lw=1.5)
    plt.xlabel("Energy")
    plt.ylabel("LDOS")
    plt.title(f"Local Density of States at site {site_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "ldos.png"), dpi=180)
    plt.close()

    return site_id, {
        "plot": os.path.join(artifact_dir, "ldos.png"),
        "data": os.path.join(artifact_dir, "ldos_data.npz"),
    }
