from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


QUANTUM_PROPERTIES = ["Ui", "ni", "Ci", "Qprime_mask", "ΔUi", "LDOS@0", "LDOS@Ui"]
SURFACE_PROPERTIES = ["Ui", "ni", "Ci", "ΔUi"]


def _as_float_list(array) -> list[float | None]:
    arr = np.asarray(array, dtype=float).reshape(-1)
    output: list[float | None] = []
    for value in arr:
        if np.isnan(value) or np.isinf(value):
            output.append(None)
        else:
            output.append(float(value))
    return output


def _as_int_list(array) -> list[int]:
    arr = np.asarray(array, dtype=int).reshape(-1)
    return [int(value) for value in arr]


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _clean_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_scalar(item) for item in value]
    return value


def _static_from_npz(npz_data) -> dict[str, Any]:
    if "static_data" not in npz_data:
        raise KeyError("run_static.npz does not contain 'static_data'.")
    static_data = npz_data["static_data"][0]
    if not isinstance(static_data, dict):
        raise TypeError("Expected run_static['static_data'][0] to be a dict-like object.")
    return static_data


def _snapshot_file_candidates(run_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(run_dir)
    candidates = sorted(root.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    result: list[Path] = []
    for path in candidates:
        if path.name == "run_static.npz":
            continue
        if path.name in {"dos_data.npz", "ldos_data.npz"}:
            continue
        try:
            data = np.load(path, allow_pickle=True)
            files = set(data.files)
        except Exception:
            continue
        if {"Ui", "ni", "Qprime"}.issubset(files):
            result.append(path)
    return result


def load_run_static(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(run_dir) / "run_static.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_static.npz in {run_dir}")
    data = np.load(path, allow_pickle=True)
    static = _static_from_npz(data)
    return {
        "Qsites": np.asarray(static["Qsites"], dtype=int),
        "coords_q": np.asarray(static["coords_q"], dtype=float),
        "site_ids_all": np.asarray(static["site_ids_all"], dtype=int),
        "coords_all": np.asarray(static["coords_all"], dtype=float),
        "raw": static,
    }


def setup_geometry_data(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    coords_all = np.asarray(static_data["coords_all"], dtype=float)
    if coords_all.size == 0:
        raise ValueError("Static geometry does not contain any coordinates.")

    raw = static_data.get("raw", {})
    materials = raw.get("materials_all", np.full(len(coords_all), "unknown", dtype=object))
    geometry_params = raw.get("geometry_params", {})
    gate_info = raw.get("gate_info", {})

    return {
        "site_ids": _as_int_list(static_data["site_ids_all"]),
        "coordinates": [[_clean_scalar(value) for value in row] for row in coords_all.tolist()],
        "materials": [str(value) for value in np.asarray(materials, dtype=object).tolist()],
        "qsite_ids": _as_int_list(static_data["Qsites"]),
        "bounds_min": _as_float_list(np.nanmin(coords_all, axis=0)),
        "bounds_max": _as_float_list(np.nanmax(coords_all, axis=0)),
        "geometry_params": _clean_scalar(geometry_params),
        "gate_info": _clean_scalar(gate_info),
    }


def _range_from_values(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if np.isclose(low, high):
        pad = 1e-12 if low == 0 else 1e-6 * abs(low)
        low -= pad
        high += pad
    return low, high


def setup_field_data(run_dir: str | os.PathLike[str], snapshot_name: str, prop_name: str) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    previous = _previous_snapshot_data(run_dir, snapshot_name)
    values_all = scalar_values_surface(static_data, snapshot_data, snapshot_name, previous, prop_name)
    color_min, color_max = _range_from_values(values_all)
    return {
        "snapshot": snapshot_name,
        "property": prop_name,
        "site_ids": _as_int_list(static_data["site_ids_all"]),
        "values": _as_float_list(values_all),
        "color_min": float(color_min),
        "color_max": float(color_max),
    }


def list_snapshots(run_dir: str | os.PathLike[str]) -> list[str]:
    return [path.name for path in _snapshot_file_candidates(run_dir)]


def load_snapshot(run_dir: str | os.PathLike[str], snapshot_name: str) -> dict[str, Any]:
    if snapshot_name in {"run_static", "run_static.npz"}:
        path = Path(run_dir) / "run_static.npz"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_name}")
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    path = Path(run_dir) / snapshot_name
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_name}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _previous_snapshot_name(run_dir: str | os.PathLike[str], snapshot_name: str) -> str | None:
    if snapshot_name in {"run_static", "run_static.npz"}:
        return None
    snapshots = list_snapshots(run_dir)
    if snapshot_name not in snapshots:
        raise FileNotFoundError(f"Snapshot not found: {snapshot_name}")
    index = snapshots.index(snapshot_name)
    if index == 0:
        return None
    return snapshots[index - 1]


def _previous_snapshot_data(run_dir: str | os.PathLike[str], snapshot_name: str) -> dict[str, Any] | None:
    previous_name = _previous_snapshot_name(run_dir, snapshot_name)
    if previous_name is None:
        return None
    return load_snapshot(run_dir, previous_name)


def qprime_mask_from_data(static_data: dict[str, Any], snapshot_data: dict[str, Any]) -> np.ndarray:
    return np.isin(np.asarray(static_data["Qsites"], dtype=int), np.asarray(snapshot_data["Qprime"], dtype=int))


def ldos_at_energy(static_data: dict[str, Any], snapshot_data: dict[str, Any], energy_mode: str) -> np.ndarray:
    if "ildos" not in snapshot_data:
        return np.full(len(static_data["Qsites"]), np.nan, dtype=float)

    ildos = snapshot_data["ildos"]
    ui_all = np.asarray(snapshot_data["Ui"], dtype=float)
    qsites = np.asarray(static_data["Qsites"], dtype=int)
    values = np.full(len(qsites), np.nan, dtype=float)

    for local_index, site_id in enumerate(qsites):
        try:
            energy = np.asarray(ildos[local_index][0], dtype=float)
            rho = np.asarray(ildos[local_index][1], dtype=float)
            if energy.size < 2:
                continue
            target_energy = float(ui_all[site_id]) if energy_mode == "Ui" else 0.0
            values[local_index] = np.interp(target_energy, energy, rho, left=np.nan, right=np.nan)
        except Exception:
            values[local_index] = np.nan
    return values


def scalar_values_quantum(
    static_data: dict[str, Any],
    snapshot_data: dict[str, Any],
    snapshot_name: str,
    previous_snapshot_data: dict[str, Any] | None,
    prop_name: str,
) -> np.ndarray:
    qsites = np.asarray(static_data["Qsites"], dtype=int)
    qmask = qprime_mask_from_data(static_data, snapshot_data)

    if prop_name == "Ui":
        return np.asarray(snapshot_data["Ui"], dtype=float)[qsites]
    if prop_name == "ni":
        return np.asarray(snapshot_data["ni"], dtype=float)[qsites]
    if prop_name == "Ci":
        values = np.full(len(qsites), np.nan, dtype=float)
        ci = np.asarray(snapshot_data["Ci"], dtype=float)
        qprime = np.asarray(snapshot_data["Qprime"], dtype=int)
        for index, site_id in enumerate(qprime[: len(ci)]):
            match = np.where(qsites == site_id)[0]
            if len(match):
                values[int(match[0])] = ci[index]
        return values
    if prop_name == "Qprime_mask":
        return qmask.astype(float)
    if prop_name == "ΔUi":
        if previous_snapshot_data is None:
            return np.zeros(len(qsites), dtype=float)
        return (
            np.asarray(snapshot_data["Ui"], dtype=float)[qsites]
            - np.asarray(previous_snapshot_data["Ui"], dtype=float)[qsites]
        )
    if prop_name == "LDOS@0":
        return ldos_at_energy(static_data, snapshot_data, energy_mode="0")
    if prop_name == "LDOS@Ui":
        return ldos_at_energy(static_data, snapshot_data, energy_mode="Ui")
    raise ValueError(f"Unsupported quantum property: {prop_name}")


def scalar_values_surface(
    static_data: dict[str, Any],
    snapshot_data: dict[str, Any],
    snapshot_name: str,
    previous_snapshot_data: dict[str, Any] | None,
    prop_name: str,
) -> np.ndarray:
    site_ids_all = np.asarray(static_data["site_ids_all"], dtype=int)
    if prop_name == "Ui":
        return np.asarray(snapshot_data["Ui"], dtype=float)[site_ids_all]
    if prop_name == "ni":
        return np.asarray(snapshot_data["ni"], dtype=float)[site_ids_all]
    if prop_name == "ΔUi":
        if previous_snapshot_data is None:
            return np.zeros(len(site_ids_all), dtype=float)
        return (
            np.asarray(snapshot_data["Ui"], dtype=float)[site_ids_all]
            - np.asarray(previous_snapshot_data["Ui"], dtype=float)[site_ids_all]
        )
    if prop_name == "Ci":
        values = np.full(len(site_ids_all), np.nan, dtype=float)
        ci = np.asarray(snapshot_data["Ci"], dtype=float)
        qprime = np.asarray(snapshot_data["Qprime"], dtype=int)
        index_by_site = {int(site_id): index for index, site_id in enumerate(site_ids_all)}
        for index, site_id in enumerate(qprime[: len(ci)]):
            if int(site_id) in index_by_site:
                values[index_by_site[int(site_id)]] = ci[index]
        return values
    raise ValueError(f"Unsupported surface property: {prop_name}")


def _mapper_range_from_qprime(values: np.ndarray, qmask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values_q = values[qmask]
    values_q = values_q[np.isfinite(values_q)]
    if values_q.size == 0:
        values_q = values[np.isfinite(values)]
    if values_q.size == 0:
        return 0.0, 1.0
    low = float(np.nanmin(values_q))
    high = float(np.nanmax(values_q))
    if np.isclose(low, high):
        pad = 1e-12 if low == 0 else 1e-6 * abs(low)
        low -= pad
        high += pad
    return low, high


def _points_near_segment(points_xy: np.ndarray, p0: np.ndarray, p1: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    length_sq = float(np.dot(v, v))
    if length_sq <= 0:
        return np.zeros(len(points_xy), dtype=bool), np.zeros(len(points_xy), dtype=float)

    offsets = np.asarray(points_xy, dtype=float) - p0[None, :]
    t = (offsets @ v) / length_sq
    t_clipped = np.clip(t, 0.0, 1.0)
    projection = p0[None, :] + t_clipped[:, None] * v[None, :]
    distance = np.linalg.norm(points_xy - projection, axis=1)
    mask = distance <= (width / 2.0)
    coord_along = t_clipped * np.sqrt(length_sq)
    return mask, coord_along


def _surface_cut_indices(static_data: dict[str, Any], p0, p1, cut_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_all = np.asarray(static_data["coords_all"], dtype=float)
    points_xy = coords_all[:, :2]
    coord_vert = coords_all[:, 2]
    mask, coord_along = _points_near_segment(points_xy, np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), cut_width)
    indices = np.where(mask)[0]
    order = np.argsort(coord_along[mask])
    return indices[order], coord_along[mask][order], coord_vert[mask][order]


def _quantum_cut_indices(static_data: dict[str, Any], p0, p1, cut_width: float) -> tuple[np.ndarray, np.ndarray]:
    coords_q = np.asarray(static_data["coords_q"], dtype=float)
    mask, coord_along = _points_near_segment(coords_q[:, :2], np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), cut_width)
    indices = np.where(mask)[0]
    order = np.argsort(coord_along[mask])
    return indices[order], coord_along[mask][order]


def quantum_heatmap_data(run_dir: str | os.PathLike[str], snapshot_name: str, prop_name: str) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    previous = _previous_snapshot_data(run_dir, snapshot_name)
    qmask = qprime_mask_from_data(static_data, snapshot_data)
    values = scalar_values_quantum(static_data, snapshot_data, snapshot_name, previous, prop_name)
    color_min, color_max = _mapper_range_from_qprime(values, qmask)
    coords_q = np.asarray(static_data["coords_q"], dtype=float)
    return {
        "snapshot": snapshot_name,
        "property": prop_name,
        "site_ids": _as_int_list(static_data["Qsites"]),
        "x": _as_float_list(coords_q[:, 0]),
        "y": _as_float_list(coords_q[:, 1]),
        "values": _as_float_list(values),
        "qprime_mask": [bool(value) for value in qmask.tolist()],
        "color_min": float(color_min),
        "color_max": float(color_max),
    }


def local_ldos_data(run_dir: str | os.PathLike[str], snapshot_name: str, site_id: int) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    qsites = np.asarray(static_data["Qsites"], dtype=int)

    if int(site_id) not in qsites:
        raise ValueError(f"Site {site_id} is not part of Qsites.")
    if "ildos" not in snapshot_data:
        raise ValueError(f"Snapshot {snapshot_name} does not contain ildos.")

    local_index = int(np.where(qsites == int(site_id))[0][0])
    energy = np.asarray(snapshot_data["ildos"][local_index][0], dtype=float)
    ldos = np.asarray(snapshot_data["ildos"][local_index][1], dtype=float)
    ui = float(np.asarray(snapshot_data["Ui"], dtype=float)[int(site_id)])
    ni = float(np.asarray(snapshot_data["ni"], dtype=float)[int(site_id)])
    qprime = np.asarray(snapshot_data["Qprime"], dtype=int)
    ci_values = np.asarray(snapshot_data["Ci"], dtype=float)
    qprime_matches = np.where(qprime[: len(ci_values)] == int(site_id))[0]
    ci = float(ci_values[int(qprime_matches[0])]) if len(qprime_matches) else np.nan
    ldos_at_0 = float(np.interp(0.0, energy, ldos, left=np.nan, right=np.nan))
    ldos_at_ui = float(np.interp(ui, energy, ldos, left=np.nan, right=np.nan))

    x_dis = energy - ui
    y_dis = ldos
    ildos_dis = np.zeros_like(y_dis, dtype=float)
    if len(y_dis) > 1:
        ildos_dis[1:] = np.cumsum(0.5 * (y_dis[1:] + y_dis[:-1]) * np.diff(x_dis))

    if np.isfinite(ci):
        poisson_prediction = ni + ci * x_dis
        diff = np.abs(poisson_prediction - ildos_dis)
        finite_mask = np.isfinite(diff)
        d_u_solution = float(x_dis[np.where(finite_mask)[0][int(np.argmin(diff[finite_mask]))]]) if np.any(finite_mask) else np.nan
    else:
        poisson_prediction = np.full_like(x_dis, np.nan, dtype=float)
        d_u_solution = np.nan

    return {
        "site_id": int(site_id),
        "snapshot": snapshot_name,
        "energy": _as_float_list(energy),
        "ldos": _as_float_list(ldos),
        "Ui": _clean_scalar(ui),
        "ni": _clean_scalar(ni),
        "Ci": _clean_scalar(ci),
        "ldos_at_0": _clean_scalar(ldos_at_0),
        "ldos_at_Ui": _clean_scalar(ldos_at_ui),
        "consistency_delta_u": _as_float_list(x_dis),
        "consistency_poisson": _as_float_list(poisson_prediction),
        "consistency_integrated": _as_float_list(ildos_dis),
        "dU_solution": _clean_scalar(d_u_solution),
    }


def surface_cut_data(run_dir: str | os.PathLike[str], snapshot_name: str, prop_name: str, p0, p1, cut_width: float) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    previous = _previous_snapshot_data(run_dir, snapshot_name)
    values_all = scalar_values_surface(static_data, snapshot_data, snapshot_name, previous, prop_name)
    indices, coord_along, coord_vert = _surface_cut_indices(static_data, p0, p1, cut_width)
    values_cut = values_all[indices] if len(indices) else np.asarray([], dtype=float)
    return {
        "snapshot": snapshot_name,
        "property": prop_name,
        "p0": _clean_scalar(list(map(float, p0))),
        "p1": _clean_scalar(list(map(float, p1))),
        "cut_width": float(cut_width),
        "distance_along": _as_float_list(coord_along),
        "z": _as_float_list(coord_vert),
        "values": _as_float_list(values_cut),
    }


def ldos_cut_data(
    run_dir: str | os.PathLike[str],
    snapshot_name: str,
    p0,
    p1,
    cut_width: float,
    overlay_snapshot_names: list[str] | None,
) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    if "ildos" not in snapshot_data:
        raise ValueError(f"Snapshot {snapshot_name} does not contain ildos.")

    idx_cut, coord_along = _quantum_cut_indices(static_data, p0, p1, cut_width)
    if len(idx_cut) == 0:
        return {
            "snapshot": snapshot_name,
            "p0": _clean_scalar(list(map(float, p0))),
            "p1": _clean_scalar(list(map(float, p1))),
            "cut_width": float(cut_width),
            "distance_along": [],
            "energy": [],
            "ldos_matrix": [],
            "overlays": [],
        }

    ildos = snapshot_data["ildos"]
    energy_reference = np.asarray(ildos[idx_cut[0]][0], dtype=float)
    rho_columns = []
    for local_index in idx_cut:
        energy = np.asarray(ildos[local_index][0], dtype=float)
        rho = np.asarray(ildos[local_index][1], dtype=float)
        if len(energy) != len(energy_reference) or not np.allclose(energy, energy_reference):
            rho = np.interp(energy_reference, energy, rho, left=np.nan, right=np.nan)
        rho_columns.append(rho)
    rho_matrix = np.asarray(rho_columns, dtype=float).T

    overlays = []
    qsites = np.asarray(static_data["Qsites"], dtype=int)
    selected_overlays = []
    seen = set()
    for name in overlay_snapshot_names or []:
        if name not in seen:
            seen.add(name)
            selected_overlays.append(name)

    for overlay_name in selected_overlays:
        overlay_data = load_snapshot(run_dir, overlay_name)
        ui_values = []
        for local_index in idx_cut:
            site_id = int(qsites[local_index])
            ui_values.append(float(overlay_data["Ui"][site_id]))
        ui_values_arr = np.asarray(ui_values, dtype=float)
        ui_values_arr[(ui_values_arr < energy_reference.min()) | (ui_values_arr > energy_reference.max())] = np.nan
        overlays.append(
            {
                "snapshot": overlay_name,
                "distance_along": _as_float_list(coord_along),
                "Ui": _as_float_list(ui_values_arr),
                "is_current": overlay_name == snapshot_name,
            }
        )

    return {
        "snapshot": snapshot_name,
        "p0": _clean_scalar(list(map(float, p0))),
        "p1": _clean_scalar(list(map(float, p1))),
        "cut_width": float(cut_width),
        "distance_along": _as_float_list(coord_along),
        "energy": _as_float_list(energy_reference),
        "ldos_matrix": [[_clean_scalar(value) for value in row] for row in rho_matrix.tolist()],
        "overlays": overlays,
    }


def quantum_ui_linecut_data(
    run_dir: str | os.PathLike[str],
    snapshot_name: str,
    p0,
    p1,
    cut_width: float,
) -> dict[str, Any]:
    static_data = load_run_static(run_dir)
    snapshot_data = load_snapshot(run_dir, snapshot_name)
    idx_cut, coord_along = _quantum_cut_indices(static_data, p0, p1, cut_width)
    if len(idx_cut) == 0:
        return {
            "snapshot": snapshot_name,
            "p0": _clean_scalar(list(map(float, p0))),
            "p1": _clean_scalar(list(map(float, p1))),
            "cut_width": float(cut_width),
            "distance_along": [],
            "values": [],
        }

    qsites = np.asarray(static_data["Qsites"], dtype=int)
    ui_all = np.asarray(snapshot_data["Ui"], dtype=float)
    values = np.asarray([float(ui_all[int(qsites[local_index])]) for local_index in idx_cut], dtype=float)

    return {
        "snapshot": snapshot_name,
        "p0": _clean_scalar(list(map(float, p0))),
        "p1": _clean_scalar(list(map(float, p1))),
        "cut_width": float(cut_width),
        "distance_along": _as_float_list(coord_along),
        "values": _as_float_list(values),
    }
