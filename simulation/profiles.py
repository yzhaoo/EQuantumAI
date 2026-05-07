from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from typing import Any

import scipy.constants as sc

from agent.schemas import AgentRuntimeConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DOTGATE_SPACING0 = 0.02
DEFAULT_DOTGATE_K = 0.2
DEVICE_SHAPE_DATA_DIRS = {
    "dotgate": "dotgate_center",
    "squaregate_center": "squaregate_center",
}

BASE_REQUIRED_FIELDS = ["backgate_voltage", "magnetic_field_T", "solve_self_consistent"]

OPTIONAL_RUNTIME_FIELDS = [
    "device_shape",
    "spacing0",
    "density_k",
    "dielectric_constant",
    "gate_potential",
    "convergence_tol",
    "Ncore",
    "eta",
    "ldos_method",
]

SPEC_FIELDS = [
    "raw_query",
    "task",
    "profile",
    "lattice_type",
    "device_shape",
    "backgate_voltage",
    "magnetic_field_T",
    "solve_self_consistent",
    "spacing0",
    "density_k",
    "dielectric_constant",
    "gate_potential",
    "convergence_tol",
    "Ncore",
    "eta",
    "ldos_method",
]


def build_density_function_dotgate_center(
    spacing0: float = DEFAULT_DOTGATE_SPACING0,
    k: float = DEFAULT_DOTGATE_K,
):
    def density_function(z):
        if abs(z) < 3 * spacing0:
            return spacing0
        return spacing0 + k * z

    return density_function


def get_geoparams_hash(params: dict[str, Any], *funcs) -> str:
    params_repr: dict[str, str] = {}
    for key, value in params.items():
        if callable(value):
            try:
                params_repr[key] = inspect.getsource(value).strip()
            except Exception:
                params_repr[key] = str(value)
        else:
            params_repr[key] = str(value)

    for func in funcs:
        try:
            params_repr[func.__name__] = inspect.getsource(func).strip()
        except Exception:
            params_repr[func.__name__] = str(func)

    params_str = str(sorted(params_repr.items()))
    return hashlib.md5(params_str.encode()).hexdigest()


def build_dotgate_center_profile(
    project_root: str,
    spacing0: float = DEFAULT_DOTGATE_SPACING0,
    density_k: float = DEFAULT_DOTGATE_K,
    lattice_type: str = "square",
    device_shape: str = "dotgate",
) -> dict[str, Any]:
    density_function = build_density_function_dotgate_center(spacing0=spacing0, k=density_k)
    geoparams = {
        "lattice_type": lattice_type,
        "box_size": ((-0.6, 0.6), (-0.6, 0.6), (-0.08, 0.08)),
        "sampling_density_function": density_function,
        "quantum_center": (0, 0, 0),
        "spacing0": spacing0,
        "density_k": density_k,
    }

    data_dir_name = DEVICE_SHAPE_DATA_DIRS.get(device_shape, "dotgate_center")
    data_root = os.path.join(project_root, "Datas", data_dir_name)
    setup_root = os.path.join(data_root, "setup")
    setup_assets_root = os.path.join(project_root, "Datas", "Setup")
    setup_hash = get_geoparams_hash(geoparams, density_function)
    blend_files = {
        "dotgate": os.path.join(setup_assets_root, "dotgate.blend"),
        "squaregate_center": os.path.join(setup_assets_root, "squraegate_center.blend"),
    }

    return {
        "name": "dotgate_center",
        "lattice_type": lattice_type,
        "supported_lattice_types": ["square", "honeycomb"],
        "device_shape": device_shape,
        "supported_device_shapes": ["dotgate", "squaregate_center"],
        "geoparams": geoparams,
        "data_root": data_root,
        "setup_root": setup_root,
        "expected_setup_dir": os.path.join(setup_root, f"setup_{setup_hash}"),
        "config_filename": "updated_sites_dot.json",
        "sites_filename": "sites.json",
        "hash_filename": "geoparams_hash.txt",
        "readme_filename": "README.txt",
        "setup_assets_root": setup_assets_root,
        "blend_file": blend_files.get(device_shape, blend_files["dotgate"]),
        "blender_script": os.path.join(setup_assets_root, "assign_point.py"),
        "blend_files": blend_files,
        "density_defaults": {
            "spacing0": float(spacing0),
            "density_k": float(density_k),
        },
        "boundary_defaults": {
            "gate": {"potential": -1.0},
            "dielectric": {"dielectric_constant": 4.0},
            "backgate": {"potential": 0.0},
        },
    }


PROFILES = {
    "dotgate_center": build_dotgate_center_profile(PROJECT_ROOT),
}


def empty_spec(default_profile: str = "dotgate_center") -> dict[str, Any]:
    return {
        "raw_query": "",
        "task": None,
        "profile": default_profile,
        "lattice_type": None,
        "device_shape": None,
        "backgate_voltage": None,
        "magnetic_field_T": None,
        "solve_self_consistent": None,
        "spacing0": None,
        "density_k": None,
        "dielectric_constant": None,
        "gate_potential": None,
        "convergence_tol": None,
        "Ncore": None,
        "eta": None,
        "ldos_method": None,
    }


def normalize_spec(spec: dict[str, Any] | None, default_profile: str = "dotgate_center") -> dict[str, Any]:
    normalized = empty_spec(default_profile=default_profile)
    if spec is None:
        spec = {}
    for key in SPEC_FIELDS:
        if key in spec:
            normalized[key] = spec[key]
    if normalized["profile"] is None:
        normalized["profile"] = default_profile
    profile_defaults = PROFILES.get(normalized["profile"])
    if profile_defaults is not None:
        if normalized["lattice_type"] is None:
            normalized["lattice_type"] = profile_defaults.get("lattice_type")
        if normalized["device_shape"] is None:
            normalized["device_shape"] = profile_defaults.get("device_shape")
    return normalized


def get_required_fields(spec: dict[str, Any]) -> list[str]:
    del spec
    return list(BASE_REQUIRED_FIELDS)


def get_missing_fields(spec: dict[str, Any]) -> list[str]:
    normalized = normalize_spec(spec, default_profile=spec.get("profile", "dotgate_center") if spec else "dotgate_center")
    return [field for field in get_required_fields(normalized) if normalized.get(field) is None]


def get_profile_defaults(profile_name: str) -> dict[str, Any]:
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown profile: {profile_name}")
    return PROFILES[profile_name]


def get_runtime_profile(
    spec_or_profile_name,
    spacing0: float | None = None,
    density_k: float | None = None,
    lattice_type: str | None = None,
    device_shape: str | None = None,
) -> dict[str, Any]:
    if isinstance(spec_or_profile_name, dict):
        profile_name = spec_or_profile_name.get("profile", "dotgate_center")
        spacing0 = spec_or_profile_name.get("spacing0", spacing0)
        density_k = spec_or_profile_name.get("density_k", density_k)
        lattice_type = spec_or_profile_name.get("lattice_type", lattice_type)
        device_shape = spec_or_profile_name.get("device_shape", device_shape)
    else:
        profile_name = spec_or_profile_name

    base_profile = get_profile_defaults(profile_name)
    resolved_spacing0 = float(
        spacing0 if spacing0 is not None else base_profile["density_defaults"]["spacing0"]
    )
    resolved_density_k = float(
        density_k if density_k is not None else base_profile["density_defaults"]["density_k"]
    )
    resolved_lattice_type = lattice_type or base_profile["lattice_type"]
    resolved_device_shape = device_shape or base_profile["device_shape"]

    if profile_name == "dotgate_center":
        return build_dotgate_center_profile(
            PROJECT_ROOT,
            spacing0=resolved_spacing0,
            density_k=resolved_density_k,
            lattice_type=resolved_lattice_type,
            device_shape=resolved_device_shape,
        )

    raise KeyError(f"Unsupported runtime profile builder for {profile_name!r}")


def merge_spec(existing_spec: dict[str, Any] | None, new_partial_spec: dict[str, Any] | None) -> dict[str, Any]:
    existing = normalize_spec(existing_spec)
    new_partial_raw = dict(new_partial_spec or {})
    new_partial = normalize_spec(new_partial_spec, default_profile=existing.get("profile", "dotgate_center"))

    merged = dict(existing)
    for key in SPEC_FIELDS:
        if key == "raw_query":
            continue
        if key == "solve_self_consistent" and new_partial_raw.get("solve_self_consistent") is None:
            continue
        value = new_partial.get(key)
        if value is not None:
            merged[key] = value

    old_raw = (existing.get("raw_query") or "").strip()
    new_raw = (new_partial.get("raw_query") or "").strip()
    if old_raw and new_raw:
        merged["raw_query"] = old_raw + "\n" + new_raw
    elif new_raw:
        merged["raw_query"] = new_raw
    else:
        merged["raw_query"] = old_raw

    return merged


def get_default_spec_values(
    config_or_args: AgentRuntimeConfig | Any,
    profile_name: str,
) -> dict[str, Any]:
    profile = get_profile_defaults(profile_name)
    return {
        "device_shape": getattr(config_or_args, "device_shape", profile["device_shape"]),
        "spacing0": float(profile["density_defaults"]["spacing0"]),
        "density_k": float(profile["density_defaults"]["density_k"]),
        "dielectric_constant": float(profile["boundary_defaults"]["dielectric"]["dielectric_constant"]),
        "gate_potential": float(profile["boundary_defaults"]["gate"]["potential"]),
        "convergence_tol": [
            float(getattr(config_or_args, "tol_poisson", 1e-3)),
            float(getattr(config_or_args, "tol_ildos", 1e-2)),
        ],
        "Ncore": int(getattr(config_or_args, "ncore", 1)),
        "eta": float(getattr(config_or_args, "eta", 0.00015)),
        "ldos_method": getattr(config_or_args, "ldos_method", "ED"),
    }


def apply_defaults_to_spec(spec: dict[str, Any], defaults: dict[str, Any], fields: list[str] | None = None) -> dict[str, Any]:
    updated = dict(spec)
    for field in fields or OPTIONAL_RUNTIME_FIELDS:
        if updated.get(field) is None and field in defaults:
            value = defaults[field]
            updated[field] = list(value) if isinstance(value, list) else value
    return updated


def build_boundary_conditions(profile: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    updates = copy.deepcopy(profile["boundary_defaults"])
    updates["backgate"]["potential"] = float(spec["backgate_voltage"])
    if spec.get("gate_potential") is not None:
        updates["gate"]["potential"] = float(spec["gate_potential"])
    if spec.get("dielectric_constant") is not None:
        updates["dielectric"]["dielectric_constant"] = float(spec["dielectric_constant"])
    return updates


def ensure_compatible(spec: dict[str, Any], profile: dict[str, Any]) -> None:
    requested_lattice = spec["lattice_type"]
    requested_device_shape = spec.get("device_shape")
    supported_lattice_types = profile.get("supported_lattice_types")
    supported_device_shapes = profile.get("supported_device_shapes")

    if supported_lattice_types is not None:
        if requested_lattice not in supported_lattice_types:
            raise ValueError(
                f"Query requested lattice_type={requested_lattice!r}, "
                f"but profile {profile['name']!r} only supports {supported_lattice_types!r}."
            )
    elif requested_lattice != profile["lattice_type"]:
        raise ValueError(
            f"Query requested lattice_type={requested_lattice!r}, "
            f"but profile {profile['name']!r} is configured for {profile['lattice_type']!r}."
        )

    if supported_device_shapes is not None and requested_device_shape not in supported_device_shapes:
        raise ValueError(
            f"Query requested device_shape={requested_device_shape!r}, "
            f"but profile {profile['name']!r} only supports {supported_device_shapes!r}."
        )


def resolve_runtime_spec(
    spec: dict[str, Any],
    args: AgentRuntimeConfig | Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize_spec(spec)
    runtime_args = args or AgentRuntimeConfig().to_namespace()
    defaults = get_default_spec_values(runtime_args, normalized["profile"])
    resolved_spec = apply_defaults_to_spec(normalized, defaults, OPTIONAL_RUNTIME_FIELDS)
    profile = get_runtime_profile(resolved_spec)
    ensure_compatible(resolved_spec, profile)
    return resolved_spec, profile


def magnetic_field_to_phi(b_field_t: float, unit_cell_area: float) -> float:
    return b_field_t * sc.e * unit_cell_area / sc.h
