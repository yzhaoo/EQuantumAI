from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from agent.schemas import AgentRuntimeConfig
from simulation import artifacts, profiles, runner, setup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL_CONTRACT_PATH = PROJECT_ROOT / "equantum_mcp_like.yaml"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_profiles",
        "description": "Return available profiles, supported lattice types, device shapes, and defaults.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_spec",
        "description": "Create a new simulation spec from explicit field values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "string", "default": "dotgate_center"},
                "task": {"type": ["string", "null"]},
                "lattice_type": {"type": ["string", "null"]},
                "device_shape": {"type": ["string", "null"]},
                "backgate_voltage": {"type": ["number", "null"]},
                "magnetic_field_T": {"type": ["number", "null"]},
                "solve_self_consistent": {"type": ["boolean", "null"]},
                "raw_query": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "clone_spec",
        "description": "Clone an existing spec so the agent can branch an experiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "update_spec",
        "description": "Apply partial field updates to an existing spec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "updates": {"type": "object"},
            },
            "required": ["spec", "updates"],
        },
    },
    {
        "name": "set_boundary_conditions",
        "description": "Map boundary-condition updates onto spec fields in a planning-safe way.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "updates": {"type": "object"},
                "reinitialize": {"type": "boolean", "default": True},
            },
            "required": ["spec", "updates"],
        },
    },
    {
        "name": "validate_spec",
        "description": "Check required fields and profile compatibility.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "generate_setup",
        "description": "Resolve or build the setup/config for a spec without running the full solve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "require_config": {"type": "boolean", "default": True},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "run_task",
        "description": "Execute the task encoded in a validated spec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "require_manual_check": {"type": "boolean", "default": False},
                "snapshot_mode": {"type": "string", "default": "step"},
                "snapshot_every": {"type": "integer", "default": 1},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "run_ldos",
        "description": "Convenience wrapper that forces task=ldos and executes the run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "require_manual_check": {"type": "boolean", "default": False},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "run_dos",
        "description": "Convenience wrapper that forces task=dos and executes the run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "require_manual_check": {"type": "boolean", "default": False},
            },
            "required": ["spec"],
        },
    },
    {
        "name": "read_run_summary",
        "description": "Load the saved query spec and run summary for a completed run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_dir": {"type": "string"},
            },
            "required": ["artifact_dir"],
        },
    },
    {
        "name": "compare_ldos_runs",
        "description": "Compare two completed LDOS runs using their saved LDOS artifacts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_a": {"type": ["object", "null"]},
                "run_b": {"type": ["object", "null"]},
                "artifact_dir_a": {"type": ["string", "null"]},
                "artifact_dir_b": {"type": ["string", "null"]},
            },
        },
    },
]


def _to_llm_tool_schema(tool_def: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool_def["name"],
        "description": tool_def["description"],
        "parameters": copy.deepcopy(tool_def["input_schema"]),
        "strict": False,
    }


def list_tool_definitions() -> dict[str, Any]:
    contract_text = TOOL_CONTRACT_PATH.read_text() if TOOL_CONTRACT_PATH.exists() else None
    return {
        "contract_path": str(TOOL_CONTRACT_PATH),
        "contract_available": TOOL_CONTRACT_PATH.exists(),
        "contract_text": contract_text,
        "tools": copy.deepcopy(TOOL_DEFINITIONS),
        "llm_tools": [_to_llm_tool_schema(tool_def) for tool_def in TOOL_DEFINITIONS],
    }


def execute_tool_call(name: str, arguments: dict[str, Any], config: AgentRuntimeConfig | None = None, **kwargs) -> dict[str, Any]:
    runtime_config = config or AgentRuntimeConfig()
    handlers = {
        "list_profiles": _handle_list_profiles,
        "create_spec": _handle_create_spec,
        "clone_spec": _handle_clone_spec,
        "update_spec": _handle_update_spec,
        "set_boundary_conditions": _handle_set_boundary_conditions,
        "validate_spec": _handle_validate_spec,
        "generate_setup": _handle_generate_setup,
        "run_task": _handle_run_task,
        "run_ldos": _handle_run_ldos,
        "run_dos": _handle_run_dos,
        "read_run_summary": _handle_read_run_summary,
        "compare_ldos_runs": _handle_compare_ldos_runs,
    }
    try:
        handler = handlers[name]
    except KeyError as exc:
        available = ", ".join(sorted(handlers))
        raise ValueError(f"Unknown tool {name!r}. Available tools: {available}.") from exc
    return handler(arguments or {}, runtime_config, **kwargs)


def _normalize_spec_payload(spec: dict[str, Any] | None, profile_name: str = "dotgate_center") -> dict[str, Any]:
    return profiles.normalize_spec(spec, default_profile=profile_name)


def _resolve_spec(spec: dict[str, Any], config: AgentRuntimeConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    return profiles.resolve_runtime_spec(spec, args=config.to_namespace())


def _serialize_profile(name: str, profile: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    defaults = profiles.get_default_spec_values(config.to_namespace(), name)
    return {
        "name": name,
        "supported_lattice_types": list(profile.get("supported_lattice_types", [])),
        "supported_device_shapes": list(profile.get("supported_device_shapes", [])),
        "density_defaults": dict(profile.get("density_defaults", {})),
        "boundary_defaults": copy.deepcopy(profile.get("boundary_defaults", {})),
        "default_runtime_values": defaults,
    }


def _handle_list_profiles(_: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    profile_summaries = [
        _serialize_profile(name, profile, config) for name, profile in profiles.PROFILES.items()
    ]
    return {"profiles": profile_summaries}


def _handle_create_spec(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    profile_name = str(arguments.get("profile") or config.profile or "dotgate_center")
    spec = profiles.empty_spec(default_profile=profile_name)
    updates = {key: value for key, value in arguments.items() if key in profiles.SPEC_FIELDS}
    spec = profiles.merge_spec(spec, updates)
    return {"spec": spec}


def _handle_clone_spec(arguments: dict[str, Any], _: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    return {"spec": copy.deepcopy(spec)}


def _handle_update_spec(arguments: dict[str, Any], _: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    updates = dict(arguments.get("updates") or {})
    updated = profiles.merge_spec(spec, updates)
    return {"spec": updated}


def _handle_set_boundary_conditions(arguments: dict[str, Any], _: AgentRuntimeConfig) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    updates = dict(arguments.get("updates") or {})
    mapped_updates: dict[str, Any] = {}

    gate_update = updates.get("gate") or {}
    dielectric_update = updates.get("dielectric") or {}
    backgate_update = updates.get("backgate") or {}

    if "potential" in gate_update:
        mapped_updates["gate_potential"] = gate_update["potential"]
    if "dielectric_constant" in dielectric_update:
        mapped_updates["dielectric_constant"] = dielectric_update["dielectric_constant"]
    if "potential" in backgate_update:
        mapped_updates["backgate_voltage"] = backgate_update["potential"]

    updated_spec = profiles.merge_spec(spec, mapped_updates)
    return {
        "spec": updated_spec,
        "bc_summary": {
            "applied_updates": mapped_updates,
            "reinitialize": bool(arguments.get("reinitialize", True)),
        },
    }


def _extract_ldos_from_static_reference(artifact_dir: Path) -> dict[str, Any]:
    static_path = artifact_dir / "run_static.npz"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing static reference: {static_path}")
    data = np.load(static_path, allow_pickle=True)
    if "ildos" not in data.files:
        raise KeyError(f"Static reference does not contain ildos data: {static_path}")
    ildos = np.asarray(data["ildos"], dtype=float)
    if ildos.size == 0:
        raise ValueError(f"Static reference ildos array is empty: {static_path}")
    static_data = data["static_data"][0]
    qsites = np.asarray(static_data["Qsites"], dtype=int)
    coords_q = np.asarray(static_data["coords_q"], dtype=float)
    center_idx = int(np.argmin(np.linalg.norm(coords_q, axis=1)))
    return {
        "site_id": int(qsites[center_idx]),
        "energy": np.asarray(ildos[center_idx, 0, :], dtype=float),
        "ldos": np.asarray(ildos[center_idx, 1, :], dtype=float),
        "source": str(static_path),
    }


def _handle_validate_spec(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    warnings: list[str] = []
    missing_fields = profiles.get_missing_fields(spec)
    if missing_fields:
        return {
            "ok": False,
            "missing_fields": missing_fields,
            "warnings": warnings,
            "resolved_spec": spec,
        }

    try:
        resolved_spec, profile = _resolve_spec(spec, config)
    except Exception as exc:
        return {
            "ok": False,
            "missing_fields": [],
            "warnings": [str(exc)],
            "resolved_spec": spec,
        }

    if resolved_spec.get("solve_self_consistent") is False:
        warnings.append("solve_self_consistent is false; some downstream workflows may assume FSC execution.")

    return {
        "ok": True,
        "missing_fields": [],
        "warnings": warnings,
        "resolved_spec": resolved_spec,
        "profile_name": profile["name"],
    }


def _handle_generate_setup(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    require_config = bool(arguments.get("require_config", True))
    resolved_spec, profile = _resolve_spec(spec, config)
    setup_dir, config_file = setup.resolve_setup(profile, resolved_spec, require_config=require_config)
    return {
        "spec": resolved_spec,
        "setup_dir": setup_dir,
        "config_file": config_file,
        "resolved_profile": {
            "name": profile["name"],
            "device_shape": profile.get("device_shape"),
            "lattice_type": profile.get("lattice_type"),
        },
    }


def _handle_run_task(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    missing_fields = profiles.get_missing_fields(spec)
    if missing_fields:
        raise ValueError(
            "Spec is incomplete for execution. Missing required fields: "
            + ", ".join(missing_fields)
            + ". Validate or ask the user to clarify them before running."
        )
    require_manual_check = bool(arguments.get("require_manual_check", False))
    snapshot_mode = str(arguments.get("snapshot_mode", "step"))
    snapshot_every = int(arguments.get("snapshot_every", 1))
    result = runner.run_spec(
        spec,
        config=config,
        require_manual_check=require_manual_check,
        snapshot_mode=snapshot_mode,
        snapshot_every=snapshot_every,
        **kwargs
    )
    return {
        "run": {
            "artifact_dir": result.get("artifact_dir"),
            "task": result.get("task"),
            "profile": result.get("profile"),
            "lattice_type": result.get("lattice_type"),
            "solve_self_consistent": result.get("solve_self_consistent"),
            "scf_executed": result.get("scf_executed"),
            "magnetic_field_T": result.get("magnetic_field_T"),
            "backgate_voltage_V": result.get("backgate_voltage_V"),
        },
        "result": result,
    }


def _handle_run_ldos(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    updated_spec = profiles.merge_spec(spec, {"task": "ldos"})
    return _handle_run_task(
        {
            "spec": updated_spec,
            "require_manual_check": arguments.get("require_manual_check", False),
        },
        config,
        **kwargs
    )


def _handle_run_dos(arguments: dict[str, Any], config: AgentRuntimeConfig, **kwargs) -> dict[str, Any]:
    spec = _normalize_spec_payload(arguments.get("spec"))
    updated_spec = profiles.merge_spec(spec, {"task": "dos"})
    return _handle_run_task(
        {
            "spec": updated_spec,
            "require_manual_check": arguments.get("require_manual_check", False),
        },
        config,
        **kwargs
    )


def _handle_read_run_summary(arguments: dict[str, Any], _: AgentRuntimeConfig) -> dict[str, Any]:
    artifact_dir = Path(str(arguments["artifact_dir"])).resolve()
    result = artifacts.load_run_summary(str(artifact_dir))
    spec = artifacts.load_query_spec(str(artifact_dir))
    return {
        "artifact_dir": str(artifact_dir),
        "spec": spec,
        "result": result,
    }


def _artifact_dir_from_run_like(run_like: dict[str, Any] | None, explicit_artifact_dir: Any) -> Path:
    candidate = explicit_artifact_dir
    if candidate is None and isinstance(run_like, dict):
        candidate = run_like.get("artifact_dir")
    if not candidate:
        raise ValueError("Expected either an artifact_dir_* argument or a run_* object containing artifact_dir.")
    return Path(str(candidate)).resolve()


def _load_ldos_artifact(artifact_dir: Path) -> dict[str, Any]:
    ldos_path = artifact_dir / "ldos_data.npz"
    if ldos_path.exists():
        data = np.load(ldos_path, allow_pickle=True)
        energy = np.asarray(data["energy"], dtype=float)
        ldos = np.asarray(data["ldos"], dtype=float)
        site_id = int(np.asarray(data["site_id"]).item())
        return {
            "site_id": site_id,
            "energy": energy,
            "ldos": ldos,
            "source": str(ldos_path),
        }
    return _extract_ldos_from_static_reference(artifact_dir)


def _handle_compare_ldos_runs(arguments: dict[str, Any], _: AgentRuntimeConfig) -> dict[str, Any]:
    artifact_dir_a = _artifact_dir_from_run_like(arguments.get("run_a"), arguments.get("artifact_dir_a"))
    artifact_dir_b = _artifact_dir_from_run_like(arguments.get("run_b"), arguments.get("artifact_dir_b"))

    summary_a = artifacts.load_run_summary(str(artifact_dir_a))
    summary_b = artifacts.load_run_summary(str(artifact_dir_b))
    spec_a = artifacts.load_query_spec(str(artifact_dir_a))
    spec_b = artifacts.load_query_spec(str(artifact_dir_b))
    ldos_a = _load_ldos_artifact(artifact_dir_a)
    ldos_b = _load_ldos_artifact(artifact_dir_b)

    energy_a = ldos_a["energy"]
    energy_b = ldos_b["energy"]
    rho_a = ldos_a["ldos"]
    rho_b = ldos_b["ldos"]

    energy_overlap_min = max(float(np.min(energy_a)), float(np.min(energy_b)))
    energy_overlap_max = min(float(np.max(energy_a)), float(np.max(energy_b)))
    if energy_overlap_min >= energy_overlap_max:
        raise ValueError("The two LDOS runs do not have an overlapping energy range.")

    common_energy = energy_a if np.array_equal(energy_a, energy_b) else np.linspace(
        energy_overlap_min,
        energy_overlap_max,
        int(min(len(energy_a), len(energy_b))),
    )

    aligned_a = np.interp(common_energy, energy_a, rho_a)
    aligned_b = np.interp(common_energy, energy_b, rho_b)
    delta = aligned_b - aligned_a

    max_abs_index = int(np.argmax(np.abs(delta)))
    integrated_abs_delta = float(np.trapz(np.abs(delta), common_energy))
    rms_delta = float(np.sqrt(np.mean(delta ** 2)))

    comparison = {
        "artifact_dir_a": str(artifact_dir_a),
        "artifact_dir_b": str(artifact_dir_b),
        "site_id_a": ldos_a["site_id"],
        "site_id_b": ldos_b["site_id"],
        "source_a": ldos_a.get("source"),
        "source_b": ldos_b.get("source"),
        "energy": common_energy.tolist(),
        "ldos_a": aligned_a.tolist(),
        "ldos_b": aligned_b.tolist(),
        "delta": delta.tolist(),
    }
    summary = {
        "task_a": summary_a.get("task"),
        "task_b": summary_b.get("task"),
        "magnetic_field_T_a": summary_a.get("magnetic_field_T"),
        "magnetic_field_T_b": summary_b.get("magnetic_field_T"),
        "backgate_voltage_V_a": summary_a.get("backgate_voltage_V"),
        "backgate_voltage_V_b": summary_b.get("backgate_voltage_V"),
        "site_id_a": ldos_a["site_id"],
        "site_id_b": ldos_b["site_id"],
        "energy_overlap": [energy_overlap_min, energy_overlap_max],
        "rms_delta": rms_delta,
        "integrated_abs_delta": integrated_abs_delta,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "max_abs_delta_energy": float(common_energy[max_abs_index]),
        "mean_ldos_a": float(np.mean(aligned_a)),
        "mean_ldos_b": float(np.mean(aligned_b)),
        "spec_a": spec_a,
        "spec_b": spec_b,
    }
    return {
        "comparison": comparison,
        "summary": summary,
    }
