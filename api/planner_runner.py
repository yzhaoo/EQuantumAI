from __future__ import annotations

from typing import Any, Callable

from agent.schemas import AgentRuntimeConfig
from api.tools import execute_tool_call
from simulation import profiles

StatusCallback = Callable[[str], None]
LogCallback = Callable[[str], None]
MetadataCallback = Callable[[dict[str, Any]], None]
EventCallback = Callable[[str, dict[str, Any] | None], None]
ManualCheckCallback = Callable[[dict[str, Any]], bool]
AbortCheckCallback = Callable[[], bool]


def _noop_status(_: str) -> None:
    return None


def _noop_log(_: str) -> None:
    return None


def _noop_metadata(_: dict[str, Any]) -> None:
    return None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _extract_last_spec_from_trace(tool_trace: list[dict[str, Any]], fallback_profile: str) -> dict[str, Any]:
    spec: dict[str, Any] | None = None
    for entry in tool_trace:
        result = entry.get("result") or {}
        if isinstance(result.get("spec"), dict):
            spec = result["spec"]
        if isinstance(result.get("resolved_spec"), dict):
            spec = result["resolved_spec"]
        if isinstance(result.get("summary"), dict):
            summary = result["summary"]
            for key in ("spec_a", "spec_b"):
                if isinstance(summary.get(key), dict):
                    spec = summary[key]
    return profiles.normalize_spec(spec, default_profile=fallback_profile)


def execute_planner_plan(
    approved_plan: list[dict[str, Any]],
    config: AgentRuntimeConfig,
    *,
    original_request: str = "",
    status: StatusCallback | None = None,
    log: LogCallback | None = None,
    metadata: MetadataCallback | None = None,
    event: EventCallback | None = None,
    manual_check: ManualCheckCallback | None = None,
    should_abort: AbortCheckCallback | None = None,
) -> dict[str, Any]:
    status_cb = status or _noop_status
    log_cb = log or _noop_log
    metadata_cb = metadata or _noop_metadata

    tool_trace: list[dict[str, Any]] = []
    latest_artifact_dir: str | None = None

    total_steps = len(approved_plan)
    for index, step in enumerate(approved_plan, start=1):
        if should_abort and should_abort():
            from simulation.runner import RunAbortedError
            raise RunAbortedError("Simulation aborted by user request.")

        tool_name = str(step.get("tool_name"))
        purpose = step.get("purpose")
        arguments = _coerce_dict(step.get("arguments"))

        if tool_name in ["run_task", "run_ldos", "run_dos", "validate_spec", "generate_setup"]:
            spec = arguments.get("spec")
            if isinstance(spec, dict):
                from api.planner import _planner_required_defaults
                req_defaults = _planner_required_defaults(config, spec.get("profile", config.profile))
                for k, v in req_defaults.items():
                    if spec.get(k) is None:
                        spec[k] = v

        status_cb(f"planner_step_{index}_of_{total_steps}")
        log_cb(f"[planner] Step {index}/{total_steps}: {tool_name}")
        if purpose:
            log_cb(f"[planner] Purpose: {purpose}")

        result = execute_tool_call(
            tool_name,
            arguments,
            config=config,
            status=status_cb,
            log=log_cb,
            metadata=metadata_cb,
            event=event,
            manual_check=manual_check,
            should_abort=should_abort,
        )
        tool_trace.append(
            {
                "tool_name": tool_name,
                "purpose": purpose,
                "arguments": arguments,
                "result": result,
            }
        )

        run_payload = result.get("run") if isinstance(result, dict) else None
        result_payload = result.get("result") if isinstance(result, dict) else None
        if isinstance(run_payload, dict) and isinstance(run_payload.get("artifact_dir"), str):
            latest_artifact_dir = run_payload["artifact_dir"]
            metadata_cb({"artifact_dir": latest_artifact_dir})
        elif isinstance(result_payload, dict) and isinstance(result_payload.get("artifact_dir"), str):
            latest_artifact_dir = result_payload["artifact_dir"]
            metadata_cb({"artifact_dir": latest_artifact_dir})

        if tool_name == "validate_spec" and isinstance(result, dict) and result.get("ok") is False:
            error_message = "Planner validation failed."
            missing_fields = result.get("missing_fields")
            warnings = result.get("warnings")
            if isinstance(missing_fields, list) and missing_fields:
                error_message += " Missing required fields: " + ", ".join(str(item) for item in missing_fields)
            if isinstance(warnings, list) and warnings:
                error_message += " Warnings: " + "; ".join(str(item) for item in warnings)
            log_cb(f"[planner] {error_message}")
            raise ValueError(error_message)

    final_spec = _extract_last_spec_from_trace(tool_trace, config.profile)
    return {
        "mode": "planner",
        "original_request": original_request,
        "tool_trace": tool_trace,
        "spec": final_spec,
        "artifact_dir": latest_artifact_dir,
    }
