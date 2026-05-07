from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from agent import parser as agent_parser
from agent.schemas import AgentRuntimeConfig
from api.tools import execute_tool_call, list_tool_definitions
from simulation import profiles

_MAX_HISTORY_TURNS = 24


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]

    text_parts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                text_parts.append(content["text"])
    return "\n".join(part for part in text_parts if part.strip()).strip()


def _responses_request(
    *,
    api_key: str,
    base_url: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/responses"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _tool_name_list() -> list[str]:
    return [tool["name"] for tool in list_tool_definitions()["tools"]]


def _normalize_history(history: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(history, list):
        return normalized
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip().lower()
        text = str(entry.get("text") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        normalized.append({"role": role, "text": text})
    return normalized[-_MAX_HISTORY_TURNS:]


def _append_history(history: list[dict[str, str]], role: str, text: str) -> list[dict[str, str]]:
    updated = list(history)
    cleaned_text = text.strip()
    if cleaned_text:
        updated.append({"role": role, "text": cleaned_text})
    return updated[-_MAX_HISTORY_TURNS:]


def _history_text(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior conversation)"
    return "\n\n".join(f"{entry['role'].title()}:\n{entry['text']}" for entry in history)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _extract_request_hints(text: str, default_profile: str) -> dict[str, Any]:
    parsed = agent_parser.parse_query_partial(text, default_profile=default_profile)
    return {
        key: value
        for key, value in parsed.items()
        if key in profiles.SPEC_FIELDS and value is not None
    }


def _collect_request_hints(
    *,
    conversation_history: list[dict[str, str]] | None,
    original_request: str | None,
    latest_message: str,
    default_profile: str,
) -> dict[str, Any]:
    merged = profiles.empty_spec(default_profile=default_profile)
    candidate_texts: list[str] = []
    if original_request:
        candidate_texts.append(original_request)
    if conversation_history:
        candidate_texts.extend(
            entry["text"] for entry in conversation_history if entry.get("role") == "user" and entry.get("text")
        )
    if latest_message:
        candidate_texts.append(latest_message)
    for text in candidate_texts:
        merged = profiles.merge_spec(merged, _extract_request_hints(text, default_profile))
    return merged


def _normalize_completed_runs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized[-4:]


def _planner_required_defaults(config: AgentRuntimeConfig, profile_name: str) -> dict[str, Any]:
    profile = profiles.get_profile_defaults(profile_name)
    return {
        "backgate_voltage": float(profile["boundary_defaults"]["backgate"]["potential"]),
        "magnetic_field_T": 0.0,
        "solve_self_consistent": not bool(getattr(config, "no_scf", False)),
    }


def _planner_plan_prompt(tool_names: list[str]) -> str:
    tool_list = ", ".join(tool_names)
    return (
        "You are the EQuantum experiment planner. "
        "Assume you are a postdoctoral researcher in a lab assisting a professor with planning computational experiments. "
        "Answer in a professional, careful, and conservative way. "
        "Use a first-person narrative when writing the human-readable plan and confirmations. "
        "Do not execute tools. Instead, produce an explicit execution plan for the user's simulation request. "
        "Also produce a short human-readable natural-language plan with numbered steps under a 'Plan:' style heading. "
        "Return the exact tools to call, in order, with concrete JSON arguments for each one. "
        "For each step, arguments_json must be a valid JSON object encoded as a string, like '{}' or '{\"spec\": {...}}'. "
        "Do not use markdown fences, comments, Python dict syntax, or explanatory text inside arguments_json. "
        "Only use argument names that actually exist in the provided tool schemas. "
        "Do not invent helper fields such as spec_ref, label, method, runs, export_plots, boundary_conditions, no_scf, name, or comparison_metrics unless they are explicitly present in a tool schema. "
        "When the user refers to ED, TF, or KPM-style methods, store that choice in spec.ldos_method. "
        "Use 'kpm' or 'kmeanssample' only as values of ldos_method, never as a free-standing method field. "
        "Use create_spec and update_spec to set spec fields, and pass those values inside the spec object when calling validate_spec, generate_setup, run_task, or any visualization/comparison tool. "
        "If the user asks for comparisons, with/without self-consistency, or multiple field values, expand that into multiple runs and a comparison step. "
        "The run itself no longer needs a final DOS-vs-LDOS task selection. Treat the simulation spec as complete once the physical setup and solver settings are complete. "
        "Choose result visualization tools after the run when the user asks what to inspect or compare. "
        "Use the provided conversation history and prior request context to resolve referential follow-ups like 'restart the plan', 'same setup', 'run it again', or 'change B to 2 T'. "
        "If completed runs are provided by the backend, you may use them for post-run visualization or comparison plans instead of proposing a new simulation. "
        "If the latest user message is shorthand, reconstruct the intended task from that prior context instead of treating it as a brand-new standalone request. "
        "Always include validate_spec before any run_task step. "
        "Available tools are: "
        f"{tool_list}. "
        "Use any structured request hints provided by the backend as strong evidence for user-specified values such as magnetic_field_T, backgate_voltage, and solve_self_consistent. "
        "If those fields are still omitted after considering the user's request, the backend may apply conservative defaults later and ask for confirmation."
    )


def _pending_plan_reply_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["approve", "revise", "clarify"],
            },
            "reason": {"type": "string"},
            "revised_request": {"type": ["string", "null"]},
            "assistant_message": {"type": "string"},
        },
        "required": ["action", "reason", "revised_request", "assistant_message"],
        "additionalProperties": False,
    }


def _interpret_pending_plan_reply(
    *,
    user_message: str,
    original_request: str,
    pending_plan: list[dict[str, Any]],
    plan_summary: str | None,
    conversation_history: list[dict[str, str]] | None,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Planner mode requires OpenAI.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    body = {
        "model": config.openai_model,
        "instructions": (
            "You are interpreting a user's reply to a previously proposed EQuantum execution plan. "
            "Assume you are a postdoctoral researcher assisting a professor. "
            "Be professional and conservative, and use first-person wording in assistant_message. "
            "Classify whether the user is approving the existing plan, asking to revise it, or is still unclear. "
            "Use the stored plan context, not just keyword matching. "
            "If the user is approving, set action='approve'. "
            "If they are asking to change parameters or otherwise alter the plan, set action='revise' and provide revised_request. "
            "If they are ambiguous or asking a side question without approval, set action='clarify'."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Conversation history:\n{_history_text(conversation_history or [])}\n\n"
                            f"Original request:\n{original_request}\n\n"
                            f"Plan summary:\n{plan_summary or '-'}\n\n"
                            f"Pending plan:\n{json.dumps(pending_plan, indent=2, sort_keys=True)}\n\n"
                            f"Latest user reply:\n{user_message}"
                        ),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_pending_plan_reply",
                "strict": True,
                "schema": _pending_plan_reply_schema(),
            }
        },
    }
    payload = _responses_request(api_key=api_key, base_url=base_url, body=body)
    result = json.loads(_extract_output_text(payload))
    result["_debug_request_body"] = body
    return result


def _plan_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "human_readable_plan": {"type": "string"},
            "needs_confirmation_message": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "enum": tool_names},
                        "purpose": {"type": "string"},
                        "arguments_json": {"type": "string"},
                    },
                    "required": ["tool_name", "purpose", "arguments_json"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "human_readable_plan", "needs_confirmation_message", "tool_calls"],
        "additionalProperties": False,
    }


def _generate_plan(
    message: str,
    config: AgentRuntimeConfig,
    *,
    conversation_history: list[dict[str, str]] | None = None,
    original_request: str | None = None,
    last_plan_summary: str | None = None,
    last_tool_trace: list[dict[str, Any]] | None = None,
    request_hints: dict[str, Any] | None = None,
    completed_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Planner mode requires OpenAI.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    tool_names = _tool_name_list()
    prompt = _planner_plan_prompt(tool_names)

    body = {
        "model": config.openai_model,
        "instructions": prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Runtime defaults:\n{json.dumps(config.to_namespace().__dict__, indent=2, sort_keys=True)}\n\n"
                            f"Conversation history:\n{_history_text(conversation_history or [])}\n\n"
                            f"Prior original request:\n{original_request or '-'}\n\n"
                            f"Structured request hints:\n{json.dumps(request_hints or {}, indent=2, sort_keys=True)}\n\n"
                            f"Completed runs available for post-run visualization or comparison:\n{json.dumps(completed_runs or [], indent=2, sort_keys=True)}\n\n"
                            f"Previous plan summary:\n{last_plan_summary or '-'}\n\n"
                            f"Previous tool trace:\n{json.dumps(last_tool_trace or [], indent=2, sort_keys=True)}\n\n"
                            f"Latest user request:\n{message}"
                        ),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_planner_plan",
                "strict": True,
                "schema": _plan_schema(tool_names),
            }
        },
    }

    payload = _responses_request(api_key=api_key, base_url=base_url, body=body)
    plan = json.loads(_extract_output_text(payload))
    plan["tool_calls"] = _normalize_plan_steps(list(plan.get("tool_calls", [])))
    plan["_debug_request_body"] = body
    return plan


def _summarize_results(
    *,
    original_request: str,
    tool_trace: list[dict[str, Any]],
    config: AgentRuntimeConfig,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Planner execution finished."

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    body = {
        "model": config.openai_model,
        "instructions": (
            "You are the EQuantum experiment summarizer. "
            "Assume you are a postdoctoral researcher reporting back to a professor. "
            "Write professionally, conservatively, and in first-person narrative. "
            "Write a concise summary of the completed simulation workflow and the main result. "
            "If a comparison was performed, mention what was compared and reference the most relevant metrics or artifact directories."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Original request:\n{original_request}\n\n"
                            f"Tool trace:\n{json.dumps(tool_trace, indent=2, sort_keys=True)}"
                        ),
                    }
                ],
            }
        ],
    }
    payload = _responses_request(api_key=api_key, base_url=base_url, body=body)
    return _extract_output_text(payload) or "Planner execution finished."


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


def _normalize_plan_steps(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_steps: list[dict[str, Any]] = []
    for step in tool_calls:
        raw_arguments = str(step.get("arguments_json", "{}")).strip()
        if not raw_arguments:
            raw_arguments = "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner returned invalid arguments_json for tool {step.get('tool_name')!r}: {exc}"
            ) from exc
        if not isinstance(arguments, dict):
            raise ValueError(
                f"Planner returned non-object arguments_json for tool {step.get('tool_name')!r}."
            )
        normalized_steps.append(
            {
                "tool_name": step.get("tool_name"),
                "purpose": step.get("purpose"),
                "arguments": arguments,
            }
        )
    return normalized_steps


def _render_plan_message(plan: dict[str, Any]) -> str:
    lines = [str(plan["summary"]).strip()]
    human_readable_plan = str(plan.get("human_readable_plan") or "").strip()
    if human_readable_plan:
        lines.extend(["", human_readable_plan])
    lines.extend(["", "Planned tool calls:"])
    for index, step in enumerate(plan.get("tool_calls", []), start=1):
        tool_name = step.get("tool_name", "unknown_tool")
        purpose = step.get("purpose", "")
        arguments = json.dumps(step.get("arguments", {}), sort_keys=True)
        lines.append(f"{index}. `{tool_name}`")
        if purpose:
            lines.append(f"   {purpose}")
        lines.append(f"   args: {arguments}")
    lines.append("")
    lines.append(str(plan["needs_confirmation_message"]).strip())
    return "\n".join(lines)


def _format_default_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(v):g}" if isinstance(v, (int, float)) else str(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _estimate_default_fields(spec: dict[str, Any], resolved_spec: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    defaults = profiles.get_default_spec_values(config.to_namespace(), resolved_spec["profile"])
    default_fields: dict[str, Any] = {}

    for field, default_value in defaults.items():
        if resolved_spec.get(field) == default_value:
            default_fields[field] = resolved_spec.get(field)
    return default_fields


def _build_spec_review(
    plan_steps: list[dict[str, Any]],
    config: AgentRuntimeConfig,
    *,
    request_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_tools = {"run_task", "validate_spec", "generate_setup"}
    spec_state: dict[str, Any] | None = None
    review_runs: list[dict[str, Any]] = []
    aggregate_missing_fields: list[str] = []
    resolved_plan_steps: list[dict[str, Any]] = []
    normalized_request_hints = profiles.normalize_spec(request_hints, default_profile=config.profile)

    for index, step in enumerate(plan_steps, start=1):
        tool_name = str(step.get("tool_name") or "")
        arguments = _coerce_dict(step.get("arguments"))
        resolved_step = {
            "tool_name": step.get("tool_name"),
            "purpose": step.get("purpose"),
            "arguments": dict(arguments),
        }

        if tool_name == "create_spec":
            result = execute_tool_call("create_spec", arguments, config=config)
            spec_result = result.get("spec")
            if isinstance(spec_result, dict):
                spec_state = dict(spec_result)
        elif tool_name == "clone_spec":
            result = execute_tool_call("clone_spec", {"spec": arguments.get("spec") or spec_state}, config=config)
            spec_result = result.get("spec")
            if isinstance(spec_result, dict):
                spec_state = dict(spec_result)
        elif tool_name == "update_spec":
            result = execute_tool_call(
                "update_spec",
                {"spec": arguments.get("spec") or spec_state, "updates": arguments.get("updates") or {}},
                config=config,
            )
            spec_result = result.get("spec")
            if isinstance(spec_result, dict):
                spec_state = dict(spec_result)
        elif tool_name == "set_boundary_conditions":
            result = execute_tool_call(
                "set_boundary_conditions",
                {
                    "spec": arguments.get("spec") or spec_state,
                    "updates": arguments.get("updates") or {},
                    "reinitialize": arguments.get("reinitialize", True),
                },
                config=config,
            )
            spec_result = result.get("spec")
            if isinstance(spec_result, dict):
                spec_state = dict(spec_result)
        elif tool_name in run_tools:
            requested_spec = _coerce_dict(arguments.get("spec")) or _coerce_dict(spec_state)
            requested_spec = profiles.merge_spec(normalized_request_hints, requested_spec)
            if tool_name == "run_task" and requested_spec.get("task") is None and isinstance(spec_state, dict):
                requested_spec = dict(spec_state)

            required_defaults = _planner_required_defaults(config, requested_spec.get("profile", config.profile))
            required_default_fields: dict[str, Any] = {}
            for field_name, default_value in required_defaults.items():
                if requested_spec.get(field_name) is None:
                    requested_spec[field_name] = default_value
                    required_default_fields[field_name] = default_value

            validation = execute_tool_call("validate_spec", {"spec": requested_spec}, config=config)
            resolved_spec = dict(validation.get("resolved_spec") or profiles.normalize_spec(requested_spec, default_profile=config.profile))
            spec_state = dict(resolved_spec)
            combined_defaults = dict(required_default_fields)
            combined_defaults.update(_estimate_default_fields(requested_spec, resolved_spec, config))
            review_runs.append(
                {
                    "step_index": index,
                    "tool_name": tool_name,
                    "purpose": step.get("purpose"),
                    "requested_spec": requested_spec,
                    "resolved_spec": resolved_spec,
                    "defaults_used": combined_defaults,
                    "warnings": list(validation.get("warnings") or []),
                    "missing_fields": list(validation.get("missing_fields") or []),
                    "ok": bool(validation.get("ok")),
                }
            )
            for field_name in list(validation.get("missing_fields") or []):
                if field_name not in aggregate_missing_fields:
                    aggregate_missing_fields.append(field_name)
            resolved_step["arguments"]["spec"] = dict(resolved_spec)

        resolved_plan_steps.append(resolved_step)

    default_count = sum(1 for item in review_runs if item["defaults_used"])
    warning_count = sum(1 for item in review_runs if item["warnings"] or item["missing_fields"] or not item["ok"])
    if aggregate_missing_fields:
        summary = (
            "I cannot proceed to the defaults check yet because some required simulation inputs are still missing: "
            + ", ".join(aggregate_missing_fields)
            + "."
        )
        needs_confirmation_message = "Please provide the missing required values so I can finish preparing the run."
    elif default_count:
        default_parts: list[str] = []
        seen_default_parts: set[str] = set()
        for item in review_runs:
            for field, value in (item["defaults_used"] or {}).items():
                part = f"{field}={_format_default_value(value)}"
                if part not in seen_default_parts:
                    seen_default_parts.add(part)
                    default_parts.append(part)
        defaults_text = ", ".join(default_parts)
        summary = (
            "I checked the planned run specification and it relies on current defaults for some remaining settings. "
            f"I can use these defaults: {defaults_text}."
        )
        needs_confirmation_message = (
            "Reply 'use defaults' to continue, or tell me which values to change."
        )
    else:
        summary = f"I checked the planned run spec{'s' if len(review_runs) != 1 else ''}."
        if warning_count:
            summary += f" {warning_count} item{'s' if warning_count != 1 else ''} need attention."
        needs_confirmation_message = "Please review these resolved parameters. Confirm if you want me to keep them and unlock the run."

    return {
        "summary": summary,
        "runs": review_runs,
        "needs_confirmation_message": needs_confirmation_message,
        "requires_default_confirmation": bool(default_count),
        "missing_fields": aggregate_missing_fields,
        "resolved_plan_steps": resolved_plan_steps,
    }


def run_planner_turn(
    message: str,
    config: AgentRuntimeConfig,
    session_state: dict[str, Any] | None = None,
    max_iterations: int = 12,
) -> dict[str, Any]:
    del max_iterations
    current_state = dict(session_state or {})
    model_request: dict[str, Any] | None = None
    conversation_history = _normalize_history(current_state.get("conversation_history"))
    conversation_history = _append_history(conversation_history, "user", message)

    pending_plan = current_state.get("pending_plan")
    approved_plan = current_state.get("approved_plan")
    pending_spec_review = current_state.get("pending_spec_review")
    original_request = str(current_state.get("original_request") or message)
    stored_human_readable_plan = current_state.get("human_readable_plan")
    completed_runs = _normalize_completed_runs(current_state.get("completed_runs"))
    request_hints = _collect_request_hints(
        conversation_history=conversation_history,
        original_request=original_request,
        latest_message=message,
        default_profile=config.profile,
    )

    if approved_plan and message.strip() == "__execute_plan__":
        tool_trace: list[dict[str, Any]] = []
        for step in approved_plan:
            tool_name = str(step["tool_name"])
            arguments = _coerce_dict(step.get("arguments"))
            try:
                result = execute_tool_call(tool_name, arguments, config=config)
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            tool_trace.append(
                {
                    "tool_name": tool_name,
                    "purpose": step.get("purpose"),
                    "arguments": arguments,
                    "result": result,
                }
            )

        final_message = _summarize_results(
            original_request=original_request,
            tool_trace=tool_trace,
            config=config,
        )
        final_spec = _extract_last_spec_from_trace(tool_trace, config.profile)
        return {
            "status": "completed",
            "message": final_message,
            "spec": final_spec,
            "missing_fields": [],
            "session_state": {
                "mode": "planner",
                "pending_plan": None,
                "approved_plan": None,
                "original_request": original_request,
                "tool_trace": tool_trace,
                "request_hints": request_hints,
                "conversation_history": _append_history(conversation_history, "assistant", final_message),
            },
            "result": {
                "mode": "planner",
                "tool_trace": tool_trace,
            },
        }

    if pending_spec_review and approved_plan:
        plan_reply = _interpret_pending_plan_reply(
            user_message=message,
            original_request=original_request,
            pending_plan=approved_plan,
            plan_summary=current_state.get("plan_summary"),
            conversation_history=conversation_history,
            config=config,
        )
        model_request = _coerce_dict(plan_reply.pop("_debug_request_body"))

        if plan_reply["action"] == "approve":
            assistant_message = "Spec review approved. Start the run when you're ready."
            review_runs = pending_spec_review.get("runs") if isinstance(pending_spec_review, dict) else []
            preview_spec = (
                dict(review_runs[0].get("resolved_spec") or {})
                if review_runs and isinstance(review_runs[0], dict)
                else profiles.normalize_spec(None, default_profile=config.profile)
            )
            return {
                "status": "running",
                "message": assistant_message,
                "spec": preview_spec,
                "missing_fields": [],
                "session_state": {
                    "mode": "planner",
                    "pending_plan": None,
                    "approved_plan": approved_plan,
                    "pending_spec_review": None,
                    "original_request": original_request,
                    "tool_trace": [],
                    "plan_summary": current_state.get("plan_summary"),
                    "human_readable_plan": stored_human_readable_plan,
                    "request_hints": request_hints,
                    "conversation_history": _append_history(conversation_history, "assistant", assistant_message),
                },
                "result": {
                    "mode": "planner",
                    "spec_review": pending_spec_review,
                    "model_request": model_request,
                },
            }

        if plan_reply["action"] == "revise":
            revised_request = str(plan_reply.get("revised_request") or "").strip()
            if not revised_request:
                revised_request = original_request + "\nUpdate: " + message
            plan = _generate_plan(
                revised_request,
                config,
                conversation_history=conversation_history,
                original_request=original_request,
                last_plan_summary=str(current_state.get("plan_summary") or ""),
                last_tool_trace=_coerce_trace(current_state.get("tool_trace")),
                request_hints=request_hints,
                completed_runs=completed_runs,
            )
            model_request = _coerce_dict(plan.pop("_debug_request_body"))
            rendered_message = _render_plan_message(plan)
            return {
                "status": "needs_clarification",
                "message": rendered_message,
                "spec": profiles.normalize_spec(None, default_profile=config.profile),
                "missing_fields": [],
                "session_state": {
                    "mode": "planner",
                    "pending_plan": list(plan.get("tool_calls", [])),
                    "approved_plan": None,
                    "pending_spec_review": None,
                    "plan_summary": plan.get("summary"),
                    "human_readable_plan": plan.get("human_readable_plan"),
                    "original_request": revised_request,
                    "tool_trace": [],
                    "request_hints": request_hints,
                    "conversation_history": _append_history(conversation_history, "assistant", rendered_message),
                },
                "result": {
                    "mode": "planner",
                    "plan_preview": plan,
                    "model_request": model_request,
                },
            }

        assistant_message = str(
            plan_reply.get("assistant_message")
            or "I still have the reviewed spec ready. Please confirm it or tell me what to change."
        )
        return {
            "status": "needs_clarification",
            "message": assistant_message,
            "spec": profiles.normalize_spec(None, default_profile=config.profile),
            "missing_fields": [],
            "session_state": {
                **current_state,
                "conversation_history": _append_history(conversation_history, "assistant", assistant_message),
            },
            "result": {
                "mode": "planner",
                "spec_review": pending_spec_review,
                "model_request": model_request,
            },
        }

    if pending_plan:
        plan_reply = _interpret_pending_plan_reply(
            user_message=message,
            original_request=original_request,
            pending_plan=pending_plan,
            plan_summary=current_state.get("plan_summary"),
            conversation_history=conversation_history,
            config=config,
        )
        model_request = _coerce_dict(plan_reply.pop("_debug_request_body"))

        if plan_reply["action"] == "approve":
            spec_review = _build_spec_review(list(pending_plan), config, request_hints=request_hints)
            missing_fields = list(spec_review.get("missing_fields") or [])
            assistant_message = (
                f"{spec_review['summary']}\n\n{spec_review['needs_confirmation_message']}"
            )
            resolved_plan_steps = list(spec_review.get("resolved_plan_steps") or pending_plan)
            return {
                "status": "needs_clarification",
                "message": assistant_message,
                "spec": (
                    dict(spec_review["runs"][0]["resolved_spec"])
                    if spec_review.get("runs")
                    else profiles.normalize_spec(None, default_profile=config.profile)
                ),
                "missing_fields": missing_fields,
                "session_state": {
                    "mode": "planner",
                    "pending_plan": resolved_plan_steps if missing_fields else None,
                    "approved_plan": None if missing_fields else resolved_plan_steps,
                    "pending_spec_review": None if missing_fields else spec_review,
                    "original_request": original_request,
                    "tool_trace": [],
                    "plan_summary": current_state.get("plan_summary"),
                    "human_readable_plan": stored_human_readable_plan,
                    "request_hints": request_hints,
                    "conversation_history": _append_history(conversation_history, "assistant", assistant_message),
                },
                "result": {
                    "mode": "planner",
                    "spec_review": spec_review,
                    "model_request": model_request,
                    **(
                        {
                            "plan_preview": {
                                "summary": current_state.get("plan_summary"),
                                "human_readable_plan": stored_human_readable_plan,
                                "tool_calls": resolved_plan_steps,
                            }
                        }
                        if missing_fields
                        else {}
                    ),
                },
            }

        if plan_reply["action"] == "revise":
            revised_request = str(plan_reply.get("revised_request") or "").strip()
            if not revised_request:
                revised_request = original_request + "\nUpdate: " + message
            plan = _generate_plan(
                revised_request,
                config,
                conversation_history=conversation_history,
                original_request=original_request,
                last_plan_summary=str(current_state.get("plan_summary") or ""),
                last_tool_trace=_coerce_trace(current_state.get("tool_trace")),
                request_hints=request_hints,
                completed_runs=completed_runs,
            )
            model_request = _coerce_dict(plan.pop("_debug_request_body"))
            rendered_message = _render_plan_message(plan)
            return {
                "status": "needs_clarification",
                "message": rendered_message,
                "spec": profiles.normalize_spec(None, default_profile=config.profile),
                "missing_fields": [],
                "session_state": {
                    "mode": "planner",
                    "pending_plan": list(plan.get("tool_calls", [])),
                    "approved_plan": None,
                    "pending_spec_review": None,
                    "plan_summary": plan.get("summary"),
                    "human_readable_plan": plan.get("human_readable_plan"),
                    "original_request": revised_request,
                    "tool_trace": [],
                    "request_hints": request_hints,
                    "conversation_history": _append_history(conversation_history, "assistant", rendered_message),
                },
                "result": {
                    "mode": "planner",
                    "plan_preview": plan,
                    "model_request": model_request,
                },
            }

        assistant_message = str(
            plan_reply.get("assistant_message")
            or "I still have the previous plan ready. Please confirm it or tell me what to change."
        )
        return {
            "status": "needs_clarification",
            "message": assistant_message,
            "spec": profiles.normalize_spec(None, default_profile=config.profile),
            "missing_fields": [],
            "session_state": {
                **current_state,
                "conversation_history": _append_history(conversation_history, "assistant", assistant_message),
            },
            "result": {
                "mode": "planner",
                "plan_preview": {
                    "summary": current_state.get("plan_summary"),
                    "human_readable_plan": stored_human_readable_plan,
                    "tool_calls": pending_plan,
                },
                "model_request": model_request,
            },
        }

    plan = _generate_plan(
        message,
        config,
        conversation_history=conversation_history,
        original_request=current_state.get("original_request"),
        last_plan_summary=str(current_state.get("plan_summary") or ""),
        last_tool_trace=_coerce_trace(current_state.get("tool_trace")),
        request_hints=request_hints,
        completed_runs=completed_runs,
    )
    model_request = _coerce_dict(plan.pop("_debug_request_body"))
    rendered_message = _render_plan_message(plan)
    return {
        "status": "needs_clarification",
        "message": rendered_message,
        "spec": profiles.normalize_spec(None, default_profile=config.profile),
        "missing_fields": [],
        "session_state": {
            "mode": "planner",
            "pending_plan": list(plan.get("tool_calls", [])),
            "approved_plan": None,
            "pending_spec_review": None,
            "plan_summary": plan.get("summary"),
            "human_readable_plan": plan.get("human_readable_plan"),
            "original_request": message,
            "tool_trace": [],
            "request_hints": request_hints,
            "conversation_history": _append_history(conversation_history, "assistant", rendered_message),
        },
        "result": {
            "mode": "planner",
            "plan_preview": plan,
            "model_request": model_request,
        },
    }
