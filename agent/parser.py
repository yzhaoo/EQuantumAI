from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from agent.schemas import AgentResponse, AgentRuntimeConfig
from simulation import profiles, setup

OPTIONAL_CONFIRM_FIELDS = list(profiles.OPTIONAL_RUNTIME_FIELDS)
REQUIRED_FIELDS_BY_TASK = dict(profiles.REQUIRED_FIELDS_BY_TASK)
SPEC_FIELDS = list(profiles.SPEC_FIELDS)
PROFILES = profiles.PROFILES
TURN_INTENTS = [
    "update_spec",
    "confirm_defaults",
    "confirm_run",
    "confirm_setup_generation",
    "reject_defaults",
    "unknown",
]
UPDATE_FIELDS = [
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
NUM_PATTERN = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def build_runtime_args(config: AgentRuntimeConfig | None = None, **overrides):
    resolved = config or AgentRuntimeConfig()
    payload = {
        **resolved.to_namespace().__dict__,
        **overrides,
    }
    return AgentRuntimeConfig(**payload).to_namespace()


def normalize_spec(spec: dict[str, Any] | None, default_profile: str = "dotgate_center") -> dict[str, Any]:
    return profiles.normalize_spec(spec, default_profile=default_profile)


def merge_spec(base_spec: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    return profiles.merge_spec(base_spec, updates)


def apply_defaults(spec: dict[str, Any], args=None) -> dict[str, Any]:
    runtime_args = args or build_runtime_args(profile=spec.get("profile", "dotgate_center"))
    defaults = get_default_spec_values(runtime_args, spec["profile"])
    return profiles.apply_defaults_to_spec(spec, defaults, OPTIONAL_CONFIRM_FIELDS)


def get_missing_fields(spec: dict[str, Any]) -> list[str]:
    return profiles.get_missing_fields(spec)


def get_runtime_profile(spec_or_profile_name, **kwargs):
    return profiles.get_runtime_profile(spec_or_profile_name, **kwargs)


def ensure_compatible(spec: dict[str, Any], profile: dict[str, Any]) -> None:
    profiles.ensure_compatible(spec, profile)


def build_boundary_conditions(profile: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return profiles.build_boundary_conditions(profile, spec)


def magnetic_field_to_phi(b_field_t: float, unit_cell_area: float) -> float:
    return profiles.magnetic_field_to_phi(b_field_t, unit_cell_area)


def get_default_spec_values(args, profile_name):
    return profiles.get_default_spec_values(args, profile_name)


def get_required_fields(spec):
    return profiles.get_required_fields(spec)


def clarification_question_for_field(field_name):
    questions = {
        "task": "Do you want DOS or LDOS?",
        "lattice_type": "What lattice type should I use?",
        "device_shape": "Which device shape should I use, for example dotgate or squaregate_center?",
        "backgate_voltage": "What backgate voltage should I use, in volts?",
        "magnetic_field_T": "What magnetic field should I use, in Tesla?",
        "solve_self_consistent": "Do you want the full self-consistent FSC calculation? Reply yes or no.",
        "spacing0": "What spacing0 should I use for the sampling density function?",
        "density_k": "What k value should I use for the sampling density function?",
    }
    return questions.get(field_name, f"Please provide {field_name}.")


def parse_float_fragment(text):
    return float(text.replace(",", "").strip())


def extract_value(query, patterns, field_name):
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return parse_float_fragment(match.group(1))
    raise ValueError(f"Could not parse {field_name} from query: {query!r}")


def extract_value_optional(query, patterns):
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return parse_float_fragment(match.group(1))
    return None


def infer_task(query):
    lowered = query.lower()
    if "ldos" in lowered or "local density of states" in lowered:
        return "ldos"
    if "density of state" in lowered or "density of states" in lowered or re.search(r"\bdos\b", lowered):
        return "dos"
    raise ValueError("Could not infer whether you want DOS or LDOS from the query.")


def infer_lattice_type(query):
    lowered = query.lower()
    explicit_match = re.search(
        r"\blattice(?:\s+type)?\s*(?:=|to|to be|as|is)?\s*(square|honeycomb)\b",
        lowered,
    )
    if explicit_match:
        return explicit_match.group(1)
    if "honeycomb lattice" in lowered or "graphene" in lowered or re.search(r"\bhoneycomb\b", lowered):
        return "honeycomb"
    if "square lattice" in lowered or "sqaure lattice" in lowered or re.search(r"\bsquare\b", lowered):
        return "square"
    raise ValueError("Could not infer the lattice type from the query.")


def infer_task_optional(query):
    try:
        return infer_task(query)
    except ValueError:
        return None


def infer_lattice_type_optional(query):
    try:
        return infer_lattice_type(query)
    except ValueError:
        return None


def infer_device_shape_optional(query):
    lowered = query.lower()
    if (
        "squaregate center" in lowered
        or "squaregate_center" in lowered
        or "square gate center" in lowered
        or "square center gate" in lowered
        or "square center-gate" in lowered
        or "center square gate" in lowered
    ):
        return "squaregate_center"
    if "dotgate" in lowered or "dot gate" in lowered:
        return "dotgate"
    return None


def parse_query_partial(query, default_profile="dotgate_center"):
    solve_self_consistent = None
    if re.search(r"\b(no scf|skip self[- ]consistent|without self[- ]consistent)\b", query, flags=re.IGNORECASE):
        solve_self_consistent = False
    elif re.search(r"\bself[- ]consistent\b", query, flags=re.IGNORECASE):
        solve_self_consistent = True

    convergence_match = re.search(
        rf"(?:convergence[_ ]?tol(?:erance)?|fsc\.convergence_tol)\s*(?:=|to)?\s*(?:\[|\()?\s*({NUM_PATTERN})\s*[, ]+\s*({NUM_PATTERN})",
        query,
        flags=re.IGNORECASE,
    )
    convergence_tol = None
    if convergence_match:
        convergence_tol = [float(convergence_match.group(1)), float(convergence_match.group(2))]

    lowered = query.lower()
    ldos_method = None
    if re.search(r"\bkmeans(?:sample)?\b", lowered):
        ldos_method = "kmeanssample"
    elif re.search(r"\b(tf|thomas[- ]?fermi)\b", lowered):
        ldos_method = "TF"
    elif re.search(r"\b(ed|exact diagonalization)\b", lowered):
        ldos_method = "ED"

    return {
        "raw_query": query,
        "task": infer_task_optional(query),
        "profile": default_profile,
        "lattice_type": infer_lattice_type_optional(query),
        "device_shape": infer_device_shape_optional(query),
        "backgate_voltage": extract_value_optional(
            query,
            [
                r"backgate(?:\s+voltage)?\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*v?\b",
                r"\bvbg\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*v?\b",
            ],
        ),
        "magnetic_field_T": extract_value_optional(
            query,
            [
                rf"magnetic\s+field\s*(?:=|to)?\s*({NUM_PATTERN})\s*t\b",
                rf"\bb\s*(?:=|to)?\s*({NUM_PATTERN})\s*(?:t(?:esla)?)?\b",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*(?:magnetic\s+field|field|b)\b",
                rf"({NUM_PATTERN})\s*t(?:esla)?\b",
            ],
        ),
        "solve_self_consistent": solve_self_consistent,
        "spacing0": extract_value_optional(
            query,
            [
                rf"\bspacing0\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bspacing\s*0\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bbase\s+spacing\s*(?:=|to)?\s*({NUM_PATTERN})\b",
            ],
        ),
        "density_k": extract_value_optional(
            query,
            [
                rf"\bdensity[_ ]?k\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bsampling[_ ]?k\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bk\s*(?:=|to)?\s*({NUM_PATTERN})\b",
            ],
        ),
        "dielectric_constant": extract_value_optional(
            query,
            [
                rf"dielectric(?:\s+constant)?\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bepsilon(?:_r)?\s*(?:=|to)?\s*({NUM_PATTERN})\b",
            ],
        ),
        "gate_potential": extract_value_optional(
            query,
            [
                rf"(?<!back)gate(?:[_ ]+potential)?\s*(?:=|to)?\s*({NUM_PATTERN})\s*v?\b",
                rf"top\s+gate(?:\s+potential)?\s*(?:=|to)?\s*({NUM_PATTERN})\s*v?\b",
            ],
        ),
        "convergence_tol": convergence_tol,
        "Ncore": (
            int(extract_value_optional(
                query,
                [
                    rf"\bncore\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                    rf"\bcores?\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                    rf"\buse\s*({NUM_PATTERN})\s*for\s*ncore\b",
                    rf"\buse\s*({NUM_PATTERN})\s*for\s*cores?\b",
                ],
            ))
            if extract_value_optional(
                query,
                [
                    rf"\bncore\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                    rf"\bcores?\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                    rf"\buse\s*({NUM_PATTERN})\s*for\s*ncore\b",
                    rf"\buse\s*({NUM_PATTERN})\s*for\s*cores?\b",
                ],
            ) is not None
            else None
        ),
        "eta": extract_value_optional(
            query,
            [
                rf"\beta\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bbroadening\s*(?:=|to)?\s*({NUM_PATTERN})\b",
            ],
        ),
        "ldos_method": ldos_method,
    }


def empty_updates(default_profile="dotgate_center"):
    return {field: None for field in UPDATE_FIELDS}


def explicit_fields_from_updates(updates):
    normalized = dict(empty_updates())
    normalized.update(updates or {})
    return [field for field in UPDATE_FIELDS if normalized.get(field) is not None]


def normalize_turn_parse(parsed, default_profile="dotgate_center"):
    normalized = {
        "raw_query": "",
        "intent": "unknown",
        "updates": empty_updates(default_profile=default_profile),
        "_parser": None,
        "_fallback_used": False,
        "_parser_error": None,
    }
    if parsed is None:
        normalized["updates"]["profile"] = default_profile
        return normalized

    normalized["raw_query"] = parsed.get("raw_query", "") if isinstance(parsed, dict) else ""
    intent = parsed.get("intent", "unknown") if isinstance(parsed, dict) else "unknown"
    normalized["intent"] = intent if intent in TURN_INTENTS else "unknown"
    updates = dict(empty_updates(default_profile=default_profile))
    if isinstance(parsed, dict):
        updates.update(parsed.get("updates", {}) or {})
        for meta_key in ["_parser", "_fallback_used", "_parser_error"]:
            if meta_key in parsed:
                normalized[meta_key] = parsed[meta_key]
    if updates.get("profile") is None:
        updates["profile"] = None
    normalized["updates"] = updates
    return normalized


def user_turn_parse_schema(default_profile="dotgate_center"):
    return {
        "type": "object",
        "properties": {
            "raw_query": {"type": "string"},
            "intent": {"type": "string", "enum": TURN_INTENTS},
            "updates": {
                "type": "object",
                "properties": {
                    "task": {"type": ["string", "null"], "enum": ["dos", "ldos", None]},
                    "profile": {"type": ["string", "null"], "enum": [default_profile, None]},
                    "lattice_type": {"type": ["string", "null"], "enum": ["square", "honeycomb", None]},
                    "device_shape": {"type": ["string", "null"], "enum": ["dotgate", "squaregate_center", None]},
                    "backgate_voltage": {"type": ["number", "null"]},
                    "magnetic_field_T": {"type": ["number", "null"]},
                    "solve_self_consistent": {"type": ["boolean", "null"]},
                    "spacing0": {"type": ["number", "null"]},
                    "density_k": {"type": ["number", "null"]},
                    "dielectric_constant": {"type": ["number", "null"]},
                    "gate_potential": {"type": ["number", "null"]},
                    "convergence_tol": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "Ncore": {"type": ["integer", "null"]},
                    "eta": {"type": ["number", "null"]},
                    "ldos_method": {"type": ["string", "null"], "enum": ["TF", "ED", "kmeanssample", None]},
                },
                "required": UPDATE_FIELDS,
                "additionalProperties": False,
            },
        },
        "required": ["raw_query", "intent", "updates"],
        "additionalProperties": False,
    }


def build_user_turn_parser_prompt(default_profile="dotgate_center"):
    return (
        "You are a parser for a quantum simulation agent. "
        "Convert the latest user message into a structured partial update. "
        "Use the current session context to interpret terse replies. "
        "Do not invent values. Return only JSON matching the schema. "
        "Only include values in updates when the latest user message explicitly states or strongly implies them. "
        "Understand synonyms: "
        "B, field, magnetic field -> magnetic_field_T in Tesla. "
        "Vbg, backgate -> backgate_voltage in volts. "
        "cores, ncore, CPU cores -> Ncore. "
        "broadening -> eta. "
        "ED, exact diagonalization -> ldos_method='ED'. "
        "TF, Thomas-Fermi -> ldos_method='TF'. "
        "kmeans -> ldos_method='kmeanssample'. "
        "'start', 'run', 'proceed' -> confirm_run. "
        "'use defaults', 'use default', 'ok defaults', 'all good', 'continue', 'looks fine' -> confirm_defaults "
        "when the session is currently asking you to confirm defaults. "
        "'generate setup', 'create setup' -> confirm_setup_generation. "
        "'no', 'change defaults', 'not defaults' -> reject_defaults. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        f"If no profile is stated, do not invent one; use null in updates. The default profile is {default_profile}."
    )


def extract_json_text_from_response(payload):
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]

    raise ValueError("OpenAI response did not contain output_text content.")


def llm_parse_user_turn(query, default_profile, api_key, model, base_url, context):
    system_prompt = build_user_turn_parser_prompt(default_profile)
    context_text = json.dumps(context, indent=2, sort_keys=True)
    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": f"Session context:\n{context_text}"}]},
            {"role": "user", "content": [{"type": "input_text", "text": f"Latest user message:\n{query}"}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_user_turn_parse",
                "strict": True,
                "schema": user_turn_parse_schema(default_profile),
            }
        },
    }

    url = base_url.rstrip("/") + "/responses"
    data = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    parsed = json.loads(extract_json_text_from_response(payload))
    parsed["raw_query"] = query
    parsed["_parser"] = f"openai:{model}"
    parsed["_fallback_used"] = False
    return normalize_turn_parse(parsed, default_profile=default_profile)


def regex_parse_user_turn(
    query,
    default_profile="dotgate_center",
    current_spec=None,
    missing_fields=None,
    pending_default_fields=None,
    awaiting_defaults_confirmation=False,
    awaiting_setup_generation_confirmation=False,
    awaiting_run_confirmation=False,
):
    partial_spec = parse_query_partial(query, default_profile=default_profile)
    updates = {field: partial_spec.get(field) for field in UPDATE_FIELDS}
    updates["profile"] = None
    if not explicit_fields_from_updates(updates):
        target_fields = list(missing_fields or [])
        if awaiting_defaults_confirmation:
            target_fields = list(pending_default_fields or [])
        if len(target_fields) == 1:
            target_field = target_fields[0]
            parsed_value = parse_reply_for_field(target_field, query)
            if parsed_value is not None and target_field in updates:
                updates[target_field] = parsed_value
    explicit_updates = explicit_fields_from_updates(updates)
    intent = "unknown"
    if awaiting_setup_generation_confirmation and is_setup_generation_confirmation_reply(query):
        intent = "confirm_setup_generation"
    elif awaiting_run_confirmation and is_start_confirmation_reply(query):
        intent = "confirm_run"
    elif awaiting_defaults_confirmation and is_affirmative_reply(query):
        intent = "confirm_defaults"
    elif awaiting_defaults_confirmation and is_negative_reply(query):
        intent = "reject_defaults"
    elif explicit_updates:
        intent = "update_spec"

    return normalize_turn_parse(
        {
            "raw_query": query,
            "intent": intent,
            "updates": updates,
            "_parser": "regex",
            "_fallback_used": True,
        },
        default_profile=default_profile,
    )


def parser_name_from_args(args):
    return getattr(args, "parser", "regex")


def parse_user_turn(
    user_text,
    args,
    current_spec=None,
    missing_fields=None,
    pending_default_fields=None,
    awaiting_defaults_confirmation=False,
    awaiting_setup_generation_confirmation=False,
    awaiting_run_confirmation=False,
):
    default_profile = getattr(args, "profile", "dotgate_center")
    requested_parser = parser_name_from_args(args)
    context = {
        "current_spec": normalize_spec(current_spec, default_profile=default_profile)
        if current_spec is not None
        else normalize_spec(None, default_profile=default_profile),
        "missing_fields": list(missing_fields or []),
        "pending_default_fields": list(pending_default_fields or []),
        "awaiting_defaults_confirmation": bool(awaiting_defaults_confirmation),
        "awaiting_setup_generation_confirmation": bool(awaiting_setup_generation_confirmation),
        "awaiting_run_confirmation": bool(awaiting_run_confirmation),
    }

    if requested_parser == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = getattr(args, "openai_model", "gpt-4o-mini")
        allow_fallback = not getattr(args, "strict_openai", False)

        if api_key:
            try:
                return llm_parse_user_turn(
                    user_text,
                    default_profile=default_profile,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    context=context,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                if not allow_fallback:
                    raise
                turn_parse = regex_parse_user_turn(
                    user_text,
                    default_profile=default_profile,
                    current_spec=current_spec,
                    missing_fields=missing_fields,
                    pending_default_fields=pending_default_fields,
                    awaiting_defaults_confirmation=awaiting_defaults_confirmation,
                    awaiting_setup_generation_confirmation=awaiting_setup_generation_confirmation,
                    awaiting_run_confirmation=awaiting_run_confirmation,
                )
                turn_parse["_parser"] = f"regex_fallback_after_openai_error:{type(exc).__name__}"
                turn_parse["_fallback_used"] = True
                turn_parse["_parser_error"] = str(exc)
                return turn_parse

        if getattr(args, "strict_openai", False):
            raise RuntimeError("OPENAI_API_KEY is not set and strict OpenAI parsing was requested.")

        turn_parse = regex_parse_user_turn(
            user_text,
            default_profile=default_profile,
            current_spec=current_spec,
            missing_fields=missing_fields,
            pending_default_fields=pending_default_fields,
            awaiting_defaults_confirmation=awaiting_defaults_confirmation,
            awaiting_setup_generation_confirmation=awaiting_setup_generation_confirmation,
            awaiting_run_confirmation=awaiting_run_confirmation,
        )
        turn_parse["_parser"] = "regex_fallback_no_api_key"
        turn_parse["_fallback_used"] = True
        turn_parse["_parser_error"] = "OPENAI_API_KEY is not set."
        return turn_parse

    turn_parse = regex_parse_user_turn(
        user_text,
        default_profile=default_profile,
        current_spec=current_spec,
        missing_fields=missing_fields,
        pending_default_fields=pending_default_fields,
        awaiting_defaults_confirmation=awaiting_defaults_confirmation,
        awaiting_setup_generation_confirmation=awaiting_setup_generation_confirmation,
        awaiting_run_confirmation=awaiting_run_confirmation,
    )
    if requested_parser != "regex":
        turn_parse["_parser"] = f"regex_fallback_from_{requested_parser}"
        turn_parse["_fallback_used"] = True
    return turn_parse


def parse_numeric_reply(text, unit_patterns=None):
    patterns = []
    if unit_patterns:
        patterns.extend(unit_patterns)
    patterns.append(r"^\s*(-?\d+(?:\.\d+)?)\s*$")
    return extract_value_optional(text, patterns)


def parse_reply_for_field(field_name, text):
    lowered = text.lower().strip()

    if field_name == "task":
        return infer_task_optional(text)
    if field_name == "lattice_type":
        return infer_lattice_type_optional(text)
    if field_name == "device_shape":
        return infer_device_shape_optional(text)
    if field_name == "backgate_voltage":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                r"^\s*(-?\d+(?:\.\d+)?)\s*v(?:olt|olts)?\s*$",
                r"^\s*backgate(?:\s+voltage)?\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*v?\s*$",
            ],
        )
    if field_name == "magnetic_field_T":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                r"^\s*(-?\d+(?:\.\d+)?)\s*t(?:esla)?\s*$",
                r"^\s*b\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*(?:t(?:esla)?)?\s*$",
                r"^\s*magnetic\s+field\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*t?\s*$",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*(?:magnetic\s+field|field|b)\b",
            ],
        )
    if field_name == "solve_self_consistent":
        if is_affirmative_reply(text):
            return True
        if is_negative_reply(text) or re.search(
            r"\b(no scf|skip self[- ]consistent|without self[- ]consisten(?:t|cy))\b",
            lowered,
        ):
            return False
        return None
    if field_name == "spacing0":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                rf"\bspacing0\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bspacing\s*0\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bbase\s+spacing\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*spacing0\b",
            ],
        )
    if field_name == "density_k":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                rf"\bdensity[_ ]?k\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bsampling[_ ]?k\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\bk\s*(?:=|to)?\s*({NUM_PATTERN})\b",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*(?:density[_ ]?k|sampling[_ ]?k|k)\b",
            ],
        )
    if field_name == "dielectric_constant":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                rf"^\s*({NUM_PATTERN})\s*$",
                rf"^\s*dielectric(?:\s+constant)?\s*(?:=|to)?\s*({NUM_PATTERN})\s*$",
            ],
        )
    if field_name == "gate_potential":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                rf"({NUM_PATTERN})\s*v(?:olt|olts)?",
                rf"(?<!back)gate(?:[_ ]+potential)?\s*(?:=|to)?\s*({NUM_PATTERN})\s*v?",
            ],
        )
    if field_name == "Ncore":
        value = parse_numeric_reply(
            text,
            unit_patterns=[
                rf"\bncore\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                rf"\bcores?\s*(?:=|to|just)?\s*({NUM_PATTERN})\b",
                rf"\buse\s*({NUM_PATTERN})\s*cores?\b",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*ncore\b",
                rf"\buse\s*({NUM_PATTERN})\s*for\s*cores?\b",
                rf"^\s*({NUM_PATTERN})\s*$",
            ],
        )
        return int(value) if value is not None else None
    if field_name == "eta":
        return parse_numeric_reply(
            text,
            unit_patterns=[
                rf"^\s*({NUM_PATTERN})\s*$",
                rf"^\s*eta\s*(?:=|to)?\s*({NUM_PATTERN})\s*$",
            ],
        )
    if field_name == "ldos_method":
        if re.search(r"\bkmeans(?:sample)?\b", lowered):
            return "kmeanssample"
        if re.search(r"\b(tf|thomas[- ]?fermi)\b", lowered):
            return "TF"
        if re.search(r"\b(ed|exact diagonalization)\b", lowered):
            return "ED"
        return None
    if field_name == "convergence_tol":
        match = re.search(
            rf"(?:convergence[_ ]?tol(?:erance)?|fsc\.convergence_tol)?\s*(?:=|to)?\s*(?:\[|\()?\s*({NUM_PATTERN})\s*[, ]+\s*({NUM_PATTERN})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return [float(match.group(1)), float(match.group(2))]
        return None
    return None


def sanitize_optional_confirmation_fields(spec, explicit_fields):
    sanitized = dict(spec)
    for field in OPTIONAL_CONFIRM_FIELDS:
        if field not in explicit_fields:
            sanitized[field] = None
    return sanitized


def fields_needing_default_confirmation(spec, explicit_fields, defaults):
    pending = []
    for field in OPTIONAL_CONFIRM_FIELDS:
        if field in explicit_fields:
            continue
        if spec.get(field) is None and field in defaults:
            pending.append(field)
    return pending


def format_default_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:g}" for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_defaults_confirmation_message(fields, defaults):
    parts = [f"{field}={format_default_value(defaults[field])}" for field in fields]
    joined = ", ".join(parts)
    return (
        "I can use the current defaults for the remaining settings: "
        f"{joined}. Reply 'use defaults' to continue, or tell me which values to change."
    )


def is_affirmative_reply(text):
    return bool(
        re.search(
            r"\b(yes|yep|use defaults|use default|sounds good|looks good|looks fine|all good|all set|ok|okay|continue|proceed|keep them|g(?:o|e)\s+ahead)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def is_negative_reply(text):
    return bool(re.search(r"\b(no|change|override|different|not those)\b", text, flags=re.IGNORECASE))


def is_start_confirmation_reply(text):
    return bool(re.search(r"\b(yes|yep|start|run|go ahead|proceed|confirm|do it|looks good|ok|okay)\b", text, flags=re.IGNORECASE))


def is_setup_generation_confirmation_reply(text):
    return bool(
        re.search(
            r"\b(generate setup|create setup|new setup|go ahead|proceed|yes|ok|okay|do it|continue)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def build_setup_generation_confirmation_message(spec, setup_info):
    return (
        "These geometry/discretization parameters will generate a new setup before the simulation runs: "
        f"device_shape={format_default_value(spec.get('device_shape'))}, "
        f"lattice_type={format_default_value(spec.get('lattice_type'))}, "
        f"spacing0={format_default_value(spec.get('spacing0'))}, "
        f"k={format_default_value(spec.get('density_k'))}. "
        f"The new setup folder will be {setup_info['setup_dir']}. "
        "Reply 'generate setup' to continue, or tell me what to change."
    )


def build_final_confirmation_message(spec):
    summary = (
        f"task={spec.get('task')}, "
        f"lattice_type={spec.get('lattice_type')}, "
        f"device_shape={format_default_value(spec.get('device_shape'))}, "
        f"backgate_voltage={format_default_value(spec.get('backgate_voltage'))}, "
        f"magnetic_field_T={format_default_value(spec.get('magnetic_field_T'))}, "
        f"solve_self_consistent={format_default_value(spec.get('solve_self_consistent'))}, "
        f"spacing0={format_default_value(spec.get('spacing0'))}, "
        f"k={format_default_value(spec.get('density_k'))}, "
        f"dielectric_constant={format_default_value(spec.get('dielectric_constant'))}, "
        f"gate_potential={format_default_value(spec.get('gate_potential'))}, "
        f"convergence_tol={format_default_value(spec.get('convergence_tol'))}, "
        f"Ncore={format_default_value(spec.get('Ncore'))}, "
        f"eta={format_default_value(spec.get('eta'))}, "
        f"ldos_method={format_default_value(spec.get('ldos_method'))}"
    )
    return (
        "Please confirm before I start the simulation with: "
        f"{summary}. Reply 'start' to run, or tell me what to change."
    )


def build_session_state(
    spec,
    args,
    history=None,
    active=True,
    explicit_fields=None,
    pending_default_fields=None,
    defaults_confirmed=False,
    run_confirmed=False,
    setup_generation_confirmed=False,
    setup_generation_signature=None,
    last_parser_debug=None,
):
    normalized = normalize_spec(spec, default_profile=getattr(args, "profile", "dotgate_center"))
    defaults = get_default_spec_values(args, normalized["profile"])
    return {
        "active": active,
        "profile": normalized["profile"],
        "parser": parser_name_from_args(args),
        "openai_model": getattr(args, "openai_model", "gpt-4o-mini"),
        "spec": normalized,
        "missing_fields": get_missing_fields(normalized),
        "explicit_fields": list(explicit_fields or []),
        "default_values": defaults,
        "pending_default_fields": list(pending_default_fields or []),
        "defaults_confirmed": bool(defaults_confirmed),
        "run_confirmed": bool(run_confirmed),
        "setup_generation_confirmed": bool(setup_generation_confirmed),
        "setup_generation_signature": setup_generation_signature,
        "last_parser_debug": last_parser_debug,
        "history": list(history or []),
    }


def process_session_spec(spec, session_state, args, execute=True):
    missing_fields = get_missing_fields(spec)
    session_state["spec"] = normalize_spec(spec, default_profile=session_state.get("profile", "dotgate_center"))
    session_state["missing_fields"] = missing_fields
    session_state["active"] = True

    if missing_fields:
        question = clarification_question_for_field(missing_fields[0])
        session_state.setdefault("history", []).append({"role": "assistant", "text": question})
        return AgentResponse(
            status="needs_clarification",
            message=question,
            spec=session_state["spec"],
            missing_fields=list(session_state.get("missing_fields", [])),
            session_state=session_state,
        )

    pending_default_fields = session_state.get("pending_default_fields", [])
    if pending_default_fields and not session_state.get("defaults_confirmed", False):
        message = build_defaults_confirmation_message(pending_default_fields, session_state["default_values"])
        session_state.setdefault("history", []).append({"role": "assistant", "text": message})
        return AgentResponse(
            status="needs_clarification",
            message=message,
            spec=session_state["spec"],
            missing_fields=list(session_state.get("missing_fields", [])),
            session_state=session_state,
        )

    setup_info = setup.get_setup_generation_info(session_state["spec"])
    session_state["pending_setup_generation"] = setup_info["requires_generation"]
    session_state["pending_setup_dir"] = setup_info["setup_dir"]
    current_signature = setup_info["setup_dir"]
    if setup_info["requires_generation"]:
        prior_signature = session_state.get("setup_generation_signature")
        is_confirmed = session_state.get("setup_generation_confirmed", False) and prior_signature == current_signature
        if not is_confirmed:
            message = build_setup_generation_confirmation_message(session_state["spec"], setup_info)
            session_state.setdefault("history", []).append({"role": "assistant", "text": message})
            session_state["setup_generation_confirmed"] = False
            session_state["setup_generation_signature"] = current_signature
            return AgentResponse(
                status="needs_clarification",
                message=message,
                spec=session_state["spec"],
                missing_fields=list(session_state.get("missing_fields", [])),
                session_state=session_state,
            )
    else:
        session_state["setup_generation_confirmed"] = False
        session_state["setup_generation_signature"] = None

    if not session_state.get("run_confirmed", False):
        message = build_final_confirmation_message(session_state["spec"])
        session_state.setdefault("history", []).append({"role": "assistant", "text": message})
        return AgentResponse(
            status="needs_clarification",
            message=message,
            spec=session_state["spec"],
            missing_fields=list(session_state.get("missing_fields", [])),
            session_state=session_state,
        )

    if execute:
        from simulation.runner import run_spec

        session_state["active"] = False
        result = run_spec(session_state["spec"], config=AgentRuntimeConfig(**{
            "parser": getattr(args, "parser", "regex"),
            "openai_model": getattr(args, "openai_model", "gpt-4o-mini"),
            "strict_openai": getattr(args, "strict_openai", False),
            "profile": getattr(args, "profile", "dotgate_center"),
            "device_shape": getattr(args, "device_shape", "dotgate"),
            "output_dir": getattr(args, "output_dir", None),
            "no_scf": getattr(args, "no_scf", False),
            "ldos_method": getattr(args, "ldos_method", "ED"),
            "ncore": getattr(args, "ncore", 1),
            "moments": getattr(args, "moments", 256),
            "n_random": getattr(args, "n_random", 10),
            "eta": getattr(args, "eta", 0.00015),
            "eps": getattr(args, "eps", 0.05),
            "kernel": getattr(args, "kernel", "jackson"),
            "energy_points": getattr(args, "energy_points", 1024),
            "tol_poisson": getattr(args, "tol_poisson", 1e-3),
            "tol_ildos": getattr(args, "tol_ildos", 1e-2),
        }))
        session_state["history"].append({"role": "assistant", "text": "Simulation completed."})
        return AgentResponse(
            status="completed",
            message="Simulation completed.",
            spec=session_state["spec"],
            missing_fields=list(session_state.get("missing_fields", [])),
            session_state=session_state,
            result=result,
        )

    session_state["active"] = False
    session_state["history"].append({"role": "assistant", "text": "Specification complete. Starting simulation."})
    return AgentResponse(
        status="running",
        message="Specification complete. Starting simulation.",
        spec=session_state["spec"],
        missing_fields=list(session_state.get("missing_fields", [])),
        session_state=session_state,
    )


def start_turn(user_text: str, args=None, execute: bool = False) -> AgentResponse:
    runtime_args = args or build_runtime_args()
    turn_parse = parse_user_turn(user_text, runtime_args)
    partial_spec = {"raw_query": user_text, **turn_parse["updates"]}
    default_profile = turn_parse["updates"].get("profile") or getattr(runtime_args, "profile", "dotgate_center")
    partial_spec["profile"] = default_profile
    explicit_fields = explicit_fields_from_updates(turn_parse["updates"])
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, explicit_fields)
    defaults = get_default_spec_values(runtime_args, default_profile)
    pending_default_fields = fields_needing_default_confirmation(partial_spec, explicit_fields, defaults)
    session_state = build_session_state(
        partial_spec,
        runtime_args,
        history=[{"role": "user", "text": user_text}],
        active=True,
        explicit_fields=explicit_fields,
        pending_default_fields=pending_default_fields,
        defaults_confirmed=(len(pending_default_fields) == 0),
        run_confirmed=False,
        setup_generation_confirmed=False,
        setup_generation_signature=None,
        last_parser_debug={
            "parser": turn_parse.get("_parser"),
            "intent": turn_parse.get("intent"),
            "updates": turn_parse.get("updates"),
            "fallback_used": turn_parse.get("_fallback_used", False),
            "parser_error": turn_parse.get("_parser_error"),
        },
    )
    if turn_parse.get("intent") == "confirm_defaults" and not pending_default_fields:
        session_state["defaults_confirmed"] = True
    if turn_parse.get("intent") == "confirm_run":
        session_state["run_confirmed"] = True
    return process_session_spec(session_state["spec"], session_state, runtime_args, execute=execute)


def continue_turn(session_state: dict[str, Any], user_text: str, args=None, execute: bool = False) -> AgentResponse:
    runtime_args = args or build_runtime_args()
    existing_state = dict(session_state or {})
    existing_spec = normalize_spec(existing_state.get("spec"), default_profile=existing_state.get("profile", getattr(runtime_args, "profile", "dotgate_center")))
    history = list(existing_state.get("history", []))
    history.append({"role": "user", "text": user_text})
    explicit_fields = list(existing_state.get("explicit_fields", []))
    default_values = dict(existing_state.get("default_values", get_default_spec_values(runtime_args, existing_spec["profile"])))
    pending_default_fields = list(existing_state.get("pending_default_fields", []))
    defaults_confirmed = bool(existing_state.get("defaults_confirmed", False))
    run_confirmed = bool(existing_state.get("run_confirmed", False))
    setup_generation_confirmed = bool(existing_state.get("setup_generation_confirmed", False))
    setup_generation_signature = existing_state.get("setup_generation_signature")
    prior_missing = existing_state.get("missing_fields") or get_missing_fields(existing_spec)
    awaiting_defaults_confirmation = bool(pending_default_fields and not defaults_confirmed)
    awaiting_setup_generation_confirmation = bool(existing_state.get("pending_setup_generation") and not setup_generation_confirmed)
    awaiting_run_confirmation = bool(not pending_default_fields and defaults_confirmed and not run_confirmed and not awaiting_setup_generation_confirmation)

    turn_parse = parse_user_turn(
        user_text,
        runtime_args,
        current_spec=existing_spec,
        missing_fields=prior_missing,
        pending_default_fields=pending_default_fields,
        awaiting_defaults_confirmation=awaiting_defaults_confirmation,
        awaiting_setup_generation_confirmation=awaiting_setup_generation_confirmation,
        awaiting_run_confirmation=awaiting_run_confirmation,
    )
    partial_spec = {"raw_query": user_text, **turn_parse["updates"]}
    partial_spec["profile"] = turn_parse["updates"].get("profile") or existing_spec.get("profile", getattr(runtime_args, "profile", "dotgate_center"))
    new_explicit_fields = explicit_fields_from_updates(turn_parse["updates"])
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, new_explicit_fields)
    merged_spec = merge_spec(existing_spec, partial_spec)
    explicit_fields = sorted(set(explicit_fields) | set(new_explicit_fields))

    for field_name in prior_missing:
        if merged_spec.get(field_name) is None:
            parsed_value = parse_reply_for_field(field_name, user_text)
            if parsed_value is not None:
                merged_spec[field_name] = parsed_value
                explicit_fields = sorted(set(explicit_fields) | {field_name})

    for field_name in OPTIONAL_CONFIRM_FIELDS:
        parsed_value = parse_reply_for_field(field_name, user_text)
        if parsed_value is not None and merged_spec.get(field_name) != parsed_value:
            merged_spec[field_name] = parsed_value
            explicit_fields = sorted(set(explicit_fields) | {field_name})

    if pending_default_fields and not defaults_confirmed:
        meaningful_spec_change_detected = any(
            merged_spec.get(field_name) != existing_spec.get(field_name)
            for field_name in SPEC_FIELDS
            if field_name not in {"raw_query", "solve_self_consistent"}
        )
        override_detected = any(partial_spec.get(field_name) is not None for field_name in pending_default_fields)

        for field_name in list(pending_default_fields):
            if merged_spec.get(field_name) is None:
                parsed_value = parse_reply_for_field(field_name, user_text)
                if parsed_value is not None:
                    merged_spec[field_name] = parsed_value
                    explicit_fields = sorted(set(explicit_fields) | {field_name})
                    override_detected = True

        pending_default_fields = [field_name for field_name in pending_default_fields if merged_spec.get(field_name) is None]

        if turn_parse.get("intent") == "reject_defaults" or is_negative_reply(user_text):
            message = (
                "Please tell me which defaults to change. "
                "You can specify dielectric constant, gate potential, convergence tolerance, Ncore, eta, or ldos method."
            )
            updated_state = build_session_state(
                merged_spec,
                runtime_args,
                history=history + [{"role": "assistant", "text": message}],
                active=True,
                explicit_fields=explicit_fields,
                pending_default_fields=pending_default_fields,
                defaults_confirmed=False,
                run_confirmed=False,
                setup_generation_confirmed=setup_generation_confirmed,
                setup_generation_signature=setup_generation_signature,
                last_parser_debug={
                    "parser": turn_parse.get("_parser"),
                    "intent": turn_parse.get("intent"),
                    "updates": turn_parse.get("updates"),
                    "fallback_used": turn_parse.get("_fallback_used", False),
                    "parser_error": turn_parse.get("_parser_error"),
                },
            )
            return AgentResponse(
                status="needs_clarification",
                message=message,
                spec=updated_state["spec"],
                missing_fields=list(updated_state.get("missing_fields", [])),
                session_state=updated_state,
            )

        if turn_parse.get("intent") == "confirm_defaults" or override_detected or meaningful_spec_change_detected:
            merged_spec = profiles.apply_defaults_to_spec(merged_spec, default_values, pending_default_fields)
            defaults_confirmed = True
            pending_default_fields = []

    if not pending_default_fields and defaults_confirmed:
        setup_reply_changed_spec = any(
            merged_spec.get(field_name) != existing_spec.get(field_name)
            for field_name in SPEC_FIELDS
            if field_name not in {"raw_query", "solve_self_consistent"}
        )
        existing_setup_signature = setup.get_setup_generation_info(existing_spec)["setup_dir"]
        new_setup_signature = setup.get_setup_generation_info(merged_spec)["setup_dir"]
        setup_signature_changed = existing_setup_signature != new_setup_signature
        if setup_reply_changed_spec:
            if setup_signature_changed:
                setup_generation_confirmed = False
                setup_generation_signature = None
                run_confirmed = False
        elif existing_state.get("pending_setup_generation") and (
            turn_parse.get("intent") == "confirm_setup_generation" or is_setup_generation_confirmation_reply(user_text)
        ):
            setup_generation_confirmed = True
            setup_generation_signature = existing_state.get("pending_setup_dir")

    awaiting_setup_generation_confirmation = bool(existing_state.get("pending_setup_generation")) and not setup_generation_confirmed

    if not pending_default_fields and defaults_confirmed and not run_confirmed and not awaiting_setup_generation_confirmation:
        if turn_parse.get("intent") == "confirm_run":
            run_confirmed = True
        else:
            run_confirmed = is_start_confirmation_reply(user_text)

    updated_state = build_session_state(
        merged_spec,
        runtime_args,
        history=history,
        active=True,
        explicit_fields=explicit_fields,
        pending_default_fields=(pending_default_fields if not defaults_confirmed else []),
        defaults_confirmed=defaults_confirmed,
        run_confirmed=run_confirmed,
        setup_generation_confirmed=setup_generation_confirmed,
        setup_generation_signature=setup_generation_signature,
        last_parser_debug={
            "parser": turn_parse.get("_parser"),
            "intent": turn_parse.get("intent"),
            "updates": turn_parse.get("updates"),
            "fallback_used": turn_parse.get("_fallback_used", False),
            "parser_error": turn_parse.get("_parser_error"),
        },
    )
    if updated_state["pending_default_fields"] and defaults_confirmed:
        updated_state["pending_default_fields"] = []
    return process_session_spec(updated_state["spec"], updated_state, runtime_args, execute=execute)


def parse_turn(user_text: str, args=None, session_state: dict[str, Any] | None = None, execute: bool = False) -> AgentResponse:
    if session_state:
        return continue_turn(session_state, user_text, args=args, execute=execute)
    return start_turn(user_text, args=args, execute=execute)
