import argparse
import dataclasses
import glob
import hashlib
import inspect
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

MPL_CACHE_DIR = os.path.join("/tmp", "equantum_mpl_cache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)

import matplotlib

matplotlib.use(os.environ.get("EQUANTUM_MPL_BACKEND", "Agg"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
EQUANTUM_DIR = os.path.join(PACKAGE_ROOT, "Equantum")

if load_dotenv is not None:
    for dotenv_path in [
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.join(PROJECT_ROOT, ".env.local"),
        os.path.join(PACKAGE_ROOT, ".env"),
        os.path.join(PACKAGE_ROOT, ".env.local"),
    ]:
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path, override=False)

if EQUANTUM_DIR not in sys.path:
    sys.path.insert(0, EQUANTUM_DIR)

from EQsystem import System
from fsc import FSC

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None


def density_function_dotgate_center(z):
    spacing0 = 0.02
    k = 0.2
    if abs(z) < 3 * spacing0:
        return spacing0
    return spacing0 + k * z


def get_geoparams_hash(params, *funcs):
    params_repr = {}
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


def build_dotgate_center_profile(project_root):
    geoparams = {
        "lattice_type": "square",
        "box_size": ((-0.6, 0.6), (-0.6, 0.6), (-0.08, 0.08)),
        "sampling_density_function": density_function_dotgate_center,
        "quantum_center": (0, 0, 0),
    }

    data_root = os.path.join(project_root, "Datas", "dotgate_center")
    setup_root = os.path.join(data_root, "setup")
    setup_hash = get_geoparams_hash(geoparams, density_function_dotgate_center)

    return {
        "name": "dotgate_center",
        "lattice_type": "square",
        "geoparams": geoparams,
        "data_root": data_root,
        "setup_root": setup_root,
        "expected_setup_dir": os.path.join(setup_root, f"setup_{setup_hash}"),
        "config_filename": "updated_sites_dot.json",
        "boundary_defaults": {
            "gate": {"potential": -1.0},
            "dielectric": {"dielectric_constant": 4.0},
            "backgate": {"potential": 0.0},
        },
    }


PROFILES = {
    "dotgate_center": build_dotgate_center_profile(PROJECT_ROOT),
}

REQUIRED_FIELDS_BY_TASK = {
    "dos": ["lattice_type", "backgate_voltage", "magnetic_field_T"],
    "ldos": ["lattice_type", "backgate_voltage", "magnetic_field_T"],
}

OPTIONAL_CONFIRM_FIELDS = [
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
    "backgate_voltage",
    "magnetic_field_T",
    "solve_self_consistent",
    "dielectric_constant",
    "gate_potential",
    "convergence_tol",
    "Ncore",
    "eta",
    "ldos_method",
]

NUM_PATTERN = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def resolve_setup_dir(profile):
    expected_config = os.path.join(profile["expected_setup_dir"], profile["config_filename"])
    if os.path.exists(expected_config):
        return profile["expected_setup_dir"]

    pattern = os.path.join(profile["setup_root"], "setup_*", profile["config_filename"])
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No setup file matching {profile['config_filename']} was found under {profile['setup_root']}."
        )

    if len(candidates) > 1:
        print(
            f"Warning: expected setup hash not found, falling back to the newest available setup among {len(candidates)} matches.",
            file=sys.stderr,
        )

    newest_config = max(candidates, key=os.path.getmtime)
    return os.path.dirname(newest_config)


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
    if "square lattice" in lowered or "sqaure lattice" in lowered or re.search(r"\bsquare\b", lowered):
        return "square"
    if "honeycomb lattice" in lowered or "graphene" in lowered or re.search(r"\bhoneycomb\b", lowered):
        return "honeycomb"
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


def empty_spec(default_profile="dotgate_center"):
    return {
        "raw_query": "",
        "task": None,
        "profile": default_profile,
        "lattice_type": None,
        "backgate_voltage": None,
        "magnetic_field_T": None,
        "solve_self_consistent": True,
        "dielectric_constant": None,
        "gate_potential": None,
        "convergence_tol": None,
        "Ncore": None,
        "eta": None,
        "ldos_method": None,
    }


def normalize_spec(spec, default_profile="dotgate_center"):
    normalized = empty_spec(default_profile=default_profile)
    if spec is None:
        return normalized
    for key in SPEC_FIELDS:
        if key in spec:
            normalized[key] = spec[key]
    if normalized["profile"] is None:
        normalized["profile"] = default_profile
    if normalized["solve_self_consistent"] is None:
        normalized["solve_self_consistent"] = True
    return normalized


def parse_query(query, default_profile="dotgate_center"):
    task = infer_task(query)
    lattice_type = infer_lattice_type(query)
    backgate_voltage = extract_value(
        query,
        [
            r"backgate(?:\s+voltage)?\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*v?\b",
            r"\bvbg\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*v?\b",
        ],
        "backgate voltage",
    )
    magnetic_field_t = extract_value(
        query,
        [
            r"magnetic\s+field\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*t\b",
            r"\bb\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*t\b",
            r"(-?\d+(?:\.\d+)?)\s*t(?:esla)?\b",
        ],
        "magnetic field",
    )

    return {
        "raw_query": query,
        "task": task,
        "profile": default_profile,
        "lattice_type": lattice_type,
        "backgate_voltage": backgate_voltage,
        "magnetic_field_T": magnetic_field_t,
        "solve_self_consistent": True,
    }


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
        convergence_tol = [
            float(convergence_match.group(1)),
            float(convergence_match.group(2)),
        ]

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
                rf"\bb\s*(?:=|to)?\s*({NUM_PATTERN})\s*t\b",
                rf"({NUM_PATTERN})\s*t(?:esla)?\b",
            ],
        ),
        "solve_self_consistent": solve_self_consistent,
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


if BaseModel is not None:
    class QuerySpecModel(BaseModel):
        raw_query: str = Field(description="Original user request")
        task: str = Field(description="Simulation target", pattern="^(dos|ldos)$")
        profile: str = Field(description="Simulation profile name")
        lattice_type: str = Field(description="Lattice type", pattern="^(square|honeycomb)$")
        backgate_voltage: float = Field(description="Backgate voltage in volts")
        magnetic_field_T: float = Field(description="Magnetic field in Tesla")
        solve_self_consistent: bool = Field(description="Whether to run the FSC loop")
        dielectric_constant: Optional[float] = Field(default=None, description="Dielectric constant")
        gate_potential: Optional[float] = Field(default=None, description="Gate potential in volts")
        convergence_tol: Optional[list[float]] = Field(default=None, description="Two convergence tolerances [poisson, ildos]")
        Ncore: Optional[int] = Field(default=None, description="Number of CPU cores")
        eta: Optional[float] = Field(default=None, description="Broadening parameter")
        ldos_method: Optional[str] = Field(default=None, description="LDOS method")

    class PartialQuerySpecModel(BaseModel):
        raw_query: str = Field(description="Original user request")
        task: Optional[str] = Field(default=None, description="Simulation target")
        profile: str = Field(description="Simulation profile name")
        lattice_type: Optional[str] = Field(default=None, description="Lattice type")
        backgate_voltage: Optional[float] = Field(default=None, description="Backgate voltage in volts")
        magnetic_field_T: Optional[float] = Field(default=None, description="Magnetic field in Tesla")
        solve_self_consistent: Optional[bool] = Field(default=True, description="Whether to run the FSC loop")
        dielectric_constant: Optional[float] = Field(default=None, description="Dielectric constant")
        gate_potential: Optional[float] = Field(default=None, description="Gate potential in volts")
        convergence_tol: Optional[list[float]] = Field(default=None, description="Two convergence tolerances [poisson, ildos]")
        Ncore: Optional[int] = Field(default=None, description="Number of CPU cores")
        eta: Optional[float] = Field(default=None, description="Broadening parameter")
        ldos_method: Optional[str] = Field(default=None, description="LDOS method")

else:
    @dataclasses.dataclass
    class QuerySpecModel:
        raw_query: str
        task: str
        profile: str
        lattice_type: str
        backgate_voltage: float
        magnetic_field_T: float
        solve_self_consistent: bool
        dielectric_constant: Optional[float] = None
        gate_potential: Optional[float] = None
        convergence_tol: Optional[list[float]] = None
        Ncore: Optional[int] = None
        eta: Optional[float] = None
        ldos_method: Optional[str] = None

    @dataclasses.dataclass
    class PartialQuerySpecModel:
        raw_query: str
        task: Optional[str] = None
        profile: str = "dotgate_center"
        lattice_type: Optional[str] = None
        backgate_voltage: Optional[float] = None
        magnetic_field_T: Optional[float] = None
        solve_self_consistent: Optional[bool] = True
        dielectric_constant: Optional[float] = None
        gate_potential: Optional[float] = None
        convergence_tol: Optional[list[float]] = None
        Ncore: Optional[int] = None
        eta: Optional[float] = None
        ldos_method: Optional[str] = None


def query_spec_schema(default_profile="dotgate_center"):
    return {
        "type": "object",
        "properties": {
            "raw_query": {"type": "string"},
            "task": {"type": "string", "enum": ["dos", "ldos"]},
            "profile": {"type": "string", "enum": [default_profile]},
            "lattice_type": {"type": "string", "enum": ["square", "honeycomb"]},
            "backgate_voltage": {"type": "number"},
            "magnetic_field_T": {"type": "number"},
            "solve_self_consistent": {"type": "boolean"},
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
        "required": [
            "raw_query",
            "task",
            "profile",
            "lattice_type",
            "backgate_voltage",
            "magnetic_field_T",
            "solve_self_consistent",
            "dielectric_constant",
            "gate_potential",
            "convergence_tol",
            "Ncore",
            "eta",
            "ldos_method",
        ],
        "additionalProperties": False,
    }


def partial_query_spec_schema(default_profile="dotgate_center"):
    return {
        "type": "object",
        "properties": {
            "raw_query": {"type": "string"},
            "task": {"type": ["string", "null"], "enum": ["dos", "ldos", None]},
            "profile": {"type": "string", "enum": [default_profile]},
            "lattice_type": {"type": ["string", "null"], "enum": ["square", "honeycomb", None]},
            "backgate_voltage": {"type": ["number", "null"]},
            "magnetic_field_T": {"type": ["number", "null"]},
            "solve_self_consistent": {"type": ["boolean", "null"]},
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
        "required": [
            "raw_query",
            "task",
            "profile",
            "lattice_type",
            "backgate_voltage",
            "magnetic_field_T",
            "solve_self_consistent",
            "dielectric_constant",
            "gate_potential",
            "convergence_tol",
            "Ncore",
            "eta",
            "ldos_method",
        ],
        "additionalProperties": False,
    }


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


def llm_parse_query(query, default_profile, api_key, model, base_url):
    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return only structured JSON that matches the provided schema. "
        "Interpret user intent conservatively. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        "Use the provided profile unless the user explicitly asks for another supported profile. "
        "Set solve_self_consistent=true unless the user clearly asks to skip self-consistency."
    )

    request_body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_query_spec",
                "strict": True,
                "schema": query_spec_schema(default_profile),
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
    parsed["profile"] = default_profile
    return parsed


def llm_parse_query_partial(query, default_profile, api_key, model, base_url):
    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return only structured JSON that matches the provided schema. "
        "Populate only values explicitly stated or strongly implied in the latest user message. "
        "If a field is missing, return null for that field. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        "Use the provided profile unless the user explicitly asks for another supported profile. "
        "Set solve_self_consistent=false only if the user clearly asks to skip self-consistency."
    )

    request_body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_partial_query_spec",
                "strict": True,
                "schema": partial_query_spec_schema(default_profile),
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
    parsed["profile"] = default_profile
    return normalize_spec(parsed, default_profile=default_profile)


def langchain_parse_query(query, default_profile, model):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain parser requires langchain-openai. Install it with: pip install -U langchain-openai"
        ) from exc

    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return a structured object matching the schema exactly. "
        "Interpret the user request conservatively. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        f"Use profile='{default_profile}' unless the user explicitly asks for a different supported profile. "
        "Set solve_self_consistent=true unless the user clearly asks to skip self-consistency."
    )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        use_responses_api=True,
    )

    structured_llm = llm.with_structured_output(
        QuerySpecModel,
        method="json_schema",
        strict=True,
    )

    result = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", query),
        ]
    )

    if hasattr(result, "model_dump"):
        parsed = result.model_dump()
    elif dataclasses.is_dataclass(result):
        parsed = dataclasses.asdict(result)
    else:
        raise TypeError(f"Unexpected LangChain structured output type: {type(result).__name__}")

    parsed["raw_query"] = query
    parsed["profile"] = default_profile
    return parsed


def langchain_parse_query_partial(query, default_profile, model):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain parser requires langchain-openai. Install it with: pip install -U langchain-openai"
        ) from exc

    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return a structured object matching the schema exactly. "
        "Populate only values explicitly stated or strongly implied in the latest user message. "
        "If a field is missing, return null for that field. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        f"Use profile='{default_profile}' unless the user explicitly asks for a different supported profile. "
        "Set solve_self_consistent=false only if the user clearly asks to skip self-consistency."
    )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        use_responses_api=True,
    )

    structured_llm = llm.with_structured_output(
        PartialQuerySpecModel,
        method="json_schema",
        strict=True,
    )

    result = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", query),
        ]
    )

    if hasattr(result, "model_dump"):
        parsed = result.model_dump()
    elif dataclasses.is_dataclass(result):
        parsed = dataclasses.asdict(result)
    else:
        raise TypeError(f"Unexpected LangChain structured output type: {type(result).__name__}")

    parsed["raw_query"] = query
    parsed["profile"] = default_profile
    return normalize_spec(parsed, default_profile=default_profile)


def llm_parse_query_partial_with_context(query, default_profile, api_key, model, base_url, current_spec=None, pending_fields=None):
    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return only structured JSON that matches the provided schema. "
        "This is a multi-turn clarification workflow. "
        "Use the current partially-filled spec and the latest user message together. "
        "Only update values explicitly stated or strongly implied by the latest user message. "
        "If the user gives a terse follow-up like 'use 2 for Ncore', infer the intended field from the latest message in context. "
        "Do not invent values that were not stated. "
        "If a field is not updated in the latest message, return null for that field. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        "Use the provided profile unless the user explicitly asks for another supported profile. "
        "Set solve_self_consistent=false only if the user clearly asks to skip self-consistency."
    )

    context_text = json.dumps(
        {
            "current_spec": normalize_spec(current_spec, default_profile=default_profile),
            "pending_fields": list(pending_fields or []),
        },
        indent=2,
        sort_keys=True,
    )

    request_body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"Current session context:\n{context_text}"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"Latest user message:\n{query}"}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "equantum_partial_query_spec_with_context",
                "strict": True,
                "schema": partial_query_spec_schema(default_profile),
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
    parsed["profile"] = default_profile
    return normalize_spec(parsed, default_profile=default_profile)


def langchain_parse_query_partial_with_context(query, default_profile, model, current_spec=None, pending_fields=None):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain parser requires langchain-openai. Install it with: pip install -U langchain-openai"
        ) from exc

    system_prompt = (
        "You are a parser for a quantum simulation CLI. "
        "Return a structured object matching the schema exactly. "
        "This is a multi-turn clarification workflow. "
        "Use the current partially-filled spec and the latest user message together. "
        "Only update values explicitly stated or strongly implied by the latest user message. "
        "If the user gives a terse follow-up like 'use 2 for Ncore', infer the intended field from the latest message in context. "
        "Do not invent values that were not stated. "
        "If a field is not updated in the latest user message, return null for that field. "
        "Map 'density of state' or 'density of states' to task='dos'. "
        "Map 'local density of states' or 'ldos' to task='ldos'. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        f"Use profile='{default_profile}' unless the user explicitly asks for a different supported profile. "
        "Set solve_self_consistent=false only if the user clearly asks to skip self-consistency."
    )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        use_responses_api=True,
    )

    structured_llm = llm.with_structured_output(
        PartialQuerySpecModel,
        method="json_schema",
        strict=True,
    )

    context_text = json.dumps(
        {
            "current_spec": normalize_spec(current_spec, default_profile=default_profile),
            "pending_fields": list(pending_fields or []),
        },
        indent=2,
        sort_keys=True,
    )

    result = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", f"Current session context:\n{context_text}"),
            ("human", f"Latest user message:\n{query}"),
        ]
    )

    if hasattr(result, "model_dump"):
        parsed = result.model_dump()
    elif dataclasses.is_dataclass(result):
        parsed = dataclasses.asdict(result)
    else:
        raise TypeError(f"Unexpected LangChain structured output type: {type(result).__name__}")

    parsed["raw_query"] = query
    parsed["profile"] = default_profile
    return normalize_spec(parsed, default_profile=default_profile)


def parse_query_with_openai(query, default_profile, model, allow_fallback=True):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = "regex_fallback_no_api_key"
            return spec
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    try:
        spec = llm_parse_query_partial(
            query=query,
            default_profile=default_profile,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        spec["_parser"] = f"openai:{model}"
        return spec
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = f"regex_fallback_after_openai_error:{type(exc).__name__}"
            spec["_parser_error"] = str(exc)
            return spec
        raise


def parse_query_with_langchain(query, default_profile, model, allow_fallback=True):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = "regex_fallback_no_api_key"
            return spec
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    try:
        spec = langchain_parse_query_partial(
            query=query,
            default_profile=default_profile,
            model=model,
        )
        spec["_parser"] = f"langchain_openai:{model}"
        return spec
    except Exception as exc:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = f"regex_fallback_after_langchain_error:{type(exc).__name__}"
            spec["_parser_error"] = str(exc)
            return spec
        raise


def get_required_fields(spec):
    task = spec.get("task")
    if task is None:
        return ["task"]
    return list(REQUIRED_FIELDS_BY_TASK.get(task, []))


def get_missing_fields(spec):
    normalized = normalize_spec(spec, default_profile=spec.get("profile", "dotgate_center") if spec else "dotgate_center")
    return [field for field in get_required_fields(normalized) if normalized.get(field) is None]


def is_spec_complete(spec):
    return len(get_missing_fields(spec)) == 0


def clarification_question_for_field(field_name):
    questions = {
        "task": "Do you want DOS or LDOS?",
        "lattice_type": "What lattice type should I use?",
        "backgate_voltage": "What backgate voltage should I use, in volts?",
        "magnetic_field_T": "What magnetic field should I use, in Tesla?",
    }
    return questions.get(field_name, f"Please provide {field_name}.")


def merge_spec(existing_spec, new_partial_spec):
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
                r"^\s*b\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*t?\s*$",
                r"^\s*magnetic\s+field\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*t?\s*$",
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


def parser_name_from_args(args):
    return getattr(args, "parser", "langchain")


def get_default_spec_values(args, profile_name):
    profile = PROFILES[profile_name]
    return {
        "dielectric_constant": float(profile["boundary_defaults"]["dielectric"]["dielectric_constant"]),
        "gate_potential": float(profile["boundary_defaults"]["gate"]["potential"]),
        "convergence_tol": [float(getattr(args, "tol_poisson", 1e-3)), float(getattr(args, "tol_ildos", 1e-2))],
        "Ncore": int(getattr(args, "ncore", 1)),
        "eta": float(getattr(args, "eta", 0.00015)),
        "ldos_method": getattr(args, "ldos_method", "ED"),
    }


def explicit_fields_from_spec(spec):
    explicit = []
    for key in SPEC_FIELDS:
        if key in {"raw_query", "profile"}:
            continue
        value = spec.get(key)
        if value is not None:
            explicit.append(key)
    return explicit


def explicit_fields_from_turn(parsed_spec, default_profile="dotgate_center"):
    normalized = normalize_spec(parsed_spec, default_profile=default_profile)
    explicit = []
    for key in SPEC_FIELDS:
        if key in {"raw_query", "profile"}:
            continue
        value = normalized.get(key)
        if value is not None:
            explicit.append(key)
    return explicit


def sanitize_optional_confirmation_fields(spec, explicit_fields):
    sanitized = dict(spec)
    for field in OPTIONAL_CONFIRM_FIELDS:
        if field not in explicit_fields:
            sanitized[field] = None
    return sanitized


def apply_defaults_to_spec(spec, defaults, fields):
    updated = dict(spec)
    for field in fields:
        if updated.get(field) is None and field in defaults:
            value = defaults[field]
            updated[field] = list(value) if isinstance(value, list) else value
    return updated


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
    return bool(re.search(r"\b(yes|yep|use defaults|sounds good|looks good|ok|okay|go ahead|keep them)\b", text, flags=re.IGNORECASE))


def is_negative_reply(text):
    return bool(re.search(r"\b(no|change|override|different|not those)\b", text, flags=re.IGNORECASE))


def is_start_confirmation_reply(text):
    return bool(re.search(r"\b(yes|yep|start|run|go ahead|proceed|confirm|do it|looks good|ok|okay)\b", text, flags=re.IGNORECASE))


def build_final_confirmation_message(spec):
    summary = (
        f"task={spec.get('task')}, "
        f"lattice_type={spec.get('lattice_type')}, "
        f"backgate_voltage={format_default_value(spec.get('backgate_voltage'))}, "
        f"magnetic_field_T={format_default_value(spec.get('magnetic_field_T'))}, "
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


def parse_user_text_partial(user_text, args, current_spec=None, pending_fields=None):
    parser_name = parser_name_from_args(args)
    default_profile = getattr(args, "profile", "dotgate_center")
    model = getattr(args, "openai_model", "gpt-4o-mini")
    allow_fallback = not getattr(args, "strict_openai", False)

    if parser_name == "langchain":
        if current_spec is not None or pending_fields:
            return parse_query_with_langchain_context(
                user_text,
                default_profile=default_profile,
                model=model,
                current_spec=current_spec,
                pending_fields=pending_fields,
                allow_fallback=allow_fallback,
            )
        return parse_query_with_langchain(
            user_text,
            default_profile=default_profile,
            model=model,
            allow_fallback=allow_fallback,
        )
    if parser_name == "openai":
        if current_spec is not None or pending_fields:
            return parse_query_with_openai_context(
                user_text,
                default_profile=default_profile,
                model=model,
                current_spec=current_spec,
                pending_fields=pending_fields,
                allow_fallback=allow_fallback,
            )
        return parse_query_with_openai(
            user_text,
            default_profile=default_profile,
            model=model,
            allow_fallback=allow_fallback,
        )
    spec = parse_query_partial(user_text, default_profile=default_profile)
    spec["_parser"] = "regex"
    return spec


def parse_query_with_openai_context(query, default_profile, model, current_spec=None, pending_fields=None, allow_fallback=True):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = "regex_fallback_no_api_key"
            return spec
        raise RuntimeError("OPENAI_API_KEY is not set and strict OpenAI parsing was requested.")

    try:
        spec = llm_parse_query_partial_with_context(
            query,
            default_profile=default_profile,
            api_key=api_key,
            model=model,
            base_url=base_url,
            current_spec=current_spec,
            pending_fields=pending_fields,
        )
        spec["_parser"] = "openai"
        return spec
    except Exception:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = "regex_fallback_openai_error"
            return spec
        raise


def parse_query_with_langchain_context(query, default_profile, model, current_spec=None, pending_fields=None, allow_fallback=True):
    try:
        spec = langchain_parse_query_partial_with_context(
            query,
            default_profile=default_profile,
            model=model,
            current_spec=current_spec,
            pending_fields=pending_fields,
        )
        spec["_parser"] = "langchain"
        return spec
    except Exception:
        if allow_fallback:
            spec = parse_query_partial(query, default_profile=default_profile)
            spec["_parser"] = "regex_fallback_langchain_error"
            return spec
        raise


def build_session_state(spec, args, history=None, active=True, explicit_fields=None, pending_default_fields=None, defaults_confirmed=False, run_confirmed=False):
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
        "history": list(history or []),
    }


def agent_response(status, message, spec, session_state, result=None):
    response = {
        "status": status,
        "message": message,
        "spec": normalize_spec(spec, default_profile=session_state.get("profile", "dotgate_center")),
        "missing_fields": list(session_state.get("missing_fields", [])),
        "session_state": session_state,
    }
    if result is not None:
        response["result"] = result
    return response


def process_session_spec(spec, session_state, args, execute=True):
    missing_fields = get_missing_fields(spec)
    session_state["spec"] = normalize_spec(spec, default_profile=session_state.get("profile", "dotgate_center"))
    session_state["missing_fields"] = missing_fields
    session_state["active"] = True

    if missing_fields:
        question = clarification_question_for_field(missing_fields[0])
        session_state.setdefault("history", []).append({"role": "assistant", "text": question})
        return agent_response(
            "needs_clarification",
            question,
            session_state["spec"],
            session_state,
        )

    pending_default_fields = session_state.get("pending_default_fields", [])
    if pending_default_fields and not session_state.get("defaults_confirmed", False):
        message = build_defaults_confirmation_message(pending_default_fields, session_state["default_values"])
        session_state.setdefault("history", []).append({"role": "assistant", "text": message})
        return agent_response(
            "needs_clarification",
            message,
            session_state["spec"],
            session_state,
        )

    if not session_state.get("run_confirmed", False):
        message = build_final_confirmation_message(session_state["spec"])
        session_state.setdefault("history", []).append({"role": "assistant", "text": message})
        return agent_response(
            "needs_clarification",
            message,
            session_state["spec"],
            session_state,
        )

    if execute:
        session_state["active"] = False
        result = run_query(session_state["spec"], args)
        session_state["history"].append({"role": "assistant", "text": "Simulation completed."})
        return agent_response(
            "completed",
            "Simulation completed.",
            session_state["spec"],
            session_state,
            result=result,
        )

    session_state["active"] = False
    session_state["history"].append({"role": "assistant", "text": "Specification complete. Starting simulation."})
    return agent_response(
        "running",
        "Specification complete. Starting simulation.",
        session_state["spec"],
        session_state,
    )


def start_agent_turn(user_text, args, execute=True):
    partial_spec = parse_user_text_partial(user_text, args)
    partial_spec["raw_query"] = user_text
    explicit_reference_spec = parse_query_partial(
        user_text,
        default_profile=getattr(args, "profile", "dotgate_center"),
    )
    explicit_fields = sorted(
        set(explicit_fields_from_spec(explicit_reference_spec))
        | set(explicit_fields_from_turn(partial_spec, default_profile=getattr(args, "profile", "dotgate_center")))
    )
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, explicit_fields)
    defaults = get_default_spec_values(args, partial_spec.get("profile", getattr(args, "profile", "dotgate_center")))
    pending_default_fields = fields_needing_default_confirmation(partial_spec, explicit_fields, defaults)
    session_state = build_session_state(
        partial_spec,
        args,
        history=[{"role": "user", "text": user_text}],
        active=True,
        explicit_fields=explicit_fields,
        pending_default_fields=pending_default_fields,
        defaults_confirmed=(len(pending_default_fields) == 0),
        run_confirmed=False,
    )
    return process_session_spec(session_state["spec"], session_state, args, execute=execute)


def continue_agent_turn(session_state, user_text, args, execute=True):
    existing_state = dict(session_state or {})
    existing_spec = normalize_spec(existing_state.get("spec"), default_profile=existing_state.get("profile", getattr(args, "profile", "dotgate_center")))
    history = list(existing_state.get("history", []))
    history.append({"role": "user", "text": user_text})
    explicit_fields = list(existing_state.get("explicit_fields", []))
    default_values = dict(existing_state.get("default_values", get_default_spec_values(args, existing_spec["profile"])))
    pending_default_fields = list(existing_state.get("pending_default_fields", []))
    defaults_confirmed = bool(existing_state.get("defaults_confirmed", False))
    run_confirmed = bool(existing_state.get("run_confirmed", False))
    prior_missing = existing_state.get("missing_fields") or get_missing_fields(existing_spec)

    partial_spec = parse_user_text_partial(
        user_text,
        args,
        current_spec=existing_spec,
        pending_fields=pending_default_fields or prior_missing,
    )
    partial_spec["raw_query"] = user_text
    explicit_reference_spec = parse_query_partial(
        user_text,
        default_profile=existing_spec.get("profile", getattr(args, "profile", "dotgate_center")),
    )
    new_explicit_fields = sorted(
        set(explicit_fields_from_spec(explicit_reference_spec))
        | set(explicit_fields_from_turn(partial_spec, default_profile=existing_spec.get("profile", getattr(args, "profile", "dotgate_center"))))
    )
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, new_explicit_fields)
    merged_spec = merge_spec(existing_spec, partial_spec)
    explicit_fields = sorted(set(explicit_fields) | set(new_explicit_fields))

    for field_name in prior_missing:
        if merged_spec.get(field_name) is None:
            parsed_value = parse_reply_for_field(field_name, user_text)
            if parsed_value is not None:
                merged_spec[field_name] = parsed_value
                explicit_fields = sorted(set(explicit_fields) | {field_name})

    if pending_default_fields and not defaults_confirmed:
        override_detected = any(
            partial_spec.get(field_name) is not None
            for field_name in pending_default_fields
        )
        for field_name in list(pending_default_fields):
            if merged_spec.get(field_name) is None:
                parsed_value = parse_reply_for_field(field_name, user_text)
                if parsed_value is not None:
                    merged_spec[field_name] = parsed_value
                    explicit_fields = sorted(set(explicit_fields) | {field_name})
                    override_detected = True

        pending_default_fields = [
            field_name for field_name in pending_default_fields
            if merged_spec.get(field_name) is None
        ]

        if is_affirmative_reply(user_text) or override_detected:
            merged_spec = apply_defaults_to_spec(merged_spec, default_values, pending_default_fields)
            defaults_confirmed = True
            pending_default_fields = []
        elif is_negative_reply(user_text):
            message = (
                "Please tell me which defaults to change. "
                "You can specify dielectric constant, gate potential, convergence tolerance, Ncore, eta, or ldos method."
            )
            updated_state = build_session_state(
                merged_spec,
                args,
                history=history + [{"role": "assistant", "text": message}],
                active=True,
                explicit_fields=explicit_fields,
                pending_default_fields=pending_default_fields,
                defaults_confirmed=False,
                run_confirmed=False,
            )
            return agent_response(
                "needs_clarification",
                message,
                updated_state["spec"],
                updated_state,
            )

    if not pending_default_fields and defaults_confirmed and not run_confirmed:
        run_confirmed = is_start_confirmation_reply(user_text)

    updated_state = build_session_state(
        merged_spec,
        args,
        history=history,
        active=True,
        explicit_fields=explicit_fields,
        pending_default_fields=(
            pending_default_fields
            if not defaults_confirmed
            else []
        ),
        defaults_confirmed=defaults_confirmed,
        run_confirmed=run_confirmed,
    )
    if updated_state["pending_default_fields"] and defaults_confirmed:
        updated_state["pending_default_fields"] = []
    return process_session_spec(updated_state["spec"], updated_state, args, execute=execute)


def magnetic_field_to_phi(b_field_t, unit_cell_area):
    return b_field_t * sc.e * unit_cell_area / sc.h


def build_boundary_conditions(profile, spec):
    updates = json.loads(json.dumps(profile["boundary_defaults"]))
    updates["backgate"]["potential"] = float(spec["backgate_voltage"])
    if spec.get("gate_potential") is not None:
        updates["gate"]["potential"] = float(spec["gate_potential"])
    if spec.get("dielectric_constant") is not None:
        updates["dielectric"]["dielectric_constant"] = float(spec["dielectric_constant"])
    return updates


def ensure_compatible(spec, profile):
    if spec["lattice_type"] != profile["lattice_type"]:
        raise ValueError(
            f"Query requested lattice_type={spec['lattice_type']!r}, "
            f"but profile {profile['name']!r} is configured for {profile['lattice_type']!r}."
        )


def summarize_run(fsc, spec, phi, artifact_dir):
    return {
        "query": spec["raw_query"],
        "task": spec["task"],
        "profile": spec["profile"],
        "lattice_type": spec["lattice_type"],
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


def save_dos_artifacts(artifact_dir, energy, rho):
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


def save_ldos_artifacts(artifact_dir, fsc, site_mode="center"):
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

    return site_id


def make_artifact_dir(profile, base_output_dir=None):
    root = base_output_dir or os.path.join(profile["setup_root"], "agent_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(root, timestamp)
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def run_query(spec, args):
    profile = PROFILES[spec["profile"]]
    ensure_compatible(spec, profile)
    spec = apply_defaults_to_spec(spec, get_default_spec_values(args, spec["profile"]), OPTIONAL_CONFIRM_FIELDS)

    setup_dir = resolve_setup_dir(profile)
    config_file = os.path.join(setup_dir, profile["config_filename"])
    artifact_dir = make_artifact_dir(profile, args.output_dir)

    syst = System(
        profile["geoparams"],
        config_file=config_file,
        ifqsystem=True,
        quantum_builder="default",
    )

    phi = magnetic_field_to_phi(spec["magnetic_field_T"], syst.unit_cell_area)
    qparams = {"Ufunc": lambda site: 0, "phi": phi}
    fsc = FSC(syst, ifinitial=False, qparams=qparams, approx="TF")

    fsc.update_BC(build_boundary_conditions(profile, spec), ifinitial=True)
    fsc.Ncore = int(spec.get("Ncore", args.ncore))
    fsc.convergence_tol = list(spec.get("convergence_tol", [args.tol_poisson, args.tol_ildos]))

    if spec["solve_self_consistent"] and not args.no_scf:
        fsc.solve(
            syst,
            save=True,
            snapshot_mode="final_only",
            snapshot_folder=artifact_dir,
            ldos_method=spec.get("ldos_method", args.ldos_method),
            save_ildos=True,
            eta=spec.get("eta", args.eta),
            M=args.moments,
            eps=args.eps,
            kernel=args.kernel,
        )

    energy_grid = np.linspace(-6 * syst.t, 6 * syst.t, args.energy_points)

    result = summarize_run(fsc, spec, phi, artifact_dir)
    result["setup_dir"] = setup_dir
    result["config_file"] = config_file

    if spec["task"] == "dos":
        energy, rho = fsc.qsystem.get_dos(
            w=energy_grid,
            M=args.moments,
            n_random=args.n_random,
            eps=args.eps,
            kernel=args.kernel,
        )
        save_dos_artifacts(artifact_dir, energy, rho)
        result["artifacts"] = {
            "plot": os.path.join(artifact_dir, "dos.png"),
            "data": os.path.join(artifact_dir, "dos_data.npz"),
        }
    else:
        site_id = save_ldos_artifacts(artifact_dir, fsc, site_mode="center")
        result["ldos_site_id"] = site_id
        result["artifacts"] = {
            "plot": os.path.join(artifact_dir, "ldos.png"),
            "data": os.path.join(artifact_dir, "ldos_data.npz"),
        }

    with open(os.path.join(artifact_dir, "query_spec.json"), "w") as handle:
        json.dump(spec, handle, indent=2)

    with open(os.path.join(artifact_dir, "run_summary.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Natural-language query runner for EQuantum dotgate_center workflows."
    )
    parser.add_argument("query", help="Natural-language request, for example: calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1T")
    parser.add_argument("--profile", default="dotgate_center", choices=sorted(PROFILES.keys()))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parser", default="langchain", choices=["langchain", "openai", "regex"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--strict-openai", action="store_true")
    parser.add_argument("--no-scf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ldos-method", default="ED", choices=["TF", "ED", "kmeanssample"])
    parser.add_argument("--ncore", type=int, default=20)
    parser.add_argument("--moments", type=int, default=256)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.00015)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--kernel", default="jackson")
    parser.add_argument("--energy-points", type=int, default=1024)
    parser.add_argument("--tol-poisson", type=float, default=1e-3)
    parser.add_argument("--tol-ildos", type=float, default=1e-2)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    response = start_agent_turn(args.query, args, execute=not args.dry_run)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
