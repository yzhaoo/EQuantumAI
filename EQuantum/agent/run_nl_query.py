import argparse
import contextlib
import dataclasses
import glob
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
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


DEFAULT_DOTGATE_SPACING0 = 0.02
DEFAULT_DOTGATE_K = 0.2
DEVICE_SHAPE_DATA_DIRS = {
    "dotgate": "dotgate_center",
    "squaregate_center": "squaregate_center",
}


def build_density_function_dotgate_center(spacing0=DEFAULT_DOTGATE_SPACING0, k=DEFAULT_DOTGATE_K):
    def density_function(z):
        if abs(z) < 3 * spacing0:
            return spacing0
        return spacing0 + k * z

    return density_function


def density_function_dotgate_center(z):
    spacing0 = DEFAULT_DOTGATE_SPACING0
    k = DEFAULT_DOTGATE_K
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


def build_dotgate_center_profile(
    project_root,
    spacing0=DEFAULT_DOTGATE_SPACING0,
    density_k=DEFAULT_DOTGATE_K,
    lattice_type="square",
    device_shape="dotgate",
):
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

REQUIRED_FIELDS_BY_TASK = {
    "dos": ["lattice_type", "backgate_voltage", "magnetic_field_T"],
    "ldos": ["lattice_type", "backgate_voltage", "magnetic_field_T"],
}

OPTIONAL_CONFIRM_FIELDS = [
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


def get_profile_defaults(profile_name):
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown profile: {profile_name}")
    return PROFILES[profile_name]


def get_runtime_profile(spec_or_profile_name, spacing0=None, density_k=None, lattice_type=None, device_shape=None):
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


def setup_metadata_paths(profile):
    setup_dir = profile["expected_setup_dir"]
    return {
        "setup_dir": setup_dir,
        "sites_file": os.path.join(setup_dir, profile.get("sites_filename", "sites.json")),
        "config_file": os.path.join(setup_dir, profile["config_filename"]),
        "hash_file": os.path.join(setup_dir, profile.get("hash_filename", "geoparams_hash.txt")),
        "readme_file": os.path.join(setup_dir, profile.get("readme_filename", "README.txt")),
    }


def export_setup_sites(profile):
    paths = setup_metadata_paths(profile)
    if os.path.exists(paths["sites_file"]):
        return paths

    os.makedirs(paths["setup_dir"], exist_ok=True)
    syst = System(profile["geoparams"], assign_mat=False, ifqsystem=False)
    syst.export_sites(filename=paths["sites_file"])

    setup_hash = os.path.basename(paths["setup_dir"]).replace("setup_", "", 1)
    with open(paths["hash_file"], "w") as handle:
        handle.write(setup_hash)

    with open(paths["readme_file"], "w") as handle:
        handle.write("Parameter settings for this setup:\n")
        handle.write(f"lattice_type: {profile['geoparams']['lattice_type']}\n")
        handle.write(f"box_size: {profile['geoparams']['box_size']}\n")
        handle.write(f"spacing0: {profile['density_defaults']['spacing0']}\n")
        handle.write(f"density_k: {profile['density_defaults']['density_k']}\n")
        handle.write(f"quantum_center: {profile['geoparams']['quantum_center']}\n")

    return paths


def build_blender_update_command(profile, spec):
    paths = setup_metadata_paths(profile)
    device_shape = spec.get("device_shape") or profile.get("device_shape")
    blend_files = profile.get("blend_files", {})
    default_blend_file = blend_files.get(device_shape, profile.get("blend_file", ""))
    blend_file = os.environ.get("EQUANTUM_BLENDER_FILE", default_blend_file)
    env_script_file = os.environ.get("EQUANTUM_BLENDER_SCRIPT")
    script_file = env_script_file or profile.get("blender_script", "")
    if env_script_file and not os.path.exists(env_script_file):
        script_file = profile.get("blender_script", "")
    blender_bin = resolve_blender_executable(os.environ.get("EQUANTUM_BLENDER_BIN", "blender"))
    command_template = os.environ.get("EQUANTUM_BLENDER_COMMAND_TEMPLATE")

    format_vars = {
        "blender_bin": blender_bin,
        "blend_file": blend_file,
        "script_file": script_file or "",
        "setup_dir": paths["setup_dir"],
        "sites_file": paths["sites_file"],
        "config_file": paths["config_file"],
        "profile": profile["name"],
        "device_shape": device_shape or "",
        "spacing0": spec.get("spacing0"),
        "density_k": spec.get("density_k"),
    }

    if command_template:
        return shlex.split(command_template.format(**format_vars))

    if not blend_file:
        raise FileNotFoundError("No Blender file configured. Set EQUANTUM_BLENDER_FILE or profile['blend_file'].")
    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"Blender file not found: {blend_file}")
    if not script_file:
        raise FileNotFoundError(
            "No Blender update script configured. Set EQUANTUM_BLENDER_SCRIPT or EQUANTUM_BLENDER_COMMAND_TEMPLATE."
        )
    if not os.path.exists(script_file):
        raise FileNotFoundError(f"Blender update script not found: {script_file}")

    return [
        blender_bin,
        "--background",
        blend_file,
        "--python",
        script_file,
        "--",
        "--sites",
        paths["sites_file"],
        "--output",
        paths["config_file"],
        "--setup-dir",
        paths["setup_dir"],
        "--profile",
        profile["name"],
        "--device-shape",
        str(device_shape),
        "--spacing0",
        str(spec.get("spacing0")),
        "--density-k",
        str(spec.get("density_k")),
    ]


def resolve_blender_executable(blender_bin):
    if blender_bin and os.path.isabs(blender_bin) and os.path.exists(blender_bin):
        return blender_bin

    if blender_bin:
        resolved = shutil.which(blender_bin)
        if resolved:
            return resolved

    common_candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender.app/Contents/MacOS/blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/blender"),
    ]
    for candidate in common_candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Blender executable not found. "
        "Install Blender and make sure 'blender' is on PATH, or set "
        "EQUANTUM_BLENDER_BIN to the full executable path, for example "
        "'/Applications/Blender.app/Contents/MacOS/Blender'."
    )


def run_blender_site_update(profile, spec):
    paths = export_setup_sites(profile)
    if os.path.exists(paths["config_file"]):
        return paths

    command = build_blender_update_command(profile, spec)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n"
            "The Blender executable could not be launched. "
            "Set EQUANTUM_BLENDER_BIN to the full Blender executable path."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            "Blender site update failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not os.path.exists(paths["config_file"]):
        raise FileNotFoundError(
            f"Blender command completed, but expected config file was not created: {paths['config_file']}"
        )
    return paths


def ensure_profile_setup(profile, spec, require_config=True):
    paths = export_setup_sites(profile)
    if not require_config:
        return paths
    if os.path.exists(paths["config_file"]):
        return paths
    return run_blender_site_update(profile, spec)


def get_setup_generation_info(spec):
    profile = get_runtime_profile(spec)
    paths = setup_metadata_paths(profile)
    return {
        "profile": profile,
        "setup_dir": paths["setup_dir"],
        "sites_file": paths["sites_file"],
        "config_file": paths["config_file"],
        "requires_generation": not os.path.exists(paths["config_file"]),
    }


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


def empty_spec(default_profile="dotgate_center"):
    return {
        "raw_query": "",
        "task": None,
        "profile": default_profile,
        "lattice_type": None,
        "device_shape": None,
        "backgate_voltage": None,
        "magnetic_field_T": None,
        "solve_self_consistent": True,
        "spacing0": None,
        "density_k": None,
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
            r"\bb\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)\s*(?:t(?:esla)?)?\b",
            r"(-?\d+(?:\.\d+)?)\s*t(?:esla)?\b",
        ],
        "magnetic field",
    )

    return {
        "raw_query": query,
        "task": task,
        "profile": default_profile,
        "lattice_type": lattice_type,
        "device_shape": infer_device_shape_optional(query),
        "backgate_voltage": backgate_voltage,
        "magnetic_field_T": magnetic_field_t,
        "solve_self_consistent": True,
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


if BaseModel is not None:
    class UpdateSpecModel(BaseModel):
        task: Optional[str] = Field(default=None, description="Simulation target")
        profile: Optional[str] = Field(default=None, description="Simulation profile name")
        lattice_type: Optional[str] = Field(default=None, description="Lattice type")
        device_shape: Optional[str] = Field(default=None, description="Device geometry / Blender shape selection")
        backgate_voltage: Optional[float] = Field(default=None, description="Backgate voltage in volts")
        magnetic_field_T: Optional[float] = Field(default=None, description="Magnetic field in Tesla")
        solve_self_consistent: Optional[bool] = Field(default=None, description="Whether to run the FSC loop")
        spacing0: Optional[float] = Field(default=None, description="Base sampling spacing used by density_function")
        density_k: Optional[float] = Field(default=None, description="Slope parameter k used by density_function")
        dielectric_constant: Optional[float] = Field(default=None, description="Dielectric constant")
        gate_potential: Optional[float] = Field(default=None, description="Gate potential in volts")
        convergence_tol: Optional[list[float]] = Field(default=None, description="Two convergence tolerances [poisson, ildos]")
        Ncore: Optional[int] = Field(default=None, description="Number of CPU cores")
        eta: Optional[float] = Field(default=None, description="Broadening parameter")
        ldos_method: Optional[str] = Field(default=None, description="LDOS method")

    class UserTurnParseModel(BaseModel):
        raw_query: str = Field(description="Original user request")
        intent: str = Field(description="Intent of the latest user turn")
        updates: UpdateSpecModel = Field(description="Structured partial spec updates")

    class QuerySpecModel(BaseModel):
        raw_query: str = Field(description="Original user request")
        task: str = Field(description="Simulation target", pattern="^(dos|ldos)$")
        profile: str = Field(description="Simulation profile name")
        lattice_type: str = Field(description="Lattice type", pattern="^(square|honeycomb)$")
        device_shape: Optional[str] = Field(default=None, description="Device geometry / Blender shape selection")
        backgate_voltage: float = Field(description="Backgate voltage in volts")
        magnetic_field_T: float = Field(description="Magnetic field in Tesla")
        solve_self_consistent: bool = Field(description="Whether to run the FSC loop")
        spacing0: Optional[float] = Field(default=None, description="Base sampling spacing used by density_function")
        density_k: Optional[float] = Field(default=None, description="Slope parameter k used by density_function")
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
        device_shape: Optional[str] = Field(default=None, description="Device geometry / Blender shape selection")
        backgate_voltage: Optional[float] = Field(default=None, description="Backgate voltage in volts")
        magnetic_field_T: Optional[float] = Field(default=None, description="Magnetic field in Tesla")
        solve_self_consistent: Optional[bool] = Field(default=True, description="Whether to run the FSC loop")
        spacing0: Optional[float] = Field(default=None, description="Base sampling spacing used by density_function")
        density_k: Optional[float] = Field(default=None, description="Slope parameter k used by density_function")
        dielectric_constant: Optional[float] = Field(default=None, description="Dielectric constant")
        gate_potential: Optional[float] = Field(default=None, description="Gate potential in volts")
        convergence_tol: Optional[list[float]] = Field(default=None, description="Two convergence tolerances [poisson, ildos]")
        Ncore: Optional[int] = Field(default=None, description="Number of CPU cores")
        eta: Optional[float] = Field(default=None, description="Broadening parameter")
        ldos_method: Optional[str] = Field(default=None, description="LDOS method")

else:
    @dataclasses.dataclass
    class UpdateSpecModel:
        task: Optional[str] = None
        profile: Optional[str] = None
        lattice_type: Optional[str] = None
        device_shape: Optional[str] = None
        backgate_voltage: Optional[float] = None
        magnetic_field_T: Optional[float] = None
        solve_self_consistent: Optional[bool] = None
        spacing0: Optional[float] = None
        density_k: Optional[float] = None
        dielectric_constant: Optional[float] = None
        gate_potential: Optional[float] = None
        convergence_tol: Optional[list[float]] = None
        Ncore: Optional[int] = None
        eta: Optional[float] = None
        ldos_method: Optional[str] = None

    @dataclasses.dataclass
    class UserTurnParseModel:
        raw_query: str
        intent: str
        updates: UpdateSpecModel

    @dataclasses.dataclass
    class QuerySpecModel:
        raw_query: str
        task: str
        profile: str
        lattice_type: str
        device_shape: Optional[str] = None
        backgate_voltage: float
        magnetic_field_T: float
        solve_self_consistent: bool
        spacing0: Optional[float] = None
        density_k: Optional[float] = None
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
        device_shape: Optional[str] = None
        backgate_voltage: Optional[float] = None
        magnetic_field_T: Optional[float] = None
        solve_self_consistent: Optional[bool] = True
        spacing0: Optional[float] = None
        density_k: Optional[float] = None
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
            "device_shape": {"type": ["string", "null"], "enum": ["dotgate", "squaregate_center", None]},
            "backgate_voltage": {"type": "number"},
            "magnetic_field_T": {"type": "number"},
            "solve_self_consistent": {"type": "boolean"},
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
        "required": [
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
        "required": [
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
        ],
        "additionalProperties": False,
    }


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
        "'use defaults', 'ok defaults' -> confirm_defaults. "
        "'generate setup', 'create setup' -> confirm_setup_generation. "
        "'no', 'change defaults', 'not defaults' -> reject_defaults. "
        "Normalize obvious typos like 'sqaure' to 'square'. "
        f"If no profile is stated, do not invent one; use null in updates. The default profile is {default_profile}."
    )


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
    return normalize_turn_parse(parsed, default_profile=default_profile)


def langchain_parse_user_turn(query, default_profile, model, context):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain parser requires langchain-openai. Install it with: pip install -U langchain-openai"
        ) from exc

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        use_responses_api=True,
    )
    structured_llm = llm.with_structured_output(
        UserTurnParseModel,
        method="json_schema",
        strict=True,
    )
    context_text = json.dumps(context, indent=2, sort_keys=True)
    result = structured_llm.invoke(
        [
            ("system", build_user_turn_parser_prompt(default_profile)),
            ("human", f"Session context:\n{context_text}"),
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
    return normalize_turn_parse(parsed, default_profile=default_profile)


def log_parser_debug(args, turn_parse):
    if getattr(args, "debug_parser", False):
        debug_payload = {
            "parser": turn_parse.get("_parser"),
            "intent": turn_parse.get("intent"),
            "updates": turn_parse.get("updates"),
            "fallback_used": turn_parse.get("_fallback_used", False),
            "parser_error": turn_parse.get("_parser_error"),
        }
        print(json.dumps(debug_payload, indent=2), file=sys.stderr)


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
    parser_name = parser_name_from_args(args)
    default_profile = getattr(args, "profile", "dotgate_center")
    model = getattr(args, "openai_model", "gpt-4o-mini")
    allow_fallback = not getattr(args, "strict_openai", False)
    context = {
        "current_spec": normalize_spec(current_spec, default_profile=default_profile) if current_spec is not None else normalize_spec(None, default_profile=default_profile),
        "missing_fields": list(missing_fields or []),
        "pending_default_fields": list(pending_default_fields or []),
        "awaiting_defaults_confirmation": bool(awaiting_defaults_confirmation),
        "awaiting_setup_generation_confirmation": bool(awaiting_setup_generation_confirmation),
        "awaiting_run_confirmation": bool(awaiting_run_confirmation),
    }

    if parser_name == "regex":
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
        log_parser_debug(args, turn_parse)
        return turn_parse

    if parser_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            if allow_fallback:
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
                log_parser_debug(args, turn_parse)
                return turn_parse
            raise RuntimeError("OPENAI_API_KEY is not set and strict OpenAI parsing was requested.")
        try:
            turn_parse = llm_parse_user_turn(user_text, default_profile, api_key, model, base_url, context)
            turn_parse["_parser"] = f"openai:{model}"
            log_parser_debug(args, turn_parse)
            return turn_parse
        except Exception as exc:
            if allow_fallback:
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
                turn_parse["_parser_error"] = str(exc)
                log_parser_debug(args, turn_parse)
                return turn_parse
            raise

    if parser_name == "langchain":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if allow_fallback:
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
                log_parser_debug(args, turn_parse)
                return turn_parse
            raise RuntimeError("OPENAI_API_KEY is not set and strict LangChain parsing was requested.")
        try:
            turn_parse = langchain_parse_user_turn(user_text, default_profile, model, context)
            turn_parse["_parser"] = f"langchain_openai:{model}"
            log_parser_debug(args, turn_parse)
            return turn_parse
        except Exception as exc:
            if allow_fallback:
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
                turn_parse["_parser"] = f"regex_fallback_after_langchain_error:{type(exc).__name__}"
                turn_parse["_parser_error"] = str(exc)
                log_parser_debug(args, turn_parse)
                return turn_parse
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
    log_parser_debug(args, turn_parse)
    return turn_parse


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
        "device_shape": "Which device shape should I use, for example dotgate or squaregate_center?",
        "backgate_voltage": "What backgate voltage should I use, in volts?",
        "magnetic_field_T": "What magnetic field should I use, in Tesla?",
        "spacing0": "What spacing0 should I use for the sampling density function?",
        "density_k": "What k value should I use for the sampling density function?",
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


def parser_name_from_args(args):
    return getattr(args, "parser", "langchain")


def get_default_spec_values(args, profile_name):
    profile = get_profile_defaults(profile_name)
    return {
        "device_shape": getattr(args, "device_shape", profile["device_shape"]),
        "spacing0": float(profile["density_defaults"]["spacing0"]),
        "density_k": float(profile["density_defaults"]["density_k"]),
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


def combine_explicit_fields(reference_fields, parsed_fields, allow_optional_fields=None):
    allowed_optional = set(allow_optional_fields or [])
    combined = set(reference_fields or [])
    for field in parsed_fields or []:
        if field in OPTIONAL_CONFIRM_FIELDS and field not in combined and field not in allowed_optional:
            continue
        combined.add(field)
    return sorted(combined)


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

    setup_info = get_setup_generation_info(session_state["spec"])
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
            return agent_response(
                "needs_clarification",
                message,
                session_state["spec"],
                session_state,
            )
    else:
        session_state["setup_generation_confirmed"] = False
        session_state["setup_generation_signature"] = None

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
    turn_parse = parse_user_turn(user_text, args)
    partial_spec = {"raw_query": user_text, **turn_parse["updates"]}
    default_profile = turn_parse["updates"].get("profile") or getattr(args, "profile", "dotgate_center")
    partial_spec["profile"] = default_profile
    explicit_fields = explicit_fields_from_updates(turn_parse["updates"])
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, explicit_fields)
    defaults = get_default_spec_values(args, default_profile)
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
    setup_generation_confirmed = bool(existing_state.get("setup_generation_confirmed", False))
    setup_generation_signature = existing_state.get("setup_generation_signature")
    prior_missing = existing_state.get("missing_fields") or get_missing_fields(existing_spec)
    awaiting_defaults_confirmation = bool(pending_default_fields and not defaults_confirmed)
    awaiting_setup_generation_confirmation = bool(existing_state.get("pending_setup_generation") and not setup_generation_confirmed)
    awaiting_run_confirmation = bool(not pending_default_fields and defaults_confirmed and not run_confirmed and not awaiting_setup_generation_confirmation)

    turn_parse = parse_user_turn(
        user_text,
        args,
        current_spec=existing_spec,
        missing_fields=prior_missing,
        pending_default_fields=pending_default_fields,
        awaiting_defaults_confirmation=awaiting_defaults_confirmation,
        awaiting_setup_generation_confirmation=awaiting_setup_generation_confirmation,
        awaiting_run_confirmation=awaiting_run_confirmation,
    )
    partial_spec = {"raw_query": user_text, **turn_parse["updates"]}
    partial_spec["profile"] = turn_parse["updates"].get("profile") or existing_spec.get("profile", getattr(args, "profile", "dotgate_center"))
    new_explicit_fields = explicit_fields_from_updates(turn_parse["updates"])
    partial_spec = sanitize_optional_confirmation_fields(partial_spec, new_explicit_fields)
    merged_spec = merge_spec(existing_spec, partial_spec)
    explicit_fields = sorted(set(explicit_fields) | set(new_explicit_fields))
    parser_mode = parser_name_from_args(args)
    if parser_mode == "regex" or turn_parse.get("_fallback_used", False):
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

        if turn_parse.get("intent") == "reject_defaults":
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
            return agent_response(
                "needs_clarification",
                message,
                updated_state["spec"],
                updated_state,
            )
        if turn_parse.get("intent") == "confirm_defaults" or override_detected or meaningful_spec_change_detected:
            merged_spec = apply_defaults_to_spec(merged_spec, default_values, pending_default_fields)
            defaults_confirmed = True
            pending_default_fields = []
        elif parser_mode == "regex" and is_negative_reply(user_text):
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
            return agent_response(
                "needs_clarification",
                message,
                updated_state["spec"],
                updated_state,
            )

    if not pending_default_fields and defaults_confirmed:
        setup_reply_changed_spec = any(
            merged_spec.get(field_name) != existing_spec.get(field_name)
            for field_name in SPEC_FIELDS
            if field_name not in {"raw_query", "solve_self_consistent"}
        )
        existing_setup_signature = get_setup_generation_info(existing_spec)["setup_dir"]
        new_setup_signature = get_setup_generation_info(merged_spec)["setup_dir"]
        setup_signature_changed = existing_setup_signature != new_setup_signature
        if setup_reply_changed_spec:
            if setup_signature_changed:
                setup_generation_confirmed = False
                setup_generation_signature = None
                run_confirmed = False
        elif existing_state.get("pending_setup_generation") and turn_parse.get("intent") == "confirm_setup_generation":
            setup_generation_confirmed = True
            setup_generation_signature = existing_state.get("pending_setup_dir")
        elif existing_state.get("pending_setup_generation") and parser_mode == "regex" and is_setup_generation_confirmation_reply(user_text):
            setup_generation_confirmed = True
            setup_generation_signature = existing_state.get("pending_setup_dir")

    awaiting_setup_generation_confirmation = bool(existing_state.get("pending_setup_generation")) and not setup_generation_confirmed

    if not pending_default_fields and defaults_confirmed and not run_confirmed and not awaiting_setup_generation_confirmation:
        if turn_parse.get("intent") == "confirm_run":
            run_confirmed = True
        elif parser_mode == "regex":
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


def summarize_run(fsc, spec, phi, artifact_dir):
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
    profile = get_runtime_profile(spec)
    ensure_compatible(spec, profile)
    spec = apply_defaults_to_spec(spec, get_default_spec_values(args, spec["profile"]), OPTIONAL_CONFIRM_FIELDS)
    setup_paths = ensure_profile_setup(profile, spec, require_config=True)
    setup_dir = setup_paths["setup_dir"]
    config_file = setup_paths["config_file"]
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
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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
    parser.add_argument("--device-shape", default=get_profile_defaults("dotgate_center")["device_shape"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parser", default="langchain", choices=["langchain", "openai", "regex"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--strict-openai", action="store_true")
    parser.add_argument("--debug-parser", action="store_true")
    parser.add_argument("--no-scf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ldos-method", default="ED", choices=["TF", "ED", "kmeanssample"])
    parser.add_argument("--ncore", type=int, default=1)
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
