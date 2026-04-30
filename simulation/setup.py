from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
from typing import Any

from simulation import profiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUANTUM_DIR = os.path.join(PROJECT_ROOT, "EQuantum", "Equantum")
if EQUANTUM_DIR not in sys.path:
    sys.path.insert(0, EQUANTUM_DIR)


def resolve_setup_dir(profile: dict[str, Any]) -> str:
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


def setup_metadata_paths(profile: dict[str, Any]) -> dict[str, str]:
    setup_dir = profile["expected_setup_dir"]
    return {
        "setup_dir": setup_dir,
        "sites_file": os.path.join(setup_dir, profile.get("sites_filename", "sites.json")),
        "config_file": os.path.join(setup_dir, profile["config_filename"]),
        "hash_file": os.path.join(setup_dir, profile.get("hash_filename", "geoparams_hash.txt")),
        "readme_file": os.path.join(setup_dir, profile.get("readme_filename", "README.txt")),
    }


def export_setup_sites(profile: dict[str, Any]) -> dict[str, str]:
    from EQuantum.Equantum.EQsystem import System

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


def resolve_blender_executable(blender_bin: str) -> str:
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


def build_blender_update_command(profile: dict[str, Any], spec: dict[str, Any]) -> list[str]:
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


def run_blender_site_update(profile: dict[str, Any], spec: dict[str, Any]) -> dict[str, str]:
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


def ensure_profile_setup(profile: dict[str, Any], spec: dict[str, Any], require_config: bool = True) -> dict[str, str]:
    paths = export_setup_sites(profile)
    if not require_config:
        return paths
    if os.path.exists(paths["config_file"]):
        return paths
    return run_blender_site_update(profile, spec)


def get_setup_generation_info(spec: dict[str, Any]) -> dict[str, Any]:
    profile = profiles.get_runtime_profile(spec)
    paths = setup_metadata_paths(profile)
    return {
        "profile": profile,
        "setup_dir": paths["setup_dir"],
        "sites_file": paths["sites_file"],
        "config_file": paths["config_file"],
        "requires_generation": not os.path.exists(paths["config_file"]),
    }


def resolve_setup(profile: dict[str, Any], spec: dict[str, Any], require_config: bool = True) -> tuple[str, str]:
    setup_paths = ensure_profile_setup(profile, spec, require_config=require_config)
    return setup_paths["setup_dir"], setup_paths["config_file"]
