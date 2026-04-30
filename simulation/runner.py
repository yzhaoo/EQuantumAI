from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUANTUM_DIR = os.path.join(PROJECT_ROOT, "EQuantum", "Equantum")
if EQUANTUM_DIR not in sys.path:
    sys.path.insert(0, EQUANTUM_DIR)

from EQuantum.Equantum.EQsystem import System
from EQuantum.Equantum.fsc import FSC
from EQuantum.agent.manual_boundary_check import generate_manual_boundary_check

from agent.schemas import AgentResponse, AgentRuntimeConfig
from simulation import artifacts, profiles, setup

StatusCallback = Callable[[str], None]
LogCallback = Callable[[str], None]
ManualCheckCallback = Callable[[dict[str, Any]], bool]


def _noop_status(_: str) -> None:
    return None


def _noop_log(_: str) -> None:
    return None


def _default_manual_check(_: dict[str, Any]) -> bool:
    return True


def run_spec(
    spec: dict[str, Any],
    config: AgentRuntimeConfig | None = None,
    *,
    status: StatusCallback | None = None,
    log: LogCallback | None = None,
    manual_check: ManualCheckCallback | None = None,
    require_manual_check: bool = False,
    snapshot_mode: str = "step",
    snapshot_every: int = 1,
) -> dict[str, Any]:
    runtime_config = config or AgentRuntimeConfig()
    status_cb = status or _noop_status
    log_cb = log or _noop_log
    manual_cb = manual_check or _default_manual_check
    args = runtime_config.to_namespace()

    spec, profile = profiles.resolve_runtime_spec(spec, args)
    setup_dir, config_file = setup.resolve_setup(profile, spec, require_config=True)
    artifact_dir = artifacts.make_artifact_dir(profile, runtime_config.output_dir)

    log_cb(f"Using setup directory: {setup_dir}")
    log_cb(f"Artifacts will be saved to: {artifact_dir}")
    artifacts.save_query_spec(artifact_dir, spec)

    status_cb("building_system")
    syst = System(
        profile["geoparams"],
        config_file=config_file,
        ifqsystem=True,
        quantum_builder="default",
    )
    log_cb("System built successfully.")
    log_cb(f"Number of sites: {syst.num_sites}")
    log_cb(f"Number of quantum sites: {len(syst.Qsites)}")

    status_cb("initializing_fsc")
    phi = profiles.magnetic_field_to_phi(spec["magnetic_field_T"], syst.unit_cell_area)
    log_cb(f"Magnetic field B = {spec['magnetic_field_T']} T")
    log_cb(f"Converted flux phi = {phi}")
    qparams = {"Ufunc": lambda site: 0, "phi": phi}
    fsc = FSC(syst, ifinitial=False, qparams=qparams, approx="TF")
    fsc.update_BC(profiles.build_boundary_conditions(profile, spec), ifinitial=True)
    for site_id, fsite in fsc.sites.items():
        if site_id in syst.sites:
            syst.sites[site_id].potential = fsite.potential
            syst.sites[site_id].charge = fsite.charge
    log_cb("Boundary conditions updated.")

    fsc.Ncore = int(spec["Ncore"])
    fsc.convergence_tol = list(spec["convergence_tol"])
    fsc.save_static_reference(artifact_dir)
    log_cb("Saved static FSC reference.")

    manual_check_payload = None
    if require_manual_check:
        status_cb("manual_check_required")
        manual_check_payload = generate_manual_boundary_check(fsc, artifact_dir)
        log_cb(manual_check_payload["message"])
        log_cb(f"Manual boundary plot saved to: {manual_check_payload['plot_path']}")
        if not manual_cb(manual_check_payload):
            raise RuntimeError("Manual boundary check was rejected.")

    status_cb("solving")
    log_cb(f"Starting FSC solve with snapshot_mode={snapshot_mode!r}.")
    fsc.solve(
        syst,
        save=True,
        snapshot_mode=snapshot_mode,
        snapshot_every=snapshot_every,
        snapshot_folder=artifact_dir,
        ldos_method=spec["ldos_method"],
        save_ildos=True,
        eta=spec["eta"],
        M=runtime_config.moments,
        eps=runtime_config.eps,
        kernel=runtime_config.kernel,
    )
    log_cb("FSC solve finished.")

    status_cb("exporting_artifacts")
    energy_grid = np.linspace(-6 * syst.t, 6 * syst.t, runtime_config.energy_points)
    result = artifacts.summarize_run(fsc, spec, phi, artifact_dir)
    result["setup_dir"] = setup_dir
    result["config_file"] = config_file

    if spec["task"] == "dos":
        energy, rho = fsc.qsystem.get_dos(
            w=energy_grid,
            M=runtime_config.moments,
            n_random=runtime_config.n_random,
            eps=runtime_config.eps,
            kernel=runtime_config.kernel,
        )
        result["artifacts"] = artifacts.save_dos_artifacts(artifact_dir, energy, rho)
    else:
        site_id, artifact_paths = artifacts.save_ldos_artifacts(artifact_dir, fsc, site_mode="center")
        result["ldos_site_id"] = site_id
        result["artifacts"] = artifact_paths

    if manual_check_payload is not None:
        result["manual_check"] = manual_check_payload

    artifacts.save_run_summary(artifact_dir, result)
    return result


def run_agent_response(
    response: AgentResponse,
    config: AgentRuntimeConfig | None = None,
    *,
    status: StatusCallback | None = None,
    log: LogCallback | None = None,
    manual_check: ManualCheckCallback | None = None,
    require_manual_check: bool = False,
) -> AgentResponse:
    if response.status != "running":
        return response

    result = run_spec(
        response.spec,
        config=config,
        status=status,
        log=log,
        manual_check=manual_check,
        require_manual_check=require_manual_check,
    )
    completed_session = dict(response.session_state)
    completed_session["active"] = False
    completed_session["missing_fields"] = []
    completed_session.setdefault("history", []).append({"role": "assistant", "text": "Simulation completed."})
    return AgentResponse(
        status="completed",
        message="Simulation completed.",
        spec=dict(response.spec),
        missing_fields=[],
        session_state=completed_session,
        result=result,
    )


def format_result_summary(result: dict[str, Any]) -> str:
    artifacts_info = result.get("artifacts", {})
    lines = [
        f"Task: {result.get('task')}",
        f"Profile: {result.get('profile')}",
        f"Lattice: {result.get('lattice_type')}",
        f"Artifact dir: {result.get('artifact_dir')}",
    ]
    if artifacts_info.get("plot"):
        lines.append(f"Plot: {artifacts_info['plot']}")
    if artifacts_info.get("data"):
        lines.append(f"Data: {artifacts_info['data']}")
    if result.get("manual_check", {}).get("plot_path"):
        lines.append(f"Boundary check: {result['manual_check']['plot_path']}")
    return os.linesep.join(lines)
