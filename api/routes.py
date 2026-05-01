from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from agent import parser
from agent.schemas import AgentRuntimeConfig
from api.schemas import (
    AbortRunResponse,
    AgentTurnRequest,
    AgentTurnResponse,
    CreateRunRequest,
    CreateRunResponse,
    ManualCheckRequest,
    RunStateResponse,
    ViewerLdosCutRequest,
    ViewerLdosCutResponse,
    ViewerQuantumHeatmapResponse,
    ViewerSiteLdosResponse,
    ViewerSnapshotsResponse,
    ViewerSurfaceCutRequest,
    ViewerSurfaceCutResponse,
)
from api.runtime import resolve_artifact_dir
from api.state import RUN_MANAGER
from simulation import viewer_data

router = APIRouter()


def _config_from_payload(payload) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        parser=payload.parser,
        openai_model=payload.openai_model,
        strict_openai=payload.strict_openai,
        profile=payload.profile,
        device_shape=payload.device_shape,
        output_dir=payload.output_dir,
        no_scf=getattr(payload, "no_scf", False),
        ldos_method=payload.ldos_method,
        ncore=payload.ncore,
        moments=payload.moments,
        n_random=payload.n_random,
        eta=payload.eta,
        eps=payload.eps,
        kernel=payload.kernel,
        energy_points=payload.energy_points,
        tol_poisson=payload.tol_poisson,
        tol_ildos=payload.tol_ildos,
    )


def _require_record(run_id: str):
    record = RUN_MANAGER.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


def _require_artifact_dir(run_id: str) -> Path:
    record = _require_record(run_id)
    artifact_dir = resolve_artifact_dir(record.run_dir)
    if artifact_dir is None:
        raise HTTPException(status_code=404, detail="Run artifact directory is not available yet")
    if not artifact_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run artifact directory does not exist: {artifact_dir}")
    return artifact_dir


def _viewer_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/agent/turn", response_model=AgentTurnResponse)
def agent_turn(payload: AgentTurnRequest):
    config = _config_from_payload(payload)
    response = parser.parse_turn(
        payload.message,
        args=config.to_namespace(),
        session_state=payload.session_state,
        execute=payload.execute,
    )
    return response.to_dict()


@router.post("/runs", response_model=CreateRunResponse)
def create_run(payload: CreateRunRequest):
    config = _config_from_payload(payload)
    record = RUN_MANAGER.create_run(
        spec=payload.spec,
        config=config,
        require_manual_check=payload.require_manual_check,
    )
    return {"run_id": record.id, "status": record.status}


@router.get("/runs/{run_id}", response_model=RunStateResponse)
def get_run(run_id: str):
    record = _require_record(run_id)
    return {
        "run_id": record.id,
        "status": record.status,
        "spec": record.spec,
        "result": record.result,
        "error": record.error,
        "logs": record.logs,
        "events": record.events,
        "manual_check_pending": record.manual_check_pending,
        "manual_check_payload": record.manual_check_payload,
    }


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    record = _require_record(run_id)
    return {"run_id": run_id, "events": record.events}


@router.post("/runs/{run_id}/manual-check")
def resolve_manual_check(run_id: str, payload: ManualCheckRequest):
    record = _require_record(run_id)
    if not record.manual_check_pending:
        raise HTTPException(status_code=409, detail="No manual check is pending for this run")
    record.resolve_manual_check(payload.approved)
    return {"run_id": run_id, "approved": payload.approved}


@router.post("/runs/{run_id}/abort", response_model=AbortRunResponse)
def abort_run(run_id: str):
    record = _require_record(run_id)
    status = record.request_abort()
    return {"run_id": run_id, "status": status}


@router.get("/runs/{run_id}/viewer/snapshots", response_model=ViewerSnapshotsResponse)
def get_viewer_snapshots(run_id: str):
    artifact_dir = _require_artifact_dir(run_id)
    return {
        "snapshots": _viewer_call(viewer_data.list_snapshots, artifact_dir),
        "has_static": (artifact_dir / "run_static.npz").exists(),
    }


@router.get("/runs/{run_id}/viewer/quantum-heatmap", response_model=ViewerQuantumHeatmapResponse)
def get_quantum_heatmap(
    run_id: str,
    snapshot: str = Query(...),
    property: str = Query(...),
):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(viewer_data.quantum_heatmap_data, artifact_dir, snapshot, property)


@router.get("/runs/{run_id}/viewer/site-ldos", response_model=ViewerSiteLdosResponse)
def get_site_ldos(
    run_id: str,
    snapshot: str = Query(...),
    site_id: int = Query(...),
):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(viewer_data.local_ldos_data, artifact_dir, snapshot, site_id)


@router.post("/runs/{run_id}/viewer/surface-cut", response_model=ViewerSurfaceCutResponse)
def post_surface_cut(run_id: str, payload: ViewerSurfaceCutRequest):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(
        viewer_data.surface_cut_data,
        artifact_dir,
        payload.snapshot,
        payload.property,
        payload.p0,
        payload.p1,
        payload.cut_width,
    )


@router.post("/runs/{run_id}/viewer/ldos-cut", response_model=ViewerLdosCutResponse)
def post_ldos_cut(run_id: str, payload: ViewerLdosCutRequest):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(
        viewer_data.ldos_cut_data,
        artifact_dir,
        payload.snapshot,
        payload.p0,
        payload.p1,
        payload.cut_width,
        payload.overlay_snapshots,
    )
