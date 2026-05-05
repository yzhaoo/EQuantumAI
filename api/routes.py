from __future__ import annotations

import json
import traceback
import urllib.error
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from agent import parser
from agent.schemas import AgentRuntimeConfig
from api.planner import run_planner_turn
from api.schemas import (
    AbortRunResponse,
    AgentTurnRequest,
    AgentTurnResponse,
    CreateRunRequest,
    CreateRunResponse,
    HistoryRunItem,
    ManualCheckRequest,
    PlannerExecuteRequest,
    PlannerTurnRequest,
    RunStateResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinitionsResponse,
    ViewerLdosCutRequest,
    ViewerLdosCutResponse,
    ViewerQuantumHeatmapResponse,
    ViewerSetupGeometryResponse,
    ViewerSetupFieldResponse,
    ViewerSiteLdosResponse,
    ViewerSnapshotsResponse,
    ViewerSurfaceCutRequest,
    ViewerSurfaceCutResponse,
)
from api.runtime import PROJECT_ROOT, list_history_run_dirs, load_json, resolve_artifact_dir, resolve_history_run_dir
from api.state import RUN_MANAGER
from api.tools import execute_tool_call, list_tool_definitions
from simulation import viewer_data

router = APIRouter()


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""
    if not body:
        return str(exc)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body
    error = parsed.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return body


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


def _history_item(run_dir: Path) -> dict:
    spec = load_json(run_dir / "query_spec.json", default={}) or {}
    result = load_json(run_dir / "run_summary.json", default={}) or {}
    snapshots = viewer_data.list_snapshots(run_dir)
    has_static = (run_dir / "run_static.npz").exists()
    return {
        "run_path": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "run_name": run_dir.name,
        "artifact_dir": str(run_dir),
        "profile": spec.get("profile") or result.get("profile"),
        "task": spec.get("task") or result.get("task"),
        "status": "completed" if result else ("saved" if has_static or snapshots else "incomplete"),
        "has_static": has_static,
        "snapshot_count": len(snapshots),
        "created_at": run_dir.name,
        "spec": spec or None,
        "result": result or None,
    }


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


@router.post("/agent/plan", response_model=AgentTurnResponse)
def agent_plan(payload: PlannerTurnRequest):
    config = _config_from_payload(payload)
    try:
        return run_planner_turn(
            payload.message,
            config,
            session_state=payload.session_state,
            max_iterations=payload.max_iterations,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise HTTPException(status_code=502, detail=_http_error_detail(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/planner/runs", response_model=CreateRunResponse)
def create_planner_run(payload: PlannerExecuteRequest):
    config = _config_from_payload(payload)
    record = RUN_MANAGER.create_planner_run(
        approved_plan=payload.approved_plan,
        config=config,
        original_request=payload.original_request,
    )
    return {"run_id": record.id, "status": record.status}


@router.get("/tools", response_model=ToolDefinitionsResponse)
def get_tools():
    return list_tool_definitions()


@router.post("/tools/call", response_model=ToolCallResponse)
def call_tool(payload: ToolCallRequest):
    config = _config_from_payload(payload)
    try:
        result = execute_tool_call(payload.tool_name, payload.arguments, config=config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "tool_name": payload.tool_name,
        "ok": True,
        "result": result,
        "error": None,
    }


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


@router.get("/runs/{run_id}/viewer/setup-geometry", response_model=ViewerSetupGeometryResponse)
def get_setup_geometry(run_id: str):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(viewer_data.setup_geometry_data, artifact_dir)


@router.get("/runs/{run_id}/viewer/setup-field", response_model=ViewerSetupFieldResponse)
def get_setup_field(
    run_id: str,
    snapshot: str = Query(...),
    property: str = Query(...),
):
    artifact_dir = _require_artifact_dir(run_id)
    return _viewer_call(viewer_data.setup_field_data, artifact_dir, snapshot, property)


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


@router.get("/history/runs", response_model=list[HistoryRunItem])
def list_history_runs():
    return [_history_item(run_dir) for run_dir in list_history_run_dirs()]


@router.get("/history/runs/{run_path:path}/viewer/snapshots", response_model=ViewerSnapshotsResponse)
def get_history_viewer_snapshots(run_path: str):
    artifact_dir = resolve_history_run_dir(run_path)
    return {
        "snapshots": _viewer_call(viewer_data.list_snapshots, artifact_dir),
        "has_static": (artifact_dir / "run_static.npz").exists(),
    }


@router.get("/history/runs/{run_path:path}/viewer/setup-geometry", response_model=ViewerSetupGeometryResponse)
def get_history_setup_geometry(run_path: str):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(viewer_data.setup_geometry_data, artifact_dir)


@router.get("/history/runs/{run_path:path}/viewer/setup-field", response_model=ViewerSetupFieldResponse)
def get_history_setup_field(
    run_path: str,
    snapshot: str = Query(...),
    property: str = Query(...),
):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(viewer_data.setup_field_data, artifact_dir, snapshot, property)


@router.get("/history/runs/{run_path:path}/viewer/quantum-heatmap", response_model=ViewerQuantumHeatmapResponse)
def get_history_quantum_heatmap(
    run_path: str,
    snapshot: str = Query(...),
    property: str = Query(...),
):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(viewer_data.quantum_heatmap_data, artifact_dir, snapshot, property)


@router.get("/history/runs/{run_path:path}/viewer/site-ldos", response_model=ViewerSiteLdosResponse)
def get_history_site_ldos(
    run_path: str,
    snapshot: str = Query(...),
    site_id: int = Query(...),
):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(viewer_data.local_ldos_data, artifact_dir, snapshot, site_id)


@router.post("/history/runs/{run_path:path}/viewer/surface-cut", response_model=ViewerSurfaceCutResponse)
def post_history_surface_cut(run_path: str, payload: ViewerSurfaceCutRequest):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(
        viewer_data.surface_cut_data,
        artifact_dir,
        payload.snapshot,
        payload.property,
        payload.p0,
        payload.p1,
        payload.cut_width,
    )


@router.post("/history/runs/{run_path:path}/viewer/ldos-cut", response_model=ViewerLdosCutResponse)
def post_history_ldos_cut(run_path: str, payload: ViewerLdosCutRequest):
    artifact_dir = resolve_history_run_dir(run_path)
    return _viewer_call(
        viewer_data.ldos_cut_data,
        artifact_dir,
        payload.snapshot,
        payload.p0,
        payload.p1,
        payload.cut_width,
        payload.overlay_snapshots,
    )


@router.get("/history/runs/{run_path:path}", response_model=HistoryRunItem)
def get_history_run(run_path: str):
    run_dir = resolve_history_run_dir(run_path)
    return _history_item(run_dir)
