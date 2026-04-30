from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent import parser
from agent.schemas import AgentRuntimeConfig
from api.schemas import (
    AgentTurnRequest,
    AgentTurnResponse,
    CreateRunRequest,
    CreateRunResponse,
    ManualCheckRequest,
    RunStateResponse,
)
from api.state import RUN_MANAGER

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
    record = RUN_MANAGER.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
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
    record = RUN_MANAGER.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "events": record.events}


@router.post("/runs/{run_id}/manual-check")
def resolve_manual_check(run_id: str, payload: ManualCheckRequest):
    record = RUN_MANAGER.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not record.manual_check_pending:
        raise HTTPException(status_code=409, detail="No manual check is pending for this run")
    record.resolve_manual_check(payload.approved)
    return {"run_id": run_id, "approved": payload.approved}
