from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentTurnRequest(BaseModel):
    message: str
    session_state: dict[str, Any] | None = None
    execute: bool = False
    parser: str = "openai"
    openai_model: str = "gpt-4o-mini"
    strict_openai: bool = False
    profile: str = "dotgate_center"
    device_shape: str = "dotgate"
    output_dir: str | None = None
    ldos_method: str = "ED"
    ncore: int = 1
    moments: int = 256
    n_random: int = 10
    eta: float = 0.00015
    eps: float = 0.05
    kernel: str = "jackson"
    energy_points: int = 1024
    tol_poisson: float = 1e-3
    tol_ildos: float = 1e-2


class AgentTurnResponse(BaseModel):
    status: str
    message: str
    spec: dict[str, Any]
    missing_fields: list[str]
    session_state: dict[str, Any]
    result: dict[str, Any] | None = None


class CreateRunRequest(BaseModel):
    spec: dict[str, Any]
    parser: str = "openai"
    openai_model: str = "gpt-4o-mini"
    strict_openai: bool = False
    profile: str = "dotgate_center"
    device_shape: str = "dotgate"
    output_dir: str | None = None
    no_scf: bool = False
    ldos_method: str = "ED"
    ncore: int = 1
    moments: int = 256
    n_random: int = 10
    eta: float = 0.00015
    eps: float = 0.05
    kernel: str = "jackson"
    energy_points: int = 1024
    tol_poisson: float = 1e-3
    tol_ildos: float = 1e-2
    require_manual_check: bool = False


class CreateRunResponse(BaseModel):
    run_id: str
    status: str


class ManualCheckRequest(BaseModel):
    approved: bool


class EventItem(BaseModel):
    type: Literal["status", "log", "manual_check", "result", "error"]
    message: str | None = None
    payload: dict[str, Any] | None = None
    timestamp: str


class RunStateResponse(BaseModel):
    run_id: str
    status: str
    spec: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    events: list[EventItem] = Field(default_factory=list)
    manual_check_pending: bool = False
    manual_check_payload: dict[str, Any] | None = None
