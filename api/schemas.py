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


class AbortRunResponse(BaseModel):
    run_id: str
    status: str


class ViewerSnapshotsResponse(BaseModel):
    snapshots: list[str] = Field(default_factory=list)
    has_static: bool


class ViewerSetupGeometryResponse(BaseModel):
    site_ids: list[int]
    coordinates: list[list[float | None]]
    materials: list[str]
    qsite_ids: list[int] = Field(default_factory=list)
    bounds_min: list[float | None]
    bounds_max: list[float | None]
    geometry_params: dict[str, Any] = Field(default_factory=dict)
    gate_info: dict[str, Any] = Field(default_factory=dict)


class ViewerSetupFieldResponse(BaseModel):
    snapshot: str
    property: str
    site_ids: list[int]
    values: list[float | None]
    color_min: float
    color_max: float


class ViewerQuantumHeatmapResponse(BaseModel):
    snapshot: str
    property: str
    site_ids: list[int]
    x: list[float | None]
    y: list[float | None]
    values: list[float | None]
    qprime_mask: list[bool]
    color_min: float
    color_max: float


class ViewerSiteLdosResponse(BaseModel):
    site_id: int
    snapshot: str
    energy: list[float | None]
    ldos: list[float | None]
    Ui: float | None = None
    ni: float | None = None
    Ci: float | None = None
    ldos_at_0: float | None = None
    ldos_at_Ui: float | None = None
    consistency_delta_u: list[float | None] = Field(default_factory=list)
    consistency_poisson: list[float | None] = Field(default_factory=list)
    consistency_integrated: list[float | None] = Field(default_factory=list)
    dU_solution: float | None = None


class ViewerSurfaceCutRequest(BaseModel):
    snapshot: str
    property: str
    p0: list[float]
    p1: list[float]
    cut_width: float = 0.05


class ViewerSurfaceCutResponse(BaseModel):
    snapshot: str
    property: str
    p0: list[float | None]
    p1: list[float | None]
    cut_width: float
    distance_along: list[float | None]
    z: list[float | None]
    values: list[float | None]


class ViewerOverlayCurve(BaseModel):
    snapshot: str
    distance_along: list[float | None]
    Ui: list[float | None]
    is_current: bool


class ViewerLdosCutRequest(BaseModel):
    snapshot: str
    p0: list[float]
    p1: list[float]
    cut_width: float = 0.05
    overlay_snapshots: list[str] = Field(default_factory=list)


class ViewerLdosCutResponse(BaseModel):
    snapshot: str
    p0: list[float | None]
    p1: list[float | None]
    cut_width: float
    distance_along: list[float | None]
    energy: list[float | None]
    ldos_matrix: list[list[float | None]]
    overlays: list[ViewerOverlayCurve] = Field(default_factory=list)


class EventItem(BaseModel):
    type: Literal["status", "log", "manual_check", "result", "error"]
    message: str | None = None
    payload: dict[str, Any] | None = None
    timestamp: str


class HistoryRunItem(BaseModel):
    run_path: str
    run_name: str
    profile: str | None = None
    task: str | None = None
    status: str
    has_static: bool
    snapshot_count: int = 0
    created_at: str | None = None
    spec: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


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
