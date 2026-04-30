from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentRuntimeConfig:
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

    def to_namespace(self):
        from argparse import Namespace

        return Namespace(**asdict(self))


@dataclass
class AgentResponse:
    status: str
    message: str
    spec: dict[str, Any]
    missing_fields: list[str]
    session_state: dict[str, Any]
    result: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentResponse":
        return cls(
            status=str(payload["status"]),
            message=str(payload["message"]),
            spec=dict(payload.get("spec", {})),
            missing_fields=list(payload.get("missing_fields", [])),
            session_state=dict(payload.get("session_state", {})),
            result=payload.get("result"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "message": self.message,
            "spec": self.spec,
            "missing_fields": self.missing_fields,
            "session_state": self.session_state,
        }
        if self.result is not None:
            payload["result"] = self.result
        return payload


@dataclass
class RunArtifacts:
    plot: str | None = None
    data: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.plot is not None:
            payload["plot"] = self.plot
        if self.data is not None:
            payload["data"] = self.data
        payload.update(self.extras)
        return payload


@dataclass
class RunResult:
    summary: dict[str, Any]
    artifacts: RunArtifacts | None = None
    manual_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.summary)
        if self.artifacts is not None:
            payload["artifacts"] = self.artifacts.to_dict()
        if self.manual_check is not None:
            payload["manual_check"] = self.manual_check
        return payload
