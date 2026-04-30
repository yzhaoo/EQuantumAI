from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent import parser
from agent.schemas import AgentResponse, AgentRuntimeConfig


def resume_after_manual_check_cancel(response: AgentResponse) -> AgentResponse:
    cancel_message = (
        "Manual boundary check canceled. Tell me what to change, "
        "or reply 'start' to reset and run again with the same parameters."
    )
    resumed_session = dict(response.session_state)
    resumed_session["active"] = True
    resumed_session["run_confirmed"] = False
    resumed_session["missing_fields"] = []
    resumed_session.setdefault("history", []).append({"role": "assistant", "text": cancel_message})
    return AgentResponse(
        status="needs_clarification",
        message=cancel_message,
        spec=dict(response.spec),
        missing_fields=[],
        session_state=resumed_session,
        result=None,
    )


@dataclass
class AgentSession:
    config: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    session_state: dict[str, Any] | None = None

    def args(self):
        return parser.build_runtime_args(self.config)

    def turn(self, user_text: str, execute: bool = False) -> AgentResponse:
        response = parser.parse_turn(
            user_text,
            args=self.args(),
            session_state=self.session_state,
            execute=execute,
        )
        self.session_state = dict(response.session_state)
        return response

    def reset(self) -> None:
        self.session_state = None


def start_session(config: AgentRuntimeConfig | None = None) -> AgentSession:
    return AgentSession(config=config or AgentRuntimeConfig())
