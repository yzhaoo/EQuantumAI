from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from typing import Any

from agent.schemas import AgentRuntimeConfig
from simulation.runner import run_spec


def _now() -> str:
    return datetime.now().isoformat()


class RunRecord:
    def __init__(self, spec: dict[str, Any], config: AgentRuntimeConfig, require_manual_check: bool):
        self.id = str(uuid.uuid4())
        self.spec = dict(spec)
        self.config = config
        self.require_manual_check = require_manual_check
        self.status = "queued"
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.logs: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.manual_check_pending = False
        self.manual_check_payload: dict[str, Any] | None = None
        self._approval: bool | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None

    def add_event(self, event_type: str, message: str | None = None, payload: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "message": message,
            "payload": payload,
            "timestamp": _now(),
        }
        self.events.append(event)

    def set_status(self, status: str) -> None:
        with self._condition:
            self.status = status
            self.add_event("status", message=status)
            self._condition.notify_all()

    def add_log(self, line: str) -> None:
        with self._condition:
            self.logs.append(line)
            self.add_event("log", message=line)
            self._condition.notify_all()

    def request_manual_check(self, payload: dict[str, Any]) -> bool:
        with self._condition:
            self.manual_check_pending = True
            self.manual_check_payload = payload
            self._approval = None
            self.add_event("manual_check", message="manual_check_required", payload=payload)
            self._condition.notify_all()
            while self._approval is None:
                self._condition.wait()
            approved = bool(self._approval)
            self.manual_check_pending = False
            self._approval = None
            self._condition.notify_all()
            return approved

    def resolve_manual_check(self, approved: bool) -> None:
        with self._condition:
            self._approval = bool(approved)
            self.manual_check_pending = False
            self.add_event("manual_check", message="manual_check_resolved", payload={"approved": bool(approved)})
            self._condition.notify_all()

    def mark_completed(self, result: dict[str, Any]) -> None:
        with self._condition:
            self.status = "completed"
            self.result = result
            self.add_event("result", message="completed", payload=result)
            self._condition.notify_all()

    def mark_failed(self, error: str) -> None:
        with self._condition:
            self.status = "failed"
            self.error = error
            self.add_event("error", message=error)
            self._condition.notify_all()


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create_run(self, spec: dict[str, Any], config: AgentRuntimeConfig, require_manual_check: bool) -> RunRecord:
        record = RunRecord(spec=spec, config=config, require_manual_check=require_manual_check)
        with self._lock:
            self._runs[record.id] = record
        record.add_event("status", message="queued")
        record._thread = threading.Thread(target=self._run_worker, args=(record,), daemon=True)
        record._thread.start()
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def _run_worker(self, record: RunRecord) -> None:
        try:
            result = run_spec(
                record.spec,
                config=record.config,
                status=record.set_status,
                log=record.add_log,
                manual_check=record.request_manual_check,
                require_manual_check=record.require_manual_check,
            )
            record.mark_completed(result)
        except Exception:
            record.mark_failed(traceback.format_exc())


RUN_MANAGER = RunManager()
