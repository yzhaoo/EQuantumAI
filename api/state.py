from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from agent.schemas import AgentRuntimeConfig
from api.runtime import (
    PROJECT_ROOT,
    default_state,
    ensure_run_dir,
    load_state,
    manual_check_path,
    request_path,
    save_state,
    set_error,
)


class RunRecord:
    def __init__(
        self,
        spec: dict[str, Any],
        config: AgentRuntimeConfig,
        require_manual_check: bool,
        *,
        mode: str = "simulation",
        approved_plan: list[dict[str, Any]] | None = None,
        original_request: str = "",
    ):
        self.id = str(uuid.uuid4())
        self.spec = dict(spec)
        self.config = config
        self.require_manual_check = require_manual_check
        self.mode = mode
        self.approved_plan = list(approved_plan or [])
        self.original_request = original_request
        self.run_dir = ensure_run_dir(self.id)
        self.process: subprocess.Popen[str] | None = None
        self.status = "queued"
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.logs: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.manual_check_pending = False
        self.manual_check_payload: dict[str, Any] | None = None
        self._lock = threading.Lock()

        save_state(self.run_dir, default_state(self.id, self.spec))
        self._write_request_file()

    def _write_request_file(self) -> None:
        payload = {
            "mode": self.mode,
            "spec": self.spec,
            "config": self.config.to_namespace().__dict__,
            "require_manual_check": self.require_manual_check,
            "approved_plan": self.approved_plan,
            "original_request": self.original_request,
        }
        from api.runtime import atomic_write_json

        atomic_write_json(request_path(self.run_dir), payload)

    def launch(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "api.worker_entry",
            "--run-dir",
            str(self.run_dir),
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
            text=True,
        )
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            state = load_state(self.run_dir)
            if not state:
                return
            self.status = state.get("status", self.status)
            self.result = state.get("result")
            self.error = state.get("error")
            self.logs = list(state.get("logs", []))
            self.events = list(state.get("events", []))
            self.manual_check_pending = bool(state.get("manual_check_pending", False))
            self.manual_check_payload = state.get("manual_check_payload")

    def resolve_manual_check(self, approved: bool) -> None:
        payload = {"approved": bool(approved)}
        from api.runtime import atomic_write_json

        atomic_write_json(manual_check_path(self.run_dir), payload)
        self.refresh()

    def request_abort(self) -> str:
        self.refresh()
        if self.status in {"completed", "failed", "aborted"}:
            return self.status

        if self.process is None:
            set_error(self.run_dir, "aborted", "Simulation aborted before worker launch.")
            self.refresh()
            return self.status

        if self.process.poll() is not None:
            set_error(self.run_dir, "aborted", "Simulation aborted after worker exit.")
            self.refresh()
            return self.status

        # Terminate the whole process group so future multi-core or child-worker
        # trees are torn down together rather than leaving orphan processes.
        self._terminate_process_group(force=False)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self.process.poll() is not None:
                break
            time.sleep(0.1)

        if self.process.poll() is None:
            self._terminate_process_group(force=True)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        set_error(self.run_dir, "aborted", "Simulation aborted by user request.")
        self.refresh()
        return self.status

    def _terminate_process_group(self, *, force: bool) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return
        if sys.platform != "win32":
            import os

            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            return
        try:
            if force:
                self.process.kill()
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create_run(self, spec: dict[str, Any], config: AgentRuntimeConfig, require_manual_check: bool) -> RunRecord:
        record = RunRecord(spec=spec, config=config, require_manual_check=require_manual_check)
        with self._lock:
            self._runs[record.id] = record
        record.launch()
        return record

    def create_planner_run(
        self,
        approved_plan: list[dict[str, Any]],
        config: AgentRuntimeConfig,
        *,
        original_request: str = "",
    ) -> RunRecord:
        record = RunRecord(
            spec={},
            config=config,
            require_manual_check=False,
            mode="planner",
            approved_plan=approved_plan,
            original_request=original_request,
        )
        with self._lock:
            self._runs[record.id] = record
        record.launch()
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            record = self._runs.get(run_id)
        if record is not None:
            record.refresh()
        return record


RUN_MANAGER = RunManager()
