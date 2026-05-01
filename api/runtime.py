from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path(tempfile.gettempdir()) / "equantum_api_runs"

REQUEST_FILENAME = "request.json"
STATE_FILENAME = "state.json"
MANUAL_CHECK_FILENAME = "manual_check_response.json"


def now_iso() -> str:
    return datetime.now().isoformat()


def ensure_run_dir(run_id: str) -> Path:
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def request_path(run_dir: Path) -> Path:
    return run_dir / REQUEST_FILENAME


def state_path(run_dir: Path) -> Path:
    return run_dir / STATE_FILENAME


def manual_check_path(run_dir: Path) -> Path:
    return run_dir / MANUAL_CHECK_FILENAME


def resolve_artifact_dir(run_dir: Path) -> Path | None:
    state = load_state(run_dir)
    candidate = state.get("artifact_dir")
    if not candidate:
        result = state.get("result") or {}
        candidate = result.get("artifact_dir")
    if not candidate:
        return None
    return Path(candidate)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    with open(path) as handle:
        return json.load(handle)


def default_state(run_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "run_id": run_id,
        "status": "queued",
        "spec": spec,
        "result": None,
        "error": None,
        "logs": [],
        "events": [
            {
                "type": "status",
                "message": "queued",
                "payload": None,
                "timestamp": timestamp,
            }
        ],
        "manual_check_pending": False,
        "manual_check_payload": None,
        "pid": None,
        "updated_at": timestamp,
    }


def load_state(run_dir: Path) -> dict[str, Any]:
    return load_json(state_path(run_dir), default={})


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    atomic_write_json(state_path(run_dir), state)


def append_event(
    run_dir: Path,
    *,
    event_type: str,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = load_state(run_dir)
    state.setdefault("events", []).append(
        {
            "type": event_type,
            "message": message,
            "payload": payload,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state


def append_log(run_dir: Path, line: str) -> dict[str, Any]:
    state = load_state(run_dir)
    state.setdefault("logs", []).append(line)
    state.setdefault("events", []).append(
        {
            "type": "log",
            "message": line,
            "payload": None,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state


def set_status(run_dir: Path, status: str) -> dict[str, Any]:
    state = load_state(run_dir)
    state["status"] = status
    state.setdefault("events", []).append(
        {
            "type": "status",
            "message": status,
            "payload": None,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state


def set_pid(run_dir: Path, pid: int) -> dict[str, Any]:
    state = load_state(run_dir)
    state["pid"] = pid
    save_state(run_dir, state)
    return state


def set_metadata(run_dir: Path, **metadata: Any) -> dict[str, Any]:
    state = load_state(run_dir)
    for key, value in metadata.items():
        state[key] = value
    save_state(run_dir, state)
    return state


def set_manual_check(
    run_dir: Path,
    *,
    pending: bool,
    payload: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    state = load_state(run_dir)
    state["manual_check_pending"] = pending
    state["manual_check_payload"] = payload
    state.setdefault("events", []).append(
        {
            "type": "manual_check",
            "message": message,
            "payload": payload,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state


def set_result(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    state = load_state(run_dir)
    state["status"] = "completed"
    state["result"] = result
    state["error"] = None
    state["manual_check_pending"] = False
    state.setdefault("events", []).append(
        {
            "type": "result",
            "message": "completed",
            "payload": result,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state


def set_error(run_dir: Path, status: str, error: str) -> dict[str, Any]:
    state = load_state(run_dir)
    state["status"] = status
    state["error"] = error
    state["manual_check_pending"] = False
    state.setdefault("events", []).append(
        {
            "type": "error",
            "message": error,
            "payload": None,
            "timestamp": now_iso(),
        }
    )
    save_state(run_dir, state)
    return state
