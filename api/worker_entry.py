from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.schemas import AgentRuntimeConfig
from api.runtime import (
    append_event,
    append_log,
    load_json,
    manual_check_path,
    request_path,
    set_error,
    set_manual_check,
    set_metadata,
    set_pid,
    set_result,
    set_status,
)
from api.planner_runner import execute_planner_plan
from simulation.runner import RunAbortedError, run_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single EQuantum simulation worker process.")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def wait_for_manual_check(run_dir: Path, payload: dict) -> bool:
    set_manual_check(
        run_dir,
        pending=True,
        payload=payload,
        message="manual_check_required",
    )
    response_file = manual_check_path(run_dir)
    while True:
        if response_file.exists():
            response = load_json(response_file, default={})
            try:
                response_file.unlink()
            except FileNotFoundError:
                pass
            approved = bool(response.get("approved", False))
            set_manual_check(
                run_dir,
                pending=False,
                payload={"approved": approved},
                message="manual_check_resolved",
            )
            return approved
        time.sleep(0.5)


def should_abort_check(run_dir: Path) -> bool:
    try:
        state = load_json(run_dir / "state.json", default={})
        return state.get("status") == "aborted"
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    request = load_json(request_path(run_dir), default={})
    mode = str(request.get("mode", "simulation"))
    spec = dict(request.get("spec", {}))
    config = AgentRuntimeConfig(**dict(request.get("config", {})))
    require_manual_check = bool(request.get("require_manual_check", False))
    approved_plan = list(request.get("approved_plan", []))
    original_request = str(request.get("original_request", ""))

    set_pid(run_dir, os.getpid())

    try:
        if mode == "planner":
            result = execute_planner_plan(
                approved_plan,
                config=config,
                original_request=original_request,
                status=lambda status: set_status(run_dir, status),
                log=lambda line: append_log(run_dir, line),
                metadata=lambda payload: set_metadata(run_dir, **payload),
                event=lambda t, p: append_event(run_dir, event_type=t, payload=p),
                manual_check=lambda payload: wait_for_manual_check(run_dir, payload),
                should_abort=lambda: should_abort_check(run_dir),
            )
        else:
            result = run_spec(
                spec,
                config=config,
                status=lambda status: set_status(run_dir, status),
                log=lambda line: append_log(run_dir, line),
                manual_check=lambda payload: wait_for_manual_check(run_dir, payload),
                metadata=lambda payload: set_metadata(run_dir, **payload),
                event=lambda t, p: append_event(run_dir, event_type=t, payload=p),
                require_manual_check=require_manual_check,
                should_abort=lambda: should_abort_check(run_dir),
            )
        set_result(run_dir, result)
        return 0
    except RunAbortedError as exc:
        set_error(run_dir, "aborted", str(exc))
        return 2
    except Exception:
        set_error(run_dir, "failed", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
