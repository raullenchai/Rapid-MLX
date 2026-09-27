"""In-process CUA run registry for the server API.

One run at a time (the computer is the shared resource). The GUI (or any
loopback client) creates a run, polls events, approves sign-in gates, and
cancels. The loop itself is unchanged; this adapter only adds a status
surface, numbered events, an approval event, and a stop event.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field

from rapid_mlx.cua.config import CUAConfig, PlannerConfig, resolve_planner
from rapid_mlx.cua.loop import run as run_loop

MAX_CONCURRENT_RUNS = 1


class CUARunConflictError(RuntimeError):
    pass


class CUARunNotFoundError(RuntimeError):
    pass


@dataclass
class CUAServiceRun:
    run_id: str
    app: str
    goal: str
    config: CUAConfig
    status: str = "running"
    final_summary: str = ""
    error: str = ""
    run_dir: str = ""
    created_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _approve_event: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _awaiting: bool = False

    def emit(self, event: dict) -> None:
        with self._lock:
            event = {"seq": len(self.events) + 1, "ts": time.time(), **event}
            self.events.append(event)
        kind = event.get("kind")
        if kind == "gate":
            self._awaiting = True
            self.status = "awaiting_approval"
        elif kind == "gate_resolved":
            self._awaiting = False
            if self.status == "awaiting_approval":
                self.status = "running"

    async def wait_for_approval(self, reason: str, timeout: float) -> bool:
        self.emit({"kind": "gate_detail", "reason": reason, "timeout_s": timeout})
        try:
            await asyncio.wait_for(self._approve_event.wait(), timeout=timeout)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False
        finally:
            self._awaiting = False
            self._approve_event.clear()

    def approve(self) -> bool:
        if not self._awaiting:
            return False
        self._approve_event.set()
        return True

    def cancel(self) -> None:
        self._stop_event.set()

    def view(self, events_after: int = 0) -> dict:
        with self._lock:
            events = [e for e in self.events if e["seq"] > events_after]
        return {
            "run_id": self.run_id,
            "app": self.app,
            "goal": self.goal,
            "status": self.status,
            "final_summary": self.final_summary,
            "error": self.error,
            "planner": self.config.planner.describe() if self.config else "n/a",
            "events_after_seq": events_after,
            "events": events,
            "run_dir": str(self.run_dir),
        }


class CUAService:
    """Owns the single active CUA run and its background task."""

    def __init__(self) -> None:
        self._runs: dict[str, CUAServiceRun] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": r.run_id,
                "app": r.app,
                "goal": r.goal,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in self._runs.values()
        ]

    def get(self, run_id: str) -> CUAServiceRun:
        run = self._runs.get(run_id)
        if run is None:
            raise CUARunNotFoundError(f"no such CUA run: {run_id}")
        return run

    async def create(
        self,
        app: str,
        goal: str,
        planner: str = "cloud-glm",
        planner_model: str | None = None,
        planner_url: str | None = None,
        open_url: str = "",
        allowed_domain: str = "",
        max_steps: int = 12,
        human_login: bool = False,
    ) -> CUAServiceRun:
        active = [
            r
            for r in self._runs.values()
            if r.status in {"running", "awaiting_approval"}
        ]
        if len(active) >= MAX_CONCURRENT_RUNS:
            raise CUARunConflictError(
                "another CUA run is active; cancel it before starting a new one"
            )
        try:
            planner_cfg: PlannerConfig = resolve_planner(
                planner, url_override=planner_url, model_override=planner_model
            )
        except ValueError as exc:
            raise ValueError(f"bad planner: {exc}") from exc
        config = CUAConfig(
            planner=planner_cfg,
            allowed_domain=allowed_domain,
            human_login=human_login,
        )
        run_id = uuid.uuid4().hex[:12]
        run = CUAServiceRun(run_id=run_id, app=app, goal=goal, config=config)
        run.run_dir = ""
        self._runs[run_id] = run
        run.emit({"kind": "started", "app": app, "goal": goal[:200]})

        task = asyncio.create_task(
            run_loop(
                config,
                app,
                goal,
                open_url=open_url,
                max_steps=max_steps,
                event_sink=run.emit,
                gate=lambda reason: run.wait_for_approval(reason, config.pause_timeout),
                stop_event=run._stop_event,
            )
        )
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._finalize(run, t))
        return run

    def _finalize(self, run: CUAServiceRun, task: asyncio.Task) -> None:
        trace: dict = {}
        try:
            trace = task.result() or {}
        except Exception as exc:  # noqa: BLE001 - surface failure to the client
            run.status = "failed"
            run.error = str(exc)[:400]
            run.emit({"kind": "terminal", "status": "failed", "error": run.error})
            return
        run.final_summary = str(trace.get("final_summary", ""))
        status = trace.get("status", "incomplete")
        run.status = {
            "done": "completed",
            "stopped": "stopped",
            "stalled": "stalled",
        }.get(status, status if status in {"failed"} else "incomplete")
        run.emit(
            {
                "kind": "terminal",
                "status": run.status,
                "final_summary": run.final_summary,
            }
        )


_SERVICE = CUAService()


def get_cua_service() -> CUAService:
    return _SERVICE


async def close_cua_service() -> None:
    """Cancel any active run (server shutdown)."""
    for run in _SERVICE._runs.values():
        if run.status in {"running", "awaiting_approval"}:
            run.cancel()
    for task in _SERVICE._tasks.values():
        task.cancel()
