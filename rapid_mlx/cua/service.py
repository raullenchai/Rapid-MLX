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
from urllib.parse import urlparse

from rapid_mlx.cua.config import CUAConfig, PlannerConfig, load_config, resolve_planner
from rapid_mlx.cua.loop import run as run_loop
from rapid_mlx.cua.planner import assert_loopback_url

MAX_CONCURRENT_RUNS = 1
MAX_RETAINED_RUNS = 100


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
            # Cursor metadata is service-owned.  A sink payload must not be able
            # to forge sequence numbers or timestamps and break pagination.
            event = {**event, "seq": len(self.events) + 1, "ts": time.time()}
            self.events.append(event)
            kind = event.get("kind")
            if kind == "started" and event.get("run_dir"):
                self.run_dir = str(event["run_dir"])
            elif kind == "gate":
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
            with self._lock:
                self._awaiting = False
            self._approve_event.clear()

    def approve(self) -> bool:
        with self._lock:
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
        self._closing = False

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
        planner: str = "local-27b",
        planner_model: str | None = None,
        planner_url: str | None = None,
        open_url: str = "",
        allowed_domain: str = "",
        max_steps: int = 12,
        human_login: bool = False,
    ) -> CUAServiceRun:
        if self._closing:
            raise CUARunConflictError("CUA service is shutting down")
        app = app.strip()
        goal = goal.strip()
        if not app or not goal:
            raise ValueError("app and goal must contain non-whitespace text")
        if open_url:
            parsed_open_url = urlparse(open_url)
            if (
                parsed_open_url.scheme not in {"http", "https"}
                or not parsed_open_url.hostname
            ):
                raise ValueError("open_url must be an absolute HTTP(S) URL")
        active_tasks = [task for task in self._tasks.values() if not task.done()]
        active_runs = [
            run
            for run in self._runs.values()
            if run.status in {"running", "awaiting_approval"}
        ]
        if active_tasks or len(active_runs) >= MAX_CONCURRENT_RUNS:
            raise CUARunConflictError(
                "another CUA run is active; cancel it before starting a new one"
            )
        try:
            planner_cfg: PlannerConfig = resolve_planner(
                planner, url_override=planner_url, model_override=planner_model
            )
        except ValueError as exc:
            raise ValueError(f"bad planner: {exc}") from exc
        # Fail synchronously with HTTP 400 instead of accepting a run that is
        # guaranteed to die in its background task.
        assert_loopback_url(planner_cfg.url)
        stored_config = load_config()
        fast_ranker_url = str(stored_config.get("fast_ranker_url", ""))
        if fast_ranker_url:
            assert_loopback_url(fast_ranker_url)
        config = CUAConfig(
            planner=planner_cfg,
            fast_ranker_url=fast_ranker_url,
            allowed_domain=allowed_domain,
            human_login=human_login,
        )
        run_id = uuid.uuid4().hex[:12]
        run = CUAServiceRun(run_id=run_id, app=app, goal=goal, config=config)
        run.run_dir = ""
        self._prune_runs()
        self._runs[run_id] = run

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

        def finalize(completed: asyncio.Task) -> None:
            self._finalize(run, completed)

        task.add_done_callback(finalize)
        return run

    def _prune_runs(self) -> None:
        terminal = sorted(
            (
                run
                for run in self._runs.values()
                if run.run_id not in self._tasks or self._tasks[run.run_id].done()
            ),
            key=lambda run: run.created_at,
        )
        while len(self._runs) >= MAX_RETAINED_RUNS and terminal:
            expired = terminal.pop(0)
            self._runs.pop(expired.run_id, None)
            self._tasks.pop(expired.run_id, None)

    @staticmethod
    def _has_terminal_event(run: CUAServiceRun) -> bool:
        with run._lock:
            return any(event.get("kind") == "terminal" for event in run.events)

    def cancel(self, run_id: str) -> CUAServiceRun:
        run = self.get(run_id)
        run.cancel()
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        return run

    def _finalize(self, run: CUAServiceRun, task: asyncio.Task) -> None:
        self._tasks.pop(run.run_id, None)
        trace: dict = {}
        try:
            trace = task.result() or {}
        except asyncio.CancelledError:
            run.status = "stopped"
            run.final_summary = "cancelled by client"
            if not self._has_terminal_event(run):
                run.emit(
                    {
                        "kind": "terminal",
                        "status": "stopped",
                        "final_summary": run.final_summary,
                    }
                )
            return
        except Exception as exc:  # noqa: BLE001 - surface failure to the client
            run.status = "failed"
            run.error = str(exc)[:400]
            if not self._has_terminal_event(run):
                run.emit({"kind": "terminal", "status": "failed", "error": run.error})
            return
        run.final_summary = str(trace.get("final_summary", ""))
        status = trace.get("status", "incomplete")
        run.status = {
            "done": "completed",
            "stopped": "stopped",
            "stalled": "stalled",
            "incomplete": "stalled",
        }.get(status, status if status in {"failed"} else "failed")
        if not self._has_terminal_event(run):
            run.emit(
                {
                    "kind": "terminal",
                    "status": run.status,
                    "final_summary": run.final_summary,
                }
            )

    async def close(self) -> None:
        """Stop active work and wait for every background task to settle."""
        self._closing = True
        tasks = list(self._tasks.values())
        for run in self._runs.values():
            run.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


_SERVICE = CUAService()


def get_cua_service() -> CUAService:
    return _SERVICE


async def close_cua_service() -> None:
    """Cancel any active run (server shutdown)."""
    global _SERVICE
    service = _SERVICE
    await service.close()
    if _SERVICE is service:
        _SERVICE = CUAService()
