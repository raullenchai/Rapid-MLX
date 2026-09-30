"""In-process CUA run registry for the server API.

One run at a time (the computer is the shared resource). The GUI (or any
loopback client) creates a run, polls events, approves sign-in gates, and
cancels. The loop itself is unchanged; this adapter only adds a status
surface, numbered events, an approval event, and a stop event.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import NamedTuple
from urllib.parse import urlparse

from rapid_mlx.computer_use import backend
from rapid_mlx.cua.config import CUAConfig, PlannerConfig, load_config, resolve_planner
from rapid_mlx.cua.loop import run as run_loop
from rapid_mlx.cua.planner import assert_loopback_url, validate_planner_url

MAX_CONCURRENT_RUNS = 1
MAX_RETAINED_RUNS = 100


def _normalize_allowed_domain(value: str) -> str:
    domain = value.strip().lower().rstrip(".")
    if not domain:
        return ""
    if len(domain) > 200 or not re.fullmatch(
        r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
        domain,
    ):
        raise ValueError(f"invalid allowed_domain: {value!r}")
    return domain


class CUARunConflictError(RuntimeError):
    pass


class CUARunNotFoundError(RuntimeError):
    pass


class CUARequestIdentityConflictError(RuntimeError):
    pass


class CUARequestIdentityNotFoundError(RuntimeError):
    pass


class CUAGateMismatchError(RuntimeError):
    pass


class CUAGateDecisionConflictError(RuntimeError):
    pass


class _RunCreateIdentity(NamedTuple):
    app: str
    goal: str
    planner: str
    planner_model: str | None
    planner_url: str | None
    open_url: str
    allowed_domain: str
    max_steps: int
    human_login: bool
    window_id: str | None
    bundle_id: str | None
    process_start_time: float | None
    targets: tuple[tuple[str, str, int, str, str, str | None, float | None], ...]
    initial_target_id: str | None


@dataclass
class CUAServiceRun:
    run_id: str
    app: str
    goal: str
    config: CUAConfig
    window_id: str | None = None
    client_request_id: str | None = None
    targets: list[dict] = field(default_factory=list)
    active_target_id: str | None = None
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
    _pending_gate: dict | None = None
    _resolved_gate_id: str | None = None

    def emit(self, event: dict) -> None:
        with self._lock:
            # Cursor metadata is service-owned.  A sink payload must not be able
            # to forge sequence numbers or timestamps and break pagination.
            if event.get("kind") == "started" and event.get("run_dir"):
                self.run_dir = str(event["run_dir"])
            # Trace paths are host-private implementation details.  Custom GUI
            # clients receive the event, but never the local filesystem path.
            event = {key: value for key, value in event.items() if key != "run_dir"}
            if self.targets and self.active_target_id and not event.get("target_id"):
                event["target_id"] = self.active_target_id
            kind = event.get("kind")
            if event.get("target_id"):
                self.active_target_id = str(event["target_id"])
                active = next(
                    (
                        t
                        for t in self.targets
                        if t["target_id"] == self.active_target_id
                    ),
                    None,
                )
                if active is not None:
                    self.app = str(active["app"])
                    self.window_id = str(active["window_id"])
            if kind == "gate":
                # Allocate the event and identity before publishing the gate.
                # A GUI may decide immediately after observing this event, so
                # wait_for_approval must reuse both objects.
                self._approve_event = asyncio.Event()
                self._pending_gate = {
                    "gate_id": uuid.uuid4().hex,
                    "kind": "approval",
                    "reason": str(event.get("reason") or "approval required"),
                    "requested_at": time.time(),
                }
                for key in ("action", "target", "target_id"):
                    if event.get(key):
                        self._pending_gate[key] = event[key]
                self._resolved_gate_id = None
                event["gate_id"] = self._pending_gate["gate_id"]
                self._awaiting = True
                self.status = "awaiting_approval"
            elif kind == "gate_detail" and self._pending_gate is not None:
                event["gate_id"] = self._pending_gate["gate_id"]
                if self._pending_gate.get("target_id"):
                    event["target_id"] = self._pending_gate["target_id"]
            elif kind == "gate_resolved" and self._resolved_gate_id is not None:
                event["gate_id"] = self._resolved_gate_id
                self._resolved_gate_id = None
            event = {**event, "seq": len(self.events) + 1, "ts": time.time()}
            self.events.append(event)
            if kind == "gate_resolved":
                self._awaiting = False
                if self.status == "awaiting_approval":
                    self.status = "running"

    async def wait_for_approval(self, reason: str, timeout: float) -> bool:
        with self._lock:
            # Production emits ``gate`` first. Direct SDK/test callers still
            # receive a fresh one-shot event and identity here.
            if not self._awaiting or self._pending_gate is None:
                self._approve_event = asyncio.Event()
                self._pending_gate = {
                    "gate_id": uuid.uuid4().hex,
                    "kind": "approval",
                    "reason": reason,
                    "requested_at": time.time(),
                }
                self._resolved_gate_id = None
                self._awaiting = True
            gate = self._pending_gate
            approval_event = self._approve_event
            gate["reason"] = reason
            gate["expires_at"] = time.time() + timeout
            self.status = "awaiting_approval"
        self.emit({"kind": "gate_detail", "reason": reason, "timeout_s": timeout})
        try:
            await asyncio.wait_for(approval_event.wait(), timeout=timeout)
            with self._lock:
                return bool(gate.get("approved"))
        except (asyncio.TimeoutError, TimeoutError):
            return False
        finally:
            with self._lock:
                if self._approve_event is approval_event and self._pending_gate is gate:
                    self._awaiting = False
                    self._pending_gate = None
                    self._resolved_gate_id = str(gate["gate_id"])
            approval_event.clear()

    def resolve_gate(self, approved: bool, *, gate_id: str | None = None) -> bool:
        with self._lock:
            if not self._awaiting:
                return False
            if self._pending_gate is None:
                self._pending_gate = {
                    "gate_id": uuid.uuid4().hex,
                    "kind": "approval",
                    "reason": "approval required",
                }
            current_gate_id = str(self._pending_gate["gate_id"])
            gate_target_id = self._pending_gate.get("target_id")
            if gate_target_id is not None and gate_target_id != self.active_target_id:
                raise CUAGateMismatchError(
                    "approval gate is stale because the active target changed"
                )
            if gate_id is not None and gate_id != current_gate_id:
                raise CUAGateMismatchError(
                    f"approval gate {gate_id!r} is stale; current gate is {current_gate_id!r}"
                )
            if "approved" in self._pending_gate:
                if bool(self._pending_gate["approved"]) == approved:
                    return True
                raise CUAGateDecisionConflictError(
                    f"approval gate {current_gate_id!r} already has a different decision"
                )
            self._pending_gate["approved"] = approved
            approval_event = self._approve_event
        approval_event.set()
        return True

    def approve(self) -> bool:
        return self.resolve_gate(True)

    def cancel(self) -> None:
        self._stop_event.set()

    def view(self, events_after: int = 0) -> dict:
        with self._lock:
            events = [e for e in self.events if e["seq"] > events_after]
            current_last_seq = self.events[-1]["seq"] if self.events else 0
            delivered_through = (
                events[-1]["seq"] if events else min(events_after, current_last_seq)
            )
            return {
                "run_id": self.run_id,
                "app": self.app,
                "goal": self.goal,
                "status": self.status,
                "final_summary": self.final_summary,
                "error": self.error,
                "planner": self.config.planner.describe() if self.config else "n/a",
                "events_after_seq": delivered_through,
                "events": events,
                "pending_gate": (
                    dict(self._pending_gate) if self._pending_gate else None
                ),
                "window_id": self.window_id,
                "targets": [dict(target) for target in self.targets],
                "active_target_id": self.active_target_id,
            }


class CUAService:
    """Owns the single active CUA run and its background task."""

    def __init__(self) -> None:
        self._runs: dict[str, CUAServiceRun] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._request_runs: dict[str, tuple[_RunCreateIdentity, str]] = {}
        self._create_lock = asyncio.Lock()
        self._closing = False

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": r.run_id,
                "app": r.app,
                "goal": r.goal,
                "status": r.status,
                "created_at": r.created_at,
                "window_id": r.window_id,
                "active_target_id": r.active_target_id,
            }
            for r in self._runs.values()
        ]

    def get(self, run_id: str) -> CUAServiceRun:
        run = self._runs.get(run_id)
        if run is None:
            raise CUARunNotFoundError(f"no such CUA run: {run_id}")
        return run

    def get_by_request_id(self, client_request_id: str) -> CUAServiceRun:
        entry = self._request_runs.get(client_request_id)
        if entry is None:
            raise CUARequestIdentityNotFoundError(
                f"no CUA run for client request: {client_request_id}"
            )
        run = self._runs.get(entry[1])
        if run is None:
            # Keep both retention indexes coherent even if a caller mutates the
            # registry in a test or future maintenance path.
            self._request_runs.pop(client_request_id, None)
            raise CUARequestIdentityNotFoundError(
                f"no CUA run for client request: {client_request_id}"
            )
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
        window_id: str | None = None,
        bundle_id: str | None = None,
        process_start_time: float | None = None,
        client_request_id: str | None = None,
        targets: list[dict] | None = None,
        initial_target_id: str | None = None,
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
        if window_id is not None and open_url:
            raise ValueError("open_url cannot be used with a selected window")
        if targets is not None:
            if not 2 <= len(targets) <= 3:
                raise ValueError("multi-target runs require two or three targets")
            target_ids = [str(target.get("target_id", "")) for target in targets]
            if len(target_ids) != len(set(target_ids)) or not all(target_ids):
                raise ValueError("multi-target runs require unique target_id values")
            if initial_target_id not in set(target_ids):
                raise ValueError("initial_target_id must name one submitted target")
            for target in targets:
                if str(target.get("app")) != f"pid:{int(target.get('pid', 0))}":
                    raise ValueError("target app must exactly match pid:<pid>")
            initial = next(
                target for target in targets if target["target_id"] == initial_target_id
            )
            if app != str(initial["app"]):
                raise ValueError("app must match the initial target app")
            if window_id is not None or allowed_domain or open_url:
                raise ValueError(
                    "targets cannot be combined with window_id, allowed_domain, or open_url"
                )
        identity = _RunCreateIdentity(
            app=app,
            goal=goal,
            planner=planner,
            planner_model=planner_model,
            planner_url=planner_url,
            open_url=open_url,
            allowed_domain=allowed_domain,
            max_steps=max_steps,
            human_login=human_login,
            window_id=window_id,
            bundle_id=bundle_id,
            process_start_time=process_start_time,
            targets=tuple(
                (
                    str(target["target_id"]),
                    str(target["app"]),
                    int(target["pid"]),
                    str(target["window_id"]),
                    _normalize_allowed_domain(str(target.get("allowed_domain", ""))),
                    target.get("bundle_id"),
                    target.get("process_start_time"),
                )
                for target in (targets or [])
            ),
            initial_target_id=initial_target_id,
        )
        async with self._create_lock:
            if client_request_id is not None:
                previous = self._request_runs.get(client_request_id)
                if previous is not None:
                    if previous[0] != identity:
                        raise CUARequestIdentityConflictError(
                            "client_request_id was already used with a different request"
                        )
                    return self.get(previous[1])
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
            return await self._create_locked(identity, client_request_id)

    async def _create_locked(
        self,
        identity: _RunCreateIdentity,
        client_request_id: str | None,
    ) -> CUAServiceRun:
        """Validate and commit one run while ``_create_lock`` is held."""
        (
            app,
            goal,
            planner,
            planner_model,
            planner_url,
            open_url,
            allowed_domain,
            max_steps,
            human_login,
            window_id,
            bundle_id,
            process_start_time,
            requested_targets,
            initial_target_id,
        ) = identity
        try:
            planner_cfg: PlannerConfig = resolve_planner(
                planner, url_override=planner_url, model_override=planner_model
            )
        except ValueError as exc:
            raise ValueError(f"bad planner: {exc}") from exc
        # Fail synchronously with HTTP 400 instead of accepting a run that is
        # guaranteed to die in its background task. User-consented cloud
        # brains (keyed presets) are allowed non-loopback HTTPS endpoints;
        # validate_planner_url enforces that inside Planner — this pre-flight
        # only catches the synchronous error early.
        validate_planner_url(planner_cfg.url, allow_remote=planner_cfg.allow_remote)
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
        selected_window_id: str | None = None
        selected_app: dict | None = None
        frozen_targets: list[dict] = []
        if requested_targets:
            for (
                target_id,
                selector,
                pid,
                requested_window,
                target_domain,
                expected_bundle,
                expected_start_time,
            ) in requested_targets:
                selection = await asyncio.to_thread(
                    backend.validate_window, selector, requested_window
                )
                app_info = dict(selection["app"])
                if int(app_info.get("pid", -1)) != pid:
                    raise ValueError(f"target {target_id!r} process identity changed")
                if (
                    expected_bundle is not None
                    and str(app_info.get("bundleId") or "").casefold()
                    != expected_bundle.casefold()
                ):
                    raise ValueError(f"target {target_id!r} app identity changed")
                if (
                    expected_start_time is not None
                    and float(app_info.get("processStartTime") or 0)
                    != expected_start_time
                ):
                    raise ValueError(f"target {target_id!r} process identity changed")
                bundle = str(app_info.get("bundleId") or "").lower()
                browser = bundle in {
                    "com.apple.safari",
                    "com.apple.safaritechnologypreview",
                } or bundle.startswith(
                    (
                        "com.google.chrome",
                        "com.microsoft.edgemac",
                        "org.chromium.chromium",
                    )
                )
                if browser and not target_domain.strip():
                    raise ValueError(
                        f"browser target {target_id!r} requires allowed_domain"
                    )
                frozen_targets.append(
                    {
                        "target_id": target_id,
                        "app": selector,
                        "pid": pid,
                        "window_id": str(selection["window_id"]),
                        "allowed_domain": target_domain.strip().lower().rstrip("."),
                        "expected_app": app_info,
                    }
                )
            active = next(
                target
                for target in frozen_targets
                if target["target_id"] == initial_target_id
            )
            selected_window_id = active["window_id"]
            selected_app = active["expected_app"]
            app = active["app"]
        elif window_id is not None:
            selection = await asyncio.to_thread(backend.validate_window, app, window_id)
            selected_window_id = str(selection["window_id"])
            selected_app = dict(selection["app"])
            if (
                bundle_id is not None
                and str(selected_app.get("bundleId") or "").casefold()
                != bundle_id.casefold()
            ):
                raise ValueError("selected app identity changed")
            if (
                process_start_time is not None
                and float(selected_app.get("processStartTime") or 0)
                != process_start_time
            ):
                raise ValueError("selected process identity changed")
        if self._closing:
            raise CUARunConflictError("CUA service is shutting down")
        run_id = uuid.uuid4().hex[:12]
        run = CUAServiceRun(
            run_id=run_id,
            app=app,
            goal=goal,
            config=config,
            window_id=selected_window_id,
            client_request_id=client_request_id,
            targets=[
                {key: value for key, value in target.items() if key != "expected_app"}
                for target in frozen_targets
            ],
            active_target_id=initial_target_id,
        )
        run.run_dir = ""
        self._prune_runs()
        self._runs[run_id] = run
        if client_request_id is not None:
            self._request_runs[client_request_id] = (identity, run_id)

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
                window_id=selected_window_id,
                backend_app=(
                    f"pid:{selected_app['pid']}" if selected_app is not None else None
                ),
                expected_app=selected_app,
                targets=frozen_targets or None,
                initial_target_id=initial_target_id,
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
            if expired.client_request_id is not None:
                entry = self._request_runs.get(expired.client_request_id)
                if entry is not None and entry[1] == expired.run_id:
                    self._request_runs.pop(expired.client_request_id, None)

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
