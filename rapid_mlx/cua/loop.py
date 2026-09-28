"""The agent loop: observe -> fast checks -> slow plan -> act -> assess.

Replaces the GUI-verifier POC loop with the productized native-AX pipeline:
- observation: fresh accessibility snapshot every step (no stale caches)
- fast thinking: local laya outcome routing + no-progress fixation gate
- slow thinking: user-configured planner (cloud or on-device)
- consent: credential/commerce hard stops, optional sign-in human gate
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rapid_mlx.computer_use import backend
from rapid_mlx.computer_use.errors import ComputerUseError
from rapid_mlx.cua import config as config_mod
from rapid_mlx.cua import gates
from rapid_mlx.cua.config import CUAConfig
from rapid_mlx.cua.fast import FastOutcomeRanker, NoProgressTracker
from rapid_mlx.cua.gates import ConsentError
from rapid_mlx.cua.planner import Planner


def _tree_signature(snapshot: dict) -> str:
    return hashlib.sha1(
        snapshot.get("tree_text", "").encode("utf-8", "replace")
    ).hexdigest()[:12]


def _open_url(app: str, url: str) -> None:
    subprocess.run(["open", "-a", app, url], check=False, timeout=15)


class CUARun:
    """One agent run.

    `event_sink` receives progress dicts (plan/executed/gate/terminal) for
    server/GUI consumers; `gate` is an async approval callback that replaces
    the file sentinel when a UI can ask the human directly; `stop_event`
    cooperatively cancels the run between steps.
    """

    def __init__(
        self,
        config: CUAConfig,
        app: str,
        goal: str,
        run_dir: Path,
        event_sink: Callable[[dict], None] | None = None,
        gate: Callable[[str], Any] | None = None,
        stop_event: asyncio.Event | None = None,
        window_id: str | None = None,
        backend_app: str | None = None,
        expected_app: dict | None = None,
    ):
        self.config = config
        self.app = app
        self.goal = goal
        self.run_dir = run_dir
        self.event_sink = event_sink
        self.gate = gate
        self.stop_event = stop_event or asyncio.Event()
        self.window_id = window_id
        self.backend_app = backend_app or app
        self.expected_app = dict(expected_app) if expected_app is not None else None
        run_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.trace: dict = {
            "app": app,
            "goal": goal,
            "planner": config.planner.describe(),
            "steps": [],
            "window_id": window_id,
        }
        self.tracker = NoProgressTracker()
        self._last_execution_failed = False
        self._failed_completion_rejections = 0
        self._trusted_transient_window_id: str | None = None
        self._empty_snapshots = 0
        self._terminal_emitted = False
        self.ranker = (
            FastOutcomeRanker(config.fast_ranker_url)
            if config.fast_ranker_url
            else None
        )

    def _record(self, step: dict) -> None:
        self.trace["steps"].append(step)
        (self.run_dir / "trace.json").write_text(
            json.dumps(self.trace, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _emit(self, event: dict) -> None:
        if event.get("kind") == "terminal":
            self._terminal_emitted = True
        if self.event_sink is None:
            return
        try:
            self.event_sink(event)
        except Exception:  # noqa: BLE001 - events must never kill the run
            pass

    async def _request_approval(
        self, reason: str, *, action: str = "", target: str = ""
    ) -> bool:
        """Ask the human to approve one explicitly described action."""
        if self.gate is None:
            marker = (
                "APPROVE_SIGNIN"
                if reason == "sign-in"
                else f"APPROVE_ACTION_{hashlib.sha256(reason.encode()).hexdigest()[:12]}"
            )
            if reason == "sign-in":
                return await gates.wait_for_human(
                    self.run_dir, marker, self.config.pause_timeout
                )
            return await gates.wait_for_human(
                self.run_dir, marker, self.config.pause_timeout, reason=reason
            )
        event = {"kind": "gate", "reason": reason}
        if action:
            event.update({"action": action, "target": target})
        self._emit(event)
        try:
            approved = bool(await self.gate(reason))
        except Exception:  # noqa: BLE001 - a broken gate must not hang the run
            approved = False
        self._emit(
            {
                "kind": "gate_resolved",
                "reason": reason,
                "approved": approved,
                "action": action,
                "target": target,
            }
        )
        return approved

    async def _request_signin_approval(self) -> bool:
        """Ask the human to approve a sign-in pause.

        With a `gate` callback (server/GUI mode) the pause is surfaced as
        gate/gate_resolved events and resolved through the callback; without
        one (CLI mode) the file sentinel is the approval channel.
        """
        return await self._request_approval("sign-in")

    @staticmethod
    def _target(snapshot: dict, index: int) -> dict:
        return next(
            (e for e in snapshot.get("elements", []) if e.get("index") == index), {}
        )

    @staticmethod
    def _target_identity(target: dict) -> tuple[str, ...] | None:
        required = ("index", "role", "label", "x", "y", "width", "height", "center")
        if any(field not in target for field in required):
            return None
        return tuple(
            json.dumps(target.get(field), ensure_ascii=False, sort_keys=True)
            for field in (*required, "subrole", "actions", "source_window_id")
        )

    @staticmethod
    def _window_identity(snapshot: dict) -> tuple[str, str, str, str, str] | None:
        app = snapshot.get("app")
        if (
            not isinstance(app, dict)
            or "pid" not in app
            or "window_index" not in snapshot
        ):
            return None
        return (
            str(app.get("pid")),
            str(app.get("bundle_id", "")),
            str(app.get("name", "")),
            str(snapshot.get("window_index")),
            str(snapshot.get("window_id", "")),
        )

    def _check_domain(self, url: str) -> str | None:
        allowed = self.config.allowed_domain.strip().lower().rstrip(".")
        if not allowed:
            return None
        if not url:
            return "domain guard: current URL could not be read; refusing to act"
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not hostname:
            return f"domain guard: current URL {url!r} is not a valid HTTP(S) URL"
        if hostname != allowed and not hostname.endswith(f".{allowed}"):
            return (
                f"domain guard: current URL {url!r} is outside "
                f"--allowed-domain {allowed!r}"
            )
        return None

    def _get_app_state(
        self, *, screenshot: bool, transient_baseline: set[str] | None = None
    ) -> dict:
        kwargs: dict[str, Any] = {"screenshot": screenshot, "use_cache": False}
        if self.window_id is not None:
            kwargs["window_id"] = self.window_id
            if self._trusted_transient_window_id is not None:
                kwargs["trusted_transient_window_id"] = (
                    self._trusted_transient_window_id
                )
            if transient_baseline is not None:
                kwargs["transient_baseline_window_ids"] = transient_baseline
        snapshot = backend.get_app_state(self.backend_app, **kwargs)
        transient = snapshot.get("transient_window")
        self._trusted_transient_window_id = (
            str(transient["window_id"]) if isinstance(transient, dict) else None
        )
        if self.expected_app is not None:
            observed = snapshot.get("app") or {}
            for key in ("pid", "bundleId", "name"):
                expected = self.expected_app.get(key)
                if expected is not None and observed.get(key) != expected:
                    raise ComputerUseError(
                        "target_drift",
                        f"selected app identity changed for pid {self.expected_app.get('pid')}",
                    )
        return snapshot

    async def _execute(self, plan: dict, snapshot: dict) -> dict:
        action = plan["action"]
        index = plan.get("element_index", -1)
        result: dict = {"action": action}
        try:
            if action == "click":
                result.update(
                    backend.click(self.backend_app, index, expected_snapshot=snapshot)
                )
            elif action == "fill":
                result.update(
                    backend.set_value(
                        self.backend_app,
                        index,
                        plan.get("text", ""),
                        expected_snapshot=snapshot,
                    )
                )
            elif action == "press":
                backend.click(
                    self.backend_app,
                    index,
                    expected_snapshot=snapshot,
                    focus_only=True,
                )
                await asyncio.sleep(0.2)
                result.update(
                    backend.press_key(
                        self.backend_app,
                        plan.get("key", "Enter"),
                        expected_snapshot=snapshot,
                        element_index=index,
                    )
                )
            elif action == "scroll":
                result.update(
                    backend.scroll(
                        self.backend_app,
                        plan.get("direction", "down"),
                        1.0,
                        expected_snapshot=snapshot,
                    )
                )
            elif action == "wait":
                await asyncio.sleep(2.0)
                result.update({"ok": True})
        except ComputerUseError as exc:
            # A tool failure is an action-level outcome, not a run-level
            # crash: the tracker records it and the next step re-observes.
            result.update(
                {
                    "ok": False,
                    "error": exc.message,
                    "error_code": exc.code,
                    "recovery": list(exc.recovery),
                    "executed": False,
                }
            )
            return result
        result["executed"] = action != "wait"
        return result

    async def step(self, planner: Planner, step_no: int) -> dict | None:
        """One loop iteration. Returns a terminal record or None to continue."""
        if self.stop_event.is_set():
            self.trace["status"] = "stopped"
            self.trace["final_summary"] = "cancelled by client"
            self._record({"step": step_no, "stop": "cancelled by client"})
            self._emit({"kind": "terminal", "status": "stopped"})
            return {"status": "stopped", "reason": "cancelled by client"}
        try:
            snapshot = self._get_app_state(screenshot=not planner.text_only)
        except ComputerUseError as exc:
            if self.window_id is not None:
                reason = f"selected window unavailable: {exc.message}"
                self.trace["status"] = "stopped"
                self.trace["final_summary"] = reason
                self._record({"step": step_no, "stop": reason, "error_code": exc.code})
                self._emit(
                    {
                        "kind": "terminal",
                        "status": "stopped",
                        "reason": reason,
                        "error": exc.code,
                    }
                )
                return {"status": "stopped", "reason": reason, "error": exc.code}
            # AX watchdog tripped (wedged app accessibility service): treat as
            # an unusable snapshot and let the honest-stop path handle it.
            snapshot = {
                "app": {"name": self.app},
                "elements": [],
                "tree_text": "",
                "ax_unavailable": True,
                "snapshot_error": str(exc),
            }
        if not snapshot.get("elements") or snapshot.get("ax_unavailable"):
            # Dogfooding find (2026-09-27): when the target app's AX tree is
            # unavailable (e.g. Chrome's accessibility service wedged), an
            # empty snapshot used to reach the planner, whose guesses (index 0)
            # crashed the whole run as "incomplete". Fail honestly instead.
            self._empty_snapshots += 1
            reason = (
                f"accessibility tree for {self.app!r} is unavailable "
                f"({self._empty_snapshots} empty snapshot(s))"
            )
            self._record({"step": step_no, "stop": reason})
            self._emit(
                {
                    "kind": "executed",
                    "step": step_no,
                    "action": "observe",
                    "outcome": "unavailable",
                    "tree_changed": False,
                }
            )
            if self._empty_snapshots >= 2:
                self.trace["status"] = "stopped"
                self.trace["final_summary"] = reason
                self._emit({"kind": "terminal", "status": "stopped", "reason": reason})
                return {"status": "stopped", "reason": reason}
            return None
        self._empty_snapshots = 0
        url_now = backend.read_url(
            self.backend_app, window_id=self.window_id or snapshot.get("window_id")
        )
        guard = self._check_domain(url_now)
        if guard:
            self.trace["guard_stop"] = guard
            self._record({"step": step_no, "stop": guard, "url": url_now})
            return {"status": "stopped", "reason": guard}

        progress_hint = (
            self.tracker.take_hint(snapshot) if self.tracker.should_intervene() else ""
        )
        plan, raw, latency, attempts = await planner.plan(
            self.goal,
            snapshot,
            self.history,
            self.config.allowed_domain,
            progress_hint,
        )
        target = self._target(snapshot, plan.get("element_index", -1))
        target_label = str(target.get("label", ""))
        try:
            gates.check_plan_consents(plan, target_label)
        except ConsentError as exc:
            self.trace["consent_stop"] = str(exc)
            self._record({"step": step_no, "plan": plan, "consent_stop": str(exc)})
            return {"status": "stopped", "reason": str(exc)}

        self._emit(
            {
                "kind": "plan",
                "step": step_no,
                "action": plan["action"],
                "step_instruction": plan["step_instruction"],
                "element_index": plan.get("element_index", -1),
                "target_label": target_label[:120],
                "latency_s": round(latency, 2),
            }
        )
        if plan["action"] == "done":
            if self._last_execution_failed:
                self._failed_completion_rejections += 1
                reason = (
                    "the previous action failed; use the fresh observation to recover"
                )
                self.history.append(
                    {
                        "step": step_no,
                        "action": "done",
                        "instruction": plan["step_instruction"][:120],
                        "outcome": "no_effect",
                        "executed": False,
                        "error": reason,
                    }
                )
                self.tracker.record(plan, "no_effect")
                self._record(
                    {
                        "step": step_no,
                        "plan": plan,
                        "completion_rejected": reason,
                    }
                )
                if self._failed_completion_rejections >= 2:
                    return {"status": "stopped", "reason": reason}
                return None
            self.trace["final_summary"] = plan["final_summary"]
            self._record({"step": step_no, "plan": plan, "latency_s": latency})
            return {"status": "done", "summary": plan["final_summary"]}

        if self.config.human_login and gates.looks_like_sign_in(snapshot):
            approved = await self._request_signin_approval()
            if not approved:
                self.trace["human_gate"] = "sign-in gate timed out"
                self._record({"step": step_no, "plan": plan, "gate": "timeout"})
                return {"status": "stopped", "reason": "sign-in gate not approved"}

        approval = gates.consequential_action(plan, target_label)
        if approval is not None:
            approval_reason = f"{approval.reason}; app={self.app!r}"
            approved = await self._request_approval(
                approval_reason, action=approval.action, target=approval.target
            )
            if not approved:
                self.trace["human_gate"] = f"{approval.kind} not approved"
                self._record({"step": step_no, "plan": plan, "gate": "not approved"})
                return {
                    "status": "stopped",
                    "reason": f"{approval.kind} not approved",
                }

            # Approval binds to the observed target. Re-observe after the human
            # pause and fail closed if the indexed control or domain changed.
            try:
                fresh = self._get_app_state(screenshot=not planner.text_only)
            except ComputerUseError as exc:
                reason = (
                    f"selected window unavailable after approval: {exc.message}"
                    if self.window_id is not None
                    else f"could not revalidate approved target: {exc}"
                )
                self._record(
                    {
                        "step": step_no,
                        "plan": plan,
                        "gate": "stale",
                        "stop": reason,
                        "error_code": exc.code,
                    }
                )
                return {"status": "stopped", "reason": reason, "error": exc.code}
            fresh_url = backend.read_url(
                self.backend_app, window_id=self.window_id or fresh.get("window_id")
            )
            fresh_guard = self._check_domain(fresh_url)
            fresh_target = self._target(fresh, plan.get("element_index", -1))
            fresh_label = str(fresh_target.get("label", ""))
            original_target_identity = self._target_identity(target)
            fresh_target_identity = self._target_identity(fresh_target)
            original_window_identity = self._window_identity(snapshot)
            fresh_window_identity = self._window_identity(fresh)
            stale = any(
                (
                    original_target_identity is None,
                    fresh_target_identity is None,
                    original_target_identity != fresh_target_identity,
                    original_window_identity is None,
                    fresh_window_identity is None,
                    original_window_identity != fresh_window_identity,
                    snapshot.get("window") != fresh.get("window"),
                    fresh_url != url_now,
                    _tree_signature(fresh) != _tree_signature(snapshot),
                )
            )
            try:
                gates.check_plan_consents(plan, fresh_label)
            except ConsentError as exc:
                self.trace["consent_stop"] = str(exc)
                self._record({"step": step_no, "plan": plan, "consent_stop": str(exc)})
                return {"status": "stopped", "reason": str(exc)}
            if fresh_guard or stale:
                reason = fresh_guard or "approved target changed before execution"
                self._record(
                    {"step": step_no, "plan": plan, "gate": "stale", "stop": reason}
                )
                result = {"status": "stopped", "reason": reason}
                if self.window_id is not None and stale:
                    result["error"] = "window_stale"
                return result
            snapshot = fresh
            url_now = fresh_url

        # Planning and human approval are await points during which the active
        # browser location can change. Enforce the domain boundary again at
        # the last possible moment before any input is dispatched.
        if self.window_id is not None and approval is None:
            try:
                fresh = self._get_app_state(screenshot=not planner.text_only)
            except ComputerUseError as exc:
                reason = f"selected window unavailable before action: {exc.message}"
                self._record(
                    {
                        "step": step_no,
                        "plan": plan,
                        "stop": reason,
                        "error_code": exc.code,
                    }
                )
                return {"status": "stopped", "reason": reason, "error": exc.code}
            if self._window_identity(snapshot) != self._window_identity(
                fresh
            ) or snapshot.get("window") != fresh.get("window"):
                reason = "selected window moved or was replaced before action"
                self._record({"step": step_no, "plan": plan, "stop": reason})
                return {"status": "stopped", "reason": reason, "error": "window_stale"}
            original_target = self._target(snapshot, plan.get("element_index", -1))
            fresh_target = self._target(fresh, plan.get("element_index", -1))
            if self._target_identity(original_target) != self._target_identity(
                fresh_target
            ):
                reason = "planned target changed before action"
                self._record({"step": step_no, "plan": plan, "stop": reason})
                return {"status": "stopped", "reason": reason, "error": "target_stale"}
            snapshot = fresh
        pre_action_url = backend.read_url(
            self.backend_app, window_id=self.window_id or snapshot.get("window_id")
        )
        pre_action_guard = self._check_domain(pre_action_url)
        if pre_action_guard:
            self.trace["guard_stop"] = pre_action_guard
            self._record(
                {
                    "step": step_no,
                    "plan": plan,
                    "stop": pre_action_guard,
                    "url": pre_action_url,
                }
            )
            return {"status": "stopped", "reason": pre_action_guard}
        url_now = pre_action_url
        before_sig = _tree_signature(snapshot)
        executed = await self._execute(plan, snapshot)
        await asyncio.sleep(1.2)
        try:
            after = self._get_app_state(
                screenshot=False,
                transient_baseline=set(snapshot.get("visible_window_ids", [])),
            )
        except ComputerUseError as exc:
            if self.window_id is None:
                raise
            reason = f"selected window unavailable after action: {exc.message}"
            self._record(
                {"step": step_no, "plan": plan, "stop": reason, "error_code": exc.code}
            )
            return {"status": "stopped", "reason": reason, "error": exc.code}
        after_sig = _tree_signature(after)
        url_after = backend.read_url(
            self.backend_app, window_id=self.window_id or after.get("window_id")
        )
        delta = {
            "executed": executed,
            "tree_changed": before_sig != after_sig,
            "url_before": url_now,
            "url_after": url_after,
        }

        execution_rejected = executed.get("ok") is False
        verification = executed.get("verified")
        verification_failed = verification is False
        execution_failed = execution_rejected or verification_failed
        if execution_failed:
            outcome = "no_effect"
        elif verification is True:
            outcome = "success"
        else:
            outcome = "uncertain"
        ranker_latency = 0.0
        if execution_failed:
            # Executor rejection and explicit failed verification are observed
            # facts. A probabilistic verifier must never relabel either one.
            delta["fast_outcome"] = {
                "outcome": outcome,
                "confidence": 1.0,
                "source": (
                    "execution" if execution_rejected else "execution-verification"
                ),
            }
        elif verification is True:
            # Exact readback is stronger evidence than a semantic classifier.
            delta["fast_outcome"] = {
                "outcome": outcome,
                "confidence": 1.0,
                "source": "execution-verification",
            }
        elif self.ranker is not None:
            try:
                verdict, ranker_latency = await self.ranker.assess(
                    self.goal, plan, delta
                )
                # The current ranker sees only coarse structured deltas. Keep
                # its result for diagnostics, but do not promote an unverified
                # dispatch to user-visible success (or suppress recovery).
                delta["fast_outcome"] = {**verdict, "advisory": True}
            except (RuntimeError, KeyError, ValueError):
                delta["fast_outcome"] = {"outcome": "unavailable"}
        self.tracker.record(plan, outcome)
        event = {
            "kind": "executed",
            "step": step_no,
            "action": plan["action"],
            "outcome": outcome,
            "tree_changed": delta["tree_changed"],
            "url_after": url_after[:120],
        }
        history_entry = {
            "step": step_no,
            "action": plan["action"],
            "instruction": plan["step_instruction"][:120],
            "outcome": outcome,
            "url_after": url_after[:120],
        }
        if execution_rejected:
            event.update(
                {
                    "error": str(executed.get("error", "action was not executed"))[
                        :240
                    ],
                    "error_code": str(executed.get("error_code", "execution_failed")),
                }
            )
            history_entry.update(
                {
                    "executed": False,
                    "error": event["error"],
                    "error_code": event["error_code"],
                    "recovery": list(executed.get("recovery", [])),
                }
            )
        if execution_failed:
            self._last_execution_failed = True
        elif executed.get("executed") is True:
            # A dispatched action followed by the fresh `after` observation is
            # enough to let the planner assess completion on its next turn.
            self._last_execution_failed = False
            self._failed_completion_rejections = 0
        self._emit(event)
        self.history.append(history_entry)
        self._record(
            {
                "step": step_no,
                "plan": plan,
                "plan_attempts": attempts,
                "planner_latency_s": round(latency, 2),
                "execution": executed,
                "state_delta": delta,
                "protocol_outcome": outcome,
                "fast_ranker_latency_s": round(ranker_latency, 3),
                "progress_interventions": self.tracker.interventions,
            }
        )
        if self.tracker.exhausted():
            reason = "planner fixated after repeated interventions"
            self.trace["final_summary"] = plan["final_summary"] or reason
            self.trace["stalled"] = True
            return {"status": "stalled", "reason": reason}
        return None


async def run(
    config: CUAConfig,
    app: str,
    goal: str,
    open_url: str = "",
    max_steps: int | None = None,
    planner: Planner | None = None,
    event_sink: Callable[[dict], None] | None = None,
    gate: Callable[[str], Any] | None = None,
    stop_event: asyncio.Event | None = None,
    window_id: str | None = None,
    backend_app: str | None = None,
    expected_app: dict | None = None,
) -> dict:
    """Run the loop. Pass `planner` to inject a custom brain (SDK/testing use)."""
    run_dir = config_mod.RUNS_DIR / (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    if open_url:
        await asyncio.to_thread(_open_url, app, open_url)
        await asyncio.sleep(6.0)
    if planner is None:
        planner = Planner(
            url=config.planner.url,
            model=config.planner.model,
            reasoning_effort=config.planner.reasoning_effort,
            text_only=config.planner.text_only,
            timeout=config.planner.timeout,
            api_key=config.planner.api_key,
            allow_remote=config.planner.allow_remote,
        )
    cua_run = CUARun(
        config,
        app,
        goal,
        run_dir,
        event_sink=event_sink,
        gate=gate,
        stop_event=stop_event,
        window_id=window_id,
        backend_app=backend_app,
        expected_app=expected_app,
    )
    cua_run._emit(
        {"kind": "started", "app": app, "run_dir": str(run_dir), "window_id": window_id}
    )
    limit = max_steps or config.max_steps
    terminal: dict = {"status": "incomplete"}
    try:
        for step_no in range(1, limit + 1):
            print(
                f"[cua] step {step_no}: planning with {config.planner.describe()}",
                flush=True,
            )
            result = await cua_run.step(planner, step_no)
            if result is not None:
                terminal = result
                break
        else:
            cua_run.trace["max_steps_reached"] = limit
            reason = f"maximum step count reached ({limit})"
            cua_run.trace["final_summary"] = reason
            terminal = {"status": "stalled", "reason": reason}
    except (ValueError, KeyError) as exc:
        # A planner repair exhaustion or malformed plan must not surface as a
        # bare traceback or raw model response in the user-facing result.
        reason = "planner did not return a valid action after one repair attempt"
        cua_run.trace["planner_error"] = str(exc)[:800]
        cua_run.trace["status"] = "stopped"
        cua_run.trace["final_summary"] = reason
        terminal = {"status": "stopped", "reason": reason}
    except asyncio.CancelledError:
        reason = "cancelled by client"
        cua_run.trace["final_summary"] = reason
        terminal = {"status": "stopped", "reason": reason}
        raise
    except Exception as exc:
        terminal = {"status": "failed", "reason": str(exc)[:400]}
        raise
    finally:
        cua_run.trace["status"] = terminal.get("status", "incomplete")
        reason = str(terminal.get("reason", ""))
        error = str(terminal.get("error", ""))
        if reason and not cua_run.trace.get("final_summary"):
            cua_run.trace["final_summary"] = reason
        if not cua_run._terminal_emitted:
            public_status = {
                "done": "completed",
                "incomplete": "stalled",
            }.get(str(terminal.get("status")), str(terminal.get("status")))
            cua_run._emit(
                {
                    "kind": "terminal",
                    "status": public_status,
                    "final_summary": str(cua_run.trace.get("final_summary", "")),
                    **({"reason": reason} if reason else {}),
                    **({"error": error} if error else {}),
                }
            )
        (run_dir / "trace.json").write_text(
            json.dumps(cua_run.trace, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        await planner.close()
        if cua_run.ranker is not None:
            await cua_run.ranker.close()
    cua_run.trace["run_dir"] = str(run_dir)
    print(f"[cua] finished: {terminal} trace={run_dir / 'trace.json'}", flush=True)
    return cua_run.trace
