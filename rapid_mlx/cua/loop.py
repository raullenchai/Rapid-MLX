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
from pathlib import Path
from urllib.parse import urlparse

from rapid_mlx.computer_use import backend
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
    def __init__(self, config: CUAConfig, app: str, goal: str, run_dir: Path):
        self.config = config
        self.app = app
        self.goal = goal
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.trace: dict = {
            "app": app,
            "goal": goal,
            "planner": config.planner.describe(),
            "steps": [],
        }
        self.tracker = NoProgressTracker()
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

    async def _execute(self, plan: dict, snapshot: dict) -> dict:
        action = plan["action"]
        index = plan.get("element_index", -1)
        result: dict = {"action": action}
        if action == "click":
            result.update(backend.click(self.app, index))
        elif action == "fill":
            backend.click(self.app, index)
            result.update(backend.set_value(self.app, index, plan.get("text", "")))
        elif action == "press":
            backend.click(self.app, index)
            await asyncio.sleep(0.2)
            result.update(backend.press_key(self.app, plan.get("key", "Enter")))
        elif action == "scroll":
            result.update(backend.scroll(self.app, plan.get("direction", "down"), 1.0))
        elif action == "wait":
            await asyncio.sleep(2.0)
            result.update({"ok": True})
        result["executed"] = action != "wait"
        return result

    async def step(self, planner: Planner, step_no: int) -> dict | None:
        """One loop iteration. Returns a terminal record or None to continue."""
        snapshot = backend.get_app_state(
            self.app, screenshot=not planner.text_only, use_cache=False
        )
        url_now = backend.read_url(self.app)
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
        target: dict = next(
            (
                e
                for e in snapshot.get("elements", [])
                if e["index"] == plan.get("element_index")
            ),
            {},
        )
        target_label = str(target.get("label", ""))
        try:
            gates.check_plan_consents(plan, target_label)
        except ConsentError as exc:
            self.trace["consent_stop"] = str(exc)
            self._record({"step": step_no, "plan": plan, "consent_stop": str(exc)})
            return {"status": "stopped", "reason": str(exc)}

        if plan["action"] == "done":
            self.trace["final_summary"] = plan["final_summary"]
            self._record({"step": step_no, "plan": plan, "latency_s": latency})
            return {"status": "done", "summary": plan["final_summary"]}

        if self.config.human_login and gates.looks_like_sign_in(snapshot):
            approved = await gates.wait_for_human(
                self.run_dir, "APPROVE_SIGNIN", self.config.pause_timeout
            )
            if not approved:
                self.trace["human_gate"] = "sign-in gate timed out"
                self._record({"step": step_no, "plan": plan, "gate": "timeout"})
                return {"status": "stopped", "reason": "sign-in gate not approved"}

        before_sig = _tree_signature(snapshot)
        executed = await self._execute(plan, snapshot)
        await asyncio.sleep(1.2)
        after = backend.get_app_state(self.app, screenshot=False, use_cache=False)
        after_sig = _tree_signature(after)
        url_after = backend.read_url(self.app)
        delta = {
            "executed": executed,
            "tree_changed": before_sig != after_sig,
            "url_before": url_now,
            "url_after": url_after,
        }

        outcome = "uncertain"
        ranker_latency = 0.0
        if self.ranker is not None:
            try:
                verdict, ranker_latency = await self.ranker.assess(
                    self.goal, plan, delta
                )
                outcome = verdict["outcome"]
                delta["fast_outcome"] = verdict
            except (RuntimeError, KeyError, ValueError):
                delta["fast_outcome"] = {"outcome": "unavailable"}
        self.tracker.record(plan, outcome)
        self.history.append(
            {
                "step": step_no,
                "action": plan["action"],
                "instruction": plan["step_instruction"][:120],
                "outcome": outcome,
                "url_after": url_after[:120],
            }
        )
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
        )
    cua_run = CUARun(config, app, goal, run_dir)
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
    finally:
        cua_run.trace["status"] = terminal.get("status", "incomplete")
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
