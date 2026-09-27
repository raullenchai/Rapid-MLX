"""Native macOS Accessibility CUA runner (DOM-free, semantic actions).

Same protocol as run_poc.py (semantic actions, strict-schema plans, optional
grounding verifier, optional laya outcome ranker) but the backend is the
macOS Accessibility tree + system screenshots instead of Playwright DOM
injection. No browser-specific code: any app with an AX tree is drivable.

Usage:
  python ax_runner.py --app "Google Chrome" \
    --planner-url http://127.0.0.1:18888/v1/chat/completions \
    --planner-model GLM-5.3-Flash-EXL3 --reasoning-effort low \
    --fast-ranker-url http://127.0.0.1:18700/v1/rank \
    --open-url https://www.wikipedia.org/ \
    --max-steps 10 --goal "在维基百科搜索 Apple Silicon 并打开词条"

POC quality: no AXObserver eventing, no settable-value fill probe, single app.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path

from run_poc import (  # noqa: E402
    FORBIDDEN_RE,
    FastOutcomeRanker,
    Planner,
)

from rapid_mlx.computer_use import ax_driver

# Special-key keycodes (HID usage). Everything else goes through unicode typing.
KEYCODES = {
    "Return": 36,
    "Enter": 36,
    "Escape": 53,
    "Tab": 48,
    "Space": 49,
    "Delete": 51,
    "ArrowDown": 125,
    "ArrowUp": 126,
    "ArrowLeft": 123,
    "ArrowRight": 124,
}
FILL_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
URL_FIELD_HINTS = ("address", "search bar", "url")


def log(msg: str) -> None:
    print(msg, flush=True)


def front_window(app_name: str) -> dict:
    """Return the main on-screen window (bounds + window number) of the app."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    best = None
    for window in windows:
        owner = window.get("kCGWindowOwnerName", "")
        if app_name.lower() not in str(owner).lower():
            continue
        if window.get("kCGWindowLayer", 99) != 0:
            continue
        bounds = window.get("kCGWindowBounds", {})
        area = bounds.get("Width", 0) * bounds.get("Height", 0)
        if best is None or area > best[0]:
            best = (area, window)
    if best is None:
        raise RuntimeError(f"no on-screen window for {app_name!r}")
    return best[1]


def capture_window(app_name: str, out_png: Path) -> bool:
    window = front_window(app_name)
    number = window.get("kCGWindowNumber")
    result = subprocess.run(
        ["screencapture", "-x", "-o", f"-l{number}", str(out_png)],
        capture_output=True,
    )
    return (
        result.returncode == 0 and out_png.exists() and out_png.stat().st_size > 8_000
    )


def _read_url_from_ax(targets: list[dict], app_name: str = "") -> str:
    """Browser URL for the domain guard: AXValue first, then AppleScript.

    Chrome's active-tab address bar often exposes an empty AXValue, so the
    AX scan alone under-reports; AppleScript (Automation TCC) is the
    reliable fallback, and the window title is the last resort.
    """
    for entry in targets:
        live = entry.get("element")
        if live is None:
            continue
        value = ax_driver._get(live, "AXValue")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    if app_name:
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "{app_name}" to get URL of active tab of front window',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            url = result.stdout.strip()
            if url.startswith(("http://", "https://")):
                return url
        except Exception:  # noqa: BLE001 - guard best-effort
            pass
    return ""


def _tree_signature(targets: list[dict]) -> str:
    payload = sorted(
        (t["role"], t["text"]) for t in targets if t["role"] != "AXStaticText"
    )
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[
        :10
    ]


def _text_context(targets: list[dict]) -> str:
    lines = []
    for entry in targets:
        rect = entry.get("rect")
        geometry = (
            f"@{entry['center'][0]},{entry['center'][1]}" if entry.get("center") else ""
        )
        press = " pressable" if "AXPress" in entry["actions"] else ""
        fill = " settable" if entry["role"] in FILL_ROLES else ""
        lines.append(
            f"{entry['target_id']} [{entry['role']}{press}{fill}] {entry['text'][:90]} {geometry}"
            + (
                f" rect={rect}"
                if rect and entry["role"] in FILL_ROLES | {"AXButton", "AXLink"}
                else ""
            )
        )
    return "\n".join(lines)


def _wait_tree_settle(
    app_name: str, before_sig: str, tries: int = 10, delay: float = 0.6
) -> str:
    """Poll the AX tree until it stops changing (page loads are async)."""
    signature = before_sig
    for _ in range(tries):
        time.sleep(delay)
        targets = ax_driver.collect(app_name, keep_elements=True, max_windows=1)
        signature = _tree_signature(targets)
        if signature != before_sig:
            before_sig = signature
            tries = min(tries, 2)  # allow one more change window
    return signature


def execute(plan: dict, targets: list[dict], app_name: str) -> dict:
    """Execute one semantic plan step over AX. Returns an execution record."""
    action = plan["action"]
    record: dict[str, object] = {"action": action}
    if action in {"done", "wait"}:
        if action == "wait":
            time.sleep(2.0)
        return record
    target_id = plan.get("target_id", "")
    match = next((t for t in targets if t["target_id"] == target_id), None)
    if action == "scroll":
        from Quartz import (
            CGEventCreateScrollWheelEvent,
            CGEventPost,
            kCGHIDEventTap,
            kCGScrollEventUnitLine,
        )

        direction = plan.get("direction") or "down"
        delta = -3 if direction == "up" else 3
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, delta)
        CGEventPost(kCGHIDEventTap, event)
        record["executed"] = "scroll"
        return record
    if match is None:
        record["error"] = f"unknown target_id {target_id!r}"
        return record
    element = match.get("element")
    if action == "click":
        if element is not None and "AXPress" in match["actions"]:
            import ApplicationServices as AS  # noqa: N817

            err = AS.AXUIElementPerformAction(element, "AXPress")
            record["executed"] = (
                "AXPress" if err == AS.kAXErrorSuccess else f"AXPress-err{err}"
            )
            if err == AS.kAXErrorSuccess:
                return record
        center = match.get("center")
        if not center:
            record["error"] = "no AXPress and no geometry"
            return record
        ax_driver._cg_click(float(center[0]), float(center[1]))
        record["executed"] = "CGEvent-click"
        return record
    if action == "fill":
        center = match.get("center")
        if not center:
            record["error"] = "fill target without geometry"
            return record
        ax_driver._cg_click(float(center[0]), float(center[1]))
        time.sleep(0.3)
        ax_driver._press_key(0, modifiers=ax_driver.FLAG_COMMAND)  # Cmd+A select all
        time.sleep(0.1)
        ax_driver._press_key(51)  # Delete
        ax_driver._type_text(plan.get("text", ""))
        if plan.get("submit"):
            time.sleep(0.2)
            ax_driver._press_key(36)  # Return
            record["executed"] = "fill+submit"
        else:
            record["executed"] = "fill"
        return record
    if action == "press":
        center = match.get("center")
        if center:
            ax_driver._cg_click(float(center[0]), float(center[1]))
        key = plan.get("key") or "Return"
        keycode = KEYCODES.get(key)
        if keycode is None:
            record["error"] = f"unsupported key {key!r}"
            return record
        time.sleep(0.15)
        ax_driver._press_key(keycode)
        record["executed"] = f"press:{key}"
        return record
    record["error"] = f"unsupported action {action!r}"
    return record


def _url_allowed(url: str, allowed_domain: str) -> bool:
    if not url:
        return True
    host = url.split("://", 1)[-1].split("/", 1)[0].lower().removeprefix("www.")
    return (
        (not allowed_domain)
        or host == allowed_domain
        or host.endswith("." + allowed_domain)
    )


async def run(args: argparse.Namespace) -> None:
    run_dir = Path("/tmp/ax-runs") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    planner = Planner(
        args.planner_url,
        args.planner_model,
        reasoning_effort=args.reasoning_effort,
        text_only=args.planner_text_only,
    )
    ranker = (
        FastOutcomeRanker(args.fast_ranker_url, args.fast_ranker_model)
        if args.fast_ranker_url
        else None
    )

    if args.open_url:
        subprocess.run(["open", "-a", args.app, args.open_url], check=False)
        time.sleep(4.0)

    allowed_domain = (args.allowed_domain or "").lower()
    history: list[dict] = []
    trace: dict = {
        "goal": args.goal,
        "app": args.app,
        "driver": "ax",
        "planner_model": args.planner_model,
        "steps": [],
        "started_at": time.time(),
    }
    log(f"[ax-runner] app={args.app!r} run={run_dir} goal={args.goal[:60]!r}")

    try:
        for step_no in range(1, args.max_steps + 1):
            targets = ax_driver.collect(args.app, keep_elements=True, max_windows=1)
            url_now = _read_url_from_ax(targets, args.app)
            if url_now and not _url_allowed(url_now, allowed_domain):
                log(f"[ax-runner] guard: URL left allowed domain: {url_now}")
                trace["guard_stop"] = url_now
                break
            screenshot = run_dir / f"step-{step_no:02d}-before.png"
            if not capture_window(args.app, screenshot):
                log("[ax-runner] screencapture failed (Screen Recording permission?)")
            for entry in targets:
                entry.pop("element", None)

            plan, raw_plan, latency, attempts = await planner.plan(
                args.goal,
                screenshot if screenshot.exists() else screenshot,
                _text_context(targets),
                {t["target_id"] for t in targets},
                history,
                False,
                FORBIDDEN_RE,
            )
            log(
                f"[ax-runner] step {step_no}: {plan['action']} "
                f"{plan.get('target_id', '')} {plan.get('step_instruction', '')[:60]} "
                f"({latency:.1f}s)"
            )
            record = {
                "step": step_no,
                "url": url_now,
                "plan": plan,
                "planner_latency_s": latency,
                "before_png": str(screenshot),
            }
            if plan["action"] == "done":
                record["terminal"] = True
                trace["final_summary"] = plan.get("final_summary") or "Task completed."
                trace["steps"].append(record)
                break

            sig_before = _tree_signature(targets)
            execution = execute(plan, targets, args.app)
            record["execution"] = execution
            settle_delay = (
                2.5 if plan.get("submit") or plan["action"] == "click" else 1.0
            )
            time.sleep(settle_delay)
            sig_after = _wait_tree_settle(args.app, sig_before)
            after_png = run_dir / f"step-{step_no:02d}-after.png"
            capture_window(args.app, after_png)
            url_after = _read_url_from_ax(
                ax_driver.collect(args.app, keep_elements=False)
            )

            delta = {
                "url_changed": bool(url_after and url_after != url_now),
                "tree_changed": sig_after != sig_before,
                "url_after": url_after,
            }
            record["state_delta"] = delta
            record["after_png"] = str(after_png)

            outcome = (
                "success"
                if (delta["tree_changed"] or delta["url_changed"])
                else "no_effect"
            )
            record["protocol_outcome"] = outcome
            if ranker is not None:
                try:
                    assessed, rank_latency = await ranker.assess(
                        args.goal,
                        {
                            "action": plan["action"],
                            "instruction": plan.get("step_instruction", ""),
                            "target": plan.get("target_id", ""),
                        },
                        delta,
                    )
                    record["fast_assessment"] = assessed
                except Exception as exc:  # noqa: BLE001
                    record["fast_assessment"] = {"error": str(exc)}
            if outcome == "no_effect":
                try:
                    reflection, _, _ = await planner.reflect(
                        args.goal,
                        plan.get("step_instruction", ""),
                        screenshot,
                        after_png,
                        url_now,
                        url_after,
                        delta,
                    )
                    record["reflection"] = reflection
                except Exception as exc:  # noqa: BLE001
                    record["reflection"] = {"error": str(exc)}
            history.append(
                {
                    "step": step_no,
                    "action": plan["action"],
                    "instruction": plan.get("step_instruction", ""),
                    "outcome": outcome,
                    "delta": delta,
                }
            )
            trace["steps"].append(record)
            (run_dir / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8"
            )
    finally:
        trace["finished_at"] = time.time()
        (run_dir / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        await planner.close()
        if ranker is not None:
            await ranker.close()
    log(f"[ax-runner] done: {run_dir / 'trace.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--planner-url", required=True)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--fast-ranker-url", default=None)
    parser.add_argument("--fast-ranker-model", default=None)
    parser.add_argument("--planner-text-only", action="store_true")
    parser.add_argument("--open-url", default=None, help="URL to open before planning")
    parser.add_argument(
        "--allowed-domain", default="", help="pin navigation to one host"
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--goal", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
