"""Tests for the productized CUA loop (rapid_mlx.cua).

All external effects are stubbed: no Accessibility calls, no HTTP servers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapid_mlx.cua import gates
from rapid_mlx.cua.config import (
    CUAConfig,
    PlannerConfig,
    load_config,
    resolve_planner,
)
from rapid_mlx.cua.fast import NoProgressTracker
from rapid_mlx.cua.gates import ConsentError
from rapid_mlx.cua.planner import validate_plan


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    from rapid_mlx.cua import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cua-config.json")
    monkeypatch.setattr(config_mod, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


# ---------------------------------------------------------------- config


def test_defaults_created_on_first_load(config_dir):
    data = load_config()
    assert "cloud-glm" in data["presets"]
    assert "local-9b" in data["presets"]
    assert (config_dir / "cua-config.json").exists()


def test_resolve_preset(config_dir):
    planner = resolve_planner("local-9b")
    assert planner.text_only is True
    assert "18702" in planner.url


def test_resolve_custom_url_requires_model(config_dir):
    with pytest.raises(ValueError, match="--planner-model"):
        resolve_planner("http://127.0.0.1:9999/v1/chat/completions")
    planner = resolve_planner(
        "http://127.0.0.1:9999/v1/chat/completions", model_override="m"
    )
    assert planner.model == "m"


def test_resolve_unknown_preset_lists_known(config_dir):
    with pytest.raises(ValueError, match="local-9b"):
        resolve_planner("nope")


# ---------------------------------------------------------------- plan validation


def test_click_requires_index():
    with pytest.raises(ValueError, match="element_index"):
        validate_plan({"action": "click", "step_instruction": "x", "element_index": -1})
    plan = validate_plan(
        {"action": "click", "step_instruction": "x", "element_index": 7}
    )
    assert plan["element_index"] == 7


def test_press_key_allowlist():
    with pytest.raises(ValueError, match="press key"):
        validate_plan(
            {"action": "press", "step_instruction": "x", "element_index": 1, "key": "Q"}
        )
    plan = validate_plan(
        {"action": "press", "step_instruction": "x", "element_index": 1, "key": "Enter"}
    )
    assert plan["key"] == "Enter"


def test_done_requires_summary():
    with pytest.raises(ValueError, match="final_summary"):
        validate_plan({"action": "done", "step_instruction": "x", "final_summary": ""})


def test_sensitive_plan_rejected():
    with pytest.raises(ValueError, match="credentials"):
        validate_plan(
            {
                "action": "fill",
                "step_instruction": "enter password",
                "element_index": 2,
                "text": "hunter2",
            }
        )


def test_unknown_index_rejected():
    with pytest.raises(ValueError, match="unknown element_index"):
        validate_plan(
            {"action": "click", "step_instruction": "x", "element_index": 99},
            valid_indexes={1, 2, 3},
        )


# ---------------------------------------------------------------- fast path


def test_tracker_detects_repetition_and_resets():
    tracker = NoProgressTracker()
    plan = {"action": "click", "step_instruction": "Click the search button"}
    for _ in range(3):
        tracker.record(plan, "success")
    assert tracker.should_intervene()
    hint = tracker.take_hint({"elements": [{"index": 1, "label": "Back"}]})
    assert "PROGRESS CHECK" in hint
    assert "search button" in hint
    assert not tracker.should_intervene()


def test_tracker_detects_stall():
    tracker = NoProgressTracker()
    for outcome in ("no_effect", "no_effect", "wrong_effect"):
        tracker.record({"step_instruction": f"step {outcome}"}, outcome)
    assert tracker.should_intervene()


def test_tracker_exhaustion():
    tracker = NoProgressTracker()
    plan = {"step_instruction": "same"}
    for _ in range(6):
        tracker.record(plan, "success")
        if tracker.should_intervene():
            tracker.take_hint({})
    assert tracker.exhausted()


# ---------------------------------------------------------------- gates


def test_commerce_fill_rejected():
    with pytest.raises(ConsentError, match="cart"):
        gates.check_plan_consents(
            {"action": "fill", "step_instruction": "x", "text": "1"},
            target_label="Add to cart",
        )


def test_credentials_in_target_label_rejected():
    with pytest.raises(ConsentError, match="credentials"):
        gates.check_plan_consents(
            {"action": "fill", "step_instruction": "fill the field", "text": "x"},
            target_label="Card number",
        )


def test_normal_plan_passes():
    gates.check_plan_consents(
        {"action": "click", "step_instruction": "open the article", "text": ""},
        target_label="Apple Silicon - Wikipedia",
    )


def test_sign_in_detection():
    snapshot = {
        "elements": [
            {"label": "Sign in"},
            {"role": "AXSecureTextField", "label": "Password"},
        ]
    }
    assert gates.looks_like_sign_in(snapshot)
    assert not gates.looks_like_sign_in(
        {"elements": [{"label": "Sign in to Wikipedia"}]}
    )


def test_human_gate_approve_file(tmp_path):
    import threading

    marker = "APPROVE_SIGNIN"

    def approve_later():
        import time

        time.sleep(0.5)
        (tmp_path / marker).write_text("")

    threading.Thread(target=approve_later, daemon=True).start()
    assert gates.wait_for_human(tmp_path, marker, timeout=5.0)


# ---------------------------------------------------------------- loop


class _FakePlanner:
    """Scripted planner: repeats one instruction, then gives up or finishes."""

    def __init__(self, plans, text_only=True):
        self.plans = list(plans)
        self.text_only = text_only
        self.calls = 0

    async def plan(self, goal, snapshot, history, allowed_domain="", progress_hint=""):
        self.calls += 1
        return dict(self.plans[min(self.calls - 1, len(self.plans) - 1)]), "", 0.01, []

    async def close(self):
        pass


@pytest.fixture()
def fake_backend(monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    state = {"steps": 0}

    def fake_get_app_state(app, screenshot=True, use_cache=True):
        state["steps"] += 1
        return {
            "app": {"name": app},
            "elements": [{"index": 1, "label": "Search", "role": "AXTextField"}],
            "tree_text": "[1] AXTextField Search",
            "screenshot_png": b"png" if screenshot else None,
        }

    monkeypatch.setattr(backend_mod, "get_app_state", fake_get_app_state)
    monkeypatch.setattr(
        backend_mod, "read_url", lambda app: "https://www.wikipedia.org/"
    )
    monkeypatch.setattr(
        backend_mod,
        "click",
        lambda app, index, **kw: {"ok": True, "mode": "AXPress"},
    )
    monkeypatch.setattr(
        backend_mod,
        "set_value",
        lambda app, index, value: {"ok": True, "verified": True, "actual": value},
    )
    monkeypatch.setattr(
        backend_mod, "press_key", lambda app, key: {"ok": True, "key": key}
    )
    return backend_mod


def _make_config(tmp_path):
    return CUAConfig(
        planner=PlannerConfig(
            preset="test", url="http://127.0.0.1:1/v1/chat/completions", model="fake"
        ),
        fast_ranker_url="",
    )


def test_loop_done_path(config_dir, fake_backend, tmp_path, monkeypatch):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    config = _make_config(tmp_path)
    planner = _FakePlanner(
        [
            {
                "action": "click",
                "step_instruction": "open the article",
                "element_index": 1,
                "final_summary": "",
            },
            {
                "action": "done",
                "step_instruction": "finish",
                "final_summary": "opened Apple Silicon",
            },
        ]
    )
    trace = asyncio.run(
        loop_mod.run(
            config, "Google Chrome", "open article", max_steps=5, planner=planner
        )
    )
    assert trace["status"] == "done"
    assert trace["final_summary"] == "opened Apple Silicon"
    assert len(trace["steps"]) == 2


def test_loop_fixation_stall(config_dir, fake_backend, tmp_path, monkeypatch):
    """The 9B music failure mode: same instruction forever -> stalled, not hung."""
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    config = _make_config(tmp_path)
    fixated = {
        "action": "click",
        "step_instruction": "click search button",
        "element_index": 1,
        "final_summary": "",
    }
    planner = _FakePlanner([fixated], text_only=True)
    trace = asyncio.run(
        loop_mod.run(
            config, "Google Chrome", "play music", max_steps=12, planner=planner
        )
    )
    assert trace["status"] == "stalled"
    assert trace["stalled"] is True
    # two intervention cycles before giving up
    assert planner.calls <= 7


def test_loop_domain_guard(config_dir, fake_backend, tmp_path, monkeypatch):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    config = _make_config(tmp_path)
    config.allowed_domain = "example.com"
    trace = asyncio.run(loop_mod.run(config, "Google Chrome", "goal", max_steps=3))
    assert trace["status"] == "stopped"
    assert "domain guard" in trace.get("guard_stop", "")


def test_loop_writes_trace(config_dir, fake_backend, tmp_path, monkeypatch):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    config = _make_config(tmp_path)
    planner = _FakePlanner(
        [
            {
                "action": "done",
                "step_instruction": "finish",
                "final_summary": "done deal",
            }
        ]
    )
    trace = asyncio.run(
        loop_mod.run(config, "Google Chrome", "g", max_steps=2, planner=planner)
    )
    trace_path = Path(trace["run_dir"]) / "trace.json"
    assert trace_path.exists()
    saved = json.loads(trace_path.read_text())
    assert saved["status"] == "done"
