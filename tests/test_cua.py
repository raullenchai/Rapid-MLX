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


def test_press_rejects_unknown_index():
    with pytest.raises(ValueError, match="unknown element_index"):
        validate_plan(
            {
                "action": "press",
                "step_instruction": "submit",
                "element_index": 99,
                "key": "Enter",
            },
            valid_indexes={1, 2},
        )


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
    import asyncio
    import threading

    marker = "APPROVE_SIGNIN"

    def approve_later():
        import time

        time.sleep(0.5)
        (tmp_path / marker).write_text("")

    threading.Thread(target=approve_later, daemon=True).start()
    assert asyncio.run(gates.wait_for_human(tmp_path, marker, timeout=5.0))


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


def test_loop_honors_a_preexisting_stop_request(
    config_dir, fake_backend, tmp_path, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    planner = _FakePlanner(
        [{"action": "done", "step_instruction": "finish", "final_summary": "done"}]
    )
    events: list[dict] = []

    async def scenario():
        stop_event = asyncio.Event()
        stop_event.set()
        return await loop_mod.run(
            _make_config(tmp_path),
            "Google Chrome",
            "goal",
            max_steps=2,
            planner=planner,
            event_sink=events.append,
            stop_event=stop_event,
        )

    trace = asyncio.run(scenario())
    assert trace["status"] == "stopped"
    assert trace["final_summary"] == "cancelled by client"
    assert planner.calls == 0
    assert [event["kind"] for event in events].count("terminal") == 1


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


def test_loop_empty_ax_tree_stops_honestly(
    config_dir, fake_backend, tmp_path, monkeypatch
):
    """Regression (2026-09-27 dogfood): Chrome's AX service wedged mid-run and
    get_app_state returned empty snapshots. The planner then guessed index 0
    and the ValueError crashed the run as "incomplete". The loop must stop
    with a readable reason after two empty snapshots instead."""
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    def empty_state(app, screenshot=True, use_cache=True):
        return {"app": {"name": app}, "elements": [], "tree_text": ""}

    monkeypatch.setattr(loop_mod.backend, "get_app_state", empty_state)
    monkeypatch.setattr(loop_mod.backend, "read_url", lambda app: "")
    config = _make_config(tmp_path)
    planner = _FakePlanner(
        [{"action": "done", "step_instruction": "x", "final_summary": "y"}],
        text_only=True,
    )
    events: list[dict] = []
    trace = asyncio.run(
        loop_mod.run(
            config,
            "Google Chrome",
            "goal",
            max_steps=5,
            planner=planner,
            event_sink=events.append,
        )
    )
    assert trace["status"] == "stopped"
    assert "accessibility tree" in (trace.get("final_summary") or "")
    assert planner.calls == 0  # planner never sees an empty snapshot
    kinds = [e["kind"] for e in events]
    assert kinds.count("executed") == 2  # one observe per empty snapshot
    assert kinds[-1] == "terminal"


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


def test_commerce_click_rejected():
    """Adversarial: clicking 'Add to cart' must trip the consent gate too."""
    with pytest.raises(ConsentError, match="cart"):
        gates.check_plan_consents(
            {"action": "click", "step_instruction": "add the item"},
            target_label="Add to cart",
        )


def test_commerce_press_and_unspaced_chinese_rejected():
    with pytest.raises(ConsentError, match="cart"):
        gates.check_plan_consents(
            {"action": "press", "step_instruction": "submit", "key": "Enter"},
            target_label="Place order",
        )
    with pytest.raises(ConsentError, match="cart"):
        gates.check_plan_consents(
            {"action": "click", "step_instruction": "立即购买商品"},
            target_label="继续",
        )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/page", None),
        ("https://docs.example.com/page", None),
        ("https://example.com.evil.test/", "outside"),
        ("https://evil.test/?next=example.com", "outside"),
        ("", "could not be read"),
    ],
)
def test_domain_guard_matches_hostname_boundary(tmp_path, url, expected):
    from rapid_mlx.cua.loop import CUARun

    config = _make_config(tmp_path)
    config.allowed_domain = "example.com"
    guard = CUARun(config, "Chrome", "g", tmp_path / "run")._check_domain(url)
    if expected is None:
        assert guard is None
    else:
        assert expected in guard


def test_wait_uses_async_sleep(config_dir, fake_backend, tmp_path, monkeypatch):
    """The loop must not block the event loop during wait/backoff."""
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    slept = []

    class _FakeTime:
        @staticmethod
        async def sleep(seconds):
            slept.append(seconds)

    monkeypatch.setattr(loop_mod.asyncio, "sleep", _FakeTime.sleep)
    config = _make_config(tmp_path)
    planner = _FakePlanner(
        [
            {
                "action": "wait",
                "step_instruction": "let the page settle",
                "final_summary": "",
            },
            {
                "action": "done",
                "step_instruction": "finish",
                "final_summary": "waited then finished",
            },
        ]
    )
    trace = asyncio.run(
        loop_mod.run(config, "Google Chrome", "g", max_steps=3, planner=planner)
    )
    assert trace["status"] == "done"
    assert 2.0 in slept and 1.2 in slept


def test_max_steps_returns_stalled_instead_of_crashing(
    config_dir, fake_backend, tmp_path, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    planner = _FakePlanner(
        [
            {
                "action": "wait",
                "step_instruction": "keep waiting",
                "final_summary": "",
            }
        ]
    )
    events: list[dict] = []
    trace = asyncio.run(
        loop_mod.run(
            _make_config(tmp_path),
            "Chrome",
            "g",
            max_steps=1,
            planner=planner,
            event_sink=events.append,
        )
    )
    assert trace["status"] == "stalled"
    assert trace["max_steps_reached"] == 1
    assert events[-1]["kind"] == "terminal"
    assert events[-1]["status"] == "stalled"


def test_loop_reports_invalid_plan_and_runtime_failures(
    config_dir, fake_backend, tmp_path, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)

    class FailingPlanner:
        text_only = True

        def __init__(self, error):
            self.error = error
            self.closed = False

        async def plan(self, *args, **kwargs):
            raise self.error

        async def close(self):
            self.closed = True

    invalid = FailingPlanner(ValueError("bad element"))
    invalid_events: list[dict] = []
    trace = asyncio.run(
        loop_mod.run(
            _make_config(tmp_path),
            "Chrome",
            "g",
            planner=invalid,
            event_sink=invalid_events.append,
        )
    )
    assert trace["status"] == "stopped"
    assert "bad element" in trace["final_summary"]
    assert invalid_events[-1]["status"] == "stopped"
    assert invalid.closed is True

    runtime = FailingPlanner(RuntimeError("planner offline"))
    runtime_events: list[dict] = []
    with pytest.raises(RuntimeError, match="planner offline"):
        asyncio.run(
            loop_mod.run(
                _make_config(tmp_path),
                "Chrome",
                "g",
                planner=runtime,
                event_sink=runtime_events.append,
            )
        )
    assert runtime_events[-1]["status"] == "failed"
    assert runtime.closed is True


def test_loop_cancellation_and_event_sink_fail_closed(
    config_dir, fake_backend, tmp_path, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)

    class BlockingPlanner:
        text_only = True

        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def plan(self, *args, **kwargs):
            self.started.set()
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    async def scenario():
        planner = BlockingPlanner()
        events: list[dict] = []
        task = asyncio.create_task(
            loop_mod.run(
                _make_config(tmp_path),
                "Chrome",
                "g",
                planner=planner,
                event_sink=events.append,
            )
        )
        await planner.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return planner, events

    planner, events = asyncio.run(scenario())
    assert planner.closed is True
    assert events[-1]["status"] == "stopped"

    runner = loop_mod.CUARun(
        _make_config(tmp_path),
        "Chrome",
        "g",
        tmp_path / "bad-sink",
        event_sink=lambda _event: (_ for _ in ()).throw(RuntimeError("sink")),
    )
    runner._emit({"kind": "progress"})

    async def broken_gate(_reason):
        raise RuntimeError("gate unavailable")

    runner.gate = broken_gate
    assert asyncio.run(runner._request_signin_approval()) is False


# ---------------------------------------------------------------- cli dispatch


def test_cli_planners_and_config(capsys, config_dir):
    from rapid_mlx.cua.cli import main

    assert main(["planners"]) == 0
    out = capsys.readouterr().out
    assert "local-9b" in out and "cloud-glm" in out

    assert main(["config", "--show"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "presets" in data

    assert (
        main(["config", "--set", "presets.local-9b.url", "http://127.0.0.1:9/v1"]) == 0
    )
    assert main(["config", "--set", "bogus.key", "x"]) == 2


def test_cli_run_dispatch_and_planner_error(capsys, config_dir, monkeypatch, tmp_path):
    import rapid_mlx.cua.cli as cli_mod
    import rapid_mlx.cua.loop as loop_mod

    seen = {}

    async def fake_run(config, app, goal, open_url="", max_steps=None, planner=None):
        seen["planner"] = config.planner.preset
        seen["goal"] = goal
        return {"status": "done", "final_summary": "ok"}

    monkeypatch.setattr(loop_mod, "run", fake_run)
    rc = cli_mod.main(
        [
            "run",
            "--app",
            "Chrome",
            "--goal",
            "g",
            "--planner",
            "local-9b",
            "--max-steps",
            "4",
        ]
    )
    assert rc == 0
    assert seen["planner"] == "local-9b" and seen["goal"] == "g"

    rc = cli_mod.main(["run", "--app", "Chrome", "--goal", "g", "--planner", "nope"])
    assert rc == 2


# ---------------------------------------------------------------- planner client


class _FakeResponse:
    def __init__(self, content=None, status_error=False):
        self._content = content
        self.is_error = status_error
        self.status_code = 500 if status_error else 200
        self.text = "server exploded" if status_error else ""

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _make_planner(monkeypatch, responses):
    from rapid_mlx.cua import planner as planner_mod

    p = planner_mod.Planner(
        url="http://127.0.0.1:9/v1/chat/completions", model="m", text_only=True
    )
    queue = list(responses)

    async def fake_post(url, json=None):
        return _FakeResponse(queue.pop(0))

    monkeypatch.setattr(p.client, "post", fake_post)
    return p


def test_plan_repairs_invalid_then_accepts(monkeypatch, fake_backend):
    import asyncio

    snapshot = fake_backend.get_app_state("Chrome", screenshot=False)
    planner = _make_planner(
        monkeypatch,
        [
            '{"action":"click","step_instruction":"x","element_index":999,"text":"","key":"","direction":"","final_summary":""}',
            '{"action":"click","step_instruction":"x","element_index":1,"text":"","key":"","direction":"","final_summary":""}',
        ],
    )
    plan, raw, latency, attempts = asyncio.run(
        planner.plan("g", snapshot, [], allowed_domain="", progress_hint="")
    )
    assert plan["element_index"] == 1
    assert len(attempts) == 2 and attempts[0]["error"]


def test_plan_http_error_raises(monkeypatch, fake_backend):
    import asyncio

    snapshot = fake_backend.get_app_state("Chrome", screenshot=False)
    planner = _make_planner(monkeypatch, [])

    async def fake_post(url, json=None):
        return _FakeResponse(status_error=True)

    monkeypatch.setattr(planner.client, "post", fake_post)
    with pytest.raises(RuntimeError, match="planner HTTP 500"):
        asyncio.run(planner.plan("g", snapshot, []))


def test_plan_attaches_screenshot_for_vision(monkeypatch, fake_backend):
    import asyncio
    import io as _io

    from PIL import Image

    from rapid_mlx.cua import planner as planner_mod

    buffer = _io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    snapshot = fake_backend.get_app_state("Chrome", screenshot=False)
    snapshot["screenshot_png"] = buffer.getvalue()

    seen = {}
    planner = planner_mod.Planner(
        url="http://127.0.0.1:9/v1/chat/completions", model="m", text_only=False
    )

    async def fake_post(url, json=None):
        seen["content_kinds"] = [c["type"] for c in json["messages"][0]["content"]]
        return _FakeResponse(
            '{"action":"wait","step_instruction":"s","element_index":-1,"text":"","key":"","direction":"","final_summary":""}'
        )

    monkeypatch.setattr(planner.client, "post", fake_post)
    asyncio.run(planner.plan("g", snapshot, []))
    assert seen["content_kinds"] == ["text", "image_url"]


def test_reflect_text_only(monkeypatch):
    import asyncio

    planner = _make_planner(
        monkeypatch,
        [
            '{"outcome":"no_effect","evidence":"nothing changed","recommended_recovery":"retry"}'
        ],
    )
    verdict, latency = asyncio.run(
        planner.reflect("g", "step", "u1", "u1", {"tree_changed": False})
    )
    assert verdict["outcome"] == "no_effect"


def test_planner_rejects_non_loopback():
    from rapid_mlx.cua.planner import Planner

    with pytest.raises(ValueError, match="loopback"):
        Planner(url="http://10.0.0.5:8888/v1", model="m")


def test_fast_ranker_rejects_non_loopback():
    from rapid_mlx.cua.fast import FastOutcomeRanker

    with pytest.raises(ValueError, match="literal IP"):
        FastOutcomeRanker("https://ranker.example/v1/rank")


def test_data_url_roundtrip():
    import base64
    import io as _io

    from PIL import Image

    from rapid_mlx.cua.planner import data_url

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    url = data_url(buffer.getvalue())
    assert url.startswith("data:image/png;base64,")
    image = Image.open(_io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (4, 4)


def test_config_recovers_from_invalid_json(config_dir):
    path = config_dir / "cua-config.json"
    path.write_text("{broken")
    assert "cloud-glm" in load_config()["presets"]


def test_fast_ranker_success_error_and_assess(monkeypatch):
    import asyncio

    from rapid_mlx.cua.fast import FastOutcomeRanker

    class Response:
        is_error = False
        status_code = 200
        text = ""

        def json(self):
            return {
                "ranked": [
                    {
                        "candidate": "The requested computer action succeeded.",
                        "prob": "0.9",
                    }
                ]
            }

    ranker = FastOutcomeRanker("http://127.0.0.1:9/v1/rank")

    async def post(*args, **kwargs):
        return Response()

    monkeypatch.setattr(ranker.client, "post", post)
    verdict, latency = asyncio.run(ranker.assess("goal", {"action": "click"}, {}))
    assert verdict == {
        "outcome": "success",
        "confidence": 0.9,
        "source": "system-one-rank",
    }
    assert latency >= 0

    Response.is_error = True
    Response.status_code = 503
    Response.text = "down"
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(ranker.rank("context", ["answer"]))
    asyncio.run(ranker.close())


def test_tracker_empty_instruction_and_empty_snapshot_hint():
    tracker = NoProgressTracker(stall_limit=1)
    tracker.record({"step_instruction": ""}, "success")
    assert not tracker.should_intervene()
    tracker.record({"step_instruction": ""}, "uncertain")
    assert tracker.should_intervene()
    hint = tracker.take_hint({})
    assert "(empty snapshot)" in hint
    assert "consecutive" in hint


def test_human_gate_timeout(tmp_path, monkeypatch):
    import asyncio

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(gates.asyncio, "sleep", no_sleep)
    assert not asyncio.run(gates.wait_for_human(tmp_path, "NEVER", timeout=-1.0))


def test_execute_all_action_variants(fake_backend, tmp_path, monkeypatch):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    monkeypatch.setattr(
        fake_backend,
        "scroll",
        lambda app, direction, pages: {"ok": True, "direction": direction},
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(loop_mod.asyncio, "sleep", no_sleep)
    runner = loop_mod.CUARun(_make_config(tmp_path), "Chrome", "g", tmp_path / "run")
    snapshot = {"elements": []}
    fill = asyncio.run(
        runner._execute(
            {"action": "fill", "element_index": 1, "text": "hello"}, snapshot
        )
    )
    press = asyncio.run(
        runner._execute(
            {"action": "press", "element_index": 1, "key": "Enter"}, snapshot
        )
    )
    scroll = asyncio.run(
        runner._execute({"action": "scroll", "direction": "up"}, snapshot)
    )
    assert fill["verified"] and press["key"] == "Enter"
    assert scroll["direction"] == "up"


def test_loop_invalid_domain_consent_and_human_timeout(
    fake_backend, tmp_path, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    config = _make_config(tmp_path)
    config.allowed_domain = "example.com"
    runner = loop_mod.CUARun(config, "Chrome", "g", tmp_path / "invalid-domain")
    assert "valid HTTP" in runner._check_domain("file:///tmp/page")

    config.allowed_domain = ""
    commerce = _FakePlanner(
        [
            {
                "action": "click",
                "step_instruction": "checkout now",
                "element_index": 1,
                "final_summary": "",
            }
        ]
    )
    result = asyncio.run(runner.step(commerce, 1))
    assert result["status"] == "stopped" and "cart" in result["reason"]

    config.human_login = True
    runner = loop_mod.CUARun(config, "Chrome", "g", tmp_path / "human")
    normal = _FakePlanner(
        [
            {
                "action": "click",
                "step_instruction": "continue",
                "element_index": 1,
                "final_summary": "",
            }
        ]
    )
    monkeypatch.setattr(loop_mod.gates, "looks_like_sign_in", lambda _: True)

    async def timeout(*args):
        return False

    monkeypatch.setattr(loop_mod.gates, "wait_for_human", timeout)
    result = asyncio.run(runner.step(normal, 1))
    assert result["status"] == "stopped" and "not approved" in result["reason"]


def test_loop_ranker_success_and_unavailable(fake_backend, tmp_path, monkeypatch):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(loop_mod.asyncio, "sleep", no_sleep)
    runner = loop_mod.CUARun(_make_config(tmp_path), "Chrome", "g", tmp_path / "rank")

    class Ranker:
        async def assess(self, *args):
            return {"outcome": "success", "confidence": 1.0}, 0.01

    runner.ranker = Ranker()
    planner = _FakePlanner(
        [
            {
                "action": "click",
                "step_instruction": "open",
                "element_index": 1,
                "final_summary": "",
            }
        ]
    )
    assert asyncio.run(runner.step(planner, 1)) is None
    assert runner.trace["steps"][-1]["protocol_outcome"] == "success"

    class BrokenRanker:
        async def assess(self, *args):
            raise RuntimeError("offline")

    runner.ranker = BrokenRanker()
    assert asyncio.run(runner.step(planner, 2)) is None
    assert runner.trace["steps"][-1]["state_delta"]["fast_outcome"] == {
        "outcome": "unavailable"
    }


def test_run_constructs_clients_opens_url_and_closes(
    config_dir, fake_backend, monkeypatch
):
    import asyncio

    from rapid_mlx.cua import loop as loop_mod

    monkeypatch.setattr(loop_mod, "backend", fake_backend)
    opened = []
    system_opened = []
    monkeypatch.setattr(
        loop_mod.subprocess,
        "run",
        lambda command, **kwargs: system_opened.append((command, kwargs)),
    )
    loop_mod._open_url("Safari", "https://example.com")
    assert system_opened[0][0] == ["open", "-a", "Safari", "https://example.com"]
    monkeypatch.setattr(
        loop_mod, "_open_url", lambda app, url: opened.append((app, url))
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(loop_mod.asyncio, "sleep", no_sleep)
    closed = []

    class PlannerFactory(_FakePlanner):
        def __init__(self, **kwargs):
            super().__init__(
                [
                    {
                        "action": "done",
                        "step_instruction": "finish",
                        "final_summary": "ok",
                    }
                ],
                text_only=kwargs["text_only"],
            )

        async def close(self):
            closed.append("planner")

    class RankerFactory:
        def __init__(self, _url):
            pass

        async def close(self):
            closed.append("ranker")

    monkeypatch.setattr(loop_mod, "Planner", PlannerFactory)
    monkeypatch.setattr(loop_mod, "FastOutcomeRanker", RankerFactory)
    config = _make_config(config_dir)
    config.fast_ranker_url = "http://127.0.0.1:9/v1/rank"
    trace = asyncio.run(
        loop_mod.run(config, "Chrome", "g", open_url="https://example.com")
    )
    assert trace["status"] == "done"
    assert opened == [("Chrome", "https://example.com")]
    assert closed == ["planner", "ranker"]


def test_cli_remaining_dispatch_paths(capsys, config_dir, monkeypatch):
    import runpy
    import sys
    import types

    import rapid_mlx.cua.cli as cli_mod
    import rapid_mlx.cua.loop as loop_mod

    assert (
        cli_mod.main(["config", "--set", "fast_ranker_url", "http://127.0.0.1:8"]) == 0
    )
    assert cli_mod.main(["config"]) == 0
    capsys.readouterr()

    async def incomplete(*args, **kwargs):
        return {"status": "stopped", "guard_stop": "outside domain"}

    monkeypatch.setattr(loop_mod, "run", incomplete)
    rc = cli_mod.main(
        [
            "run",
            "--app",
            "Chrome",
            "--goal",
            "g",
            "--planner",
            "local-9b",
            "--planner-vision",
            "--no-fast-ranker",
        ]
    )
    assert rc == 1 and "NOT DONE" in capsys.readouterr().out

    original = cli_mod.build_parser
    parser = types.SimpleNamespace(
        parse_args=lambda argv: types.SimpleNamespace(cua_command="unknown"),
        print_help=lambda: None,
    )
    monkeypatch.setattr(cli_mod, "build_parser", lambda: parser)
    assert cli_mod.main([]) == 2
    monkeypatch.setattr(cli_mod, "build_parser", original)

    monkeypatch.setattr(sys, "argv", ["rapid_mlx.cua", "planners"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("rapid_mlx.cua.__main__", run_name="__main__")
    assert exc.value.code == 0
    capsys.readouterr()

    monkeypatch.delitem(sys.modules, "rapid_mlx.cua.cli", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("rapid_mlx.cua.cli", run_name="__main__")
    assert exc.value.code == 0
    capsys.readouterr()

    import rapid_mlx.cli as root_cli

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "cua", "planners"])
    assert root_cli.main() == 0
    assert "local-9b" in capsys.readouterr().out


def test_planner_validation_and_helpers(monkeypatch):
    import asyncio

    from rapid_mlx.cua import planner as planner_mod

    assert planner_mod.extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(ValueError, match="no JSON"):
        planner_mod.extract_json("nothing")
    with pytest.raises(ValueError, match="JSON object"):
        planner_mod.extract_json("[1]")
    with monkeypatch.context() as patch:
        patch.setattr(planner_mod.json, "loads", lambda _text: [])
        with pytest.raises(ValueError, match="JSON object"):
            planner_mod.extract_json("{}")
    with pytest.raises(ValueError, match="unsupported action"):
        validate_plan({"action": "launch"})
    with pytest.raises(ValueError, match="integer"):
        validate_plan({"action": "click", "element_index": None})
    with pytest.raises(ValueError, match="requires text"):
        validate_plan({"action": "fill", "element_index": 1, "text": ""})
    assert validate_plan({"action": "scroll", "direction": "up"})["direction"] == "up"
    assert validate_plan({"action": "wait", "direction": "up"})["direction"] == ""
    with pytest.raises(ValueError, match="HTTP"):
        planner_mod.assert_loopback_url("file:///tmp/socket")
    with pytest.raises(ValueError, match="literal IP"):
        planner_mod.assert_loopback_url("http://localhost:9/v1")

    planner = planner_mod.Planner(
        url="http://127.0.0.1:9/v1", model="m", reasoning_effort="low"
    )
    seen = {}

    async def post(url, json=None):
        seen.update(json)
        return _FakeResponse('{"ok":true}')

    monkeypatch.setattr(planner.client, "post", post)
    assert asyncio.run(planner._ask([], 5, {}, "test")) == '{"ok":true}'
    assert seen["reasoning_effort"] == "low"
    assert asyncio.run(planner_mod.ask_with_timeout(asyncio.sleep(0, result=7), 1)) == 7
    asyncio.run(planner.close())
