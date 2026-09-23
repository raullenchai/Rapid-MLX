# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx import cli, server
from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle
from rapid_mlx.service import helpers
from rapid_mlx.telemetry import registry, server_start


@pytest.fixture(autouse=True)
def _reset_state():
    server_start._reset_for_tests()
    yield
    server_start._reset_for_tests()


def _capture(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props: events.append((event, dict(props))),
    )
    return events


def test_attempted_then_ready_exactly_once(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.attempted("ignored", load_policy="lazy")
    server_start.ready()
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "ready"]
    assert all(name == "server_start_state" for name, _ in events)
    assert events[0][1] == {
        "state": "attempted",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert events[1][1] == {
        "state": "ready",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert all(registry.validate(name, props) == props for name, props in events)


@pytest.mark.parametrize(
    "stage",
    ["resolve", "download", "preflight", "prepare", "engine_start", "bind"],
)
def test_each_failure_stage_is_the_only_terminal(monkeypatch, stage):
    events = _capture(monkeypatch)
    server_start.attempted("flux-schnell", load_policy="lazy")
    server_start.failed(stage)
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == stage
    assert "failure_stage" not in events[0][1]


def test_invalid_values_are_omitted_or_ignored(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted(SimpleNamespace(), load_policy="surprise")
    server_start.failed("surprise")
    server_start.ready()

    assert events == [
        (
            "server_start_state",
            {"state": "attempted", "model_type": "other"},
        ),
        (
            "server_start_state",
            {"state": "ready", "model_type": "other"},
        ),
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3.5-4b-4bit", "eager"),
        ("flux-schnell", "lazy"),
        ("ltx-2.3-mlx-q4", "lazy"),
        ("kokoro", "lazy"),
    ],
)
def test_load_policy_matches_adapter_lanes(model, expected):
    assert server_start.load_policy(model) == expected


def test_explicit_lazy_load_overrides_adapter_policy():
    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"
    assert server_start.load_policy("qwen3.5-4b-4bit", lazy_load=True) == "lazy"


@pytest.mark.asyncio
async def test_post_ready_lazy_503_does_no_server_start_telemetry_work(monkeypatch):
    events = _capture(monkeypatch)

    class FailingLazyEngine:
        _loaded = False

        async def start(self):
            raise RuntimeError("lazy load failed")

        async def stop(self):
            self._loaded = False

    engine = FailingLazyEngine()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    monkeypatch.setattr(
        helpers,
        "get_config",
        lambda: SimpleNamespace(primary_model_lifecycle=lifecycle),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="lazy")
    server_start.ready()

    imports: list[str] = []
    real_import = builtins.__import__

    def reject_server_start_import(name, *args, **kwargs):
        if name == "rapid_mlx.telemetry.server_start":
            imports.append(name)
            raise AssertionError("request path imported server_start telemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rapid_mlx.telemetry.server_start")
    monkeypatch.setattr(builtins, "__import__", reject_server_start_import)

    with pytest.raises(HTTPException) as caught:
        await helpers.ensure_engine_ready(engine)

    assert caught.value.status_code == 503
    assert imports == []
    assert [props["state"] for _, props in events] == ["attempted", "ready"]


def test_emitter_base_exception_cannot_change_host_result(monkeypatch):
    def explode(*_args, **_kwargs):
        raise SystemExit(91)

    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr("rapid_mlx.telemetry.track.track", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.failed("preflight")


def test_disabled_telemetry_never_initializes_sender(monkeypatch):
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)

    def explode():
        raise AssertionError("disabled telemetry initialized the sender")

    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.ready()


def test_attempt_setup_failure_is_inert(monkeypatch):
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.install_atexit",
        lambda: (_ for _ in ()).throw(RuntimeError("sender unavailable")),
    )

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")


def test_load_policy_falls_back_to_eager_when_classification_fails(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.model_type",
        lambda _model: (_ for _ in ()).throw(RuntimeError("classifier unavailable")),
    )

    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"


def test_cli_main_emits_attempted_before_serve_preflight(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import consent_runtime

    model = tmp_path / "local-model"
    model.mkdir()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(consent_runtime, "startup", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda command: None)
    monkeypatch.setattr(
        server_start,
        "load_policy",
        lambda selected, *, lazy_load: calls.append(("policy", selected)) or "eager",
    )
    monkeypatch.setattr(
        server_start,
        "attempted",
        lambda selected, *, load_policy: calls.append(("attempted", load_policy)),
    )
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: calls.append(("serve", args.model)),
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(model)])

    cli.main()

    assert calls == [
        ("policy", str(model)),
        ("attempted", "eager"),
        ("serve", str(model)),
    ]


def test_failure_context_preserves_original_exception(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    with (
        pytest.raises(SystemExit) as caught,
        server_start.failure_stage("preflight"),
    ):
        raise SystemExit(2)

    assert caught.value.code == 2
    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == "preflight"


def _stub_download_entry(monkeypatch):
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_pull_failed",
        lambda *_a, **_kw: None,
    )


def test_resolve_timeout_emits_resolve_before_preserving_exit(monkeypatch):
    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError()),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    assert caught.value.code == 1
    assert [(props["state"], props.get("failure_stage")) for _, props in events] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_definitive_download_404_emits_download(monkeypatch):
    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("404 missing")),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(RuntimeError, match="not found on HuggingFace"):
        cli._ensure_model_downloaded("owner/model")

    assert [(props["state"], props.get("failure_stage")) for _, props in events] == [
        ("attempted", None),
        ("failed", "download"),
    ]


@pytest.mark.parametrize(
    "decorator", [cli._capture_start_failures, server._capture_start_failures]
)
def test_entrypoint_guard_never_replaces_host_exception(monkeypatch, decorator):
    monkeypatch.setattr(
        server_start,
        "fail_current",
        lambda: (_ for _ in ()).throw(SystemExit(91)),
    )

    @decorator
    def fail_host():
        raise ValueError("host failure")

    with pytest.raises(ValueError, match="host failure"):
        fail_host()
