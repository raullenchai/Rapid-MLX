# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import socket
import sys
from types import SimpleNamespace

import pytest
import requests
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
    assert all("failure_stage" not in props for _, props in events)
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


def _hub_response(status_code: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://huggingface.co/owner/model"
    response.request = requests.Request("GET", response.url).prepare()
    return response


def test_render_hub_error_not_found_has_repo_discovery_next_steps():
    from huggingface_hub.errors import RepositoryNotFoundError

    failure = RepositoryNotFoundError("private raw detail", response=_hub_response(404))
    outer = RuntimeError("outer private detail")
    outer.__cause__ = failure

    rendered = cli.render_hub_error(outer, "owner/model")

    assert rendered is not None
    assert "owner/model" in rendered
    assert "rapid-mlx models" in rendered
    assert "mlx-community/Qwen3.5-9B-4bit" in rendered
    assert "private raw detail" not in rendered


@pytest.mark.parametrize("status_code", [401, 403])
def test_render_hub_error_gated_has_access_and_auth_next_steps(status_code):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    failure = (
        GatedRepoError("private raw detail", response=_hub_response(status_code))
        if status_code == 403
        else HfHubHTTPError("private raw detail", response=_hub_response(status_code))
    )

    rendered = cli.render_hub_error(failure, "owner/model")

    assert rendered is not None
    assert "https://huggingface.co/owner/model" in rendered
    assert "huggingface-cli login" in rendered
    assert "HF_TOKEN" in rendered
    assert "private raw detail" not in rendered


@pytest.mark.parametrize(
    "kind",
    ["local-entry", "offline-mode", "requests", "dns", "timeout"],
)
def test_render_hub_error_offline_has_network_cache_next_steps(kind):
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    failure = {
        "local-entry": LocalEntryNotFoundError("private cache"),
        "offline-mode": OfflineModeIsEnabled("private mode"),
        "requests": requests.ConnectionError("private host"),
        "dns": socket.gaierror("private dns"),
        "timeout": TimeoutError("private timeout"),
    }[kind]
    rendered = cli.render_hub_error(failure, "owner/model")

    assert rendered is not None
    assert "network" in rendered.lower()
    assert "HF_HUB_OFFLINE" in rendered
    assert "cached model" in rendered
    assert "private" not in rendered


def test_render_hub_error_ignores_context_and_unknown_errors():
    contextual = ValueError("legacy private detail")
    contextual.__context__ = requests.ConnectionError("ignored context")

    assert cli.render_hub_error(contextual, "owner/model") is None

    cyclic = RuntimeError("cycle")
    cyclic.__cause__ = cyclic
    assert cli.render_hub_error(cyclic, "owner/model") is None

    class ExplodingCauseError(RuntimeError):
        def __getattribute__(self, name):
            if name == "__cause__":
                raise KeyboardInterrupt
            return super().__getattribute__(name)

    assert cli.render_hub_error(ExplodingCauseError(), "owner/model") is None


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
    assert [
        (props["state"], props.get("failure_stage"))
        for name, props in events
        if name == "server_start_state"
    ] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_definitive_download_not_found_fails_resolve_with_next_steps(
    monkeypatch, capsys
):
    from huggingface_hub.errors import RepositoryNotFoundError

    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(
            sha="abc",
            siblings=[SimpleNamespace(size=1024, rfilename="weights.safetensors")],
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            RepositoryNotFoundError("private", response=_hub_response(404))
        ),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    assert caught.value.code == 1
    assert "rapid-mlx models" in capsys.readouterr().err
    assert [
        (props["state"], props.get("failure_stage"))
        for name, props in events
        if name == "server_start_state"
    ] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_gated_download_fails_fast_instead_of_printing_retry(monkeypatch, capsys):
    from huggingface_hub.errors import HfHubHTTPError

    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            HfHubHTTPError("private", response=_hub_response(403))
        ),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert "https://huggingface.co/owner/model" in captured.err
    assert "huggingface-cli login" in captured.err
    assert "server will retry" not in captured.out + captured.err


def test_gated_metadata_fails_fast_before_download(monkeypatch, capsys):
    from huggingface_hub.errors import GatedRepoError

    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            GatedRepoError("private", response=_hub_response(403))
        ),
    )
    downloaded = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: downloaded.append(True),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert downloaded == []
    assert "https://huggingface.co/owner/model" in captured.err
    assert "server will retry" not in captured.out + captured.err


def test_offline_metadata_warns_and_continues_to_download(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("private endpoint")
        ),
    )
    downloaded = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: downloaded.append(True) or "/tmp/fake",
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert downloaded == [True]
    assert "HF_HUB_OFFLINE" in captured.err
    assert "private endpoint" not in captured.out + captured.err


def test_offline_download_keeps_retry_path_with_next_steps(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("private endpoint")
        ),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert "HF_HUB_OFFLINE" in captured.err
    assert "server will retry" in captured.err
    assert "private endpoint" not in captured.out + captured.err


def test_unknown_download_error_keeps_legacy_message(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("legacy detail")),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Pre-download skipped (ValueError); server will retry." in captured.out


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
