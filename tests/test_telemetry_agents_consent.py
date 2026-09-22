# SPDX-License-Identifier: Apache-2.0
"""End-to-end pins for agent setup and explicit consent telemetry."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import uuid
from dataclasses import replace
from datetime import datetime
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import rapid_mlx
import rapid_mlx.cli as cli
from rapid_mlx.agents import adapter as agent_adapter
from rapid_mlx.agents import get_profile, list_profiles, setup
from rapid_mlx.agents import telemetry as agent_telemetry
from rapid_mlx.agents.base import AgentConfigSpec
from rapid_mlx.telemetry import consent_runtime, emit, posthog_sender, state
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts
from rapid_mlx.telemetry.consent_decision import DISCLOSURE_REVISION

STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
INSTALL_ID = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
SESSION_ID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
FACTS = PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-ultra",
    memory_gb=64,
    python_version="3.11",
)


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def loopback_telemetry(tmp_path, monkeypatch):
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    telemetry_dir = tmp_path / ".rapid-mlx"
    telemetry_dir.mkdir()
    state.consent_path().write_text(
        "consent: true\n"
        "prompted_version: 0.15.1\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: INSTALL_ID)
    monkeypatch.setattr(emit, "session_id", lambda: SESSION_ID)
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: "7-29"
    )
    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)
    track_module._reset_for_tests()
    consent_runtime._reset_runtime_state_for_tests()
    posthog_sender._reset_for_tests()
    state.set_cli_kill_switch(False)

    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    monkeypatch.setenv(
        posthog_sender.POSTHOG_URL_ENV,
        f"http://{server.server_address[0]}:{server.server_port}/batch/",
    )

    def loopback_post(url: str, body: bytes, timeout: float) -> int:
        target = urlsplit(url)
        connection = HTTPConnection(target.hostname, target.port, timeout=timeout)
        try:
            connection.request(
                "POST",
                target.path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    sender = posthog_sender.PostHogSender(post=loopback_post)
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        posthog_sender.get_sender().flush(2.0)
        posthog_sender._reset_for_tests()
        track_module._reset_for_tests()
        consent_runtime._reset_runtime_state_for_tests()
        state.set_cli_kill_switch(False)
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def _items(server: HTTPServer) -> list[dict[str, object]]:
    return [
        item
        for body in server.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
    ]


def _event_view(item: dict[str, object]) -> dict[str, object]:
    assert set(item) == {"uuid", "event", "distinct_id", "timestamp", "properties"}
    assert str(uuid.UUID(str(item["uuid"]))) == item["uuid"]
    assert datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00"))
    assert item["distinct_id"] == INSTALL_ID
    properties = dict(item["properties"])  # type: ignore[arg-type]
    expected_common = {
        "app_version": "0.15.1",
        "surface": "cli",
        "os": "darwin",
        "os_version": "25.3",
        "arch": "arm64",
        "chip": "m3-ultra",
        "memory_gb": 64,
        "python_version": "3.11",
        "install_id": INSTALL_ID,
        "session_id": SESSION_ID,
        "channel": "stable",
        "days_since_first_run_bucket": "7-29",
        "$geoip_disable": True,
        "$process_person_profile": False,
    }
    assert {key: properties.pop(key) for key in expected_common} == expected_common
    return {"event": item["event"], "properties": properties}


def test_agent_setup_events_cover_success_and_all_failure_classes(
    loopback_telemetry, tmp_path, monkeypatch
):
    base_url = (
        f"http://{loopback_telemetry.server_address[0]}:"
        f"{loopback_telemetry.server_port}"
    )
    success_path = tmp_path / "success.json"
    success_plan = setup.SetupPlan(
        "continue",
        "Continue.dev",
        success_path,
        {},
        {"models": []},
        base_url,
        "model",
    )
    setup.apply_setup_plan(success_plan)
    agent_telemetry.track_agent_configured(success_plan.agent)

    with pytest.raises(ValueError, match="first-class safe setup flow"):
        setup.build_setup_plan("private-profile", base_url, "model")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        setup._load_yaml_mapping(invalid_yaml, "deepseek-harness")

    changed_path = tmp_path / "changed.json"
    changed_path.write_text("{}", encoding="utf-8")
    changed_plan = setup.SetupPlan(
        "claude-code",
        "Claude Code",
        changed_path,
        {},
        {"env": {}},
        base_url,
        "model",
    )
    changed_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after preview"):
        setup.apply_setup_plan(changed_plan)

    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text("token: changed\n", encoding="utf-8")
    credentials_plan = setup.SetupPlan(
        "deepseek-harness",
        "DeepSeek Harness",
        tmp_path / "settings.yaml",
        {},
        {},
        base_url,
        "model",
        "yaml",
        credentials_path,
        {},
        {},
    )
    with pytest.raises(RuntimeError, match="changed after preview"):
        setup.apply_setup_plan(credentials_plan)

    real_replace = os.replace

    def fail_target_replace(source, target):
        if Path(target) == tmp_path / "write-failed.yaml":
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_target_replace)
    write_failed_plan = setup.SetupPlan(
        "deepseek-harness",
        "DeepSeek Harness",
        tmp_path / "write-failed.yaml",
        {},
        {"value": 1},
        base_url,
        "model",
        "yaml",
    )
    with pytest.raises(OSError, match="disk full"):
        setup.apply_setup_plan(write_failed_plan)
    monkeypatch.setattr(os, "replace", real_replace)

    class Response:
        def __init__(self, status: int, body: bytes = b"") -> None:
            self.status = status
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return self.body

    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("refused")
        ),
    )
    with pytest.raises(RuntimeError, match="server is not ready"):
        setup.verify_server(base_url, "model", agent="continue")

    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(503),
    )
    with pytest.raises(RuntimeError, match="health returned HTTP 503"):
        setup.verify_server(base_url, "model", agent="continue")

    responses = iter(
        (
            Response(200),
            Response(200, b'{"data": []}'),
            Response(200),
            Response(200, b'{"data": [{"id": "a"}, {"id": "b"}]}'),
        )
    )
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *_a, **_k: next(responses)
    )
    with pytest.raises(RuntimeError, match="reported no models"):
        setup.verify_server(base_url, "model", agent="continue")
    with pytest.raises(RuntimeError, match="does not advertise model"):
        setup.verify_server(base_url, "missing", agent="continue")

    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "agent_configured", "properties": {"agent": "continue"}},
        {
            "event": "agent_configure_failed",
            "properties": {"agent": "other", "error_class": "no_safe_setup_flow"},
        },
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "deepseek-harness",
                "error_class": "config_invalid",
            },
        },
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "claude-code",
                "error_class": "config_changed",
            },
        },
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "deepseek-harness",
                "error_class": "config_changed",
            },
        },
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "deepseek-harness",
                "error_class": "config_write_failed",
            },
        },
        {
            "event": "agent_configure_failed",
            "properties": {"agent": "continue", "error_class": "server_not_ready"},
        },
        {
            "event": "agent_configure_failed",
            "properties": {"agent": "continue", "error_class": "server_not_ready"},
        },
        {
            "event": "agent_configure_failed",
            "properties": {"agent": "continue", "error_class": "server_no_models"},
        },
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "continue",
                "error_class": "model_not_advertised",
            },
        },
    ]
    wire = json.dumps(_items(loopback_telemetry), sort_keys=True)
    forbidden_values = [str(tmp_path), socket.gethostname(), "127.0.0.1"]
    username = os.environ.get("USER", "")
    if len(username) >= 6:
        forbidden_values.append(username)
    for forbidden in forbidden_values:
        if forbidden:
            assert forbidden not in wire


@pytest.mark.parametrize("agent_name", [profile.name for profile in list_profiles()])
def test_every_agent_profile_emits_one_configured_event(
    agent_name, loopback_telemetry, tmp_path, monkeypatch
):
    profile = get_profile(agent_name)
    assert profile is not None
    if agent_name in {"claude-code", "continue", "deepseek-harness"}:
        if agent_name == "claude-code":
            monkeypatch.setattr(
                setup.claude_code,
                "current_config_path",
                lambda: tmp_path / "claude-code.json",
            )
        elif agent_name == "continue":
            monkeypatch.setattr(
                setup.continue_dev,
                "current_config_path",
                lambda: tmp_path / "continue.json",
            )
        plan = setup.build_setup_plan(
            agent_name,
            "http://127.0.0.1:8000/v1",
            "test-model",
        )
        setup.apply_setup_plan(plan)
        agent_telemetry.track_agent_configured(plan.agent)
    else:
        summary = agent_adapter.setup_agent_config(
            profile,
            "http://127.0.0.1:8000/v1",
            "test-model",
        )
        assert not summary.startswith("Cannot")

    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "agent_configured", "properties": {"agent": agent_name}}
    ]


def test_env_profile_dry_run_emits_nothing(loopback_telemetry):
    profile = get_profile("aider")
    assert profile is not None
    agent_adapter.setup_agent_config(profile, dry_run=True)
    posthog_sender.get_sender().flush(2.0)
    assert _items(loopback_telemetry) == []


def test_adapter_second_identical_setup_emits_nothing(
    loopback_telemetry, tmp_path, monkeypatch
):
    profile = get_profile("opencode")
    assert profile is not None
    monkeypatch.setenv("OPENCODE_HOME", str(tmp_path))

    agent_adapter.setup_agent_config(profile, model_id="test-model")
    posthog_sender.get_sender().flush(2.0)
    assert len(_items(loopback_telemetry)) == 1

    agent_adapter.setup_agent_config(profile, model_id="test-model")
    posthog_sender.get_sender().flush(2.0)
    assert len(_items(loopback_telemetry)) == 1


def test_atomic_write_reports_changed_existing_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("before", encoding="utf-8")
    assert agent_adapter._atomic_write(path, "after") is True
    assert path.read_text(encoding="utf-8") == "after"


def test_dry_run_failure_emits_nothing(loopback_telemetry):
    def explode(*_args, **_kwargs):
        raise ValueError("broken profile")

    broken = SimpleNamespace(name="aider", render_config=explode)
    with pytest.raises(ValueError, match="broken profile"):
        agent_adapter.setup_agent_config(broken, dry_run=True)
    posthog_sender.get_sender().flush(2.0)
    assert _items(loopback_telemetry) == []


def test_saved_first_class_config_with_dead_server_emits_only_failure(
    loopback_telemetry, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        setup.continue_dev, "current_config_path", lambda: tmp_path / "continue.json"
    )
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("refused")
        ),
    )
    args = SimpleNamespace(
        agent_name="continue",
        base_url="http://127.0.0.1:1/v1",
        test=False,
        setup=True,
        model="test-model",
        agent_version=None,
        dry_run=False,
        yes=True,
        no_check=False,
    )

    with pytest.raises(SystemExit) as raised:
        cli.agents_command(args)
    assert raised.value.code == 1
    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "continue",
                "error_class": "server_not_ready",
            },
        }
    ]


def test_first_class_second_identical_setup_emits_nothing(
    loopback_telemetry, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        setup.continue_dev, "current_config_path", lambda: tmp_path / "continue.json"
    )
    args = SimpleNamespace(
        agent_name="continue",
        base_url="http://127.0.0.1:8000/v1",
        test=False,
        setup=True,
        model="test-model",
        agent_version=None,
        dry_run=False,
        yes=True,
        no_check=True,
    )

    cli.agents_command(args)
    posthog_sender.get_sender().flush(2.0)
    assert len(_items(loopback_telemetry)) == 1

    cli.agents_command(args)
    posthog_sender.get_sender().flush(2.0)
    assert len(_items(loopback_telemetry)) == 1


def test_first_class_dry_run_failure_emits_nothing(
    loopback_telemetry, tmp_path, monkeypatch
):
    (tmp_path / "settings.yaml").write_text("key: [unterminated\n", encoding="utf-8")
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    monkeypatch.setattr(agent_adapter, "fetch_context_window", lambda *_args: 32768)
    monkeypatch.setattr(agent_adapter, "fetch_reasoning_support", lambda *_args: True)
    args = SimpleNamespace(
        agent_name="deepseek-harness",
        base_url="http://127.0.0.1:8000/v1",
        test=False,
        setup=True,
        model="test-model",
        agent_version=None,
        dry_run=True,
        yes=True,
        no_check=True,
    )

    with pytest.raises(state.yaml.YAMLError):
        cli.agents_command(args)
    posthog_sender.get_sender().flush(2.0)
    assert _items(loopback_telemetry) == []


@pytest.mark.parametrize("agent_name", ["claude-code", "continue"])
def test_json_write_failure_has_agent_and_never_reports_success(
    agent_name, loopback_telemetry, tmp_path, monkeypatch
):
    plan = setup.SetupPlan(
        agent_name,
        agent_name,
        tmp_path / f"{agent_name}.json",
        {},
        {"configured": True},
        "http://127.0.0.1:8000/v1",
        "test-model",
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(setup.launch_common, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="disk full"):
        setup.apply_setup_plan(plan)
    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": agent_name,
                "error_class": "config_write_failed",
            },
        }
    ]


def test_non_oserror_write_failure_is_not_misclassified(
    loopback_telemetry, tmp_path, monkeypatch
):
    class FatalWrite(BaseException):
        pass

    plan = setup.SetupPlan(
        "continue",
        "Continue.dev",
        tmp_path / "continue.json",
        {},
        {"configured": True},
        "http://127.0.0.1:8000/v1",
        "test-model",
    )

    def fail_write(*_args, **_kwargs):
        raise FatalWrite

    monkeypatch.setattr(setup.launch_common, "atomic_write_json", fail_write)
    with pytest.raises(FatalWrite):
        setup.apply_setup_plan(plan)
    posthog_sender.get_sender().flush(2.0)
    assert _items(loopback_telemetry) == []


def test_adapter_failure_outcomes_use_closed_mapping(loopback_telemetry, monkeypatch):
    opencode = get_profile("opencode")
    codex = get_profile("codex")
    hermes = get_profile("hermes")
    assert opencode is not None and codex is not None and hermes is not None

    def explode(*_args, **_kwargs):
        raise ValueError("broken profile")

    broken_render = SimpleNamespace(name="aider", render_config=explode)
    with pytest.raises(ValueError, match="broken profile"):
        agent_adapter.setup_agent_config(broken_render)

    with monkeypatch.context() as scoped:
        scoped.setattr(agent_adapter, "_render_hermes_runtime_toolsets", explode)
        with pytest.raises(ValueError, match="broken profile"):
            agent_adapter.setup_agent_config(hermes)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            agent_adapter,
            "_resolve_config_path",
            lambda _cfg: (_ for _ in ()).throw(OSError("read-only directory")),
        )
        assert agent_adapter.setup_agent_config(opencode).startswith("Cannot prepare")

    with monkeypatch.context() as scoped:
        scoped.setattr(agent_adapter, "_resolve_config_path", explode)
        with pytest.raises(ValueError, match="broken profile"):
            agent_adapter.setup_agent_config(opencode)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            agent_adapter,
            "_atomic_write",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
        )
        assert agent_adapter.setup_agent_config(codex).startswith(
            "Cannot write Codex model catalog"
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            agent_adapter,
            "_merge_file_config",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("cannot read")),
        )
        assert agent_adapter.setup_agent_config(opencode).startswith(
            "Cannot read existing config"
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            agent_adapter,
            "_merge_file_config",
            lambda *_a, **_k: (_ for _ in ()).throw(
                agent_adapter._MergeParseError("invalid")
            ),
        )
        assert agent_adapter.setup_agent_config(opencode).startswith(
            "Cannot parse existing config"
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(agent_adapter, "_merge_file_config", lambda *_a, **_k: "{}")
        scoped.setattr(
            agent_adapter,
            "_atomic_write",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
        )
        assert agent_adapter.setup_agent_config(opencode).startswith(
            "Cannot write config"
        )

    no_path = replace(
        opencode,
        config=AgentConfigSpec(type="json", path=None, template="{}"),
    )
    assert agent_adapter.setup_agent_config(no_path) == (
        "No config to write (template not specified)"
    )

    posthog_sender.get_sender().flush(2.0)
    views = [_event_view(item) for item in _items(loopback_telemetry)]
    assert [view["properties"] for view in views] == [
        {"agent": "aider", "error_class": "other"},
        {"agent": "hermes", "error_class": "other"},
        {"agent": "opencode", "error_class": "config_write_failed"},
        {"agent": "opencode", "error_class": "other"},
        {"agent": "codex", "error_class": "config_write_failed"},
        {"agent": "opencode", "error_class": "other"},
        {"agent": "opencode", "error_class": "config_invalid"},
        {"agent": "opencode", "error_class": "config_write_failed"},
        {"agent": "opencode", "error_class": "other"},
    ]
    assert {view["event"] for view in views} == {"agent_configure_failed"}


def test_yaml_syntax_error_has_known_agent(loopback_telemetry, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("key: [unterminated\n", encoding="utf-8")
    with pytest.raises(Exception):
        setup._load_yaml_mapping(path, "deepseek-harness")
    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "deepseek-harness",
                "error_class": "config_invalid",
            },
        }
    ]


def test_apply_invalid_yaml_uses_plan_agent(loopback_telemetry, tmp_path):
    path = tmp_path / "broken-plan.yaml"
    path.write_text("key: [unterminated\n", encoding="utf-8")
    plan = setup.SetupPlan(
        "deepseek-harness",
        "DeepSeek Harness",
        path,
        {},
        {"configured": True},
        "http://127.0.0.1:8000/v1",
        "test-model",
        "yaml",
    )
    with pytest.raises(Exception):
        setup.apply_setup_plan(plan)
    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {
            "event": "agent_configure_failed",
            "properties": {
                "agent": "deepseek-harness",
                "error_class": "config_invalid",
            },
        }
    ]


def test_agent_telemetry_registry_failure_is_best_effort(monkeypatch):
    from rapid_mlx.telemetry import registry

    monkeypatch.setattr(
        registry,
        "load_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("missing registry")),
    )
    agent_telemetry.track_agent_configured("aider")
    agent_telemetry.track_agent_configure_failed("other", "aider")


def test_opt_out_flushes_before_write_and_reversed_order_loses_event(
    loopback_telemetry,
):
    cli.telemetry_command(SimpleNamespace(telemetry_action="disable"))
    assert state.get_consent_state().consent is False  # type: ignore[union-attr]
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "telemetry_opted_out", "properties": {"via": "cli"}}
    ]

    # The load-bearing control: mutating first closes the sender's live gate.
    track_module.track("telemetry_opted_out", {"via": "cli"})
    posthog_sender.get_sender().flush(2.0)
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "telemetry_opted_out", "properties": {"via": "cli"}}
    ]


def test_opt_in_writes_before_capture_and_consent_event_cap_is_five(
    loopback_telemetry,
):
    state.record_consent(False, rapid_mlx_version="0.15.1")
    actions = ("enable", "disable", "enable", "disable", "enable", "disable", "enable")
    for action in actions:
        cli.telemetry_command(SimpleNamespace(telemetry_action=action))
    posthog_sender.get_sender().flush(2.0)

    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "telemetry_opted_in", "properties": {"via": "cli"}},
        {"event": "telemetry_opted_out", "properties": {"via": "cli"}},
        {"event": "telemetry_opted_in", "properties": {"via": "cli"}},
        {"event": "telemetry_opted_out", "properties": {"via": "cli"}},
        {"event": "telemetry_opted_in", "properties": {"via": "cli"}},
    ]


def test_opt_in_delivers_required_notice_before_capture(loopback_telemetry, capfd):
    state.consent_path().write_text(
        "consent: true\nprompted_version: 0.15.1\n", encoding="utf-8"
    )
    consent_runtime._reset_runtime_state_for_tests()

    cli._track_telemetry_opted_in()
    posthog_sender.get_sender().flush(2.0)

    assert "anonymous usage reporting" in capfd.readouterr().err
    consent = state.yaml.safe_load(state.consent_path().read_text(encoding="utf-8"))
    assert consent["notice_revision_seen"] == DISCLOSURE_REVISION
    assert [_event_view(item) for item in _items(loopback_telemetry)] == [
        {"event": "telemetry_opted_in", "properties": {"via": "cli"}}
    ]


def test_opt_in_notice_delivery_failure_keeps_upload_latched(
    loopback_telemetry, monkeypatch
):
    state.consent_path().write_text(
        "consent: true\nprompted_version: 0.15.1\n", encoding="utf-8"
    )
    consent_runtime._reset_runtime_state_for_tests()
    monkeypatch.setattr(
        consent_runtime, "deliver_notice_if_needed", lambda _decision: False
    )

    cli._track_telemetry_opted_in()
    posthog_sender.get_sender().flush(2.0)

    assert consent_runtime.notice_was_delivered() is False
    assert _items(loopback_telemetry) == []


def test_real_cli_enable_and_disable_each_flush_exactly_one_event(
    loopback_telemetry, tmp_path
):
    site = tmp_path / "site"
    package = site / "rapid_mlx"
    shutil.copytree(
        Path(rapid_mlx.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (package / "telemetry" / "_release_stamp.json").write_text(
        json.dumps({"channel": "stable", "posthog_key": STAMP.posthog_key}),
        encoding="utf-8",
    )
    metadata = site / "rapid_mlx-0.15.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 0.15.1\n",
        encoding="utf-8",
    )
    endpoint = (
        f"http://{loopback_telemetry.server_address[0]}:"
        f"{loopback_telemetry.server_port}/batch/"
    )
    cli_code = "from rapid_mlx.cli import cli_entrypoint; cli_entrypoint()"

    for home_name, initial, action, expected, marker_present in (
        ("enable-noticed", False, "enable", "telemetry_opted_in", True),
        ("disable", True, "disable", "telemetry_opted_out", True),
        ("enable-needs-notice", False, "enable", "telemetry_opted_in", False),
    ):
        home = tmp_path / home_name
        consent_dir = home / ".rapid-mlx"
        consent_dir.mkdir(parents=True)
        (consent_dir / "telemetry-consent.yaml").write_text(
            f"consent: {str(initial).lower()}\n"
            "prompted_version: 0.15.1\n"
            + (
                f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
                if marker_present
                else ""
            ),
            encoding="utf-8",
        )
        before = len(_items(loopback_telemetry))
        env = os.environ.copy()
        for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
            env.pop(name, None)
        env.update(
            {
                "HOME": str(home),
                "USER": "rc",
                "PYTHONPATH": str(site),
                posthog_sender.POSTHOG_URL_ENV: endpoint,
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", cli_code, "telemetry", action],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        observed = _items(loopback_telemetry)[before:]
        assert [item["event"] for item in observed] == [expected]
        if action == "enable" and not marker_present:
            assert "anonymous usage reporting is ON" in result.stderr
            assert "anonymous usage reporting" not in result.stdout
            consent = state.yaml.safe_load(
                (consent_dir / "telemetry-consent.yaml").read_text(encoding="utf-8")
            )
            assert consent["notice_revision_seen"] == DISCLOSURE_REVISION


def test_reset_from_refusal_emits_no_consent_event(loopback_telemetry):
    state.record_consent(False, rapid_mlx_version="0.15.1")
    cli.telemetry_command(SimpleNamespace(telemetry_action="reset"))
    posthog_sender.get_sender().flush(2.0)
    assert _items(loopback_telemetry) == []


def test_consent_event_failures_never_break_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    def explode(*_args, **_kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)
    monkeypatch.setattr(track_module, "track", explode)
    cli._track_telemetry_opted_out()
    cli._track_telemetry_opted_in()


def test_enable_write_failure_emits_nothing(loopback_telemetry, monkeypatch):
    import rapid_mlx.telemetry as telemetry

    monkeypatch.setattr(
        telemetry,
        "record_consent",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")),
    )
    with pytest.raises(SystemExit) as raised:
        cli.telemetry_command(SimpleNamespace(telemetry_action="enable"))
    assert raised.value.code == 1
    assert _items(loopback_telemetry) == []


def test_refresh_decision_fail_closed_edges(monkeypatch):
    from rapid_mlx.telemetry.consent_decision import ProcessRole

    monkeypatch.setattr(consent_runtime, "_resolved_role", ProcessRole.DESKTOP)
    assert consent_runtime.refresh_decision().upload_now is False

    monkeypatch.setattr(consent_runtime, "_resolved_role", ProcessRole.HEADLESS_CLI)
    monkeypatch.setattr(consent_runtime, "read_stored_consent", lambda: None)
    assert consent_runtime.refresh_decision().upload_now is False


def test_confirm_plan_handles_eof(monkeypatch, tmp_path):
    plan = setup.SetupPlan(
        "continue",
        "Continue.dev",
        tmp_path / "continue.json",
        {},
        {},
        "http://127.0.0.1:8000/v1",
        "model",
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError)
    )
    assert setup.confirm_plan(plan) is False
