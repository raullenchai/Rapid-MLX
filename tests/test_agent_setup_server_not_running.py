# SPDX-License-Identifier: Apache-2.0
"""``agents <name> --setup`` before ``rapid-mlx serve`` is up.

A refused connection means no server is running yet — the normal first-run
order — so setup finishes with the one command that starts a matching server
instead of failing. Only a generic writer with no model to write still refuses.
"""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.agents import adapter, get_profile, setup
from rapid_mlx.agents import telemetry as agent_telemetry
from rapid_mlx.agents.server_hint import (
    ServerNotRunningError,
    is_connection_refused,
    is_local_base_url,
    start_server_command,
)

pytestmark = pytest.mark.usefixtures("agent_setup_host")


@pytest.fixture(autouse=True)
def short_readiness_wait(monkeypatch):
    real_wait = setup.wait_for_server

    def bounded(*args, **kwargs):
        return real_wait(*args, timeout=0.01, **kwargs)

    monkeypatch.setattr(setup, "wait_for_server", bounded)


@pytest.fixture
def dead_url():
    """HTTP is refused by agent_setup_host, regardless of real listeners."""
    port = 18731
    return f"http://127.0.0.1:{port}/v1", port


@pytest.fixture
def events(monkeypatch):
    recorded: list[tuple] = []

    def configured(agent):
        recorded.append(("configured", agent))

    def failed(error_class, agent, cause=None):
        recorded.append(("configure_failed", error_class, agent, cause))

    for module in (agent_telemetry, setup, adapter):
        if hasattr(module, "track_agent_configured"):
            monkeypatch.setattr(module, "track_agent_configured", configured)
        monkeypatch.setattr(module, "track_agent_configure_failed", failed)
    return recorded


def _args(agent, base_url, **overrides):
    values = dict(
        agent_name=agent,
        base_url=base_url,
        test=False,
        setup=True,
        model=None,
        agent_version=None,
        dry_run=False,
        yes=True,
        no_check=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ConnectionRefusedError(), True),
        (urllib.error.URLError(ConnectionRefusedError(61, "refused")), True),
        (urllib.error.URLError("refused"), False),
        (urllib.error.URLError(TimeoutError()), False),
        (TimeoutError(), False),
        (ValueError("bad json"), False),
    ],
)
def test_only_a_refused_connection_counts_as_not_running(exc, expected):
    assert is_connection_refused(exc) is expected


def test_start_command_uses_requested_model_and_pins_port():
    profile = get_profile("opencode")
    assert (
        start_server_command(profile, "http://localhost:8123/v1", "my-model")
        == "rapid-mlx serve my-model --port 8123"
    )


@pytest.mark.parametrize(
    "ram,cached,expected",
    [
        (512.0, False, "qwen3.6-35b-4bit"),
        (16.0, False, "qwen3.5-4b-4bit"),
        (64.0, True, "qwen3.5-9b-4bit"),
    ],
)
def test_start_command_falls_back_to_recommended_model_for_default(
    monkeypatch, ram, cached, expected
):
    # Recovery uses the first fitting cached model, then the first fitting model.
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: ram)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.is_repo_cached", lambda repo: cached and "9B" in repo
    )
    profile = get_profile("claude-code")
    assert start_server_command(profile, "http://127.0.0.1:8000/v1", "default") == (
        f"rapid-mlx serve {expected} --port 8000"
    )


def test_start_command_without_recommendation_uses_placeholder():
    profile = SimpleNamespace(recommended_models=[])
    assert (
        start_server_command(profile, "http://localhost:9000", "default")
        == "rapid-mlx serve <model> --port 9000"
    )


def test_start_command_binds_ipv6_loopback_explicitly():
    # serve binds 127.0.0.1 by default; a config pointing at [::1] needs
    # the server on the IPv6 loopback instead.
    profile = SimpleNamespace(recommended_models=[])
    assert (
        start_server_command(profile, "http://[::1]:8123/v1", "m")
        == "rapid-mlx serve m --port 8123 --host ::1"
    )


def test_start_command_quotes_model_and_survives_unparseable_url():
    profile = SimpleNamespace(recommended_models=[])
    assert start_server_command(profile, "ftp://x", "a b") == "rapid-mlx serve 'a b'"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000/v1", True),
        ("http://127.0.0.1:8000", True),
        ("http://[::1]:8000/v1", True),
        ("http://192.168.1.5:8000/v1", False),
        ("http://studio.local:8000/v1", False),
        ("ftp://x", False),
    ],
)
def test_only_local_urls_count_as_startable(url, expected):
    assert is_local_base_url(url) is expected


# --- verify_server ---------------------------------------------------------


def test_verify_server_refused_raises_untracked_not_running(dead_url, events):
    base_url, _ = dead_url
    with pytest.raises(ServerNotRunningError, match="no server is running"):
        setup.verify_server(base_url, "default", agent="continue", timeout=1.0)
    assert events == []


def test_verify_server_refused_remote_endpoint_is_still_a_failure(monkeypatch, events):
    def refused(*_args, **_kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(setup.urllib.request, "urlopen", refused)
    with pytest.raises(RuntimeError, match="server is not ready") as raised:
        setup.verify_server("http://192.168.1.5:8000/v1", "m", agent="continue")
    assert not isinstance(raised.value, ServerNotRunningError)
    assert events == [
        ("configure_failed", "server_not_ready", "continue", "remote_unreachable")
    ]


def test_verify_server_other_connection_failure_is_still_tracked(monkeypatch, events):
    def timeout(*_args, **_kwargs):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(setup.urllib.request, "urlopen", timeout)
    with pytest.raises(RuntimeError, match="server is not ready") as raised:
        setup.verify_server("http://127.0.0.1:8000/v1", "m", agent="continue")
    assert not isinstance(raised.value, ServerNotRunningError)
    assert events == [
        ("configure_failed", "server_not_ready", "continue", "startup_timeout")
    ]


# --- first-class flow ------------------------------------------------------


@pytest.fixture
def continue_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(setup.continue_dev, "current_config_path", lambda: path)
    return path


def test_first_class_setup_writes_offline(dead_url, events, continue_config, capsys):
    base_url, port = dead_url
    cli.agents_command(_args("continue", base_url, model="test-model"))
    output = capsys.readouterr().out
    assert continue_config.exists()
    assert f"--port {port}" in output
    assert "Connection check passed" not in output
    assert events == [("configured", "continue")]


def test_first_class_no_server_without_model_fails(
    dead_url, events, continue_config, capsys
):
    base_url, _ = dead_url
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(_args("continue", base_url))
    assert exc.value.code == 1
    assert not continue_config.exists()
    assert base_url in capsys.readouterr().out
    assert events == [
        ("configure_failed", "server_not_running", "continue", "connection_refused")
    ]


def test_first_class_setup_with_live_server_reports_check(
    monkeypatch, events, continue_config, capsys
):
    monkeypatch.setattr(setup, "verify_server", lambda *_a, **_k: "served-model")

    cli.agents_command(_args("continue", "http://127.0.0.1:8000/v1", model="m"))

    output = capsys.readouterr().out
    assert "Connection check passed (model: served-model)." in output
    assert "Start it with:" not in output
    assert events == [("configured", "continue")]


# --- generic flow ----------------------------------------------------------


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def test_generic_setup_with_model_writes_offline(dead_url, events, codex_home, capsys):
    base_url, port = dead_url
    cli.agents_command(_args("codex", base_url, model="local-qwen"))
    output = capsys.readouterr().out
    assert (codex_home / "config.toml").exists()
    assert base_url in (codex_home / "config.toml").read_text()
    assert f"--port {port}" in output
    assert "Connection check passed" not in output
    assert events == [("configured", "codex")]


def test_generic_setup_without_model_refuses_with_start_command(
    dead_url, events, codex_home, capsys
):
    base_url, port = dead_url

    with pytest.raises(SystemExit) as raised:
        cli.agents_command(_args("codex", base_url))

    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert not (codex_home / "config.toml").exists()
    assert f"No Rapid-MLX server on http://127.0.0.1:{port}" in output
    assert f"rapid-mlx agents codex --setup --yes --base-url {base_url}" in output
    assert "Traceback" not in output
    assert events == [
        ("configure_failed", "server_not_running", "codex", "connection_refused")
    ]


def test_generic_refusal_omits_default_base_url(
    monkeypatch, events, codex_home, capsys
):
    def refused(*_args, **_kwargs):
        raise ServerNotRunningError("no server is running at http://localhost:8000")

    monkeypatch.setattr(setup, "verify_server", refused)
    monkeypatch.setattr(adapter, "_detect_running_model", lambda _url: (None, None))
    with pytest.raises(SystemExit):
        cli.agents_command(_args("codex", "http://localhost:8000/v1"))
    output = capsys.readouterr().out
    assert "--setup --yes\n" in output
    assert "--base-url" not in output
