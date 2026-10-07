# SPDX-License-Identifier: Apache-2.0
"""``agents <name> --setup`` before ``rapid-mlx serve`` is up.

A refused connection means no server is running yet — the normal first-run
order — so setup finishes with the one command that starts a matching server
instead of failing. Only a generic writer with no model to write still refuses.
"""

from __future__ import annotations

import socket
import urllib.error
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.agents import adapter, get_profile, setup
from rapid_mlx.agents import telemetry as agent_telemetry
from rapid_mlx.agents.server_hint import (
    ServerNotRunningError,
    is_connection_refused,
    start_server_command,
)


@pytest.fixture
def dead_url():
    """A loopback base URL whose port has nothing listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1", port


@pytest.fixture
def events(monkeypatch):
    recorded: list[tuple] = []

    def configured(agent):
        recorded.append(("configured", agent))

    def failed(error_class, agent):
        recorded.append(("configure_failed", error_class, agent))

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


def test_start_command_falls_back_to_recommended_model_for_default():
    profile = get_profile("claude-code")
    assert start_server_command(profile, "http://127.0.0.1:8000/v1", "default") == (
        f"rapid-mlx serve {profile.recommended_models[0]} --port 8000"
    )


def test_start_command_without_recommendation_uses_placeholder_and_host():
    profile = SimpleNamespace(recommended_models=[])
    assert (
        start_server_command(profile, "http://192.168.1.5:9000", "default")
        == "rapid-mlx serve <model> --port 9000 --host 192.168.1.5"
    )


def test_start_command_quotes_model_and_survives_unparseable_url():
    profile = SimpleNamespace(recommended_models=[])
    assert start_server_command(profile, "ftp://x", "a b") == "rapid-mlx serve 'a b'"


# --- verify_server ---------------------------------------------------------


def test_verify_server_refused_raises_untracked_not_running(dead_url, events):
    base_url, _ = dead_url
    with pytest.raises(ServerNotRunningError, match="no server is running"):
        setup.verify_server(base_url, "default", agent="continue", timeout=1.0)
    assert events == []


def test_verify_server_other_connection_failure_is_still_tracked(monkeypatch, events):
    def timeout(*_args, **_kwargs):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(setup.urllib.request, "urlopen", timeout)
    with pytest.raises(RuntimeError, match="server is not ready") as raised:
        setup.verify_server("http://127.0.0.1:8000/v1", "m", agent="continue")
    assert not isinstance(raised.value, ServerNotRunningError)
    assert events == [("configure_failed", "server_not_ready", "continue")]


# --- first-class flow ------------------------------------------------------


@pytest.fixture
def continue_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(setup.continue_dev, "current_config_path", lambda: path)
    return path


def test_first_class_setup_saves_config_and_explains_start(
    dead_url, events, continue_config, capsys
):
    base_url, port = dead_url

    cli.agents_command(_args("continue", base_url, model="test-model"))

    output = capsys.readouterr().out
    assert continue_config.exists()
    assert "test-model" in continue_config.read_text()
    assert f"Server not running yet at http://127.0.0.1:{port}" in output
    assert f"Start it with:  rapid-mlx serve test-model --port {port}" in output
    assert "connection check failed" not in output
    assert events == [("configured", "continue")]


def test_first_class_unchanged_config_with_no_server_emits_nothing(
    dead_url, events, continue_config, capsys
):
    base_url, _ = dead_url
    cli.agents_command(_args("continue", base_url, model="test-model"))
    events.clear()

    cli.agents_command(_args("continue", base_url, model="test-model"))

    output = capsys.readouterr().out
    assert "Already configured" in output
    assert "Start it with:" in output
    assert events == []


# --- generic flow ----------------------------------------------------------


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def test_generic_setup_with_model_writes_and_explains_start(
    dead_url, events, codex_home, capsys
):
    base_url, port = dead_url

    cli.agents_command(_args("codex", base_url, model="local-qwen"))

    output = capsys.readouterr().out
    assert 'model = "local-qwen"' in (codex_home / "config.toml").read_text()
    assert "Codex CLI configured!" in output
    assert "Connection check passed" not in output
    assert f"Start it with:  rapid-mlx serve local-qwen --port {port}" in output
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
    recommended = get_profile("codex").recommended_models[0]
    assert (
        f"Start the server first:  rapid-mlx serve {recommended} --port {port}"
        in output
    )
    assert (
        f"rapid-mlx agents codex --setup --model <name> --base-url {base_url}" in output
    )
    assert "Traceback" not in output
    assert events == [("configure_failed", "server_not_ready", "codex")]


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
    assert "--setup --model <name>\n" in output
    assert "--base-url" not in output
