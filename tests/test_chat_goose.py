"""Goose CLI startup contract without loading a model."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import chat_goose


def test_goose_cli_resources_match_desktop_harness():
    desktop = Path(__file__).resolve().parents[1] / "apps/rapid-mac/Work"
    pairs = {
        "inference_proxy.py": desktop / "work-inference-proxy.py",
        "helper_watchdog.py": desktop / "work-helper-watchdog.py",
        "prefix.json": desktop / "goose-prefix.json",
    }
    for name, source in pairs.items():
        assert (chat_goose.ASSETS / name).read_bytes() == source.read_bytes()


def test_goose_cli_starts_desktop_helpers_and_runs_pinned_harness(
    tmp_path, monkeypatch
):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "What causes tides?")
    monkeypatch.setenv("PATH", "/opt/user-tools:/usr/bin:/bin")
    commands = []
    environments = []

    def capture_goose(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(chat_goose.subprocess, "run", capture_goose)
    args = SimpleNamespace(
        mcp_config=None,
        mcp_max_rounds=8,
        response_timeout=600,
        context_length=None,
        system=None,
        temperature=0.7,
        max_tokens=2048,
        think=False,
        disable_prefix_cache=True,
        output_format="stream-json",
        resume=True,
    )

    assert chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=False
    )
    workspace = next((tmp_path / "cache" / "workspaces").iterdir())
    assert (workspace / "chat-contract.json").is_file()
    assert (workspace / "config/custom_providers/custom_rapid_local.json").is_file()
    assert (workspace / "bash").is_file()
    assert len(commands) == 1
    assert "--no-profile" not in commands[0]
    assert "--no-session" not in commands[0]
    assert "--resume" in commands[0]
    assert commands[0][commands[0].index("--with-builtin") + 1] == chat_goose.BUILTINS
    assert "/opt/user-tools" in environments[0]["PATH"]
    assert "WORK_PROXY_BEARER" not in environments[0]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"mcp_config": "tools.json"}, "--mcp-config"),
        ({"mcp_max_rounds": 12}, "--mcp-max-rounds"),
        ({"response_timeout": 30}, "--response-timeout"),
    ],
)
def test_goose_cli_explains_unsupported_flags(monkeypatch, capsys, changes, reason):
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    values = {"mcp_config": None, "mcp_max_rounds": 8, "response_timeout": 600}
    values.update(changes)
    assert not chat_goose.run(
        SimpleNamespace(**values),
        "http://127.0.0.1:8123",
        "test-model",
        max_tokens_explicit=False,
    )
    error = capsys.readouterr().err
    assert error.count("Goose unavailable") == 1
    assert reason in error


def test_goose_cli_falls_back_for_remote_server(monkeypatch, capsys):
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    args = SimpleNamespace(mcp_config=None, mcp_max_rounds=8, response_timeout=600)
    assert not chat_goose.run(
        args, "https://example.com", "test-model", max_tokens_explicit=False
    )
    assert "requires a local HTTP Rapid server" in capsys.readouterr().err


def test_goose_cli_explains_download_failure(monkeypatch, capsys):
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        chat_goose, "_goose_binary", lambda: (_ for _ in ()).throw(OSError("offline"))
    )
    args = SimpleNamespace(mcp_config=None, mcp_max_rounds=8, response_timeout=600)
    assert not chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=False
    )
    assert capsys.readouterr().err.count("Goose unavailable (offline)") == 1
