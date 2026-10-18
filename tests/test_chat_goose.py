"""Goose CLI startup contract without loading a model."""

from pathlib import Path
from types import SimpleNamespace

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
    commands = []

    def capture_goose(command, **_kwargs):
        commands.append(command)
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
    assert commands[0][commands[0].index("--with-builtin") + 1] == chat_goose.BUILTINS


def test_goose_cli_falls_back_for_remote_server(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    args = SimpleNamespace(mcp_config=None, mcp_max_rounds=8, response_timeout=600)
    assert not chat_goose.run(
        args, "https://example.com", "test-model", max_tokens_explicit=False
    )
