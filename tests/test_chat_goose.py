"""Goose CLI startup contract without loading a model."""

import hashlib
import io
import json
import os
import tarfile
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


def test_goose_archive_is_verified_and_cached(tmp_path, monkeypatch):
    executable = b"#!/bin/sh\nexit 0\n"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as package:
        entry = tarfile.TarInfo("goose")
        entry.size = len(executable)
        package.addfile(entry, io.BytesIO(executable))
    payload = archive.getvalue()
    monkeypatch.setattr(chat_goose.sys, "platform", "darwin")
    monkeypatch.setattr(
        chat_goose.os, "uname", lambda: SimpleNamespace(machine="arm64")
    )
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(chat_goose, "GOOSE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        chat_goose, "GOOSE_BINARY_SHA256", hashlib.sha256(executable).hexdigest()
    )
    downloads = []

    def download(_url, timeout):
        downloads.append(timeout)
        return io.BytesIO(payload)

    monkeypatch.setattr(chat_goose.urllib.request, "urlopen", download)
    binary = chat_goose._goose_binary()
    assert binary.read_bytes() == executable
    assert chat_goose._goose_binary() == binary
    assert downloads == [120]
    binary.write_bytes(b"tampered")
    (tmp_path / "binary.sha256").write_text(hashlib.sha256(b"tampered").hexdigest())
    assert chat_goose._goose_binary().read_bytes() == executable
    assert downloads == [120]


def test_context_limit_uses_authenticated_server_metadata(monkeypatch):
    requests = []

    def metadata(request, timeout):
        requests.append(
            (request.full_url, request.get_header("Authorization"), timeout)
        )
        return io.BytesIO(b'{"data":[{"id":"test-model","context_window":8192}]}')

    monkeypatch.setattr(chat_goose.urllib.request, "urlopen", metadata)
    assert (
        chat_goose._served_context_limit(
            "http://127.0.0.1:8123", "test-model", "real-server-key"
        )
        == 8192
    )
    assert requests == [
        ("http://127.0.0.1:8123/v1/models", "Bearer real-server-key", 5)
    ]


def test_goose_cli_starts_desktop_helpers_and_runs_pinned_harness(
    tmp_path, monkeypatch
):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose, "_served_context_limit", lambda *_: 8192)
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "What causes tides?")
    monkeypatch.setenv("PATH", "/opt/user-tools:/usr/bin:/bin")
    workspace = (
        tmp_path / "cache/workspaces" / hashlib.sha256(os.getcwd().encode()).hexdigest()
    )
    (workspace / "config").mkdir(parents=True)
    (workspace / "config/permission.yaml").write_text("native Goose grants\n")
    commands = []
    environments = []
    snapshots = []

    def capture_goose(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        name = kwargs["env"]["GOOSE_PROVIDER"]
        snapshots.append(
            (
                json.loads(
                    next((workspace / "runs").glob("*/chat-contract.json")).read_text()
                ),
                json.loads(
                    (workspace / "config/custom_providers" / f"{name}.json").read_text()
                ),
            )
        )
        return 0

    monkeypatch.setattr(chat_goose, "_run_goose", capture_goose)
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
    assert snapshots[0][0]["use_goose_sampling_defaults"] is True
    assert snapshots[0][1]["models"][0]["context_limit"] == 8192
    assert (workspace / "config/permission.yaml").read_text() == "native Goose grants\n"
    assert not list((workspace / "runs").iterdir())
    assert len(commands) == 1
    assert "--no-profile" not in commands[0]
    assert "--no-session" not in commands[0]
    assert "--resume" in commands[0]
    assert (
        commands[0][commands[0].index("--provider") + 1]
        == environments[0]["GOOSE_PROVIDER"]
    )
    assert commands[0][commands[0].index("--with-builtin") + 1] == chat_goose.BUILTINS
    assert "/opt/user-tools" in environments[0]["PATH"]
    assert "WORK_PROXY_BEARER" not in environments[0]


def test_goose_cli_maps_sampling_and_system_flags(tmp_path, monkeypatch):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "Hello")
    snapshots = []

    def capture_goose(_command, **kwargs):
        workspace = Path(kwargs["env"]["HOME"])
        name = kwargs["env"]["GOOSE_PROVIDER"]
        snapshots.append(
            (
                json.loads(
                    next((workspace / "runs").glob("*/chat-contract.json")).read_text()
                ),
                json.loads(
                    (workspace / "config/custom_providers" / f"{name}.json").read_text()
                ),
            )
        )
        return 0

    monkeypatch.setattr(chat_goose, "_run_goose", capture_goose)
    args = SimpleNamespace(
        mcp_config=None,
        mcp_max_rounds=8,
        response_timeout=600,
        context_length=8192,
        system="Be concise",
        temperature=0.2,
        max_tokens=512,
        think=True,
        disable_prefix_cache=True,
        output_format="text",
        resume=False,
    )
    assert chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=True
    )
    contract, provider = snapshots[0]
    assert contract["system_prompt"] == "Be concise"
    assert contract["use_goose_sampling_defaults"] is False
    assert contract["temperature"] == 0.2
    assert contract["max_tokens"] == 512
    assert contract["enable_thinking"] is True
    assert provider["models"][0] == {"name": "test-model", "context_limit": 8192}


def test_explicit_default_sampling_flags_override_goose_defaults(tmp_path, monkeypatch):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "Hello")
    captured = []

    def capture(_command, **kwargs):
        workspace = Path(kwargs["env"]["HOME"])
        captured.append(
            json.loads(
                next((workspace / "runs").glob("*/chat-contract.json")).read_text()
            )
        )
        return 0

    monkeypatch.setattr(chat_goose, "_run_goose", capture)
    args = SimpleNamespace(
        mcp_config=None,
        mcp_max_rounds=8,
        response_timeout=600,
        context_length=8192,
        system=None,
        temperature=0.7,
        think=False,
        _temperature_was_explicit=True,
        _think_was_explicit=True,
        max_tokens=2048,
        disable_prefix_cache=True,
        output_format="text",
        resume=False,
    )
    assert chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=False
    )
    assert captured[0]["use_goose_sampling_defaults"] is False
    assert captured[0]["enable_thinking"] is False


def test_early_goose_exit_replays_prompt_in_builtin_chat(tmp_path, monkeypatch, capsys):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "Keep this prompt")
    monkeypatch.setattr(chat_goose, "_run_goose", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(chat_goose, "_metric_sequence", lambda *_args: 0)
    args = SimpleNamespace(
        mcp_config=None,
        mcp_max_rounds=8,
        response_timeout=600,
        context_length=8192,
        system=None,
        temperature=0.7,
        think=False,
        max_tokens=2048,
        disable_prefix_cache=True,
        output_format="text",
        resume=False,
    )
    assert not chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=False
    )
    assert args._goose_pending_prompt == "Keep this prompt"
    assert capsys.readouterr().err.count("Goose unavailable") == 1


def test_piped_stream_json_runs_goose_without_stdout_banner(
    tmp_path, monkeypatch, capsys
):
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(chat_goose, "_goose_binary", lambda: binary)
    monkeypatch.setattr(chat_goose, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat_goose.sys, "stdin", io.StringIO("Piped prompt\n"))
    commands = []
    monkeypatch.setattr(
        chat_goose, "_run_goose", lambda command, **_: commands.append(command) or 0
    )
    args = SimpleNamespace(
        mcp_config=None,
        mcp_max_rounds=8,
        response_timeout=600,
        context_length=8192,
        system=None,
        temperature=0.7,
        think=False,
        max_tokens=2048,
        disable_prefix_cache=True,
        output_format="stream-json",
        resume=False,
    )
    assert chat_goose.run(
        args, "http://127.0.0.1:8123", "test-model", max_tokens_explicit=False
    )
    assert "--interactive" not in commands[0]
    assert commands[0][commands[0].index("--text") + 1] == "Piped prompt"
    assert capsys.readouterr().out == ""


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
