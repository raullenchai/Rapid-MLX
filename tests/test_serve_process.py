# SPDX-License-Identifier: Apache-2.0
"""Entry-point and subcommand regressions for process discovery (#5584)."""

import time
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.serve_process import serve_subcommand_index

# These are the final server argv forms after launchers exec their children.
LAUNCHERS = [
    ["rapid-mlx"],
    ["/opt/homebrew/bin/rapid-mlx"],
    ["rmlx"],
    ["vllm-mlx"],
    ["python", "/venv/bin/rapid-mlx"],
    [
        "Python",
        "/Users/me/Library/Application Support/pipx/venvs/rapid-mlx/bin/rapid-mlx",
    ],
    ["/uv/tools/rapid-mlx/bin/python3", "/uv/tools/rapid-mlx/bin/rapid-mlx"],
    ["/uv/cache/env/bin/python3.12", "/uv/cache/env/bin/rapid-mlx"],
    ["python3", "-m", "rapid_mlx"],
    ["python3", "-m", "rapid_mlx.cli"],
    ["python3", "-X", "serve", "-m", "rapid_mlx.cli"],
    ["python3", "-m", "vllm_mlx"],
    ["python3", "-m", "vllm_mlx.cli"],
    [
        "/Applications/Rapid.app/Contents/Resources/rapid-mlx/python/bin/python3.12",
        "-P",
        "-u",
        "-s",
        "-m",
        "rapid_mlx.cli",
    ],
    ["python3.12", "-Pus", "-mrapid_mlx.cli"],
    [
        "python",
        "-W",
        "ignore",
        "-X",
        "dev",
        "--check-hash-based-pycs",
        "default",
        "-m",
        "rapid_mlx.cli",
    ],
    ["python", "-Wignore", "-Xdev", "--", "/bin/rapid-mlx"],
]

NOT_SERVERS = [
    [],
    ["python"],
    ["rapid-mlx"],
    ["grep", "rapid-mlx", "serve"],
    ["vim", "rapid_mlx/serve.py"],
    ["vim", "rapid_mlx/serve.py", "serve"],
    ["python", "other.py", "serve", "--x", "rapid_mlx"],
    ["python", "other.py", "-m", "rapid_mlx", "serve"],
    ["python", "-c", "rapid_mlx", "serve"],
    ["python", "-", "rapid-mlx", "serve"],
    ["pythonista", "-m", "rapid_mlx", "serve"],
    ["python-launcher", "/bin/rapid-mlx", "serve"],
    ["python", "-m"],
    ["python", "-W"],
    ["python", "--"],
    ["python", "-m", "rapid_mlx_other", "serve"],
    ["python", "-m", "other.rapid_mlx", "serve"],
    ["python", "/tmp/not-rapid-mlx", "serve"],
    ["not-rapid-mlx", "serve"],
    ["rapid-mlx", "agents", "serve"],
    ["rapid-mlx", "chat", "--model", "serve"],
    ["rapid-mlx", "--help", "serve"],
    ["rapid-mlx", "--version", "serve"],
    ["rapid-mlx", "--unknown", "serve"],
    ["rapid-mlx", "--no-banner"],
    ["python", "-m", "rapid_mlx.cli", "models", "serve"],
    ["/usr/bin/caffeinate", "-is", "rapid-mlx", "serve", "tiny"],
    ["uv", "run", "rapid-mlx", "serve", "tiny"],
    ["uv", "run", "--with", "rapid-mlx", "python", "-m", "rapid_mlx.cli", "serve"],
    ["pipx", "run", "rapid-mlx", "serve", "tiny"],
    ["sh", "-c", "rapid-mlx", "serve"],
]


@pytest.mark.parametrize("entry", LAUNCHERS)
@pytest.mark.parametrize(
    "globals", [[], ["--no-banner", "--no-telemetry", "--disable-version-check"]]
)
def test_supported_launchers(entry, globals, monkeypatch, agent_discovery_defaults):
    import psutil

    argv = [*entry, *globals, "serve", "tiny", "--port", "8462"]
    assert serve_subcommand_index(argv) == len(entry) + len(globals)
    proc = SimpleNamespace(
        pid=123,
        info={"pid": 123, "cmdline": argv, "create_time": time.time()},
        net_connections=lambda **_k: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_a: [proc])
    servers = agent_discovery_defaults()
    assert [(s.pid, s.port, s.model) for s in servers] == [(123, 8462, "tiny")]
    assert [row[:3] for row in cli._scan_running_servers()] == [(123, "8462", "tiny")]


@pytest.mark.parametrize("argv", NOT_SERVERS)
def test_rejects_foreign_commands_and_wrappers(
    argv, monkeypatch, agent_discovery_defaults
):
    import psutil

    assert serve_subcommand_index(argv) is None
    # No process metadata/socket access is needed for a non-server.
    proc = SimpleNamespace(info={"cmdline": argv})
    monkeypatch.setattr(psutil, "process_iter", lambda *_a: [proc])
    assert agent_discovery_defaults() == []
    assert cli._scan_running_servers() == []


@pytest.mark.parametrize("executable", ["/bin/not-rapid-mlx", "/bin/rapid-mlx-backup"])
def test_legacy_service_rejects_lookalike_executable(executable):
    from rapid_mlx.headless_service.status import _parse_legacy_serve

    with pytest.raises(ValueError, match="unrecognized legacy service executable"):
        _parse_legacy_serve([executable, "serve", "tiny"])


def test_legacy_service_uses_shared_subcommand_matching():
    from rapid_mlx.headless_service.status import _parse_legacy_serve

    assert _parse_legacy_serve(
        ["/bin/rapid-mlx", "--no-banner", "serve", "tiny", "--port", "8000"]
    ) == (
        "/bin/rapid-mlx",
        "tiny",
        "127.0.0.1",
        8000,
    )
    with pytest.raises(ValueError, match="unrecognized legacy service executable"):
        _parse_legacy_serve(["/bin/rapid-mlx", "agents", "serve"])
