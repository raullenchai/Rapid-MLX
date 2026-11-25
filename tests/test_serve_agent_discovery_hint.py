# SPDX-License-Identifier: Apache-2.0
"""Coding-agent discovery hint printed once per interactive ``serve`` start."""

from __future__ import annotations

import logging
import socket
import urllib.request

import pytest

import rapid_mlx.connect as connect
import rapid_mlx.server as server
from rapid_mlx.config import get_config


@pytest.fixture(autouse=True)
def _isolate_banner_state(monkeypatch, tmp_path):
    cfg = get_config()
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("RAPID_MLX_NO_BANNER", raising=False)
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    monkeypatch.setenv("CI", "1")  # Keep newsletter hint out of these assertions.
    monkeypatch.setattr(server, "_CODING_AGENTS_HINT_SHOWN", False)
    # Most tests model a later start: the once-per-install Next block has
    # already been claimed, so the one-line hint is what the banner prints.
    from rapid_mlx.first_run import claim_serve_next_tip

    claim_serve_next_tip()
    saved = (
        server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.bind_listen_fd,
        cfg.model_alias,
        cfg.model_name,
        cfg.api_key,
        cfg.embedding_engine,
        cfg.rerank_engine,
    )
    server._engine = None
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.bind_listen_fd = None
    cfg.model_alias = "qwen3.5-4b-4bit"
    cfg.model_name = "qwen3.5-4b-4bit"
    cfg.api_key = None
    cfg.embedding_engine = None
    cfg.rerank_engine = None
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    yield
    (
        server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.bind_listen_fd,
        cfg.model_alias,
        cfg.model_name,
        cfg.api_key,
        cfg.embedding_engine,
        cfg.rerank_engine,
    ) = saved


def test_render_coding_agents_hint_format():
    assert (
        connect.render_coding_agents_hint(["opencode", "codex"])
        == "  Coding agents found: opencode, codex -> rapid-mlx launch opencode\n"
    )
    assert connect.render_coding_agents_hint([]) == ""


def test_serve_ready_prints_coding_agents_hint_once_per_start_on_tty(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    # Fresh install: the Next block has not been shown yet.
    (tmp_path / "state" / "serve_next_tip_shown").unlink()
    installed = {"opencode", "codex"}
    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/usr/local/bin/{name}" if name in installed else None,
    )

    server.print_ready_banner()
    first = capsys.readouterr().out
    server.print_ready_banner()
    second = capsys.readouterr().out
    server.print_ready_banner()
    third = capsys.readouterr().out

    expected = "  Coding agents found: codex, opencode -> rapid-mlx launch codex"
    # The once-per-install Next block owns the very first banner; the hint
    # must not duplicate its launch commands.
    assert "Next:" in first
    assert first.count(expected) == 0
    # From the next start the one-line hint appears exactly once per process.
    assert second.count(expected) == 1
    assert second.index("  Ready: http://localhost:8766") < second.index(expected)
    assert "Next:" not in second
    assert third.count(expected) == 0
    assert "  Ready: http://localhost:8766" in third


def test_serve_ready_detects_profile_binaries_including_claude_pi_hermes(
    monkeypatch, capsys
):
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    installed = {"claude", "pi", "hermes"}
    monkeypatch.setattr(
        "shutil.which", lambda name: f"/opt/bin/{name}" if name in installed else None
    )

    server.print_ready_banner()
    out = capsys.readouterr().out
    assert (
        "  Coding agents found: claude-code, pi, hermes -> rapid-mlx launch claude-code"
        in out
    )


def test_discovery_hint_writes_nothing_to_disk_and_makes_no_network_calls(
    monkeypatch, tmp_path
):
    def forbid(*_args, **_kwargs):
        raise AssertionError("network call is forbidden during agent discovery")

    def forbid_write(*_args, **_kwargs):
        raise AssertionError("disk write is forbidden during agent discovery")

    import builtins
    import http.client
    import os
    import pathlib

    # Network primitives: DNS, raw sockets, http.client and urllib.
    monkeypatch.setattr(urllib.request, "urlopen", forbid)
    monkeypatch.setattr(socket, "create_connection", forbid)
    monkeypatch.setattr(socket, "getaddrinfo", forbid)
    monkeypatch.setattr(socket.socket, "connect", forbid)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", forbid)
    # Filesystem mutation primitives: open() for writing, os and pathlib writers.
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            forbid_write()
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("mkdir", "makedirs", "replace", "rename", "remove", "unlink"):
        monkeypatch.setattr(os, name, forbid_write)
    for name in ("write_text", "write_bytes", "mkdir", "touch", "unlink"):
        monkeypatch.setattr(pathlib.Path, name, forbid_write)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/opencode" if name == "opencode" else None
    )
    before = set(tmp_path.rglob("*"))
    hint = connect.render_coding_agents_hint()
    after = set(tmp_path.rglob("*"))
    assert hint == "  Coding agents found: opencode -> rapid-mlx launch opencode\n"
    assert before == after


@pytest.mark.parametrize(
    "gate",
    [
        None,
        "non_tty",
        "quiet_warning",
        "quiet_error",
        "no_banner_env",
        "desktop_sidecar",
        "watchdog_child",
        "process_role_sidecar",
        "json_mode",
        "no_agents",
        "audio_lane",
        "retrieval_lane",
    ],
)
def test_serve_ready_coding_agents_hint_suppressed(monkeypatch, capsys, gate):
    cfg = get_config()
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: gate != "non_tty")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            None
            if gate == "no_agents"
            else ("/usr/bin/opencode" if name == "opencode" else None)
        ),
    )
    if gate == "quiet_warning":
        monkeypatch.setattr(
            logging.getLogger(), "getEffectiveLevel", lambda: logging.WARNING
        )
    elif gate == "quiet_error":
        monkeypatch.setattr(
            logging.getLogger(), "getEffectiveLevel", lambda: logging.ERROR
        )
    elif gate == "no_banner_env":
        monkeypatch.setenv("RAPID_MLX_NO_BANNER", "1")
    elif gate == "desktop_sidecar":
        monkeypatch.setattr(
            "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
            lambda: True,
        )
    elif gate == "watchdog_child":
        monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "4242")
    elif gate == "process_role_sidecar":
        monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    elif gate == "json_mode":
        monkeypatch.setattr(cfg, "json_output", True, raising=False)
    elif gate == "audio_lane":
        cfg.model_alias = "kokoro"
        cfg.model_name = "kokoro"
    elif gate == "retrieval_lane":
        cfg.model_alias = None
        cfg.model_name = None
        cfg.embedding_engine = object()

    server.print_ready_banner()
    out = capsys.readouterr().out
    if gate is None:
        assert "Coding agents found: opencode -> rapid-mlx launch opencode" in out
    else:
        assert "Coding agents found:" not in out
