# SPDX-License-Identifier: Apache-2.0
"""The newsletter pointer appears once after an interactive serve is ready."""

import fcntl
import logging
import sys

import pytest

import rapid_mlx.server as server
from rapid_mlx import cli, newsletter_hint, update_refresh
from rapid_mlx.config import get_config
from rapid_mlx.newsletter_hint import (
    NEWSLETTER_TEXT,
    _interactive_stdout,
    maybe_print_newsletter_hint,
)
from rapid_mlx.telemetry.consent_decision import ProcessRole
from tests.test_serve_listen_fd import (  # noqa: F401
    _free_tcp_port,
    _minimal_serve_ns,
    stub_heavy_serve_deps,
)


@pytest.fixture
def terminal(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_interactive_stdout", lambda: True)
    monkeypatch.setattr(
        newsletter_hint, "detect_role", lambda: ProcessRole.INTERACTIVE_CLI
    )
    monkeypatch.setattr(newsletter_hint, "_running_in_desktop_sidecar", lambda: False)
    monkeypatch.setattr(update_refresh, "_notice_printed", False)
    return tmp_path / "state" / "newsletter_hint_shown"


@pytest.mark.parametrize("generic_banner", [False, True])
def test_prints_once_across_two_serves(terminal, monkeypatch, capsys, generic_banner):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    cfg = get_config()
    saved = (
        cfg.bind_host,
        cfg.bind_port,
        cfg.bind_listen_fd,
        cfg.model_alias,
        cfg.api_key,
    )
    cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd = "localhost", 8766, None
    cfg.model_alias, cfg.api_key = "lfm2.5-1b-4bit", None
    level = logging.WARNING if generic_banner else logging.INFO
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: level)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    monkeypatch.setattr(server, "_text_lane_image_note", lambda _cfg: None)
    try:
        server.print_ready_banner()
        first = capsys.readouterr().out
        server.print_ready_banner()
        second = capsys.readouterr().out
    finally:
        (
            cfg.bind_host,
            cfg.bind_port,
            cfg.bind_listen_fd,
            cfg.model_alias,
            cfg.api_key,
        ) = saved
    assert "  Ready: http://localhost:8766" in first
    assert first.index("  Ready:") < first.index(NEWSLETTER_TEXT)
    assert first.count(NEWSLETTER_TEXT) == 1
    assert NEWSLETTER_TEXT not in second
    assert terminal.exists()


@pytest.mark.parametrize(
    "gate", ["non_tty", "ci", "desktop", "watchdog", "update", "marker"]
)
def test_suppressed_gates(terminal, monkeypatch, capsys, gate):
    if gate == "non_tty":
        monkeypatch.setattr(newsletter_hint, "_interactive_stdout", lambda: False)
    elif gate == "ci":
        monkeypatch.setenv("CI", "")  # Presence, even empty, means CI.
    elif gate == "desktop":
        monkeypatch.setattr(newsletter_hint, "detect_role", lambda: ProcessRole.SIDECAR)
    elif gate == "watchdog":
        monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "4242")
        from rapid_mlx.telemetry.consent_runtime import detect_role

        monkeypatch.setattr(newsletter_hint, "detect_role", detect_role)
    elif gate == "marker":
        terminal.parent.mkdir(parents=True)
        terminal.touch()
    maybe_print_newsletter_hint(update_notice_shown=gate == "update")
    assert capsys.readouterr().out == ""
    assert terminal.exists() is (gate == "marker")


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar"),
        ("RAPID_MLX_WATCHDOG_PPID", "4242"),
    ],
)
def test_managed_spawn_uses_existing_role_detector(terminal, monkeypatch, env, value):
    from rapid_mlx.telemetry.consent_runtime import detect_role

    monkeypatch.setenv(env, value)
    assert detect_role() is ProcessRole.SIDECAR
    monkeypatch.setattr(newsletter_hint, "detect_role", detect_role)
    maybe_print_newsletter_hint()
    assert not terminal.exists()


def test_desktop_app_path_is_managed(terminal, monkeypatch, capsys):
    monkeypatch.setattr(newsletter_hint, "_running_in_desktop_sidecar", lambda: True)
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == ""
    assert not terminal.exists()


def test_update_notice_defers_pointer(terminal, capsys):
    maybe_print_newsletter_hint(update_notice_shown=True)
    assert not terminal.exists()
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == NEWSLETTER_TEXT + "\n"
    assert terminal.exists()


def test_stdout_probe(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "fileno", lambda: 1)
    monkeypatch.setattr(newsletter_hint.os, "tcgetpgrp", lambda _fd: 42)
    monkeypatch.setattr(newsletter_hint.os, "getpgrp", lambda: 42)
    assert newsletter_hint._interactive_stdout()


@pytest.mark.parametrize("probe", ["background", "error"])
def test_background_job_does_not_consume_marker(terminal, monkeypatch, capsys, probe):
    monkeypatch.setattr(newsletter_hint, "_interactive_stdout", _interactive_stdout)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "fileno", lambda: 1)
    monkeypatch.setattr(newsletter_hint.os, "getpgrp", lambda: 42)

    def terminal_group(_fd):
        if probe == "error":
            raise OSError("no controlling terminal")
        return 43

    monkeypatch.setattr(newsletter_hint.os, "tcgetpgrp", terminal_group)
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == ""
    assert not terminal.exists()


def test_background_update_before_print_defers_pointer(terminal, monkeypatch, capsys):
    from rapid_mlx import _version_check as vc

    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "get_latest_version", lambda: "1.1.0")
    monkeypatch.setattr(update_refresh, "_seen_versions", set())
    real_flock = fcntl.flock

    def announce_while_acquiring_lock(lock, mode):
        real_flock(lock, mode)
        update_refresh._refresh()

    monkeypatch.setattr(fcntl, "flock", announce_while_acquiring_lock)
    maybe_print_newsletter_hint(update_notice_shown=False)
    output = capsys.readouterr()
    assert vc.format_staleness_warning("1.0.0", "1.1.0") in output.err
    assert output.out == ""
    assert not terminal.exists()


def test_marker_write_failure_is_silent(terminal, monkeypatch, capsys):
    class FailedMarker:
        def touch(self, **_kwargs):
            raise OSError("read only")

    # Keep the real lock/print path, but make only the marker write fail.
    monkeypatch.setattr("pathlib.Path.touch", FailedMarker.touch)
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == NEWSLETTER_TEXT + "\n"
    assert not terminal.exists()


def test_lock_contention_does_not_block_ready(terminal, capsys):
    terminal.parent.mkdir(parents=True)
    with (terminal.parent / "newsletter_hint.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        maybe_print_newsletter_hint()
    assert capsys.readouterr().out == ""
    assert not terminal.exists()


def test_print_failure_does_not_mark(terminal, monkeypatch):
    def fail(*_args, **_kwargs):
        raise BrokenPipeError("closed")

    monkeypatch.setattr("builtins.print", fail)
    maybe_print_newsletter_hint()
    assert not terminal.exists()


def test_state_failure_does_not_affect_serve(terminal, monkeypatch, capsys):
    def fail():
        raise OSError("state unavailable")

    monkeypatch.setattr(newsletter_hint, "_state_dir", fail)
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == ""


def test_dim_and_no_color(terminal, monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR")
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == f"\x1b[2m{NEWSLETTER_TEXT}\x1b[0m\n"
    terminal.unlink()
    monkeypatch.setenv("TERM", "dumb")
    maybe_print_newsletter_hint()
    assert capsys.readouterr().out == NEWSLETTER_TEXT + "\n"


@pytest.mark.parametrize("announced", ["0.9.0", None])
@pytest.mark.usefixtures("stub_heavy_serve_deps", "scheduler_config_stub")
def test_serve_threads_update_notice(monkeypatch, announced):
    from rapid_mlx import _version_check

    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kw: announced
    )
    captured = {}
    monkeypatch.setattr(
        "rapid_mlx._uvicorn._run_server", lambda app, **kw: captured.update(kw)
    )
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    cli.serve_command(_minimal_serve_ns(port=_free_tcp_port()))
    callback = captured["on_server_accepting"]
    if announced:
        assert callback.func is server.print_ready_banner
        assert callback.keywords == {"update_notice_shown": True}
    else:
        assert callback is server.print_ready_banner
