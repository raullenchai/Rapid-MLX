# SPDX-License-Identifier: Apache-2.0
"""One-time newsletter pointer under the interactive ``serve`` banner.

The pointer is a single dim line, printed at most once per machine, only
for a human at a terminal. Every other surface — piped output, CI, Desktop
sidecars, managed spawns, a run that already showed the update notice, a
machine that already saw it — stays byte-clean. The marker write happens
after the print and any filesystem failure is swallowed.
"""

from __future__ import annotations

import logging
import sys

import pytest

import rapid_mlx.server as server
from rapid_mlx import newsletter_hint
from rapid_mlx.config import get_config
from rapid_mlx.newsletter_hint import (
    NEWSLETTER_TEXT,
    NEWSLETTER_URL,
    marker_path,
    maybe_print_newsletter_hint,
    newsletter_line,
    should_show_newsletter_hint,
)

# ---------------------------------------------------------------------------
# The exact line users see
# ---------------------------------------------------------------------------


def test_newsletter_text_is_the_documented_line():
    assert (
        f"  New models that run on your Mac, the day they drop: {NEWSLETTER_URL}"
    ) == NEWSLETTER_TEXT


def test_newsletter_line_plain_without_color():
    assert newsletter_line(color=False) == NEWSLETTER_TEXT


def test_newsletter_line_dimmed_with_color():
    assert newsletter_line(color=True) == f"\x1b[2m{NEWSLETTER_TEXT}\x1b[0m"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, True),
        ({"NO_COLOR": "1"}, False),
        ({"NO_COLOR": ""}, True),
        ({"TERM": "dumb"}, False),
        ({"TERM": "xterm-256color"}, True),
    ],
)
def test_color_enabled_respects_terminal_opt_outs(monkeypatch, env, expected):
    for name in ("NO_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert newsletter_hint._color_enabled() is expected


# ---------------------------------------------------------------------------
# Pure gate — every exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"stdout_isatty": False}, False),
        ({"ci": True}, False),
        ({"managed_spawn": True}, False),
        ({"update_notice_shown": True}, False),
        ({"marker_exists": True}, False),
        ({"stdout_isatty": False, "marker_exists": True}, False),
    ],
)
def test_should_show_gate(overrides, expected):
    gates = {
        "stdout_isatty": True,
        "ci": False,
        "managed_spawn": False,
        "update_notice_shown": False,
        "marker_exists": False,
    }
    gates.update(overrides)
    assert should_show_newsletter_hint(**gates) is expected


# ---------------------------------------------------------------------------
# Marker location + lifecycle
# ---------------------------------------------------------------------------


def test_marker_lives_in_the_existing_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    assert marker_path() == tmp_path / "state" / "newsletter_hint_shown"


def test_prints_once_and_writes_marker(monkeypatch, capsys):
    monkeypatch.setattr(newsletter_hint, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_ci", lambda: False)
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)
    monkeypatch.setenv("NO_COLOR", "1")

    assert maybe_print_newsletter_hint() is True
    out = capsys.readouterr().out
    assert out == NEWSLETTER_TEXT + "\n"
    assert marker_path().exists()

    # Second serve on the same machine: banner unchanged, no pointer.
    assert maybe_print_newsletter_hint() is False
    assert capsys.readouterr().out == ""


def test_existing_marker_suppresses(monkeypatch, capsys):
    monkeypatch.setattr(newsletter_hint, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_ci", lambda: False)
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)
    marker_path().parent.mkdir(parents=True, exist_ok=True)
    marker_path().write_text("")

    assert maybe_print_newsletter_hint() is False
    assert capsys.readouterr().out == ""


def test_update_notice_suppresses_this_run_but_retries_later(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(newsletter_hint, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_ci", lambda: False)
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)

    assert maybe_print_newsletter_hint(update_notice_shown=True) is False
    assert capsys.readouterr().out == ""
    assert not marker_path().exists()

    # A later serve that did not show the update notice still gets the line.
    assert maybe_print_newsletter_hint(update_notice_shown=False) is True
    assert capsys.readouterr().out.startswith(NEWSLETTER_TEXT)


def test_marker_write_failure_is_silent_and_retries(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(newsletter_hint, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_ci", lambda: False)
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)

    calls: list[int] = []
    real_claim = newsletter_hint._claim_marker

    def flaky_claim():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("read-only state dir")
        real_claim()

    monkeypatch.setattr(newsletter_hint, "_claim_marker", flaky_claim)

    # The line still prints; the failure never reaches serve.
    assert maybe_print_newsletter_hint() is True
    assert capsys.readouterr().out.startswith(NEWSLETTER_TEXT)
    assert not marker_path().exists()

    # Because the marker was not written, a later run may try again.
    assert maybe_print_newsletter_hint() is True
    assert marker_path().exists()


def test_print_failure_writes_no_marker(monkeypatch):
    monkeypatch.setattr(newsletter_hint, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(newsletter_hint, "_ci", lambda: False)
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)

    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr("builtins.print", broken_print)

    assert maybe_print_newsletter_hint() is False
    assert not marker_path().exists()


def test_tty_probe_failure_fails_closed(monkeypatch, capsys):
    monkeypatch.setattr(
        sys.stdout, "isatty", lambda: (_ for _ in ()).throw(OSError("closed"))
    )
    monkeypatch.setattr(newsletter_hint, "_managed_spawn", lambda: False)
    assert maybe_print_newsletter_hint() is False
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Managed-spawn detection reuses the process-role detector
# ---------------------------------------------------------------------------


def test_desktop_sidecar_is_managed(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    assert newsletter_hint._managed_spawn() is True


@pytest.mark.parametrize("ppid", ["2", "4242"])
def test_watchdog_child_is_managed(monkeypatch, ppid):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", ppid)
    assert newsletter_hint._managed_spawn() is True


def test_detector_failure_fails_closed(monkeypatch):
    def boom():
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr("rapid_mlx.telemetry.consent_runtime.detect_role", boom)
    assert newsletter_hint._managed_spawn() is True


# ---------------------------------------------------------------------------
# Integration with the real post-bind banner
# ---------------------------------------------------------------------------


@pytest.fixture
def interactive_banner(monkeypatch, tmp_path):
    """Make ``print_ready_banner`` take its interactive branch."""
    cfg = get_config()
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    saved = (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd)
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.bind_listen_fd = None
    cfg.model_alias = "lfm2.5-1b-4bit"
    cfg.api_key = None
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )
    monkeypatch.setattr("rapid_mlx.server._text_lane_image_note", lambda _cfg: None)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    yield cfg
    cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd = saved


def test_banner_prints_pointer_under_ready_line(
    interactive_banner, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready: http://localhost:8766" in out
    assert NEWSLETTER_TEXT in out
    assert out.index("Ready: http://localhost:8766") < out.index(NEWSLETTER_TEXT)
    assert marker_path().exists()


def test_banner_skips_pointer_when_update_notice_shown(
    interactive_banner, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    server.print_ready_banner(update_notice_shown=True)
    out = capsys.readouterr().out
    assert "Ready:" in out
    assert NEWSLETTER_TEXT not in out
    assert not marker_path().exists()


def test_banner_skips_pointer_when_marker_present(
    interactive_banner, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    marker_path().parent.mkdir(parents=True, exist_ok=True)
    marker_path().write_text("")
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready:" in out
    assert NEWSLETTER_TEXT not in out


def test_banner_skips_pointer_for_piped_output(interactive_banner, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready:" in out
    assert NEWSLETTER_TEXT not in out


# ---------------------------------------------------------------------------
# CLI plumbing: the update-notice state reaches the post-bind banner
# ---------------------------------------------------------------------------

from rapid_mlx import cli  # noqa: E402
from tests.test_serve_listen_fd import (  # noqa: E402,F401
    _free_tcp_port,
    _minimal_serve_ns,
    stub_heavy_serve_deps,
)


def test_run_uvicorn_binds_update_notice_state(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "rapid_mlx._uvicorn._run_server", lambda app, **kw: captured.update(kw)
    )
    ns = _minimal_serve_ns(port=9000, host="127.0.0.1")
    cli._run_uvicorn(object(), ns, "info", update_notice_shown=True)

    callback = captured["on_server_accepting"]
    assert callback.func is server.print_ready_banner
    assert callback.keywords == {"update_notice_shown": True}


@pytest.mark.parametrize(
    ("announced", "expected"),
    [("0.9.0", True), (None, False)],
)
def test_serve_command_threads_announced_update_into_banner(
    stub_heavy_serve_deps,  # noqa: F811 - fixture imported from the sibling module
    monkeypatch,
    announced,
    expected,
):
    from rapid_mlx import _version_check

    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kw: announced
    )
    captured: dict = {}
    monkeypatch.setattr(
        "rapid_mlx._uvicorn._run_server", lambda app, **kw: captured.update(kw)
    )
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)

    cli.serve_command(_minimal_serve_ns(port=_free_tcp_port()))

    callback = captured["on_server_accepting"]
    assert callback.func is server.print_ready_banner
    assert callback.keywords == {"update_notice_shown": expected}
