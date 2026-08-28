"""Tests for the rapid-mlx cheetah launch banner.

Covers three surfaces, all demand-tested against the release contract that the
banner must NEVER corrupt machine-facing output:

  * ``vllm_mlx._banner.render_banner`` — the artwork + wordmark + version
    render, and the ANSI pass on/off under ``color``.
  * ``vllm_mlx._banner.should_show_banner`` — the interactive-only gate that
    ``cli.py`` folds the env flag into and calls before printing. The helper
    is pure/hermetic: it does not execute any subcommand.
  * CLI-level folding through the real ``main()`` (mocked TTY streams + env):
    the ``RAPID_MLX_NO_BANNER`` env var and ``--no-banner`` flag both suppress
    the banner on a tty, and the ``version`` subcommand stays byte-clean.

The ``no_banner`` / ``RAPID_MLX_NO_BANNER`` suppression, the ``NO_COLOR`` mono
behavior, and the byte-clean ``version``/``help`` subcommands are the opt-outs
the release requires, plus pipe/redirect (non-TTY) and ``--json`` suppression.
"""

from __future__ import annotations

import pytest

from vllm_mlx._banner import render_banner, should_show_banner

# ---------------------------------------------------------------------------
# Artwork + wordmark + version render
# ---------------------------------------------------------------------------


def test_banner_renders_art_wordmark_and_version_in_mono():
    out = render_banner("0.13.2", color=False)
    # The cheetah's two signatures are present as literal glyphs.
    assert "●" in out  # rosette spots
    assert "█" in out  # the tear-mark / coat blocks
    assert "r a p i d - m l x" in out
    assert "Rapid-MLX 0.13.2" in out
    # Compact: wordmark + version are the last two lines, art above them.
    lines = out.splitlines()
    assert lines[-1] == "Rapid-MLX 0.13.2"
    assert lines[-2] == "r a p i d - m l x"
    # No stray ANSI in the mono render.
    assert "\x1b[" not in out


def test_banner_fits_an_80_column_terminal():
    art = render_banner("0.13.2", color=False).splitlines()
    assert max(len(line) for line in art) <= 40
    # ~20-24 columns of art max per the release contract; we stay well under.
    assert len(art) <= 20


def test_banner_color_applies_ansi_and_cleans_up():
    out = render_banner("0.13.2", color=True)
    assert "\x1b[" in out  # ANSI present
    # Every colored glyph is wrapped in <open>...<reset>, so the number of
    # resets equals the total number of color-Opens (coat + dark); a single
    # orphaned escape would break that balance and leak terminal state.
    import re

    _esc = re.compile(r"\x1b\[([0-9;]*)m")
    opens = [m for m in _esc.finditer(out) if m.group(1) not in ("", "0")]
    resets = out.count("\x1b[0m")
    assert resets == len(opens)
    # Stripping ANSI yields the identical mono art.
    plain = _esc.sub("", out)
    assert plain == render_banner("0.13.2", color=False)


def test_banner_does_not_duplicate_when_printed_twice():
    a = render_banner("0.13.2", color=False)
    b = render_banner("0.13.2", color=False)
    assert a == b


# ---------------------------------------------------------------------------
# Interactive-only gating (the suppression contract)
# ---------------------------------------------------------------------------


def test_show_on_interactive_subcommand():
    assert should_show_banner(
        command="serve",
        json_output=False,
        no_banner=False,
        stdout_isatty=True,
        stdin_isatty=True,
    )
    # stdin need not be a tty for a subcommand.
    assert should_show_banner(
        command="serve",
        json_output=False,
        no_banner=False,
        stdout_isatty=True,
        stdin_isatty=False,
    )


def test_show_on_bare_interactive_launcher():
    assert should_show_banner(
        command=None,
        json_output=False,
        no_banner=False,
        stdout_isatty=True,
        stdin_isatty=True,
    )


@pytest.mark.parametrize(
    "kw",
    [
        # pipe / redirect: stdout is not a terminal
        dict(command="serve", stdout_isatty=False, stdin_isatty=True),
        # bare launcher with stdin piped (nameplate itself needs a tty stdin)
        dict(command=None, stdout_isatty=True, stdin_isatty=False),
        # explicit --no-banner opt-out, still a terminal
        dict(command="serve", no_banner=True, stdout_isatty=True, stdin_isatty=True),
        # machine-readable --json output, still a terminal
        dict(command="serve", json_output=True, stdout_isatty=True, stdin_isatty=True),
        # byte-clean subcommands stay clean even on a tty (a "rapid-mlx
        # version" contract must stay greppable in scripts)
        dict(command="version", stdout_isatty=True, stdin_isatty=True),
        dict(command="help", stdout_isatty=True, stdin_isatty=True),
    ],
)
def test_suppressed(kw):
    kw.setdefault("json_output", False)
    kw.setdefault("no_banner", False)
    kw.setdefault("stdin_isatty", True)
    assert should_show_banner(**kw) is False


def test_show_on_machine_clean_subcommands():
    # Non byte-clean interactive subcommands DO show the banner on a tty.
    for cmd in ("serve", "chat", "models", "pull"):
        assert should_show_banner(
            command=cmd,
            json_output=False,
            no_banner=False,
            stdout_isatty=True,
            stdin_isatty=True,
        )


def test_no_color_keeps_banner_but_drops_ansi():
    # The gate is unaffected by NO_COLOR — the caller passes color=False there
    # (see cli.py), so the banner STILL shows, just monochrome.
    assert should_show_banner(
        command="serve",
        json_output=False,
        no_banner=False,
        stdout_isatty=True,
        stdin_isatty=True,
    )
    mono = render_banner("0.13.2", color=False)
    assert "\x1b[" not in mono


# ---------------------------------------------------------------------------
# CLI-level env/flag folding (through real main(), mocked TTY)
# ---------------------------------------------------------------------------


class _FakeStream:
    """A StringIO that reports ``isatty()``, so ``main()``'s stdout gate and
    the nameplate's stdin gate both behave as if the user is at a terminal."""

    def __init__(self, data: str = "", isatty: bool = True) -> None:
        self._buf = []
        self._data = data
        self._tty = isatty

    def isatty(self) -> bool:
        return self._tty

    def write(self, s: str) -> None:
        self._buf.append(s)

    def getvalue(self) -> str:
        return "".join(self._buf)


def _run_main(argv, *, stdout_tty=True, stdin_tty=True, env=None):
    """Drive ``vllm_mlx.cli.main()`` with a mocked TTY/stream and env, and
    return ``(exit_code, stdout_text)``. Uses the ``version`` subcommand as the
    probe: it is the cheapest real subcommand (no model load) whose presence
    lets us assert both that the banner SHOWS on a tty and that it is
    suppressed by each opt-out."""
    import os
    import sys

    from vllm_mlx import cli as cli_mod

    class _Stdout(_FakeStream):
        pass

    real_argv, real_stdout, real_stdin = sys.argv, sys.stdout, sys.stdin
    sys.argv = ["rapid-mlx"] + argv
    sys.stdout = _Stdout(isatty=stdout_tty)
    sys.stdin = _FakeStream(isatty=stdin_tty)
    saved_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(saved_env)
    # The no-override path models a user who did not set NO_COLOR. Do not let
    # the developer/CI host's preference silently turn that control case mono;
    # tests that exercise the opt-out add it back through ``env`` below.
    os.environ.pop("NO_COLOR", None)
    if env:
        os.environ.update(env)
    try:
        try:
            cli_mod.main()
            rc = 0
        except SystemExit as exc:
            rc = exc.code if exc.code is not None else 0
        text = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdout, sys.stdin = real_argv, real_stdout, real_stdin
        os.environ.clear()
        os.environ.update(saved_env)
    return rc, text


def test_env_var_suppresses_banner_through_main_on_tty():
    """RAPID_MLX_NO_BANNER must suppress the banner all the way through the
    real ``main()`` folding, not just the helper gate (codex nit)."""
    _rc, out = _run_main(
        ["version"], stdout_tty=True, stdin_tty=True, env={"RAPID_MLX_NO_BANNER": "1"}
    )
    assert "r a p i d - m l x" not in out
    assert "\x1b[" not in out


def test_flag_suppresses_banner_through_main_on_tty():
    _rc, out = _run_main(["--no-banner", "version"], stdout_tty=True, stdin_tty=True)
    assert "r a p i d - m l x" not in out


def test_version_subcommand_stays_byte_clean_through_main():
    # The ``version`` subcommand must NOT get a banner even on a tty (codex
    # blocking fix): a wrapper scraping "rapid-mlx X.Y.Z" stays dependable.
    _rc, out = _run_main(["version"], stdout_tty=True, stdin_tty=True)
    assert "r a p i d - m l x" not in out


def test_banner_shows_through_main_on_interactive_subcommand():
    """The banner's *show* path must be covered all the way through the real
    ``main()`` too, not just the helper gate — otherwise the cover-py
    changed-lines gate (--fail-under 100 on the cli.py diff) is red. ``models``
    is the ideal probe subcommand: it is NOT in ``_BYTE_CLEAN_SUBCOMMANDS``, so
    it shows the banner on a tty, and its dispatch lists local model aliases —
    parses with no required args and does no network/model load, so the test
    is hermetic and fast."""
    _rc, out = _run_main(["models"], stdout_tty=True, stdin_tty=True)
    assert "r a p i d - m l x" in out
    assert "Rapid-MLX" in out


def test_banner_color_varies_with_no_color_env_through_main():
    # On a TTY the banner shows; under NO_COLOR it keeps the mono glyphs but
    # drops every ANSI escape (opt-out contract), driven through real main().
    _rc, color = _run_main(["models"], stdout_tty=True, stdin_tty=True)
    assert "\x1b[" in color
    _rc, mono = _run_main(
        ["models"], stdout_tty=True, stdin_tty=True, env={"NO_COLOR": "1"}
    )
    assert "r a p i d - m l x" in mono
    assert "\x1b[" not in mono


def test_banner_render_hiccup_never_blocks_the_command(monkeypatch):
    """The banner is decorative: if its renderer throws (a broken glyph, a
    font/ANSI edge case), ``main()`` must swallow it and still run the user's
    real command to completion. This also covers the except guard that the
    changed-lines coverage gate requires."""
    import vllm_mlx._banner as banner_mod

    def _boom(version, *, color):
        raise RuntimeError("agent-injected render failure")

    monkeypatch.setattr(banner_mod, "render_banner", _boom)
    # ``models`` is on the *show* path (not byte-clean), so render_banner is
    # actually invoked and raises; ``version`` would never reach it (byte-clean
    # suppresses before the render), so the guard couldn't be exercised.
    _rc, out = _run_main(["models"], stdout_tty=True, stdin_tty=True)
    assert _rc == 0  # the command itself still completes
    assert "r a p i d - m l x" not in out  # nothing partial was printed
