# SPDX-License-Identifier: Apache-2.0
"""Pin the first-run consent prompt's skip rules.

If any of these guards regresses, we either nag users (prompting on
non-interactive subcommands or re-prompting after they answered) OR
silently never ask (skipping when we shouldn't, leaving telemetry
permanently off without disclosure). Both are bugs the original issue
explicitly calls out as deal-breakers.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    return tmp_path


def _stub_tty(monkeypatch, *, in_=True, out=True):
    """Force ``sys.stdin.isatty()`` / ``sys.stdout.isatty()`` regardless
    of the actual test runner environment (CI, tmux, etc.)."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: in_)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: out)


def test_skips_when_consent_already_recorded(fake_home, monkeypatch, capsys):
    """User already answered — never re-prompt, even on later interactive runs."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import record_consent

    record_consent(False, rapid_mlx_version="0.6.33")
    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


def test_skips_when_env_var_set(fake_home, monkeypatch, capsys):
    """Env var already governs state — no need to ask."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


def test_skips_when_cli_no_telemetry(fake_home, monkeypatch, capsys):
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve", cli_no_telemetry=True)
    assert capsys.readouterr().out == ""


def test_skips_when_stdin_not_tty(fake_home, monkeypatch, capsys):
    """Pipes / CI / daemons must NOT see a prompt that hangs the process."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch, in_=False)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "subcommand",
    ["version", "help", "models", "ps", "info", "telemetry"],
)
def test_skips_for_non_interactive_subcommands(
    fake_home, monkeypatch, capsys, subcommand
):
    """One-shot info commands must stay quiet (and grep-friendly)."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent(subcommand)
    assert capsys.readouterr().out == ""


def test_skips_when_subcommand_none(fake_home, monkeypatch, capsys):
    """`rapid-mlx` with no subcommand prints help — no prompt."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent(None)
    assert capsys.readouterr().out == ""


def test_yes_records_consent_true(fake_home, monkeypatch, capsys):
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    out = capsys.readouterr().out
    # The post-opt-in confirmation must thank the user and surface the
    # disable/audit affordances so they always know how to revisit.
    lower = out.lower()
    assert "thank you" in lower
    assert "rapid-mlx telemetry disable" in lower


def test_no_records_consent_false(fake_home, monkeypatch, capsys):
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "n")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.consent is False
    out = capsys.readouterr().out
    assert "stays off" in out.lower()


def test_empty_answer_defaults_to_no(fake_home, monkeypatch):
    """Pressing enter at the y/N prompt defaults to N — same as Ollama,
    same as Homebrew's modern prompt. The disclosure says ``y/N`` so the
    default has to be N or it's a lie."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.consent is False


def test_eof_during_prompt_does_not_record(fake_home, monkeypatch, capsys):
    """Ctrl-D mid-prompt is "I changed my mind" — must NOT record a
    silent No that prevents future prompting."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)

    def boom():
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    maybe_prompt_for_consent("serve")
    # No consent recorded → next interactive run will re-prompt.
    assert get_consent_state() is None


def test_records_prompted_version_correctly(fake_home, monkeypatch):
    """``status`` shows users when they were prompted and on which
    version. If we record the wrong version, future schema-version
    bumps can't reliably re-prompt only stale users."""
    from vllm_mlx import __version__ as actual_version
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.prompted_version == actual_version


def test_unwritable_home_does_not_crash_cli(fake_home, monkeypatch, capsys):
    """If recording consent fails (read-only home, disk full, etc.), the
    CLI must NOT crash — telemetry consent is never a reason for the
    user's actual subcommand to fail."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")

    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    # Make every path operation in record_consent fail.
    monkeypatch.setattr(
        "vllm_mlx.telemetry.consent.record_consent",
        lambda *a, **kw: (_ for _ in ()).throw(boom()),
    )
    # Should swallow the error and return cleanly.
    maybe_prompt_for_consent("serve")
