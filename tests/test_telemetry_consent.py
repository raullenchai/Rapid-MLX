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
    # Round 3: ``maybe_prompt_for_consent`` returns True when it just
    # collected first-time consent. The cli uses this to suppress the
    # SAME run's lifecycle emit (which captured pre-prompt argv).
    assert maybe_prompt_for_consent("serve") is True

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
    # Even a "no" answer counts as "just collected" for the lifecycle
    # skip — the contract is per-invocation, not per-yes.
    assert maybe_prompt_for_consent("serve") is True

    state = get_consent_state()
    assert state is not None
    assert state.consent is False
    out = capsys.readouterr().out
    assert "stays off" in out.lower()


def test_skip_paths_return_false(fake_home, monkeypatch, capsys):
    """Round 3 codex review wired the return type into a contract: the
    cli relies on it to skip the SAME run's lifecycle emit only when
    the prompt actually fired. Every skip path must return False so a
    cached-consent run still emits normally."""
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import record_consent

    # Cached consent → no prompt → False (lifecycle emit must run).
    record_consent(True, rapid_mlx_version="0.0.0+test")
    assert maybe_prompt_for_consent("serve") is False

    # CLI override skip.
    assert maybe_prompt_for_consent("serve", cli_no_telemetry=True) is False

    # Non-interactive subcommand skip.
    assert maybe_prompt_for_consent("version") is False


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


def test_post_record_oserror_still_reports_just_collected(
    fake_home, monkeypatch, capsys
):
    """Round 14 codex review: after ``record_consent(True)`` had
    already succeeded, an ``OSError`` from a subsequent print (e.g.
    SIGPIPE from a closed parent pipe) flipped the return value back
    to ``False``. The CLI then treated the run as "not just collected"
    and emitted same-run ``session_start`` / ``session_end`` events
    for the argv that ran BEFORE the disclosure — directly violating
    the disclosure's "nothing from before this prompt" promise.

    Pin: once consent is persisted, the return value is True even if
    one of the chatter prints raises OSError."""
    from vllm_mlx.telemetry import consent as consent_mod
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "n")

    # Make the opt-out chatter path raise OSError (this print runs
    # AFTER record_consent has persisted the decision). The pre-record
    # prints are unaffected — they go through the normal stdout.
    def _explode():
        raise OSError("simulated SIGPIPE from closed parent pipe")

    monkeypatch.setattr(consent_mod, "consent_path", _explode)

    # Despite the OSError, the function must report that consent was
    # just collected so cli.py suppresses same-run lifecycle emit.
    assert maybe_prompt_for_consent("serve") is True

    # And the consent is genuinely on disk.
    state = get_consent_state()
    assert state is not None
    assert state.consent is False


def test_pre_record_oserror_returns_false(fake_home, monkeypatch, capsys):
    """Companion to ``test_post_record_oserror_still_reports_just_collected``:
    an OSError BEFORE ``record_consent`` succeeds must keep returning
    ``False``. Otherwise we'd skip the lifecycle emit for an invocation
    where consent was never actually collected — same disclosure
    violation in the opposite direction."""
    from vllm_mlx.telemetry import consent as consent_mod
    from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
    from vllm_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")

    # Make ``client_id_path`` (called inside the pre-prompt disclosure
    # print) raise OSError. Reaches the outer except BEFORE
    # record_consent runs, so just_collected stays False.
    def _explode():
        raise OSError("simulated stdout-closed during disclosure")

    monkeypatch.setattr(consent_mod, "client_id_path", _explode)

    assert maybe_prompt_for_consent("serve") is False
    # And nothing was persisted.
    assert get_consent_state() is None


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
