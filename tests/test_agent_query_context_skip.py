# SPDX-License-Identifier: Apache-2.0
"""Pin the soft-skip behavior when a harness refuses init on context window.

Regression coverage for issue #655: the v0.7.27 release SOP's M3 gauntlet
G7b aborted because Hermes Agent refuses to initialize on Qwen3.5-9B-4bit
(advertised context window 32K, Hermes minimum 64K). The subprocess
exited 0 and wrote the refusal to stdout, so before this fix:

  - ``_agent_query`` returned ``(output, None)``
  - ``_test_e2e_file_read`` saw "no expected substring" and reported FAIL
  - The harness sweep ended 27p 7f 0e 0s and ``set -e`` aborted the gauntlet

After the fix, ``_agent_query`` detects the refusal pattern and propagates
a ``SKIP:``-prefixed err sentinel; the three ``_test_e2e_*`` callers
honor that prefix and report SKIP instead of FAIL. The harness sweep
stays honest (0 FAIL, N SKIP) and the gauntlet continues to the
remaining gates.
"""

from __future__ import annotations

from unittest.mock import patch

from vllm_mlx.agents.testing import (
    TestStatus,
    _agent_query,
    _err_to_status,
    _test_e2e_chat,
    _test_e2e_file_read,
    _test_e2e_terminal,
)


def _stub_completed_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Build a stand-in for the subprocess.run() result."""

    class _Stub:
        def __init__(self):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    return _Stub()


# --------------------------------------------------------------------------- #
# _err_to_status: the three-bucket mapping                                    #
# --------------------------------------------------------------------------- #


def test_err_to_status_not_found_is_skip():
    assert _err_to_status("Binary not found: hermes") is TestStatus.SKIP


def test_err_to_status_skip_prefix_is_skip():
    assert (
        _err_to_status("SKIP: agent refused init — context window") is TestStatus.SKIP
    )


def test_err_to_status_timeout_is_error():
    assert _err_to_status("TIMEOUT") is TestStatus.ERROR


def test_err_to_status_arbitrary_is_error():
    assert _err_to_status("Agent error: server issue") is TestStatus.ERROR


# --------------------------------------------------------------------------- #
# _agent_query: the detection happens once at the subprocess boundary         #
# --------------------------------------------------------------------------- #


HERMES_REFUSAL_STDOUT = (
    "Failed to initialize agent: Model mlx-community/Qwen3.5-9B-4bit has a "
    "context window of 32,768 tokens, which is below the minimum 64,000 "
    "required by Hermes Agent.\n"
)

# Exactly the bytes the real hermes binary prints — it hard-wraps at
# ~100 cols so the phrase "context window" ends up split as
# "context\nwindow". A literal substring check misses this case, which
# is what slipped through #659 round-1 and re-broke the gauntlet. Keep
# this fixture verbatim so any future regression of the detection
# layer's whitespace tolerance trips the test, not the user.
HERMES_REFUSAL_STDOUT_WRAPPED = (
    "Failed to initialize agent: Model mlx-community/Qwen3.5-9B-4bit has a context\n"
    "window of 32,768 tokens, which is below the minimum 64,000 required by Hermes\n"
    "Agent.  Choose a model with at least 64K context.\n"
)


def test_agent_query_detects_context_refusal_as_skip():
    """The Hermes-style "context window below minimum" refusal → SKIP err."""
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=HERMES_REFUSAL_STDOUT),
        ),
    ):
        out, err = _agent_query(
            "hermes", "hermes chat -q '{query}' -Q", "hi", timeout=10
        )
    assert out is None, (
        "On refusal, output must be suppressed so downstream tests route via err"
    )
    assert err is not None
    assert err.startswith("SKIP:"), (
        f"Refusal must be propagated with a SKIP: prefix so _test_e2e_* "
        f"recognize it; got err={err!r}"
    )
    # The actionable bits — model name and the actual vs required token
    # counts — must survive into the SKIP message (codex NIT #659).
    # Without them, the user has to dig in the server log to learn what
    # the harness wanted vs what the model offered.
    assert "Qwen3.5-9B-4bit" in err, (
        f"SKIP message must carry the model name; got err={err!r}"
    )
    assert "32,768" in err and "64,000" in err, (
        f"SKIP message must carry the advertised vs minimum context values; "
        f"got err={err!r}"
    )


def test_agent_query_detects_context_refusal_when_phrase_is_line_wrapped():
    """Direct regression for the #659 round-1 verify-pass finding.

    The real hermes binary hard-wraps stderr at ~100 cols, so the
    phrase "context window" arrives split as "context\\nwindow". The
    initial #659 fix used a literal ``"context window" in output``
    substring check that missed this — gauntlet hermes line was still
    ``27p 7f 0e 0s`` after the merge. Detection must collapse
    whitespace first so the wrapped form is recognized.
    """
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=HERMES_REFUSAL_STDOUT_WRAPPED),
        ),
    ):
        out, err = _agent_query(
            "hermes", "hermes chat -q '{query}' -Q", "hi", timeout=10
        )
    assert out is None
    assert err is not None and err.startswith("SKIP:"), (
        f"Wrapped-phrase refusal must STILL be detected; got err={err!r}"
    )
    # And the model name + numbers still survive into the message after
    # whitespace normalization.
    assert "Qwen3.5-9B-4bit" in err, f"model name lost; err={err!r}"
    assert "32,768" in err and "64,000" in err, f"context numbers lost; err={err!r}"


def test_agent_query_passes_through_normal_output():
    """A clean subprocess success returns ``(output, None)`` as before."""
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/codex"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout="The answer is 4.\n"),
        ),
    ):
        out, err = _agent_query(
            "codex", "codex -q '{query}'", "what is 2+2?", timeout=10
        )
    assert err is None
    assert out is not None and "4" in out


def test_agent_query_does_not_match_unrelated_failures():
    """\"Failed to initialize agent\" alone (no \"context window\") is NOT a SKIP.

    Other init failures (missing API key, bad config, etc) should remain
    ERROR — auto-skipping would mask real harness regressions.
    """
    bad_init_no_context = (
        "Failed to initialize agent: missing OPENAI_API_KEY environment variable\n"
    )
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=bad_init_no_context),
        ),
    ):
        out, err = _agent_query(
            "hermes", "hermes chat -q '{query}' -Q", "hi", timeout=10
        )
    # Output passes through (not None); downstream tests then route via
    # "no expected substring" → FAIL, which is the correct signal for a
    # genuine harness misconfiguration.
    assert err is None
    assert out is not None and "OPENAI_API_KEY" in out


# --------------------------------------------------------------------------- #
# Each of the three _test_e2e_* functions must route SKIP-prefixed err → SKIP #
# --------------------------------------------------------------------------- #


def test_e2e_chat_routes_skip_prefix_to_skip_status():
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=HERMES_REFUSAL_STDOUT),
        ),
    ):
        result = _test_e2e_chat("hermes", "hermes chat -q '{query}' -Q", timeout=10)
    assert result.status is TestStatus.SKIP, (
        f"e2e_chat must SKIP on Hermes context-window refusal (#655), not FAIL/ERROR. "
        f"Got status={result.status}, message={result.message!r}"
    )


def test_e2e_file_read_routes_skip_prefix_to_skip_status():
    """Direct regression for the gauntlet line in #655."""
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=HERMES_REFUSAL_STDOUT),
        ),
    ):
        result = _test_e2e_file_read(
            "hermes", "hermes chat -q '{query}' -Q", timeout=10
        )
    assert result.status is TestStatus.SKIP
    assert "context window" in (result.message or "").lower() or "context window" in (
        result.message or ""
    )


def test_e2e_terminal_routes_skip_prefix_to_skip_status():
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=_stub_completed_proc(stdout=HERMES_REFUSAL_STDOUT),
        ),
    ):
        result = _test_e2e_terminal(
            "hermes", "hermes chat -q '{query}' -Q", timeout=10, agent_name="hermes"
        )
    assert result.status is TestStatus.SKIP


def test_e2e_tests_still_error_on_genuine_failure():
    """A real subprocess crash must still be ERROR, not SKIP — guards against
    over-broad SKIP routing masking regressions."""
    with (
        patch("vllm_mlx.agents.testing.shutil.which", return_value="/fake/hermes"),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            side_effect=TimeoutError("simulated timeout"),
        ),
    ):
        result = _test_e2e_file_read("hermes", "hermes chat -q '{query}' -Q", timeout=1)
    # TimeoutError raised inside _agent_query → caught by the bare ``except
    # Exception`` and turned into err=str(exc); the resulting err is NOT
    # "not found" and does NOT start with "SKIP:", so the test routes to
    # ERROR — which is the correct signal for a genuine subprocess failure.
    assert result.status is TestStatus.ERROR, (
        f"A genuine subprocess failure must remain ERROR (not SKIP). "
        f"Got {result.status}, message={result.message!r}"
    )
