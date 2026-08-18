# SPDX-License-Identifier: Apache-2.0
"""Pin that `_test_e2e_chat` cannot pass on an agent that never answered (#1981).

The e2e chat probe asked "What is 2+2?" and accepted a bare ``4`` anywhere in
the agent CLI's stdout+stderr, while `_agent_query` never looked at the child's
exit status. A CLI that never reached the server therefore reported PASS:

    dsh: request failed: connect ECONNREFUSED 127.0.0.1:8477   ->  PASS
                                                       ^ this "4"

An HTTP 404, a timestamp, a token count or a version string does it just as
well, on all 13 agent profiles, including the release gate's
``bench --tier harness`` sweep.

Both halves of the fix need their own coverage, because neither is sufficient:

* **the assertion** — dsh exits 0 even when it fails outright (its profile's
  ``known_issues`` records this), so for that CLI the exit status carries no
  signal at all and only the expected answer can decide;
* **the exit status** — an agent whose failure text happens to contain the
  expected token would otherwise still pass, and an evidence-less run deserves
  to be reported as the launch failure it is rather than as a wrong answer.

And the fix must not overshoot: an agent that produced the evidence and *then*
exited non-zero has still demonstrated the capability (#1598 established the
same for an agent that never terminates), so the sibling file-read and terminal
probes are exercised here too.

Everything below drives a REAL subprocess. The bug is at the subprocess
boundary — an unread ``returncode`` — and a mock asserting we read the
attribute would only restate the diff.
"""

from __future__ import annotations

import shlex
import sys

from vllm_mlx.agents.testing import (
    E2E_CHAT_EXPECTED,
    E2E_CHAT_QUERY,
    E2E_FIRST_LINE,
    TestStatus,
    _agent_query,
    _err_to_status,
    _test_e2e_chat,
    _test_e2e_file_read,
    _test_e2e_terminal,
    _test_plain_chat,
)

# Long enough that a slow machine never flakes, short enough that a regression
# fails the suite in seconds instead of hanging it.
_TIMEOUT_S = 8

# The proven reproducer from #1981, verbatim: the whole of what a dsh that
# cannot reach the server prints. Its only "4"s live in the port number.
CONNECT_REFUSED = "dsh: request failed: connect ECONNREFUSED 127.0.0.1:8477\n"

# The other shape, and the reason an exit-code check alone is not the fix:
# dsh returns 0 here. The "4"s come from an HTTP status and a version string.
NO_ADAPTER = (
    'NO_ADAPTER: no adapter registered for provider "rapid-mlx" '
    "(dsh 0.1.0-rc.4, HTTP 404)\n"
)


def _fake_cli(stdout: str = "", stderr: str = "", exit_code: int = 0) -> str:
    """A stand-in agent CLI with a scripted answer and exit status."""
    script = (
        "import sys; "
        f"sys.stdout.write({stdout!r}); "
        f"sys.stderr.write({stderr!r}); "
        f"sys.exit({exit_code})"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"


def _hanging_cli(stdout: str = "", stderr: str = "") -> str:
    """A CLI that prints, flushes, and then never terminates."""
    script = (
        "import sys, time; "
        f"sys.stdout.write({stdout!r}); sys.stdout.flush(); "
        f"sys.stderr.write({stderr!r}); sys.stderr.flush(); "
        "time.sleep(30)"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"


def _echoing_cli(exit_code: int = 1) -> str:
    """A CLI that prints the prompt back and never answers it."""
    script = "import sys; sys.stdout.write('> ' + sys.argv[1] + chr(10)); "
    script += f"sys.exit({exit_code})"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"


# --------------------------------------------------------------------------- #
# The reported false green                                                    #
# --------------------------------------------------------------------------- #


def test_e2e_chat_fails_when_the_cli_never_reached_the_server():
    """The #1981 reproducer: a connection refusal must not read as an answer."""
    result = _test_e2e_chat(
        sys.executable, _fake_cli(stderr=CONNECT_REFUSED, exit_code=1), _TIMEOUT_S
    )

    assert result.status is not TestStatus.PASS, (
        "an agent CLI that never reached the server reported PASS — the port "
        "number in its error text satisfied the assertion"
    )
    assert result.status is TestStatus.ERROR, (
        f"a CLI that exited non-zero without answering is a launch failure, "
        f"not a wrong answer; got {result.status} / {result.message!r}"
    )
    assert "ECONNREFUSED" in result.message, (
        f"the report must name what actually went wrong: {result.message!r}"
    )


def test_e2e_chat_fails_on_a_zero_exit_failure_whose_text_contains_a_four():
    """The half an exit-code check cannot cover: dsh fails and still returns 0.

    If this test can be made to pass by looking at ``returncode``, the fix is
    in the wrong place — for dsh the exit status is a constant.
    """
    result = _test_e2e_chat(
        sys.executable, _fake_cli(stdout=NO_ADAPTER, exit_code=0), _TIMEOUT_S
    )

    assert result.status is TestStatus.FAIL, (
        f"a provider error that never produced an answer passed the chat probe; "
        f"got {result.status} / {result.message!r}"
    )


def test_e2e_chat_rejects_the_expected_answer_inside_a_longer_number():
    """A timestamp that merely contains the digits is not an answer."""
    result = _test_e2e_chat(
        sys.executable,
        _fake_cli(stderr="[1777777923] dsh: request failed\n", exit_code=0),
        _TIMEOUT_S,
    )

    assert result.status is TestStatus.FAIL, (
        f"the expected answer was accepted as a fragment of a longer number — "
        f"exactly the failure mode being fixed; got {result.status}"
    )


def test_a_failed_run_gets_no_credit_for_what_it_printed_on_stderr():
    """The same false green from the other side (codex review, round 1).

    Evidence beats a non-zero exit — so a failure diagnostic that happens to
    quote the expected token would buy a PASS if diagnostics counted as
    evidence. They do not: a process that reported failure is credited only
    with what it wrote to stdout, where every profile's CLI writes its answer.
    """
    result = _test_e2e_chat(
        sys.executable,
        _fake_cli(
            stderr=f"ERROR: expected {E2E_CHAT_EXPECTED}; request failed\n",
            exit_code=1,
        ),
        _TIMEOUT_S,
    )

    assert result.status is TestStatus.ERROR, (
        f"a failed run passed because its own error message mentioned the "
        f"expected answer; got {result.status} / {result.message!r}"
    )


def test_a_failed_run_gets_no_credit_for_a_sentinel_on_stderr():
    """Same rule for the file-read probe — one place decides, not two."""
    file_read = _test_e2e_file_read(
        sys.executable,
        _fake_cli(stderr=f"ERROR: never found {E2E_FIRST_LINE}\n", exit_code=1),
        _TIMEOUT_S,
    )

    assert file_read.status is TestStatus.ERROR, file_read.message


def test_a_hung_run_gets_no_credit_for_what_it_printed_on_stderr():
    """The timeout carve-out obeys the same rule (codex review, round 2).

    ``TIMEOUT`` is the other err that loses to evidence, so if a hung CLI's
    stderr counted, the stderr-diagnostic false green would simply move here.
    """
    chat = _test_e2e_chat(
        sys.executable,
        _hanging_cli(stderr=f"ERROR: expected {E2E_CHAT_EXPECTED}\n"),
        timeout=1,
    )
    file_read = _test_e2e_file_read(
        sys.executable,
        _hanging_cli(stderr=f"ERROR: never found {E2E_FIRST_LINE}\n"),
        timeout=1,
    )

    assert chat.status is TestStatus.ERROR, (
        f"a hung run passed on its own error text: {chat.message!r}"
    )
    assert file_read.status is TestStatus.ERROR, (
        f"a hung run passed on its own error text: {file_read.message!r}"
    )


def test_a_hung_run_still_passes_on_evidence_it_printed_on_stdout():
    """...and #1598's carve-out itself must survive that narrowing."""
    result = _test_e2e_chat(
        sys.executable, _hanging_cli(stdout=f"{E2E_CHAT_EXPECTED}\n"), timeout=1
    )

    assert result.status is TestStatus.PASS, result.message
    assert "did not terminate" in result.message, result.message


def test_e2e_chat_rejects_a_malformed_digit_grouping():
    """Separator tolerance must not manufacture the answer out of other digits.

    Deleting every separator that sat between two digits also turned "7777 77"
    into the expected sum — a fresh way to pass without answering.
    """
    for reply in ("7777 77\n", "77 7777\n", "7,77777\n", "1,777777\n", "777777,000\n"):
        result = _test_e2e_chat(sys.executable, _fake_cli(stdout=reply), _TIMEOUT_S)
        assert result.status is TestStatus.FAIL, (
            f"{reply.strip()!r} was normalized into the expected answer: "
            f"{result.status}"
        )
    # Real thousands grouping still counts.
    for reply in ("777 777\n", "777,777\n"):
        ok = _test_e2e_chat(sys.executable, _fake_cli(stdout=reply), _TIMEOUT_S)
        assert ok.status is TestStatus.PASS, f"{reply.strip()!r}: {ok.message}"


def test_e2e_chat_rejects_a_signed_or_fractional_near_miss():
    """ "-777777" and "777777.5" are different numbers, not sloppy right ones."""
    for reply in ("-777777\n", "777777.5\n", "12.777777\n"):
        result = _test_e2e_chat(sys.executable, _fake_cli(stdout=reply), _TIMEOUT_S)
        assert result.status is TestStatus.FAIL, (
            f"{reply.strip()!r} was accepted as the answer to "
            f"{E2E_CHAT_QUERY!r}: {result.status}"
        )
    # ...while a sentence-ending period is punctuation, not a decimal point.
    ok = _test_e2e_chat(
        sys.executable, _fake_cli(stdout="The answer is 777777.\n"), _TIMEOUT_S
    )
    assert ok.status is TestStatus.PASS, ok.message


def test_e2e_chat_is_not_satisfied_by_a_cli_echoing_the_prompt():
    """Why the expected value is derived, not a sentinel handed to the agent.

    A sentinel word in the prompt would also be un-guessable from an error
    string — and would be echoed straight back by any CLI that prints the
    prompt it was given (``codex exec`` does), which is the same false green
    wearing a different hat.
    """
    assert E2E_CHAT_EXPECTED not in E2E_CHAT_QUERY, (
        "the expected answer appears in the prompt; every prompt-echoing CLI "
        "now passes the chat probe without answering"
    )

    result = _test_e2e_chat(sys.executable, _echoing_cli(exit_code=1), _TIMEOUT_S)

    assert result.status is not TestStatus.PASS, (
        f"a CLI that only echoed the prompt passed: {result.message!r}"
    )


# --------------------------------------------------------------------------- #
# ...without breaking the honest cases                                        #
# --------------------------------------------------------------------------- #


def test_e2e_chat_passes_on_a_correct_answer():
    """The guard has to let the working case through, or it is just a red X."""
    result = _test_e2e_chat(
        sys.executable, _fake_cli(stdout=f"{E2E_CHAT_EXPECTED}\n"), _TIMEOUT_S
    )

    assert result.status is TestStatus.PASS, result.message


def test_e2e_chat_accepts_a_digit_grouped_answer():
    """Models write long numbers with separators; that is still the answer."""
    result = _test_e2e_chat(
        sys.executable,
        _fake_cli(stdout="123456 + 654321 = 777,777\n"),
        _TIMEOUT_S,
    )

    assert result.status is TestStatus.PASS, result.message


def test_e2e_chat_passes_when_the_agent_answered_then_exited_nonzero():
    """Exit status describes how the process ended, not whether it worked.

    Some CLIs answer and then exit non-zero on their way out. Making a
    non-zero exit fatal would turn those into a red release gate, so evidence
    wins — the exit status only decides what an evidence-LESS run is called.

    The answer is on stdout and the noise on stderr, which is the split
    `_agent_query` relies on: see the stderr-diagnostic test above for the
    other half of this rule.
    """
    result = _test_e2e_chat(
        sys.executable,
        _fake_cli(
            stdout=f"{E2E_CHAT_EXPECTED}\n", stderr="warning: session\n", exit_code=3
        ),
        _TIMEOUT_S,
    )

    assert result.status is TestStatus.PASS, (
        f"an agent that answered correctly was failed for its exit code: "
        f"{result.message!r}"
    )
    assert "exited non-zero" in result.message, (
        f"the non-zero exit must still be reported on the PASS: {result.message!r}"
    )


def test_file_read_still_passes_when_the_cli_exits_nonzero_with_the_sentinel():
    """The exit-code signal must not regress the already-correct siblings."""
    result = _test_e2e_file_read(
        sys.executable, _fake_cli(stdout=f"{E2E_FIRST_LINE}\n", exit_code=1), _TIMEOUT_S
    )

    assert result.status is TestStatus.PASS, result.message


def test_terminal_keeps_a_broken_run_fatal_because_its_marker_is_in_the_prompt():
    """The terminal probe gets no evidence carve-out, and that is on purpose.

    Its marker is handed to the agent in the prompt ("Run 'echo <marker>'"),
    so a CLI that echoes the prompt on its way out prints the marker without
    ever opening a shell. Letting that overrule a crash would be the #1981
    false green again, one probe over (codex review, round 4).
    """
    marker = "rapidmlx_codex_test"
    echoed = _test_e2e_terminal(
        sys.executable, _echoing_cli(exit_code=1), _TIMEOUT_S, "codex"
    )
    printed = _test_e2e_terminal(
        sys.executable,
        _fake_cli(stdout=f"{marker}\n", exit_code=1),
        _TIMEOUT_S,
        "codex",
    )

    assert echoed.status is TestStatus.ERROR, (
        f"a CLI that echoed the prompt and died passed the terminal probe: "
        f"{echoed.message!r}"
    )
    assert printed.status is TestStatus.ERROR, (
        f"a broken run must stay fatal for the terminal probe: {printed.message!r}"
    )
    # ...while a healthy run that echoes the marker is still a PASS.
    healthy = _test_e2e_terminal(
        sys.executable, _fake_cli(stdout=f"{marker}\n"), _TIMEOUT_S, "codex"
    )
    assert healthy.status is TestStatus.PASS, healthy.message


def test_file_read_reports_the_launch_failure_instead_of_a_wrong_answer():
    """No evidence and a non-zero exit is an ERROR, not a FAIL, everywhere."""
    result = _test_e2e_file_read(
        sys.executable, _fake_cli(stderr=CONNECT_REFUSED, exit_code=1), _TIMEOUT_S
    )

    assert result.status is TestStatus.ERROR, (
        f"got {result.status} / {result.message!r}"
    )


# --------------------------------------------------------------------------- #
# The API-level sibling                                                       #
# --------------------------------------------------------------------------- #


def _plain_chat_verdict(content: str, monkeypatch) -> TestStatus:
    """Grade `content` as `_test_plain_chat` would, without a live server."""
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._api_call",
        lambda *_a, **_k: {"choices": [{"message": {"content": content}}]},
    )
    return _test_plain_chat("http://localhost:8000/v1", "model").status


def test_plain_chat_wants_the_number_four_not_the_digit(monkeypatch):
    """ "1234", "0.4", "-4" and "4.5" are not answers to "what is 2+2".

    Same family as the e2e false green: grading a digit rather than a number
    lets an unrelated value satisfy the assertion (codex review, round 3).
    """
    for wrong in ("1234", "0.4", "-4", "4.5", "The id is a4b.", "4,000"):
        assert _plain_chat_verdict(wrong, monkeypatch) is TestStatus.FAIL, (
            f"{wrong!r} was accepted as the answer to 2+2"
        )


def test_plain_chat_still_accepts_a_real_answer(monkeypatch):
    for right in ("4", "4.", "2+2 = 4", "The answer is **4**"):
        assert _plain_chat_verdict(right, monkeypatch) is TestStatus.PASS, (
            f"{right!r} was rejected — plain_chat must stay the easy control"
        )


# --------------------------------------------------------------------------- #
# The err sentinel itself                                                     #
# --------------------------------------------------------------------------- #


def test_agent_query_keeps_the_output_alongside_the_exit_err():
    """The err is soft: callers must still be able to grade what was printed."""
    out, err = _agent_query(
        sys.executable,
        _fake_cli(stdout="partial work\n", exit_code=2),
        "unused",
        timeout=_TIMEOUT_S,
    )

    assert out is not None and "partial work" in out, (
        "dropping the output on a non-zero exit would break every "
        "evidence-beats-exit-status grader"
    )
    assert err is not None and err.startswith("EXIT:2"), err


def test_agent_query_reports_a_clean_exit_as_before():
    out, err = _agent_query(
        sys.executable, _fake_cli(stdout="ok\n"), "unused", timeout=_TIMEOUT_S
    )

    assert err is None, f"a clean run must stay err-free: {err!r}"
    assert out == "ok\n"


def test_exit_err_is_not_misrouted_to_skip_by_the_childs_own_words():
    """The EXIT err carries the child's text, and that text is not ours.

    ``_err_to_status`` maps any err containing "not found" to SKIP (the
    binary-missing case). A child that printed "command not found" would
    otherwise vanish from the report as a skip.
    """
    err = "EXIT:127 agent CLI exited non-zero — dsh: command not found"

    assert _err_to_status(err) is TestStatus.ERROR, (
        "a failed agent run was reported as a skip because its own error text "
        "said 'not found'"
    )
