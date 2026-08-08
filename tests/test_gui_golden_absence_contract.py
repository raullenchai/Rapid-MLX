# SPDX-License-Identifier: Apache-2.0
"""A golden flow may only claim something is ABSENT from an observation it has.

``gui-golden-flows.sh`` proves that things are gone: that Settings closed, that
no video-generation alias reached a chat surface. Written as

    jq -e '[.data.ui_elements[]? | select(...)] | length == 0'

that claim is also satisfied by never having looked. ``rapid-ax`` walks the
accessibility tree with three silent ways to fall short of a full inventory — an
``AXChildren`` read that fails, the depth cap, the record cap — and each removes
a subtree while the dump still says ``success: true``.

It matters because the flow it guards is the one standing between users and
#1603: eight video-generation aliases reaching the picker and dead-ending at
"Couldn't start … Try again" *after* a download of up to 64 GB. A test that can
pass without looking is not a guard against that returning.

The fix is a completeness signal (``data.walk.complete``) the assertion gates
on, and a helper that refuses to answer while it is false. These tests pin the
helper's three-outcome contract, and lint the flows so the raw idiom cannot come
back — it had already been copied to a third site (#1673) before it was fixed
once.

Pure bash + jq, no GUI, no Swift, no GPU: the functions are extracted from the
real script so a copy here cannot drift away from what actually runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLOWS = _REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "gui-golden-flows.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="needs bash and jq, which the golden flows require anyway",
)


def _extract(name: str) -> str:
    """Pull one shell function out of the flows script, verbatim."""
    source = _FLOWS.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}$", source, re.MULTILINE | re.DOTALL
    )
    assert match, f"{name}() not found in {_FLOWS} — did it get renamed?"
    return match.group(0)


# The filter every catalog assertion uses, in the "does it match?" polarity the
# helper expects.
PRESENT_FILTER = (
    '[.data.ui_elements[]? | select(.identifier == "fake-video-alias")] | length > 0'
)


def _dump(elements, *, complete=True, success=True, reasons=None, walk=True):
    payload = {
        "success": success,
        "data": {
            "pid": 4242,
            "ui_elements": elements,
            "windows": {"titles": ["Rapid-MLX"], "complete": True},
        },
    }
    if walk:
        payload["data"]["walk"] = {
            "complete": complete,
            "scope": "window-forest",
            "reasons": reasons or [],
        }
    return payload


def _match(tmp_path, payload, filter_=PRESENT_FILTER) -> int:
    """Exit status of ax_elements_match against ``payload``."""
    dump = tmp_path / "dump.json"
    dump.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    script = textwrap.dedent(
        f"""
        set -uo pipefail
        {_extract("ax_elements_match")}
        ax_elements_match "$1" "$2"
        """
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", str(dump), filter_],
        capture_output=True,
        text=True,
    ).returncode


# ---------------------------------------------------------------------------
# The three outcomes. Folding the third into "absent" is the whole bug.
# ---------------------------------------------------------------------------


def test_a_match_in_a_complete_dump_is_present(tmp_path):
    assert _match(tmp_path, _dump([{"identifier": "fake-video-alias"}])) == 0


def test_no_match_in_a_complete_dump_is_absent(tmp_path):
    assert _match(tmp_path, _dump([{"identifier": "rapid.chat.compose"}])) == 1


def test_an_incomplete_walk_cannot_answer(tmp_path):
    """The case the whole change exists for: nothing matched, but a subtree is
    missing, so "nothing matched" is not an observation of absence."""
    payload = _dump(
        [{"identifier": "rapid.chat.compose"}],
        complete=False,
        reasons=["AXChildren was unreadable on 1 element(s) (last AXError -25204)"],
    )
    assert _match(tmp_path, payload) == 2


def test_a_dump_without_a_walk_signal_cannot_answer(tmp_path):
    """An older driver, or one whose output shape drifted, proves nothing."""
    assert _match(tmp_path, _dump([{"identifier": "x"}], walk=False)) == 2


def test_an_unsuccessful_dump_cannot_answer(tmp_path):
    assert _match(tmp_path, _dump([], success=False)) == 2


def test_ui_elements_that_is_not_an_array_cannot_answer(tmp_path):
    """``[]?`` swallows a structural failure, so without the type check a
    malformed dump reads as a confident "absent"."""
    payload = _dump([])
    payload["data"]["ui_elements"] = {"oops": True}
    assert _match(tmp_path, payload) == 2


def test_unparseable_json_cannot_answer(tmp_path):
    assert _match(tmp_path, "<html>not a dump</html>") == 2


def test_a_broken_query_cannot_answer(tmp_path):
    """jq exits 1 for "false" and 3 for "does not compile". Only the first is an
    answer; treating the second as absence is how a typo becomes a green test."""
    assert _match(tmp_path, _dump([]), filter_=".data.ui_elements[") == 2


def test_an_empty_element_array_cannot_answer(tmp_path):
    """A complete dump always holds at least the application record, so an empty
    array is not an app with nothing in it — it is a dump that is not one of
    ours. Reading it as absence is outcome 2 collapsing into outcome 1."""
    assert _match(tmp_path, _dump([])) == 2


def test_a_complete_walk_that_found_no_match_is_a_real_absence(tmp_path):
    """The gate must not be so strict that no flow can ever prove anything."""
    assert (
        _match(
            tmp_path,
            _dump([{"depth": 0}, {"identifier": "rapid.chat.compose"}]),
        )
        == 1
    )


def test_a_streaming_filter_cannot_hide_a_match(tmp_path):
    """`jq -e` reports the LAST value a filter emits. A per-element filter over
    [match, non-match] yields `true, false` and exits 1 — "absent", having just
    matched. The helper requires exactly one boolean, so this is refused rather
    than answered wrongly."""
    per_element = '.data.ui_elements[]? | (.identifier == "fake-video-alias")'
    payload = _dump(
        [{"identifier": "fake-video-alias"}, {"identifier": "rapid.chat.compose"}]
    )
    assert _match(tmp_path, payload, filter_=per_element) == 2


def test_a_filter_emitting_a_non_boolean_cannot_answer(tmp_path):
    """`… | length` emits a number; jq calls 0 falsy, so a count of zero would
    read as absence and any other count as presence — right by accident until
    a filter emits a string or null."""
    counting = "[.data.ui_elements[]?] | length"
    assert _match(tmp_path, _dump([{"depth": 0}]), filter_=counting) == 2


# ---------------------------------------------------------------------------
# assert_ax_absent turns those three outcomes into pass / fail / fail-loudly.
# ---------------------------------------------------------------------------


def _assert_absent(tmp_path, payload, *, driver="false") -> subprocess.CompletedProcess:
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(payload))
    # `set -euo pipefail` matches production: the retry loop and the helpers
    # must not merely return the right status, they must not abort the run
    # getting there. `sleep` is shadowed so the loop does not cost 5 s.
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        sleep() {{ :; }}
        die() {{ echo "DIE: $*" >&2; exit 9; }}
        AX_DRIVER={driver}
        APP_PID=4242
        {_extract("ax_elements_match")}
        {_extract("redump_evidence")}
        {_extract("assert_ax_absent")}
        {_extract("walk_reasons")}
        assert_ax_absent "$1" "$2" "a video-gen alias reached the chat surface"
        """
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", str(dump), PRESENT_FILTER],
        capture_output=True,
        text=True,
    )


def test_absence_in_a_complete_dump_passes(tmp_path):
    result = _assert_absent(tmp_path, _dump([{"identifier": "rapid.chat.compose"}]))
    assert result.returncode == 0, result.stderr


def test_a_present_element_fails_with_the_callers_message(tmp_path):
    result = _assert_absent(tmp_path, _dump([{"identifier": "fake-video-alias"}]))
    assert result.returncode == 9
    assert "a video-gen alias reached the chat surface" in result.stderr


def test_an_incomplete_dump_fails_loudly_rather_than_passing(tmp_path):
    """The regression this guards: an unobservable dump must never be reported
    as a clean bill of health, and the reason must reach the log."""
    payload = _dump(
        [{"identifier": "rapid.chat.compose"}],
        complete=False,
        reasons=["the record cap of 12000 was reached"],
    )
    result = _assert_absent(tmp_path, payload)
    assert result.returncode == 9, result.stdout + result.stderr
    assert "cannot rule out" in result.stderr
    assert "record cap of 12000" in result.stderr
    # The retries ran against a driver that could not produce anything. The
    # dump the caller captured is the artifact a human debugs from, and it is
    # also where that reason was read from, so it must survive them.
    assert json.loads((tmp_path / "dump.json").read_text()) == payload
    assert not (tmp_path / "dump.json.retry").exists()


def test_a_driver_that_succeeds_with_garbage_does_not_destroy_the_evidence(tmp_path):
    """`true` exits 0 and writes nothing. Judging the retry by exit status
    alone promotes that empty file over the only dump that carried a reason,
    and the failure message then explains nothing."""
    payload = _dump(
        [{"depth": 0}],
        complete=False,
        reasons=["the record cap of 12000 was reached"],
    )
    result = _assert_absent(tmp_path, payload, driver="true")
    assert result.returncode == 9
    assert "record cap of 12000" in result.stderr
    assert json.loads((tmp_path / "dump.json").read_text()) == payload
    assert not (tmp_path / "dump.json.retry").exists()


# ---------------------------------------------------------------------------
# Lint: the raw idiom must not come back. It had already been copied to a third
# assertion (#1673) between the issue being filed and being fixed.
# ---------------------------------------------------------------------------


# The shapes an "it is not there" claim takes when written by hand.
#
# A speed bump, not a proof. `| not` is deliberately NOT on this list: the flows
# use it for ordinary per-element negation (`select(… | startswith("X") | not)`)
# and flagging that would be noise, so `any(…) | not` gets through. So does a
# count bound to a shell variable, or a filter assembled from pieces. What the
# list covers is what people actually reach for, which is what let the same
# assertion be copied to a third flow while #1670 was open.
_ABSENCE_IDIOMS = (
    r"length\s*(?:==|<=)\s*0",  # […] | length == 0
    r"\)\s*==\s*0",  # (… | length) == 0
    r"0\s*==\s*[(\[]",  # 0 == (… | length)
    r"==\s*\[\s*\]",  # […] == []
)

# How far either side of a `ui_elements` reference the idiom may sit. Behind as
# well as ahead, because `0 == (…)` puts the tell first.
_LINT_LOOKBEHIND = 160
_LINT_LOOKAHEAD = 400


def _lint_offenders(source: str) -> list[str]:
    offenders = []
    for anchor in re.finditer(r"ui_elements", source):
        start = max(0, anchor.start() - _LINT_LOOKBEHIND)
        window = source[start : anchor.end() + _LINT_LOOKAHEAD]
        for idiom in _ABSENCE_IDIOMS:
            hit = re.search(idiom, window)
            if hit:
                offenders.append(window[: hit.end()][-140:])
                break
    return offenders


def test_no_flow_proves_absence_by_counting_elements_itself():
    offenders = _lint_offenders(_FLOWS.read_text())
    assert not offenders, (
        "prove absence with assert_ax_absent, which refuses to answer from an "
        "incomplete walk; counting elements yourself is satisfied by never "
        "having looked:\n" + "\n---\n".join(offenders)
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "jq -e '[.data.ui_elements[]? | select(.identifier == \"X\")] | length == 0'",
        "jq -e '([.data.ui_elements[]? | select(.identifier == \"X\")] | length) == 0'",
        "jq -e '0 == ([.data.ui_elements[]? | select(.identifier == \"X\")] | length)'",
        "jq -e '[.data.ui_elements[]? | select(.identifier == \"X\")] == []'",
    ],
)
def test_the_lint_catches_the_idioms_people_actually_write(snippet):
    """A lint nobody has aimed at its own target is decoration."""
    assert _lint_offenders(snippet), snippet


def test_the_lint_does_not_flag_ordinary_negation():
    """`| not` inside a `select` is how the flows filter elements; treating it
    as an absence claim would make the lint noise, and a noisy lint gets
    disabled. The cost is that `any(…) | not` gets through — a known hole, not
    an oversight."""
    ordinary = (
        'jq -e \'.data.ui_elements[]? | select((.identifier // "") '
        '| startswith("Settings.Category.") | not)\''
    )
    assert not _lint_offenders(ordinary)


def test_the_helper_gates_on_the_completeness_signal():
    """Pin the premise. If the helper stops consulting `walk.complete`, every
    test above still passes while the flows go back to proving absence from a
    walk that may be clipped."""
    assert "data.walk.complete == true" in _extract("ax_elements_match")


def test_the_catalog_flow_establishes_the_catalogue_loaded_before_denying_it():
    """The absence assertions must be preceded by a positive one.

    A surface still fetching its model list contains neither the video-gen
    alias nor any other, so the absence check passes without the filter under
    test having been exercised. This pins the ORDER: each `wait_ax_match` comes
    before its `assert_ax_absent`."""
    source = _FLOWS.read_text()
    flow = source[source.index("flow_catalog_integrity() {") :]
    flow = flow[: flow.index("\n}\n")]
    calls = re.findall(r"^\s*(wait_ax_match|assert_ax_absent)\b", flow, re.MULTILINE)
    assert calls == [
        "wait_ax_match",
        "assert_ax_absent",
        "wait_ax_match",
        "assert_ax_absent",
    ], calls


# ---------------------------------------------------------------------------
# The producer. These run the real driver, so they need macOS and a Swift
# toolchain; everything above is text and bash, and runs anywhere.
# ---------------------------------------------------------------------------

_DRIVER = _REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "rapid-ax.swift"

_needs_swift = pytest.mark.skipif(
    not sys.platform.startswith("darwin") or shutil.which("swift") is None,
    reason="the driver is a Swift script and only runs on macOS",
)


def _run_driver(pid: int) -> dict:
    result = subprocess.run(
        ["swift", str(_DRIVER), "dump", str(pid)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@_needs_swift
def test_the_driver_refuses_to_vouch_for_a_process_it_cannot_read():
    """pid 1 has no accessibility tree we can reach. The dump still reports
    `success: true` with a single record — exactly the shape that used to
    satisfy an absence assertion without one element having been observed."""
    payload = _run_driver(1)
    assert payload["success"] is True
    walk = payload["data"]["walk"]
    assert walk["complete"] is False
    assert walk["reasons"], "a refusal with no reason is unfixable"
    assert walk["scope"] == "window-forest"
    assert any("no searchable text" in r for r in walk["reasons"]), walk["reasons"]


@_needs_swift
def test_a_read_failure_alone_does_not_condemn_the_dump():
    """The first version of this signal was unshippable, and only running it
    showed that.

    Measured across a full golden-flow suite: 5 of 77 real dumps carried one
    failed read of a searched attribute (`AXError -25200`), in the same Settings
    panels every run — structural, not a lost race, so the retry loop could
    never recover and `ax-baseline.py` refused the whole suite. A signal that
    cannot be satisfied is worse than one that is slightly coarse.

    The rule is now: a failed read costs completeness only when it leaves the
    element with NOTHING searchable. An element whose title would not read but
    whose identifier did is still found by every filter that tests identifiers.
    Finder is the control — an ordinary app must come back clean.
    """
    pids = subprocess.run(
        ["pgrep", "-x", "Finder"], capture_output=True, text=True
    ).stdout.split()
    if not pids:
        pytest.skip("Finder is not running")
    walk = _run_driver(int(pids[0]))["data"]["walk"]
    assert walk["complete"] is True, walk["reasons"]
    assert "elements_with_unreadable_fields" in walk, (
        "the count has to be reported even when it costs nothing — `reasons` "
        "only ever explains why `complete` is false, so without this the "
        "artifact cannot say a read failed at all"
    )


@_needs_swift
def test_the_driver_vouches_for_an_ordinary_application():
    """The other direction, and the one that decides whether this is usable at
    all: a signal that goes false on a healthy app makes every flow die."""
    pids = subprocess.run(
        ["pgrep", "-x", "Finder"], capture_output=True, text=True
    ).stdout.split()
    if not pids:
        pytest.skip("Finder is not running")
    walk = _run_driver(int(pids[0]))["data"]["walk"]
    assert walk["complete"] is True, walk["reasons"]


# ---------------------------------------------------------------------------
# A structural baseline is the same claim at a larger scale: everything in the
# committed snapshot is here, and nothing else is.
# ---------------------------------------------------------------------------


def test_a_baseline_cannot_be_taken_from_an_incomplete_dump(tmp_path):
    """Otherwise comparison passes on a clipped tree whose recorded prefix
    happens to match, and `--update` commits the clipped tree as the truth."""
    dump = tmp_path / "clipped.json"
    dump.write_text(
        json.dumps(
            _dump(
                [{"depth": 0, "role": "AXApplication"}],
                complete=False,
                reasons=["the record cap of 12000 was reached"],
            )
        )
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "ax-baseline.py"),
            "check",
            str(dump),
            "--baseline",
            str(baseline),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not a complete observation" in result.stderr + result.stdout
