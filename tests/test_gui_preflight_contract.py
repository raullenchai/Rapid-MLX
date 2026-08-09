# SPDX-License-Identifier: Apache-2.0
"""Static contract for the golden-flow GUI preflight.

Two different environment faults make the golden flows fail in exactly the same
misleading way — the flow waits 20 s and dies on "main window did not appear",
which reads as "the app is broken":

  * the controlling process does not hold TCC Accessibility, so every AX read
    fails and the dump contains nothing but the application root;
  * the screen is locked, so Accessibility is fine and other processes' trees
    read perfectly, but no application can present a window.

Both were hit for real while this gate was being built. The preflight turns
each into a one-line, correctly-named failure before anything launches. These
tests exist because deleting it would not break a single flow on a healthy
machine — it would only quietly restore the misdiagnosis on an unhealthy one.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "apps/rapid-mac/scripts/rapid-ax.swift"
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"


def test_trust_command_checks_permission_reach_and_lock():
    source = DRIVER.read_text()
    trust = source.split('CommandLine.arguments[1] == "trust"', 1)[1]
    trust = trust.split("guard CommandLine.arguments.count >= 3", 1)[0]

    # The system's opinion about us...
    assert "AXIsProcessTrusted()" in trust
    # ...a real cross-process read, because that opinion is not proof...
    assert "AXUIElementCopyAttributeValue(" in trust
    # ...and the one signal neither of those can carry.
    assert "CGSessionCopyCurrentDictionary()" in trust
    assert "CGSSessionScreenIsLocked" in trust


def test_trust_success_requires_all_three_signals():
    """`success` must not be satisfiable by a subset.

    A locked screen still reports trusted == true and a successful read, so an
    AND that forgets the lock bit is green in exactly the case that matters.
    """
    source = DRIVER.read_text()
    line = next(line for line in source.splitlines() if 'payload["success"]' in line)
    assert "trusted" in line
    assert "readSucceeded" in line
    assert "screenLocked" in line


def test_each_fault_fails_with_its_own_message():
    source = DRIVER.read_text()
    assert source.count("fail(") >= 4
    # Naming the fault is the entire point; a shared message would rebuild the
    # ambiguity this preflight exists to remove.
    assert "the screen is locked" in source
    assert "NOT trusted for Accessibility" in source


def test_preflight_runs_before_any_persona_starts():
    source = HARNESS.read_text()
    require_tools = source.split("require_tools() {", 1)[1].split("\n}", 1)[0]
    assert "require_ax_trust" in require_tools
    # Before the peekaboo branch, so the cheap universal check is not skipped
    # for the flows that return early.
    assert require_tools.index("require_ax_trust") < require_tools.index(
        "flow_requires_peekaboo"
    )
    # And require_tools itself runs before the dispatcher picks a flow.
    assert source.index("require_tools\n") < source.index(
        'case "$FLOW" in\n    fresh-install)'
    )


def test_peekaboo_requirement_is_default_deny():
    """A new flow must be assumed to need peekaboo until stated otherwise.

    The inverse — listing the flows that DO need it — is silent when someone
    adds a flow and forgets, and the failure lands on a runner as an unrelated
    "command not found".
    """
    source = HARNESS.read_text()
    body = source.split("flow_requires_peekaboo() {", 1)[1].split("\n}", 1)[0]
    assert "*) return 0 ;;" in body, "the catch-all must REQUIRE peekaboo"
    peekaboo_free = {
        "chat-restore",
        "restored-tools",
        "tool-loop-budget",
        "chat-depth",
        "slow-stream-stop",
        "model-crash-recovery",
        "image-generation",
    }
    named = {
        flow
        for line in body.splitlines()
        if "return 1" in line
        for flow in line.split(")", 1)[0].strip().split("|")
    }
    assert named == peekaboo_free
