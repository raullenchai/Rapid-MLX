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
DOGFOOD = ROOT / "apps/rapid-mac/scripts/dogfood-isolate.sh"
FAKE_SIDECAR = ROOT / "apps/rapid-mac/scripts/fake-rapid-mlx.sh"


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
        "fresh-install",
        "message-actions",
        "cached-quickstart",
        "cached-curated-tradeup",
        "download-progress",
        "settings-persistence",
        "settings-mtp",
        "restored-tools",
        "tool-loop-budget",
        "chat-depth",
        "math-rendering",
        "browse-all-destination",
        "no-dead-controls",
        "catalog-integrity",
        "update-state",
        "launch-integrations",
        "slow-stream-stop",
        "model-crash-recovery",
        "low-memory-choice",
        "chat-document-attachment",
        "image-generation",
        "dictation",
        "audio-readiness",
        "window-close-prompt",
        "resident-load-rejected",
    }
    named = {
        flow
        for line in body.splitlines()
        if "return 1" in line
        for flow in line.split(")", 1)[0].strip().split("|")
    }
    assert named == peekaboo_free


def test_dogfood_launcher_isolates_port_and_disables_heuristic_sweep():
    """A throwaway persona must never reap an operator's real server (#1618)."""
    source = DOGFOOD.read_text()
    launcher = source.split('cat > "$LAUNCHER" <<LAUNCHEOF', 1)[1].split(
        "LAUNCHEOF", 1
    )[0]
    assert 'export RAPID_DESKTOP_PORT="$ISOLATED_PORT"' in launcher
    assert "export RAPID_DESKTOP_NO_PORT_SWEEP=1" in launcher
    assert "49152" in source and "65535" in source

    # The macOS GoldenFlow is the executable proof: it keeps an
    # operator-shaped listener in the default 8000-8009 window while launching
    # the persona, then
    # checks both process survival and the persona's actual bound port.
    flow = (
        HARNESS.read_text().split("flow_cached_quickstart() {", 1)[1].split("\n}", 1)[0]
    )
    assert "serve operator-owned" in flow
    assert "os.setsid()" in flow
    assert "{8000..8009}" in flow
    assert 'kill -0 "$OPERATOR_SERVER_PID"' in flow
    assert ".port >= 49152 and .port <= 65535" in flow


def test_harness_reaps_its_own_fake_before_relaunch_without_global_sweep():
    source = HARNESS.read_text()
    relaunch = source.split("relaunch_persona() {", 1)[1].split("\n}", 1)[0]
    assert relaunch.index("stop_app") < relaunch.index("cleanup_fake_sidecars")
    assert relaunch.index("cleanup_fake_sidecars") < relaunch.index(
        '"$PERSONA/launch.sh"'
    )


def test_fresh_install_fixture_contains_the_real_starter():
    """The starter assertion is meaningless if the fake catalog omits it."""
    flow = HARNESS.read_text().split("flow_fresh_install() {", 1)[1].split("\n}", 1)[0]
    fake = FAKE_SIDECAR.read_text()
    assert "start_persona fresh-install FAKE_INCLUDE_STARTER=1" in flow
    assert 'if _setting("FAKE_INCLUDE_STARTER") == "1":' in fake
    assert 'print("lfm2.5-1b-4bit' in fake


def test_audio_baseline_waits_for_residency_poll_to_settle():
    flow = (
        HARNESS.read_text().split("flow_audio_readiness() {", 1)[1].split("\n}", 1)[0]
    )
    correlated_row = "any(range(1; $elements | length);"
    resident_alias = '$elements[.].value == "fake-qwen3-tts"'
    resident_lock = '$elements[. - 1].description == "Lock"'
    settled_guard = '[[ "$speech_resident" == 1 ]]'
    launch_check = 'press "$OUT/speech-resident.json" Sidebar.Launch'
    return_to_audio = 'press "$OUT/launch-from-audio.json" Sidebar.Audio'
    switch = 'press "$OUT/audio-after-launch.json" Audio.Mode.Dictation'
    assert correlated_row in flow
    assert resident_alias in flow
    assert resident_lock in flow
    assert settled_guard in flow
    assert launch_check in flow
    assert return_to_audio in flow
    assert switch in flow
    assert flow.index(resident_alias) < flow.index(launch_check)
    assert flow.index(resident_lock) < flow.index(launch_check)
    assert flow.index(settled_guard) < flow.index(launch_check)
    assert flow.index(launch_check) < flow.index(return_to_audio) < flow.index(switch)
