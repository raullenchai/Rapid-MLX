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

import json
import signal
import subprocess
import sys
import time
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


def test_ax_dump_omits_non_finite_numbers_before_json_serialization():
    source = DRIVER.read_text()
    json_value = source.split("func jsonValue", 1)[1].split("\n}", 1)[0]
    assert "number.doubleValue.isFinite ? number : nil" in json_value
    bounds = source.split('record["bounds"]', 1)[0].rsplit("if let origin = point", 1)[
        1
    ]
    assert "origin.x.isFinite" in bounds and "extent.height.isFinite" in bounds


def test_ax_escape_posts_a_real_key_without_answering_nonmodal_consent():
    source = DRIVER.read_text()
    key = source.split('if command == "key" {', 1)[1].split("\n}", 1)[0]
    assert 'wanted == "escape"' in key
    assert "virtualKey: 53" in key
    assert "down.postToPid(pid)" in key
    assert "up.postToPid(pid)" in key

    fresh_install = (
        HARNESS.read_text().split("flow_fresh_install() {", 1)[1].split("\n}", 1)[0]
    )
    assert '"$AX_DRIVER" key "$APP_PID" escape' in fresh_install
    assert (
        "wait_identifier TelemetryConsent.PostValueBanner"
        ' "$OUT/post-value-consent-after-escape.json"'
    ) in fresh_install
    assert "TelemetryConsent.PostValue.Decline" in fresh_install
    assert (
        "explicit No thanks did not dismiss the telemetry invitation" in fresh_install
    )
    assert "dismissed telemetry invitation returned after relaunch" in fresh_install


def test_active_switch_selects_the_fresh_native_menu_item_by_identifier():
    """The driver must not report success merely because key events posted."""
    driver = DRIVER.read_text()
    selection = driver.split('case "select-menu-item":', 1)[1].split(
        'case "set-scroll-value":', 1
    )[0]
    assert "findFreshElement(identifier: itemIdentifier)" in selection
    assert "freshRoots = attribute(" in selection
    assert "application, kAXChildrenAttribute" in selection
    assert "AXUIElementPerformAction(menuItem, kAXPressAction" in selection
    assert "menu item identifier not found after opening" in selection
    assert "selected_title" not in selection

    flow = (
        HARNESS.read_text()
        .split("flow_model_switch_active_request() {", 1)[1]
        .split("\n}", 1)[0]
    )
    assert (
        'select-menu-item "$APP_PID" ModelPickerBar.ModelMenu '
        "\\\n        ModelPickerBar.Alias.fake-external-alias"
    ) in flow
    assert '"$AX_DRIVER" click-center "$APP_PID" ModelSwitchGuard.Cancel' in flow
    assert '"$AX_DRIVER" press "$APP_PID" ModelSwitchGuard.Cancel' not in flow
    assert '"$AX_DRIVER" key "$APP_PID" escape' not in flow


def test_fresh_install_proves_the_telemetry_boundary_with_a_loopback_sink():
    source = HARNESS.read_text()
    sink = source.split("start_telemetry_sink() {", 1)[1].split("\n}", 1)[0]
    fresh_install = source.split("flow_fresh_install() {", 1)[1].split("\n}", 1)[0]

    assert 'LoopbackSinkServer(("127.0.0.1", 0), Sink)' in sink
    assert "HTTPServer.server_bind" in sink and "getfqdn" in sink
    assert '"method": "POST"' in sink
    assert '"path": self.path' in sink
    assert '"bytes": length' in sink
    assert '"event": event' in sink
    assert '"timestamp": timestamp' in sink
    assert '"activation_kind": activation_kind' in sink
    assert '"activation_surface": activation_surface' in sink
    assert '"activation_keys": activation_keys' in sink
    assert 'RAPID_MLX_TELEMETRY_ENDPOINT="http://127.0.0.1:' in fresh_install
    expected_stages = (
        "before-onboarding",
        "before-first-value",
        "post-value-before-decision",
        "after-decline",
        "declined-relaunch",
    )
    for stage in expected_stages:
        assert f"assert_no_telemetry_requests {stage}" in fresh_install

    assert "asked for telemetry before the first working feature" in fresh_install
    assert "did not show exactly one telemetry invitation" in fresh_install
    assert fresh_install.index("assert_no_telemetry_requests before-first-value") < (
        fresh_install.index('send_prompt "Say hello in one short sentence."')
    )
    assert fresh_install.index(
        "assert_no_telemetry_requests post-value-before-decision"
    ) < fresh_install.index("TelemetryConsent.PostValue.Decline")
    assert fresh_install.index("relaunch_persona") < fresh_install.index(
        "assert_no_telemetry_requests declined-relaunch"
    )
    assert fresh_install.index("assert_no_telemetry_requests declined-relaunch") < (
        fresh_install.index("assert_one_telemetry_request settings-opt-in")
    )
    assert "Settings.Privacy.TelemetryToggle" in fresh_install
    positive_control = source.split("assert_one_telemetry_request() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert '.requests[0].path == "/v1/events"' in positive_control
    assert ".requests[0].bytes > 0" in positive_control
    assert '.requests[0].event == "session_start"' in positive_control
    assert ".requests[0].timestamp >= .not_before" in positive_control
    assert 'sleep "$settling_seconds"' in positive_control
    assert "loopback telemetry sink exited while settling" in positive_control
    assert "opt_in_not_before" in fresh_install
    assert "TelemetryConsent.PostValue.Share" in fresh_install
    assert (
        "assert_share_activation_requests share-accepted first_chat_reply"
        in fresh_install
    )
    assert "activation_seen_desktop_first_chat_reply" in fresh_install


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
        "cached-variant-collapse",
        "model-switch-active-request",
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
        "update-busy",
        "campaign-banner",
        "launch-integrations",
        "slow-stream-stop",
        "model-crash-recovery",
        "low-memory-choice",
        "chat-document-attachment",
        "chat-multimodal-attachments",
        "image-generation",
        "dictation",
        "dictation-rc2-upgrade",
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
        "launch_persona_app append"
    )


def test_fake_sidecar_cleanup_waits_then_escalates_without_touching_operator(tmp_path):
    """Regression for #2676: cleanup must finish before persona deletion."""
    graceful_code = """
import signal
import sys
import time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
Path(sys.argv[1]).touch()
while True:
    time.sleep(0.05)
"""
    stubborn_code = """
import signal
import sys
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).touch()
while True:
    time.sleep(0.05)
"""
    ready_paths = [tmp_path / name for name in ("graceful", "stubborn", "operator")]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                graceful_code,
                str(ready_paths[0]),
                "serve",
                "fake-graceful",
            ]
        ),
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                stubborn_code,
                str(ready_paths[1]),
                "serve",
                "fake-stubborn",
            ]
        ),
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                graceful_code,
                str(ready_paths[2]),
                "serve",
                "fake-graceful-backup",
            ]
        ),
    ]
    graceful, stubborn, operator = processes
    for _ in range(100):
        if all(path.exists() for path in ready_paths):
            break
        time.sleep(0.01)
    assert all(path.exists() for path in ready_paths)
    events = tmp_path / "fake-events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps({"event": "server_started", "pid": proc.pid, "alias": alias})
            for proc, alias in (
                (graceful, "fake-graceful"),
                (stubborn, "fake-stubborn"),
                # Simulate a recycled pid whose new alias merely shares the
                # recorded alias prefix. It is not owned by this harness run.
                (operator, "fake-graceful"),
            )
        )
        + "\n"
    )
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'harness="$1"; event_out="$2"; set --; '
                'source "$harness"; OUT="$event_out"; cleanup_fake_sidecars',
                "cleanup-test",
                str(HARNESS),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert graceful.wait(timeout=1) == 0
        assert stubborn.wait(timeout=1) == -signal.SIGKILL
        assert operator.poll() is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)


def test_fake_sidecar_cleanup_fails_closed_on_partial_ownership_log(tmp_path):
    ready = tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import signal
import sys
import time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
Path(sys.argv[1]).touch()
while True:
    time.sleep(0.05)
""",
            str(ready),
            "serve",
            "fake-partial",
        ]
    )
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.01)
    assert ready.exists()
    (tmp_path / "fake-events.jsonl").write_text(
        json.dumps(
            {"event": "server_started", "pid": process.pid, "alias": "fake-partial"}
        )
        + '\n{"event":"server_started"'
    )

    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'harness="$1"; event_out="$2"; set --; '
                'source "$harness"; OUT="$event_out"; cleanup_fake_sidecars',
                "cleanup-test",
                str(HARNESS),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "could not parse fake sidecar ownership log" in result.stderr
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=1)


def test_stop_app_drains_its_owned_process_group(tmp_path):
    """Unreported catalogue children must exit before their HOME is deleted."""
    leader = tmp_path / "leader.py"
    child_pid_file = tmp_path / "child.pid"
    child_ready = tmp_path / "child.ready"
    leader.write_text(
        """
import os
import subprocess
import sys
import time
from pathlib import Path

os.setsid()
child = subprocess.Popen([
    sys.argv[3],
    "-c",
    "import signal,sys,time; from pathlib import Path; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "Path(sys.argv[1]).touch(); time.sleep(60)",
    sys.argv[2],
])
for _ in range(100):
    if Path(sys.argv[2]).exists():
        break
    time.sleep(0.01)
Path(sys.argv[1]).write_text(str(child.pid))
while True:
    time.sleep(1)
"""
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'harness="$1"; leader="$2"; pid_file="$3"; ready="$4"; python="$5"; '
            'set --; source "$harness"; '
            '"$python" "$leader" "$pid_file" "$ready" "$python" & APP_PID=$!; '
            'for _ in {1..100}; do test -s "$pid_file" && break; sleep 0.01; done; '
            'test -s "$pid_file"; child_pid="$(cat "$pid_file")"; '
            'stop_app; child_state="$(ps -p "$child_pid" -o stat= 2>/dev/null '
            '| tr -d "[:space:]" || true)"; '
            'test -z "$child_state" || test "${child_state#Z}" != "$child_state"',
            "process-group-test",
            str(HARNESS),
            str(leader),
            str(child_pid_file),
            str(child_ready),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_exit_trap_turns_cleanup_failure_into_failed_result(tmp_path):
    """A passing journey cannot hide a retained process or persona."""
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"status": "pass", "exit_code": 0}))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'harness="$1"; out="$2"; set --; source "$harness"; '
            'OUT_ROOT="$out"; FLOW="cleanup-contract"; APP_SOURCE="fixture.app"; '
            "RESULT_WRITTEN=1; cleanup_persona() { return 1; }; "
            "cleanup_operator_server() { return 0; }; "
            "cleanup_telemetry_sink() { return 0; }; "
            "trap finish EXIT; exit 0",
            "finish-test",
            str(HARNESS),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    evidence = json.loads(result_path.read_text())
    assert evidence["status"] == "fail"
    assert evidence["exit_code"] != 0


def test_fresh_install_fixture_contains_the_real_starter():
    """The starter assertion is meaningless if the fake catalog omits it."""
    flow = HARNESS.read_text().split("flow_fresh_install() {", 1)[1].split("\n}", 1)[0]
    fake = FAKE_SIDECAR.read_text()
    assert "start_persona fresh-install FAKE_INCLUDE_STARTER=1" in flow
    assert 'if _setting("FAKE_INCLUDE_STARTER") == "1":' in fake
    assert 'print("lfm2.5-1b-4bit' in fake


def test_start_model_waits_for_an_interactive_readiness_action():
    """A mounted SwiftUI button can still reject an AX press while disabled."""
    source = HARNESS.read_text()
    helper = source.split("start_model() {", 1)[1].split("\n}", 1)[0]
    assert "wait_identifier_enabled Readiness.Action" in helper
    assert helper.index("wait_identifier_enabled Readiness.Action") < helper.index(
        'press "$OUT/readiness-start.json" Readiness.Action'
    )
    assert 'identifier == "MemoryWarning.Confirm" and .enabled == true' in helper
    assert '"$AX_DRIVER" click-center "$APP_PID" MemoryWarning.Confirm' in helper

    driver = DRIVER.read_text()
    click = driver.split('case "click-center":', 1)[1].split(
        'case "set-scroll-value":', 1
    )[0]
    assert "kAXPositionAttribute" in click and "kAXSizeAttribute" in click
    assert ".leftMouseDown" in click and ".leftMouseUp" in click


def test_image_inflight_baseline_uses_an_event_backed_warmup_phase():
    flow = (
        HARNESS.read_text().split("flow_image_generation() {", 1)[1].split("\n}", 1)[0]
    )
    fake = FAKE_SIDECAR.read_text()

    assert (
        'FAKE_IMAGE_FIRST_WARMUP_ACK="$OUT_ROOT/image-generation/ig-warmup-ack"' in flow
    )
    wire_gate = 'wait_fake_event \'.event == "image_request"'
    assert wire_gate in flow
    assert flow.index(wire_gate) < flow.index('see_main "$OUT/ig-inflight.json"')
    assert '_setting("FAKE_IMAGE_FIRST_WARMUP_ACK")' in fake
    assert '"running": self.running and not self.warming_up' in fake
    assert "while not os.path.exists(first_warmup_ack)" in fake
    assert '[[ "$inflight" == 1 ]]' in flow
    acknowledgement = ': > "$OUT/ig-warmup-ack"'
    assert flow.index('[[ "$inflight" == 1 ]]') < flow.index(acknowledgement)
    assert flow.index(acknowledgement) < flow.index(
        "baseline image-generation.inflight"
    )


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
