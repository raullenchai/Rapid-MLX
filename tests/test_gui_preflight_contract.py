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
import textwrap
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


def test_fresh_install_settles_transcript_before_structural_baseline():
    """A transient scroll affordance must not become golden structure."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"
    fresh_install = source.split("flow_fresh_install() {", 1)[1].split("\n}", 1)[0]

    assert 'select(.identifier == "Transcript.JumpToBottom")' in helper
    assert 'press "$destination" Transcript.JumpToBottom "$press_result"' in helper
    assert 'select(.identifier == "ChatView.SendOrStopButton")' in helper
    assert ".bounds.x > $compose_x" in helper
    assert '--argjson scroll_x "$scroll_x"' in helper
    assert 'die "Jump to latest did not physically settle' in helper

    banner = fresh_install.index("wait_identifier TelemetryConsent.PostValueBanner")
    settle = fresh_install.index("settle_transcript_at_bottom")
    baseline = fresh_install.index("baseline fresh-install.post-value-consent")
    assert banner < settle < baseline


def test_transcript_settler_waits_for_physical_scroll_stability(tmp_path):
    """A hidden button alone is insufficient while the scroll view is moving."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    fixtures = [
        (0.25, True),
        (0.45, False),
        (0.80, False),
        (1.00, False),
        (1.00, False),
    ]
    for index, (value, has_button) in enumerate(fixtures):
        elements = [
            {
                "role": "AXScrollBar",
                "value": 1.0,
                "bounds": {"x": 180, "width": 17, "height": 320},
            },
            {
                "role": "AXScrollBar",
                "value": value,
                "bounds": {"x": 704, "width": 17, "height": 320},
            },
            {
                "identifier": "ChatView.SendOrStopButton",
                "bounds": {"x": 658, "width": 28, "height": 28},
            },
        ]
        if has_button:
            elements.append(
                {
                    "identifier": "Transcript.JumpToBottom",
                    "bounds": {"x": 444.5, "width": 33, "height": 33},
                }
            )
        (tmp_path / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )

    script = textwrap.dedent(
        f"""
        set -u
        fixture_dir={str(tmp_path)!r}
        calls=0
        see_main() {{
            local destination="$1" index="$calls"
            (( index > 4 )) && index=4
            cp "$fixture_dir/fixture-$index.json" "$destination"
            calls=$((calls + 1))
        }}
        press() {{ :; }}
        die() {{ printf '%s\\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture_dir/current.json" "$fixture_dir/press.json"
        printf '%s\\n' "$calls"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "5"


def test_transcript_settler_waits_when_jump_button_is_initially_absent(tmp_path):
    """Pinned state can precede AppKit reaching the physical tail."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    for index, value in enumerate((0.80, 1.00, 1.00)):
        elements = [
            {
                "role": "AXScrollBar",
                "value": value,
                "bounds": {"x": 704, "width": 17, "height": 320},
            },
            {
                "identifier": "ChatView.SendOrStopButton",
                "bounds": {"x": 658, "width": 28, "height": 28},
            },
        ]
        (tmp_path / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )

    script = textwrap.dedent(
        f"""
        set -u
        fixture_dir={str(tmp_path)!r}
        calls=0
        see_main() {{
            local destination="$1" index="$calls"
            (( index > 2 )) && index=2
            cp "$fixture_dir/fixture-$index.json" "$destination"
            calls=$((calls + 1))
        }}
        press() {{ printf 'unexpected press\\n' >&2; exit 98; }}
        die() {{ printf '%s\\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture_dir/current.json" "$fixture_dir/press.json"
        printf '%s\\n' "$calls"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "3"


def test_transcript_settler_accepts_stable_visible_tail_after_scrollbar_hides(tmp_path):
    """A short reply can fit after Jump and make AppKit remove its overlay bar."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    initial_elements = [
        {
            "role": "AXButton",
            "identifier": "ChatView.SendOrStopButton",
            "bounds": {"x": 658, "y": 546, "width": 28, "height": 28},
        },
        {
            "role": "AXScrollBar",
            "value": 0.25,
            "bounds": {"x": 704, "y": 171, "width": 16, "height": 318},
        },
        {"role": "AXButton", "identifier": "Transcript.JumpToBottom"},
    ]
    settled_elements = [
        {
            "role": "AXButton",
            "identifier": "ChatView.SendOrStopButton",
            "bounds": {"x": 658, "y": 546, "width": 28, "height": 28},
        },
        {
            "role": "AXScrollArea",
            "bounds": {"x": 201, "y": 171, "width": 519, "height": 318},
        },
        {
            "role": "AXButton",
            "identifier": "ChatView.Message.Retry.00000000-0000-0000-0000-000000000000",
            "bounds": {"x": 277, "y": 383, "width": 24, "height": 24},
        },
    ]
    for index, elements in enumerate(
        (initial_elements, settled_elements, settled_elements)
    ):
        (fixture_dir / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )

    script = textwrap.dedent(
        f"""
        set -euo pipefail
        fixture_dir={str(fixture_dir)!r}
        index=0
        calls=0
        see_main() {{
            cp "$fixture_dir/fixture-$index.json" "$1"
            (( index < 2 )) && index=$((index + 1)) || true
            calls=$((calls + 1))
        }}
        press() {{ :; }}
        die() {{ printf '%s\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture_dir/current.json" "$fixture_dir/press.json"
        printf '%s\n' "$calls"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "3"


def test_transcript_settler_accepts_short_reply_without_scrollbar(tmp_path):
    """A fitting first reply can be at its tail before AppKit mounts any bar."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "role": "AXScrollArea",
                            "bounds": {
                                "x": 201,
                                "y": 171,
                                "width": 519,
                                "height": 318,
                            },
                        },
                        {
                            "role": "AXButton",
                            "identifier": "ChatView.Message.Retry.reply",
                            "bounds": {
                                "x": 277,
                                "y": 383,
                                "width": 24,
                                "height": 24,
                            },
                        },
                        {
                            "role": "AXButton",
                            "identifier": "ChatView.SendOrStopButton",
                            "bounds": {
                                "x": 658,
                                "y": 546,
                                "width": 28,
                                "height": 28,
                            },
                        },
                    ]
                }
            }
        )
    )

    script = textwrap.dedent(
        f"""
        set -euo pipefail
        fixture={str(fixture)!r}
        calls=0
        see_main() {{ cp "$fixture" "$1"; calls=$((calls + 1)); }}
        press() {{ printf 'unexpected press\n' >&2; exit 98; }}
        die() {{ printf '%s\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture.current.json" "$fixture.press.json"
        printf '%s\n' "$calls"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "3"


def test_transcript_settler_rechecks_after_a_stale_jump_press(tmp_path):
    """A vanished stale AX element still needs the physical tail proof."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    initial_elements = [
        {
            "role": "AXScrollBar",
            "value": 0.25,
            "bounds": {"x": 704, "width": 17, "height": 320},
        },
        {
            "identifier": "ChatView.SendOrStopButton",
            "bounds": {"x": 658, "width": 28, "height": 28},
        },
        {"identifier": "Transcript.JumpToBottom"},
    ]
    settled_elements = [
        {
            "role": "AXScrollBar",
            "value": 1.0,
            "bounds": {"x": 704, "width": 17, "height": 320},
        },
        {
            "identifier": "ChatView.SendOrStopButton",
            "bounds": {"x": 658, "width": 28, "height": 28},
        },
    ]
    for index, elements in enumerate(
        (initial_elements, settled_elements, settled_elements, settled_elements)
    ):
        (fixture_dir / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )

    script = textwrap.dedent(
        f"""
        set -u
        fixture_dir={str(fixture_dir)!r}
        calls=0
        see_main() {{
            local destination="$1" index="$calls"
            (( index > 3 )) && index=3
            cp "$fixture_dir/fixture-$index.json" "$destination"
            calls=$((calls + 1))
        }}
        press() {{ return 1; }}
        die() {{ printf '%s\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture_dir/current.json" "$fixture_dir/press.json"
        printf '%s\n' "$calls"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "4"


def test_transcript_settler_rejects_failed_press_when_jump_remains(tmp_path):
    """A live but unpressable affordance must continue to fail closed."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    elements = [
        {
            "role": "AXScrollBar",
            "value": 0.25,
            "bounds": {"x": 704, "width": 17, "height": 320},
        },
        {
            "identifier": "ChatView.SendOrStopButton",
            "bounds": {"x": 658, "width": 28, "height": 28},
        },
        {"identifier": "Transcript.JumpToBottom"},
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"data": {"ui_elements": elements}}))

    script = textwrap.dedent(
        f"""
        set -u
        fixture={str(fixture)!r}
        see_main() {{ cp "$fixture" "$1"; }}
        press() {{ return 1; }}
        die() {{ printf '%s\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture.current.json" "$fixture.press.json"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 97
    assert "was not pressable" in completed.stderr


def test_transcript_settler_rejects_a_stable_intermediate_position(tmp_path):
    """Progress plus a hidden button is not proof that the tail was reached."""
    source = HARNESS.read_text()
    helper_body = source.split("settle_transcript_at_bottom() {", 1)[1].split("\n}", 1)[
        0
    ]
    helper = f"settle_transcript_at_bottom() {{{helper_body}\n}}"

    fixtures = [(0.25, True), (0.45, False)]
    for index, (value, has_button) in enumerate(fixtures):
        elements = [
            {
                "role": "AXScrollBar",
                "value": value,
                "bounds": {"x": 704, "width": 17, "height": 320},
            },
            {
                "identifier": "ChatView.SendOrStopButton",
                "bounds": {"x": 658, "width": 28, "height": 28},
            },
        ]
        if has_button:
            elements.append(
                {
                    "identifier": "Transcript.JumpToBottom",
                    "bounds": {"x": 444.5, "width": 33, "height": 33},
                }
            )
        (tmp_path / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )

    script = textwrap.dedent(
        f"""
        set -u
        fixture_dir={str(tmp_path)!r}
        calls=0
        see_main() {{
            local destination="$1" index="$calls"
            (( index > 1 )) && index=1
            cp "$fixture_dir/fixture-$index.json" "$destination"
            calls=$((calls + 1))
        }}
        press() {{ :; }}
        die() {{ printf '%s\\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        settle_transcript_at_bottom "$fixture_dir/current.json" "$fixture_dir/press.json"
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 97
    assert "did not physically settle" in completed.stderr


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


def test_cached_curated_tradeup_confirms_the_quickstart_memory_sheet():
    """The onboarding sheet and main-window warning use different AX IDs."""
    source = HARNESS.read_text()
    flow = source.split("flow_cached_curated_tradeup() {", 1)[1].split("\n}", 1)[0]

    assert "wait_fake_event_after_start" in flow
    assert "Quickstart.Memory.Load" in flow
    assert "Quickstart.Memory.LoadAnyway" in flow


def test_cached_curated_tradeup_waits_for_health_and_bounded_ui_readiness():
    """A spawn event is not health, and hosted UI readiness gets 60 seconds."""
    source = HARNESS.read_text()
    flow = source.split("flow_cached_curated_tradeup() {", 1)[1].split("\n}", 1)[0]

    assert 'wait_fake_sidecar_health "qwen3.5-4b-4bit" "cached 16 GB starter"' in flow
    assert (
        'wait_identifier Quickstart.Ready.StartChatting \\\n        "$OUT/ready-confirmation.json" 240'
        in flow
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
        "cached-quickstart",
        "cached-curated-tradeup",
        "cached-variant-collapse",
        "model-switch-active-request",
        "download-progress",
        "settings-persistence",
        "settings-mtp",
        "chat-depth",
        "browse-all-destination",
        "no-dead-controls",
        "catalog-integrity",
        "update-state",
        "update-busy",
        "campaign-banner",
        "launch-integrations",
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
    """Start waits for an interactive action and its actually selected alias."""
    source = HARNESS.read_text()
    helper = source.split("start_model() {", 1)[1].split("\n}", 1)[0]
    assert "wait_identifier_enabled Readiness.Action" in helper
    assert helper.index("wait_identifier_enabled Readiness.Action") < helper.index(
        'press "$OUT/readiness-start.json" Readiness.Action'
    )
    assert "wait_fake_event_after_start" in helper
    assert (
        'selected_alias="$(element_field "$OUT/readiness-start.json" '
        'ModelPickerBar.ModelMenu value)"'
    ) in helper
    assert r"and .alias == \"$selected_alias\"" in helper
    assert r"and .alias == \"$FAKE_ALIAS\"" not in helper

    driver = DRIVER.read_text()
    click = driver.split('case "click-center":', 1)[1].split(
        'case "set-scroll-value":', 1
    )[0]
    assert "kAXPositionAttribute" in click and "kAXSizeAttribute" in click
    assert ".leftMouseDown" in click and ".leftMouseUp" in click


def test_wait_send_idle_follows_an_intentionally_deferred_auto_start(tmp_path):
    """A low-memory relaunch stays idle until the flow performs visible consent."""
    source = HARNESS.read_text()
    helper_body = source.split("wait_send_idle() {", 1)[1].split("\n}", 1)[0]
    helper = f"wait_send_idle() {{{helper_body}\n}}"

    assert 'identifier == "Readiness.Action"' in helper
    assert 'and .description == "Start"' in helper
    assert "and .enabled == true" in helper
    assert '"$AX_DRIVER" click-center "$APP_PID" Readiness.Action' in helper
    assert "follow_memory_confirmation_edge" in helper
    assert helper.index('"$AX_DRIVER" click-center "$APP_PID" Readiness.Action') < (
        helper.index("follow_memory_confirmation_edge")
    )
    assert "deferred-start" in helper

    fixtures = [
        [
            {
                "identifier": "ChatView.SendOrStopButton",
                "description": "Send message",
                "enabled": False,
                "help": "fake-alias is still starting",
            },
            {
                "identifier": "Readiness.Action",
                "description": "Start",
                "enabled": True,
            },
        ],
        [
            {
                "identifier": "ChatView.SendOrStopButton",
                "description": "Send message",
                "enabled": False,
                "help": "fake-alias is still starting",
            },
            {"identifier": "MemoryWarning.Confirm", "enabled": True},
        ],
        [
            {
                "identifier": "ChatView.SendOrStopButton",
                "description": "Send message",
                "enabled": False,
            }
        ],
        [
            {
                "identifier": "ChatView.SendOrStopButton",
                "description": "Send message",
                "enabled": False,
            }
        ],
    ]
    for index, elements in enumerate(fixtures):
        (tmp_path / f"fixture-{index}.json").write_text(
            json.dumps({"data": {"ui_elements": elements}})
        )
    driver = tmp_path / "driver.sh"
    driver.write_text('#!/bin/bash\nprintf "%s\\n" "$3" >> "$CLICK_LOG"\n')
    driver.chmod(0o755)
    click_log = tmp_path / "clicks.txt"

    script = textwrap.dedent(
        f"""
        set -euo pipefail
        fixture_dir={str(tmp_path)!r}
        AX_DRIVER={str(driver)!r}
        CLICK_LOG={str(click_log)!r}
        export CLICK_LOG
        APP_PID=42
        calls=0
        follow_memory_confirmation_edge() {{
            MEMORY_CONFIRMATION_SIGNATURE="$3"
            MEMORY_CONFIRMATION_POLLS="$4"
            MEMORY_CONFIRMATION_ATTEMPTS="$5"
            MEMORY_CONFIRMATION_VISIBLE=0
            if jq -e '.data.ui_elements[]?
                       | select(.identifier == "MemoryWarning.Confirm"
                                and .enabled == true)' "$1" >/dev/null; then
                "$AX_DRIVER" click-center "$APP_PID" MemoryWarning.Confirm > "$2"
                MEMORY_CONFIRMATION_VISIBLE=1
            fi
        }}
        see_main() {{
            local destination="$1" index="$calls"
            (( index > 3 )) && index=3
            cp "$fixture_dir/fixture-$index.json" "$destination"
            calls=$((calls + 1))
        }}
        log() {{ :; }}
        die() {{ printf '%s\n' "$*" >&2; exit 97; }}
        sleep() {{ :; }}
        {helper}
        wait_send_idle "$fixture_dir/current.json" 8
        """
    )
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert click_log.read_text().splitlines() == [
        "Readiness.Action",
        "MemoryWarning.Confirm",
    ]


def test_start_model_witnesses_the_selected_download_alias(tmp_path):
    """Fresh install starts the downloaded pick, not the persona's fallback."""
    source = HARNESS.read_text()
    helper = (
        "start_model() {"
        + source.split("start_model() {", 1)[1].split("\n}", 1)[0]
        + "\n}"
    )
    capture = tmp_path / "predicate.txt"
    result = subprocess.run(
        [
            "bash",
            "-c",
            helper
            + r"""
set -euo pipefail
OUT="$1"
CAPTURE="$2"
wait_identifier_enabled() { :; }
element_field() {
    if [[ "$2" == "Readiness.Action" ]]; then
        printf 'Start\n'
    else
        printf 'lfm2.5-1b-4bit\n'
    fi
}
press() { :; }
wait_fake_event_after_start() { printf '%s\n' "$1" > "$CAPTURE"; }
wait_send_idle() { :; }
die() { printf '%s\n' "$*" >&2; exit 1; }
start_model
""",
            "start-model-selected-alias",
            str(tmp_path),
            str(capture),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().strip() == (
        '.event == "server_started" and .alias == "lfm2.5-1b-4bit"'
    )


def test_direct_model_starts_follow_enabled_memory_confirmation_branches():
    """Every explicit fake-sidecar start must tolerate real host pressure."""
    source = HARNESS.read_text()
    enabled = source.split("memory_confirmation_enabled() {", 1)[1].split("\n}", 1)[0]
    confirm = source.split("confirm_memory_warning_from_tree() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert ".identifier == $id and .enabled == true" in enabled
    assert 'click-center "$APP_PID" "$identifier"' in confirm
    assert "|| return 1" in confirm

    wait = source.split("wait_fake_event_after_start() {", 1)[1].split("\n}", 1)[0]
    assert 'jq -e -s "any(.[]; $predicate)"' in wait
    assert "follow_memory_confirmation_edge" in wait
    assert "confirmation_signatures" in wait
    assert "confirmation_attempts" in wait
    assert wait.index("follow_memory_confirmation_edge") < wait.index('die "$what"')

    cached = source.split("flow_cached_quickstart() {", 1)[1].split("\n}", 1)[0]
    assert "wait_fake_event_after_start" in cached
    assert "Quickstart.Memory.Load" in cached
    assert "Quickstart.Memory.LoadAnyway" in cached

    image = source.split("flow_image_generation() {", 1)[1].split("\n}", 1)[0]
    assert "wait_fake_event_after_start" in image
    assert r"and .alias == \"$FAKE_IMAGE_ALIAS\"" in image

    resident = source.split("flow_resident_load_rejected() {", 1)[1].split("\n}", 1)[0]
    assert resident.count("wait_fake_event_after_start") == 2
    assert "resident-chat" in resident
    assert "resident-image" in resident

    audio = source.split("flow_audio_readiness() {", 1)[1].split("\n}", 1)[0]
    assert "wait_fake_event_after_start" in audio
    assert 'and .alias == "fake-qwen3-tts"' in audio


def test_memory_confirmation_helper_handles_quickstart_revalidation(tmp_path):
    """A tight confirmation may return as unsafe after live-memory revalidation."""
    source = HARNESS.read_text()

    def shell_function(name: str) -> str:
        return (
            name + "() {" + source.split(name + "() {", 1)[1].split("\n}", 1)[0] + "\n}"
        )

    helpers = "\n".join(
        shell_function(name)
        for name in (
            "memory_confirmation_enabled",
            "memory_confirmation_signature",
            "confirm_memory_warning_from_tree",
            "follow_memory_confirmation_edge",
        )
    )

    driver = tmp_path / "driver.sh"
    driver.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$3" >> "$CLICKS"\n'
        '[[ "$3" != "MemoryWarning.Fail" ]] || exit 1\n'
        "printf '{\"success\":true}\\n'\n"
    )
    driver.chmod(0o755)
    tight = tmp_path / "tight.json"
    unsafe = tmp_path / "unsafe.json"
    tight.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {"identifier": "Quickstart.Memory.Load", "enabled": True}
                    ]
                }
            }
        )
    )
    unsafe.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "identifier": "Quickstart.Memory.LoadAnyway",
                            "enabled": True,
                        }
                    ]
                }
            }
        )
    )
    absent = tmp_path / "absent.json"
    absent.write_text(json.dumps({"data": {"ui_elements": []}}))
    main_tight = tmp_path / "main-tight.json"
    main_unsafe = tmp_path / "main-unsafe.json"
    main_tight.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "identifier": "MemoryWarning.Confirm",
                            "enabled": True,
                            "label": "Load model",
                        }
                    ]
                }
            }
        )
    )
    main_unsafe.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "identifier": "MemoryWarning.Confirm",
                            "enabled": True,
                            "label": "Load anyway (risky)",
                        }
                    ]
                }
            }
        )
    )
    main_unsafe_disabled = tmp_path / "main-unsafe-disabled.json"
    main_unsafe_disabled.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "identifier": "MemoryWarning.Confirm",
                            "enabled": False,
                            "label": "Load anyway (risky)",
                        }
                    ]
                }
            }
        )
    )
    failed_delivery = tmp_path / "failed-delivery.json"
    failed_delivery.write_text(
        json.dumps(
            {
                "data": {
                    "ui_elements": [
                        {
                            "identifier": "MemoryWarning.Fail",
                            "enabled": True,
                            "label": "Load model",
                        }
                    ]
                }
            }
        )
    )
    evidence = tmp_path / "evidence.json"
    clicks = tmp_path / "clicks.txt"
    script = f"""
set -euo pipefail
AX_DRIVER="$1"
APP_PID=42
EVIDENCE="$7"
export CLICKS="$9"
log() {{ :; }}
die() {{ printf '%s\\n' "$*" >&2; exit 1; }}
{helpers}
load_signature=""; load_polls=0; load_attempts=0
anyway_signature=""; anyway_polls=0; anyway_attempts=0
scan_quickstart() {{
    follow_memory_confirmation_edge "$1" "$EVIDENCE" \\
        "$load_signature" "$load_polls" "$load_attempts" Quickstart.Memory.Load
    load_signature="$MEMORY_CONFIRMATION_SIGNATURE"
    load_polls="$MEMORY_CONFIRMATION_POLLS"
    load_attempts="$MEMORY_CONFIRMATION_ATTEMPTS"
    follow_memory_confirmation_edge "$1" "$EVIDENCE" \\
        "$anyway_signature" "$anyway_polls" "$anyway_attempts" Quickstart.Memory.LoadAnyway
    anyway_signature="$MEMORY_CONFIRMATION_SIGNATURE"
    anyway_polls="$MEMORY_CONFIRMATION_POLLS"
    anyway_attempts="$MEMORY_CONFIRMATION_ATTEMPTS"
}}
scan_quickstart "$2"
scan_quickstart "$2"
scan_quickstart "$4"
scan_quickstart "$3"
scan_quickstart "$3"
main_signature=""; main_polls=0; main_attempts=0
scan_main() {{
    follow_memory_confirmation_edge "$1" "$EVIDENCE" \\
        "$main_signature" "$main_polls" "$main_attempts" MemoryWarning.Confirm
    main_signature="$MEMORY_CONFIRMATION_SIGNATURE"
    main_polls="$MEMORY_CONFIRMATION_POLLS"
    main_attempts="$MEMORY_CONFIRMATION_ATTEMPTS"
}}
scan_main "$5"
scan_main "$5"
scan_main "$6"
scan_main "$6"
for _ in {{1..20}}; do scan_main "$6"; done
# Disabling and re-enabling the same mounted semantic decision must not mint a
# fresh delivery budget after the three-attempt cap has been consumed.
scan_main "${{10}}"
for _ in {{1..20}}; do scan_main "$6"; done
fail_signature=""; fail_polls=0; fail_attempts=0
scan_failure() {{
    follow_memory_confirmation_edge "$1" "$EVIDENCE" \\
        "$fail_signature" "$fail_polls" "$fail_attempts" MemoryWarning.Fail
    fail_signature="$MEMORY_CONFIRMATION_SIGNATURE"
    fail_polls="$MEMORY_CONFIRMATION_POLLS"
    fail_attempts="$MEMORY_CONFIRMATION_ATTEMPTS"
}}
for _ in {{1..20}}; do scan_failure "$8"; done
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "confirmation-contract",
            str(driver),
            str(tight),
            str(unsafe),
            str(absent),
            str(main_tight),
            str(main_unsafe),
            str(evidence),
            str(failed_delivery),
            str(clicks),
            str(main_unsafe_disabled),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    recorded = clicks.read_text().splitlines()
    assert recorded[:2] == [
        "Quickstart.Memory.Load",
        "Quickstart.Memory.LoadAnyway",
    ]
    # Tight is clicked once, then the semantically new unsafe presentation
    # gets at most three spaced delivery attempts despite remaining visible or
    # temporarily disabled before it becomes interactive again.
    assert recorded[2:6] == ["MemoryWarning.Confirm"] * 4
    # A failing driver consumes the same spaced budget instead of retrying on
    # every 250 ms poll forever.
    assert recorded[6:] == ["MemoryWarning.Fail"] * 3


def test_ready_wait_confirms_memory_warning_after_session_restore():
    """Automatic restore can show the same sheet without calling start_model."""
    source = HARNESS.read_text()
    wait = source.split("wait_send_idle() {", 1)[1].split("\n}", 1)[0]
    assert "follow_memory_confirmation_edge" in wait
    assert "memory_confirmation_signature" in wait
    assert "memory_confirmation_attempts" in wait
    assert 'MEMORY_CONFIRMATION_VISIBLE" == 1' in wait
    assert wait.index("follow_memory_confirmation_edge") < wait.index(
        'identifier == "ChatView.SendOrStopButton"'
    )
    assert "continue" in wait


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
