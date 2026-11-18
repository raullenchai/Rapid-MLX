"""Run with python3 -m unittest discover -s Tests/GUIGoldenFlows -p 'test_*.py'."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "scripts/gui-golden-flows.sh").read_text()


def function(name):
    return re.search(r"^" + name + r"\(\) \{\n.*?^\}", SCRIPT, re.M | re.S)[0]


class HarnessIsolationTests(unittest.TestCase):
    def test_chat_lifecycle_flows_select_fixture_before_starting(self):
        # Exercise each journey's real setup through its first prompt. Skip
        # may win discovery (starter alias) or lose it (cached fixture alias).
        for flow in ("flow_chat_restore", "flow_model_crash_recovery"):
            for initial_alias, selection in (
                ("lfm2.5-2.6b-4bit", "delayed"),
                ("fake-alias", "delayed"),
                ("fake-alias", "ready"),
                ("fake-alias", "idle"),
                ("fake-alias", "idle-localized"),
                ("fake-alias", "start-disabled-then-enabled"),
                ("fake-alias", "auto-start"),
                ("fake-alias", "start-stays-disabled"),
                ("fake-alias", "never-ready"),
                ("fake-alias", "alias-drift"),
                ("lfm2.5-2.6b-4bit", "ready"),
                ("lfm2.5-2.6b-4bit", "missing-event"),
                ("lfm2.5-2.6b-4bit", "disabled-picker"),
                ("lfm2.5-2.6b-4bit", "missing-picker"),
                ("lfm2.5-2.6b-4bit", "ignored"),
                ("lfm2.5-2.6b-4bit", "failed"),
            ):
                with self.subTest(
                    flow=flow, initial=initial_alias, selection=selection
                ):
                    shell = (
                        "\n".join(
                            function(name)
                            for name in (
                                "wait_identifier_enabled",
                                "start_fixture_chat_model",
                                "wait_fake_event_after_start",
                                "wait_send_idle",
                                flow,
                            )
                        )
                        + r"""
OUT_ROOT="$1"
OUT="$1"
FAKE_ALIAS=fake-alias
APP_PID=123
AX_DRIVER=fixture_driver
selected_alias="$2"
selection="$3"
polls=0
selection_requested=0
selection_attempts=0
start_presses=0
fixture_started=0
initial_alias="$selected_alias"
# Idle fixtures emit no launch event until Start is pressed or auto-start wins.
if [[ "$selected_alias" == fake-alias && "$selection" != idle* && "$selection" != start-* && "$selection" != auto-start ]]; then
    printf '{"event":"server_started","alias":"fake-alias"}\n' > "$OUT/fake-events.jsonl"
fi
follow_memory_confirmation_edge() {
    MEMORY_CONFIRMATION_SIGNATURE=''
    MEMORY_CONFIRMATION_POLLS=0
    MEMORY_CONFIRMATION_ATTEMPTS=0
    MEMORY_CONFIRMATION_CLICKED=0
    MEMORY_CONFIRMATION_VISIBLE=0
}
log() { :; }
start_persona() { :; }
dismiss_first_run() { :; }
sleep() { :; }
die() { echo "$*" >&2; exit 1; }
fixture_driver() {
    [[ "$*" == 'select-menu-item 123 ModelPickerBar.ModelMenu ModelPickerBar.Alias.fake-alias' ]] || exit 2
    [[ "$polls" -ge 2 ]] || die "picker used before enabled"
    selection_attempts=$((selection_attempts + 1))
    # Reselecting the current item succeeds without emitting a launch event.
    [[ "$selected_alias" != fake-alias ]] || return 0
    [[ "$selection" != failed ]] || return 1
    selection_requested=1
    if [[ "$selection" != missing-event ]]; then
        printf '{"event":"server_started","alias":"fake-alias"}\n' >> "$OUT/fake-events.jsonl"
    fi
}
see_main() {
    polls=$((polls + 1))
    if [[ "$selection_requested" == 1 && "$selection" != ignored ]] \
       && { [[ "$selection" != delayed ]] || [[ "$polls" -ge 4 ]]; }; then
        selected_alias=fake-alias
    fi
    local picker_enabled=false readiness='' start_enabled=true start_label=Start
    if [[ "$polls" -ge 2 && "$selection" != disabled-picker ]]; then
        picker_enabled=true
    fi
    # The old selection may expose a disabled action. The selected cached
    # fixture can already be ready and expose no readiness action at all.
    if [[ "$selection" == auto-start && "$polls" -ge 4 && "$fixture_started" == 0 ]]; then
        fixture_started=1
        printf '{"event":"server_started","alias":"fake-alias"}\n' >> "$OUT/fake-events.jsonl"
    fi
    if [[ "$fixture_started" == 0 ]] \
       && { [[ "$selection" == idle* || "$selection" == start-* || "$selection" == auto-start ]]; }; then
        if [[ "$selection" == start-stays-disabled || "$selection" == auto-start ]] \
           || { [[ "$selection" == start-disabled-then-enabled && "$polls" -lt 4 ]]; }; then
            start_enabled=false
        fi
        [[ "$selection" != idle-localized ]] || start_label=启动
        readiness=",{\"identifier\":\"Readiness.Action\",\"description\":\"$start_label\",\"enabled\":$start_enabled}"
    elif [[ "$selection" == never-ready ]] \
       || { [[ "$selection" == delayed ]] && [[ "$polls" -lt 6 ]]; }; then
        readiness=',{"identifier":"Composer.LoadStatus"}'
    elif [[ "$selected_alias" == fake-alias || "$selection" == alias-drift ]]; then
        readiness=',{"identifier":"ChatView.SendOrStopButton","description":"Send message","enabled":true}'
        if [[ "$selection" == alias-drift && "$polls" -ge 4 ]]; then
            selected_alias=lfm2.5-2.6b-4bit
        fi
    else
        readiness=',{"identifier":"Readiness.Action","enabled":false}'
    fi
    if [[ "$selection" == missing-picker ]]; then
        printf '{"data":{"ui_elements":[]}}\n' > "$1"
    else
        printf '{"data":{"ui_elements":[{"identifier":"ModelPickerBar.ModelMenu","enabled":%s,"value":"%s"}%s]}}\n' \
            "$picker_enabled" "$selected_alias" "$readiness" > "$1"
    fi
}
element_field() { jq -r --arg id "$2" '.data.ui_elements[] | select(.identifier == $id) | .value' "$1"; }
press() {
    [[ "$2" == Readiness.Action ]] || die "unexpected press: $2"
    jq -e '.data.ui_elements[] | select(.identifier == "Readiness.Action"
        and (.description == "Start" or .description == "启动") and .enabled == true)' \
        "$1" >/dev/null || die "Start pressed before enabled"
    [[ "$fixture_started" == 0 ]] || die "fixture started twice"
    start_presses=$((start_presses + 1))
    fixture_started=1
    printf '{"event":"server_started","alias":"fake-alias"}\n' >> "$OUT/fake-events.jsonl"
    printf '{}\n' > "$3"
}
start_model() { die "cached fixture must not be started twice"; }
send_prompt() {
    local expected_presses=0
    if [[ "$selection" == idle* || "$selection" == start-disabled-then-enabled ]]; then
        expected_presses=1
    fi
    [[ "$start_presses" == "$expected_presses" ]] || die "unexpected Start press count: $start_presses"
    if [[ "$selection" == start-disabled-then-enabled || "$selection" == auto-start ]]; then
        [[ "$polls" -ge 4 ]] || die "startup polling skipped"
    fi
    if [[ "$initial_alias" == fake-alias ]]; then
        [[ "$selection_attempts" == 0 ]] || die "fixture model unnecessarily reselected"
    else
        [[ "$selection_attempts" == 1 ]] || die "fixture model not selected exactly once"
    fi
    [[ "$(element_field "$OUT/fixture-model-ready.json" ModelPickerBar.ModelMenu value)" == fake-alias ]] \
        || die "prompt sent on wrong alias"
    echo "started $selected_alias"
    exit 0
}
"""
                        + "\n"
                        + flow
                    )
                    with tempfile.TemporaryDirectory() as tmp:
                        result = subprocess.run(
                            [
                                "/bin/bash",
                                "-eu",
                                "-o",
                                "pipefail",
                                "-c",
                                shell,
                                "test",
                                tmp,
                                initial_alias,
                                selection,
                            ],
                            text=True,
                            capture_output=True,
                        )
                    if selection in (
                        "ignored",
                        "failed",
                        "missing-event",
                        "disabled-picker",
                        "missing-picker",
                        "never-ready",
                        "alias-drift",
                        "start-stays-disabled",
                    ):
                        self.assertNotEqual(result.returncode, 0)
                        self.assertNotIn("started", result.stdout)
                        if selection == "start-stays-disabled":
                            self.assertEqual(
                                result.stderr,
                                "chat fixture model did not become ready\n",
                            )
                        self.assertIn(
                            "ModelPickerBar.ModelMenu"
                            if selection in ("disabled-picker", "missing-picker")
                            else (
                                "composer never settled"
                                if selection == "never-ready"
                                else "fixture model"
                            ),
                            result.stderr,
                        )
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, "started fake-alias\n")

    def test_crash_recovery_pins_fixture_before_start_and_checks_respawn_alias(self):
        for restarted_alias in ("fake-alias", "lfm2.5-2.6b-4bit"):
            with (
                self.subTest(restarted_alias=restarted_alias),
                tempfile.TemporaryDirectory() as tmp,
            ):
                shell = (
                    function("start_fixture_chat_model")
                    + "\n"
                    + function("flow_model_crash_recovery")
                    + r"""
OUT_ROOT="$1"
OUT="$1"
FAKE_ALIAS=fake-alias
APP_PID=123
AX_DRIVER=driver
log() { :; }
die() { echo "$*" >&2; exit 1; }
start_persona() { :; }
dismiss_first_run() { :; }
driver() {
    [[ "$*" == 'select-menu-item 123 ModelPickerBar.ModelMenu ModelPickerBar.Alias.fake-alias' ]] || exit 1
    touch "$OUT/fixture-selected"
}
wait_identifier_enabled() { echo starter > "$2"; }
wait_fake_event_after_start() {
    [[ -f "$OUT/fixture-selected" ]] || die "started without selecting fixture"
    printf '{"event": "server_started", "alias": "fake-alias"}\n' > "$OUT/fake-events.jsonl"
}
see_main() {
    printf '{"data":{"ui_elements":[{"identifier":"ChatView.SendOrStopButton"}]}}\n' > "$1"
}
element_field() {
    if [[ "$1" == */fixture-picker-ready.json ]]; then echo starter
    elif [[ "$2" == ModelPickerBar.ModelMenu ]]; then echo fake-alias
    else echo 'Send message'; fi
}
wait_send_idle() { :; }
baseline() { touch "$OUT/baseline-checked"; }
cleanup_persona() { :; }
RESTARTED_ALIAS="$2"
send_prompt() {
    printf '{"event": "server_started", "alias": "%s"}\n' "$RESTARTED_ALIAS" >> "$OUT/fake-events.jsonl"
}
flow_model_crash_recovery
"""
                )
                result = subprocess.run(
                    ["/bin/bash", "-eu", "-c", shell, "test", tmp, restarted_alias],
                    text=True,
                    capture_output=True,
                )
                if restarted_alias == "fake-alias":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue((Path(tmp) / "baseline-checked").exists())
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("other than the crash fixture", result.stderr)
                    self.assertFalse((Path(tmp) / "baseline-checked").exists())

    def test_persona_language_is_pinned_across_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "home").mkdir()
            shell = (
                function("start_persona")
                + "\n"
                + function("relaunch_persona")
                + r"""
ROOT=/stub
APP_SOURCE=/stub/app
OUT_ROOT="$1"
cleanup_persona() { :; }
seed_persona_feature_defaults() { :; }
stop_app() { :; }
cleanup_fake_sidecars() { :; }
wait_for_window() { :; }
env() { :; }
jq() { :; }
mktemp() { printf '%s\n' "$OUT_ROOT"; }
/stub/scripts/dogfood-isolate.sh() { echo /stub/app; }
/usr/libexec/PlistBuddy() { echo test.bundle; }
launch_persona_app() { printf '<%s>' "$@"; printf '\n'; }
start_persona default
relaunch_persona
start_persona localized RAPID_GUI_APP_LANGUAGE=zh-Hans FAKE_OTHER=value
relaunch_persona
start_persona reset
relaunch_persona
start_persona empty RAPID_GUI_APP_LANGUAGE=
relaunch_persona
"""
            )
            result = subprocess.run(
                ["/bin/bash", "-eu", "-c", shell, "test", tmp],
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"<{mode}><-AppleLanguages><({language})>"
                for language in ("en", "zh-Hans", "en", "en")
                for mode in ("truncate", "append")
            ],
        )

    def test_baselines_only_contain_cjk_in_explicitly_localized_journeys(self):
        # The zh-Hans localized-photo-hint journey has no committed baseline.
        # Add exact localized baseline filenames here if it gains any.
        localized_baselines = set()
        cjk = re.compile(
            r"[\u2e80-\u2fff\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf"
            r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f"
            r"\U00020000-\U000323af]"
        )
        snapshots = ROOT / "Tests/GUIGoldenFlows/__Snapshots__"
        for path in sorted(snapshots.rglob("*.txt")):
            with self.subTest(baseline=path.name):
                if path.name not in localized_baselines:
                    self.assertNotRegex(path.read_text(), cjk)

    def test_feature_inventory_matches_product_defaults(self):
        table = subprocess.check_output(
            [
                "bash",
                "-c",
                function("persona_feature_defaults") + "\npersona_feature_defaults",
            ],
            text=True,
        )
        seeds = dict(
            (key, (kind, value))
            for key, kind, value in (line.split() for line in table.splitlines())
        )
        # Intentionally fixed: every seed addition/removal needs a source-default
        # decision here, even when a new seed happens to match another parser.
        expected_keys = {
            "Rapid.experimental.videoGenerationEnabled",
            "Rapid.experimental.computerUseEnabled",
            "Rapid.experimental.localSearchEnabled",
            "Rapid.experimental.communityBenchmarkEnabled",
            "Rapid.experimental.shareComputeEnabled",
            "rapid.mcp.connectors.enabled.v1",
            "rapid.mcp.approval.mode.v1",
            "rapid.window.hideDockChoice",
            "SUAutomaticallyUpdate",
            "rapid.perf.modelOverrides.v1",
            "rapid.server.auto_start_on_launch.v1",
            "rapid.server.confirm_active_request_switch.v1",
            "rapid.picker.show_all_models.v1",
            "Rapid.showLogs",
            "rapid.memory.enabled",
            "dictation.enabled",
            "dictation.archiveAudio",
            "rapid.appearance.v1",
            "rapid.webSearch.provider",
            "rapid.tools.browse.mode.v1",
            "rapid.tools.enabled.web_search",
            "rapid.tools.enabled.browse",
            "rapid.tools.enabled.weather",
            "rapid.tools.enabled.read_document",
        }
        self.assertEqual(set(seeds), expected_keys)
        self.assertEqual(len(table.splitlines()), len(seeds), "duplicate seed key")
        defaults = {}

        def source(path):
            return (ROOT / "Sources/Rapid" / path).read_text()

        def capture(text, pattern):
            matches = re.findall(pattern, text, re.M)
            self.assertEqual(len(matches), 1, f"ambiguous/missing default: {pattern}")
            return matches[0]

        def constant(text, name):
            return capture(
                text,
                r"\bstatic let "
                + re.escape(name)
                + r'(?:\s*:\s*String)?\s*=\s*"([^"\n]+)"',
            )

        def record(key, kind, value):
            self.assertNotIn(key, defaults, "duplicate source-default coverage")
            defaults[key] = (kind, value)

        for path in (ROOT / "Sources/Rapid").rglob("*FeatureConfig.swift"):
            text = path.read_text()
            record(
                constant(text, "enabledKey"),
                "bool",
                capture(text, r"\bstatic let defaultEnabled\s*=\s*(true|false)\b"),
            )

        for path, preference in (
            ("Server/AutoStartDecision.swift", "AutoStartPreference"),
            ("Server/ServerResidency.swift", "ModelSwitchConfirmationPreference"),
        ):
            text = capture(source(path), r"enum " + preference + r"\s*\{([\s\S]*?)^\}")
            record(
                constant(text, "storageKey"),
                "bool",
                capture(
                    text,
                    r"\bstatic let defaultValue(?:\s*:\s*Bool)?\s*=\s*(true|false)\b",
                ),
            )

        # Resolve both halves of AppStorage's key-constant/literal declaration.
        for key_source, key_constant, view, qualified_key in (
            (
                "Server/ModelPickerVisibility.swift",
                "showAllStorageKey",
                "UI/ModelPickerBar.swift",
                "ModelPickerVisibility.showAllStorageKey",
            ),
            (
                "UI/ContentView.swift",
                "showLogsKey",
                "UI/ContentView.swift",
                "ContentView.showLogsKey",
            ),
        ):
            record(
                constant(source(key_source), key_constant),
                "bool",
                capture(
                    source(view),
                    r"@AppStorage\("
                    + re.escape(qualified_key)
                    + r"\)\s*private var \w+(?:\s*:\s*Bool)?\s*=\s*(true|false)\b",
                ),
            )

        text = source("Dictation/DictationController.swift")
        record(
            constant(text, "archiveAudio"),
            "bool",
            capture(
                text,
                r"self.archiveAudio = defaults.object\(forKey: Keys.archiveAudio\)"
                r" as\? Bool \?\? (true|false)\b",
            ),
        )

        # Enum cases here have implicit raw strings; insist that remains true.
        for path, key_constant, pattern in (
            (
                "MCP/MCPToolApprovalStore.swift",
                "modeKey",
                r'self.mode = Mode\(rawValue: defaults.string\(forKey: Self.modeKey\) \?\? ""\) \?\? \.(\w+)',
            ),
            (
                "Tools/BrowseApprovalStore.swift",
                "modeKey",
                r'self.mode = Mode\(rawValue: defaults.string\(forKey: Self.modeKey\) \?\? ""\) \?\? \.(\w+)',
            ),
            (
                "UI/DockVisibilityPromptStore.swift",
                "choiceKey",
                r'let initial = HideDockChoice\(rawValue: raw \?\? ""\) \?\? \.(\w+)',
            ),
            (
                "UI/AppearanceConfig.swift",
                "storageKey",
                r"\} else \{\s*self.mode = \.(\w+)\s*\}",
            ),
            (
                "Tools/WebSearchProvider.swift",
                "providerKey",
                r"\} else \{\s*self.provider = \.(\w+)\s*\}",
            ),
        ):
            text = source(path)
            value = capture(text, pattern)
            self.assertRegex(
                text, r"(?m)^ *case " + re.escape(value) + r" *$", msg=path
            )
            record(constant(text, key_constant), "string", value)

        text = source("Updater/SparkleUpdateController.swift")
        key, value = capture(
            text, r'object\(forKey: "([^"\n]+)"\) as\? Bool \?\? (true|false)\b'
        )
        record(key, "bool", value)

        # Verify that built-in tools default to enabled when no preference is stored.
        text = source("Chat/ChatViewModel.swift")
        template = capture(
            text,
            r'private static func toolEnabledKey\(_ name: String\) -> String \{\s*"([^"\n]+)"\s*\}',
        )
        self.assertRegex(
            text,
            r"let enabled = toolDefaults.object\(forKey: Self.toolEnabledKey\(name\)\) as\? Bool"
            r"\s*\?\? true",
        )
        self.assertIn("if !enabled { disabled.insert(name) }", text)
        self.assertIn("var disabled = Set<String>()", text)
        self.assertIn("self.disabledTools = disabled", text)
        for name in (
            "web_search",
            "browse",
            "weather",
            "read_document",
        ):
            record(
                template.replace(r"\(name)", name),
                "bool",
                "true",
            )

        # No explicit scalar default: UserDefaults.bool returns false on an
        # absent key; MTP decodes a sparse per-model map (empty means no override).
        # Locations are relative to Sources/Rapid; key presence guards renames.
        allowlist = {
            # MCP/MCPConfigStore.swift:53 (implicit UserDefaults bool default)
            "rapid.mcp.connectors.enabled.v1": (
                "MCP/MCPConfigStore.swift",
                "bool",
                "false",
            ),
            # Chat/MemoryStore.swift:70 (implicit UserDefaults bool default)
            "rapid.memory.enabled": ("Chat/MemoryStore.swift", "bool", "false"),
            # Dictation/DictationController.swift:265 (test override, then implicit bool default)
            "dictation.enabled": (
                "Dictation/DictationController.swift",
                "bool",
                "false",
            ),
            # Server/ModelPerfConfigStore.swift:49-51 (missing blob -> empty override map)
            "rapid.perf.modelOverrides.v1": (
                "Server/ModelPerfConfigStore.swift",
                "data",
                "7b7d",
            ),
        }
        for key, (path, kind, value) in allowlist.items():
            self.assertIn('"' + key + '"', source(path), path)
            record(key, kind, value)

        # Retain the obsolete preference in isolation seeds so older app
        # binaries remain isolated too. Current Desktop ignores this key.
        legacy_key = "Rapid.experimental.communityBenchmarkEnabled"
        self.assertEqual(seeds[legacy_key], ("bool", "false"))
        source_keys = expected_keys - {legacy_key}
        self.assertEqual(
            set(defaults), source_keys, "every active seed needs a source default"
        )
        for key in sorted(source_keys):
            with self.subTest(key=key):
                self.assertEqual(seeds[key], defaults[key])

    def run_wait(self, sequence):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for index, state in enumerate(sequence):
                elements = (
                    []
                    if state == "missing"
                    else [
                        {
                            "identifier": identifier,
                            **(
                                {}
                                if state == "unreadable"
                                else {
                                    "enabled": state
                                    in ("ready", "duplicate_sidebar", "duplicate_chat")
                                }
                            ),
                        }
                        for identifier in (
                            ("Sidebar.Residency.Unload",) * 2
                            if state == "duplicate_sidebar"
                            else ("ChatView.Residency.Unload",) * 2
                            if state == "duplicate_chat"
                            else (
                                "Sidebar.Residency.Unload",
                                "ChatView.Residency.Unload",
                            )
                        )
                    ]
                )
                (directory / f"{index}.json").write_text(
                    json.dumps({"data": {"ui_elements": elements}})
                )
            shell = (
                function("wait_residency_idle")
                + """
index=0
see_main() {
    local fixture=$index
    [[ "$fixture" -lt "$limit" ]] || fixture=$((limit - 1))
    cp "$directory/$fixture.json" "$1"
    index=$((index + 1))
}
sleep() { :; }
die() { echo "$*" >&2; exit 1; }
wait_residency_idle "$directory/result.json"
echo "$index"
"""
            )
            return subprocess.run(
                ["bash", "-eu", "-c", shell],
                text=True,
                capture_output=True,
                env={
                    "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
                    "directory": str(directory),
                    "limit": str(len(sequence)),
                },
            )

    def test_residency_wait_resets_after_busy_or_missing_reads(self):
        result = self.run_wait(
            [
                "ready",
                "ready",
                "busy",
                "ready",
                "missing",
                "ready",
                "unreadable",
                "ready",
                "ready",
                "ready",
                "ready",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "11")

    def test_stably_disabled_or_unreadable_controls_never_pass(self):
        for state in ("busy", "missing", "unreadable"):
            result = self.run_wait([state])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not settle enabled", result.stderr)

    def test_duplicate_identifier_cannot_replace_the_missing_control(self):
        for state in ("duplicate_sidebar", "duplicate_chat"):
            with self.subTest(state=state):
                result = self.run_wait([state])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("did not settle enabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
