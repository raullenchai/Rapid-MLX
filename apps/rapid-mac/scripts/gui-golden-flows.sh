#!/usr/bin/env bash
# Release-grade AX-first GUI journeys for Rapid-MLX Desktop.
#
# Each flow runs in a unique bundle-id + HOME, targets elements by semantic
# accessibility identifiers, and writes JSON/screenshot evidence. The bundled
# fake sidecar keeps the suite deterministic and prevents model-related OOMs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_SOURCE="${RAPID_GUI_SOURCE_APP:-$ROOT/build/Rapid-MLX Desktop.app}"
OUT_ROOT="${RAPID_GUI_GOLDEN_OUT:-/tmp/rapid-gui-golden-$(date -u +%Y%m%dT%H%M%SZ)}"
BRIDGE="${PEEKABOO_BRIDGE_SOCKET:-$HOME/Library/Application Support/Peekaboo/daemon.sock}"
FLOW="all"
KEEP=0
APP_PID=""
PERSONA=""
OUT=""
MAIN_WINDOW_ID=""
BUNDLE_ID=""
AX_DRIVER=""
RESULT_WRITTEN=0

usage() {
    cat <<'EOF'
Usage: gui-golden-flows.sh [--flow NAME] [--keep]

Flows: fresh-install, settings-persistence, chat-restore, slow-stream-stop,
       model-crash-recovery, low-memory-choice, update-state, no-dead-controls,
       catalog-integrity, all

Environment:
  RAPID_GUI_SOURCE_APP   built .app to test
  RAPID_GUI_GOLDEN_OUT  artifact directory
  PEEKABOO_BRIDGE_SOCKET Peekaboo bridge socket
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flow) FLOW="${2:?--flow requires a name}"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

log() { printf '[gui-golden] %s\n' "$*"; }
die() { printf '[gui-golden] FAIL: %s\n' "$*" >&2; exit 1; }
pb() { peekaboo "$@" --bridge-socket "$BRIDGE"; }

cleanup_persona() {
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    APP_PID=""
    if [[ -n "$OUT" && -f "$OUT/fake-events.jsonl" ]]; then
        while read -r fake_pid; do
            [[ "$fake_pid" =~ ^[0-9]+$ ]] || continue
            local command
            command="$(ps -p "$fake_pid" -o command= 2>/dev/null || true)"
            if [[ "$command" == *"serve fake-alias"* ]]; then
                kill "$fake_pid" 2>/dev/null || true
            fi
        done < <(jq -r 'select(.event == "server_started") | .pid' "$OUT/fake-events.jsonl" 2>/dev/null | sort -u)
    fi
    if [[ "$KEEP" == 0 && -n "$BUNDLE_ID" ]]; then
        defaults delete "$BUNDLE_ID" >/dev/null 2>&1 || true
    fi
    if [[ "$KEEP" == 0 && -n "$PERSONA" && -d "$PERSONA" ]]; then
        rm -rf "$PERSONA"
    fi
    PERSONA=""
    BUNDLE_ID=""
}

finish() {
    local status=$?
    set +e
    if [[ "$status" -ne 0 && "$RESULT_WRITTEN" == 0 && -d "$OUT_ROOT" ]]; then
        jq -n --arg status fail --arg flow "$FLOW" --arg app "$APP_SOURCE" \
            --argjson exit_code "$status" \
            '{status: $status, flow: $flow, app: $app, exit_code: $exit_code}' \
            > "$OUT_ROOT/result.json" 2>/dev/null || true
    fi
    cleanup_persona
}
trap finish EXIT
trap 'cleanup_persona; exit 130' INT
trap 'cleanup_persona; exit 143' TERM

require_tools() {
    [[ -d "$APP_SOURCE" ]] || die "built app not found: $APP_SOURCE"
    for tool in peekaboo jq; do
        command -v "$tool" >/dev/null || die "$tool is required"
    done
    AX_DRIVER="$OUT_ROOT/rapid-ax"
    swiftc "$ROOT/scripts/rapid-ax.swift" -o "$AX_DRIVER"
    pb permissions status --json > "$OUT_ROOT/permissions.json"
    jq -e '.success and ([.data.permissions[] | select(.isRequired) | .isGranted] | all)' \
        "$OUT_ROOT/permissions.json" >/dev/null \
        || die "Peekaboo needs Accessibility and Screen Recording permissions"
}

start_persona() {
    local name="$1"
    shift
    cleanup_persona
    OUT="$OUT_ROOT/$name"
    PERSONA="$(mktemp -d "/tmp/rapid-golden-${name}.XXXXXX")"
    mkdir -p "$OUT"
    "$ROOT/scripts/dogfood-isolate.sh" "$APP_SOURCE" "$PERSONA" \
        > "$OUT/isolated-app.txt" 2> "$OUT/isolate.log"
    local isolated_app
    isolated_app="$(cat "$OUT/isolated-app.txt")"
    BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$isolated_app/Contents/Info.plist")"
    local config="$PERSONA/home/.rapid-golden-fake.json"
    jq -n --arg event_log "$OUT/fake-events.jsonl" '{FAKE_EVENT_LOG: $event_log}' > "$config"
    local assignment key value updated
    for assignment in "$@"; do
        key="${assignment%%=*}"
        value="${assignment#*=}"
        updated="$config.next"
        jq --arg key "$key" --arg value "$value" '.[$key] = $value' "$config" > "$updated"
        mv "$updated" "$config"
    done
    env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
        FAKE_EVENT_LOG="$OUT/fake-events.jsonl" "$@" \
        "$PERSONA/launch.sh" > "$OUT/app.log" 2>&1 &
    APP_PID=$!
    wait_for_window
}

relaunch_persona() {
    stop_app
    env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
        FAKE_EVENT_LOG="$OUT/fake-events.jsonl" \
        "$PERSONA/launch.sh" >> "$OUT/app.log" 2>&1 &
    APP_PID=$!
    wait_for_window
}

stop_app() {
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$APP_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -9 "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    APP_PID=""
}

wait_for_window() {
    local windows="$OUT/windows.json"
    for _ in {1..80}; do
        kill -0 "$APP_PID" 2>/dev/null || die "app exited before opening a window"
        pb list windows --app "PID:$APP_PID" --json > "$windows" 2>/dev/null || true
        MAIN_WINDOW_ID="$(jq -r '(.data.windows // []) | map(select(.title == "Rapid-MLX"))[0].window_id // empty' "$windows" 2>/dev/null)"
        [[ -n "$MAIN_WINDOW_ID" ]] && return
        sleep 0.25
    done
    die "main window did not appear"
}

see_main() {
    local destination="$1"
    pb list windows --app "PID:$APP_PID" --json > "$OUT/windows-current.json"
    local current_main
    current_main="$(jq -r '(.data.windows // []) | map(select(.title == "Rapid-MLX"))[0].window_id // empty' "$OUT/windows-current.json")"
    [[ -n "$current_main" ]] && MAIN_WINDOW_ID="$current_main"
    "$AX_DRIVER" dump "$APP_PID" > "$destination"
}

wait_identifier() {
    local identifier="$1" destination="$2" attempts="${3:-80}"
    for ((i=0; i<attempts; i++)); do
        see_main "$destination"
        if jq -e --arg id "$identifier" '.data.ui_elements[]? | select(.identifier == $id)' "$destination" >/dev/null; then
            return
        fi
        sleep 0.25
    done
    die "timed out waiting for AX identifier $identifier"
}

element_field() {
    local tree="$1" identifier="$2" field="$3"
    jq -r --arg id "$identifier" --arg field "$field" \
        '.data.ui_elements[] | select(.identifier == $id) | .[$field] // empty' "$tree" | head -1
}

press() {
    local tree="$1" identifier="$2" evidence="$3"
    jq -e --arg id "$identifier" '.data.ui_elements[]? | select(.identifier == $id)' "$tree" >/dev/null \
        || { printf '[gui-golden] AX identifier missing: %s\n' "$identifier" >&2; return 1; }
    "$AX_DRIVER" press "$APP_PID" "$identifier" > "$evidence" || return 1
    jq -e '.success' "$evidence" >/dev/null \
        || { printf '[gui-golden] AXPress failed: %s\n' "$identifier" >&2; return 1; }
}

dismiss_first_run() {
    local tree="$OUT/first-run.json"
    see_main "$tree"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$tree" >/dev/null; then
        if ! press "$tree" TelemetryConsent.DontShare "$OUT/consent.json" 2>/dev/null; then
            read -r x y < <(jq -r '.data.ui_elements[] | select(.identifier == "TelemetryConsent.DontShare") | [(.bounds.x + .bounds.width / 2), (.bounds.y + .bounds.height / 2)] | @tsv' "$tree")
            pb click --coords "$x,$y" --global-coords --app "PID:$APP_PID" --json > "$OUT/consent-coordinate-fallback.json"
        fi
        sleep 0.5
        see_main "$tree"
    fi
    if jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Skip")' "$tree" >/dev/null; then
        press "$tree" Quickstart.Skip "$OUT/quickstart-skip.json"
        sleep 0.5
        see_main "$tree"
    fi
    if jq -e '.data.ui_elements[]? | select(.identifier == "DockHidePrompt.NoButton")' "$tree" >/dev/null; then
        press "$tree" DockHidePrompt.NoButton "$OUT/dock-no.json"
        sleep 0.5
    fi
    wait_identifier rapid.chat.compose "$OUT/steady.json"
}

open_settings() {
    pb menu click --app "PID:$APP_PID" --item 'Settings…' --json > "$OUT/open-settings.json"
    for _ in {1..40}; do
        pb list windows --app "PID:$APP_PID" --json > "$OUT/settings-windows.json"
        SETTINGS_WINDOW_ID="$(jq -r '.data.windows[]? | select(.title == "Settings") | .window_id' "$OUT/settings-windows.json" | head -1)"
        [[ -n "$SETTINGS_WINDOW_ID" ]] && return
        sleep 0.25
    done
    die "Settings window did not open"
}

see_settings() {
    "$AX_DRIVER" dump "$APP_PID" > "$1"
}

start_model() {
    wait_identifier Readiness.Action "$OUT/readiness-start.json"
    press "$OUT/readiness-start.json" Readiness.Action "$OUT/start-model.json"
    for _ in {1..120}; do
        see_main "$OUT/readiness-ready.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "ChatView.SendOrStopButton" and .description == "Send message")' "$OUT/readiness-ready.json" >/dev/null \
            && grep -q '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null; then return; fi
        sleep 0.25
    done
    die "fake model did not become ready"
}

send_prompt() {
    local prompt="$1" prefix="$2"
    see_main "$OUT/${prefix}-compose.json"
    "$AX_DRIVER" set-value "$APP_PID" rapid.chat.compose "$prompt" > "$OUT/${prefix}-type.json"
    see_main "$OUT/${prefix}-draft.json"
    press "$OUT/${prefix}-draft.json" ChatView.SendOrStopButton "$OUT/${prefix}-send.json"
}

assert_tree_text() {
    local tree="$1" needle="$2"
    jq -e --arg needle "$needle" '(.data.ui_elements | tostring) | contains($needle)' "$tree" >/dev/null \
        || die "AX tree does not contain expected text: $needle"
}

flow_fresh_install() {
    log "1/6 fresh install and onboarding"
    start_persona fresh-install
    see_main "$OUT/consent-visible.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$OUT/consent-visible.json" >/dev/null \
        || die "fresh install did not show telemetry consent"
    dismiss_first_run
    for id in Sidebar.NewChat Sidebar.Launch rapid.chat.compose ChatView.SendOrStopButton ModelPickerBar.ModelMenu; do
        jq -e --arg id "$id" '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/steady.json" >/dev/null \
            || die "post-onboarding shell missing $id"
    done
    pb image --window-id "$MAIN_WINDOW_ID" --path "$OUT/final.png" --json > "$OUT/final-image.json"
    cleanup_persona
}

flow_settings_persistence() {
    log "2/6 settings and persistence"
    start_persona settings-persistence
    dismiss_first_run
    open_settings
    see_settings "$OUT/settings-root.json"
    press "$OUT/settings-root.json" Settings.Category.models "$OUT/settings-models-open.json"
    sleep 0.3
    see_settings "$OUT/models-before.json"
    local preference_key="rapid.picker.show_all_models.v1"
    press "$OUT/models-before.json" Settings.Models.ShowAllModelsToggle "$OUT/models-toggle.json"
    for _ in {1..20}; do
        [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 1 ]] && break
        sleep 0.1
    done
    [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 1 ]] \
        || die "GUI toggle did not persist true to isolated preferences"
    see_settings "$OUT/models-after.json"
    relaunch_persona
    dismiss_first_run
    open_settings
    see_settings "$OUT/settings-relaunch.json"
    press "$OUT/settings-relaunch.json" Settings.Category.models "$OUT/settings-models-reopen.json"
    sleep 0.3
    see_settings "$OUT/models-persisted.json"
    press "$OUT/models-persisted.json" Settings.Models.ShowAllModelsToggle "$OUT/models-toggle-after-relaunch.json"
    for _ in {1..20}; do
        [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 0 ]] && break
        sleep 0.1
    done
    [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 0 ]] \
        || die "relaunch did not restore the persisted toggle state"
    cleanup_persona
}

flow_chat_restore() {
    log "3/6 basic chat and session restore"
    start_persona chat-restore
    dismiss_first_run
    start_model
    send_prompt "golden restore marker" chat
    for _ in {1..100}; do
        see_main "$OUT/chat-complete.json"
        if jq -e '(.data.ui_elements | tostring) | contains("deterministic content")' "$OUT/chat-complete.json" >/dev/null; then break; fi
        sleep 0.2
    done
    assert_tree_text "$OUT/chat-complete.json" "golden restore marker"
    assert_tree_text "$OUT/chat-complete.json" "deterministic content"
    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/chat-restored.json"
    assert_tree_text "$OUT/chat-restored.json" "golden restore marker"
    local conversation_id
    conversation_id="$(jq -r '.data.ui_elements[] | select((.identifier // "") | startswith("Sidebar.Conversation.")) | .identifier' "$OUT/chat-restored.json" | head -1)"
    [[ -n "$conversation_id" ]] || die "restored conversation row was not exposed to AX"
    press "$OUT/chat-restored.json" "$conversation_id" "$OUT/open-restored-conversation.json"
    sleep 0.2
    see_main "$OUT/chat-restored-transcript.json"
    assert_tree_text "$OUT/chat-restored-transcript.json" "deterministic content"
    cleanup_persona
}

flow_slow_stream_stop() {
    log "4/6 controlled slow stream and Stop"
    start_persona slow-stream-stop FAKE_INTER_TOKEN_SLEEP_S=0.01 FAKE_CONTENT_REPEAT=20000
    dismiss_first_run
    start_model
    send_prompt "golden stop marker" slow
    for _ in {1..40}; do
        see_main "$OUT/slow-streaming.json"
        if [[ "$(element_field "$OUT/slow-streaming.json" ChatView.SendOrStopButton description)" == "Stop generating" ]]; then break; fi
        sleep 0.1
    done
    [[ "$(element_field "$OUT/slow-streaming.json" ChatView.SendOrStopButton description)" == "Stop generating" ]] \
        || die "send button never transitioned to Stop generating"
    press "$OUT/slow-streaming.json" ChatView.SendOrStopButton "$OUT/slow-stop.json"
    for _ in {1..40}; do
        see_main "$OUT/slow-stopped.json"
        [[ "$(element_field "$OUT/slow-stopped.json" ChatView.SendOrStopButton description)" == "Send message" ]] && break
        sleep 0.1
    done
    [[ "$(element_field "$OUT/slow-stopped.json" ChatView.SendOrStopButton description)" == "Send message" ]] \
        || die "Stop did not restore Send state"
    for _ in {1..100}; do
        grep -q '"event": "chat_cancelled"' "$OUT/fake-events.jsonl" 2>/dev/null && break
        sleep 0.05
    done
    grep -q '"event": "chat_cancelled"' "$OUT/fake-events.jsonl" \
        || die "fake server did not observe stream cancellation"
    if grep -q '"event": "chat_finished"' "$OUT/fake-events.jsonl"; then
        die "slow response finished instead of being stopped early"
    fi
    jq -n '{success: true, assertion: "UI returned to Send and server observed cancellation"}' \
        > "$OUT/stop-assertion.json"
    cleanup_persona
}

flow_model_crash_recovery() {
    log "5/6 model lifecycle and crash recovery"
    start_persona model-crash-recovery FAKE_DIE_AFTER_CHUNKS=2 \
        FAKE_DIE_ONCE_STATE="$OUT_ROOT/model-crash-recovery/died-once"
    dismiss_first_run
    start_model
    send_prompt "golden crash marker" crash
    for _ in {1..160}; do
        see_main "$OUT/crash-recovered.json"
        local starts
        starts="$(grep -c '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null || true)"
        if [[ "$starts" -ge 2 ]] && jq -e '.data.ui_elements[]? | select(.identifier == "ChatView.SendOrStopButton")' "$OUT/crash-recovered.json" >/dev/null; then
            break
        fi
        sleep 0.25
    done
    [[ "$(grep -c '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null || true)" -ge 2 ]] \
        || die "server did not respawn after the simulated crash"
    for _ in {1..80}; do
        see_main "$OUT/crash-ready.json"
        if [[ "$(element_field "$OUT/crash-ready.json" ChatView.SendOrStopButton description)" == "Send message" ]]; then break; fi
        sleep 0.25
    done
    [[ "$(element_field "$OUT/crash-ready.json" ChatView.SendOrStopButton description)" == "Send message" ]] \
        || die "model was not ready after crash recovery"
    jq -n --argjson starts "$(grep -c '"event": "server_started"' "$OUT/fake-events.jsonl")" \
        '{success: true, assertion: "sidecar crashed once, respawned, and returned to ready", server_starts: $starts}' \
        > "$OUT/recovery-assertion.json"
    cleanup_persona
}

flow_low_memory_choice() {
    log "6/6 low-memory onboarding escape"
    start_persona low-memory-choice

    local tree="$OUT/onboarding.json"
    see_main "$tree"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$tree" >/dev/null; then
        press "$tree" TelemetryConsent.DontShare "$OUT/consent.json"
    fi
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_identifier Quickstart.Choice.qwen3-0.6b-4bit "$OUT/model-choices.json"

    local fallback_label
    fallback_label="$(element_field "$OUT/model-choices.json" Quickstart.Choice.qwen3-0.6b-4bit description)"
    [[ "$fallback_label" == *"Lowest memory"* ]] \
        || die "low-memory choice is missing its spoken category label"
    [[ "$fallback_label" == *"less accurate"* ]] \
        || die "low-memory choice hides its quality trade-off"
    [[ "$fallback_label" == *"not recommended for tools"* ]] \
        || die "low-memory choice hides its tool-use limitation"
    press "$OUT/model-choices.json" Quickstart.Choice.qwen3-0.6b-4bit "$OUT/select-low-memory.json"
    see_main "$OUT/low-memory-selected.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Choice.qwen3-0.6b-4bit")' \
        "$OUT/low-memory-selected.json" >/dev/null \
        || die "selecting the low-memory choice dismissed or replaced the chooser"
    jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Footer.Primary")' \
        "$OUT/low-memory-selected.json" >/dev/null \
        || die "selecting the low-memory choice left no Download & start action"
    local sheet_region
    sheet_region="$(jq -r '.data.ui_elements[] | select(.role == "AXSheet") | [.bounds.x, .bounds.y, .bounds.width, .bounds.height] | map(round) | @csv' "$OUT/low-memory-selected.json" | head -1)"
    [[ -n "$sheet_region" ]] || die "Quickstart sheet bounds are absent from AX"
    pb app switch --to "PID:$APP_PID" --verify --json > "$OUT/focus-before-image.json"
    pb image --mode area --region "$sheet_region" --path "$OUT/low-memory-selected.png" --json \
        > "$OUT/low-memory-selected-image.json"

    jq -n '{success: true, assertion: "onboarding exposes and selects an honestly labelled sub-1B low-memory fallback"}' \
        > "$OUT/low-memory-assertion.json"
    cleanup_persona
}

flow_update_state() {
    # Settings > App must name the version the app actually IS.
    #
    # This is the cheap end of a real failure: the update manifest the app
    # falls back on (dl.rapidmlx.com/latest.json) sat at 0.11.0 for four
    # releases (#1612). Anything consuming a stale manifest reports a version
    # that disagrees with CFBundleShortVersionString, and this assertion
    # catches exactly that mismatch without needing network state.
    start_persona update-state
    dismiss_first_run
    open_settings
    see_main "$OUT/update-settings.json"
    press "$OUT/update-settings.json" Settings.Category.app "$OUT/update-open-app.json"
    wait_identifier Settings.App.UpToDate "$OUT/update-app-panel.json"

    local shown expected
    shown="$(element_field "$OUT/update-app-panel.json" Settings.App.UpToDate value)"
    expected="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' \
        "$APP_SOURCE/Contents/Info.plist" 2>/dev/null)"
    [[ -n "$expected" ]] || die "could not read CFBundleShortVersionString"
    [[ "$shown" == *"$expected"* ]] \
        || die "update panel says '$shown' but the app is $expected"
    log "  update state names the running version ($expected)"
    cleanup_persona
}

flow_no_dead_controls() {
    # Every advertised Settings control must do something observable.
    #
    # Journey-shaped flows never found this class; an inventory-shaped one
    # finds all of it. Recovery buttons that highlighted, accepted the click
    # and did nothing (#1595); toggles that reported success without changing
    # value (#1608); a tray item that fired and reported nowhere (#1605).
    start_persona no-dead-controls
    dismiss_first_run
    open_settings
    see_main "$OUT/dead-before.json"

    local category
    for category in models modelManagement tools appearance privacy app; do
        press "$OUT/dead-before.json" "Settings.Category.$category" \
            "$OUT/dead-open-$category.json" \
            || die "Settings category $category is not pressable"
        see_main "$OUT/dead-panel-$category.json"
        # Count only the PANEL's own controls. The six `Settings.Category.*`
        # buttons are present on every panel, so counting all `Settings.*`
        # identifiers is vacuous — it goes green on a completely unlabelled
        # panel, which is precisely the state Tools is in today.
        local count
        count="$(jq '[.data.ui_elements[]?
                      | select((.identifier // "") | startswith("Settings."))
                      | select((.identifier // "") | startswith("Settings.Category.") | not)]
                     | length' "$OUT/dead-panel-$category.json")"
        [[ "$count" -gt 0 ]] \
            || die "Settings > $category exposes no identified controls of its own"
        log "  $category: $count identified controls"
    done
    cleanup_persona
}

flow_catalog_integrity() {
    # A model that cannot chat must never be offered as one.
    #
    # Eight video-generation aliases reached the picker and Model Management
    # looking ordinary; selecting one dead-ended at "Couldn't start ... Try
    # again" forever, reachable AFTER downloading up to 64 GB (#1603). The
    # fake sidecar emits a `[video:gen]`-tagged row so this proves the FILTER,
    # not today's registry contents.
    start_persona catalog-integrity
    dismiss_first_run
    see_main "$OUT/catalog-main.json"

    jq -e '[.data.ui_elements[]? | select([(.identifier // ""), (.value // ""), (.title // ""), (.description // "")] | map(tostring) | join(" ") | test("fake-video-alias"))] | length == 0' \
        "$OUT/catalog-main.json" >/dev/null \
        || die "a video-gen alias reached the chat surface"

    open_settings
    see_main "$OUT/catalog-settings.json"
    press "$OUT/catalog-settings.json" Settings.Category.modelManagement \
        "$OUT/catalog-open-mm.json"
    see_main "$OUT/catalog-model-management.json"
    jq -e '[.data.ui_elements[]? | select([(.identifier // ""), (.value // ""), (.title // ""), (.description // "")] | map(tostring) | join(" ") | test("fake-video-alias"))] | length == 0' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "a video-gen alias reached Model Management"
    log "  no video-gen alias on either catalog surface"
    cleanup_persona
}

if [[ -d "$OUT_ROOT" && -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]]; then
    RESULT_WRITTEN=1
    die "artifact directory is not empty: $OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"
require_tools
case "$FLOW" in
    fresh-install) flow_fresh_install ;;
    settings-persistence) flow_settings_persistence ;;
    chat-restore) flow_chat_restore ;;
    slow-stream-stop) flow_slow_stream_stop ;;
    model-crash-recovery) flow_model_crash_recovery ;;
    low-memory-choice) flow_low_memory_choice ;;
    update-state) flow_update_state ;;
    no-dead-controls) flow_no_dead_controls ;;
    catalog-integrity) flow_catalog_integrity ;;
    all)
        flow_fresh_install
        flow_settings_persistence
        flow_chat_restore
        flow_slow_stream_stop
        flow_model_crash_recovery
        flow_low_memory_choice
        flow_update_state
        flow_no_dead_controls
        flow_catalog_integrity
        ;;
    *) die "unknown flow: $FLOW" ;;
esac

jq -n --arg status pass --arg flow "$FLOW" --arg app "$APP_SOURCE" \
    '{status: $status, flow: $flow, app: $app}' > "$OUT_ROOT/result.json"
RESULT_WRITTEN=1
log "PASS — $FLOW"
log "artifacts: $OUT_ROOT"
