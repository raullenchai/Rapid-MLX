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
BASELINE_TOOL="$ROOT/scripts/ax-baseline.py"
BASELINE_DIR="${RAPID_GUI_BASELINE_DIR:-$ROOT/Tests/GUIGoldenFlows/__Snapshots__}"
# The fixture alias is scrubbed out of baselines so renaming the fake model
# is not a structural change to the UI.
FAKE_ALIAS="fake-alias"
UPDATE_BASELINES=0
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
Usage: gui-golden-flows.sh [--flow NAME] [--keep] [--update-baselines]

Flows: fresh-install, settings-persistence, chat-restore, slow-stream-stop,
       model-crash-recovery, low-memory-choice, loaded-model-benchmark,
       update-state, no-dead-controls, catalog-integrity,
       browse-all-destination, all

Options:
  --update-baselines  rewrite the committed AX structural baselines instead of
                      comparing against them. Intended UI changes land as a
                      reviewable diff under Tests/GUIGoldenFlows/__Snapshots__.

Environment:
  RAPID_GUI_SOURCE_APP   built .app to test
  RAPID_GUI_GOLDEN_OUT  artifact directory
  RAPID_GUI_BASELINE_DIR AX structural baseline directory
  PEEKABOO_BRIDGE_SOCKET Peekaboo bridge socket
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flow) FLOW="${2:?--flow requires a name}"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --update-baselines) UPDATE_BASELINES=1; shift ;;
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
    for tool in peekaboo jq python3; do
        command -v "$tool" >/dev/null || die "$tool is required"
    done
    [[ -f "$BASELINE_TOOL" ]] || die "AX baseline normalizer not found: $BASELINE_TOOL"
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

# AX can expose a newly opened/refocused Settings window before AppKit has
# finished realizing every window subtree. A single immediate dump therefore
# sometimes loses the main-window children or the category rail entirely.
# Require the semantic anchors we care about and two identical normalized
# trees before snapshotting or pressing from the tree.
wait_settings_stable() {
    local destination="$1"
    shift
    local candidate="$destination.candidate" previous="$destination.previous.txt"
    local normalized="$destination.normalized.txt" stable=0
    rm -f "$previous"
    for _ in {1..80}; do
        see_settings "$candidate"
        local complete=1 identifier
        for identifier in rapid.chat.compose Settings.Category.modelManagement "$@"; do
            jq -e --arg id "$identifier" \
                '.data.ui_elements[]? | select(.identifier == $id)' \
                "$candidate" >/dev/null || { complete=0; break; }
        done
        if [[ "$complete" == 1 ]]; then
            python3 "$BASELINE_TOOL" normalize "$candidate" --scrub "$FAKE_ALIAS" \
                --output "$normalized"
            if [[ -f "$previous" ]] && cmp -s "$previous" "$normalized"; then
                stable=$((stable + 1))
                if [[ "$stable" -ge 1 ]]; then
                    mv "$candidate" "$destination"
                    rm -f "$previous" "$normalized"
                    return
                fi
            else
                stable=0
            fi
            cp "$normalized" "$previous"
        else
            stable=0
            rm -f "$previous"
        fi
        sleep 0.25
    done
    die "Settings AX tree did not settle with required identifiers: $*"
}

start_model() {
    wait_identifier Readiness.Action "$OUT/readiness-start.json"
    press "$OUT/readiness-start.json" Readiness.Action "$OUT/start-model.json"
    # ``server_started`` says the fake bound its port; it does NOT say the app
    # has finished wiring up to it. The old gate also tested
    # ``description == "Send message"``, which is the button's label for the
    # whole startup — including while its hint still reads "<alias> is still
    # starting." So this returned early, ``send_prompt`` pressed into a closed
    # readiness gate, and the press was silently dropped (observed: 1 run in 2).
    for _ in {1..120}; do
        grep -q '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null && break
        sleep 0.25
    done
    grep -q '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null \
        || die "fake model did not become ready"
    wait_send_idle "$OUT/readiness-ready.json"
}

send_prompt() {
    local prompt="$1" prefix="$2"
    see_main "$OUT/${prefix}-compose.json"
    "$AX_DRIVER" set-value "$APP_PID" rapid.chat.compose "$prompt" > "$OUT/${prefix}-type.json"
    see_main "$OUT/${prefix}-draft.json"
    press "$OUT/${prefix}-draft.json" ChatView.SendOrStopButton "$OUT/${prefix}-send.json"
    # A press that lands while the gate is closed is dropped and the draft
    # stays in the composer — where ``assert_tree_text`` happily FINDS the
    # prompt and reports a message that was never sent. Requiring the composer
    # to drain is what makes that failure loud instead of silent.
    #
    # ``has("value")`` rather than ``.value // ""``: rapid-ax OMITS an attribute
    # whose AX read failed, so a defaulting test reads a failed read as "drained"
    # and rebuilds the very false green this exists to stop. And the composer
    # clearing is the app's story about itself — the fake's ``chat_request`` is
    # the independent witness that a request actually left the process.
    for _ in {1..40}; do
        see_main "$OUT/${prefix}-sent.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "rapid.chat.compose")
                  | select(has("value") and .value == "")' "$OUT/${prefix}-sent.json" >/dev/null \
           && grep -q '"event": "chat_request"' "$OUT/fake-events.jsonl" 2>/dev/null; then
            return
        fi
        sleep 0.25
    done
    die "no chat_request reached the sidecar, or the composer never drained: the message was never sent"
}

assert_tree_text() {
    local tree="$1" needle="$2"
    jq -e --arg needle "$needle" '(.data.ui_elements | tostring) | contains($needle)' "$tree" >/dev/null \
        || die "AX tree does not contain expected text: $needle"
}

# Structural baseline for a settled UI state. The dump is normalized (see
# scripts/ax-baseline.py) and compared against a committed tree, so a control
# that vanishes, moves in the hierarchy, changes identifier or flips
# enabled/disabled becomes a reviewable diff. Colour, spacing and typography
# are NOT covered — those stay with the PNG snapshots in Tests/RapidTests.
baseline() {
    local name="$1" tree="$2"
    local committed="$BASELINE_DIR/$name.txt"
    local observed="$OUT/$name.observed.txt"
    if [[ "$UPDATE_BASELINES" == 1 ]]; then
        python3 "$BASELINE_TOOL" check "$tree" --scrub "$FAKE_ALIAS" \
            --baseline "$committed" --observed "$observed" --update \
            || die "could not update AX structural baseline: $name"
    else
        python3 "$BASELINE_TOOL" check "$tree" --scrub "$FAKE_ALIAS" \
            --baseline "$committed" --observed "$observed" \
            || die "AX structural baseline mismatch: $name"
    fi
}

# Wait until the composer is genuinely idle before fingerprinting the tree.
#
# ``ChatView.SendOrStopButton`` publishes ``AXHelp`` only while the readiness
# gate is closed (``accessibilityHint`` is empty once ``sendAllowed`` is true),
# so its absence is a copy-independent "the model is ready" signal. The
# description check adds "no stream in flight". Without this the crash-recovery
# tree was captured mid-restart on roughly half of all runs: the button already
# reads "Send message" while the sidecar is still loading, and the transient
# "Starting …" banner then appeared in one run's baseline and not the next.
#
# Readiness is the ABSENCE of AXHelp, and there is deliberately no positive
# attribute to test instead: the button is `enabled=false` in every settled
# state, because a drained composer has nothing to send. So "not ready" and
# "ready with an empty box" differ only by the hint.
#
# That makes a *failed* AX read indistinguishable from readiness — rapid-ax
# omits an attribute it could not read. Mitigated by requiring the element's
# other attributes to have been read successfully in the same pass
# (`has("description")`, `has("enabled")`): an isolated failure of the help
# read alone, with its siblings intact, is the only remaining hole, and it has
# to happen twice in a row because the state must also be STABLE across two
# consecutive dumps. The dump walks the readiness banner before the send
# button, so a single dump can be a hybrid of two states.
wait_send_idle() {
    local destination="$1" attempts="${2:-160}" stable=0
    for ((i=0; i<attempts; i++)); do
        see_main "$destination"
        if jq -e '.data.ui_elements[]? | select(.identifier == "ChatView.SendOrStopButton"
                  and has("description") and .description == "Send message"
                  and has("enabled") and (has("help") | not))' \
            "$destination" >/dev/null; then
            stable=$((stable + 1))
            [[ "$stable" -ge 2 ]] && return
        else
            stable=0
        fi
        sleep 0.25
    done
    die "composer never settled into a ready, non-streaming state"
}

flow_fresh_install() {
    log "1/6 fresh install and onboarding"
    start_persona fresh-install
    see_main "$OUT/consent-visible.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$OUT/consent-visible.json" >/dev/null \
        || die "fresh install did not show telemetry consent"
    baseline fresh-install.consent "$OUT/consent-visible.json"
    dismiss_first_run
    for id in Sidebar.NewChat Sidebar.Launch rapid.chat.compose ChatView.SendOrStopButton ModelPickerBar.ModelMenu; do
        jq -e --arg id "$id" '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/steady.json" >/dev/null \
            || die "post-onboarding shell missing $id"
    done
    baseline fresh-install.steady "$OUT/steady.json"
    pb image --window-id "$MAIN_WINDOW_ID" --path "$OUT/final.png" --json > "$OUT/final-image.json"
    cleanup_persona
}

flow_settings_persistence() {
    log "2/6 settings and persistence"
    start_persona settings-persistence
    dismiss_first_run
    open_settings
    wait_settings_stable "$OUT/settings-root.json"
    baseline settings-persistence.settings-root "$OUT/settings-root.json"
    press "$OUT/settings-root.json" Settings.Category.modelManagement "$OUT/settings-models-open.json"
    wait_settings_stable "$OUT/models-before.json" Settings.Models.ShowAllModelsToggle
    baseline settings-persistence.models-idle "$OUT/models-before.json"
    local preference_key="rapid.picker.show_all_models.v1"
    press "$OUT/models-before.json" Settings.Models.ShowAllModelsToggle "$OUT/models-toggle.json"
    for _ in {1..20}; do
        [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 1 ]] && break
        sleep 0.1
    done
    [[ "$(defaults read "$BUNDLE_ID" "$preference_key" 2>/dev/null || true)" == 1 ]] \
        || die "GUI toggle did not persist true to isolated preferences"
    wait_settings_stable "$OUT/models-after.json" Settings.Models.ShowAllModelsToggle
    baseline settings-persistence.models-toggled "$OUT/models-after.json"
    relaunch_persona
    dismiss_first_run
    open_settings
    wait_settings_stable "$OUT/settings-relaunch.json"
    press "$OUT/settings-relaunch.json" Settings.Category.modelManagement "$OUT/settings-models-reopen.json"
    wait_settings_stable "$OUT/models-persisted.json" Settings.Models.ShowAllModelsToggle
    baseline settings-persistence.models-after-relaunch "$OUT/models-persisted.json"
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
    # The loop above breaks as soon as the transcript mentions "deterministic
    # content", which the fake emits nine chunks before the stream ends. Settle
    # first so the baseline is the finished turn, not a partial one.
    wait_send_idle "$OUT/chat-settled.json"
    baseline chat-restore.answered "$OUT/chat-settled.json"
    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/chat-restored.json"
    assert_tree_text "$OUT/chat-restored.json" "golden restore marker"
    local conversation_id
    # Match the ROW exactly. `Sidebar.Conversation.` is now a namespace, not a
    # row: it also contains `…Pin.<uuid>`, `…Unpin.<uuid>`, `…Menu.<uuid>` and
    # `…Action.*`. A prefix match can select the pin button or the ··· menu and
    # press that instead of opening the conversation — and because the restored
    # transcript is asserted *before* this press, the flow would still pass.
    conversation_id="$(jq -r '.data.ui_elements[] | (.identifier // "")
        | select(test("^Sidebar\\.Conversation\\.[0-9A-Fa-f-]{36}$"))' \
        "$OUT/chat-restored.json" | head -1)"
    [[ -n "$conversation_id" ]] || die "restored conversation row was not exposed to AX"
    press "$OUT/chat-restored.json" "$conversation_id" "$OUT/open-restored-conversation.json"
    sleep 0.2
    see_main "$OUT/chat-restored-transcript.json"
    assert_tree_text "$OUT/chat-restored-transcript.json" "deterministic content"
    wait_send_idle "$OUT/chat-restored-settled.json"
    baseline chat-restore.transcript-restored "$OUT/chat-restored-settled.json"
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
    wait_send_idle "$OUT/slow-settled.json"
    baseline slow-stream-stop.stopped "$OUT/slow-settled.json"
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
    wait_send_idle "$OUT/crash-settled.json"
    baseline model-crash-recovery.recovered "$OUT/crash-settled.json"
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

flow_loaded_model_benchmark() {
    log "7/7 benchmark the model that is already loaded"
    start_persona loaded-model-benchmark
    pb app switch --to "PID:$APP_PID" --verify --json > "$OUT/focus.json"
    dismiss_first_run
    start_model

    wait_identifier ChatView.SpeedOnThisMac "$OUT/chat-ready.json"
    press "$OUT/chat-ready.json" ChatView.SpeedOnThisMac "$OUT/open-benchmark.json"
    wait_identifier Benchmark.RunLoadedModel "$OUT/benchmark-idle.json"
    press "$OUT/benchmark-idle.json" Benchmark.RunLoadedModel "$OUT/run-benchmark.json"
    wait_identifier Benchmark.LoadedModelResult "$OUT/benchmark-result.json"

    local starts requests
    starts="$(grep -c '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null || true)"
    requests="$(grep -c '"event": "benchmark_request"' "$OUT/fake-events.jsonl" 2>/dev/null || true)"
    [[ "$starts" == 1 ]] \
        || die "speed test started a second server/model process ($starts starts)"
    [[ "$requests" == 2 ]] \
        || die "speed test did not send warm-up + measured requests to the loaded server ($requests requests)"
    jq -n --argjson starts "$starts" --argjson requests "$requests" \
        '{success: true, assertion: "speed test reused the loaded model", server_starts: $starts, benchmark_requests: $requests}' \
        > "$OUT/loaded-model-benchmark-assertion.json"
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

    # TWO identifiers satisfy this invariant, and which one appears depends on
    # something outside the app: whether a release for this version has been
    # published yet.
    #
    #   Settings.App.UpToDate        — the build matches the newest published release
    #   Settings.App.AheadOfManifest — the build is NEWER than anything published
    #
    # The second is not an edge case, it is the state every release passes
    # through: the version is bumped and the app is built before its release is
    # tagged. Waiting only for UpToDate made this flow fail during exactly the
    # window it is meant to protect — cutting 0.12.7, with the app correctly
    # reporting "Up to date — v0.12.7." under the other identifier.
    #
    # The invariant is unchanged: whichever state the panel is in, it must name
    # the version the app actually IS.
    local state shown expected
    state=""
    for _ in {1..80}; do
        see_settings "$OUT/update-app-panel.json"
        for candidate in Settings.App.UpToDate Settings.App.AheadOfManifest; do
            if jq -e --arg id "$candidate" '.data.ui_elements[]? | select(.identifier == $id)' \
                "$OUT/update-app-panel.json" >/dev/null 2>&1; then
                state="$candidate"
                break 2
            fi
        done
        sleep 0.25
    done
    [[ -n "$state" ]] \
        || die "Settings > App reported neither Settings.App.UpToDate nor Settings.App.AheadOfManifest"

    shown="$(element_field "$OUT/update-app-panel.json" "$state" value)"
    expected="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' \
        "$APP_SOURCE/Contents/Info.plist" 2>/dev/null)"
    [[ -n "$expected" ]] || die "could not read CFBundleShortVersionString"
    [[ "$shown" == *"$expected"* ]] \
        || die "update panel ($state) says '$shown' but the app is $expected"
    log "  update state names the running version ($expected, via ${state##*.})"
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
    for category in modelManagement tools appearance privacy app; do
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

flow_browse_all_destination() {
    # An advertised destination must actually be one, and must not cost the
    # user what they already chose.
    #
    # "Browse all models →" on Quickstart step 2 was implemented as one line
    # that set a dismiss flag (#1653). It was present, enabled, correctly
    # labelled and carried an AXIdentifier, so every structural check passed —
    # the wizard simply vanished, the user's pick was discarded, and they
    # landed on whatever the alphabetical fallback chose (a 7.6 GB download
    # nobody asked for). None of that is visible in a tree dump. This flow
    # presses the control and drives the whole round trip.
    start_persona browse-all-destination

    # Only the consent sheet — the wizard has to stay up, it is the subject.
    local tree="$OUT/ba-first-run.json"
    see_main "$tree"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$tree" >/dev/null; then
        press "$tree" TelemetryConsent.DontShare "$OUT/ba-consent.json" \
            || die "could not answer the telemetry consent sheet"
        sleep 0.5
    fi

    wait_identifier Quickstart.GetStarted "$OUT/ba-welcome.json"
    press "$OUT/ba-welcome.json" Quickstart.GetStarted "$OUT/ba-get-started.json" \
        || die "Quickstart.GetStarted is not pressable"
    wait_identifier Quickstart.BrowseAll "$OUT/ba-chooser.json"

    # Choose a card that is NOT the default. The bug discarded the user's
    # selection; asserting the survival of a pick nobody made proves nothing,
    # so make one, and make it a different one.
    local chosen
    chosen="$(jq -r '[.data.ui_elements[]?
                      | select((.identifier // "") | startswith("Quickstart.Choice."))
                      | select(.selected != true)][0].identifier // empty' \
              "$OUT/ba-chooser.json")"
    [[ -n "$chosen" ]] || die "the chooser offers no unselected model card to pick"
    press "$OUT/ba-chooser.json" "$chosen" "$OUT/ba-choose.json" \
        || die "$chosen is not pressable"
    sleep 0.5
    see_main "$OUT/ba-chosen.json"
    jq -e --arg id "$chosen" '.data.ui_elements[]? | select(.identifier == $id) | select(.selected == true)' \
        "$OUT/ba-chosen.json" >/dev/null \
        || die "pressing $chosen did not select it — the chooser cannot record a choice"
    log "  chose $chosen"

    press "$OUT/ba-chosen.json" Quickstart.BrowseAll "$OUT/ba-press.json" \
        || die "Quickstart.BrowseAll is not pressable"

    # 1. It opened the catalogue. `open_settings` drives the menu, so assert
    #    the window the BUTTON opened rather than opening one ourselves.
    local i
    for ((i=0; i<40; i++)); do
        pb list windows --app "PID:$APP_PID" --json > "$OUT/ba-windows.json"
        SETTINGS_WINDOW_ID="$(jq -r '.data.windows[]? | select(.title == "Settings") | .window_id' "$OUT/ba-windows.json" | head -1)"
        [[ -n "$SETTINGS_WINDOW_ID" ]] && break
        sleep 0.25
    done
    [[ -n "$SETTINGS_WINDOW_ID" ]] \
        || die "Browse all models did not open anything — it is a dismiss button again (#1653)"

    # 2. On the models tab, not merely "Settings somewhere". The wizard's own
    #    copy promises the catalogue; landing on the user's last-used tab is a
    #    different bug wearing the same green check.
    wait_settings_stable "$OUT/ba-settings.json" Settings.Models.ShowAllModelsToggle
    log "  landed on Model Management"

    # 3. Settings is actually USABLE, not merely present. A window opened
    #    behind a modal sheet still publishes its whole subtree to AX, and
    #    AXUIElementPerformAction reaches it there too — so neither the tree
    #    nor an AXPress can tell a usable window from a trapped one. Focus it,
    #    click it the way a person would, and require the panel to change.
    pb window focus --window-id "$SETTINGS_WINDOW_ID" --json > "$OUT/ba-focus.json" \
        || die "could not focus the Settings window the button opened"
    # Coordinates re-read AFTER the focus, because focusing can raise or move
    # the window and a stale point would click whatever now sits there.
    see_settings "$OUT/ba-focused.json"
    local cx cy
    read -r cx cy < <(jq -r '.data.ui_elements[]
                             | select(.identifier == "Settings.Category.privacy")
                             | [(.bounds.x + .bounds.width / 2), (.bounds.y + .bounds.height / 2)]
                             | @tsv' "$OUT/ba-focused.json")
    [[ -n "$cx" && -n "$cy" ]] || die "Settings.Category.privacy has no bounds to click"
    # ``--foreground`` is the whole point. Peekaboo's default is background
    # delivery — a coordinate hit-test followed by an accessibility action,
    # which reaches UI a person cannot, and is therefore exactly as blind to
    # "trapped behind a modal sheet" as the AXPress this replaced.
    # ``--window-id`` also pins the click to the Settings window rather than
    # whatever else the app has on screen at that point.
    pb click --coords "$cx,$cy" --global-coords --foreground \
        --window-id "$SETTINGS_WINDOW_ID" --json > "$OUT/ba-click.json" \
        || die "the Settings window did not accept a real click — it is behind the wizard sheet"
    # A real click that changed nothing is the same failure as no click at all,
    # so require the panel's own control to appear, not merely that the press
    # returned success.
    wait_settings_stable "$OUT/ba-privacy.json" Settings.Privacy.TelemetryToggle
    log "  Settings is focused and responds to a real click"

    # 4. Close it, the way the user would, and land back on the wizard with
    #    the same pick. This is the half the bug actually broke.
    #
    #    Scoped to the window, not the app: ``menu click --app`` routes Close
    #    to whichever window is key, which on a bad day is the main one — and
    #    then every assertion below runs against a wizard that was never
    #    actually returned to.
    pb menu click --window-id "$SETTINGS_WINDOW_ID" --item 'Close' --json > "$OUT/ba-close.json" \
        || die "could not close the Settings window"
    local closed=0
    for ((i=0; i<40; i++)); do
        pb list windows --app "PID:$APP_PID" --json > "$OUT/ba-windows-after.json"
        if jq -e '[.data.windows[]? | select(.title == "Settings")] | length == 0' \
            "$OUT/ba-windows-after.json" >/dev/null; then closed=1; break; fi
        sleep 0.25
    done
    # Not a cosmetic check: with Settings still open, the app-wide AX dump
    # below carries the wizard AND the Settings tree, so the round-trip
    # assertion would pass without any round trip having happened.
    [[ "$closed" == 1 ]] || die "the Settings window did not close — the round trip below would be vacuous"

    wait_identifier Quickstart.BrowseAll "$OUT/ba-after.json"
    jq -e --arg id "$chosen" '.data.ui_elements[]? | select(.identifier == $id) | select(.selected == true)' \
        "$OUT/ba-after.json" >/dev/null \
        || die "the wizard came back without the user's selection — browsing must not discard it (#1653)"
    log "  back on the wizard, $chosen still selected"
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
    loaded-model-benchmark) flow_loaded_model_benchmark ;;
    update-state) flow_update_state ;;
    no-dead-controls) flow_no_dead_controls ;;
    catalog-integrity) flow_catalog_integrity ;;
    browse-all-destination) flow_browse_all_destination ;;
    all)
        flow_fresh_install
        flow_settings_persistence
        flow_chat_restore
        flow_slow_stream_stop
        flow_model_crash_recovery
        flow_low_memory_choice
        flow_loaded_model_benchmark
        flow_update_state
        flow_no_dead_controls
        flow_catalog_integrity
        flow_browse_all_destination
        ;;
    *) die "unknown flow: $FLOW" ;;
esac

jq -n --arg status pass --arg flow "$FLOW" --arg app "$APP_SOURCE" \
    '{status: $status, flow: $flow, app: $app}' > "$OUT_ROOT/result.json"
RESULT_WRITTEN=1
log "PASS — $FLOW"
log "artifacts: $OUT_ROOT"
