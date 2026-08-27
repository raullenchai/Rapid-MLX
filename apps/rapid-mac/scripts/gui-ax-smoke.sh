#!/usr/bin/env bash
# AX-first Rapid-Mac GUI smoke using Peekaboo.
#
# This deliberately avoids model startup and mutable controls. It validates
# that the running app exposes stable semantic selectors, opens Settings via
# the application menu, navigates with AXPress, and records structured trees
# plus screenshots for diagnosis.
set -euo pipefail

APP="${RAPID_GUI_APP:-Rapid-MLX}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RAPID_GUI_OUT:-/tmp/rapid-gui-ax-${STAMP}}"
BRIDGE="${PEEKABOO_BRIDGE_SOCKET:-$HOME/Library/Application Support/Peekaboo/daemon.sock}"
mkdir -p "$OUT"

die() { printf 'gui-ax-smoke: FAIL: %s\n' "$*" >&2; exit 1; }
log() { printf 'gui-ax-smoke: %s\n' "$*"; }
pb() { peekaboo "$@" --bridge-socket "$BRIDGE"; }

observe_app() {
    local destination="$1"
    pb list windows --app "$APP" --json > "$OUT/windows-current.json"
    MAIN_WINDOW_ID="$(jq -r --arg app "$APP" \
        '.data.windows | (map(select(.title == $app and .isMainWindow == true))[0] // map(select(.title == $app))[0]) | .window_id // empty' \
        "$OUT/windows-current.json" | head -1)"
    [[ -n "$MAIN_WINDOW_ID" ]] || die "main window for $APP not found"
    pb see --window-id "$MAIN_WINDOW_ID" --json > "$destination" || true
    if ! jq -e '.success' "$destination" >/dev/null; then
        # The first sizeable untitled window is the frontmost SwiftUI sheet.
        local sheet_window_id
        sheet_window_id="$(jq -r \
            '.data.windows[] | select(.title == "" and .bounds[1][0] >= 400 and .bounds[1][1] >= 200) | .window_id' \
            "$OUT/windows-current.json" | head -1)"
        [[ -n "$sheet_window_id" ]] || return 1
        pb see --window-id "$sheet_window_id" --json > "$destination"
    fi
    jq -e '.success' "$destination" >/dev/null
}

press_identifier() {
    local tree="$1" identifier="$2" output="$3"
    local element snapshot
    element="$(jq -r --arg id "$identifier" \
        '.data.ui_elements[] | select(.identifier == $id) | .id' "$tree" | head -1)"
    snapshot="$(jq -r '.data.snapshot_id' "$tree")"
    [[ -n "$element" ]] || return 1
    pb perform-action --on "$element" --action AXPress --snapshot "$snapshot" --json > "$output"
}

command -v peekaboo >/dev/null || die "peekaboo is not installed"
command -v jq >/dev/null || die "jq is not installed"

pb permissions status --json > "$OUT/permissions.json"
jq -e '.success and ([.data.permissions[] | select(.isRequired) | .isGranted] | all)' \
    "$OUT/permissions.json" >/dev/null \
    || die "Accessibility or Screen Recording permission is missing"

pb list windows --app "$APP" --json > "$OUT/windows-initial.json"
MAIN_WINDOW_ID="$(jq -r --arg app "$APP" \
    '.data.windows | (map(select(.title == $app and .isMainWindow == true))[0] // map(select(.title == $app))[0]) | .window_id // empty' \
    "$OUT/windows-initial.json" | head -1)"
[[ -n "$MAIN_WINDOW_ID" ]] || die "main window for $APP not found"
observe_app "$OUT/main.json" || die "could not inspect main window or modal sheet"

# Fresh-install onboarding is a real release surface, but this smoke is aimed
# at the steady-state shell. Verify its selector and skip via AX so no model is
# downloaded or started.
if jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Skip")' "$OUT/main.json" >/dev/null; then
    press_identifier "$OUT/main.json" "Quickstart.Skip" "$OUT/onboarding-skip.json" \
        || die "could not AXPress Quickstart.Skip"
    sleep 1
    observe_app "$OUT/main.json" || die "could not inspect app after onboarding"
fi

if jq -e '.data.ui_elements[]? | select(.identifier == "DockHidePrompt.NoButton")' "$OUT/main.json" >/dev/null; then
    press_identifier "$OUT/main.json" "DockHidePrompt.NoButton" "$OUT/dock-prompt.json" \
        || die "could not dismiss Dock visibility prompt"
    sleep 1
    observe_app "$OUT/main.json" || die "could not inspect app after Dock prompt"
fi

for identifier in Sidebar.NewChat Sidebar.Launch rapid.chat.compose ChatView.SendOrStopButton ModelPickerBar.ModelMenu; do
    jq -e --arg id "$identifier" \
        '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/main.json" >/dev/null \
        || die "main window missing AX identifier: $identifier"
done
pb image --window-id "$MAIN_WINDOW_ID" --path "$OUT/main.png" --json \
    > "$OUT/main-image.json"

pb menu click --app "$APP" --item 'Settings…' --json > "$OUT/open-settings.json"
for _ in {1..20}; do
    pb list windows --app "$APP" --json > "$OUT/windows-settings.json"
    SETTINGS_WINDOW_ID="$(jq -r '.data.windows[] | select(.title == "Settings") | .window_id' \
        "$OUT/windows-settings.json" | head -1)"
    [[ -n "$SETTINGS_WINDOW_ID" ]] && break
    sleep 0.25
done
[[ -n "${SETTINGS_WINDOW_ID:-}" ]] || die "Settings window did not open"

pb see --window-id "$SETTINGS_WINDOW_ID" --json > "$OUT/settings.json"
for category in modelManagement tools appearance privacy app; do
    jq -e --arg id "Settings.Category.$category" \
        '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/settings.json" >/dev/null \
        || die "Settings missing category identifier: $category"
done

# The shared trailing-toggle style must preserve the native checkbox role and
# the identifiers attached by each settings panel. A visually-correct switch
# can otherwise degrade into inert AX text, which breaks both VoiceOver and
# semantic automation while remaining invisible in screenshots.
press_identifier "$OUT/settings.json" "Settings.Category.modelManagement" "$OUT/open-models.json" \
    || die "could not AXPress Settings.Category.modelManagement"
sleep 0.5
pb see --window-id "$SETTINGS_WINDOW_ID" --json > "$OUT/models.json"
for identifier in Settings.Models.ShowAllModelsToggle Settings.Models.AutoStartOnLaunchToggle; do
    jq -e --arg id "$identifier" \
        '.data.ui_elements[]? | select(.identifier == $id and .role == "checkbox")' \
        "$OUT/models.json" >/dev/null \
        || die "Models toggle is not exposed as a native AX checkbox: $identifier"
done

APPEARANCE_ID="$(jq -r '.data.ui_elements[] | select(.identifier == "Settings.Category.appearance") | .id' "$OUT/models.json")"
SNAPSHOT="$(jq -r '.data.snapshot_id' "$OUT/models.json")"
pb perform-action --on "$APPEARANCE_ID" --action AXPress --snapshot "$SNAPSHOT" --json \
    > "$OUT/open-appearance.json"
sleep 0.5
pb see --window-id "$SETTINGS_WINDOW_ID" --json > "$OUT/appearance.json"

for mode in system light dark; do
    case "$mode" in
        system) expected_description="Auto (follow system)" ;;
        light) expected_description="Light" ;;
        dark) expected_description="Dark" ;;
    esac
    jq -e --arg id "Settings.Appearance.Theme.$mode" --arg description "$expected_description" \
        '.data.ui_elements[]? | select(.identifier == $id and .description == $description)' \
        "$OUT/appearance.json" >/dev/null \
        || die "Appearance option is not semantically addressable: $mode"
done
pb image --window-id "$SETTINGS_WINDOW_ID" --path "$OUT/appearance.png" --json \
    > "$OUT/appearance-image.json"

log "PASS — semantic GUI smoke complete"
log "artifacts: $OUT"
