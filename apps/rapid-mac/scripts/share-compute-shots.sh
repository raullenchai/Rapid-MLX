#!/usr/bin/env bash
# share-compute-shots.sh — capture the Share Compute surfaces at the four
# approved review sizes (1440×900 and 720×900, light and dark).
#
# WHY THIS EXISTS
# ---------------
# The Signal Split direction is a visual spec, and `swift test` cannot review
# spacing, contrast, or clipping. This script drives the REAL app, so the
# screenshots show the shipped rendering rather than a preview approximation.
#
# It is a review tool, not a gate: it asserts nothing. The operator (or an
# agent) compares its output against the Paper artboards.
#
# WHY IT RELAUNCHES INSTEAD OF CLICKING
# -------------------------------------
# Driving the sidebar and the tab strip would need macOS Accessibility
# permission, which is granted per-binary and is not available to every
# harness (the golden-flow suite has it; a plain shell generally does not).
# Rather than depend on it, each surface gets its own launch with
# RAPID_GUI_INITIAL_SECTION / RAPID_GUI_SHARE_COMPUTE_TAB — both inert unless
# RAPID_GUI_GOLDEN_MODE=1, so nothing here can affect a normal launch.
#
# ISOLATION
# ---------
# Every run goes through `dogfood-isolate.sh`, so it gets a throwaway bundle
# identifier and a throwaway $HOME. It therefore cannot read or write the
# operator's real preferences, chat history, or contribution receipts — which
# matters here more than usual, because this script SEEDS receipts and would
# otherwise write fixture sessions into somebody's real history.
#
# The model catalog comes from the bundled fake sidecar with
# FAKE_SHARE_COMPUTE_POOL=1: two of the four pool models cached, the other two
# (Nemotron and GLM) left un-downloaded, so Share (ready models only), the
# picker, and Live Pool's download + storage rows all have something real to
# render.
#
# POOL SUMMARY
# ------------
# RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY (same golden-mode gate) injects a FIXED
# summary instead of calling pay.quicksilverpro.io. A capture must not read the
# production pool: it is currently all zeros, so the populated layout — the one
# where a long model name sits beside four metric columns — would never be
# reviewed, and live numbers would make two runs of identical code look like a
# regression. `populated` carries all four catalog models with GLM disabled;
# `empty` is the honest all-zero pool.
#
# CLEAN ARTIFACTS
# ---------------
# RAPID_GUI_SUPPRESS_REVIEW_CHROME=1 (only honoured alongside
# RAPID_GUI_GOLDEN_MODE=1) stops the app painting the bottom-trailing update
# card and GitHub star prompt. Without it the update checker reaches the real
# release feed during the capture and drops an "Update Available" panel over
# the lower-right corner of every surface under review. Nothing about update
# discovery itself is disabled — see ContentView.suppressesReviewChrome.
#
# LIFECYCLE SURFACES
# ------------------
# Preparing, Online, Session Complete, Connection Review and the open model
# picker cannot be reached without a provider key and a live pool.
# RAPID_GUI_SHARE_COMPUTE_STAGE (same golden-mode gate) renders them from
# ShareComputeReviewFixture — real view models, fixture inputs, no mocked API.
#
# Usage:
#   ./scripts/share-compute-shots.sh [output-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_SOURCE="${RAPID_SHOTS_SOURCE_APP:-$ROOT/build/Rapid-MLX Desktop.app}"
OUT_ROOT="${1:-/tmp/share-compute-shots-$(date -u +%Y%m%dT%H%M%SZ)}"

[[ -d "$APP_SOURCE" ]] || { echo "error: build the app first (scripts/build.sh)" >&2; exit 1; }
mkdir -p "$OUT_ROOT"

# Resolves the app's CGWindowID so captures are BY WINDOW rather than by screen
# rectangle — the app cannot be brought to the front without Accessibility
# permission, and a rectangle capture would photograph whatever is in front of
# it instead.
# window-id.swift is SOURCE; the executable it compiles to is generated output
# and must never land in scripts/ — a Mach-O binary sitting beside the shell
# scripts is invisible to `.gitignore` (which only covers build/ and .build/)
# and gets committed by the next `git add scripts/`. It is built into a
# throwaway directory that the EXIT trap removes, so no stale binary can
# survive a change to the .swift file either.
TOOLS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/share-compute-tools.XXXXXX")"
WINDOW_ID_TOOL="$TOOLS_DIR/window-id"

PERSONA=""
APP_PID=""

cleanup() {
    if [[ -n "$APP_PID" ]]; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    APP_PID=""
}
# One trap, every exit path (success, error, and the `set -e` abort): the
# persona and the compiled tool both go.
trap 'cleanup; [[ -n "$PERSONA" ]] && rm -rf "$PERSONA"; rm -rf "$TOOLS_DIR"' EXIT

swiftc -O -o "$WINDOW_ID_TOOL" "$ROOT/scripts/window-id.swift" \
    || { echo "error: could not build window-id" >&2; exit 1; }

# Main display size in POINTS, which is the unit both the saved window frame
# and `screencapture -R` work in. Read from system_profiler's "UI Looks like"
# line so the script needs no PyObjC (the system python3 no longer ships it).
screen_points() {
    system_profiler SPDisplaysDataType 2>/dev/null \
        | awk '/UI Looks like/ { print $4, $6; exit }'
}

# Fixture contribution history. Dated relative to now so both the "Today ·
# 2:14 PM" and the "Sep 16 · 8:40 PM" row shapes render, with mixed reward
# statuses so the status-tag column shows its three intrinsic widths.
seed_receipts() {
    local home="$1"
    local dir="$home/Library/Application Support/Rapid"
    mkdir -p "$dir"
    /usr/bin/python3 - "$dir/share-compute-receipts.json" <<'PY'
import json, sys
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
rows = [
    ("QS-8A31", "qwen3.8-27b", "Qwen3.8 27B · 4-bit", "qs-node-a8c1", 2, 6138, "available"),
    ("QS-7F24", "qwen3.6-35b", "Qwen3.6 35B", "qs-node-7f24", 30, 7560, "available"),
    ("QS-2D90", "nemotron-3.5-lightning", "Nemotron 3.5 Lightning 30B · 4-bit", "qs-node-2d90", 54, 4140, "processing"),
    ("QS-B314", "qwen3.8-27b", "Qwen3.8 27B · 4-bit", "qs-node-b314", 78, 2820, "available"),
    ("QS-45E8", "qwen3.6-35b", "Qwen3.6 35B", "qs-node-45e8", 102, 12060, "available"),
    ("QS-9C11", "qwen3.8-27b", "Qwen3.8 27B · 4-bit", None, 126, 300, "available"),
]
receipts = []
for rid, catalog, title, node, hours_ago, duration, reward in rows:
    started = now - timedelta(hours=hours_ago)
    receipts.append({
        "id": rid,
        "catalogID": catalog,
        "modelTitle": title,
        "worker": "Lori-Mac",
        "nodeID": node,
        "startedAt": started.isoformat().replace("+00:00", "Z"),
        "endedAt": (started + timedelta(seconds=duration)).isoformat().replace("+00:00", "Z"),
        "rewardStatus": reward,
        "restoreStatus": "complete",
    })
with open(sys.argv[1], "w") as stream:
    json.dump({"schemaVersion": 1, "receipts": receipts}, stream, indent=2)
PY
}

# One launch → one screenshot. `$WINDOW_RECT` is known up front because the
# window frame is pinned by preference, so the capture needs no CGWindowID
# lookup (which would need PyObjC).
# `$stage` is optional: empty for the three ordinary tabs, or one of the
# ShareComputeReviewStage raw values for a lifecycle surface.
capture_surface() {
    local name="$1" width="$2" height="$3" appearance="$4" tab="$5" out="$6" stage="${7:-}"
    cleanup
    [[ -n "$PERSONA" ]] && rm -rf "$PERSONA"
    PERSONA="$(mktemp -d "/tmp/share-compute-shots.XXXXXX")"

    local isolated_app bundle_id home screen_w screen_h window_y
    isolated_app="$("$ROOT/scripts/dogfood-isolate.sh" "$APP_SOURCE" "$PERSONA" 2>/dev/null | tail -1)"
    bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$isolated_app/Contents/Info.plist")"
    home="$PERSONA/home"
    mkdir -p "$home"
    seed_receipts "$home"

    read -r screen_w screen_h <<<"$(screen_points)"
    : "${screen_w:=1920}" "${screen_h:=1080}"
    # Leave the menu bar clear, then convert the Cocoa (bottom-left) origin to
    # the top-left rect `screencapture -R` expects.
    window_y=$((screen_h - height - 60))

    pref() { env HOME="$home" CFFIXED_USER_HOME="$home" /usr/bin/defaults write "$bundle_id" "$@" >/dev/null; }
    pref com.rapidmlx.rapid.telemetry.enabled -bool false
    pref Rapid.experimental.shareComputeEnabled -bool true
    pref quickstart.v1.done -bool true
    pref onboarding.v1.seen -bool true
    # The saved window frame is the documented lever for pinning a review size
    # (see preview.sh). The trailing four numbers are the SCREEN frame the
    # window was saved against; AppKit re-fits the window when they do not
    # match the current display, so they come from the real screen.
    pref "NSWindow Frame Rapid.MainWindow" \
        "0 ${window_y} ${width} ${height} 0 0 ${screen_w} ${screen_h}"
    # The app's OWN appearance override (Settings → Appearance), not the
    # system one. `AppleInterfaceStyle` as a per-app key does not reliably
    # force Dark on an app whose system is Light — it was tried first and the
    # "dark" captures came out light — and flipping the system appearance
    # would be a visible change to the operator's machine for a screenshot.
    pref "rapid.appearance.v1" -string "$appearance"

    # CI=true skips dogfood-isolate's host precheck, which otherwise refuses to
    # launch on a machine it considers busy.
    env CI=true \
        RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
        FAKE_SHARE_COMPUTE_POOL=1 \
        DOGFOOD_WORKING_SET_GB=0.1 \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_GUI_SUPPRESS_REVIEW_CHROME=1 \
        RAPID_GUI_INITIAL_SECTION=shareCompute \
        RAPID_GUI_SHARE_COMPUTE_TAB="$tab" \
        RAPID_GUI_SHARE_COMPUTE_STAGE="$stage" \
        RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY="${POOL_SUMMARY:-populated}" \
        RAPID_GUI_SHARE_COMPUTE_LEDGER="${LEDGER:-loaded}" \
        /usr/bin/python3 -c 'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
        "$PERSONA/launch.sh" > "$OUT_ROOT/$name-$tab-app.log" 2>&1 &
    APP_PID=$!

    # The catalog shell-out and first paint take a few seconds; there is no
    # permission-free readiness signal, so this waits generously.
    sleep 9
    kill -0 "$APP_PID" 2>/dev/null || { echo "  ! app exited early ($tab)" >&2; return 1; }
    local window_id
    window_id="$("$WINDOW_ID_TOOL" "$(pgrep -P "$APP_PID" -f "MacOS/Rapid" | head -1)" 2>/dev/null \
        || "$WINDOW_ID_TOOL" "$APP_PID" 2>/dev/null || true)"
    if [[ -z "$window_id" ]]; then
        echo "  ! no window found for $tab${stage:+ / $stage}" >&2
        cleanup
        return 1
    fi
    # -o drops the window shadow so the PNG is exactly the window.
    screencapture -x -o -l"$window_id" "$out"
    cleanup
}

run_variant() {
    local name="$1" width="$2" height="$3" appearance="$4"
    echo "==> $name (${width}×${height}, $appearance)"
    capture_surface "$name" "$width" "$height" "$appearance" share \
        "$OUT_ROOT/$name-share.png" || true
    capture_surface "$name" "$width" "$height" "$appearance" credits \
        "$OUT_ROOT/$name-credits.png" || true
    capture_surface "$name" "$width" "$height" "$appearance" livePool \
        "$OUT_ROOT/$name-live-pool.png" || true
}

# The lifecycle surfaces and the open picker. Captured light + narrow-light
# for the picker (its clipping risk is a narrow-width question) and light at
# desktop for the rest, which is where the Paper artboards for them live; the
# dark pass is covered by the theme matrix above on the tabs that share the
# same components.
run_stage() {
    local name="$1" width="$2" height="$3" appearance="$4" stage="$5" out="$6"
    echo "==> $name · $stage (${width}x${height}, $appearance)"
    capture_surface "$name" "$width" "$height" "$appearance" share \
        "$OUT_ROOT/$out.png" "$stage" || true
}

run_variant "1440x900-light" 1440 900 light
run_variant "1440x900-dark"  1440 900 dark
run_variant "720x900-light"   720 900 light
run_variant "720x900-dark"    720 900 dark

# The calm all-zero pool, which is what production currently reports. It must
# read as a normal state — real zeros, a quiet empty line, no error.
POOL_SUMMARY=empty capture_surface "1440x900-light" 1440 900 light livePool \
    "$OUT_ROOT/1440x900-light-live-pool-empty.png" || true
POOL_SUMMARY=unavailable capture_surface "1440x900-light" 1440 900 light livePool \
    "$OUT_ROOT/1440x900-light-live-pool-unavailable.png" || true

# Contributor-ledger states. `loaded` is already covered by run_variant above
# (it is the default), so these are the other four, at both sizes and themes.
for ledger in noReadKey empty revoked stale; do
    slug="$(echo "$ledger" | tr '[:upper:]' '[:lower:]')"
    for v in "1440x900-light 1440 900 light" "1440x900-dark 1440 900 dark" \
             "720x900-light 720 900 light" "720x900-dark 720 900 dark"; do
        set -- $v
        LEDGER="$ledger" capture_surface "$1" "$2" "$3" "$4" credits \
            "$OUT_ROOT/$1-credits-$slug.png" || true
    done
done

run_stage "1440x900-light" 1440 900 light modelPicker      "1440x900-light-model-picker-open"
run_stage "720x900-light"   720 900 light modelPicker      "720x900-light-model-picker-open"
run_stage "1440x900-light" 1440 900 light connectionReview "1440x900-light-connection-review"
run_stage "720x900-dark"    720 900 dark  connectionReview "720x900-dark-connection-review"
run_stage "1440x900-light" 1440 900 light preparing        "1440x900-light-preparing"
run_stage "1440x900-dark"  1440 900 dark  preparing        "1440x900-dark-preparing"
run_stage "1440x900-light" 1440 900 light online           "1440x900-light-online"
run_stage "1440x900-dark"  1440 900 dark  online           "1440x900-dark-online"
run_stage "1440x900-light" 1440 900 light sessionComplete  "1440x900-light-session-complete"
run_stage "1440x900-dark"  1440 900 dark  sessionComplete  "1440x900-dark-session-complete"
run_stage "720x900-light"   720 900 light online           "720x900-light-online"
run_stage "720x900-light"   720 900 light sessionComplete  "720x900-light-session-complete"

echo
echo "Screenshots: $OUT_ROOT"
ls -1 "$OUT_ROOT"/*.png 2>/dev/null || echo "(none captured)"
