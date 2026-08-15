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
# The fake's image-generation fixture (see fake-rapid-mlx.sh). Scrubbed from
# baselines for the same reason as FAKE_ALIAS: renaming a fixture must not read
# as a structural change to the UI.
FAKE_IMAGE_ALIAS="fake-image-alias"
UPDATE_BASELINES=0
FLOW="all"
KEEP=0
APP_PID=""
OPERATOR_SERVER_PID=""
PERSONA=""
OUT=""
MAIN_WINDOW_ID=""
BUNDLE_ID=""
AX_DRIVER=""
RESULT_WRITTEN=0
PERSONA_ENV=()

usage() {
    cat <<'EOF'
Usage: gui-golden-flows.sh [--flow NAME] [--keep] [--update-baselines]

Flows: fresh-install, cached-quickstart, download-progress, settings-persistence, chat-restore, restored-tools, tool-loop-budget, chat-depth, math-rendering, launch-integrations,
       slow-stream-stop,
       model-crash-recovery, low-memory-choice,
       update-state, window-close-prompt, no-dead-controls, catalog-integrity,
       browse-all-destination, chat-document-attachment, image-generation, audio-readiness, all

Most named regression flows drive the app through the accessibility API alone.
The preflight contract tests keep the exact allowlist in sync with
flow_requires_peekaboo below. Those flows need neither Peekaboo nor Screen
Recording, which lets them run unattended in CI (see the gui-golden-flows job
in .github/workflows/rapid-mac-ci.yml).

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
flow_requires_screen_recording() {
    case "$FLOW" in
        all|fresh-install|low-memory-choice) return 0 ;;
        *) return 1 ;;
    esac
}
# Which flows shell out to `peekaboo`, and therefore need it installed and
# permitted. Named flows drive the app through `rapid-ax` alone; everything
# else — including `all` — is assumed to need peekaboo.
#
# Default-deny is the load-bearing part. A NEW flow is treated as needing
# peekaboo until someone says otherwise, so it cannot quietly join the
# unattended subset and then fail somewhere unrelated on a machine that has no
# peekaboo. Getting this backwards would be silent; getting it wrong this way
# round is a one-line fix.
#
# Why it matters at all: `rapid-ax` needs only the Accessibility grant, which a
# GitHub-hosted macOS runner already carries in its image TCC database, and it
# is built from a source file in this repo. Peekaboo is a third-party install
# that additionally reaches its bridge socket (`--bridge-socket`, provided by
# the Peekaboo app rather than the `brew` CLI) and has its own permission
# surface. The peekaboo-free flows are therefore the set that can run
# unattended without taking on any of that.
flow_requires_peekaboo() {
    case "$FLOW" in
        cached-quickstart|download-progress|settings-persistence|chat-restore|restored-tools|tool-loop-budget|chat-depth|math-rendering|browse-all-destination|no-dead-controls|catalog-integrity|update-state|launch-integrations) return 1 ;;
        slow-stream-stop|model-crash-recovery|chat-document-attachment|image-generation|audio-readiness|window-close-prompt|resident-load-rejected) return 1 ;;
        *) return 0 ;;
    esac
}
pb_click_coords() {
    local coords="$1"
    shift
    # Peekaboo 3.0 uses screen coordinates by default and auto-focuses the
    # target window. Newer releases make those semantics explicit.
    if peekaboo click --help 2>&1 | grep -q -- --global-coords; then
        pb click --coords "$coords" --global-coords --foreground "$@"
    else
        pb click --coords "$coords" "$@"
    fi
}

cleanup_persona() {
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    APP_PID=""
    cleanup_fake_sidecars
    if [[ "$KEEP" == 0 && -n "$BUNDLE_ID" ]]; then
        defaults delete "$BUNDLE_ID" >/dev/null 2>&1 || true
    fi
    if [[ "$KEEP" == 0 && -n "$PERSONA" && -d "$PERSONA" ]]; then
        rm -rf "$PERSONA"
    fi
    PERSONA=""
    BUNDLE_ID=""
    PERSONA_ENV=()
}

cleanup_fake_sidecars() {
    if [[ -n "$OUT" && -f "$OUT/fake-events.jsonl" ]]; then
        # Pair each pid with the alias it was started for, and require the
        # live command to still name THAT alias. A bare `serve fake-alias`
        # match missed `serve fake-image-alias` entirely and left the image
        # flow's sidecar listening after the run — an orphan that then
        # contaminates the next local run. Matching the recorded alias keeps
        # the recycled-pid guard the substring test was there for, without
        # having to remember to extend it for every new fixture alias.
        while IFS=$'\t' read -r fake_pid fake_alias; do
            [[ "$fake_pid" =~ ^[0-9]+$ ]] || continue
            [[ -n "$fake_alias" && "$fake_alias" != "null" ]] || continue
            local command
            command="$(ps -p "$fake_pid" -o command= 2>/dev/null || true)"
            if [[ "$command" == *"serve $fake_alias"* ]]; then
                kill "$fake_pid" 2>/dev/null || true
            fi
        done < <(jq -r 'select(.event == "server_started")
                        | "\(.pid)\t\(.alias // "")"' \
                     "$OUT/fake-events.jsonl" 2>/dev/null | sort -u)
    fi
}

cleanup_operator_server() {
    if [[ -n "$OPERATOR_SERVER_PID" ]] && kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null; then
        kill "$OPERATOR_SERVER_PID" 2>/dev/null || true
        wait "$OPERATOR_SERVER_PID" 2>/dev/null || true
    fi
    OPERATOR_SERVER_PID=""
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
    cleanup_operator_server
}
trap finish EXIT
trap 'cleanup_persona; cleanup_operator_server; exit 130' INT
trap 'cleanup_persona; cleanup_operator_server; exit 143' TERM

# The preconditions every flow depends on and none of them can observe:
# permission to read another process's AX tree, and a session that can actually
# put a window on screen.
#
# Checked BEFORE the first app launch. Both failures otherwise look identical
# and identically wrong: the flow spends 20 s inside `wait_for_window` and dies
# on "main window did not appear", accusing the app of never opening a window
# when the truth is either that we were not allowed to look or that nothing can
# be shown at all. Both were observed for real while building this — a missing
# grant, and a Mac that locked its screen mid-run.
#
# Aimed at the Dock when one is running, because the grant has to work against
# ANOTHER process and `AXIsProcessTrusted()` alone is only the system's opinion
# about us until a real cross-process read backs it up. rapid-ax adds the lock
# check, which that read cannot supply: the Dock reads perfectly behind a lock
# screen.
require_ax_trust() {
    local dock_pid
    dock_pid="$(pgrep -x Dock | head -1 || true)"
    # rapid-ax prints the specific reason to stderr; do not restate it here and
    # risk naming the wrong one of the two.
    "$AX_DRIVER" trust ${dock_pid:+"$dock_pid"} > "$OUT_ROOT/ax-trust.json" \
        || die "GUI preconditions not met — see the rapid-ax line above and $OUT_ROOT/ax-trust.json"
}

require_tools() {
    [[ -d "$APP_SOURCE" ]] || die "built app not found: $APP_SOURCE"
    for tool in jq python3; do
        command -v "$tool" >/dev/null || die "$tool is required"
    done
    [[ -f "$BASELINE_TOOL" ]] || die "AX baseline normalizer not found: $BASELINE_TOOL"
    AX_DRIVER="$OUT_ROOT/rapid-ax"
    swiftc "$ROOT/scripts/rapid-ax.swift" -o "$AX_DRIVER"
    require_ax_trust
    flow_requires_peekaboo || return 0
    command -v peekaboo >/dev/null || die "peekaboo is required for flow: $FLOW"
    pb permissions status --json > "$OUT_ROOT/permissions.json"
    jq -e '.success and any(.data.permissions[]?; .name == "Accessibility" and .isGranted == true)' \
        "$OUT_ROOT/permissions.json" >/dev/null || die "Peekaboo needs Accessibility permission"
    if flow_requires_screen_recording; then
        jq -e '.success and ([.data.permissions[] | select(.isRequired) | .isGranted] | all)' \
            "$OUT_ROOT/permissions.json" >/dev/null \
            || die "this screenshot flow needs Screen Recording permission"
    fi
}

start_persona() {
    local name="$1"
    shift
    cleanup_persona
    OUT="$OUT_ROOT/$name"
    PERSONA_ENV=("$@")
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
    # macOS ships Bash 3.2, where expanding a declared-but-empty array under
    # `set -u` raises "unbound variable". The `+` form expands to nothing
    # when the array has no elements and preserves argv boundaries otherwise.
    for assignment in "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}"; do
        key="${assignment%%=*}"
        value="${assignment#*=}"
        updated="$config.next"
        jq --arg key "$key" --arg value "$value" '.[$key] = $value' "$config" > "$updated"
        mv "$updated" "$config"
    done
    env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
        FAKE_EVENT_LOG="$OUT/fake-events.jsonl" \
        "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}" \
        "$PERSONA/launch.sh" > "$OUT/app.log" 2>&1 &
    APP_PID=$!
    wait_for_window
}

relaunch_persona() {
    stop_app
    # A relaunch keeps the persona but starts a fresh app process. Reap only
    # the fake sidecars this harness recorded before starting it again. Before
    # #1618 the app's unsafe global port sweep hid this ownership leak by
    # killing the old fake (and potentially an operator's real server too).
    cleanup_fake_sidecars
    env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
        FAKE_EVENT_LOG="$OUT/fake-events.jsonl" \
        "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}" \
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

refresh_main_window_id() {
    MAIN_WINDOW_ID=""
    pb list windows --app "PID:$APP_PID" --json > "$OUT/windows-current.json" 2>/dev/null \
        || return 1
    MAIN_WINDOW_ID="$(jq -r '(.data.windows // []) | map(select(.title == "Rapid-MLX"))[0].window_id // empty' "$OUT/windows-current.json" 2>/dev/null)"
    [[ -n "$MAIN_WINDOW_ID" ]]
}

wait_for_window() {
    local windows="$OUT/windows.json"
    for _ in {1..80}; do
        kill -0 "$APP_PID" 2>/dev/null || die "app exited before opening a window"
        "$AX_DRIVER" dump "$APP_PID" > "$windows" 2>/dev/null || true
        if jq -e '.success == true and .data.windows.complete == true
                  and any(.data.windows.titles[]?; . == "Rapid-MLX")' \
            "$windows" >/dev/null 2>&1; then
            if flow_requires_screen_recording; then
                if ! refresh_main_window_id; then
                    sleep 0.25
                    continue
                fi
            fi
            return
        fi
        sleep 0.25
    done
    die "main window did not appear"
}

see_main() {
    local destination="$1"
    # Only screenshot flows need a CGWindow id. Never retain a stale id if
    # enumeration fails; visual evidence must target the current main window.
    if flow_requires_screen_recording && ! refresh_main_window_id; then
        die "could not refresh the main screenshot window ID"
    fi
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

wait_tree_text() {
    local needle="$1" destination="$2" attempts="${3:-80}"
    for ((i=0; i<attempts; i++)); do
        see_main "$destination"
        if jq -e --arg needle "$needle" \
            '(.data.ui_elements | tostring) | contains($needle)' \
            "$destination" >/dev/null; then
            return
        fi
        sleep 0.25
    done
    die "timed out waiting for AX text: $needle"
}

# Is a window with this title in the app's OWN accessibility tree?
#
# ``peekaboo list windows`` is NOT an oracle for this. It reports a window that
# has already been destroyed — measured: immediately after Settings closes,
# ``pb list windows`` still lists it while the AX tree does not, and it never
# catches up. A flow that polls the window list for a title to DISAPPEAR
# therefore waits forever and then reports a product bug that is not there;
# one that polls for a title to APPEAR can be satisfied by a window a previous
# persona in the same run opened. The app's own AX tree is authoritative for
# both directions.
#
# Three outcomes, not two: 0 = present, 1 = absent, 2 = could not observe.
# Folding the third into "absent" recreates the very bug above in a new place —
# one failed dump, one unparseable file, and a caller waiting for a window to
# close concludes that it closed. Callers must branch on all three.
ax_window_present() {
    local title="$1" destination="$2" status
    "$AX_DRIVER" dump "$APP_PID" > "$destination" 2>/dev/null || return 2
    # `data.windows`, NOT `ui_elements`: the driver enumerates the root's
    # children once and vouches for that list with `complete`. The element
    # array cannot answer this, because every way it comes up short — a role
    # read that failed, a title that would not read, the record cap — removes a
    # window from it silently and is indistinguishable from the window closing.
    # `titles` must be an array as well as complete: `titles[]?` below swallows
    # a structural failure, so without this a malformed list would read as a
    # confident "absent" — the third outcome collapsing back into the first.
    jq -e '.success == true and .data.windows.complete == true
           and (.data.windows.titles | type) == "array"' \
        "$destination" >/dev/null 2>&1 || return 2
    status=0
    jq -e --arg t "$title" '[.data.windows.titles[]? | select(. == $t)] | length > 0' \
        "$destination" >/dev/null 2>&1 || status=$?
    # jq exits 1 only for a well-formed query whose answer was false; anything
    # else (2 usage, 3 compile, 4 no output) is a broken observation.
    case "$status" in
        0) return 0 ;;
        1) return 1 ;;
        *) return 2 ;;
    esac
}

# Everything the transcript is showing, and nothing else.
#
# `ui_elements` is the whole app. The sidebar row for this conversation
# carries the first prompt's text in its accessibility description, and the
# composer carries whatever is typed. An assertion that searches the flat list
# can therefore be satisfied by a surface that is NOT the transcript — which
# is how "all five turns are present, in order" would pass on a transcript
# that lost four of them, as long as the sidebar still knew their names.
#
# Measured on a real five-turn dump: the sidebar row exposes `shape:prose` in
# `description` while the transcript bubble exposes it in `value`, so today
# the app-wide search happens to land on the right element. Nothing pins that.
# Give the field one `title` and the ordering assertion starts reading the
# sidebar instead, silently.
#
# Scope by the app's stable message-action identifiers. The old MarkdownUI
# implementation happened to insert an AXOpaqueProviderList, but TextKit's
# custom views correctly expose native AXStaticText nodes without that private
# provider wrapper. Start at the first user message text (immediately before
# its Copy/Edit controls) and end at the last assistant Retry control.
transcript_only() {
    python3 - "$1" "$2" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
els = json.load(open(src))["data"]["ui_elements"]
action_indexes = [
    i for i, e in enumerate(els)
    if str(e.get("identifier", "")).startswith("ChatView.Message.")
]
if not action_indexes:
    sys.exit("no transcript message actions in this dump")
first_action, last_action = min(action_indexes), max(action_indexes)
# Include the nearest preceding static text: that is the first prompt. Keep
# the search local so sidebar text can never satisfy transcript assertions.
start = next(
    (i for i in range(first_action - 1, max(-1, first_action - 8), -1)
     if els[i].get("role") == "AXStaticText"),
    first_action,
)
scoped = els[start:last_action + 1]
if not scoped:
    sys.exit("the transcript container has no children — nothing to assert on")
json.dump({"data": {"ui_elements": scoped}}, open(dst, "w"))
PYEOF
}

# Extract the first complete AX subtree whose root has ROLE. The rapid-ax dump
# is flat pre-order plus `depth`; taking the root and every following element
# until depth returns to the root level preserves the hierarchy while excluding
# covered background windows. Modal-sheet baselines must use this — AppKit keeps
# the underlying split view in the application tree even though a user cannot
# interact with it.
role_subtree_only() {
    python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys
src, role, dst = sys.argv[1:]
els = json.load(open(src))["data"]["ui_elements"]
start = next((i for i, e in enumerate(els) if e.get("role") == role), None)
if start is None:
    sys.exit(f"AX tree has no {role} subtree")
root_depth = int(els[start].get("depth", 0))
end = len(els)
for i in range(start + 1, len(els)):
    if int(els[i].get("depth", 0)) <= root_depth:
        end = i
        break
scoped = []
for element in els[start:end]:
    element = dict(element)
    element["depth"] = int(element.get("depth", root_depth)) - root_depth
    scoped.append(element)
json.dump({"data": {"ui_elements": scoped}}, open(dst, "w"))
PYEOF
}

# Turn N's prompt is in the Nth USER message and turn N's answer is in the
# Nth ASSISTANT message.
#
# Reading order alone does not say that. Every needle can sit in the right
# sequence while the answer text lives inside the user's own bubble and the
# assistant's bubble holds something else entirely — the counts, the ordering
# and the structural baseline all survive that, because nothing ties a string
# to the message it belongs to.
#
# The app's own controls are the boundary: a user message ends at its Edit
# button, an assistant message ends at its Retry button. Measured on a real
# dump, in tree order:
#
#   StaticText(prompt)  Copy  Edit        <- user message
#   Disclosure  StaticText(answer)  …  Copy  Retry   <- assistant message
assert_turns_pair_up() {
    local transcript="$1"
    shift
    python3 - "$transcript" "$@" <<'PYEOF'
import json, sys
transcript, pairs = sys.argv[1], sys.argv[2:]
els = json.load(open(transcript))["data"]["ui_elements"]

messages, buffer = [], []
for element in els:
    identifier = str(element.get("identifier") or "")
    buffer.append(str(element.get("value", "")))
    if ".Edit." in identifier:
        messages.append(("user", " ".join(buffer)))
        buffer = []
    elif ".Retry." in identifier:
        messages.append(("model", " ".join(buffer)))
        buffer = []

expected = [
    (side, text)
    for i, text in enumerate(pairs)
    for side in ("user" if i % 2 == 0 else "model",)
]
if len(messages) != len(expected):
    got = ", ".join(side for side, _ in messages)
    sys.exit(
        f"expected {len(expected)} messages alternating user/model, "
        f"found {len(messages)}: {got}"
    )
for index, ((want_side, needle), (got_side, text)) in enumerate(
    zip(expected, messages), start=1
):
    if want_side != got_side:
        sys.exit(
            f"message {index} is a {got_side} message, expected {want_side} — "
            "the transcript is not alternating"
        )
    if needle not in text:
        sys.exit(
            f"{want_side} message {index} does not contain {needle!r}; "
            f"it holds {text.strip()[:80]!r}"
        )
PYEOF
}

# Each of these strings is a whole element, not a fragment of a blob.
#
# This is the positive half of "markdown was rendered". Asserting only that
# ``` fences and | pipe rows are ABSENT cannot tell a rendered table from a
# renderer that stripped the pipes and printed one flat line, nor a rendered
# list from one that dropped the bullets. Both leave the text on screen and
# both pass an absence check.
#
# Measured: a rendered table puts every cell in its own AXStaticText
# (`qwen3.5-9b`, `5.2 GB`, `74 tok/s`), and a rendered list puts every item in
# its own node with the source marker stripped. A renderer that flattens
# either one merges them into a single node, so requiring an EXACT value match
# is what separates "rendered" from "printed".
assert_rendered_as_separate_nodes() {
    local tree="$1" label="$2"
    shift 2
    python3 - "$tree" "$label" "$@" <<'PYEOF'
import json, sys
tree, label, expected = sys.argv[1], sys.argv[2], sys.argv[3:]
els = json.load(open(tree))["data"]["ui_elements"]
values = [str(e.get("value", "")).strip() for e in els]
missing = [want for want in expected if want not in values]
if missing:
    # Distinguish "not on screen at all" from "on screen inside a bigger
    # node" — the second is the flattening regression this exists to catch.
    detail = []
    for want in missing:
        holder = next((v for v in values if want in v), None)
        detail.append(
            f"{want!r} is part of {holder[:60]!r}" if holder
            else f"{want!r} is not in the transcript at all"
        )
    sys.exit(f"{label}: not rendered as separate elements — " + "; ".join(detail))
PYEOF
}

# No list item still wearing its source marker.
#
# Measured: the renderer strips `-`/`*`/`1.` and emits the bare item text. A
# fallback to plain text puts them back, and every "does the text appear"
# assertion in this file passes on that, because the text does appear.
assert_no_literal_list_markers() {
    local tree="$1"
    python3 - "$tree" <<'PYEOF'
import json, re, sys
els = json.load(open(sys.argv[1]))["data"]["ui_elements"]
# A marker glued to its item ("- a nested point") and a marker standing alone
# in its own node ("-" next to "a nested point") are the same regression on
# screen, and the second slips past a line-prefix check while also satisfying
# an exact-match check on the item text.
# A tab after the marker is TextKit's accessible representation of a real
# NSTextList item, not raw markdown. Raw source uses ordinary spaces.
LEADING = re.compile(r"^\s*(?:[-*+] +|\d+\. +)")
BARE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s*$")
offenders = []
for e in els:
    value = str(e.get("value", ""))
    if BARE.match(value):
        offenders.append(value)
        continue
    offenders.extend(line for line in value.splitlines() if LEADING.match(line))
if offenders:
    sys.exit(
        "a list marker reached the screen verbatim — the list was printed, "
        f"not rendered: {offenders[:3]}"
    )
PYEOF
}

# The Nth assistant message, as a dump of its own.
#
# Pairing a prompt with an answer is not enough on its own: every other shape
# assertion searched the WHOLE transcript, so a restore that moved the table
# cells into the CJK bubble, or the code block under the wrong question, still
# satisfied all of them. Each shape now has to be found in the message that
# shape was sent to.
assistant_message_only() {
    python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys
src, wanted, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
els = json.load(open(src))["data"]["ui_elements"]
messages, buffer = [], []
for element in els:
    identifier = str(element.get("identifier") or "")
    buffer.append(element)
    if ".Edit." in identifier:
        messages.append(("user", buffer))
        buffer = []
    elif ".Retry." in identifier:
        messages.append(("model", buffer))
        buffer = []
models = [group for side, group in messages if side == "model"]
if len(models) < wanted:
    sys.exit(
        f"transcript holds {len(models)} assistant message(s); wanted #{wanted}"
    )
json.dump({"data": {"ui_elements": models[wanted - 1]}}, open(dst, "w"))
PYEOF
}

# Everything this suite can say about what the renderer did with the five
# shapes, in one place so the restored transcript is held to the SAME bar as
# the live one. Checking the shapes only before the relaunch leaves a restore
# that flattens the table or drops the emoji indistinguishable from a good
# one, because the counts and the structural baseline both survive it (the
# baseline normalizes every value to `text`).
#
# Takes a TRANSCRIPT-scoped dump. Handing it the whole app would let another
# subtree — a preview, a tooltip, an off-screen copy — answer for the
# transcript.
#
assert_rendered_shapes() {
    local transcript="$1" scratch="$2"
    # These two hold anywhere in the transcript: no source syntax survives,
    # in any message.
    assert_markdown_rendered "$transcript"
    assert_no_literal_list_markers "$transcript"

    # Everything else is checked INSIDE the assistant message that shape was
    # sent to. The endings matter as much as the openings: a distinctive
    # phrase near the start of a long answer passes on a stream that stopped
    # early, and the fake is deterministic, so the last words are knowable.
    local m1="$scratch-m1.json" m2="$scratch-m2.json" m3="$scratch-m3.json"
    local m4="$scratch-m4.json" m5="$scratch-m5.json"

    assistant_message_only "$transcript" 1 "$m1"
    assert_tree_text "$m1" "Only the first was ever read by anyone else."

    assistant_message_only "$transcript" 2 "$m2"
    assert_code_block_is_its_own_view "$m2" \
        "Here is the function you asked for" "def fib(n)"
    assert_tree_text "$m2" "    return a"
    assert_tree_text "$m2" "background-color"
    assert_tree_text "$m2" "@font-face"
    assert_tree_text "$m2" ".PHONY"
    assert_tree_text "$m2" "filter-out"

    assistant_message_only "$transcript" 3 "$m3"
    assert_rendered_as_separate_nodes "$m3" "table cells" \
        "qwen3.5-9b" "5.2 GB" "74 tok/s" "llama-3.1-8b" "4.5 GB" "68 tok/s"
    # AppKit exposes a native SwiftUI Table as AXOutline on macOS, with real
    # row/cell/column children and titled column headers. Pin the whole shape;
    # six loose AXStaticTexts cannot satisfy this contract (#1689).
    jq -e '[.data.ui_elements[]?] as $e
            | any($e[]; .role == "AXOutline" and .description == "Markdown table")
              and ([ $e[] | select(.role == "AXRow") ] | length >= 2)
              and ([ $e[] | select(.role == "AXCell") ] | length >= 6)
              and ([ $e[] | select(.role == "AXColumn") ] | length >= 3)
              and ([ $e[] | select(.title == "model") ] | length > 0)
              and ([ $e[] | select(.title == "size") ] | length > 0)
              and ([ $e[] | select(.title == "speed") ] | length > 0)' \
        "$m3" >/dev/null \
        || die "markdown comparison has no navigable table semantics in the AX tree (#1689)"
    assert_tree_text "$m3" "Both fit comfortably in 16 GB."

    assistant_message_only "$transcript" 4 "$m4"
    # TextKit exposes one native AXStaticText for a paragraph/list group. Its
    # NSTextList markers are tabs (`1.\t`), while raw markdown markers are
    # ordinary spaces and are rejected above. Pin every item and the native
    # marker shape without requiring MarkdownUI's former one-node-per-item
    # implementation detail.
    for item in "First, read the prompt." "Second, plan the answer." \
                "a nested point" "another one" "Third, write it down."; do
        assert_tree_text "$m4" "$item"
    done
    jq -e '[.data.ui_elements[]? | (.value // "") | tostring]
            | any(.[]; contains("1.\tFirst, read the prompt."))' "$m4" >/dev/null \
        || die "ordered list lost TextKit native list semantics"

    assistant_message_only "$transcript" 5 "$m5"
    assert_tree_text "$m5" "🎯🚀"
    assert_tree_text "$m5" "مرحبا"
    assert_tree_text "$m5" "用来检查换行和字宽"
}

# Markdown reached the renderer as markdown, not as source text.
#
# The cheapest regression here is the loudest one for a user: the renderer
# falls back to plain text and the answer arrives full of ``` fences and | pipe
# rows. Every "does the text appear" assertion in this file passes on that,
# because the text does appear — wearing its syntax.
assert_markdown_rendered() {
    local tree="$1"
    jq -e '[.data.ui_elements[]? | ((.value // "") | tostring)
            | select(contains("```"))] | length == 0' "$tree" >/dev/null \
        || die "a code fence reached the screen verbatim — markdown was printed, not rendered"
    jq -e '[.data.ui_elements[]? | ((.value // "") | tostring)
            | select(test("\\| *-{2,} *\\|"))] | length == 0' "$tree" >/dev/null \
        || die "a table separator row reached the screen verbatim — the table was not rendered"
}

# A fenced block is its own view, not a paragraph that happens to contain code.
#
# Measured: the surrounding prose sits at one depth and the code block one
# level deeper, with its newlines and indentation intact. If a refactor
# flattens that, the code still "appears" — as a wrapped, unindented,
# uncopyable smear.
assert_code_block_is_its_own_view() {
    local tree="$1" prose="$2" code="$3"
    python3 - "$tree" "$prose" "$code" <<'PYEOF'
import json, sys
tree, prose, code = sys.argv[1], sys.argv[2], sys.argv[3]
elements = json.load(open(tree))["data"]["ui_elements"]
def find(needle):
    return next((e for e in elements if needle in str(e.get("value", ""))), None)
prose_el, code_el = find(prose), find(code)
if prose_el is None:
    sys.exit(f"prose not found: {prose}")
if code_el is None:
    sys.exit(f"code not found: {code}")
if code_el is prose_el:
    sys.exit("code block was flattened into the prose accessibility node")
if "\n" not in str(code_el.get("value", "")):
    sys.exit("code block lost its line breaks")
PYEOF
}

# How many messages of each side the transcript is showing.
#
# User turns carry an Edit button, assistant turns carry a Retry button — the
# app's own distinction, not one this harness invents. Counting them is how a
# multi-turn flow proves nothing was dropped, merged or duplicated; asserting
# only that the LAST answer is on screen cannot tell a five-turn conversation
# from a one-turn one.
transcript_counts() {
    local tree="$1"
    jq -r '[.data.ui_elements[]? | (.identifier // "")]
           | { user:  [ .[] | select(startswith("ChatView.Message.Edit."))  ] | length,
               model: [ .[] | select(startswith("ChatView.Message.Retry.")) ] | length }
           | "\(.user) \(.model)"' "$tree"
}

# Counts every turn in the tree — which is only a valid completeness check
# while the WHOLE transcript is realized.
#
# The transcript is a virtualized scroll view: a message scrolled far enough out
# of view is removed from the accessibility tree, and the dump says so honestly
# with `walk.complete == true`. Measured on a 1024x681 window, `chat-depth` at
# turn 4 reported 3 user + 4 model with a complete walk, the first user bubble
# sitting at y=-429. Nothing was broken; it had simply scrolled away.
#
# So a shortfall here means one of two things, and they are not distinguishable
# from the counts alone: a dropped turn, or a window too short to hold them.
# Check the window height before reading it as a product bug.
assert_transcript_turns() {
    local tree="$1" expected="$2" counts user model
    counts="$(transcript_counts "$tree")"
    user="${counts% *}"
    model="${counts#* }"
    [[ "$user" == "$expected" && "$model" == "$expected" ]] \
        || die "expected $expected user + $expected model message(s), tree shows ${user} + ${model} (a virtualized transcript drops off-screen turns — check the window is tall enough before reading this as a dropped message)"
}

# Do these strings appear in the transcript IN THIS ORDER?
#
# A conversation that shows every turn but in the wrong order is still broken,
# and every "does the text appear" assertion in this file would pass on it.
# `ui_elements` is emitted in tree order, so position in that array is reading
# order.
assert_text_order() {
    local tree="$1"
    shift
    local needles=("$@")
    python3 - "$tree" "${needles[@]}" <<'PYEOF'
import json, sys
tree, needles = sys.argv[1], sys.argv[2:]
elements = json.load(open(tree))["data"]["ui_elements"]
haystack = [str(e.get("value", "")) + " " + str(e.get("title", "")) for e in elements]
# Position is (element index, offset inside that element), not the element
# index alone. Two needles inside ONE element used to compare equal, so a
# transcript that flattened turns into a single node — the extreme case being
# one node holding every needle — satisfied `sorted()` in any visual order.
positions = []
for needle in needles:
    hit = next(
        ((i, text.index(needle)) for i, text in enumerate(haystack) if needle in text),
        None,
    )
    if hit is None:
        sys.exit(f"transcript never shows: {needle}")
    positions.append(hit)
# Strictly increasing, not merely sorted: equal positions mean two turns share
# one element, which is itself the flattening regression.
if any(b <= a for a, b in zip(positions, positions[1:])):
    order = ", ".join(f"{n}@{p[0]}+{p[1]}" for n, p in zip(needles, positions))
    sys.exit(f"transcript is out of order: {order}")
PYEOF
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
            # The coordinate fallback is peekaboo's. A flow that declared
            # itself peekaboo-free has no fallback left, and reaching for one
            # anyway would surface as a bare "peekaboo: command not found"
            # attached to whichever assertion happened to run next.
            command -v peekaboo >/dev/null \
                || die "AXPress on TelemetryConsent.DontShare failed and $FLOW has no peekaboo fallback"
            read -r x y < <(jq -r '.data.ui_elements[] | select(.identifier == "TelemetryConsent.DontShare") | [(.bounds.x + .bounds.width / 2), (.bounds.y + .bounds.height / 2)] | @tsv' "$tree")
            pb_click_coords "$x,$y" --app "PID:$APP_PID" --json > "$OUT/consent-coordinate-fallback.json"
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
    if flow_requires_peekaboo; then
        pb menu click --app "PID:$APP_PID" --item 'Settings…' --json > "$OUT/open-settings.json"
    else
        # Settings persistence is deliberately part of the unattended,
        # AX-only suite. Use the standard macOS shortcut so that flow does
        # not quietly depend on Peekaboo just to open the window.
        osascript - "$APP_PID" > "$OUT/open-settings.json" <<'APPLESCRIPT'
on run argv
    set targetPID to (item 1 of argv) as integer
    tell application "System Events"
        set frontmost of first application process whose unix id is targetPID to true
        keystroke "," using command down
    end tell
    return "{\"success\":true,\"method\":\"command-comma\"}"
end run
APPLESCRIPT
    fi
    local probe=2 opened=0
    for _ in {1..40}; do
        probe=0
        ax_window_present Settings "$OUT/settings-windows.json" || probe=$?
        if [[ "$probe" == 0 ]]; then opened=1; break; fi
        sleep 0.25
    done
    if [[ "$opened" == 1 ]]; then
        # Screenshot flows retain the old window-id postcondition. Semantic
        # flows intentionally stop at the AX proof above, avoiding a Screen
        # Recording dependency for an ID they never consume.
        if flow_requires_screen_recording; then
            SETTINGS_WINDOW_ID=""
            for _ in {1..40}; do
                if ! pb list windows --app "PID:$APP_PID" --json > "$OUT/settings-cg-windows.json" 2>/dev/null; then
                    sleep 0.25
                    continue
                fi
                SETTINGS_WINDOW_ID="$(jq -r '.data.windows[]? | select(.title == "Settings") | .window_id' "$OUT/settings-cg-windows.json" | head -1)"
                [[ -n "$SETTINGS_WINDOW_ID" ]] && break
                sleep 0.25
            done
            [[ -n "$SETTINGS_WINDOW_ID" ]] || die "Settings opened but has no screenshot window ID"
        fi
        return
    fi
    [[ "$probe" == 2 ]] && die "could not observe whether the Settings window opened"
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
        python3 "$BASELINE_TOOL" check "$tree" \
            --scrub "$FAKE_ALIAS" --scrub "$FAKE_IMAGE_ALIAS" \
            --baseline "$committed" --observed "$observed" --update \
            || die "could not update AX structural baseline: $name"
    else
        python3 "$BASELINE_TOOL" check "$tree" \
            --scrub "$FAKE_ALIAS" --scrub "$FAKE_IMAGE_ALIAS" \
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
    # The real engine registry always contains the starter. Without this row,
    # the fake catalog makes the app correctly fall back to its only chat row
    # and the assertion below can never prove the production first-run rule.
    start_persona fresh-install FAKE_INCLUDE_STARTER=1
    see_main "$OUT/consent-visible.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' "$OUT/consent-visible.json" >/dev/null \
        || die "fresh install did not show telemetry consent"
    role_subtree_only "$OUT/consent-visible.json" AXSheet "$OUT/consent-sheet.json"
    baseline fresh-install.consent "$OUT/consent-sheet.json"
    # #1560: merely launching a fresh install must not inspect model caches
    # behind the consent sheet. Give both SwiftUI catalog tasks time to run;
    # the fake sidecar records every non-serve command before it exits.
    sleep 0.75
    if [[ -s "$OUT/fake-events.jsonl" ]] && jq -e \
        'select(.event == "command" and (.subcommand == "models" or .subcommand == "ls"))' \
        "$OUT/fake-events.jsonl" >/dev/null; then
        die "#1560: first launch probed the model catalog before user interaction"
    fi
    dismiss_first_run
    selected_model="$(element_field "$OUT/steady.json" ModelPickerBar.ModelMenu value)"
    [[ "$selected_model" == *"lfm2.5-1b-4bit"* ]] \
        || die "#1564: skipping Quickstart selected '$selected_model' instead of the small starter"
    for id in Sidebar.NewChat Sidebar.Launch rapid.chat.compose ChatView.SendOrStopButton ModelPickerBar.ModelMenu; do
        jq -e --arg id "$id" '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/steady.json" >/dev/null \
            || die "post-onboarding shell missing $id"
    done
    baseline fresh-install.steady "$OUT/steady.json"
    pb image --window-id "$MAIN_WINDOW_ID" --path "$OUT/final.png" --json > "$OUT/final-image.json"
    cleanup_persona
}

flow_cached_quickstart() {
    log "cached Quickstart starts without downloading (#1793)"
    # Reproduce #1618, not merely its configuration strings: an
    # operator-owned rapid-mlx-shaped listener is alive on the default port
    # before the dogfood app launches. The isolated persona must bind its own
    # high port without sweeping or terminating this process.
    FAKE_EVENT_LOG="$OUT_ROOT/operator-events.jsonl" \
        "$ROOT/scripts/fake-rapid-mlx.sh" serve operator-owned \
        --host 127.0.0.1 --port 8000 > "$OUT_ROOT/operator-server.log" 2>&1 &
    OPERATOR_SERVER_PID=$!
    for _ in {1..40}; do
        curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
        sleep 0.1
    done
    curl -fsS http://127.0.0.1:8000/healthz >/dev/null \
        || die "operator-shaped server did not bind :8000 for the isolation repro"
    kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null \
        || die ":8000 was already occupied; cannot establish the owned-server isolation repro"

    # Include the real cold-cache notice alongside the deterministic cached
    # fixture. Catalog output can be interleaved with prose; the chooser must
    # never promote that notice into a selectable model named "No" (#1918).
    start_persona cached-quickstart FAKE_EMPTY_CACHE_NOTICE=1

    see_main "$OUT/consent.json"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' \
        "$OUT/consent.json" >/dev/null; then
        press "$OUT/consent.json" TelemetryConsent.DontShare "$OUT/consent-dismiss.json"
    fi
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    jq -e '.data.ui_elements[]?
            | select(.identifier == "Quickstart.Progress")
            | select(.description == "Setup progress, step 1 of 4")' "$OUT/welcome.json" >/dev/null \
        || die "Quickstart welcome does not expose honest step progress"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_identifier "Quickstart.CachedModel.$FAKE_ALIAS" "$OUT/chooser.json"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Quickstart.CachedModel.No")' \
        "$OUT/chooser.json" >/dev/null; then
        die "empty-cache notice surfaced as a selectable model named No (#1918)"
    fi
    jq -e '.data.ui_elements[]?
            | select(.identifier == "Quickstart.Progress")
            | select(.description == "Setup progress, step 2 of 4")' "$OUT/chooser.json" >/dev/null \
        || die "Quickstart chooser does not advance its honest step progress"
    press "$OUT/chooser.json" "Quickstart.CachedModel.$FAKE_ALIAS" "$OUT/select-cached.json"
    see_main "$OUT/selected.json"
    assert_tree_text "$OUT/selected.json" "Start existing model"
    press "$OUT/selected.json" Quickstart.Footer.Primary "$OUT/start-existing.json"

    wait_fake_event \
        ".event == \"server_started\" and .alias == \"$FAKE_ALIAS\"" \
        "cached Quickstart did not start the selected model"
    jq -e -s 'any(.[]; .event == "server_started" and .alias == "fake-alias"
              and .port >= 49152 and .port <= 65535)' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "isolated persona did not bind its selected high port"
    kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null \
        || die "dogfood launch terminated the operator-owned :8000 server (#1618)"
    curl -fsS http://127.0.0.1:8000/healthz >/dev/null \
        || die "operator-owned :8000 server stopped responding after dogfood launch"

    # Ready is no longer completion: onboarding must hold the window until
    # the user explicitly confirms the final step. Pin both halves so a
    # future regression cannot silently restore the old auto-dismiss path.
    wait_identifier Quickstart.Ready.StartChatting "$OUT/ready-confirmation.json"
    jq -e '.data.ui_elements[]?
            | select(.identifier == "Quickstart.Progress")
            | select(.description == "Setup progress, step 4 of 4")' \
        "$OUT/ready-confirmation.json" >/dev/null \
        || die "Quickstart Ready does not report the final onboarding step"
    # SwiftUI sheets expose the covered window's AX descendants as well, so
    # the background composer may still be present in this tree. The Ready
    # action itself is the reliable contract: old auto-dismiss builds never
    # expose it, and this press is the only route that completes onboarding.
    press "$OUT/ready-confirmation.json" Quickstart.Ready.StartChatting \
        "$OUT/start-chatting.json"
    wait_identifier rapid.chat.compose "$OUT/ready.json"
    assert_tree_text "$OUT/ready.json" "chatting with fake-alias, running entirely on your Mac."
    [[ "$(jq '[.data.ui_elements[]? | select(.value? | strings | startswith("You’re chatting with fake-alias, running entirely on your Mac."))] | length' "$OUT/ready.json")" == 1 ]] \
        || die "Quickstart welcome was not seeded exactly once after confirmation"
    if jq -e -s 'any(.[]; .event == "command" and .subcommand == "pull")' \
        "$OUT/fake-events.jsonl" >/dev/null; then
        die "cached Quickstart invoked rapid-mlx pull instead of the start-only path"
    fi
    cleanup_persona
    cleanup_operator_server
}

flow_download_progress() {
    log "download progress never shows observed bytes above its total (#1550)"
    start_persona download-progress FAKE_DOWNLOAD_OVERRUN=1

    see_main "$OUT/consent.json"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.DontShare")' \
        "$OUT/consent.json" >/dev/null; then
        press "$OUT/consent.json" TelemetryConsent.DontShare "$OUT/consent-dismiss.json"
    fi
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    # The footer exists while Step 2 is still asynchronously reading the
    # catalogue. Waiting for that shared identifier can capture the transient
    # "Matching models" state before the recommendation and its size land.
    wait_tree_text "~633 MB" "$OUT/chooser.json"
    press "$OUT/chooser.json" Quickstart.Footer.Primary "$OUT/review-open.json"
    wait_identifier Quickstart.Review.Alias "$OUT/review.json"
    assert_tree_text "$OUT/review.json" "Download & start"
    press "$OUT/review.json" Quickstart.Footer.Primary "$OUT/download-start.json"

    local observed=0
    for _ in {1..40}; do
        see_main "$OUT/downloading.json"
        if jq -e '(.data.ui_elements | tostring) | contains("633 MB downloaded")' \
            "$OUT/downloading.json" >/dev/null; then
            observed=1
            break
        fi
        sleep 0.1
    done
    [[ "$observed" == 1 ]] \
        || die "overrun fixture never reached a truthful bytes-downloaded state"
    if jq -e '(.data.ui_elements | tostring)
              | test("633 MB[[:space:]]*/[[:space:]]*563 MB")' \
        "$OUT/downloading.json" >/dev/null; then
        die "download progress still shows observed bytes above its displayed total"
    fi
    jq -e -s 'any(.[]; .event == "command" and .subcommand == "pull")' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "download-progress flow never exercised the pull subprocess"
    cleanup_persona
}

flow_settings_persistence() {
    log "2/6 settings and persistence"
    start_persona settings-persistence
    dismiss_first_run
    open_settings
    wait_settings_stable "$OUT/settings-root.json"
    baseline settings-persistence.settings-root "$OUT/settings-root.json"
    press "$OUT/settings-root.json" Settings.Category.instructions "$OUT/settings-instructions-open.json"
    wait_settings_stable "$OUT/instructions-open.json" Settings.Instructions.GlobalEditor
    "$AX_DRIVER" set-value "$APP_PID" Settings.Instructions.GlobalEditor \
        "Keep answers concise and include runnable examples." > "$OUT/instructions-type.json"
    for _ in {1..20}; do
        [[ "$(defaults read "$BUNDLE_ID" rapid.custom-instructions.global.v1 2>/dev/null || true)" == \
            "Keep answers concise and include runnable examples." ]] && break
        sleep 0.1
    done
    [[ "$(defaults read "$BUNDLE_ID" rapid.custom-instructions.global.v1 2>/dev/null || true)" == \
        "Keep answers concise and include runnable examples." ]] \
        || die "global instructions did not persist to isolated preferences"
    wait_settings_stable "$OUT/instructions-saved.json" Settings.Instructions.GlobalEditor.Count
    baseline settings-persistence.instructions-saved "$OUT/instructions-saved.json"
    # #1717: configure a chosen model before it runs. This proves the panel is
    # not coupled to the current child, its model selector is addressable, and
    # a real control mutation reaches the honest "next load" state.
    press "$OUT/settings-root.json" Settings.Category.performance "$OUT/settings-performance-open.json"
    wait_settings_stable "$OUT/performance-open.json" Settings.Performance.ModelPicker
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.Performance.Panel")' \
        "$OUT/performance-open.json" >/dev/null || die "Performance settings panel did not mount"
    press "$OUT/performance-open.json" Settings.Performance.Prefix.Off "$OUT/performance-prefix-off.json"
    wait_settings_stable "$OUT/performance-saved.json" Settings.Performance.AppliesNextLoad
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.Performance.AppliesNextLoad" and (.value | contains("next time")))' \
        "$OUT/performance-saved.json" >/dev/null || die "Performance settings did not explain deferred application"
    baseline settings-persistence.performance-saved "$OUT/performance-saved.json"
    press "$OUT/performance-saved.json" Settings.Category.modelManagement "$OUT/settings-models-open.json"
    wait_settings_stable "$OUT/models-before.json" Settings.Models.ShowAllModelsToggle
    # GoldenFlow coverage for the recommendation SSOT: the running GUI must
    # render exactly the smart + fast aliases selected from the same JSON the
    # CLI consumes. This catches a missing app resource, a decoder drift, and a
    # third recommendation accidentally creeping back into a tier.
    local recommendation_json="$ROOT/../../vllm_mlx/model_recommendations.json"
    local ram_bytes
    ram_bytes="$(sysctl -n hw.memsize)"
    local expected_recommendations
    expected_recommendations="$(python3 - "$recommendation_json" "$ram_bytes" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
ram_gb = int(sys.argv[2]) / (1 << 30)
tier = payload["tiers"][0]
for candidate in payload["tiers"]:
    if ram_gb >= candidate["floor_gb"]:
        tier = candidate
print("\n".join(pick["alias"] for pick in tier["picks"]))
PY
)"
    local expected_smart expected_fast
    expected_smart="$(printf '%s\n' "$expected_recommendations" | sed -n '1p')"
    expected_fast="$(printf '%s\n' "$expected_recommendations" | sed -n '2p')"
    [[ -n "$expected_smart" && -n "$expected_fast" && "$(printf '%s\n' "$expected_recommendations" | sed -n '3p')" == "" ]] \
        || die "recommendation SSOT did not select exactly two aliases"
    jq -e --arg alias "$expected_smart" \
        '.data.ui_elements[]? | select(.identifier == ("Settings.ModelManagement.Recommended.Download." + $alias))' \
        "$OUT/models-before.json" >/dev/null || die "GUI did not render SSOT smart recommendation $expected_smart"
    jq -e --arg alias "$expected_fast" \
        '.data.ui_elements[]? | select(.identifier == ("Settings.ModelManagement.Recommended.Download." + $alias))' \
        "$OUT/models-before.json" >/dev/null || die "GUI did not render SSOT fast recommendation $expected_fast"
    [[ "$(jq '[.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.Recommended.primary" or .identifier == "Settings.ModelManagement.Recommended.alt")] | length' "$OUT/models-before.json")" -eq 2 ]] \
        || die "GUI recommendation section did not render exactly smart + fast cards"
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
    press "$OUT/settings-relaunch.json" Settings.Category.instructions "$OUT/settings-instructions-reopen.json"
    wait_settings_stable "$OUT/instructions-persisted.json" Settings.Instructions.GlobalEditor
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.Instructions.GlobalEditor" and .value == "Keep answers concise and include runnable examples.")' \
        "$OUT/instructions-persisted.json" >/dev/null \
        || die "relaunch did not restore global instructions in the editor"
    baseline settings-persistence.instructions-after-relaunch "$OUT/instructions-persisted.json"
    press "$OUT/instructions-persisted.json" Settings.Category.modelManagement "$OUT/settings-models-reopen.json"
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
    press "$OUT/chat-settled.json" ChatView.ConversationInstructions "$OUT/conversation-instructions-open-press.json"
    wait_identifier ChatView.ConversationInstructions.Editor "$OUT/conversation-instructions-open.json"
    "$AX_DRIVER" set-value "$APP_PID" ChatView.ConversationInstructions.Editor \
        "Answer this conversation as a product analyst." > "$OUT/conversation-instructions-type.json"
    see_main "$OUT/conversation-instructions-draft.json"
    press "$OUT/conversation-instructions-draft.json" ChatView.ConversationInstructions.Save \
        "$OUT/conversation-instructions-save.json"
    baseline chat-restore.conversation-instructions "$OUT/conversation-instructions-draft.json"
    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/chat-restored.json"
    assert_tree_text "$OUT/chat-restored.json" "golden restore marker"
    # Relaunch restarts the fake model too. Search results are a modal overlay
    # on the whole window, so its structural baseline otherwise races the
    # transient readiness band and residency controls behind the panel.
    wait_send_idle "$OUT/chat-restored-ready.json"

    # Conversation search is a window-level recovery path, including for
    # history that is not currently visible in the sidebar. Exercise the real
    # toolbar button, live filtering, result selection, and dismissal by
    # opening the restored transcript from the panel.
    press "$OUT/chat-restored.json" Toolbar.SearchChats "$OUT/search-open-press.json"
    wait_identifier ConversationSearch.Field "$OUT/search-open.json"
    "$AX_DRIVER" set-value "$APP_PID" ConversationSearch.Field "golden restore" \
        > "$OUT/search-type.json"
    local search_result_id=""
    for _ in {1..40}; do
        see_main "$OUT/search-filtered.json"
        search_result_id="$(jq -r '.data.ui_elements[]? | (.identifier // "")
            | select(test("^ConversationSearch\\.Result\\.[0-9A-Fa-f-]{36}$"))' \
            "$OUT/search-filtered.json" | head -1)"
        [[ -n "$search_result_id" ]] && break
        sleep 0.1
    done
    [[ -n "$search_result_id" ]] || die "conversation search did not return the restored chat"
    assert_tree_text "$OUT/search-filtered.json" "golden restore marker"
    baseline chat-restore.search-results "$OUT/search-filtered.json"
    press "$OUT/search-filtered.json" "$search_result_id" "$OUT/search-result-open.json"
    for _ in {1..40}; do
        see_main "$OUT/search-dismissed.json"
        if ! jq -e '.data.ui_elements[]? | select(.identifier == "ConversationSearch.Field")' \
            "$OUT/search-dismissed.json" >/dev/null; then break; fi
        sleep 0.1
    done
    jq -e '[.data.ui_elements[]? | select(.identifier == "ConversationSearch.Field")] | length == 0' \
        "$OUT/search-dismissed.json" >/dev/null \
        || die "opening a conversation did not dismiss the search panel"
    assert_tree_text "$OUT/search-dismissed.json" "deterministic content"

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
    press "$OUT/chat-restored-settled.json" ChatView.ConversationInstructions \
        "$OUT/conversation-instructions-reopen-press.json"
    wait_identifier ChatView.ConversationInstructions.Editor "$OUT/conversation-instructions-restored.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "ChatView.ConversationInstructions.Editor" and .value == "Answer this conversation as a product analyst.")' \
        "$OUT/conversation-instructions-restored.json" >/dev/null \
        || die "relaunch did not restore per-conversation instructions"
    press "$OUT/conversation-instructions-restored.json" ChatView.ConversationInstructions.Cancel \
        "$OUT/conversation-instructions-close.json"
    baseline chat-restore.transcript-restored "$OUT/chat-restored-settled.json"

    # #1588: these controls existed for months without ever being mounted.
    # Drive the assembled app so a future refactor cannot quietly orphan them
    # again while their unit tests stay green.
    jq -e '.data.ui_elements[]? | select(.identifier == "ContentView.ToggleLogs")' \
        "$OUT/chat-restored-settled.json" >/dev/null \
        || die "the status footer/log affordance is not mounted"
    press "$OUT/chat-restored-settled.json" ContentView.ToggleLogs "$OUT/logs-open-press.json" \
        || die "the mounted log toggle is not pressable"
    wait_identifier ContentView.LogDrawer "$OUT/logs-open.json"
    press "$OUT/logs-open.json" ContentView.ToggleLogs "$OUT/logs-close-press.json" \
        || die "the mounted log drawer cannot be closed"
    # `press` records the action response, not a fresh accessibility tree.
    # Wait for the drawer transition and inspect the settled main window.
    for _ in {1..40}; do
        see_main "$OUT/logs-closed.json"
        if ! jq -e '.data.ui_elements[]? | select(.identifier == "ContentView.LogDrawer")' \
            "$OUT/logs-closed.json" >/dev/null; then break; fi
        sleep 0.1
    done
    jq -e '.data.ui_elements[]? | select(.identifier == "ContentView.ToggleLogs")' \
        "$OUT/logs-closed.json" >/dev/null \
        || die "the status footer disappeared after closing logs"
    local select_text_id
    select_text_id="$(jq -r '.data.ui_elements[]? | (.identifier // "")
        | select(startswith("ChatView.Message.SelectText."))' \
        "$OUT/logs-closed.json" | head -1)"
    [[ -n "$select_text_id" ]] || die "completed transcript exposes no Select text action"
    press "$OUT/logs-closed.json" "$select_text_id" "$OUT/select-text-press.json" \
        || die "Select text action is not pressable"
    for _ in {1..40}; do
        see_main "$OUT/select-text-sheet.json"
        if jq -e '(.data.ui_elements | tostring) | contains("Selection here crosses paragraphs")' \
            "$OUT/select-text-sheet.json" >/dev/null; then break; fi
        sleep 0.1
    done
    assert_tree_text "$OUT/select-text-sheet.json" "Selection here crosses paragraphs"
    cleanup_persona
}

flow_math_rendering() {
    # Artifact-level coverage for #1504/#1576. The fake emits display math;
    # MathView exposes `Math:` only after SwiftMath parsed and hosted it, while
    # the safe fallback exposes `Unrenderable math:`. This therefore catches
    # both a missing font bundle and a parser/resource regression in the real
    # assembled app.
    start_persona math-rendering
    dismiss_first_run
    start_model
    send_prompt "shape:math show me the Gaussian integral" math
    wait_send_idle "$OUT/math-settled.json"
    # TextKit 2's custom block stack does not expose SwiftMath's private
    # NSViewRepresentable label through the flattened AX walk. Keep the
    # artifact gate on what it can prove reliably: the formula is segmented
    # out of prose (neither raw $$ source nor fallback is printed), while the
    # surrounding blocks survive on both sides. Unit coverage pins the exact
    # MathBlock latex payload and MathView still owns parse/resource fallback.
    assert_tree_text "$OUT/math-settled.json" "The Gaussian integral is"
    assert_tree_text "$OUT/math-settled.json" 'and inline it reads $e^{i\\pi} + 1 = 0$.'
    if jq -e '(.data.ui_elements | tostring) | contains("$$\\\\int_")' \
        "$OUT/math-settled.json" >/dev/null; then
        die "display math reached the transcript as literal source"
    fi
    if jq -e '(.data.ui_elements | tostring) | contains("Unrenderable math:")' \
        "$OUT/math-settled.json" >/dev/null; then
        die "SwiftMath took the literal-source fallback in the assembled app"
    fi
    cleanup_persona
}

flow_restored_tools() {
    log "restored conversation keeps deterministic web research"
    start_persona restored-tools RAPID_GUI_WEB_SEARCH_FIXTURE=1
    dismiss_first_run
    start_model
    send_prompt "What's a major news story from the last week?" restored-tools-first
    wait_send_idle "$OUT/first-settled.json"
    assert_tree_text "$OUT/first-settled.json" "Tool call web_search"
    assert_tree_text "$OUT/first-settled.json" "Golden technology story"

    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/restored.json"
    local conversation_id
    conversation_id="$(jq -r '.data.ui_elements[] | (.identifier // "")
        | select(test("^Sidebar\\.Conversation\\.[0-9A-Fa-f-]{36}$"))' \
        "$OUT/restored.json" | head -1)"
    [[ -n "$conversation_id" ]] || die "restored tool conversation row missing"
    press "$OUT/restored.json" "$conversation_id" "$OUT/opened.json"
    # Relaunch starts a fresh sidecar. The restored transcript can become
    # interactive before that sidecar is ready, so sending immediately races
    # the readiness gate and silently leaves the prompt in the composer.
    wait_send_idle "$OUT/restored-ready.json"
    send_prompt "What about technology? Find one concrete story and summarize it." restored-tools-followup
    wait_send_idle "$OUT/followup-settled.json"
    assert_tree_text "$OUT/followup-settled.json" "Golden technology story"

    jq -s -e '[.[] | select(.event == "chat_request")
        | select((.roles | index("tool")) != null)
        | select((.tools | index("web_search")) != null)] | length == 2' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "fresh/restored synthesis requests did not both carry web evidence and tools"
    cleanup_persona
}

flow_tool_loop_budget() {
    log "runaway tool use ends with a bounded synthesis answer"
    start_persona tool-loop-budget RAPID_GUI_WEB_SEARCH_FIXTURE=1
    dismiss_first_run
    start_model
    send_prompt "shape:tool-loop research this topic thoroughly" tool-loop-budget
    wait_send_idle "$OUT/settled.json"
    assert_tree_text "$OUT/settled.json" "Golden tool-loop synthesis"

    jq -s -e '[.[] | select(.event == "tool_loop_call")] | length == 3' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "the app did not stop after exactly three tool executions"
    jq -s -e '[.[] | select(.event == "tool_loop_synthesis" and .tool_results == 3)] | length == 1' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "the capped loop did not finish with one synthesis request"
    jq -s -e '[.[] | select(.event == "chat_request")][-1].tools == []' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "the final synthesis request still advertised tools"
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
    # Stop a stream that is actually streaming CONTENT.
    #
    # The button flips to "Stop generating" on the first delta, and that delta
    # is a REASONING token — the answer itself has not started. Pressing there
    # leaves a bubble with no content node; pressing a moment later leaves one
    # with. Both are legitimate app states, and the structural baseline can
    # only pin one of them.
    #
    # Measured, same commit: this dev machine always had the content by then
    # and the hosted runner never did, so whichever machine wrote the baseline
    # made it un-enforceable on the other — three local runs were stable, which
    # is exactly what makes this kind of race so easy to commit by accident.
    #
    # Waiting for the first content token removes the race and sharpens what
    # the flow claims to test: cancelling a response that is being produced,
    # not one that has yet to start.
    for _ in {1..80}; do
        see_main "$OUT/slow-streaming.json"
        if jq -e '(.data.ui_elements | tostring) | contains("Hello")' \
            "$OUT/slow-streaming.json" >/dev/null; then break; fi
        sleep 0.1
    done
    assert_tree_text "$OUT/slow-streaming.json" "Hello"
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

    # Release dogfood found a second Stop edge: cancelling before the first
    # content token left an unanswered user prompt in wire history. The next
    # request then answered that cancelled prompt instead of the new one.
    # Exercise that zero-content lane and prove the immediately-following turn
    # is routed from its own prompt.
    send_prompt "cancel this before content" zero-content-stop
    for _ in {1..40}; do
        see_main "$OUT/zero-content-streaming.json"
        if [[ "$(element_field "$OUT/zero-content-streaming.json" ChatView.SendOrStopButton description)" == "Stop generating" ]]; then break; fi
        sleep 0.05
    done
    [[ "$(element_field "$OUT/zero-content-streaming.json" ChatView.SendOrStopButton description)" == "Stop generating" ]] \
        || die "zero-content request never transitioned to Stop generating"
    press "$OUT/zero-content-streaming.json" ChatView.SendOrStopButton "$OUT/zero-content-stop.json"
    wait_send_idle "$OUT/zero-content-stopped.json"
    send_prompt "shape:list answer the new request" after-stop
    wait_send_idle "$OUT/after-stop-settled.json"
    assert_tree_text "$OUT/after-stop-settled.json" "Three things, in order:"
    jq -n '{success: true, assertion: "a send immediately after zero-content Stop answers the new prompt"}' \
        > "$OUT/after-stop-assertion.json"
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
    # Capture the containing window instead of relying on Peekaboo's newer
    # area/region flags. The AX assertion above still proves that the sheet is
    # present, while a window capture works with both v3.0 beta and current
    # Peekaboo releases used across our dogfood Macs.
    pb image --window-id "$MAIN_WINDOW_ID" --path "$OUT/low-memory-selected.png" --json \
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
    # PR #1907: the automatic-update preference is part of the shipped App
    # panel, not just updater plumbing. Local golden builds intentionally omit
    # SUPublicEDKey, so the control is visible but disabled; signed release
    # builds enable the same control and default it on through Info.plist.
    jq -e '.data.ui_elements[]? | select(
        .identifier == "Settings.App.AutomaticUpdatesToggle"
    )' "$OUT/update-app-panel.json" >/dev/null \
        || die "Settings > App does not expose automatic background updates"
    baseline update-state.app-panel "$OUT/update-app-panel.json"
    log "  update state names the running version ($expected, via ${state##*.})"
    cleanup_persona
}

flow_window_close_prompt() {
    # #1590: the prompt, persistence store and delegate proxy all existed, but
    # no WindowAccessor ever attached the proxy to the real main NSWindow.
    start_persona window-close-prompt
    dismiss_first_run

    # AXPress blocks while NSAlert.runModal is active, so issue the native
    # close in the background, observe/answer the sheet from a second driver,
    # then require the original close action to finish successfully.
    "$AX_DRIVER" close-window "$APP_PID" Rapid-MLX > "$OUT/close-window.json" 2> "$OUT/close-window.err" &
    local close_pid=$!
    wait_identifier DockHidePrompt.NoButton "$OUT/dock-prompt.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "DockHidePrompt.YesButton")' \
        "$OUT/dock-prompt.json" >/dev/null \
        || die "first main-window close has no Yes choice"
    jq -e '.data.ui_elements[]? | select(.identifier == "DockHidePrompt.DontAskCheckbox")' \
        "$OUT/dock-prompt.json" >/dev/null \
        || die "first main-window close has no Don.t ask again choice"
    press "$OUT/dock-prompt.json" DockHidePrompt.NoButton "$OUT/dock-prompt-no.json"
    wait "$close_pid" || die "native main-window close action failed: $(cat "$OUT/close-window.err")"

    local probe=2
    for _ in {1..40}; do
        probe=0
        ax_window_present Rapid-MLX "$OUT/after-close.json" || probe=$?
        [[ "$probe" == 1 ]] && break
        sleep 0.25
    done
    [[ "$probe" == 1 ]] || die "No choice did not close the main window normally"
    log "  first main-window close presents and resolves the Dock prompt"
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
    for category in modelManagement instructions tools connectors performance appearance privacy app; do
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

    # Presence is not behaviour. Exercise reversible controls and require the
    # AX value/selection to round-trip after each press. This catches buttons
    # that highlight under the pointer but never mutate their binding.
    press "$OUT/dead-panel-app.json" Settings.Category.appearance "$OUT/dead-open-appearance-actions.json" \
        || die "Appearance category is not pressable"
    see_main "$OUT/dead-appearance-before.json"
    press "$OUT/dead-appearance-before.json" Settings.Appearance.Theme.dark "$OUT/dead-appearance-dark-press.json" \
        || die "Dark appearance option is not pressable"
    see_main "$OUT/dead-appearance-dark.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Appearance.Theme.dark")
           | select(.selected == true or .value == 1 or .value == "1")' \
        "$OUT/dead-appearance-dark.json" >/dev/null \
        || die "Dark appearance accepted AXPress but did not become selected"
    press "$OUT/dead-appearance-dark.json" Settings.Appearance.Theme.light "$OUT/dead-appearance-light-press.json" \
        || die "Light appearance option is not pressable"
    see_main "$OUT/dead-appearance-light.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Appearance.Theme.light")
           | select(.selected == true or .value == 1 or .value == "1")' \
        "$OUT/dead-appearance-light.json" >/dev/null \
        || die "Light appearance did not restore selection"

    press "$OUT/dead-appearance-light.json" Settings.Category.privacy "$OUT/dead-open-privacy-actions.json" \
        || die "Privacy category is not pressable"
    see_main "$OUT/dead-privacy-before.json"
    local telemetry_before telemetry_after
    telemetry_before="$(element_field "$OUT/dead-privacy-before.json" Settings.Privacy.TelemetryToggle value)"
    press "$OUT/dead-privacy-before.json" Settings.Privacy.TelemetryToggle "$OUT/dead-privacy-toggle.json" \
        || die "Telemetry toggle is not pressable"
    see_main "$OUT/dead-privacy-after.json"
    telemetry_after="$(element_field "$OUT/dead-privacy-after.json" Settings.Privacy.TelemetryToggle value)"
    [[ -n "$telemetry_before" && -n "$telemetry_after" && "$telemetry_before" != "$telemetry_after" ]] \
        || die "Telemetry toggle accepted AXPress but its value did not change"
    press "$OUT/dead-privacy-after.json" Settings.Privacy.TelemetryToggle "$OUT/dead-privacy-restore.json" \
        || die "Telemetry toggle could not be restored"

    local ax_contracts=(
        "dead-panel-tools.json|Settings.Tools.Toggle.web_search|Web search"
        "dead-panel-tools.json|Settings.Tools.Toggle.browse|Browse pages"
        "dead-panel-tools.json|Settings.Tools.Toggle.weather|Weather"
        "dead-panel-tools.json|Settings.Tools.Browse.AutoApproveToggle|Approve every page automatically"
        "dead-panel-app.json|Settings.App.HideDockOnCloseToggle|Hide Dock icon when closing window"
    )
    local contract file identifier label
    for contract in "${ax_contracts[@]}"; do
        IFS='|' read -r file identifier label <<< "$contract"
        jq -e --arg identifier "$identifier" --arg label "$label" \
            '.data.ui_elements[]?
             | select(.identifier == $identifier)
             | select(.description == $label)' \
            "$OUT/$file" >/dev/null \
            || die "$identifier has no readable VoiceOver label"
    done
    log "  Settings toggles expose readable VoiceOver labels"
    cleanup_persona
}

flow_browse_all_destination() {
    # An advertised destination must actually be one, must not cost the user
    # what they already chose, and must not be a way out of setup.
    #
    # "Browse all models →" on Quickstart step 2 was implemented as one line
    # that set a dismiss flag (#1653). It was present, enabled, correctly
    # labelled and carried an AXIdentifier, so every structural check passed —
    # the wizard simply vanished, the user's pick was discarded, and they
    # landed on whatever the alphabetical fallback chose (a 7.6 GB download
    # nobody asked for). None of that is visible in a tree dump.
    #
    # The first fix sent the user to the Settings model catalogue: a second
    # window, a staged tab, and a round trip back. Paper 05.2.J · S1
    # supersedes it — the catalogue is now a micro-stage INSIDE Step 2. So the
    # assertions below are the same three questions, asked of the new
    # destination: did anything happen, did setup survive, did the pick.
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
    #
    # Find the default POSITIVELY and exclude it by identifier — never pick by
    # `.selected != true`. Measured: SwiftUI publishes AXSelected only on the
    # card that IS selected; the other three omit the attribute entirely, and
    # rapid-ax also omits an attribute whose read failed. Absence therefore
    # means "not selected OR we failed to look", and the two are
    # indistinguishable by construction, not merely on a bad day. A `!= true`
    # pick can thus hand back the default itself, after which the round trip at
    # the end asserts only that the default is still the default — which stays
    # true when the wizard throws the user's choice away, i.e. the bug walks
    # straight through the flow written to catch it (#1653).
    #
    # Requiring exactly one card to claim selection is what keeps this honest:
    # if that read is the one that failed, the count is 0 and we retry rather
    # than quietly promoting some other card to "the default".
    local i chosen=""
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ba-chooser.json"
        chosen="$(jq -r '[.data.ui_elements[]?
                          | select((.identifier // "") | startswith("Quickstart.Choice."))]
                         | (map(select(.selected == true))) as $default
                         | if ($default | length) != 1 then empty
                           else (map(select(.identifier != $default[0].identifier))[0].identifier // empty)
                           end' \
                  "$OUT/ba-chooser.json")"
        if [[ -n "$chosen" ]]; then break; fi
        sleep 0.25
    done
    [[ -n "$chosen" ]] \
        || die "the chooser never showed exactly one selected card with another to pick — AXSelected did not read cleanly"
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

    # 1. It opened the catalogue, and it opened it HERE. Wait for one of the
    #    catalogue's own surfaces rather than for the list specifically — a
    #    persona with a fake engine may legitimately land on the empty-cache or
    #    error body, and this flow is about the destination, not the contents.
    local landed=0
    for ((i=0; i<60; i++)); do
        see_main "$OUT/ba-catalog.json"
        if jq -e '[.data.ui_elements[]?
                   | select((.identifier // "") | startswith("Quickstart.BrowseAll."))]
                  | length > 0' "$OUT/ba-catalog.json" >/dev/null 2>&1; then
            landed=1; break
        fi
        sleep 0.25
    done
    [[ "$landed" == 1 ]] \
        || die "Browse all models did not open the in-window catalogue — it is a dismiss button again (#1653)"
    log "  landed on the in-window catalogue"

    # 2. Setup is still on screen, at the same public step. The bug this
    #    replaces dismissed the wizard; the fix it replaces opened a second
    #    window. Neither may happen.
    jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Progress")' \
        "$OUT/ba-catalog.json" >/dev/null \
        || die "the setup rail is gone — browsing dismissed onboarding"
    jq -e '[.data.ui_elements[]?
            | select(.identifier == "Quickstart.Step2.Kicker")
            | .value // .title // .label // ""]
           | map(select(test("STEP 2 OF 4"; "i"))) | length > 0' \
        "$OUT/ba-catalog.json" >/dev/null \
        || die "the catalogue is not reporting Step 2 of 4 — a micro-stage became a step"

    # 3. No second window. `ax_window_present` returns 1 for "read the list, not
    #    there" and 2 for "could not look" — only an explicit 1 is evidence.
    local probe=0
    ax_window_present Settings "$OUT/ba-windows.json" || probe=$?
    case "$probe" in
        0) die "Browse all models opened a Settings window — Paper 05.2.J · S1 forbids it" ;;
        2) die "could not read the app's window list, so 'no second window' is unverified" ;;
    esac
    log "  no Settings window, no second window"

    # 4. If this fixture's intentionally tiny fake catalogue contains the
    #    shortlist pick, the matching row must expose the shared selection.
    #    Usually it does not: the fake reports only fake-alias rows, while the
    #    shortlist deliberately exercises the real recommended aliases. The
    #    unconditional proof that selection survived is therefore after Back,
    #    where the chosen row is guaranteed to exist.
    if jq -e --arg id "$chosen" '.data.ui_elements[]? | select(.identifier == $id)' \
        "$OUT/ba-catalog.json" >/dev/null; then
        jq -e --arg id "$chosen" '.data.ui_elements[]? | select(.identifier == $id) | select(.selected == true)' \
            "$OUT/ba-catalog.json" >/dev/null \
            || die "the matching catalogue row lost the user's selection (#1653)"
    fi

    # 5. Back, by the visible control, returns to the shortlist with the pick
    #    intact. Escape is a shortcut for this same control, which is why the
    #    control has to exist and has to work.
    press "$OUT/ba-catalog.json" Quickstart.Footer.Back "$OUT/ba-back.json" \
        || die "the catalogue's Back control is not pressable"
    wait_identifier Quickstart.BrowseAll "$OUT/ba-after.json"
    jq -e '[.data.ui_elements[]?
            | select((.identifier // "") | startswith("Quickstart.BrowseAll."))]
           | length == 0' "$OUT/ba-after.json" >/dev/null \
        || die "Back did not leave the catalogue"
    jq -e --arg id "$chosen" '.data.ui_elements[]? | select(.identifier == $id) | select(.selected == true)' \
        "$OUT/ba-after.json" >/dev/null \
        || die "the shortlist came back without the user's selection — Back must not discard it"
    log "  back on the shortlist, $chosen still selected"
    cleanup_persona
}

flow_chat_depth() {
    # One message is not a conversation.
    #
    # `chat-restore` sends a single prompt and checks it comes back after a
    # relaunch — that covers persistence and almost nothing about chatting. It
    # cannot see a second turn landing above the first, a turn being dropped
    # when the next one starts, or a restore that brings back only the last
    # exchange. And every answer it has ever rendered was the same paragraph of
    # plain text, so the code block, the table, the list and the CJK line have
    # never once been through the renderer in this suite.
    #
    # Each turn here asks the fake for a different SHAPE of answer. The fake has
    # no model, so this is not about whether an answer is any good — judging
    # that belongs to the eval suites against a real model. It is about the work
    # the APP does differently per shape, which is exactly what a GUI gate can
    # hold.
    start_persona chat-depth
    dismiss_first_run
    start_model

    # marker | what the user would be asking | a distinctive string the answer must contain
    local -a turns=(
        "shape:prose|write the opening of a story about a lighthouse|lighthouse keeper"
        "shape:code|show me fibonacci in python|def fib(n)"
        "shape:table|compare those two models for me|qwen3.5-9b"
        "shape:list|give me three steps|nested point"
        "shape:unicode|用中文回答并带上 emoji|中文排版测试"
    )

    local index=0 spec marker prompt expect
    for spec in "${turns[@]}"; do
        index=$((index + 1))
        marker="${spec%%|*}"
        prompt="${spec#*|}"; prompt="${prompt%%|*}"
        expect="${spec##*|}"
        # The marker travels in the prompt so the fake can pick the shape, and
        # it doubles as the per-turn needle for the ordering assertion below.
        send_prompt "$marker $prompt" "turn$index"
        wait_send_idle "$OUT/turn$index-settled.json"
        assert_tree_text "$OUT/turn$index-settled.json" "$expect"
        # After turn N there must be exactly N of each, every time — not just
        # at the end, so a turn that vanishes is attributed to the turn that
        # dropped it.
        assert_transcript_turns "$OUT/turn$index-settled.json" "$index"
        log "  turn $index ($marker) rendered and both sides counted"
    done

    # Prompts AND answers, interleaved, inside the transcript only.
    #
    # Ordering the prompts alone cannot see a transcript that brings every
    # turn back but pairs the fifth answer with the first question: check one
    # side and both arrangements are equally "sorted". Interleaving is what
    # pins each answer to the prompt it belongs under.
    local -a conversation=()
    for spec in "${turns[@]}"; do
        conversation+=("${spec%%|*}" "${spec##*|}")
    done
    transcript_only "$OUT/turn5-settled.json" "$OUT/turn5-transcript.json"
    assert_text_order "$OUT/turn5-transcript.json" "${conversation[@]}"
    # …and each half is in the message that half belongs to. Reading order
    # alone would accept an answer rendered inside the user's own bubble.
    assert_turns_pair_up "$OUT/turn5-transcript.json" "${conversation[@]}"
    log "  all 5 turns present, each answer inside its own assistant message"

    # The shapes are only worth sending if something asserts on what the
    # renderer did with them — positively, not just "the source syntax is
    # absent".
    assert_rendered_shapes "$OUT/turn5-transcript.json" "$OUT/turn5"
    log "  markdown rendered: table cells and list items are their own elements,"
    log "  no raw fences, pipe rows or list markers, code block nested and intact,"
    log "  and the CJK answer kept its emoji and its right-to-left run"
    baseline chat-depth.five-turns "$OUT/turn5-settled.json"

    # Restore has to bring back the WHOLE conversation. `chat-restore` only
    # ever proved that one message survived, which a store that keeps the last
    # exchange would also pass.
    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/depth-restored.json"
    local conversation_id
    conversation_id="$(jq -r '.data.ui_elements[] | (.identifier // "")
        | select(test("^Sidebar\\.Conversation\\.[0-9A-Fa-f-]{36}$"))' \
        "$OUT/depth-restored.json" | head -1)"
    [[ -n "$conversation_id" ]] || die "restored conversation row was not exposed to AX"
    press "$OUT/depth-restored.json" "$conversation_id" "$OUT/depth-open-restored.json"
    wait_send_idle "$OUT/depth-restored-transcript.json"
    assert_transcript_turns "$OUT/depth-restored-transcript.json" 5
    # The same interleaved, transcript-scoped check as before the relaunch.
    # A restore that returns five prompts with the answers shuffled between
    # them is a broken restore, and prompt-only ordering cannot see it.
    transcript_only "$OUT/depth-restored-transcript.json" \
        "$OUT/depth-restored-scoped.json"
    assert_text_order "$OUT/depth-restored-scoped.json" "${conversation[@]}"
    assert_turns_pair_up "$OUT/depth-restored-scoped.json" "${conversation[@]}"
    # Same bar as the live transcript. Without this a restore that brought
    # every turn back but flattened the table, dropped the emoji or printed
    # the list markers would pass — the counts survive it, and the structural
    # baseline normalizes every value to `text`, so neither can see it.
    assert_rendered_shapes "$OUT/depth-restored-scoped.json" "$OUT/depth-restored"
    log "  all 5 turns restored, each answer still under its own prompt,"
    log "  and every shape still rendered the way it was before the relaunch"
    baseline chat-depth.restored "$OUT/depth-restored-transcript.json"
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
    local catalog_ready=0
    for _ in {1..40}; do
        see_main "$OUT/catalog-main.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "ModelPickerBar.ModelMenu" and .value == "fake-alias")' \
               "$OUT/catalog-main.json" >/dev/null; then
            catalog_ready=1
            break
        fi
        sleep 0.25
    done
    [[ "$catalog_ready" == 1 ]] || die "chat catalog inventory was not observed"

    # This fixture's catalog is populated by app-owned `models` / `ls` probes.
    # (The Swift subprocess test separately exercises the conditional `info`
    # sibling probe, which this fixture deliberately has no candidate for.)
    # They are implementation details, not real engine sessions (#1415).
    # The model-menu value above is the completion barrier: ModelCatalog.load
    # publishes that merged inventory only after its models/ls tasks have
    # all returned, so the event log now contains the full initial probe set.
    jq -e -s '[.[] | select(.event == "command")]
              | (map(.subcommand) | index("models") != null)
                and (map(.subcommand) | index("ls") != null)
                and all(.[]; .do_not_track == "1")' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "an internal catalog probe launched with engine telemetry enabled"

    jq -e '.data.walk.complete == true' "$OUT/catalog-main.json" >/dev/null \
        || die "could not completely observe the chat catalog"
    jq -e '[.data.ui_elements[]? | select([(.identifier // ""), (.value // ""), (.title // ""), (.description // "")] | map(tostring) | join(" ") | test("fake-video-alias"))] | length == 0' \
        "$OUT/catalog-main.json" >/dev/null \
        || die "a video-gen alias reached the chat surface"

    open_settings
    see_main "$OUT/catalog-settings.json"
    press "$OUT/catalog-settings.json" Settings.Category.modelManagement \
        "$OUT/catalog-open-mm.json"
    see_main "$OUT/catalog-model-management.json"
    jq -e '.data.walk.complete == true' "$OUT/catalog-model-management.json" >/dev/null \
        || die "could not completely observe Model Management"
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.Row.fake-alias")' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "Model Management inventory was not observed"
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.Row.fake-external-alias")' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "external model was not visible in Model Management"
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.StorageSummary")' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "Model Management disk overview was not visible"
    jq -e '.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.LargestModel")
              | [(.title // ""), (.value // ""), (.description // "")]
              | join(" ") | contains("fake-image-alias")' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "disk overview did not identify the largest managed model"
    jq -e '[.data.ui_elements[]? | select(.identifier == "Settings.ModelManagement.Delete.fake-external-alias")] | length == 0' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "external model exposed a delete action"
    jq -e '[.data.ui_elements[]? | select([(.title // ""), (.description // ""), (.help // "")] | join(" ") | test("another app"; "i"))] | length > 0' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "external model was not labelled as owned by another app"
    jq -e '[.data.ui_elements[]? | select([(.identifier // ""), (.value // ""), (.title // ""), (.description // "")] | map(tostring) | join(" ") | test("fake-video-alias"))] | length == 0' \
        "$OUT/catalog-model-management.json" >/dev/null \
        || die "a video-gen alias reached Model Management"
    log "  no video-gen alias on either catalog surface; external model is visible and read-only"
    cleanup_persona
}

flow_chat_document_attachment() {
    start_persona chat-document-attachment
    dismiss_first_run
    start_model

    local fixture="$ROOT/Tests/GUIGoldenFlows/Fixtures/chat-document.txt"
    see_main "$OUT/document-compose.json"
    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$fixture" \
        > "$OUT/document-paste.json"

    wait_identifier ChatView.Attachment.Remove.chat-document.txt \
        "$OUT/document-attached.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ChatView.Attachment.Remove.chat-document.txt")' \
        "$OUT/document-attached.json" >/dev/null \
        || die "the pasted TXT file did not become a removable attachment chip"

    send_prompt "Which region is in this document?" document
    for _ in {1..40}; do
        if jq -e -s 'any(.[]; .event == "chat_request"
                       and any(.user_texts[]?;
                           contains("BEGIN RAPID ATTACHMENT")
                           and contains("Revenue: 42")
                           and contains("Region: APAC")))' \
            "$OUT/fake-events.jsonl" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
    done
    jq -e -s 'any(.[]; .event == "chat_request"
                   and any(.user_texts[]?;
                       contains("BEGIN RAPID ATTACHMENT")
                       and contains("Revenue: 42")
                       and contains("Region: APAC")))' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "the document chip sent no extracted local text to the model"

    relaunch_persona
    dismiss_first_run
    wait_identifier Sidebar.NewChat "$OUT/document-restored-root.json"
    local conversation_id
    conversation_id="$(jq -r '.data.ui_elements[] | (.identifier // "")
        | select(test("^Sidebar\\.Conversation\\.[0-9A-Fa-f-]{36}$"))' \
        "$OUT/document-restored-root.json" | head -1)"
    [[ -n "$conversation_id" ]] || die "document conversation did not persist"
    press "$OUT/document-restored-root.json" "$conversation_id" \
        "$OUT/document-open-restored.json"
    local restored=0
    for _ in {1..40}; do
        see_main "$OUT/document-restored.json"
        if jq -e '(.data.ui_elements | tostring) | contains("chat-document.txt")' \
            "$OUT/document-restored.json" >/dev/null; then
            restored=1
            break
        fi
        sleep 0.25
    done
    [[ "$restored" == 1 ]] || die "the restored transcript lost the document chip"
    assert_tree_text "$OUT/document-restored.json" "chat-document.txt"
    if jq -e '(.data.ui_elements | tostring) | contains("Revenue: 42")' \
        "$OUT/document-restored.json" >/dev/null; then
        die "extracted document contents leaked into the visible transcript"
    fi
    cleanup_persona
}

# Wait until the fake has recorded an event matching a jq predicate.
#
# The event log is the independent witness. Every "did it work?" question in
# this flow has a UI answer and a wire answer, and only the wire answer can
# tell a render that happened from a render the UI merely drew a card for.
wait_fake_event() {
    local predicate="$1" what="$2" i
    for ((i=0; i<200; i++)); do
        if [[ -s "$OUT/fake-events.jsonl" ]] \
           && jq -e -s "any(.[]; $predicate)" "$OUT/fake-events.jsonl" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    die "$what"
}

# Put text in the Images composer and PROVE it arrived.
#
# ``set-value`` reports success on whatever element carries the identifier,
# which is not the same thing as the SwiftUI binding updating. Measured: with
# the identifier on the wrapper, the driver set the placeholder AXStaticText,
# answered {"success":true}, the prompt stayed empty, ``Images.Generate``
# stayed disabled, and the subsequent press was silently dropped — a green
# type step followed by a render that never happened. The editor now carries
# its own identifier (``rapid.images.compose``), and the gate below is what
# keeps a future regression of that wiring loud.
type_prompt() {
    local text="$1" prefix="$2" i
    "$AX_DRIVER" set-value "$APP_PID" rapid.images.compose "$text" \
        > "$OUT/$prefix-type.json"
    for ((i=0; i<40; i++)); do
        see_main "$OUT/$prefix.json"
        # The composer holds the text AND the button it gates is live. Either
        # alone can lie: the editor can hold text the binding never saw, and
        # the button is disabled for an empty prompt as well as for a model
        # that is not ready.
        if jq -e --arg t "$text" \
               '[.data.ui_elements[]?] as $e
                | (($e[] | select(.identifier == "rapid.images.compose")
                          | select(has("value") and .value == $t)) != null)
                  and (($e[] | select(.identifier == "Images.Generate")
                          | select(.enabled == true)) != null)' \
               "$OUT/$prefix.json" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    die "the prompt never reached the composer binding (Images.Generate stayed disabled): $text"
}

flow_image_generation() {
    # Text→image, the interactive half of the Images tab (#1705).
    #
    # The tab shipped with its identifiers but no journey, so nothing walked
    # it: a prompt that never reaches the wire, a progress card that never
    # clears, a gallery that shows the first render twice — all of them look
    # exactly like success in a tree dump. Each assertion below therefore pairs
    # the UI's story with the fake's recorded requests.
    #
    # No diffusion weights: the fake answers /v1/images/* with a real 1x1 PNG
    # whose bytes differ per render, after a scripted number of steps so the
    # in-flight card is observable rather than a frame between two polls.
    # RAPID_GUI_GOLDEN_MODE=1 + RAPID_SIMULATED_IMPORT_PATH together activate
    # the app's import test seam: when both are set, Images.Edit.Import imports
    # exactly this file through the same post-pick path a real picker would (see
    # ImagesView.chooseEditImage) instead of opening a native NSOpenPanel, whose
    # file browser publishes no AX identifiers and cannot be driven by injected
    # key events on an unattended CI runner. The golden-mode gate means a real
    # user's launch — which never sets it — always gets the picker even if an
    # unrelated process leaked an import path into the environment.
    # AX baseline normalization itself takes several seconds on a busy mini;
    # keep the synthetic decode tail long enough to observe after it.
    start_persona image-generation FAKE_IMAGE_STEPS=8 FAKE_IMAGE_STEP_MS=300 \
        FAKE_IMAGE_FINISH_MS=15000 \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_SIMULATED_IMPORT_PATH="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-96.png"

    dismiss_first_run

    # 1. The tab and its empty state.
    see_main "$OUT/ig-chat.json"
    press "$OUT/ig-chat.json" Sidebar.Images "$OUT/ig-open.json" \
        || die "Sidebar.Images is not pressable — the Images tab is unreachable"
    wait_identifier Images.EmptyState "$OUT/ig-empty.json"

    # 2. The picker resolved to the image model on its own.
    #    ``ImageGenViewModel.resolveAlias`` prefers a CACHED image entry and the
    #    fake marks exactly one, so this is deterministic without opening the
    #    menu. AXHelp carries "Model: <alias>" — the picker's own account of
    #    what the next render will use.
    #    Polled, not read once: ``refreshCatalog`` shells out to
    #    ``rapid-mlx models`` and ``ls`` from a `.task`, so the tab renders
    #    with an unresolved picker ("Choose a model") for as long as those two
    #    subprocesses take. A single read here fails on the app being
    #    correctly asynchronous.
    local i resolved=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/ig-empty.json"
        if jq -e --arg alias "$FAKE_IMAGE_ALIAS" \
               '.data.ui_elements[]? | select(.identifier == "Images.ModelPicker")
                | select((.help // "") | contains($alias))' "$OUT/ig-empty.json" >/dev/null; then
            resolved=1; break
        fi
        sleep 0.25
    done
    [[ "$resolved" == 1 ]] \
        || die "Images.ModelPicker never resolved to $FAKE_IMAGE_ALIAS — the tab has no model to render with"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Aspect")' "$OUT/ig-empty.json" >/dev/null \
        || die "Images.Aspect is missing — no way to choose an aspect ratio"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Resolution")' "$OUT/ig-empty.json" >/dev/null \
        || die "Images.Resolution is missing — no way to choose an output resolution"
    baseline image-generation.empty "$OUT/ig-empty.json"

    # 3. Load the model. rapid-mlx serves one model per process, so opening the
    #    tab cannot silently inherit a ready server: the readiness gate holds
    #    Generate shut until the image model is actually up.
    wait_identifier Readiness.Action "$OUT/ig-readiness.json"
    press "$OUT/ig-readiness.json" Readiness.Action "$OUT/ig-start.json" \
        || die "Readiness.Action is not pressable — the tab offers no way to load its model"
    # Match the ALIAS, not merely "a server started": the app may already have
    # started one on the chat alias at launch, and that event would satisfy a
    # bare grep while the image model never loaded at all.
    wait_fake_event \
        ".event == \"server_started\" and .alias == \"$FAKE_IMAGE_ALIAS\"" \
        "the image model never started — Readiness.Action did not switch the server"

    # ``help`` distinguishes "not ready" from "ready with an empty prompt":
    # the button is disabled in both, and only the hint separates them.
    local i ready=0
    for ((i=0; i<200; i++)); do
        see_main "$OUT/ig-ready.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "Images.Generate")
                  | select((.help // "") == "Generate")' "$OUT/ig-ready.json" >/dev/null; then
            ready=1; break
        fi
        sleep 0.25
    done
    [[ "$ready" == 1 ]] || die "Images.Generate never became ready after the model loaded"

    # 4. Generate.
    local prompt1="a cheetah on a red couch"
    local prompt2="the same cheetah, at night"
    type_prompt "$prompt1" ig-draft
    press "$OUT/ig-draft.json" Images.Generate "$OUT/ig-generate.json" \
        || die "Images.Generate is not pressable with a prompt and a ready model"

    # The in-flight card. Asserted BEFORE the result so a render that returns
    # instantly (or a card that never appears) is a failure rather than a frame
    # nobody looked at.
    local inflight=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/ig-inflight.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "Images.Cancel")' \
               "$OUT/ig-inflight.json" >/dev/null; then
            inflight=1; break
        fi
        sleep 0.1
    done
    [[ "$inflight" == 1 ]] \
        || die "no in-flight progress card: Images.Cancel never appeared during a render"
    baseline image-generation.inflight "$OUT/ig-inflight.json"

    # Sampling completion is followed by VAE decode / encoding. That tail must
    # be a named indeterminate phase, not a full 8/8 bar that appears stuck.
    local finalizing=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/ig-finalizing.json"
        if jq -e '.data.ui_elements[]?
                  | select((.value // .label // "") == "Finalizing image…")' \
               "$OUT/ig-finalizing.json" >/dev/null; then
            finalizing=1; break
        fi
        sleep 0.1
    done
    [[ "$finalizing" == 1 ]] \
        || die "the post-denoise tail never showed Finalizing image…"
    baseline image-generation.finalizing "$OUT/ig-finalizing.json"

    wait_fake_event '.event == "image_request"' \
        "no image_request reached the sidecar — the prompt was never sent"
    wait_fake_event '.event == "image_response" and .cancelled == false' \
        "the render never completed"

    # The result, and the card that has to go away again.
    local settled=0
    for ((i=0; i<200; i++)); do
        see_main "$OUT/ig-result.json"
        # `index(...) == null` is a claim of ABSENCE, and a walk that fell
        # short of a full inventory satisfies it by never having looked. The
        # driver already says whether it can vouch for `ui_elements`; require
        # that before reading a missing node as a cleared one.
        if jq -e '.success == true and .data.walk.complete == true
                  and ([.data.ui_elements[]? | .identifier // ""] as $ids
                       | ($ids | index("Images.Gallery.Thumb.1")) != null
                         and ($ids | index("Images.EmptyState")) == null
                         and ($ids | index("Images.Cancel")) == null)' \
               "$OUT/ig-result.json" >/dev/null; then
            settled=1; break
        fi
        sleep 0.25
    done
    [[ "$settled" == 1 ]] \
        || die "after the render the gallery had no thumbnail, or the empty state / progress card never cleared"
    log "  first render landed"

    # 5. Refine by re-prompting — a SECOND render, not a redraw of the first.
    type_prompt "$prompt2" ig-draft-2
    press "$OUT/ig-draft-2.json" Images.Generate "$OUT/ig-generate-2.json" \
        || die "Images.Generate is not pressable for a second render"

    local second=0
    for ((i=0; i<240; i++)); do
        see_main "$OUT/ig-result-2.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and ([.data.ui_elements[]? | .identifier // ""] as $ids
                       | ($ids | index("Images.Gallery.Thumb.2")) != null
                         and ($ids | index("Images.Cancel")) == null)' \
               "$OUT/ig-result-2.json" >/dev/null; then
            second=1; break
        fi
        sleep 0.25
    done
    [[ "$second" == 1 ]] || die "re-prompting produced no second thumbnail"
    # Exactly two requests on the wire. A UI that re-submits (a double press,
    # a silent retry) sends a third whose prompt duplicates an earlier one, so
    # the TOTAL count is the thing that catches it — the refine step is one
    # render, not one-plus-a-resend.
    jq -s '[.[] | select(.event == "image_request")] | length' \
        "$OUT/fake-events.jsonl" > "$OUT/ig-request-count.txt"
    [[ "$(cat "$OUT/ig-request-count.txt")" == "2" ]] \
        || die "the sidecar saw $(cat "$OUT/ig-request-count.txt") image requests, expected exactly 2 — a render was dropped or re-sent"
    # The ORDERED prompts, compared to what was actually typed — not a unique
    # count. `unique | length == 2` is satisfied by two prompts that are
    # merely different from each other: truncated, transformed, or swapped
    # text all pass it, and so does a first request that never carried the
    # user's words at all.
    jq -s -c '[.[] | select(.event == "image_request") | .prompt]' \
        "$OUT/fake-events.jsonl" > "$OUT/ig-prompts.json"
    local expected_prompts
    expected_prompts="$(jq -n -c --arg a "$prompt1" --arg b "$prompt2" '[$a,$b]')"
    [[ "$(cat "$OUT/ig-prompts.json")" == "$expected_prompts" ]] \
        || die "the prompts on the wire were $(cat "$OUT/ig-prompts.json"), expected $expected_prompts"
    # And the rest of the payload. Without this the picker can show and load
    # the image alias while the request names something else entirely — the
    # tab would look right and render with the wrong model.
    # `1024x1024` is what the default (square) aspect maps to. A shape-only
    # check like `^[0-9]+x[0-9]+$` passes `768x1024` — the PORTRAIT size — so
    # an aspect control wired to the wrong case would render the wrong shape
    # and the flow would call it correct.
    jq -s -e --arg alias "$FAKE_IMAGE_ALIAS" \
        'all(.[] | select(.event == "image_request");
             .model == $alias and .n == 1 and .size == "1024x1024")' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "an image request named the wrong model, asked for n != 1, or did not carry the square size 1024x1024: $(jq -s -c '[.[] | select(.event == "image_request") | {model, size, n}]' "$OUT/fake-events.jsonl")"
    # ...and the UI agreed that square was selected, so the two cannot drift
    # into agreeing on a wrong value together.
    # On `selected`, not on existence. All three ratio buttons carry the
    # identifier `Images.Aspect` and all three are always present, so
    # "there is a 1:1 button" is true no matter which one is active — an
    # assertion that cannot fail. Only the selected flag distinguishes them.
    jq -e '[.data.ui_elements[]? | select(.identifier == "Images.Aspect")
            | select(.selected == true)
            | (.description // .title // "")] == ["1:1"]' \
        "$OUT/ig-result-2.json" >/dev/null \
        || die "the selected aspect is not the square one the requests were sent with: $(jq -c '[.data.ui_elements[]? | select(.identifier == "Images.Aspect") | {d: (.description // .title), selected}]' "$OUT/ig-result-2.json")"

    # Two thumbnails is not two renders — and the two ways that can go wrong
    # need two different witnesses.
    #
    # This one is the DISPLAY side: the gallery can list a second entry and
    # bind the first result to it. Nothing above notices, because AX carries
    # no pixel data and both cases dump identically. Each thumb's label is
    # bound to its OWN prompt, so a duplicated record announces the duplicated
    # prompt. Pins the newest-first order at the same time. The render-index
    # check below is the other half — it covers the SIDECAR producing one
    # bitmap twice, which this cannot see and which cannot see this.
    local thumb1 thumb2
    thumb1="$(jq -r '.data.ui_elements[]? | select(.identifier == "Images.Gallery.Thumb.1")
                     | (.description // .title // "")' "$OUT/ig-result-2.json" | head -1)"
    thumb2="$(jq -r '.data.ui_elements[]? | select(.identifier == "Images.Gallery.Thumb.2")
                     | (.description // .title // "")' "$OUT/ig-result-2.json" | head -1)"
    [[ "$thumb1" == *"$prompt2"* ]] \
        || die "the newest thumbnail does not name the second render's prompt (got: $thumb1)"
    [[ "$thumb2" == *"$prompt1"* ]] \
        || die "the older thumbnail does not name the first render's prompt (got: $thumb2)"
    [[ "$thumb1" != "$thumb2" ]] \
        || die "both thumbnails describe the same render — the refine step redrew the first instead of adding a second"
    # ...and two DISTINCT bitmaps came back, not one artifact returned twice.
    # Compared by SHA-256 of the bytes actually sent, NOT by the response
    # index: an index is a counter, and a fixture or engine that returned one
    # image twice while still incrementing would satisfy it. The hash is the
    # only field here that is a statement about content.
    #
    # What this flow does NOT prove, stated plainly so nobody reads it as
    # more: nothing above compares the pixels the app DRAWS. AX exposes no
    # image data, so a dump cannot distinguish two thumbnails showing one
    # bitmap from two showing two. The pair of checks brackets it — the wire
    # carried two different images, and the gallery bound two different
    # records — and `filmstripThumb` takes its picture and its label from the
    # same value, so they cannot disagree without an edit that deliberately
    # reaches past its own parameter. Closing the last gap needs pixel
    # capture, which this flow is deliberately without (#1708 removed screen
    # capture from the semantic flows); that belongs with the XCUITest work
    # in #1719.
    jq -s '[.[] | select(.event == "image_response") | .sha256] | unique | length' \
        "$OUT/fake-events.jsonl" > "$OUT/ig-render-count.txt"
    [[ "$(cat "$OUT/ig-render-count.txt")" == "2" ]] \
        || die "the sidecar returned identical PNG bytes for both renders — one image produced twice: $(jq -s -c '[.[] | select(.event == "image_response") | {index, sha256}]' "$OUT/fake-events.jsonl")"

    # A thumbnail is a way back to its prompt, not just a picture.
    press "$OUT/ig-result-2.json" Images.Gallery.Thumb.2 "$OUT/ig-revisit.json" \
        || die "Images.Gallery.Thumb.2 is not pressable — the filmstrip is decorative"
    local revisited=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ig-revisited.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "rapid.images.compose")
                  | select(has("value") and (.value | test("red couch")))' \
               "$OUT/ig-revisited.json" >/dev/null; then
            revisited=1; break
        fi
        sleep 0.25
    done
    [[ "$revisited" == 1 ]] \
        || die "selecting the older thumbnail did not restore its prompt"

    # 6. Edit that generated result. This is the actual GUI contract added by
    # the feature: action -> edit mode -> multipart request -> returned image
    # becomes the next source -> exit restores generation controls.
    press "$OUT/ig-revisited.json" Images.Result.Edit "$OUT/ig-edit-open.json" \
        || die "the generated result has no pressable Edit action"
    wait_identifier Images.Edit.Source "$OUT/ig-edit-source.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Edit.Exit")' \
        "$OUT/ig-edit-source.json" >/dev/null \
        || die "edit mode has no way to exit"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Edit.Import")' \
        "$OUT/ig-edit-source.json" >/dev/null \
        || die "edit mode has no way to replace its source"

    local edit_prompt="replace the couch with a blue armchair"
    type_prompt "$edit_prompt" ig-edit-draft
    press "$OUT/ig-edit-draft.json" Images.Generate "$OUT/ig-edit-submit.json" \
        || die "Images.Generate is not pressable with an edit instruction"
    wait_fake_event '.event == "image_request" and .operation == "edit"' \
        "no multipart image edit request reached the sidecar"
    wait_fake_event '.event == "image_response" and .cancelled == false and .index == 3' \
        "the image edit never completed"

    local edited=0
    for ((i=0; i<200; i++)); do
        see_main "$OUT/ig-edit-result.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and ([.data.ui_elements[]? | .identifier // ""] as $ids
                       | ($ids | index("Images.Gallery.Thumb.3")) != null
                         and ($ids | index("Images.Edit.Source")) != null
                         and ($ids | index("Images.Cancel")) == null)' \
               "$OUT/ig-edit-result.json" >/dev/null; then
            edited=1; break
        fi
        sleep 0.25
    done
    [[ "$edited" == 1 ]] \
        || die "the edited result did not land as a third thumbnail and remain the edit source"
    jq -s -e --arg alias "$FAKE_IMAGE_ALIAS" --arg prompt "$edit_prompt" \
        '[.[] | select(.event == "image_request" and .operation == "edit")
              | {prompt, model, size, n, operation, has_image}] ==
         [{prompt:$prompt, model:$alias, size:null, n:1,
           operation:"edit", has_image:true}]' "$OUT/fake-events.jsonl" >/dev/null \
        || die "the edit request did not carry the exact prompt, model, and source image: $(jq -s -c '[.[] | select(.event == "image_request" and .operation == "edit")]' "$OUT/fake-events.jsonl")"

    # Sequential-edit invariant: the source strip now names the EDIT result's
    # instruction, not the original generation prompt.
    assert_tree_text "$OUT/ig-edit-result.json" "$edit_prompt"
    press "$OUT/ig-edit-result.json" Images.Edit.Exit "$OUT/ig-edit-exit.json" \
        || die "Images.Edit.Exit is not pressable after an edit"
    local exited=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ig-edit-exited.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and ([.data.ui_elements[]? | .identifier // ""] as $ids
                       | ($ids | index("Images.Edit.Source")) == null
                         and ($ids | index("Images.Aspect")) != null)' \
               "$OUT/ig-edit-exited.json" >/dev/null; then
            exited=1; break
        fi
        sleep 0.25
    done
    [[ "$exited" == 1 ]] \
        || die "exiting edit mode did not restore generation controls"

    baseline image-generation.generated "$OUT/ig-edit-exited.json"

    # 7. Import from disk — the SECOND door into /v1/images/edits, and the one
    #    the journey above never opens.
    #
    #    The generated-result edit walks in through Images.Result.Edit. This
    #    section drives the other entry: Images.Edit.Import -> an edit keyed to
    #    the imported file's own name. It exists because "import an image then
    #    edit it" is a distinct user contract: nothing below can pass unless the
    #    app really turns the picked file into an editable source (edit mode,
    #    "Replace source image" affordance, the file name on the source bar, and
    #    the fixture's bytes on the wire).
    #
    #    The app's own AX tree cannot reach a native NSOpenPanel — it publishes
    #    no kAXIdentifierAttribute — and injected key events cannot drive its
    #    file browser on an unattended CI runner (see the RAPID_SIMULATED_IMPORT
    #    note in start_persona). So the harness has told the app, via that seam,
    #    exactly which file Images.Edit.Import should pick. The press below goes
    #    through the app-level post-pick path for real, and every user-visible
    #    contract is still asserted here: edit mode, the replace-source
    #    affordance, the file name, and the fixture's bytes on the wire. The old
    #    filename is static; assert it landed.
    local fixture="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-96.png"
    [[ -f "$fixture" ]] || die "import fixture not found: $fixture"
    local file_basename
    file_basename="$(basename "$fixture" .png)"
    # The seam path never opens a modal, so the press completes normally; keep
    # the CannotComplete tolerance anyway (like a real pick, the composer can be
    # momentarily busy) and let the edit-mode wait below be the judge: if the
    # imported source never appears the import button is genuinely broken.
    press "$OUT/ig-edit-exited.json" Images.Edit.Import "$OUT/ig-import-press.json" \
        2>/dev/null || true

    # Entering edit mode from an import must be observably different from the
    # generated-result entry: the source is keyed to the FILE NAME, the import
    # affordance flips to "Replace source image", and the edit source bar
    # appears.
    local imported=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/ig-import-entered.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and (([.data.ui_elements[]? | .identifier // ""] | index("Images.Edit.Source")) != null)
                  and (([.data.ui_elements[]? | .identifier // ""] | index("Images.Edit.Import")) != null)
                  and ([.data.ui_elements[]? | select(.identifier == "Images.Edit.Import")
                        | (.help // .description // "")] | any(. == "Replace source image"))' \
               "$OUT/ig-import-entered.json" >/dev/null; then
            imported=1; break
        fi
        sleep 0.25
    done
    [[ "$imported" == 1 ]] \
        || die "importing the fixture did not enter edit mode with a replace-source affordance"
    assert_tree_text "$OUT/ig-import-entered.json" "$file_basename" \
        || die "the imported source does not carry the file name ($file_basename) on the edit source bar"

    # 8. Edit the imported image — the rest of the contract after a real
    #    import: type an instruction, generate, and the multipart edit request
    #    must carry the fixture bytes.
    local import_prompt="give the logo a blue background"
    type_prompt "$import_prompt" ig-import-draft
    press "$OUT/ig-import-draft.json" Images.Generate "$OUT/ig-import-submit.json" \
        || die "Images.Generate is not pressable after importing an image"
    wait_fake_event '.event == "image_request" and .operation == "edit" and .has_image == true' \
        "no multipart edit request carrying an image reached the sidecar after import"
    wait_fake_event '.event == "image_response" and .cancelled == false and .index == 4' \
        "the imported edit never completed"
    local import_done=0
    for ((i=0; i<200; i++)); do
        see_main "$OUT/ig-import-result.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and (([.data.ui_elements[]? | .identifier // ""] | index("Images.Gallery.Thumb.4")) != null)
                  and (([.data.ui_elements[]? | .identifier // ""] | index("Images.Edit.Source")) != null)
                  and (([.data.ui_elements[]? | .identifier // ""] | index("Images.Cancel")) == null)' \
               "$OUT/ig-import-result.json" >/dev/null; then
            import_done=1; break
        fi
        sleep 0.25
    done
    [[ "$import_done" == 1 ]] \
        || die "the imported edit did not land as a new thumbnail and remain the edit source"
    jq -s -e --arg alias "$FAKE_IMAGE_ALIAS" --arg prompt "$import_prompt" \
        '[.[] | select(.event == "image_request" and .operation == "edit" and .prompt == $prompt)
              | {model, n, operation, has_image}] ==
         [{model:$alias, n:1, operation:"edit", has_image:true}]' "$OUT/fake-events.jsonl" >/dev/null \
        || die "the imported edit request did not carry the exact prompt, model, and the fixture image: $(jq -s -c '[.[] | select(.event == "image_request" and .operation == "edit")]' "$OUT/fake-events.jsonl")"
    # The uploaded image must be the picked fixture. has_image only proves a
    # multipart part named "image" existed; a regression that submits the
    # previously generated image — or any other arbitrary PNG — would still
    # pass it. But the fixture cannot be compared by raw bytes: the app's
    # EditImageImporter decodes and re-encodes every import, so ancillary
    # chunks (iCCP, eXIf ...) and the IDAT stream can legitimately differ
    # across macOS encoder versions. Compare the DECODED RGBA pixel hash
    # instead, which is the user contract that matters. The fake's
    # png-rgba-sha subcommand runs the exact same decoder the request fake
    # uses, so expectation and upload can never drift.
    local expected_sha
    expected_sha="$("$ROOT/scripts/fake-rapid-mlx.sh" png-rgba-sha "$fixture")"
    [[ -n "$expected_sha" ]] \
        || die "could not compute the fixture's pixel hash: $fixture"
    jq -s -e --arg sha "$expected_sha" --arg prompt "$import_prompt" \
        '[.[] | select(.event == "image_request" and .operation == "edit" and .prompt == $prompt)
              | .image_rgba_sha256] == [$sha]' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "the uploaded image pixels do not match the fixture ($fixture, rgba sha256 $expected_sha)"

    # Exit restores generation controls — the same exit contract as the
    # generated-result journey, now after an import.
    press "$OUT/ig-import-result.json" Images.Edit.Exit "$OUT/ig-import-exit.json" \
        || die "Images.Edit.Exit is not pressable after an imported edit"
    local import_exited=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ig-import-exited.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and ([.data.ui_elements[]? | .identifier // ""] as $ids
                       | ($ids | index("Images.Edit.Source")) == null
                         and ($ids | index("Images.Aspect")) != null)' \
               "$OUT/ig-import-exited.json" >/dev/null; then
            import_exited=1; break
        fi
        sleep 0.25
    done
    [[ "$import_exited" == 1 ]] \
        || die "exiting edit mode after an import did not restore generation controls"

    log "  image-generation OK"
}
flow_resident_load_rejected() {
    start_persona resident-load-rejected FAKE_REJECT_IMAGE_LOAD=1 \
        FAKE_RESIDENT_LOAD_DELAY_MS=1500

    dismiss_first_run

    # 1. Bring the CHAT model up first so the sidecar is running and the served
    #    alias is resident - the precondition for the in-process load.
    wait_identifier Readiness.Action "$OUT/rlr-chat-readiness.json" \
        || die "no chat readiness action to bring the resident chat model up"
    press "$OUT/rlr-chat-readiness.json" Readiness.Action "$OUT/rlr-chat-start.json" \
        || die "chat Readiness.Action is not pressable - could not start the resident chat model"
    # Match the ALIAS, not merely "a server started": the fake must be serving
    # the chat alias so the residency snapshot reports it resident.
    wait_fake_event \
        ".event == \"server_started\" and .alias == \"$FAKE_ALIAS\"" \
        "the chat model never started - no resident sidecar to reject against"
    # The chat sidecar must actually reach .ready (with a child) in the app's
    # state machine BEFORE the Images load: ``ensureServing`` only takes the
    # in-process ``/v1/models/load`` path when ``readyWithChild`` is true, i.e.
    # when the chat model is already residing in this process. Bare
    # ``server_started`` only proves the fake bound its port; if we press
    # Images readiness while the chat model is still ``.starting``, the app
    # falls back to replacing the child process (a cold start) and the
    # rejection never reaches the wire (#1838). ``wait_send_idle`` blocks
    # until the ChatView readiness gate opens, which is exactly the app's
    # story that ``state == .ready``.
    wait_send_idle "$OUT/rlr-chat-ready.json"

    # 2. Go to Images and ask it to load its model.
    see_main "$OUT/rlr-ig-chat.json"
    press "$OUT/rlr-ig-chat.json" Sidebar.Images "$OUT/rlr-ig-open.json" \
        || die "Sidebar.Images is not pressable - the Images tab is unreachable"
    wait_identifier Images.EmptyState "$OUT/rlr-ig-empty.json"

    # 2.5 The picker must resolve to the image model BEFORE we press the
    #     readiness action. ``refreshCatalog`` shells out to ``rapid-mlx
    #     models`` and ``ls`` from a `.task`, so the tab renders with an
    #     unresolved picker ("Choose a model") for as long as those two
    #     subprocesses take; while unresolved, ``selectedAlias`` is empty and
    #     readiness resolves to ``.noModel``, whose action is ``.chooseModel``
    #     and renders NO ``Readiness.Action`` -- or, mid-window, a button whose
    #     load does not name the image model. Pressing too early therefore
    #     falls out of the resident ``/v1/models/load`` path entirely and the
    #     rejection never reaches the wire. ``image-generation`` does the same
    #     wait for the same reason; mirror it so this flow's press is
    #     deterministic (#1838).
    local i resolved=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/rlr-ig-empty.json"
        if jq -e --arg alias "$FAKE_IMAGE_ALIAS" \
               '.data.ui_elements[]? | select(.identifier == "Images.ModelPicker")
                | select((.help // "") | contains($alias))' "$OUT/rlr-ig-empty.json" >/dev/null; then
            resolved=1; break
        fi
        sleep 0.25
    done
    [[ "$resolved" == 1 ]] \
        || die "Images.ModelPicker never resolved to $FAKE_IMAGE_ALIAS - the Images tab has no model to load"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Aspect")' "$OUT/rlr-ig-empty.json" >/dev/null \
        || die "Images.Aspect is missing - the picker did not finish resolving"

    # 3. The readiness action routes through ensureServing and hits the
    #    in-process /v1/models/load endpoint, not a process restart.
    wait_identifier Readiness.Action "$OUT/rlr-ig-readiness.json" \
        || die "Images readiness has no action to load its model"
    press "$OUT/rlr-ig-readiness.json" Readiness.Action "$OUT/rlr-ig-start.json" \
        || die "Images Readiness.Action is not pressable - the load button is dead"

    # 4. The wire must show the load was ATTEMPTED in-process and REJECTED.
    wait_fake_event '.event == "model_load"' \
        "the Images action never issued an in-process /v1/models/load"

    # The request is deliberately held open: the tap must immediately replace
    # the CTA with a working state. Before this regression fix HF was already
    # writing the checkpoint while the UI still said "isn't downloaded yet"
    # and kept showing a pressable Download & start button.
    see_main "$OUT/rlr-in-flight.json"
    jq -e '[.data.ui_elements[]? | .value? | strings] | any(contains("Downloading or loading the image model"))' \
        "$OUT/rlr-in-flight.json" >/dev/null \
        || die "resident image download started without visible working feedback"
    if jq -e '.data.ui_elements[]? | select(.identifier == "Readiness.Action")' \
        "$OUT/rlr-in-flight.json" >/dev/null; then
        die "Download & start remained pressable while its resident load was in flight"
    fi
    wait_fake_event '.event == "model_load_rejected"' \
        "the fake did not reject the in-process image load (FAKE_REJECT_IMAGE_LOAD not applied)"

    # 5. The rejection's actionable reason must be VISIBLE on the Images
    #    surface (the readiness banner), not only in the log drawer.
    local i shown=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/rlr-shown.json"
        if jq -e '[.data.ui_elements[]?]
                  | map(((.title // "") | tostring) + " " + ((.value // "") | tostring) + " " + ((.description // "") | tostring) + " " + ((.help // "") | tostring))
                  | join(" ") | test("rapid-mlx\\[image\\]")' \
               "$OUT/rlr-shown.json" >/dev/null 2>&1; then
            shown=1; break
        fi
        sleep 0.25
    done
    [[ "$shown" == 1 ]] \
        || die "the engine's rejection reason never appeared on the Images surface - it was swallowed into the log (#1838)"

    # 6. The surface offers a recovery action, not a dead button.
    jq -e '.data.ui_elements[]? | select(.identifier == "Readiness.Action")' \
        "$OUT/rlr-shown.json" >/dev/null \
        || die "the Images readiness banner did not offer a recovery action after the rejection"

    log "  resident-load-rejected OK"
}

flow_launch_integrations() {
    log "flow: launch-integrations"
    start_persona launch-integrations
    dismiss_first_run
    see_main "$OUT/main.json"
    press "$OUT/main.json" Sidebar.Launch "$OUT/launch.json"

    # The engine-owned registry currently resolves to fourteen distinct
    # products after the overlapping Claude Code and Continue entries are
    # merged. Count the actual per-row action, not a container: putting the id
    # on the row propagates it to the Copy button in SwiftUI and makes the
    # button look addressable while preventing it from having its own stable
    # identity.
    for _ in {1..40}; do
        see_main "$OUT/launch.json"
        count="$(jq '[.data.ui_elements[]? | (.identifier // "") | select(startswith("Launch.Integration.Copy."))] | unique | length' "$OUT/launch.json")"
        [[ "$count" == 14 ]] && break
        sleep 0.25
    done
    [[ "$count" == 14 ]] || die "Launch rendered $count integrations; engine registry exposes 14 (#1715)"
    jq -e '.data.ui_elements[]? | select(.identifier == "Launch.Integration.Copy.cline")' "$OUT/launch.json" >/dev/null \
        || die "Launch omitted config-writing target Cline"
    jq -e '.data.ui_elements[]? | select(.identifier == "Launch.Integration.Copy.smolagents")' "$OUT/launch.json" >/dev/null \
        || die "Launch omitted adapter profile smolagents"
    # The two one-session launch commands are the useful fast path, not an
    # implementation-detail registry order. Keep them first and in the product
    # order promised by the Launch page.
    local first_two
    first_two="$(jq -r '[.data.ui_elements[]?
                         | select((.identifier // "")
                                  | startswith("Launch.Integration.Copy."))
                         | .identifier]
                        | .[0:2]
                        | join(",")' "$OUT/launch.json")"
    [[ "$first_two" == "Launch.Integration.Copy.claude-code,Launch.Integration.Copy.codex" ]] \
        || die "Launch did not lead with Claude Code then Codex (got: $first_two)"
    # The card itself is not the action. Every visible row must publish a
    # distinct Copy button so AX/keyboard users can invoke the same command a
    # pointer user can, and every one is disabled honestly until a live model
    # has minted a usable endpoint/key.
    local copy_count enabled_copy_count
    copy_count="$(jq '[.data.ui_elements[]?
                       | (.identifier // "")
                       | select(startswith("Launch.Integration.Copy."))]
                      | unique | length' "$OUT/launch.json")"
    [[ "$copy_count" == 14 ]] \
        || die "Launch rendered $copy_count addressable Copy buttons for 14 integrations"
    enabled_copy_count="$(jq '[.data.ui_elements[]?
                               | select(((.identifier // "") | startswith("Launch.Integration.Copy."))
                                        and .enabled == true)] | length' "$OUT/launch.json")"
    [[ "$enabled_copy_count" == 0 ]] \
        || die "Launch enabled $enabled_copy_count copy commands before a model was ready"
    baseline launch-integrations.complete "$OUT/launch.json"
    log "  launch-integrations OK"
    cleanup_persona
}

flow_audio_readiness() {
    log "flow: audio-readiness"
    # Keep `pull` alive long enough to prove Audio owns a real download job.
    # The audio server reports /healthz before its lazy engine has weights, so
    # a UI-only Ready assertion would miss the regression this flow guards.
    start_persona audio-readiness \
        FAKE_DOWNLOAD_OVERRUN=1 \
        FAKE_PARTIAL_AUDIO_CACHE=1 \
        FAKE_AUDIO_PULL_STATE="$OUT_ROOT/audio-readiness/pulled-audio.txt"
    dismiss_first_run
    see_main "$OUT/chat.json"
    press "$OUT/chat.json" Sidebar.Audio "$OUT/speech.json" \
        || die "Sidebar.Audio is not pressable"

    local i speech_ready=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/speech.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Audio.Speech.ModelPicker")' "$OUT/speech.json" >/dev/null \
           && jq -e '.data.ui_elements[]?
                     | select(.identifier == "Readiness.Action"
                              and (.description // .value // .label // "") == "Download & start")' \
                    "$OUT/speech.json" >/dev/null; then
            speech_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_ready" == 1 ]] \
        || die "Speech did not expose Chat-equivalent Download & start readiness"
    baseline audio-readiness.speech "$OUT/speech.json"

    press "$OUT/speech.json" Readiness.Action "$OUT/speech-download-start.json" \
        || die "Speech Download & start is not pressable"
    wait_fake_event \
        '.event == "command" and .subcommand == "pull" and .alias == "fake-qwen3-tts"' \
        "Speech Download & start did not invoke pull for fake-qwen3-tts"

    # Sample several fresh AX trees inside the fake's five-second pull window.
    # An early healthy audio sidecar must never turn that window into Ready.
    local speech_downloading=0
    for ((i=0; i<8; i++)); do
        see_main "$OUT/speech-downloading.json"
        if jq -e '.data.ui_elements[]?
                  | select(((.description // .value // .label // "") | tostring)
                           | startswith("Downloading fake-qwen3-tts"))' \
                 "$OUT/speech-downloading.json" >/dev/null; then
            speech_downloading=1
        fi
        if jq -e '(.data.ui_elements | tostring)
                  | contains("Ready — fake-qwen3-tts")' \
                 "$OUT/speech-downloading.json" >/dev/null; then
            die "Speech reported Ready while fake-qwen3-tts was still downloading"
        fi
        sleep 0.25
    done
    [[ "$speech_downloading" == 1 ]] \
        || die "Speech never exposed Downloading after Download & start"
    if jq -e -s 'any(.[]; .event == "server_started" and .alias == "fake-qwen3-tts")' \
        "$OUT/fake-events.jsonl" >/dev/null; then
        die "Speech started fake-qwen3-tts before its pull completed and cache was verified"
    fi

    wait_fake_event \
        '.event == "server_started" and .alias == "fake-qwen3-tts"' \
        "Speech did not start after its download completed"
    local speech_loaded=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/speech-loaded.json"
        if ! jq -e '.data.ui_elements[]?
                    | select(.identifier == "Readiness.Action")' \
                   "$OUT/speech-loaded.json" >/dev/null; then
            speech_loaded=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_loaded" == 1 ]] \
        || die "Speech stayed behind Download & start after its model became ready"

    # Residency is polled independently from Audio readiness. The sidecar can
    # be ready several frames before the sidebar's first residency snapshot;
    # switching tabs immediately made the transcription baseline alternate
    # between "no resident" and the correctly resident TTS model depending on
    # poll timing. Wait for the user-visible state that must follow readiness
    # before recording the next settled screen.
    local speech_resident=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/speech-resident.json"
        if jq -e '.data.ui_elements as $elements
                  | any(range(1; $elements | length);
                        $elements[.].identifier == "Sidebar.Residency"
                        and $elements[.].value == "fake-qwen3-tts"
                        and $elements[. - 1].identifier == "Sidebar.Residency"
                        and $elements[. - 1].description == "Lock")' \
                 "$OUT/speech-resident.json" >/dev/null; then
            speech_resident=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_resident" == 1 ]] \
        || die "ready fake-qwen3-tts never appeared as the locked resident model"

    press "$OUT/speech-resident.json" Audio.Mode.Transcription "$OUT/transcription.json" \
        || die "Audio transcription segment is not pressable"
    local transcription_ready=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/transcription.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Audio.Transcription.ModelPicker")' \
                 "$OUT/transcription.json" >/dev/null \
           && jq -e '.data.ui_elements[]?
                     | select(.identifier == "Readiness.Action"
                              and (.description // .value // .label // "") == "Download & start")' \
                    "$OUT/transcription.json" >/dev/null; then
            transcription_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$transcription_ready" == 1 ]] \
        || die "Transcription did not expose Chat-equivalent Download & start readiness"
    baseline audio-readiness.transcription "$OUT/transcription.json"

    press "$OUT/transcription.json" Readiness.Action "$OUT/transcription-start.json" \
        || die "Transcription Download & start is not pressable"
    wait_fake_event \
        '.event == "command" and .subcommand == "pull" and .alias == "fake-whisper-small"' \
        "Transcription Download & start did not invoke pull for fake-whisper-small"
    local transcription_downloading=0
    for ((i=0; i<8; i++)); do
        see_main "$OUT/transcription-downloading.json"
        if jq -e '.data.ui_elements[]?
                  | select(((.description // .value // .label // "") | tostring)
                           | startswith("Downloading fake-whisper-small"))' \
                 "$OUT/transcription-downloading.json" >/dev/null; then
            transcription_downloading=1
        fi
        if jq -e -s 'any(.[]; .event == "server_started" and .alias == "fake-whisper-small")' \
            "$OUT/fake-events.jsonl" >/dev/null; then
            die "Transcription started fake-whisper-small before its pull completed"
        fi
        sleep 0.25
    done
    [[ "$transcription_downloading" == 1 ]] \
        || die "Transcription never exposed Downloading after Download & start"
    wait_fake_event \
        '.event == "server_started" and .alias == "fake-whisper-small"' \
        "Transcription Download & start did not switch to its selected model"
    local transcription_loaded=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/transcription-loaded.json"
        if ! jq -e '.data.ui_elements[]?
                    | select(.identifier == "Readiness.Action")' \
                   "$OUT/transcription-loaded.json" >/dev/null; then
            transcription_loaded=1; break
        fi
        sleep 0.25
    done
    [[ "$transcription_loaded" == 1 ]] \
        || die "Transcription stayed behind Download & start after its model became ready"
    jq -n '{success: true,
            assertion: "Speech and Transcription each start the selected model and leave readiness"}' \
        > "$OUT/audio-readiness-actions.json"
    log "  audio-readiness OK"
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
    cached-quickstart) flow_cached_quickstart ;;
    download-progress) flow_download_progress ;;
    settings-persistence) flow_settings_persistence ;;
    chat-restore) flow_chat_restore ;;
    restored-tools) flow_restored_tools ;;
    tool-loop-budget) flow_tool_loop_budget ;;
    chat-depth) flow_chat_depth ;;
    math-rendering) flow_math_rendering ;;
    slow-stream-stop) flow_slow_stream_stop ;;
    model-crash-recovery) flow_model_crash_recovery ;;
    low-memory-choice) flow_low_memory_choice ;;
    update-state) flow_update_state ;;
    window-close-prompt) flow_window_close_prompt ;;
    no-dead-controls) flow_no_dead_controls ;;
    catalog-integrity) flow_catalog_integrity ;;
    browse-all-destination) flow_browse_all_destination ;;
    chat-document-attachment) flow_chat_document_attachment ;;
    image-generation) flow_image_generation ;;
    audio-readiness) flow_audio_readiness ;;
    resident-load-rejected) flow_resident_load_rejected ;;
    launch-integrations) flow_launch_integrations ;;
    all)
        flow_fresh_install
        flow_cached_quickstart
        flow_download_progress
        flow_settings_persistence
        flow_chat_restore
        flow_restored_tools
        flow_tool_loop_budget
        flow_chat_depth
        flow_math_rendering
        flow_slow_stream_stop
        flow_model_crash_recovery
        flow_low_memory_choice
        flow_update_state
        flow_window_close_prompt
        flow_no_dead_controls
        flow_catalog_integrity
        flow_browse_all_destination
        flow_chat_document_attachment
        flow_image_generation
        flow_audio_readiness
        flow_resident_load_rejected
        flow_launch_integrations
        ;;
    *) die "unknown flow: $FLOW" ;;
esac

jq -n --arg status pass --arg flow "$FLOW" --arg app "$APP_SOURCE" \
    '{status: $status, flow: $flow, app: $app}' > "$OUT_ROOT/result.json"
RESULT_WRITTEN=1
log "PASS — $FLOW"
log "artifacts: $OUT_ROOT"
