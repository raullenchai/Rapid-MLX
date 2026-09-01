#!/usr/bin/env bash
# Release-grade AX-first GUI journeys for Rapid-MLX Desktop.
#
# Each flow runs in a unique bundle-id + HOME, targets elements by semantic
# accessibility identifiers, and writes JSON/screenshot evidence. The bundled
# fake sidecar keeps the suite deterministic and prevents model-related OOMs.
set -euo pipefail

ORIGINAL_ARGS=("$@")

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
# Phase B made the first-run wizard's "RECOMMENDED FOR YOUR N GB MAC" row
# derive from the host's REAL physical RAM. A single committed golden
# baseline therefore can't be host-independent: a 14 GB CI runner lands on
# the 8 GB tier (lfm2.5-2.6B) while a 256 GB release Mac lands on 48+
# (qwen3.8-27B + qwen3.6-35B). The golden gate runs in BOTH places (CI and
# the operator's release Mac), so every persona that renders the chooser
# pins the same tier to keep its AX baseline deterministic.
# 8 = the 8 GB tier, which is exactly what the committed compact-chooser
# baseline captures (safe fast pick lfm2.5-1B as the starter; smart
# lfm2.5-2.6B as the optional memory-guarded capability upgrade). Pinning 8 also happens to
# be the safe tier for cached-curated-tradeup: its assertion needs
# qwen3.5-4b to stay a native curated trade-up, which 8 GB guarantees
# (16/24/32 GB hosts fold qwen3.5-4b into the recommended row instead).
GOLDEN_RAM_GB=8
GOLDEN_BRAND="Apple M1"
UPDATE_BASELINES=0
FLOW="all"
KEEP=0
APP_PID=""
OPERATOR_SERVER_PID=""
TELEMETRY_SINK_PID=""
TELEMETRY_SINK_PORT=""
TELEMETRY_SINK_LOG=""
PERSONA=""
OUT=""
MAIN_WINDOW_ID=""
BUNDLE_ID=""
AX_DRIVER=""
RESULT_WRITTEN=0
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STARTED_EPOCH="$(date +%s)"
PERSONA_ENV=()

usage() {
    cat <<'EOF'
Usage: gui-golden-flows.sh [--flow NAME] [--keep] [--update-baselines]

Flows: fresh-install, cached-quickstart, cached-curated-tradeup, cached-variant-collapse, download-progress, settings-persistence, settings-mtp, chat-restore, chat-depth, launch-integrations,
       model-crash-recovery, low-memory-choice,
       update-state, update-busy, campaign-banner, window-close-prompt, no-dead-controls, catalog-integrity,
       browse-all-destination, chat-document-attachment, chat-multimodal-attachments, image-generation, dictation, dictation-rc2-upgrade, audio-readiness, all

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

# CI may route a Desktop diff to a subset of the manifest's journey groups.
# Keep the selection check before preflight, app launch, output-directory
# creation, and cleanup registration so an unaffected named workflow step is a
# true no-op. Empty/malformed routing input fails closed by running the flow.
if [[ -n "${GUI_FLOWS:-}" && "$FLOW" != all ]]; then
    if selected="$(jq -r --arg flow "$FLOW" \
        'if type == "array" then any(.[]; . == $flow) else error("not an array") end' \
        <<<"$GUI_FLOWS" 2>/dev/null)"; then
        if [[ "$selected" != true ]]; then
            printf '[gui-golden] SKIP — %s is outside the selected risk groups\n' "$FLOW"
            exit 0
        fi
    else
        printf '[gui-golden] WARN: invalid GUI_FLOWS; running %s fail-closed\n' "$FLOW" >&2
    fi
fi

# The helper-contract tests source this file to exercise the shell guards. Host
# admission belongs to the executable entry point, not to library-style use.
if [[ "${BASH_SOURCE[0]}" == "$0" && "${RAPID_HOST_PRECHECK_HELD:-0}" != "1" && "${CI:-}" != "true" ]]; then
    export RAPID_HOST_PRECHECK_HELD=1
    # macOS Bash 3.2 treats an empty array expansion as unbound under `set -u`.
    exec "$ROOT/scripts/dogfood-host-precheck.sh" -- "$0" \
        ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}
fi

log() { printf '[gui-golden] %s\n' "$*"; }
die() { printf '[gui-golden] FAIL: %s\n' "$*" >&2; exit 1; }

# Small executable contracts shared by the real journeys and the fast Python
# harness tests. Keeping the failure inside these helpers means changing a
# journey guard from ``die`` to logging can no longer leave the unit contract
# green: the helper is exercised with both allowed and forbidden fixtures.
assert_fake_server_starts() {
    local events="$1" expected_count="$2" expected_alias="$3" phase="$4"
    local query_status=0 attempt
    # The app appends JSONL concurrently. A reader can briefly observe the
    # final record between writes; retry parse/read failures for a bounded
    # 400 ms, while a valid-but-wrong start set still fails immediately.
    # Persistently malformed or unreadable evidence remains fail-closed.
    for attempt in 1 2 3 4 5; do
        query_status=0
        jq -e -s --argjson count "$expected_count" --arg alias "$expected_alias" \
            '[.[] | select(.event == "server_started")] as $starts
             | (($starts | length) == $count
                and ($alias == "" or all($starts[]; .alias == $alias)))' \
            "$events" >/dev/null 2>&1 || query_status=$?
        case "$query_status" in
            0) return 0 ;;
            1) die "$phase observed an unexpected sidecar start set" ;;
        esac
        [[ "$attempt" == 5 ]] || sleep 0.1
    done
    die "$phase could not validate the sidecar event log"
}

# A recorded spawn is intentionally weaker than readiness: the fake writes
# ``server_started`` immediately before binding its HTTP server.  Keep the
# wire-side boundary explicit so a GUI assertion can distinguish a sidecar
# that never became reachable from a slower app state transition.
wait_fake_sidecar_health() {
    local expected_alias="$1" what="$2" attempts="${3:-40}"
    local sidecar_record sidecar_pid sidecar_port health_payload i
    sidecar_record="$(jq -rs --arg alias "$expected_alias" \
        'map(select(.event == "server_started" and .alias == $alias))
         | last | [(.pid // ""), (.port // "")] | @tsv' \
        "$OUT/fake-events.jsonl")"
    IFS=$'\t' read -r sidecar_pid sidecar_port <<< "$sidecar_record"
    [[ "$sidecar_pid" =~ ^[0-9]+$ && "$sidecar_port" =~ ^[0-9]+$ ]] \
        || die "$what did not record a valid sidecar pid and port"
    for ((i=0; i<attempts; i++)); do
        health_payload="$(curl -fsS --connect-timeout 1 --max-time 1 \
            "http://127.0.0.1:$sidecar_port/healthz" 2>/dev/null || true)"
        if jq -e --argjson pid "$sidecar_pid" --arg alias "$expected_alias" \
            '.ok == true and .pid == $pid and .alias == $alias' \
            <<< "$health_payload" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    die "$what pid=$sidecar_pid started but never served its own health on :$sidecar_port"
}

require_observed_phase() {
    local observed="$1" phase="$2"
    [[ "$observed" == 1 ]] || die "required $phase phase was not observed"
}

recorded_process_has_argv_pair() {
    local fake_pid="$1" first="$2" second="$3" command
    command="$(ps -ww -p "$fake_pid" -o command= 2>/dev/null || true)"
    [[ -n "$command" ]] || return 1

    # `ps` has no structured argv output on macOS. Split its untruncated
    # rendering with glob expansion disabled and require two adjacent exact
    # tokens rather than a substring.
    (
        set -f
        local previous="" token
        for token in $command; do
            if [[ "$previous" == "$first" && "$token" == "$second" ]]; then
                return 0
            fi
            previous="$token"
        done
        return 1
    )
}

recorded_fake_sidecar_is_live() {
    recorded_process_has_argv_pair "$1" serve "$2"
}

stop_recorded_fake_sidecar() {
    local fake_pid="$1" fake_alias="$2"
    recorded_fake_sidecar_is_live "$fake_pid" "$fake_alias" || return 0

    # Follow the server lifecycle used by the engine: request a graceful stop,
    # wait for it to finish releasing its resources, then force only the exact
    # recorded process if it ignores the deadline. Merely sending SIGTERM and
    # immediately deleting the persona races Python's final cache writes.
    kill "$fake_pid" 2>/dev/null || true
    local attempt
    for attempt in {1..40}; do
        recorded_fake_sidecar_is_live "$fake_pid" "$fake_alias" || return 0
        sleep 0.05
    done

    # Re-check the command immediately before SIGKILL. A recycled pid or a
    # process whose argv no longer names the recorded alias is not ours.
    recorded_fake_sidecar_is_live "$fake_pid" "$fake_alias" || return 0
    kill -KILL "$fake_pid" 2>/dev/null || true
    for attempt in {1..40}; do
        recorded_fake_sidecar_is_live "$fake_pid" "$fake_alias" || return 0
        sleep 0.05
    done

    printf '[gui-golden] owned fake sidecar pid=%s alias=%s did not exit\n' \
        "$fake_pid" "$fake_alias" >&2
    return 1
}

cleanup_fake_sidecars() {
    local status=0 fake_pid fake_alias records
    if [[ -n "$OUT" && -f "$OUT/fake-events.jsonl" ]]; then
        if ! records="$(jq -r 'select(.event == "server_started")
                                | "\(.pid)\t\(.alias // "-")"' \
                             "$OUT/fake-events.jsonl")"; then
            printf '[gui-golden] could not parse fake sidecar ownership log: %s\n' \
                "$OUT/fake-events.jsonl" >&2
            return 1
        fi
        # Long-lived serves that intentionally detach from the app's process
        # group are stopped through their exact PID+alias argv identity.
        while IFS=$'\t' read -r fake_pid fake_alias; do
            [[ "$fake_pid" =~ ^[0-9]+$ ]] || continue
            [[ "$fake_alias" != "-" ]] || { status=1; continue; }
            stop_recorded_fake_sidecar "$fake_pid" "$fake_alias" || status=1
        done < <(printf '%s\n' "$records" | sort -u)
    fi
    return "$status"
}

process_is_running() {
    local state
    state="$(ps -p "$1" -o stat= 2>/dev/null | tr -d '[:space:]' || true)"
    [[ -n "$state" && "$state" != Z* ]]
}

process_group_has_live_members() {
    local expected_group="$1" group state
    while read -r group state; do
        if [[ "$group" == "$expected_group" && "$state" != Z* ]]; then
            return 0
        fi
    done < <(ps -axo pgid=,stat= 2>/dev/null || true)
    return 1
}

stop_app() {
    [[ -n "$APP_PID" ]] || return 0
    local app_pid="$APP_PID" app_group="" attempt
    if ! process_is_running "$app_pid"; then
        wait "$app_pid" 2>/dev/null || true
        APP_PID=""
        return 0
    fi

    app_group="$(ps -p "$app_pid" -o pgid= 2>/dev/null | tr -d '[:space:]')"
    if [[ "$app_group" != "$app_pid" ]]; then
        # Never signal a shared or unknown process group. The launcher below
        # establishes a session whose pgid equals the app pid; a mismatch is a
        # broken ownership boundary, so stop only the known pid and preserve
        # the persona for diagnosis.
        kill "$app_pid" 2>/dev/null || true
        for attempt in {1..20}; do
            process_is_running "$app_pid" || break
            sleep 0.1
        done
        kill -KILL "$app_pid" 2>/dev/null || true
        wait "$app_pid" 2>/dev/null || true
        printf '[gui-golden] app pid=%s did not own its process group (pgid=%s)\n' \
            "$app_pid" "${app_group:-unknown}" >&2
        return 1
    fi

    # The isolated app is the leader of a harness-owned session. Signal the
    # whole group so catalogue probes that have not written lifecycle events
    # yet cannot outlive their profile and recreate Python cache files during
    # deletion.
    kill -TERM -- "-$app_group" 2>/dev/null || true
    for attempt in {1..20}; do
        process_is_running "$app_pid" || break
        sleep 0.1
    done
    if process_is_running "$app_pid"; then
        kill -KILL -- "-$app_group" 2>/dev/null || true
    fi
    wait "$app_pid" 2>/dev/null || true

    # Reaping the leader does not prove its descendants are gone. Give the
    # remaining group a second bounded drain, then force only that owned group.
    for attempt in {1..20}; do
        process_group_has_live_members "$app_group" || break
        sleep 0.1
    done
    if process_group_has_live_members "$app_group"; then
        kill -KILL -- "-$app_group" 2>/dev/null || true
        for attempt in {1..20}; do
            process_group_has_live_members "$app_group" || break
            sleep 0.1
        done
    fi
    if process_group_has_live_members "$app_group"; then
        printf '[gui-golden] owned app process group %s did not exit\n' \
            "$app_group" >&2
        return 1
    fi
    APP_PID=""
}

write_result() {
    local status="$1" exit_code="$2" finished_epoch duration_seconds
    finished_epoch="$(date +%s)"
    duration_seconds=$((finished_epoch - RUN_STARTED_EPOCH))
    jq -n --arg status "$status" --arg flow "$FLOW" --arg app "$APP_SOURCE" \
        --arg started_at "$RUN_STARTED_AT" --arg artifact_path "$OUT_ROOT" \
        --argjson duration_seconds "$duration_seconds" \
        --argjson exit_code "$exit_code" \
        '{status: $status, flow: $flow, app: $app, started_at: $started_at,
          duration_seconds: $duration_seconds, artifact_path: $artifact_path,
          exit_code: $exit_code}' > "$OUT_ROOT/result.json"
}

finish() {
    local status=$? cleanup_failed=0
    set +e
    cleanup_persona || cleanup_failed=1
    cleanup_operator_server || cleanup_failed=1
    cleanup_telemetry_sink || cleanup_failed=1
    if [[ "$status" -eq 0 && "$cleanup_failed" -ne 0 ]]; then
        status=1
    fi
    if [[ "$status" -ne 0 ]]; then
        mkdir -p "$OUT_ROOT" 2>/dev/null || true
        if [[ -d "$OUT_ROOT" ]]; then
            # Cleanup failure overrides an earlier PASS result. A retained
            # process/profile is a failed journey, never green evidence.
            write_result fail "$status" 2>/dev/null || true
            RESULT_WRITTEN=1
        fi
    fi
    trap - EXIT
    exit "$status"
}

# When sourced, expose only the executable contract helpers above. Return
# before cleanup traps, app launch, filesystem mutation, or tool preflight.
# Direct execution cannot be bypassed with an environment variable.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

pb() { peekaboo "$@" --bridge-socket "$BRIDGE"; }
flow_requires_screen_recording() {
    case "$FLOW" in
        all) return 0 ;;
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
        fresh-install|cached-quickstart|cached-curated-tradeup|cached-variant-collapse|download-progress|settings-persistence|settings-mtp|chat-restore|chat-depth|browse-all-destination|no-dead-controls|catalog-integrity|update-state|update-busy|campaign-banner|launch-integrations) return 1 ;;
        model-switch-active-request|model-crash-recovery|low-memory-choice|chat-document-attachment|chat-multimodal-attachments|image-generation|dictation|dictation-rc2-upgrade|audio-readiness|window-close-prompt|resident-load-rejected) return 1 ;;
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
    local app_stopped=1 sidecars_stopped=1
    stop_app || app_stopped=0
    cleanup_fake_sidecars || sidecars_stopped=0
    if [[ "$KEEP" == 0 && -n "$BUNDLE_ID" ]]; then
        defaults delete "$BUNDLE_ID" >/dev/null 2>&1 || true
    fi
    if [[ "$app_stopped" == 0 || "$sidecars_stopped" == 0 ]]; then
        printf '[gui-golden] preserving persona because an owned process is still live: %s\n' \
            "$PERSONA" >&2
        return 1
    fi
    if [[ "$KEEP" == 0 && -n "$PERSONA" && -d "$PERSONA" ]]; then
        if ! rm -rf "$PERSONA"; then
            # Keep the surviving tree available for ownership diagnosis and
            # prevent the EXIT trap from turning a failed cleanup into a
            # second, unobserved delete attempt.
            KEEP=1
            printf '[gui-golden] persona cleanup failed; preserving evidence: %s\n' \
                "$PERSONA" >&2
            return 1
        fi
    fi
    PERSONA=""
    BUNDLE_ID=""
    PERSONA_ENV=()
}

stop_tracked_child() {
    local label="$1" child_pid="$2" attempt
    if ! process_is_running "$child_pid"; then
        wait "$child_pid" 2>/dev/null || true
        return 0
    fi
    kill "$child_pid" 2>/dev/null || true
    for attempt in {1..20}; do
        process_is_running "$child_pid" || break
        sleep 0.1
    done
    if process_is_running "$child_pid"; then
        kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
    if process_is_running "$child_pid"; then
        printf '[gui-golden] owned %s pid=%s did not exit\n' \
            "$label" "$child_pid" >&2
        return 1
    fi
}

cleanup_operator_server() {
    [[ -n "$OPERATOR_SERVER_PID" ]] || return 0
    stop_tracked_child operator-server "$OPERATOR_SERVER_PID" || return 1
    OPERATOR_SERVER_PID=""
}

cleanup_telemetry_sink() {
    if [[ -n "$TELEMETRY_SINK_PID" ]]; then
        stop_tracked_child telemetry-sink "$TELEMETRY_SINK_PID" || return 1
    fi
    TELEMETRY_SINK_PID=""
    TELEMETRY_SINK_PORT=""
    TELEMETRY_SINK_LOG=""
}

start_telemetry_sink() {
    local artifact_dir="$1"
    local ready_file="$artifact_dir/telemetry-sink.port"
    local server_log="$artifact_dir/telemetry-sink-server.log"
    mkdir -p "$artifact_dir"
    cleanup_telemetry_sink
    TELEMETRY_SINK_LOG="$artifact_dir/telemetry-sink-requests.jsonl"
    : > "$TELEMETRY_SINK_LOG"
    rm -f "$ready_file"

    # Bind port zero so the kernel chooses a free loopback port. The sink logs
    # only request metadata: the acceptance check needs proof that nothing was
    # sent, not a second store of telemetry payloads.
    python3 - "$TELEMETRY_SINK_LOG" "$ready_file" > "$server_log" 2>&1 <<'PY' &
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

request_log, ready_file = sys.argv[1:]
write_lock = threading.Lock()


class Sink(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        event = None
        timestamp = None
        activation_kind = None
        activation_surface = None
        activation_keys = []
        try:
            envelope = json.loads(payload)
            batch = envelope.get("batch") if isinstance(envelope, dict) else None
            first = batch[0] if isinstance(batch, list) and batch else None
            if isinstance(first, dict):
                event = first.get("event") if isinstance(first.get("event"), str) else None
                timestamp = first.get("timestamp") if isinstance(first.get("timestamp"), str) else None
                activation = first.get("activation")
                if isinstance(activation, dict):
                    activation_kind = activation.get("activation_kind")
                    activation_surface = activation.get("surface")
                    activation_keys = sorted(activation)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        record = {
            "method": "POST",
            "path": self.path,
            "bytes": length,
            "event": event,
            "timestamp": timestamp,
            "activation_kind": activation_kind,
            "activation_surface": activation_surface,
            "activation_keys": activation_keys,
        }
        with write_lock, open(request_log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.send_response(202)
        self.end_headers()

    def log_message(self, _format, *_args):
        return


class LoopbackSinkServer(ThreadingTCPServer):
    # ``HTTPServer.server_bind`` resolves the runner hostname through
    # ``getfqdn`` before it writes the ready file. Hosted macOS DNS can stall
    # there even though the loopback bind itself is healthy, so this fixture
    # uses the protocol-compatible TCP server directly.
    daemon_threads = True
    allow_reuse_address = True


server = LoopbackSinkServer(("127.0.0.1", 0), Sink)
with open(ready_file, "w", encoding="utf-8") as stream:
    stream.write(str(server.server_address[1]))
    stream.flush()
    os.fsync(stream.fileno())
server.serve_forever()
PY
    TELEMETRY_SINK_PID=$!

    for _ in {1..80}; do
        if [[ -s "$ready_file" ]] && kill -0 "$TELEMETRY_SINK_PID" 2>/dev/null; then
            TELEMETRY_SINK_PORT="$(cat "$ready_file")"
            break
        fi
        sleep 0.05
    done
    [[ "$TELEMETRY_SINK_PORT" =~ ^[0-9]+$ ]] \
        || die "loopback telemetry sink did not become ready — see $server_log"
    jq -n --arg endpoint "http://127.0.0.1:$TELEMETRY_SINK_PORT/v1/events" \
        --argjson pid "$TELEMETRY_SINK_PID" \
        '{ready: true, endpoint: $endpoint, pid: $pid}' \
        > "$artifact_dir/telemetry-sink-ready.json"
}

assert_no_telemetry_requests() {
    local stage="$1"
    local evidence="$OUT/telemetry-$stage.json"
    local count
    # Give an erroneously dispatched URLSession request time to reach the
    # independent sink before declaring the boundary quiet.
    sleep 0.5
    kill -0 "$TELEMETRY_SINK_PID" 2>/dev/null \
        || die "loopback telemetry sink exited during $stage"
    count="$(wc -l < "$TELEMETRY_SINK_LOG" | tr -d '[:space:]')"
    jq -n --arg stage "$stage" --arg log "$TELEMETRY_SINK_LOG" \
        --argjson request_count "$count" \
        '{stage: $stage, request_count: $request_count, request_log: $log}' \
        > "$evidence"
    [[ "$count" == 0 ]] \
        || die "telemetry request crossed the consent boundary during $stage — see $TELEMETRY_SINK_LOG"
}

assert_one_telemetry_request() {
    local stage="$1"
    local not_before="$2"
    local evidence="$OUT/telemetry-$stage.json"
    local settling_seconds=0.5
    local count=0
    for _ in {1..80}; do
        kill -0 "$TELEMETRY_SINK_PID" 2>/dev/null \
            || die "loopback telemetry sink exited during $stage"
        count="$(wc -l < "$TELEMETRY_SINK_LOG" | tr -d '[:space:]')"
        [[ "$count" == 1 ]] && break
        [[ "$count" -gt 1 ]] && break
        sleep 0.05
    done
    # Keep the sink open after the first request so a near-following duplicate
    # or delayed pre-consent request is included in the final exact count.
    if [[ "$count" == 1 ]]; then
        sleep "$settling_seconds"
        kill -0 "$TELEMETRY_SINK_PID" 2>/dev/null \
            || die "loopback telemetry sink exited while settling $stage"
        count="$(wc -l < "$TELEMETRY_SINK_LOG" | tr -d '[:space:]')"
    fi
    jq -s --arg stage "$stage" --arg log "$TELEMETRY_SINK_LOG" \
        --arg not_before "$not_before" \
        '{stage: $stage, request_count: length, request_log: $log,
          not_before: $not_before,
          requests: map({method, path, bytes, event, timestamp})}' \
        "$TELEMETRY_SINK_LOG" > "$evidence"
    [[ "$count" == 1 ]] \
        || die "Settings opt-in did not produce exactly one telemetry request during $stage"
    jq -e '(.request_count == 1)
              and (.requests[0].method == "POST")
              and (.requests[0].path == "/v1/events")
              and (.requests[0].bytes > 0)
              and (.requests[0].event == "session_start")
              and (.requests[0].timestamp >= .not_before)' \
        "$evidence" >/dev/null \
        || die "Settings opt-in did not produce a new session_start during $stage"
}

assert_share_activation_requests() {
    local stage="$1"
    local expected_kind="$2"
    local evidence="$OUT/telemetry-$stage.json"
    local count=0
    for _ in {1..120}; do
        kill -0 "$TELEMETRY_SINK_PID" 2>/dev/null \
            || die "loopback telemetry sink exited during $stage"
        count="$(wc -l < "$TELEMETRY_SINK_LOG" | tr -d '[:space:]')"
        [[ "$count" -ge 2 ]] && break
        sleep 0.05
    done
    # A short quiet period turns "the two expected requests arrived" into
    # "exactly those two requests arrived" rather than racing a third send.
    sleep 0.5
    count="$(wc -l < "$TELEMETRY_SINK_LOG" | tr -d '[:space:]')"
    jq -s --arg stage "$stage" --arg log "$TELEMETRY_SINK_LOG" \
        --arg expected_kind "$expected_kind" \
        '{stage: $stage, request_count: length, request_log: $log,
          expected_activation_kind: $expected_kind, requests: .}' \
        "$TELEMETRY_SINK_LOG" > "$evidence"
    [[ "$count" == 2 ]] \
        || die "Share produced $count telemetry requests instead of session_start plus one activation"
    jq -e '. as $evidence |
        ([.requests[] | select(.event == "session_start")] | length) == 1
        and ([.requests[] | select(.event == "activation")] | length) == 1
        and ([.requests[] | select(
            .event == "activation"
            and .method == "POST"
            and .path == "/v1/events"
            and .bytes > 0
            and .activation_kind == $evidence.expected_activation_kind
            and .activation_surface == "desktop"
            and .activation_keys == ["activation_kind", "surface"]
        )] | length) == 1' \
        "$evidence" >/dev/null \
        || die "Share did not produce exactly one valid $expected_kind Desktop activation"
}

trap finish EXIT
# Signal handlers only select the conventional exit code. The EXIT handler is
# the single owner of cleanup and final evidence, avoiding double-cleanup and
# ensuring cancellation/timeout failures receive the same result schema.
trap 'exit 130' INT
trap 'exit 143' TERM

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

launch_persona_app() {
    local log_mode="$1"
    shift
    if [[ "$log_mode" == "append" ]]; then
        env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
            DOGFOOD_WORKING_SET_GB=0.1 \
            FAKE_EVENT_LOG="$OUT/fake-events.jsonl" \
            "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}" \
            /usr/bin/python3 -c \
            'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
            "$PERSONA/launch.sh" "$@" >> "$OUT/app.log" 2>&1 &
    else
        env RAPID_BIN="$ROOT/scripts/fake-rapid-mlx.sh" \
            DOGFOOD_WORKING_SET_GB=0.1 \
            FAKE_EVENT_LOG="$OUT/fake-events.jsonl" \
            "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}" \
            /usr/bin/python3 -c \
            'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
            "$PERSONA/launch.sh" "$@" > "$OUT/app.log" 2>&1 &
    fi
    APP_PID=$!

    local app_group="" attempt
    for attempt in {1..40}; do
        app_group="$(ps -p "$APP_PID" -o pgid= 2>/dev/null | tr -d '[:space:]')"
        [[ "$app_group" == "$APP_PID" ]] && return 0
        kill -0 "$APP_PID" 2>/dev/null || break
        sleep 0.05
    done
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    printf '[gui-golden] isolated app failed to establish its owned process group\n' >&2
    APP_PID=""
    return 1
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
    # All regression journeys except fresh-install start with a durable quiet
    # opt-out. They test their named feature, not the one-time consent invite;
    # leaving the preference absent would make the first successful result add
    # an unrelated banner midway through dozens of snapshots.
    if [[ "$name" != "fresh-install" && "$name" != "fresh-install-share" ]]; then
        env HOME="$PERSONA/home" CFFIXED_USER_HOME="$PERSONA/home" \
            /usr/bin/defaults write "$BUNDLE_ID" \
            com.rapidmlx.rapid.telemetry.enabled -bool false
    fi
    local config="$PERSONA/home/.rapid-golden-fake.json"
    jq -n \
        --arg event_log "$OUT/fake-events.jsonl" \
        --arg pull_state "$OUT/pulled-models" \
        '{FAKE_EVENT_LOG: $event_log, FAKE_PULL_STATE: $pull_state}' > "$config"
    local assignment key value updated app_language=""
    # macOS ships Bash 3.2, where expanding a declared-but-empty array under
    # `set -u` raises "unbound variable". The `+` form expands to nothing
    # when the array has no elements and preserves argv boundaries otherwise.
    for assignment in "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}"; do
        key="${assignment%%=*}"
        value="${assignment#*=}"
        if [[ "$key" == "RAPID_GUI_APP_LANGUAGE" ]]; then
            app_language="$value"
        fi
        updated="$config.next"
        jq --arg key "$key" --arg value "$value" '.[$key] = $value' "$config" > "$updated"
        mv "$updated" "$config"
    done
    if [[ -n "$app_language" ]]; then
        launch_persona_app truncate -AppleLanguages "($app_language)"
    else
        launch_persona_app truncate
    fi
    wait_for_window
}

relaunch_persona() {
    stop_app
    # A relaunch keeps the persona but starts a fresh app process. Reap only
    # the fake sidecars this harness recorded before starting it again. Before
    # #1618 the app's unsafe global port sweep hid this ownership leak by
    # killing the old fake (and potentially an operator's real server too).
    cleanup_fake_sidecars
    launch_persona_app append
    wait_for_window
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

settle_transcript_at_bottom() {
    local destination="$1" press_result="$2"
    see_main "$destination"
    # The window can expose both sidebar and transcript scrollbars. Correlate
    # the transcript bar with the compose surface: it is the first vertical
    # bar to the Send button's right. This anchor remains present after the
    # Jump-to-latest overlay hides.
    local scroll_x before_value
    scroll_x="$(jq -r '
        ([.data.ui_elements[]?
          | select(.identifier == "ChatView.SendOrStopButton")
          | .bounds.x] | first) as $compose_x
        | [.data.ui_elements[]?
           | select(.role == "AXScrollBar"
                    and (.value | type) == "number"
                    and .bounds.height > .bounds.width
                    and .bounds.x > $compose_x)]
        | sort_by(.bounds.x) | .[0].bounds.x // empty' "$destination")"
    [[ -n "$scroll_x" ]] \
        || die "could not identify the transcript scrollbar beside the compose surface"
    before_value="$(jq -r --argjson scroll_x "$scroll_x" '
        [.data.ui_elements[]?
         | select(.role == "AXScrollBar"
                  and (.value | type) == "number"
                  and ((.bounds.x - $scroll_x) | fabs) < 1)
         | .value] | first // empty' "$destination")"
    [[ -n "$before_value" ]] \
        || die "transcript exposes no measurable scroll position before Jump to latest"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Transcript.JumpToBottom")' \
        "$destination" >/dev/null; then
        press "$destination" Transcript.JumpToBottom "$press_result" \
            || die "transcript was not at its tail and Jump to latest was not pressable"
    fi
    local previous_tail_key="" stable_samples=0
    for _ in {1..60}; do
        see_main "$destination"
        local current_value tail_marker_visible tail_key=""
        current_value="$(jq -r --argjson scroll_x "$scroll_x" '
            [.data.ui_elements[]?
             | select(.role == "AXScrollBar"
                      and (.value | type) == "number"
                      and ((.bounds.x - $scroll_x) | fabs) < 1)
             | .value] | first // empty' "$destination")"
        # AppKit may remove an overlay scrollbar once a short transcript fits
        # entirely inside its viewport. In that state, the last assistant
        # action row being fully visible is stronger physical evidence than a
        # scrollbar value that no longer exists.
        tail_marker_visible="$(jq -r '
            ([.data.ui_elements[]?
              | select(.role == "AXScrollArea"
                       and .bounds.x > 200
                       and .bounds.width > 300
                       and .bounds.height > 100)]
             | sort_by(.bounds.width) | last) as $transcript
            | if $transcript == null then false else
                ([.data.ui_elements[]?
                  | select((.identifier // "")
                           | startswith("ChatView.Message.Retry."))
                  | select(.bounds.y >= $transcript.bounds.y
                           and (.bounds.y + .bounds.height)
                               <= ($transcript.bounds.y + $transcript.bounds.height))]
                 | length) > 0
              end' "$destination")"
        if [[ -n "$current_value" ]] \
            && awk -v current="$current_value" \
                'BEGIN { exit !(current >= 0.99) }'; then
            tail_key="$current_value"
        elif [[ -z "$current_value" && "$tail_marker_visible" == "true" ]]; then
            tail_key="assistant-tail-visible"
        fi
        if [[ -n "$tail_key" ]] \
            && ! jq -e '.data.ui_elements[]?
                        | select(.identifier == "Transcript.JumpToBottom")' \
                "$destination" >/dev/null; then
            if [[ "$previous_tail_key" == "$tail_key" ]]; then
                stable_samples=$((stable_samples + 1))
            else
                stable_samples=1
            fi
            if (( stable_samples >= 2 )); then
                return
            fi
        else
            stable_samples=0
        fi
        previous_tail_key="$tail_key"
        sleep 0.1
    done
    die "Jump to latest did not physically settle the transcript at its tail"
}

wait_identifier_enabled() {
    local identifier="$1" destination="$2" attempts="${3:-80}"
    for ((i=0; i<attempts; i++)); do
        see_main "$destination"
        if jq -e --arg id "$identifier" \
            '.data.ui_elements[]? | select(.identifier == $id and .enabled == true)' \
            "$destination" >/dev/null; then
            return
        fi
        sleep 0.25
    done
    die "timed out waiting for enabled AX identifier $identifier"
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

wait_selected() {
    local identifier="$1" destination="$2" attempts="${3:-80}"
    for ((i=0; i<attempts; i++)); do
        see_main "$destination"
        if jq -e --arg id "$identifier" \
            '.data.ui_elements[]?
             | select(.identifier == $id)
             | select(.selected == true or .value == 1 or .value == "1")' \
            "$destination" >/dev/null; then
            return
        fi
        sleep 0.25
    done
    die "timed out waiting for AX selection: $identifier"
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
    # Retry the press itself. SwiftUI can replace the accessibility element
    # backing a control between the dump above and the AXPress below, and the
    # press then fails with a transient invalid-element / cannot-complete error
    # (#2009 identified this and fixed three call sites inline; there are 126).
    # A single transient miss on ANY of them failed the whole gate, which is why
    # three consecutive runs of this suite failed at three DIFFERENT controls —
    # Choose File twice, then Check for updates. The identifier precheck stays
    # OUTSIDE the loop: a genuinely absent control must still fail immediately
    # rather than costing three attempts.
    local attempt
    for attempt in 1 2 3; do
        if "$AX_DRIVER" press "$APP_PID" "$identifier" > "$evidence" 2>/dev/null \
            && jq -e '.success' "$evidence" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.4
    done
    printf '[gui-golden] AXPress failed after 3 attempts: %s\n' "$identifier" >&2
    return 1
}

round_trip_toggle() {
    local identifier="$1" stem="$2"
    local before after restored
    see_main "$OUT/$stem-before.json"
    before="$(element_field "$OUT/$stem-before.json" "$identifier" value)"
    [[ -n "$before" ]] || die "$identifier exposes no AX value before its press"
    press "$OUT/$stem-before.json" "$identifier" "$OUT/$stem-press.json" \
        || die "$identifier is not pressable"
    see_main "$OUT/$stem-after.json"
    after="$(element_field "$OUT/$stem-after.json" "$identifier" value)"
    [[ -n "$after" && "$after" != "$before" ]] \
        || die "$identifier accepted AXPress but did not change value"
    press "$OUT/$stem-after.json" "$identifier" "$OUT/$stem-restore-press.json" \
        || die "$identifier could not be restored"
    see_main "$OUT/$stem-restored.json"
    restored="$(element_field "$OUT/$stem-restored.json" "$identifier" value)"
    [[ "$restored" == "$before" ]] \
        || die "$identifier did not round-trip to its original value"
}

press_and_require_selected() {
    local identifier="$1" stem="$2"
    see_main "$OUT/$stem-before.json"
    press "$OUT/$stem-before.json" "$identifier" "$OUT/$stem-press.json" \
        || die "$identifier is not pressable"
    see_main "$OUT/$stem-after.json"
    jq -e --arg identifier "$identifier" \
        '.data.ui_elements[]? | select(.identifier == $identifier)
         | select(.selected == true or .value == 1 or .value == "1")' \
        "$OUT/$stem-after.json" >/dev/null \
        || die "$identifier accepted AXPress but did not become selected"
}

dismiss_first_run() {
    local tree="$OUT/first-run.json"
    see_main "$tree"
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
    # The readiness band is mounted before its action becomes interactive
    # while the catalog finishes resolving the selected model.  Pressing the
    # merely-present button is accepted by AX but dropped by SwiftUI, which
    # made slower hosted runners wait a full minute for a sidecar that was
    # never asked to start.  Gate the interaction on the same enabled state a
    # user needs before clicking it.
    wait_identifier_enabled Readiness.Action "$OUT/readiness-start.json"
    local initial_action
    initial_action="$(element_field "$OUT/readiness-start.json" Readiness.Action description)"
    press "$OUT/readiness-start.json" Readiness.Action "$OUT/start-model.json"
    if [[ "$initial_action" == "Download" ]]; then
        local download_ready=0
        for _ in {1..240}; do
            see_main "$OUT/readiness-after-download.json"
            if jq -e '.data.ui_elements[]?
                      | select(.identifier == "Readiness.Action"
                               and .description == "Start"
                               and .enabled == true)' \
                "$OUT/readiness-after-download.json" >/dev/null; then
                download_ready=1
                break
            fi
            sleep 0.25
        done
        [[ "$download_ready" == 1 ]] \
            || die "download completed without exposing the model Start action"
        press "$OUT/readiness-after-download.json" Readiness.Action \
            "$OUT/start-downloaded-model.json"
    fi
    # ``server_started`` says the fake bound its port; it does NOT say the app
    # has finished wiring up to it. The old gate also tested
    # ``description == "Send message"``, which is the button's label for the
    # whole startup — including while its hint still reads "<alias> is still
    # starting." So this returned early, ``send_prompt`` pressed into a closed
    # readiness gate, and the press was silently dropped (observed: 1 run in 2).
    # Hosted macOS runners can spend more than 30 seconds cold-starting the
    # bundled fake sidecar after a full release build. Keep the event-based
    # readiness proof, but allow 60 seconds before declaring startup broken.
    local memory_confirmed=0
    for _ in {1..240}; do
        grep -q '"event": "server_started"' "$OUT/fake-events.jsonl" 2>/dev/null && break
        # Authoritative aliases retain their production footprint estimate
        # even though this journey launches the zero-weight fake sidecar.  A
        # memory-constrained hosted runner can therefore present the real
        # safety confirmation after Start.  Confirm only when that explicit
        # sheet exists; this keeps the product gate covered without leaving a
        # fake-model GUI journey dependent on the runner's ambient pressure.
        if [[ "$memory_confirmed" == 0 ]]; then
            see_main "$OUT/readiness-after-start.json"
            if jq -e '.data.ui_elements[]?
                      | select(.identifier == "MemoryWarning.Confirm" and .enabled == true)' \
                "$OUT/readiness-after-start.json" >/dev/null; then
                "$AX_DRIVER" click-center "$APP_PID" MemoryWarning.Confirm \
                    > "$OUT/readiness-memory-confirm.json"
                memory_confirmed=1
                log "  confirmed hosted-runner memory warning for fake sidecar"
            fi
        fi
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
    start_telemetry_sink "$OUT_ROOT/fresh-install"
    # The real engine registry always contains the starter. Without this row,
    # the fake catalog makes the app correctly fall back to its only chat row
    # and the assertion below can never prove the production first-run rule.
    start_persona fresh-install FAKE_INCLUDE_STARTER=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND" \
        RAPID_MLX_TELEMETRY_ENDPOINT="http://127.0.0.1:$TELEMETRY_SINK_PORT/v1/events"
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    assert_no_telemetry_requests before-onboarding
    if jq -e '.data.ui_elements[]? | select((.identifier? // "") | startswith("TelemetryConsent."))' \
        "$OUT/welcome.json" >/dev/null; then
        die "fresh install asked for telemetry before Rapid delivered product value"
    fi

    # Direction D owns the window rather than mounting production controls
    # behind a sheet. Pin all three responsive tiers before continuing the
    # pre-existing fresh-install journey into the production shell.
    for hidden in Sidebar.NewChat Sidebar.Launch rapid.chat.compose; do
        if jq -e --arg id "$hidden" '.data.ui_elements[]? | select(.identifier == $id)' \
            "$OUT/welcome.json" >/dev/null; then
            die "Direction D mounted production control $hidden behind onboarding"
        fi
    done
    "$AX_DRIVER" set-window-size "$APP_PID" Rapid-MLX 1400x850 > "$OUT/wide-size.json"
    see_main "$OUT/wide.json"
    baseline onboarding-direction-d.wide "$OUT/wide.json"
    "$AX_DRIVER" set-window-size "$APP_PID" Rapid-MLX 1000x760 > "$OUT/medium-size.json"
    see_main "$OUT/medium.json"
    baseline onboarding-direction-d.medium "$OUT/medium.json"
    "$AX_DRIVER" set-window-size "$APP_PID" Rapid-MLX 720x700 > "$OUT/compact-size.json"
    see_main "$OUT/compact.json"
    baseline onboarding-direction-d.compact "$OUT/compact.json"
    press "$OUT/compact.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_selected Quickstart.Choice.lfm2.5-1b-4bit "$OUT/chooser-settled.json"
    baseline onboarding-direction-d.compact-chooser "$OUT/chooser-settled.json"
    press "$OUT/chooser-settled.json" Quickstart.Footer.Back "$OUT/chooser-back.json"
    wait_identifier Quickstart.Skip "$OUT/welcome-returned.json"
    press "$OUT/welcome-returned.json" Quickstart.Skip "$OUT/quickstart-skip.json"
    wait_identifier rapid.chat.compose "$OUT/steady.json"
    selected_model="$(element_field "$OUT/steady.json" ModelPickerBar.ModelMenu value)"
    [[ "$selected_model" == *"lfm2.5-1b-4bit"* ]] \
        || die "#2219: 8 GB onboarding selected '$selected_model' instead of the compact starter"
    for id in Sidebar.NewChat Sidebar.Launch rapid.chat.compose ChatView.SendOrStopButton ModelPickerBar.ModelMenu; do
        jq -e --arg id "$id" '.data.ui_elements[]? | select(.identifier == $id)' "$OUT/steady.json" >/dev/null \
            || die "post-onboarding shell missing $id"
    done
    baseline fresh-install.steady "$OUT/steady.json"
    if jq -e '.data.ui_elements[]? | select((.identifier? // "") | startswith("TelemetryConsent."))' \
        "$OUT/steady.json" >/dev/null; then
        die "fresh install asked for telemetry before the first working feature"
    fi
    assert_no_telemetry_requests before-first-value
    # Exercise the scene/content contract, not only the constant. Before the
    # fix the declared floor was never applied and AppKit accepted ~616pt.
    # Asking for 500pt must be clamped by the live window to at least 720pt.
    "$AX_DRIVER" set-window-size "$APP_PID" Rapid-MLX 500x500 \
        > "$OUT/window-floor.json" \
        || die "the main window rejected a native resize request"
    jq -e '.actual.width >= 720 and .actual.height >= 560' \
        "$OUT/window-floor.json" >/dev/null \
        || die "the live main window did not enforce its 720x560 floor: $(jq -c .actual "$OUT/window-floor.json")"

    # Consent follows proof of value instead of blocking first launch. A real
    # completed assistant turn is the trigger; the answer remains visible and
    # usable under the non-modal invitation.
    start_model
    send_prompt "Say hello in one short sentence." "post-value-consent"
    wait_identifier TelemetryConsent.PostValueBanner "$OUT/post-value-consent-visible.json"
    # Streaming completion and scroll anchoring settle independently. Capture
    # the structural baseline only after the transcript reaches its stable tail.
    settle_transcript_at_bottom "$OUT/post-value-consent-visible.json" \
        "$OUT/post-value-consent-jump-press.json"
    assert_tree_text "$OUT/post-value-consent-visible.json" "Hello"
    [[ "$(jq '[.data.ui_elements[]? | select(.identifier == "TelemetryConsent.PostValueBanner")] | length' \
        "$OUT/post-value-consent-visible.json")" == 1 ]] \
        || die "the first successful reply did not show exactly one telemetry invitation"
    assert_no_telemetry_requests post-value-before-decision
    baseline fresh-install.post-value-consent "$OUT/post-value-consent-visible.json"
    # The invitation is part of the main window, not a modal. Escape belongs
    # to the active app interaction and must never answer a permanent privacy
    # choice on the user's behalf.
    "$AX_DRIVER" key "$APP_PID" escape > "$OUT/post-value-consent-escape.json"
    wait_identifier TelemetryConsent.PostValueBanner "$OUT/post-value-consent-after-escape.json"
    press "$OUT/post-value-consent-after-escape.json" \
        TelemetryConsent.PostValue.Decline \
        "$OUT/post-value-consent-explicit-decline.json"
    for _ in {1..40}; do
        see_main "$OUT/post-value-consent-declined.json"
        if ! jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.PostValueBanner")' \
            "$OUT/post-value-consent-declined.json" >/dev/null; then
            break
        fi
        sleep 0.25
    done
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.PostValueBanner")' \
        "$OUT/post-value-consent-declined.json" >/dev/null; then
        die "explicit No thanks did not dismiss the telemetry invitation"
    fi
    assert_no_telemetry_requests after-decline
    relaunch_persona
    wait_identifier rapid.chat.compose "$OUT/post-value-consent-relaunch.json"
    if jq -e '.data.ui_elements[]? | select(.identifier == "TelemetryConsent.PostValueBanner")' \
        "$OUT/post-value-consent-relaunch.json" >/dev/null; then
        die "dismissed telemetry invitation returned after relaunch"
    fi
    assert_no_telemetry_requests declined-relaunch
    # Positive control for the five quiet checkpoints above: use the promised
    # reversible Settings path and require the app to reach this exact sink.
    # Without this, a dropped endpoint override could make every negative
    # assertion pass while telemetry escaped to a different destination.
    open_settings
    wait_settings_stable "$OUT/telemetry-settings-rail.json" Settings.Category.privacy
    press "$OUT/telemetry-settings-rail.json" Settings.Category.privacy \
        "$OUT/telemetry-open-privacy.json" \
        || die "Privacy category is not pressable after declined consent"
    wait_settings_stable "$OUT/telemetry-settings-privacy.json" Settings.Privacy.TelemetryToggle
    [[ "$(element_field "$OUT/telemetry-settings-privacy.json" \
        Settings.Privacy.TelemetryToggle value)" == "0" ]] \
        || die "declined telemetry decision was not still off in Settings"
    # TelemetryEvent timestamps have whole-second precision. Cross a UTC
    # second boundary before the toggle so a request created during launch but
    # delayed in URLSession cannot masquerade as this positive control.
    local boundary_second opt_in_not_before
    boundary_second="$(date -u +%s)"
    while [[ "$(date -u +%s)" == "$boundary_second" ]]; do sleep 0.05; done
    opt_in_not_before="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    press "$OUT/telemetry-settings-privacy.json" Settings.Privacy.TelemetryToggle \
        "$OUT/telemetry-settings-opt-in.json" \
        || die "Settings telemetry opt-in is not pressable"
    assert_one_telemetry_request settings-opt-in "$opt_in_not_before"
    cleanup_persona
    cleanup_telemetry_sink

    # A second pristine profile takes the affirmative path. Keep it separate
    # so the decline/no-re-ask contract above remains intact while this lane
    # proves that a success retained before consent becomes one accepted,
    # content-free activation only after Share.
    log "  consent Share emits one accepted first-chat activation"
    start_telemetry_sink "$OUT_ROOT/fresh-install-share"
    start_persona fresh-install-share FAKE_INCLUDE_STARTER=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND" \
        RAPID_MLX_TELEMETRY_ENDPOINT="http://127.0.0.1:$TELEMETRY_SINK_PORT/v1/events"
    dismiss_first_run
    assert_no_telemetry_requests share-before-first-value
    start_model
    send_prompt "Say hello in one short sentence." "share-activation"
    wait_identifier TelemetryConsent.PostValueBanner "$OUT/share-consent-visible.json"
    assert_no_telemetry_requests share-before-decision
    press "$OUT/share-consent-visible.json" TelemetryConsent.PostValue.Share \
        "$OUT/share-consent-accepted.json" \
        || die "Share was not actionable after the first successful reply"
    assert_share_activation_requests share-accepted first_chat_reply
    [[ -f "$PERSONA/home/.rapid-mlx/activation_seen_desktop_first_chat_reply" ]] \
        || die "accepted first_chat_reply did not claim its once-per-install marker"
    cleanup_persona
    cleanup_telemetry_sink
}

flow_cached_quickstart() {
    log "cached Quickstart starts without downloading (#1793)"
    # Reproduce #1618, not merely its configuration strings: an
    # operator-owned rapid-mlx-shaped listener is alive in the default port
    # window
    # before the dogfood app launches. The isolated persona must bind its own
    # high port without sweeping or terminating this process.
    # A server started in another Terminal owns a different process group.
    # Non-interactive CI shells disable job control, though, so a bare `&`
    # would put this fixture in the runner shell's group alongside the app and
    # its child.  That makes an ownership-safe group shutdown look like it
    # killed the operator fixture.  Give the fixture the same isolation a real
    # operator-owned process has; `$!` remains its pid because Python execs the
    # fake in place after setsid().
    local operator_port=""
    local candidate
    for candidate in {8000..8009}; do
        if ! /usr/sbin/lsof -nP -sTCP:LISTEN -ti :"$candidate" 2>/dev/null \
            | grep -q .; then
            operator_port="$candidate"
            break
        fi
    done
    [[ -n "$operator_port" ]] \
        || die "no free port in the operator's default 8000-8009 window"

    FAKE_EVENT_LOG="$OUT_ROOT/operator-events.jsonl" \
        /usr/bin/env python3 -c \
        'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
        "$ROOT/scripts/fake-rapid-mlx.sh" serve operator-owned \
        --host 127.0.0.1 --port "$operator_port" \
        > "$OUT_ROOT/operator-server.log" 2>&1 &
    OPERATOR_SERVER_PID=$!
    local operator_bound=0
    for _ in {1..40}; do
        if kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null \
            && curl -fsS "http://127.0.0.1:$operator_port/healthz" >/dev/null 2>&1; then
            # The port was observed free immediately before spawn. Give a
            # failed bind enough time to unwind before accepting the health
            # response, so a racing listener cannot impersonate this fixture.
            sleep 0.1
            if kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null; then
                operator_bound=1
                break
            fi
        fi
        sleep 0.1
    done
    [[ "$operator_bound" == 1 ]] \
        || die "operator fixture did not own :$operator_port; cannot establish the isolation repro"

    start_persona operator-isolation

    # #1618 is specifically a launch-sweep regression. Prove the operator's
    # listener survived the isolated app launch, then release the canonical
    # port window and this probe persona before exercising cached model
    # startup. Keeping an unrelated server — or the app launched beside it —
    # alive throughout onboarding adds no ownership coverage and couples two
    # otherwise independent regression shapes.
    if ! kill -0 "$OPERATOR_SERVER_PID" 2>/dev/null; then
        echo "=== isolated app log ===" >&2
        tail -n 120 "$OUT/app.log" >&2 || true
        echo "=== operator server log ===" >&2
        tail -n 120 "$OUT_ROOT/operator-server.log" >&2 || true
        die "dogfood launch terminated the operator-owned :$operator_port server (#1618)"
    fi
    curl -fsS "http://127.0.0.1:$operator_port/healthz" >/dev/null \
        || die "operator-owned :$operator_port server stopped responding after dogfood launch"
    cleanup_operator_server

    # Include the real cold-cache notice alongside the deterministic cached
    # fixture. Catalog output can be interleaved with prose; the chooser must
    # never promote that notice into a selectable model named "No" (#1918).
    # A fresh persona keeps this onboarding assertion independent from the
    # launch-sweep assertion above.
    start_persona cached-quickstart FAKE_EMPTY_CACHE_NOTICE=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"

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
    wait_fake_sidecar_health "$FAKE_ALIAS" "cached Quickstart sidecar"
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

flow_cached_curated_tradeup() {
    log "cached hardware-fit starter stays visible past the six-row cap"
    start_persona cached-curated-tradeup FAKE_CACHED_CURATED_TRADEUP=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=16 \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"
    # Six alphabetically earlier cached rows consume the bounded "Already on
    # this Mac" presentation. The hardware-fit cached starter still has to own
    # the one visible selected row; cached preference cannot produce a hidden
    # choice with a disabled footer.
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_selected Quickstart.CachedModel.qwen3.5-4b-4bit "$OUT/chooser.json"
    jq -e '.data.ui_elements[]?
            | select(.identifier == "Quickstart.CachedModel.qwen3.5-4b-4bit")
            | select((.description // "") | contains("on disk 2.9 GB"))
            | select(((.description // "") | contains("Download 2.9 GB")) | not)' \
        "$OUT/chooser.json" >/dev/null \
        || die "cached hardware-fit starter is hidden or advertises a download"
    # Finish the exact 16 GB starter journey without a network pull. The
    # welcome is shown only after the model is ready, so it must describe that
    # achieved state instead of promising a network-dependent duration.
    press "$OUT/chooser.json" Quickstart.Footer.Primary "$OUT/start-existing.json"
    # The 16 GB fixture owns recommendation policy, not the host's live
    # pressure. A busy hosted runner can therefore (correctly) ask for the
    # existing explicit memory confirmation before it starts this zero-weight
    # fake. Handle that real branch exactly as start_model does: confirm only
    # when the enabled warning is present, then still require the independent
    # sidecar event. Without this, ambient runner pressure made the flow wait
    # for a start the app was intentionally holding for user consent.
    local memory_confirmed=0 starter_started=0
    for _ in {1..240}; do
        if [[ -s "$OUT/fake-events.jsonl" ]] \
           && jq -e -s 'any(.[]; .event == "server_started"
                                   and .alias == "qwen3.5-4b-4bit")' \
                "$OUT/fake-events.jsonl" >/dev/null 2>&1; then
            starter_started=1
            break
        fi
        if [[ "$memory_confirmed" == 0 ]]; then
            see_main "$OUT/starter-after-start.json"
            if jq -e '.data.ui_elements[]?
                      | select(.identifier == "Quickstart.Memory.LoadAnyway"
                               and .enabled == true)' \
                "$OUT/starter-after-start.json" >/dev/null; then
                "$AX_DRIVER" click-center "$APP_PID" Quickstart.Memory.LoadAnyway \
                    > "$OUT/starter-memory-confirm.json"
                memory_confirmed=1
                log "  confirmed hosted-runner memory warning for cached starter"
            fi
        fi
        sleep 0.25
    done
    [[ "$starter_started" == 1 ]] || die "cached 16 GB starter did not start"
    wait_fake_sidecar_health "qwen3.5-4b-4bit" "cached 16 GB starter"
    # ``server_started`` is emitted before the fake binds HTTP, and a
    # constrained hosted runner can spend more than the default 20-second AX
    # budget moving from healthy sidecar to the final SwiftUI readiness step.
    # Match the existing 60-second hosted-start budget without weakening the
    # independent health or exact Ready-identifier requirements.
    wait_identifier Quickstart.Ready.StartChatting \
        "$OUT/ready-confirmation.json" 240
    press "$OUT/ready-confirmation.json" Quickstart.Ready.StartChatting \
        "$OUT/start-chatting.json"
    wait_identifier rapid.chat.compose "$OUT/ready.json"
    assert_tree_text "$OUT/ready.json" "selected to fit this Mac."
    assert_tree_text "$OUT/ready.json" "ready for your first message."
    if jq -e '.. | strings | select(test("about a minute"; "i"))' \
        "$OUT/ready.json" >/dev/null; then
        die "16 GB starter welcome still promises a fixed download time"
    fi
    cleanup_persona
}

flow_cached_variant_collapse() {
    log "first-run chooser collapses cached quant siblings (#2033 finding 3)"
    start_persona cached-variant-collapse FAKE_CACHED_VARIANTS=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_identifier Quickstart.CachedModel.qwen3-0.6b-4bit "$OUT/chooser.json"
    wait_identifier Quickstart.CachedModel.qwen3-4b-4bit "$OUT/distinct-size.json"
    wait_identifier Quickstart.CachedVariants.Toggle "$OUT/collapsed.json"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Quickstart.CachedVariant.qwen3-0.6b-8bit")' \
        "$OUT/collapsed.json" >/dev/null; then
        die "cached sibling variant is visible before disclosure"
    fi

    press "$OUT/collapsed.json" Quickstart.CachedVariants.Toggle "$OUT/expand.json"
    wait_identifier Quickstart.CachedVariant.qwen3-0.6b-8bit "$OUT/expanded.json"
    cleanup_persona
}

flow_download_progress() {
    log "download progress never shows observed bytes above its total (#1550)"
    start_persona download-progress FAKE_DOWNLOAD_OVERRUN=1 \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"

    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    # Cached preference is the normal first-run policy now. Select the explicit
    # low-memory download so this journey still exercises the 633 MB overrun
    # fixture rather than accidentally starting the cached fake model.
    wait_identifier Quickstart.Choice.lfm2.5-1b-4bit "$OUT/chooser.json"
    press "$OUT/chooser.json" Quickstart.Choice.lfm2.5-1b-4bit "$OUT/select-download.json"
    wait_selected Quickstart.Choice.lfm2.5-1b-4bit "$OUT/selected-download.json"
    press "$OUT/selected-download.json" Quickstart.Footer.Primary "$OUT/review-open.json"
    wait_identifier Quickstart.Review.Alias "$OUT/review.json"
    assert_tree_text "$OUT/review.json" "Download & start"
    # AXPress is normally immediate, but AppKit can synchronously hold the
    # accessibility action until the pull task yields.  In the full suite the
    # fake pull can therefore finish before a synchronous `press` returns,
    # making the harness miss the exact in-flight state it exists to verify.
    # Drive the action concurrently with observation, then still require the
    # action itself to have succeeded.
    press "$OUT/review.json" Quickstart.Footer.Primary "$OUT/download-start.json" &
    local press_pid=$!

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
    # The progress pipe can become AX-visible a few milliseconds before the
    # separately opened JSONL witness is observable after several personas
    # have run back-to-back. Poll the independent witness like the image/audio
    # flows do instead of treating that filesystem scheduling window as a
    # product failure.
    wait_fake_event \
        '.event == "command" and .subcommand == "pull"' \
        "download-progress flow never exercised the pull subprocess"

    # Onboarding covers the global DownloadStrip, so Step 3 must provide its
    # own reachable cancellation path. Exercise the live process rather than
    # accepting a source-level identifier: cancellation must become a notice,
    # never fabricated network advice, and Back must return to the Review
    # micro-stage that launched this transfer.
    jq -e '.data.ui_elements[]?
            | select(.identifier == "Quickstart.Download.Cancel")
            | select(.enabled == true)' "$OUT/downloading.json" >/dev/null \
        || die "an active onboarding download exposes no enabled Cancel action"
    press "$OUT/downloading.json" Quickstart.Download.Cancel \
        "$OUT/download-cancel.json" \
        || die "onboarding Cancel download is not pressable"
    wait "$press_pid" \
        || die "AXPress failed while starting the download-progress fixture"
    wait_identifier Quickstart.Retry "$OUT/download-cancelled.json"
    assert_tree_text "$OUT/download-cancelled.json" "Download stopped"
    if jq -e '(.data.ui_elements | tostring) | contains("Check your connection")' \
        "$OUT/download-cancelled.json" >/dev/null; then
        die "a user-cancelled download was misdiagnosed as a network failure"
    fi
    press "$OUT/download-cancelled.json" Quickstart.Failure.BackToModelSelection \
        "$OUT/download-back.json" \
        || die "cancelled download recovery has no working Back action"
    wait_identifier Quickstart.Review.Alias "$OUT/download-review-restored.json"
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

flow_settings_mtp() {
    log "settings Qwen3.8 MTP opt-in"
    start_persona settings-mtp FAKE_SETTINGS_MTP=1
    dismiss_first_run
    open_settings
    wait_settings_stable "$OUT/settings-root.json"
    press "$OUT/settings-root.json" Settings.Category.performance "$OUT/performance-open-press.json"
    wait_identifier Settings.Performance.SpeculativeDecoding.Enabled "$OUT/performance-mtp-off.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Performance.ModelPicker"
                    and .value == "qwen3.8-27b-4bit")' \
        "$OUT/performance-mtp-off.json" >/dev/null \
        || die "Performance did not select the cached Qwen3.8 MTP fixture"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Performance.SpeculativeDecoding.Enabled"
                    and .enabled == true and .value == 0)' \
        "$OUT/performance-mtp-off.json" >/dev/null \
        || die "Qwen3.8 MTP switch was missing, disabled, or defaulted on"
    press "$OUT/performance-mtp-off.json" Settings.Performance.SpeculativeDecoding.Enabled \
        "$OUT/performance-mtp-press.json"
    wait_settings_stable "$OUT/performance-mtp-on.json" Settings.Performance.SpeculativeDecoding.Enabled
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Performance.SpeculativeDecoding.Enabled" and .value == 1)' \
        "$OUT/performance-mtp-on.json" >/dev/null \
        || die "Qwen3.8 MTP switch did not stay enabled after pressing"
    baseline settings-mtp.enabled "$OUT/performance-mtp-on.json"
    cleanup_persona

    # Every chat model gets the same stable control surface. An alias without
    # an audited registry preset must fail closed instead of hiding the row or
    # letting the user enable an unverified combination.
    start_persona settings-spec-unsupported
    dismiss_first_run
    open_settings
    wait_settings_stable "$OUT/settings-unsupported-root.json"
    press "$OUT/settings-unsupported-root.json" Settings.Category.performance \
        "$OUT/performance-unsupported-open-press.json"
    wait_identifier Settings.Performance.SpeculativeDecoding.Enabled \
        "$OUT/performance-spec-unsupported.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Performance.ModelPicker"
                    and .value == "fake-alias")' \
        "$OUT/performance-spec-unsupported.json" >/dev/null \
        || die "Performance did not select the unsupported chat fixture"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Performance.SpeculativeDecoding.Enabled"
                    and .enabled == false and .value == 0)' \
        "$OUT/performance-spec-unsupported.json" >/dev/null \
        || die "Unsupported alias did not expose a disabled speculative control"
    baseline settings-mtp.unsupported "$OUT/performance-spec-unsupported.json"
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

    # Conversation folders are a visible sidebar workflow, not just a data
    # model. Exercise the real row-menu path that creates a folder and files
    # this conversation in one intention.
    local conversation_menu_id
    conversation_menu_id="Sidebar.Conversation.Menu.${conversation_id##*.}"
    press "$OUT/chat-restored-transcript.json" "$conversation_menu_id" \
        "$OUT/folder-row-menu.json"
    wait_identifier Sidebar.Conversation.Action.MoveToNewFolder "$OUT/folder-menu.json"
    press "$OUT/folder-menu.json" Sidebar.Conversation.Action.MoveToNewFolder \
        "$OUT/folder-new-press.json"
    wait_identifier Sidebar.Folder.NameField "$OUT/folder-prompt.json"
    "$AX_DRIVER" set-value "$APP_PID" Sidebar.Folder.NameField "Golden Work" \
        > "$OUT/folder-name.json"
    # AXValue changes reach the native field before SwiftUI has necessarily
    # propagated the binding and rebuilt the alert action as enabled. Pressing
    # during that gap is a real disabled-button interaction, not a transient
    # stale-element failure, so wait for the observable enabled state first.
    wait_identifier_enabled Sidebar.Folder.Prompt.Confirm "$OUT/folder-name-set.json"
    press "$OUT/folder-name-set.json" Sidebar.Folder.Prompt.Confirm \
        "$OUT/folder-save.json"
    wait_identifier Sidebar.Folder.Toggle.Golden-Work "$OUT/folder-created.json"
    for _ in {1..40}; do
        see_main "$OUT/folder-filed.json"
        if jq -e '(.data.ui_elements | tostring) | contains("Golden Work (1)")' \
            "$OUT/folder-filed.json" >/dev/null; then break; fi
        sleep 0.1
    done
    assert_tree_text "$OUT/folder-filed.json" "Golden Work (1)"
    assert_tree_text "$OUT/folder-filed.json" "golden restore marker"

    # The renderer/encoder are unit-tested; this GUI assertion owns the other
    # half of the feature contract: the export action is reachable from a real
    # conversation row and presents the native save surface.
    press "$OUT/folder-filed.json" "$conversation_menu_id" "$OUT/export-row-menu.json"
    wait_identifier Sidebar.Conversation.Action.Export.Markdown "$OUT/export-menu.json"
    press "$OUT/export-menu.json" Sidebar.Conversation.Action.Export.Markdown \
        "$OUT/export-markdown.json"
    local export_panel_visible=0
    for _ in {1..40}; do
        if ax_window_present "Export Conversation" "$OUT/export-panel.json"; then
            export_panel_visible=1; break
        fi
        sleep 0.1
    done
    [[ "$export_panel_visible" == 1 ]] \
        || die "Markdown export did not present its save panel"
    # #2050: the save panel's window object publishes neither AXCloseButton
    # nor AXCancelButton, so `close-window` can never dismiss it. Its
    # content IS bridged into the app's AX tree, and AppKit gives the
    # cancel affordance a stable identifier — press that instead.
    press "$OUT/export-panel.json" CancelButton "$OUT/export-panel-close.json" \
        || die "export save panel could not be cancelled"

    local conversation_suffix pin_id unpin_id
    conversation_suffix="${conversation_id##*.}"
    pin_id="Sidebar.Conversation.Pin.$conversation_suffix"
    unpin_id="Sidebar.Conversation.Unpin.$conversation_suffix"
    if jq -e --arg identifier "$pin_id" \
        '.data.ui_elements[]? | select(.identifier == $identifier)' \
        "$OUT/chat-restored-transcript.json" >/dev/null; then
        press "$OUT/chat-restored-transcript.json" "$pin_id" "$OUT/pin-press.json" \
            || die "Pin conversation is not pressable"
        wait_identifier "$unpin_id" "$OUT/pinned.json"
        press "$OUT/pinned.json" "$unpin_id" "$OUT/unpin-press.json" \
            || die "Unpin conversation is not pressable"
        wait_identifier "$pin_id" "$OUT/unpinned.json"
    else
        jq -e --arg identifier "$unpin_id" \
            '.data.ui_elements[]? | select(.identifier == $identifier)' \
            "$OUT/chat-restored-transcript.json" >/dev/null \
            || die "active conversation exposes neither Pin nor Unpin"
        press "$OUT/chat-restored-transcript.json" "$unpin_id" "$OUT/unpin-first-press.json"
        wait_identifier "$pin_id" "$OUT/unpinned-first.json"
        press "$OUT/unpinned-first.json" "$pin_id" "$OUT/pin-restore-press.json"
        wait_identifier "$unpin_id" "$OUT/pinned-restored.json"
    fi
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
    press "$OUT/select-text-sheet.json" SelectText.Done "$OUT/select-text-done.json" \
        || die "Select text Done button is not pressable"

    wait_identifier Toolbar.SearchChats "$OUT/search-actions-ready.json"
    press "$OUT/search-actions-ready.json" Toolbar.SearchChats "$OUT/search-actions-open-press.json"
    wait_identifier ConversationSearch.Field "$OUT/search-actions-open.json"
    "$AX_DRIVER" set-value "$APP_PID" ConversationSearch.Field "golden" \
        > "$OUT/search-actions-type.json"
    wait_identifier ConversationSearch.Clear "$OUT/search-actions-filtered.json"
    press "$OUT/search-actions-filtered.json" ConversationSearch.Clear \
        "$OUT/search-actions-clear-press.json" \
        || die "conversation search Clear is not pressable"
    see_main "$OUT/search-actions-cleared.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ConversationSearch.Field" and .value == "")' \
        "$OUT/search-actions-cleared.json" >/dev/null \
        || die "conversation search Clear did not empty the query"
    press "$OUT/search-actions-cleared.json" ConversationSearch.Close \
        "$OUT/search-actions-close-press.json" \
        || die "conversation search Close is not pressable"
    wait_identifier Toolbar.SearchChats "$OUT/search-actions-closed.json"

    press "$OUT/search-actions-closed.json" Toolbar.SearchChats \
        "$OUT/search-new-chat-open-press.json"
    wait_identifier ConversationSearch.NewChat "$OUT/search-new-chat-open.json"
    press "$OUT/search-new-chat-open.json" ConversationSearch.NewChat \
        "$OUT/search-new-chat-press.json" \
        || die "conversation search New chat is not pressable"
    wait_identifier Sidebar.NewChat "$OUT/search-new-chat-landed.json"
    jq -e '[.data.ui_elements[]? | select(.identifier == "ConversationSearch.Field")]
           | length == 0' "$OUT/search-new-chat-landed.json" >/dev/null \
        || die "New chat did not dismiss conversation search"
    log "  conversation Pin/Unpin and search Clear/Close/New chat all produced effects"
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
    start_persona low-memory-choice \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"

    local tree="$OUT/onboarding.json"
    wait_identifier Quickstart.GetStarted "$OUT/welcome.json"
    press "$OUT/welcome.json" Quickstart.GetStarted "$OUT/get-started.json"
    wait_identifier Quickstart.Choice.lfm2.5-1b-4bit "$OUT/model-choices.json"

    local fallback_label
    fallback_label="$(element_field "$OUT/model-choices.json" Quickstart.Choice.lfm2.5-1b-4bit description)"
    [[ "$fallback_label" == *"Lowest memory"* ]] \
        || die "low-memory choice is missing its spoken category label"
    [[ "$fallback_label" == *"less accurate"* ]] \
        || die "low-memory choice hides its quality trade-off"
    [[ "$fallback_label" == *"not recommended for tools"* ]] \
        || die "low-memory choice hides its tool-use limitation"
    press "$OUT/model-choices.json" Quickstart.Choice.lfm2.5-1b-4bit "$OUT/select-low-memory.json"
    see_main "$OUT/low-memory-selected.json"
    jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Choice.lfm2.5-1b-4bit")' \
        "$OUT/low-memory-selected.json" >/dev/null \
        || die "selecting the low-memory choice dismissed or replaced the chooser"
    jq -e '.data.ui_elements[]? | select(.identifier == "Quickstart.Footer.Primary")' \
        "$OUT/low-memory-selected.json" >/dev/null \
        || die "selecting the low-memory choice left no Download & start action"
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
    # One baseline PER STATE, for the same reason the wait loop above accepts
    # both. The two panels are genuinely different trees — AheadOfManifest has
    # no ``Settings.App.RecheckCTA`` and no re-check copy beside it — so a
    # single baseline can only pin one of them, and the release window then
    # fails the flow on a UI that is behaving correctly. Collapsing the region
    # instead (the ``Footer.DesktopVersionPill`` treatment in ax-baseline.py)
    # would hide a real regression in whichever panel is not being exercised.
    baseline "update-state.app-panel.${state##*.}" "$OUT/update-app-panel.json"
    log "  update state names the running version ($expected, via ${state##*.})"
    cleanup_persona
}

# A signed release can discover the new version through Rapid's lightweight
# manifest while Sparkle is already fetching that same update in the
# background. Sparkle rejects a second foreground check in this state. The UI
# must expose the real busy state instead of leaving an enabled orange button
# whose click is silently ignored.
flow_update_busy() {
    start_persona update-busy \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_GUI_UPDATE_BUSY_FIXTURE=1
    dismiss_first_run
    open_settings
    see_main "$OUT/settings.json"
    press "$OUT/settings.json" Settings.Category.app "$OUT/open-app.json"
    wait_identifier Settings.App.UpdateBusy "$OUT/app-panel.json"

    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.App.UpdateBusy"
                    and (.description | contains("Update in progress")))' \
        "$OUT/app-panel.json" >/dev/null \
        || die "background Sparkle session has no visible progress feedback"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Settings.App.UpdateCTA")' \
             "$OUT/app-panel.json" >/dev/null; then
        die "background Sparkle session still exposes the no-op update CTA"
    fi
    baseline "update-busy.app-panel" "$OUT/app-panel.json"
    log "  background update replaces the no-op CTA with truthful progress"
    cleanup_persona
}

flow_campaign_banner() {
    # This flow owns campaign state, not the live release channel. Keep the
    # footer deterministic so a newly published app version cannot invalidate
    # the campaign's structural baseline.
    start_persona campaign-banner \
        RAPID_GUI_CAMPAIGN_PREVIEW=1 RAPIDMLX_NO_UPDATE_CHECK=1
    dismiss_first_run
    wait_identifier_enabled Campaign.Action "$OUT/campaign-visible.json"
    assert_tree_text "$OUT/campaign-visible.json" "Qwen3.5 35B is ready"
    jq -e '.data.ui_elements[]? | select(.identifier == "Campaign.Action" and .enabled == true)' \
        "$OUT/campaign-visible.json" >/dev/null \
        || die "campaign CTA is absent or disabled"
    baseline campaign-banner.visible "$OUT/campaign-visible.json"
    # Race two genuine AX activations against SwiftUI's disabled-state update.
    # DownloadManager must coalesce them to one pull even if both actions enter.
    "$AX_DRIVER" press "$APP_PID" Campaign.Action > "$OUT/campaign-action-1.json" 2>/dev/null &
    local action_pid_1=$!
    "$AX_DRIVER" press "$APP_PID" Campaign.Action > "$OUT/campaign-action-2.json" 2>/dev/null &
    local action_pid_2=$!
    wait "$action_pid_1" || true
    wait "$action_pid_2" || true
    jq -s -e 'any(.[]; .success == true)' \
        "$OUT/campaign-action-1.json" "$OUT/campaign-action-2.json" >/dev/null \
        || die "neither rapid campaign activation reached AXPress"
    wait_fake_event \
        '.event == "command" and .subcommand == "pull" and .alias == "qwen3.5-35b-4bit"' \
        "campaign CTA did not start the allowlisted model pull"
    [[ "$(jq -s '[.[] | select(.event == "command" and .subcommand == "pull" and .alias == "qwen3.5-35b-4bit")] | length' "$OUT/fake-events.jsonl")" == 1 ]] \
        || die "campaign CTA started the model pull more than once"
    wait_identifier Campaign.Dismiss "$OUT/campaign-after-action.json"
    press "$OUT/campaign-after-action.json" Campaign.Dismiss "$OUT/campaign-dismiss.json"
    for _ in {1..40}; do
        see_main "$OUT/campaign-dismissed.json"
        if ! jq -e '.data.ui_elements[]? | select(.identifier == "Campaign.Banner")' \
            "$OUT/campaign-dismissed.json" >/dev/null; then
            break
        fi
        sleep 0.25
    done
    ! jq -e '.data.ui_elements[]? | select(.identifier == "Campaign.Banner")' \
        "$OUT/campaign-dismissed.json" >/dev/null \
        || die "campaign remained visible after dismissal"

    relaunch_persona
    dismiss_first_run
    see_main "$OUT/campaign-relaunched.json"
    ! jq -e '.data.ui_elements[]? | select(.identifier == "Campaign.Banner")' \
        "$OUT/campaign-relaunched.json" >/dev/null \
        || die "dismissed campaign returned after relaunch"
    log "  campaign banner renders one typed CTA and remembers dismissal"
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
    local settings_categories=(modelManagement instructions tools connectors performance appearance privacy app)
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Settings.Category.developer")' \
        "$OUT/dead-before.json" >/dev/null; then
        settings_categories+=(developer)
    fi
    for category in "${settings_categories[@]}"; do
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
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Settings.Privacy.Link.MTPLX")' \
        "$OUT/dead-privacy-before.json" >/dev/null \
        || die "MTPLX attribution link is missing from Settings → Privacy"
    [[ "$(element_field "$OUT/dead-privacy-before.json" Settings.Privacy.Link.MTPLX enabled)" == "true" ]] \
        || die "MTPLX attribution link is disabled"
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

    # Dogfood every reversible Settings control, not merely its presence.
    # Each toggle must change and then restore its persisted value. Segmented
    # controls must publish the selection they accepted. Disclosure buttons
    # must add and then remove their body. This keeps an AXPress that only
    # produces a hover/highlight from masquerading as working UI.
    see_main "$OUT/dead-actions-start.json"
    press "$OUT/dead-actions-start.json" Settings.Category.modelManagement \
        "$OUT/dead-actions-models-open.json"
    round_trip_toggle Settings.Models.ShowAllModelsToggle dead-actions-show-all
    round_trip_toggle Settings.Models.AutoStartOnLaunchToggle dead-actions-auto-start

    see_main "$OUT/dead-actions-models-favorite-before.json"
    local favorite_before favorite_after favorite_restored
    favorite_before="$(element_field "$OUT/dead-actions-models-favorite-before.json" \
        Settings.ModelManagement.Favorite.fake-alias description)"
    press "$OUT/dead-actions-models-favorite-before.json" \
        Settings.ModelManagement.Favorite.fake-alias \
        "$OUT/dead-actions-models-favorite-press.json"
    see_main "$OUT/dead-actions-models-favorite-after.json"
    favorite_after="$(element_field "$OUT/dead-actions-models-favorite-after.json" \
        Settings.ModelManagement.Favorite.fake-alias description)"
    [[ -n "$favorite_after" && "$favorite_after" != "$favorite_before" ]] \
        || die "favorite button accepted AXPress but did not change Pin/Unpin state"
    press "$OUT/dead-actions-models-favorite-after.json" \
        Settings.ModelManagement.Favorite.fake-alias \
        "$OUT/dead-actions-models-favorite-restore-press.json"
    see_main "$OUT/dead-actions-models-favorite-restored.json"
    favorite_restored="$(element_field "$OUT/dead-actions-models-favorite-restored.json" \
        Settings.ModelManagement.Favorite.fake-alias description)"
    [[ "$favorite_restored" == "$favorite_before" ]] \
        || die "favorite button did not restore its original Pin/Unpin state"

    press "$OUT/dead-actions-models-favorite-restored.json" Settings.Category.tools \
        "$OUT/dead-actions-tools-open.json"
    # The selected backend owns the visible key row. Opening Tools must resolve
    # its cached Keychain state on its own; making users re-select the already
    # selected radio or focus the secret field is the regression from #2462.
    local key_status=""
    for _ in {1..40}; do
        see_main "$OUT/dead-actions-tools-key-status.json"
        key_status="$(element_field \
            "$OUT/dead-actions-tools-key-status.json" \
            Settings.Tools.WebSearch.KeyStatus.keenable value)"
        [[ -n "$key_status" && "$key_status" != "Checking saved key…" ]] && break
        sleep 0.1
    done
    [[ -n "$key_status" && "$key_status" != "Checking saved key…" ]] \
        || die "Tools left the active backend's saved-key status unresolved"
    log "  active web-search backend resolves its saved-key status on appearance"
    local tool_name
    for tool_name in web_search browse weather; do
        round_trip_toggle "Settings.Tools.Toggle.$tool_name" "dead-actions-tool-$tool_name"
        see_main "$OUT/dead-actions-details-$tool_name-before.json"
        press "$OUT/dead-actions-details-$tool_name-before.json" \
            "Settings.Tools.Details.$tool_name" \
            "$OUT/dead-actions-details-$tool_name-press.json"
        see_main "$OUT/dead-actions-details-$tool_name-open.json"
        jq -e --arg identifier "Settings.Tools.DetailsBody.$tool_name" \
            '.data.ui_elements[]? | select(.identifier == $identifier)' \
            "$OUT/dead-actions-details-$tool_name-open.json" >/dev/null \
            || die "Details for $tool_name pressed but revealed no body"
        press "$OUT/dead-actions-details-$tool_name-open.json" \
            "Settings.Tools.Details.$tool_name" \
            "$OUT/dead-actions-details-$tool_name-close-press.json"
        see_main "$OUT/dead-actions-details-$tool_name-closed.json"
        jq -e --arg identifier "Settings.Tools.DetailsBody.$tool_name" \
            '[.data.ui_elements[]? | select(.identifier == $identifier)] | length == 0' \
            "$OUT/dead-actions-details-$tool_name-closed.json" >/dev/null \
            || die "Details for $tool_name did not collapse"
    done
    round_trip_toggle Settings.Tools.Browse.AutoApproveToggle dead-actions-auto-approve
    press_and_require_selected Settings.Tools.WebSearch.Backend.brave dead-actions-backend-brave
    press_and_require_selected Settings.Tools.WebSearch.Backend.tavily dead-actions-backend-tavily
    press_and_require_selected Settings.Tools.WebSearch.Backend.duckduckgo dead-actions-backend-restore

    see_main "$OUT/dead-actions-tools-done.json"
    press "$OUT/dead-actions-tools-done.json" Settings.Category.connectors \
        "$OUT/dead-actions-connectors-open.json"
    round_trip_toggle Settings.Connectors.MasterToggle dead-actions-connectors-master

    see_main "$OUT/dead-actions-connectors-done.json"
    press "$OUT/dead-actions-connectors-done.json" Settings.Category.performance \
        "$OUT/dead-actions-performance-open.json"
    press_and_require_selected Settings.Performance.Prefix.On dead-actions-prefix-on
    press_and_require_selected Settings.Performance.Prefix.Off dead-actions-prefix-off
    press_and_require_selected Settings.Performance.Prefix.Default dead-actions-prefix-default
    see_main "$OUT/dead-actions-cache-before.json"
    local cache_before cache_after
    cache_before="$(element_field "$OUT/dead-actions-cache-before.json" \
        Settings.Performance.CacheBudget value)"
    "$AX_DRIVER" increment "$APP_PID" Settings.Performance.CacheBudget \
        > "$OUT/dead-actions-cache-increment.json" \
        || die "Cache budget slider rejected its native AXIncrement action"
    see_main "$OUT/dead-actions-cache-after.json"
    cache_after="$(element_field "$OUT/dead-actions-cache-after.json" \
        Settings.Performance.CacheBudget value)"
    [[ -n "$cache_after" && "$cache_after" != "$cache_before" ]] \
        || die "Cache budget slider accepted AXIncrement but stayed at $cache_after"
    "$AX_DRIVER" decrement "$APP_PID" Settings.Performance.CacheBudget \
        > "$OUT/dead-actions-cache-decrement.json" \
        || die "Cache budget slider rejected its native AXDecrement action"
    see_main "$OUT/dead-actions-cache-restored.json"
    [[ "$(element_field "$OUT/dead-actions-cache-restored.json" \
            Settings.Performance.CacheBudget value)" == "$cache_before" ]] \
        || die "Cache budget slider did not restore $cache_before"

    see_main "$OUT/dead-actions-performance-done.json"
    press "$OUT/dead-actions-performance-done.json" Settings.Category.app \
        "$OUT/dead-actions-app-open.json"
    round_trip_toggle Settings.App.HideDockOnCloseToggle dead-actions-hide-dock
    see_main "$OUT/dead-actions-app-before-recheck.json"
    # The App panel is a per-STATE tree (see the update-state flow above):
    # ``AheadOfManifest`` — the build is newer than anything published — has NO
    # ``Settings.App.RecheckCTA`` at all. That is precisely the state every
    # version-bump PR builds into (app X.Y.Z+1, manifest X.Y.Z), so an
    # unconditional press here failed 2 of 2 runs that reached this flow on the
    # 0.12.15 bump while passing on every same-version PR. Mirror update-state:
    # exercise the CTA when the panel renders it, and in AheadOfManifest assert
    # the state marker instead — a dead control in UpToDate still dies, and a
    # correctly-absent control no longer reads as one.
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Settings.App.RecheckCTA")' \
        "$OUT/dead-actions-app-before-recheck.json" >/dev/null; then
        press "$OUT/dead-actions-app-before-recheck.json" Settings.App.RecheckCTA \
            "$OUT/dead-actions-app-recheck-press.json" \
            || die "Check for updates is not pressable"
        local update_feedback=0
        for ((i=0; i<40; i++)); do
            see_main "$OUT/dead-actions-app-checked.json"
            if jq -e '(.data.ui_elements | tostring)
                      | contains("Checking for updates") or contains("Up to date")' \
                "$OUT/dead-actions-app-checked.json" >/dev/null; then
                update_feedback=1
                break
            fi
            sleep 0.25
        done
        [[ "$update_feedback" == 1 ]] \
            || die "Check for updates produced no visible state"
    else
        jq -e '.data.ui_elements[]?
               | select(.identifier == "Settings.App.AheadOfManifest")' \
            "$OUT/dead-actions-app-before-recheck.json" >/dev/null \
            || die "Settings > App shows neither Settings.App.RecheckCTA nor Settings.App.AheadOfManifest"
    fi

    # Developer exists only in debug builds. Its destructive reset is unit
    # tested separately; here the GUI contract is that every scope toggle
    # round-trips and the action opens a cancellable confirmation rather than
    # erasing immediately.
    see_main "$OUT/dead-actions-after-update.json"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Settings.Category.developer")' \
        "$OUT/dead-actions-after-update.json" >/dev/null; then
        press "$OUT/dead-actions-after-update.json" Settings.Category.developer \
            "$OUT/dead-actions-developer-open.json"
        round_trip_toggle Settings.Developer.Scope.Preferences dead-actions-developer-preferences
        round_trip_toggle Settings.Developer.Scope.Conversations dead-actions-developer-conversations
        round_trip_toggle Settings.Developer.Scope.Telemetry dead-actions-developer-telemetry
        see_main "$OUT/dead-actions-developer-before-confirm.json"
        press "$OUT/dead-actions-developer-before-confirm.json" Settings.Developer.Reonboard \
            "$OUT/dead-actions-developer-dialog-press.json" \
            || die "Erase and restart is not pressable"
        wait_identifier Settings.Developer.CancelReonboard \
            "$OUT/dead-actions-developer-dialog.json"
        local dialog_closed=0
        for ((i=0; i<40; i++)); do
            # SwiftUI can replace a confirmation-dialog AX element between a
            # tree dump and AXPress. Re-resolve and retry the semantic action;
            # the observable contract is that the dialog disappears.
            "$AX_DRIVER" press "$APP_PID" Settings.Developer.CancelReonboard \
                > "$OUT/dead-actions-developer-cancel.json" 2>/dev/null || true
            sleep 0.1
            see_main "$OUT/dead-actions-developer-cancelled.json"
            if ! jq -e '.data.ui_elements[]?
                        | select(.identifier == "Settings.Developer.CancelReonboard")' \
                "$OUT/dead-actions-developer-cancelled.json" >/dev/null; then
                dialog_closed=1; break
            fi
            sleep 0.25
        done
        [[ "$dialog_closed" == 1 ]] \
            || die "re-onboarding confirmation stayed open after Cancel"
    fi
    log "  reversible Settings controls all changed state and restored"
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
    start_persona browse-all-destination \
        RAPID_GUI_HARDWARE_FIXTURE=1 RAPID_HARDWARE_RAM_GB=$GOLDEN_RAM_GB \
        RAPID_HARDWARE_BRAND="$GOLDEN_BRAND"

    # The wizard has to stay up; it is the subject of this flow.
    local tree="$OUT/ba-first-run.json"
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
                          | select((.identifier // "") as $id
                            | ($id | startswith("Quickstart.Choice."))
                              or ($id | startswith("Quickstart.CachedModel."))
                              or ($id | startswith("Quickstart.YourPick.")))]
                         | (map(select(.selected == true))) as $default
                         | if ($default | length) != 1 then empty
                           else (map(select(.identifier != $default[0].identifier
                                    and ((.identifier // "") | startswith("Quickstart.Choice."))))[0].identifier // empty)
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

    # The composer's single plus affordance expands to exactly the two product
    # actions: documents remain available for every chat model, while photos
    # stay visible-but-disabled for this text-only fixture alias. Keeping the
    # disabled row visible teaches the capability boundary without pretending
    # Rapid itself lacks image input.
    press "$OUT/document-compose.json" ChatView.AddAttachments \
        "$OUT/attachment-menu-open.json"
    local attachment_menu_ready=0
    for _ in {1..40}; do
        see_main "$OUT/attachment-menu.json"
        if jq -e '[.data.ui_elements[]?
                   | select(.identifier == "ChatView.Attachments.UploadFile"
                            or .identifier == "ChatView.Attachments.UploadPhoto")]
                  | length == 2' "$OUT/attachment-menu.json" >/dev/null; then
            attachment_menu_ready=1; break
        fi
        sleep 0.1
    done
    [[ "$attachment_menu_ready" == 1 ]] \
        || die "the attachment plus menu did not expose Upload file and Upload photo"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ChatView.Attachments.UploadFile" and .enabled == true)' \
        "$OUT/attachment-menu.json" >/dev/null \
        || die "Upload file was not enabled for a text-only chat model"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ChatView.Attachments.UploadPhoto" and .enabled == false)' \
        "$OUT/attachment-menu.json" >/dev/null \
        || die "Upload photo did not expose the text-only model capability boundary"

    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$fixture" \
        > "$OUT/document-paste.json"

    wait_identifier ChatView.Attachment.Remove.chat-document.txt \
        "$OUT/document-attached.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ChatView.Attachment.Remove.chat-document.txt")' \
        "$OUT/document-attached.json" >/dev/null \
        || die "the pasted TXT file did not become a removable attachment chip"

    # Paste the exact same path again through the real clipboard gesture. It
    # must produce feedback without adding a second chip — otherwise the
    # shared document budget is split across duplicate copies.
    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$fixture" \
        > "$OUT/document-duplicate-paste.json"
    local duplicate_rejected=0
    for _ in {1..40}; do
        see_main "$OUT/document-duplicate-rejected.json"
        if jq -e '([.data.ui_elements[]?
                    | select(.identifier == "ChatView.Attachment.Remove.chat-document.txt")]
                   | length) == 1
                  and ((.data.ui_elements | tostring)
                       | contains("That file is already attached."))' \
            "$OUT/document-duplicate-rejected.json" >/dev/null; then
            duplicate_rejected=1; break
        fi
        sleep 0.1
    done
    [[ "$duplicate_rejected" == 1 ]] \
        || die "pasting the same document twice did not leave exactly one chip with duplicate feedback"

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

flow_chat_multimodal_attachments() {
    start_persona chat-multimodal-attachments FAKE_VISION_CHAT=1
    dismiss_first_run
    start_model

    local first="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-28.png"
    local second="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-96.png"
    local document="$ROOT/Tests/GUIGoldenFlows/Fixtures/chat-document.txt"

    see_main "$OUT/mm-compose.json"
    press "$OUT/mm-compose.json" ChatView.AddAttachments "$OUT/mm-menu-open.json"
    wait_identifier ChatView.Attachments.UploadPhoto "$OUT/mm-menu.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ChatView.Attachments.UploadPhoto" and .enabled == true)' \
        "$OUT/mm-menu.json" >/dev/null \
        || die "Upload photo was not enabled for a vision chat model"
    # Dismiss the popover by pressing its anchor again before exercising the
    # native paste ingress. The two picker actions are source-wired to the same
    # importer and pinned by AttachmentDedupTests; this flow proves the mounted
    # menu exposes the correct enabled action.
    press "$OUT/mm-menu.json" ChatView.AddAttachments "$OUT/mm-menu-close.json"

    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$first" > "$OUT/mm-first-paste.json"
    wait_identifier ChatView.Attachment.Remove.cheetah-logo-28.png "$OUT/mm-first-attached.json"
    send_prompt "Describe current image one" mm-first
    wait_send_idle "$OUT/mm-first-complete.json"

    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$second" > "$OUT/mm-second-paste.json"
    wait_identifier ChatView.Attachment.Remove.cheetah-logo-96.png "$OUT/mm-second-attached.json"
    send_prompt "Describe current image two" mm-second
    wait_send_idle "$OUT/mm-second-complete.json"

    local first_hash second_hash
    first_hash="$(python3 - "$first" <<'PY'
import base64, hashlib, pathlib, sys
url = "data:image/png;base64," + base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
print(hashlib.sha256(url.encode()).hexdigest())
PY
)"
    second_hash="$(python3 - "$second" <<'PY'
import base64, hashlib, pathlib, sys
url = "data:image/png;base64," + base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
print(hashlib.sha256(url.encode()).hexdigest())
PY
)"
    jq -e -s --arg first "$first_hash" --arg second "$second_hash" '
        [ .[] | select(.event == "chat_request"
              and .request_origin != "background_assist") ][1]
        | .user_payloads[-1].image_url_sha256 == [$second]
          and ([.user_payloads[]?.image_url_sha256[]?] | index($first) | not)
    ' "$OUT/fake-events.jsonl" >/dev/null \
        || die "the second image request resent the first image or omitted the second"

    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$document" > "$OUT/mm-document-paste.json"
    wait_identifier ChatView.Attachment.Remove.chat-document.txt "$OUT/mm-document-attached.json"
    send_prompt "Review current document" mm-document
    wait_send_idle "$OUT/mm-document-complete.json"
    jq -e -s '
        [ .[] | select(.event == "chat_request"
              and .request_origin != "background_assist") ][2]
        | ([.user_payloads[]?.image_url_sha256[]?] | length) == 0
          and (.user_payloads[-1].text
               | contains("BEGIN RAPID ATTACHMENT")
                 and contains("Revenue: 42")
                 and contains("Region: APAC"))
    ' "$OUT/fake-events.jsonl" >/dev/null \
        || die "the document request retained historical images or omitted extracted text"

    cleanup_persona
    flow_localized_photo_hint
}

flow_localized_photo_hint() {
    start_persona localized-photo-hint \
        RAPID_GUI_APP_LANGUAGE=zh-Hans \
        FAKE_SERVING_LANE_REASON=vision_memory_insufficient
    dismiss_first_run
    start_model

    local expected="此模型的文字聊天可以正常使用。照片模式需要的内存超过这台 Mac 的容量；如需添加照片，请选择内存需求更低的视觉模型。"
    local localized=0
    for _ in {1..40}; do
        see_main "$OUT/zh-main.json"
        press "$OUT/zh-main.json" ChatView.AddAttachments "$OUT/zh-menu-press.json"
        wait_identifier ChatView.Attachments.UploadPhoto "$OUT/zh-menu.json"
        if jq -e --arg expected "$expected" '
            .data.ui_elements[]?
            | select(.identifier == "ChatView.Attachments.UploadPhoto"
                     and .enabled == false
                     and .help == $expected)
        ' "$OUT/zh-menu.json" >/dev/null; then
            localized=1
            break
        fi
        press "$OUT/zh-menu.json" ChatView.AddAttachments "$OUT/zh-menu-close.json"
        sleep 0.25
    done
    [[ "$localized" == 1 ]] \
        || die "the disabled photo action never exposed its zh-Hans memory remedy"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ContentView.Settings" and .description == "设置")' \
        "$OUT/zh-menu.json" >/dev/null \
        || die "the release-shaped app did not load its compiled zh-Hans resources"

    # The disabled menu row explains the capability before an attempt; paste
    # proves the mounted chat surface treats the same fact as a transient
    # capability notice while keeping text chat usable.
    press "$OUT/zh-menu.json" ChatView.AddAttachments "$OUT/zh-menu-close.json"
    local fixture="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-28.png"
    "$AX_DRIVER" paste-file "$APP_PID" rapid.chat.compose "$fixture" \
        > "$OUT/zh-photo-paste.json"
    wait_tree_text "$expected" "$OUT/zh-photo-notice.json" 40
    send_prompt "Text chat remains available" zh-text-after-photo
    wait_send_idle "$OUT/zh-text-after-photo-complete.json"
    assert_tree_text "$OUT/zh-text-after-photo-complete.json" "Text chat remains available"
    assert_tree_text "$OUT/zh-text-after-photo-complete.json" "deterministic content"
    jq -e --arg expected "$expected" '
        [.data.ui_elements[]?
         | select([(.title // ""), (.value // ""), (.description // "")]
                  | join(" ") | contains($expected))]
        | length == 0
    ' "$OUT/zh-text-after-photo-complete.json" >/dev/null \
        || die "the photo capability notice persisted after a successful text turn"

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
        FAKE_IMAGE_STEP_MS_SEQUENCE=5000,300,5000,300,300,300,300,300 \
        FAKE_IMAGE_FIRST_WARMUP_ACK="$OUT_ROOT/image-generation/ig-warmup-ack" \
        FAKE_IMAGE_STEP_HOLD_ACK="$OUT_ROOT/image-generation/ig-eta-hold-ack" \
        FAKE_IMAGE_FINISH_MS=15000 \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_SIMULATED_IMPORT_PATH="$ROOT/Tests/RapidTests/__Snapshots__/cheetah-logo-96.png"

    # Existing renders pass step 2 immediately. Section 9 removes this file
    # for one request so its two ETA reads are event-gated, not timing-gated.
    : > "$OUT/ig-eta-hold-ack"

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
    # The picker and readiness banner are fed by separate async state. A
    # resolved picker does not guarantee that the banner has consumed the
    # same catalog refresh yet; snapshotting here used to race and capture
    # "No model chosen" on busy CI runners. Wait for the cached model's Start
    # action so the baseline describes one coherent state.
    wait_identifier Readiness.Action "$OUT/ig-empty.json"
    jq -e '[.data.ui_elements[]? | select((.identifier // "")
                                           | startswith("Images.Aspect."))]
           | length == 3' "$OUT/ig-empty.json" >/dev/null \
        || die "Images aspect options are not independently addressable"
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Resolution")' "$OUT/ig-empty.json" >/dev/null \
        || die "Images.Resolution is missing — no way to choose an output resolution"
    baseline image-generation.empty "$OUT/ig-empty.json"

    local starter_id starter_prompt
    for starter_id in Images.Starter.0 Images.Starter.1 Images.Starter.2 Images.Starter.3; do
        see_main "$OUT/ig-starter-before.json"
        starter_prompt="$(element_field "$OUT/ig-starter-before.json" "$starter_id" description)"
        [[ -n "$starter_prompt" ]] || die "$starter_id has no readable prompt"
        press "$OUT/ig-starter-before.json" "$starter_id" \
            "$OUT/ig-${starter_id##*.}-press.json" \
            || die "$starter_id is not pressable"
        see_main "$OUT/ig-starter-filled.json"
        jq -e --arg prompt "$starter_prompt" \
            '.data.ui_elements[]?
             | select(.identifier == "rapid.images.compose" and .value == $prompt)' \
            "$OUT/ig-starter-filled.json" >/dev/null \
            || die "$starter_id did not fill the image prompt"
        "$AX_DRIVER" set-value "$APP_PID" rapid.images.compose "" \
            > "$OUT/ig-starter-clear.json"
        wait_identifier "$starter_id" "$OUT/ig-starter-restored.json"
    done
    press_and_require_selected Images.Aspect.portrait ig-aspect-portrait
    press_and_require_selected Images.Aspect.landscape ig-aspect-landscape
    press_and_require_selected Images.Aspect.square ig-aspect-square
    log "  all starter prompts and aspect buttons changed the composer state"

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

    # Synchronize on the wire accepting the request before asking AX for the
    # preparing card. The fake keeps this real protocol phase open long enough
    # for one potentially expensive tree read; no retry count guesses whether
    # the button action has reached the server yet.
    wait_fake_event '.event == "image_request" and .operation == "generation"' \
        "the image request never reached the sidecar"

    # The in-flight card. Asserted BEFORE the result so a render that returns
    # instantly (or a card that never appears) is a failure rather than a frame
    # nobody looked at.  SwiftUI mounts Cancel one layout pass before the
    # indeterminate indicator the baseline owns; require both so the snapshot
    # cannot race that valid intermediate tree.
    local inflight=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/ig-inflight.json"
        if jq -e 'any(.data.ui_elements[]?; .identifier == "Images.Cancel")
                  and any(.data.ui_elements[]?; .role == "AXBusyIndicator")' \
               "$OUT/ig-inflight.json" >/dev/null; then
            inflight=1; break
        fi
        sleep 0.1
    done
    [[ "$inflight" == 1 ]] \
        || die "no settled in-flight progress card with Cancel and busy indicator"
    : > "$OUT/ig-warmup-ack"
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
    # `512x512` is what the default (square) aspect maps to. A shape-only
    # check like `^[0-9]+x[0-9]+$` passes `768x1024` — the PORTRAIT size — so
    # an aspect control wired to the wrong case would render the wrong shape
    # and the flow would call it correct.
    jq -s -e --arg alias "$FAKE_IMAGE_ALIAS" \
        'all(.[] | select(.event == "image_request");
             .model == $alias and .n == 1 and .size == "512x512")' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "an image request named the wrong model, asked for n != 1, or did not carry the square size 512x512: $(jq -s -c '[.[] | select(.event == "image_request") | {model, size, n}]' "$OUT/fake-events.jsonl")"
    # ...and the UI agreed that square was selected, so the two cannot drift
    # into agreeing on a wrong value together.
    # On `selected`, not on existence. All three ratio buttons carry the
    # identifier `Images.Aspect` and all three are always present, so
    # "there is a 1:1 button" is true no matter which one is active — an
    # assertion that cannot fail. Only the selected flag distinguishes them.
    jq -e '[.data.ui_elements[]? | select(.identifier == "Images.Aspect.square")
            | select(.selected == true)] | length == 1' \
        "$OUT/ig-result-2.json" >/dev/null \
        || die "the selected aspect is not the square one the requests were sent with"

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
                         and ($ids | index("Images.Aspect.square")) != null)' \
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
                         and ($ids | index("Images.Aspect.square")) != null)' \
               "$OUT/ig-import-exited.json" >/dev/null; then
            import_exited=1; break
        fi
        sleep 0.25
    done
    [[ "$import_exited" == 1 ]] \
        || die "exiting edit mode after an import did not restore generation controls"

    # 9. ETA evidence across unchanged samples, cancellation, and restart.
    # The fixture holds one reported step start. Capture the same
    # structured step twice during that hold: the numeric ETA must remain
    # identical even though the HUD's elapsed clock keeps advancing.
    local cancel_prompt="a cheetah render to cancel after ETA appears"
    rm -f "$OUT/ig-eta-hold-ack"
    type_prompt "$cancel_prompt" ig-eta-cancel-draft
    press "$OUT/ig-eta-cancel-draft.json" Images.Generate "$OUT/ig-eta-cancel-submit.json" \
        || die "Images.Generate is not pressable for ETA cancellation evidence"
    wait_fake_event \
        ".event == \"image_request\" and .prompt == \"$cancel_prompt\"" \
        "the ETA cancellation request never reached the sidecar"

    local eta_ready=0 eta_step_a eta_value_a eta_step_b eta_value_b
    for ((i=0; i<80; i++)); do
        see_main "$OUT/ig-eta-sample-a.json"
        eta_step_a="$(element_field "$OUT/ig-eta-sample-a.json" Images.Progress.Step value)"
        eta_value_a="$(element_field "$OUT/ig-eta-sample-a.json" Images.Progress.ETA value)"
        if [[ "$eta_step_a" == "2 / 8" && "$eta_value_a" == *"s left" ]]; then
            eta_ready=1; break
        fi
        sleep 0.1
    done
    [[ "$eta_ready" == 1 ]] \
        || die "the held denoise step never exposed a numeric ETA at step 2 / 8"
    see_main "$OUT/ig-eta-sample-b.json"
    eta_step_b="$(element_field "$OUT/ig-eta-sample-b.json" Images.Progress.Step value)"
    eta_value_b="$(element_field "$OUT/ig-eta-sample-b.json" Images.Progress.ETA value)"
    [[ "$eta_step_b" == "$eta_step_a" ]] \
        || die "the deterministic unchanged-step fixture advanced unexpectedly ($eta_step_a -> $eta_step_b)"
    [[ "$eta_value_b" == "$eta_value_a" ]] \
        || die "ETA changed while reported progress stayed at $eta_step_a ($eta_value_a -> $eta_value_b)"
    press "$OUT/ig-eta-sample-b.json" Images.Generate "$OUT/ig-eta-cancel-press.json" \
        || die "the primary Stop control is not pressable after numeric ETA appears"
    local cancel_requested=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ig-eta-cancel-requested.json"
        if jq -e 'any(.data.ui_elements[]?;
                      .identifier == "Images.Generate" and .enabled == false)
                  and any(.data.ui_elements[]?;
                          .identifier == "Images.Cancel" and .enabled == false)' \
               "$OUT/ig-eta-cancel-requested.json" >/dev/null; then
            cancel_requested=1; break
        fi
        sleep 0.05
    done
    [[ "$cancel_requested" == 1 ]] \
        || die "the primary Stop control did not enter the cancelling state"
    wait_fake_event '.event == "image_cancel"' \
        "the ETA-bearing render did not receive cancellation"
    : > "$OUT/ig-eta-hold-ack"
    local eta_cleared=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/ig-eta-cancelled.json"
        if jq -e '.success == true and .data.walk.complete == true
                  and (([.data.ui_elements[]? | .identifier // ""]
                        | index("Images.Progress.ETA")) == null)
                  and (([.data.ui_elements[]? | .identifier // ""]
                        | index("Images.Cancel")) == null)' \
               "$OUT/ig-eta-cancelled.json" >/dev/null; then
            eta_cleared=1; break
        fi
        sleep 0.1
    done
    [[ "$eta_cleared" == 1 ]] \
        || die "numeric ETA remained visible after cancellation completed"

    # A restarted request begins at a new evidence window. The old numeric ETA
    # must not flash on the new card before two completed samples exist.
    local restart_prompt="a fresh cheetah render after cancellation"
    type_prompt "$restart_prompt" ig-eta-restart-draft
    press "$OUT/ig-eta-restart-draft.json" Images.Generate "$OUT/ig-eta-restart-submit.json" \
        || die "Images.Generate is not pressable after ETA cancellation"
    wait_fake_event \
        ".event == \"image_request\" and .prompt == \"$restart_prompt\"" \
        "the post-cancellation restart never reached the sidecar"
    local restart_estimating=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/ig-eta-restart-estimating.json"
        if [[ "$(element_field "$OUT/ig-eta-restart-estimating.json" Images.Progress.ETA value)" == "Estimating…" ]]; then
            restart_estimating=1; break
        fi
        sleep 0.05
    done
    [[ "$restart_estimating" == 1 ]] \
        || die "a restarted render reused stale numeric ETA instead of estimating from fresh progress"
    press "$OUT/ig-eta-restart-estimating.json" Images.Cancel "$OUT/ig-eta-restart-cancel.json" \
        || die "the restarted ETA fixture could not be cancelled for cleanup"
    wait_fake_event \
        ".event == \"image_response\" and .cancelled == true and .index == 6" \
        "the restarted ETA fixture did not settle as cancelled"

    # 10. Deletion is destructive and session-local, so exercise both doors
    #     through the native confirmation dialog before proving the last
    #     result restores the real empty state.  A source-only contract cannot
    #     tell whether SwiftUI actually hosts either dialog action in AX.
    see_main "$OUT/ig-delete-before.json"
    local gallery_before
    gallery_before="$(jq '[.data.ui_elements[]?
                           | select((.identifier // "")
                                    | startswith("Images.Gallery.Thumb."))]
                          | length' "$OUT/ig-delete-before.json")"
    [[ "$gallery_before" -gt 0 ]] \
        || die "the deletion journey has no generated results to remove"

    press "$OUT/ig-delete-before.json" Images.Result.Delete "$OUT/ig-delete-open.json" \
        || die "the active image has no pressable Delete action"
    wait_identifier Images.Result.Delete.Keep "$OUT/ig-delete-dialog-cancel.json"
    press "$OUT/ig-delete-dialog-cancel.json" Images.Result.Delete.Keep \
        "$OUT/ig-delete-keep.json" \
        || die "the native deletion dialog has no pressable Keep action"
    wait_identifier Images.Result.Delete "$OUT/ig-delete-kept.json"
    local gallery_after_keep
    gallery_after_keep="$(jq '[.data.ui_elements[]?
                              | select((.identifier // "")
                                       | startswith("Images.Gallery.Thumb."))]
                             | length' "$OUT/ig-delete-kept.json")"
    [[ "$gallery_after_keep" == "$gallery_before" ]] \
        || die "Keep removed a generated result ($gallery_before -> $gallery_after_keep)"

    # Remove every result through the same user-visible contract.  The button
    # always targets the active result; ImageGenViewModel chooses the adjacent
    # remaining image until the final confirmation restores Images.EmptyState.
    local remaining="$gallery_after_keep"
    while [[ "$remaining" -gt 0 ]]; do
        see_main "$OUT/ig-delete-${remaining}-before.json"
        press "$OUT/ig-delete-${remaining}-before.json" Images.Result.Delete \
            "$OUT/ig-delete-${remaining}-open.json" \
            || die "Delete stopped being pressable with $remaining result(s) left"
        wait_identifier Images.Result.Delete.Confirm \
            "$OUT/ig-delete-${remaining}-dialog.json"
        press "$OUT/ig-delete-${remaining}-dialog.json" Images.Result.Delete.Confirm \
            "$OUT/ig-delete-${remaining}-confirm.json" \
            || die "the native deletion dialog has no pressable Delete Image action"
        remaining=$((remaining - 1))
        if [[ "$remaining" -gt 0 ]]; then
            wait_identifier Images.Result.Delete "$OUT/ig-delete-${remaining}-settled.json"
        else
            wait_identifier Images.EmptyState "$OUT/ig-delete-empty.json"
        fi
    done
    jq -e '.success == true and .data.walk.complete == true
           and any(.data.ui_elements[]?; .identifier == "Images.EmptyState")
           and ([.data.ui_elements[]?
                 | select((.identifier // "")
                          | startswith("Images.Gallery.Thumb."))]
                | length == 0)' "$OUT/ig-delete-empty.json" >/dev/null \
        || die "deleting the final image did not restore a gallery-free empty state"

    log "  image-generation OK"
}
flow_resident_load_rejected() {
    start_persona resident-load-rejected FAKE_REJECT_IMAGE_SIDECAR=1

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
    jq -e '.data.ui_elements[]? | select(.identifier == "Images.Aspect.square")' "$OUT/rlr-ig-empty.json" >/dev/null \
        || die "Images.Aspect is missing - the picker did not finish resolving"

    # 3. Image models are deliberately not residency-eligible. Starting one
    #    must replace the chat sidecar instead of sending the incompatible
    #    image architecture through the chat server's /v1/models/load path.
    wait_identifier Readiness.Action "$OUT/rlr-ig-readiness.json" \
        || die "Images readiness has no action to load its model"
    press "$OUT/rlr-ig-readiness.json" Readiness.Action "$OUT/rlr-ig-start.json" \
        || die "Images Readiness.Action is not pressable - the load button is dead"

    wait_fake_event ".event == \"server_start_rejected\" and .alias == \"$FAKE_IMAGE_ALIAS\"" \
        "the fake never rejected the dedicated image sidecar"
    if jq -e 'select(.event == "model_load")' "$OUT/fake-events.jsonl" >/dev/null 2>&1; then
        die "Images incorrectly issued an in-process /v1/models/load"
    fi

    # The dedicated process failure must remain actionable on the Images
    # surface, not disappear into logs after avoiding the resident 500.
    local i shown=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/rlr-shown.json"
        if jq -e '[.data.ui_elements[]?]
                  | map(((.title // "") | tostring) + " " + ((.value // "") | tostring) + " " + ((.description // "") | tostring) + " " + ((.help // "") | tostring))
                  | join(" ") | test("couldn.t load|Check the model files|choose another model"; "i")' \
               "$OUT/rlr-shown.json" >/dev/null 2>&1; then
            shown=1; break
        fi
        sleep 0.25
    done
    [[ "$shown" == 1 ]] \
        || die "the actionable image-sidecar diagnosis never appeared on the Images surface"
    jq -e '.data.ui_elements[]? | select(.identifier == "Readiness.Action")' \
        "$OUT/rlr-shown.json" >/dev/null \
        || die "the Images failure offered no recovery action"

    log "  resident-load-rejected OK"
}

flow_launch_integrations() {
    log "flow: launch-integrations"
    start_persona launch-integrations
    dismiss_first_run
    see_main "$OUT/main.json"
    press "$OUT/main.json" Sidebar.Launch "$OUT/launch.json"
    wait_tree_text "Connect your agents" "$OUT/launch.json" 40

    # ``ConnectToolsView`` renders the three-entry compatibility fallback
    # first, then replaces it with the sidecar's authoritative integration
    # registry. The heading appears before that asynchronous load completes,
    # so capturing immediately makes the baseline race between two valid UI
    # states. Settle on the fake sidecar's complete 14-entry registry before
    # asserting or recording the stopped-state structure.
    local i count=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/launch.json"
        count="$(jq '[.data.ui_elements[]?
                      | (.identifier // "")
                      | select(startswith("Launch.Integration.Copy."))]
                     | unique | length' "$OUT/launch.json")"
        [[ "$count" == 14 ]] && break
        sleep 0.1
    done
    [[ "$count" == 14 ]] \
        || die "Cold Launch did not settle on the 14-entry integration registry (got $count)"

    # Cold Launch is a beginner path, not a wall of live (copyable) commands.
    # The stopped state now stays a useful setup destination (#2297): the
    # endpoint shape and integration rows are shown as documentation, the
    # inline model picker lets a user choose a different downloaded model,
    # and the readiness banner offers Start. What must NOT happen is a
    # command the user can paste while it is still a placeholder — so every
    # `Launch.Integration.Copy.*` button must be present but `.enabled == false`
    # until the endpoint/key actually exist (Copy on a placeholder is the
    # silent-failure defect the disabled-Copy gate exists to prevent).
    enabled_count="$(jq '[.data.ui_elements[]? | select(((.identifier // "") | startswith("Launch.Integration.Copy.")) and .enabled == true)] | length' "$OUT/launch.json")"
    [[ "$enabled_count" == 0 ]] || die "Cold Launch offered $enabled_count copyable commands before the endpoint/key existed"
    jq -e '.data.ui_elements[]? | select(.identifier == "Readiness.Action")' "$OUT/launch.json" >/dev/null \
        || die "Cold Launch offered no primary model-start action"
    # The inline picker is addressable by its own menu popup
    # (``ModelPickerBar.ModelMenu``) — NOT by a composite id stamped on the
    # whole bar, which `ModelPickerBar` deliberately avoids (it propagates one
    # id onto both the popup and the (i) info button and makes them
    # indistinguishable). Assert the popup itself is present.
    jq -e '.data.ui_elements[]? | select(.identifier == "ModelPickerBar.ModelMenu")' "$OUT/launch.json" >/dev/null \
        || die "Cold Launch offered no inline model picker"
    baseline launch-integrations.complete "$OUT/launch.json"

    press "$OUT/launch.json" Sidebar.NewChat "$OUT/launch-chat.json" \
        || die "Sidebar.NewChat is not pressable from Launch"
    start_model
    see_main "$OUT/launch-model-ready.json"
    press "$OUT/launch-model-ready.json" Sidebar.Launch "$OUT/launch-ready-open.json"
    # Ready Launch leads with three common choices. The registry remains fully
    # reachable behind one explicit disclosure instead of occupying the page by
    # default.
    wait_identifier ConnectTools.MoreIntegrations "$OUT/launch-ready.json"
    local ready_copies
    ready_copies="$(jq '[.data.ui_elements[]?
                         | select(((.identifier // "")
                                   | startswith("Launch.Integration.Copy."))
                                  and .enabled == true)] | length' "$OUT/launch-ready.json")"
    [[ "$ready_copies" == 3 ]] \
        || die "Ready Launch should lead with 3 common integrations, got $ready_copies"
    press "$OUT/launch-ready.json" ConnectTools.MoreIntegrations \
        "$OUT/launch-more-press.json" \
        || die "More integrations disclosure is not pressable"
    for ((i=0; i<80; i++)); do
        see_main "$OUT/launch-ready.json"
        ready_copies="$(jq '[.data.ui_elements[]?
                              | select(((.identifier // "")
                                        | startswith("Launch.Integration.Copy."))
                                       and .enabled == true)]
                             | length' "$OUT/launch-ready.json")"
        [[ "$ready_copies" == 14 ]] && break
        sleep 0.1
    done
    [[ "$ready_copies" == 14 ]] \
        || die "Launch enabled $ready_copies of 14 copy commands after the chat model was ready"
    jq -e '.data.ui_elements[]? | select(.identifier == "Launch.Integration.Copy.cline")' "$OUT/launch-ready.json" >/dev/null \
        || die "Expanded Launch omitted config-writing target Cline"
    jq -e '.data.ui_elements[]? | select(.identifier == "Launch.Integration.Copy.smolagents")' "$OUT/launch-ready.json" >/dev/null \
        || die "Expanded Launch omitted adapter profile smolagents"
    local first_two
    first_two="$(jq -r '[.data.ui_elements[]?
                         | select((.identifier // "")
                                  | startswith("Launch.Integration.Copy."))
                         | .identifier]
                        | .[0:2]
                        | join(",")' "$OUT/launch-ready.json")"
    [[ "$first_two" == "Launch.Integration.Copy.claude-code,Launch.Integration.Copy.codex" ]] \
        || die "Launch did not lead with Claude Code then Codex (got: $first_two)"
    local integration_id copied_command
    while IFS= read -r integration_id; do
        pbcopy < /dev/null
        see_main "$OUT/launch-copy-before.json"
        press "$OUT/launch-copy-before.json" "$integration_id" \
            "$OUT/launch-copy-${integration_id##*.}.json" \
            || die "$integration_id is not pressable"
        copied_command="$(pbpaste)"
        [[ -n "$copied_command" ]] \
            || die "$integration_id pressed but copied no command"
    done < <(jq -r '[.data.ui_elements[]?
                      | select(((.identifier // "")
                                | startswith("Launch.Integration.Copy."))
                               and .enabled == true)
                      | .identifier] | unique[]' "$OUT/launch-ready.json")
    log "  all 14 integration buttons copied a non-empty ready command"
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
        FAKE_AUDIO_PULL_STATE="$OUT_ROOT/audio-readiness/pulled-audio.txt" \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_SIMULATED_AUDIO_PATH="$ROOT/../../examples/assistant_bank_en.wav" \
        RAPID_SIMULATED_SPEECH_SAVE_PATH="$OUT_ROOT/audio-readiness/saved-speech.wav"
    dismiss_first_run
    see_main "$OUT/chat.json"
    press "$OUT/chat.json" Sidebar.Audio "$OUT/dictation.json" \
        || die "Sidebar.Audio is not pressable"

    # Dictation is the Audio landing surface, but its global hotkey requires
    # real Microphone + Accessibility grants that an unattended runner must not
    # invent. Cover the reachable configuration UI and its privacy default,
    # then switch to the existing Speech workbench for its end-to-end actions.
    local i dictation_ready=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/dictation.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Model")' "$OUT/dictation.json" >/dev/null \
           && jq -e '.data.ui_elements[]?
                     | select(.identifier == "Dictation.Enable")' "$OUT/dictation.json" >/dev/null; then
            dictation_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$dictation_ready" == 1 ]] \
        || die "Audio did not open on a rendered Dictation pane"
    wait_identifier Dictation.ArchiveAudio "$OUT/dictation.json"
    [[ "$(element_field "$OUT/dictation.json" Dictation.ArchiveAudio value)" != "1" ]] \
        || die "Dictation retained raw microphone recordings without opt-in"
    # Do not snapshot this pane: Accessibility/Microphone grants and app-name
    # vocabulary suggestions legitimately differ per host. The semantic checks
    # above are the stable Golden contract.
    # Opening a tab is not a request to spend memory. #2053 removed automatic
    # loading everywhere else; the default Audio pane is the easiest place for
    # it to creep back in, because dictation *does* load a model — just later,
    # when the user actually presses the hotkey.
    assert_fake_server_starts \
        "$OUT/fake-events.jsonl" 0 "" "Opening Audio before any user action"

    press "$OUT/dictation.json" Audio.Mode.Speech "$OUT/speech-tab-press.json" \
        || die "Audio Speech segment is not pressable from Dictation"

    local speech_ready=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/speech.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Audio.Speech.ModelPicker")' "$OUT/speech.json" >/dev/null \
           && jq -e '.data.ui_elements[]?
                     | select(.identifier == "Readiness.Action"
                              and (.description // .value // .label // "") == "Download")' \
                    "$OUT/speech.json" >/dev/null; then
            speech_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_ready" == 1 ]] \
        || die "Speech did not expose download-only readiness"
    baseline audio-readiness.speech "$OUT/speech.json"

    press "$OUT/speech.json" Readiness.Action "$OUT/speech-download-start.json" \
        || die "Speech Download is not pressable"
    wait_fake_event \
        '.event == "command" and .subcommand == "pull" and .alias == "fake-qwen3-tts"' \
        "Speech Download did not invoke pull for fake-qwen3-tts"

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
        || die "Speech never exposed Downloading after Download"
    assert_fake_server_starts \
        "$OUT/fake-events.jsonl" 0 "" "Speech before pull completion"

    local speech_start_ready=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/speech-downloaded.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Readiness.Action"
                           and (.description // .value // .label // "") == "Start")' \
                 "$OUT/speech-downloaded.json" >/dev/null; then
            speech_start_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_start_ready" == 1 ]] \
        || die "Speech did not become Start-ready after its download completed"
    assert_fake_server_starts \
        "$OUT/fake-events.jsonl" 0 "" "Speech after download-only action"
    press "$OUT/speech-downloaded.json" Readiness.Action "$OUT/speech-start.json" \
        || die "Speech Start is not pressable after download"
    wait_fake_event \
        '.event == "server_started" and .alias == "fake-qwen3-tts"' \
        "Speech did not start after the explicit Start action"
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
        || die "Speech stayed behind Start after its model became ready"

    local speech_controls_ready=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/speech-controls-before.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Audio.Speech.LoadVoices"
                           and .enabled == true)' \
            "$OUT/speech-controls-before.json" >/dev/null; then
            speech_controls_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_controls_ready" == 1 ]] \
        || die "Speech became ready but Load Voices stayed disabled"
    press "$OUT/speech-controls-before.json" Audio.Speech.LoadVoices \
        "$OUT/speech-load-voices-press.json" \
        || die "Load Voices is not pressable"
    wait_fake_event '.event == "audio_voices"' \
        "Load Voices never reached the audio server"
    local voices_loaded=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/speech-voices-loaded.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Audio.Speech.VoicePicker"
                           and .value == "Golden")' \
            "$OUT/speech-voices-loaded.json" >/dev/null; then
            voices_loaded=1; break
        fi
        sleep 0.25
    done
    [[ "$voices_loaded" == 1 ]] \
        || die "Load Voices produced no selected voice"
    press "$OUT/speech-voices-loaded.json" Audio.Speech.VoicePicker \
        "$OUT/speech-voice-picker-press.json" \
        || die "Voice picker is not pressable"
    wait_identifier Audio.Speech.VoiceOption.Harbor "$OUT/speech-voice-options.json"
    press "$OUT/speech-voice-options.json" Audio.Speech.VoiceOption.Harbor \
        "$OUT/speech-voice-harbor-press.json" \
        || die "Harbor voice option is not pressable"
    see_main "$OUT/speech-voice-harbor.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "Audio.Speech.VoicePicker" and .value == "Harbor")' \
        "$OUT/speech-voice-harbor.json" >/dev/null \
        || die "voice picker did not select Harbor"

    local speed_before speed_after
    speed_before="$(element_field "$OUT/speech-voice-harbor.json" Audio.Speech.Speed value)"
    "$AX_DRIVER" increment "$APP_PID" Audio.Speech.Speed \
        > "$OUT/speech-speed-increment.json" \
        || die "speech speed rejected AXIncrement"
    see_main "$OUT/speech-speed-after.json"
    speed_after="$(element_field "$OUT/speech-speed-after.json" Audio.Speech.Speed value)"
    [[ -n "$speed_after" && "$speed_after" != "$speed_before" ]] \
        || die "speech speed accepted AXIncrement but did not change"
    "$AX_DRIVER" set-value "$APP_PID" Audio.Speech.Text "golden speech controls" \
        > "$OUT/speech-text-type.json"
    see_main "$OUT/speech-generate-ready.json"
    press "$OUT/speech-generate-ready.json" Audio.Speech.Generate \
        "$OUT/speech-generate-press.json" \
        || die "Generate Speech is not pressable"
    wait_fake_event '.event == "audio_speech"
                     and .voice == "Harbor"
                     and .text == "golden speech controls"' \
        "Generate Speech did not send the selected voice and text"
    wait_identifier Audio.Speech.Play "$OUT/speech-result.json"
    local speech_saved=0
    for ((i=0; i<40; i++)); do
        # Re-resolve the dynamic result button for every AXPress. A real
        # semantic action is required here; CGEvent coordinate clicks are not
        # trustworthy on unattended runners and can report success while TCC
        # discards the event.
        "$AX_DRIVER" press "$APP_PID" Audio.Speech.Save \
            > "$OUT/speech-save-press.json" 2>/dev/null || true
        if [[ -s "$OUT_ROOT/audio-readiness/saved-speech.wav" ]]; then
            speech_saved=1; break
        fi
        sleep 0.25
    done
    [[ "$speech_saved" == 1 ]] \
        || die "Save speech did not write the generated WAV"
    see_main "$OUT/speech-before-play.json"
    local playback_started=0
    for ((i=0; i<40; i++)); do
        "$AX_DRIVER" press "$APP_PID" Audio.Speech.Play \
            > "$OUT/speech-play-press.json" 2>/dev/null || true
        sleep 0.05
        see_main "$OUT/speech-playing.json"
        if [[ "$(element_field "$OUT/speech-playing.json" Audio.Speech.Play description)" == "Stop playback" ]]; then
            playback_started=1; break
        fi
        sleep 0.1
    done
    [[ "$playback_started" == 1 ]] \
        || die "Play speech did not enter playback state"
    "$AX_DRIVER" decrement "$APP_PID" Audio.Speech.Speed \
        > "$OUT/speech-speed-decrement.json" \
        || die "speech speed rejected AXDecrement"
    see_main "$OUT/speech-speed-restored.json"
    [[ "$(element_field "$OUT/speech-speed-restored.json" Audio.Speech.Speed value)" == "$speed_before" ]] \
        || die "speech speed did not restore its original value"
    log "  voices, voice selection, speed, and Generate Speech produced effects"

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

    # A media resident is process-wide state, not the Chat model selection.
    # Launch commands must neither advertise the TTS alias to coding agents
    # nor be copyable while their selected chat model is not serving. Since
    # #2297 the stopped Launch page renders the integration rows as
    # documentation with Copy deliberately disabled, so the guard is the
    # rows being present-but-not-copyable: the `Launch.Integration.Copy.*`
    # rows must EXIST (count > 0) and every one of them must be disabled
    # (enabled_count == 0), plus the Readiness start action must be present.
    # Counting the rows (not just checking nothing is enabled) is what
    # distinguishes "documentation rows rendered with Copy disabled" from a
    # regression where the whole Launch surface silently vanished. The
    # launch-integrations journey asserts the same present-but-not-copyable
    # contract.
    press "$OUT/speech-resident.json" Sidebar.Launch "$OUT/launch-from-audio.json" \
        || die "Sidebar.Launch is not pressable from an Audio residency"
    local launch_ready=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/launch-from-audio.json"
        if jq -e '.data.ui_elements[]? | select(.identifier == "Readiness.Action")' \
              "$OUT/launch-from-audio.json" >/dev/null \
           && [[ "$(jq '[.data.ui_elements[]?
                          | select(((.identifier // "")
                                    | startswith("Launch.Integration.Copy.")))] | length' \
                       "$OUT/launch-from-audio.json")" -gt 0 ]] \
           && [[ "$(jq '[.data.ui_elements[]?
                          | select(((.identifier // "")
                                    | startswith("Launch.Integration.Copy."))
                                   and .enabled == true)] | length' \
                       "$OUT/launch-from-audio.json")" == 0 ]]; then
            launch_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$launch_ready" == 1 ]] \
        || die "Launch exposed copyable commands, hid the launch rows, or had no chat-model start action from Audio"
    if jq -e '[.data.ui_elements[]? | .value? | strings]
              | any(contains("fake-qwen3-tts")
                    and (contains("--model")
                         or contains(" -m ")
                         or contains("ANTHROPIC_MODEL")
                         or contains("HERMES_INFERENCE_MODEL")))' \
            "$OUT/launch-from-audio.json" >/dev/null; then
        die "Launch leaked the resident Audio alias into an agent command"
    fi
    press "$OUT/launch-from-audio.json" Sidebar.Audio "$OUT/audio-after-launch.json" \
        || die "Sidebar.Audio is not pressable after checking Launch"
    wait_identifier Audio.Mode.Dictation "$OUT/audio-after-launch.json" \
        || die "Audio did not settle after returning from Launch"

    # Back to the default mode. Dictation's model lifecycle now speaks the
    # same language as every other surface: an uncached selection exposes the
    # shared Download readiness banner (no silent server-side fetch — #2053's
    # on-demand contract applies to LOADING, not downloading), the pull runs
    # through the app-wide download manager, and finishing it must not start
    # the model while dictation is off.
    press "$OUT/audio-after-launch.json" Audio.Mode.Dictation "$OUT/dictation-return-press.json" \
        || die "Audio Dictation segment is not pressable"
    local dictation_controls=0
    for ((i=0; i<40; i++)); do
        see_main "$OUT/dictation-return.json"
        if jq -e '[.data.ui_elements[]?
                   | .identifier // ""
                   | select(. == "Dictation.Model"
                            or . == "Dictation.Hotkey"
                            or . == "Dictation.Enable")]
                  | unique | length == 3' "$OUT/dictation-return.json" >/dev/null; then
            dictation_controls=1; break
        fi
        sleep 0.25
    done
    [[ "$dictation_controls" == 1 ]] \
        || die "Dictation did not expose its model, hotkey and enable controls"
    local dictation_download_ready=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/dictation-return.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Readiness.Action"
                           and (.description // .value // .label // "") == "Download")' \
                 "$OUT/dictation-return.json" >/dev/null; then
            dictation_download_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$dictation_download_ready" == 1 ]] \
        || die "Dictation did not expose download-only readiness for its uncached model"
    # No structural baseline for this pane: the Microphone and Accessibility
    # rows render a grant button only while the permission is missing, so its
    # tree legitimately differs between a fresh runner and a developer machine.
    # A snapshot would encode the runner's TCC state as the contract.
    assert_fake_server_starts \
        "$OUT/fake-events.jsonl" 1 "fake-qwen3-tts" "Opening Dictation"

    press "$OUT/dictation-return.json" Readiness.Action "$OUT/dictation-download.json" \
        || die "Dictation Download is not pressable"
    wait_fake_event \
        '.event == "command" and .subcommand == "pull" and .alias == "fake-whisper-small"' \
        "Dictation Download did not invoke pull"
    local dictation_downloaded=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/dictation-downloaded.json"
        if ! jq -e '.data.ui_elements[]?
                    | select(.identifier == "Readiness.Action")' \
                  "$OUT/dictation-downloaded.json" >/dev/null; then
            dictation_downloaded=1; break
        fi
        sleep 0.25
    done
    [[ "$dictation_downloaded" == 1 ]] \
        || die "Dictation banner did not clear after the download landed"
    assert_fake_server_starts \
        "$OUT/fake-events.jsonl" 1 "fake-qwen3-tts" "Dictation after Download"

    jq -n '{success: true,
            assertion: "Dictation is privacy-safe and inert on open, exposes the shared explicit Download lifecycle for an uncached model, and Speech keeps its explicit Download and Start"}' \
        > "$OUT/audio-readiness-actions.json"
    log "  audio-readiness OK"
    cleanup_persona
}

# Dictation is its own product journey, not merely the landing state of Audio.
# Keep this separate from audio-readiness so a regression in its controls is
# named directly in CI. Microphone and Accessibility grants are intentionally
# not globally faked: their TCC state belongs to the host. This journey opts
# into a two-key, process-local readiness fixture only around the OS permission
# and event-tap boundaries, allowing the real server/warmup state machine to be
# exercised deterministically. The stable contract is that every setup control
# is reachable, raw recordings remain opt-in, local vocabulary edits work,
# mode round-trips preserve the pane, opening it alone never starts a model,
# and enabling visibly transitions from Loading to Ready.
flow_dictation_rc2_upgrade() {
    log "flow: dictation-rc2-upgrade"
    start_persona dictation-rc2-upgrade \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_GUI_DICTATION_READINESS_FIXTURE=1 \
        FAKE_CACHED_DICTATION=1
    dismiss_first_run
    stop_app

    # Recreate the keys written by rc2 before lane ownership existed: one
    # shared last-served chat key plus independently persisted Dictation intent
    # and its audio model. The bundle id is unique to this throwaway persona.
    defaults write "$BUNDLE_ID" rapid.serve.lastAlias "fake-alias"
    defaults write "$BUNDLE_ID" dictation.enabled -bool true
    defaults write "$BUNDLE_ID" dictation.model "fake-whisper-small"
    relaunch_persona
    dismiss_first_run

    local i restored_chat=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/rc2-upgrade-chat.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "ModelPickerBar.ModelMenu"
                           and .value == "fake-alias")' \
                 "$OUT/rc2-upgrade-chat.json" >/dev/null; then
            restored_chat=1; break
        fi
        sleep 0.1
    done
    [[ "$restored_chat" == 1 ]] \
        || die "rc2 upgrade did not restore the persisted chat model"
    wait_send_idle "$OUT/rc2-upgrade-send-ready.json"
    jq -e -s 'any(.[]; .event == "server_started" and .alias == "fake-alias")
              and (any(.[]; .event == "server_started" and .alias == "fake-whisper-small") | not)' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "rc2 upgrade restored the transcription alias as process owner"
    press "$OUT/rc2-upgrade-send-ready.json" Sidebar.Audio \
        "$OUT/rc2-upgrade-audio-open.json" \
        || die "Audio is not reachable after rc2 upgrade"
    wait_identifier Dictation.Enable "$OUT/rc2-upgrade-audio.json"
    [[ "$(element_field "$OUT/rc2-upgrade-audio.json" Dictation.Enable value)" == "1" ]] \
        || die "rc2 upgrade lost persisted Dictation intent"

    log "  rc2 chat + Dictation state restored with Send enabled"
    log "  dictation-rc2-upgrade OK"
    cleanup_persona
}


flow_model_switch_active_request() {
    log "flow: model-switch-active-request"
    start_persona model-switch-active-request \
        FAKE_CHAT_RESPONSE_DELAY_MS=10000 \
        FAKE_INTER_TOKEN_SLEEP_S=0.02 \
        FAKE_CONTENT_REPEAT=250
    dismiss_first_run
    start_model
    send_prompt "shape:long keep this stream active" "busy-stream"

    # The fake residency endpoint reports the real in-flight handler count.
    # Wait for the independent handler-lifecycle fact before asking the picker
    # to switch; the app's own refresh then observes that same count.
    local i busy=0
    for ((i=0; i<80; i++)); do
        if jq -e 'select(.event == "chat_active" and .active_requests > 0)' \
            "$OUT/fake-events.jsonl" >/dev/null 2>&1; then
            busy=1; break
        fi
        sleep 0.1
    done
    [[ "$busy" == 1 ]] || die "fake residency never reported the active stream"

    see_main "$OUT/busy.json"
    # SwiftUI creates these rows only after the native menu opens. Resolve the
    # requested row from that fresh AX tree and press its stable identifier;
    # the driver fails closed if opening the menu did not expose the row.
    "$AX_DRIVER" select-menu-item "$APP_PID" ModelPickerBar.ModelMenu \
        ModelPickerBar.Alias.fake-external-alias > "$OUT/switch-requested.json" \
        || die "alternate cached model could not be selected during an active stream"
    wait_identifier ModelSwitchGuard.Cancel "$OUT/switch-guard.json"
    jq -e '.data.ui_elements[]?
           | select(.identifier == "ModelSwitchGuard.Cancel")' \
        "$OUT/switch-guard.json" >/dev/null \
        || die "active stream did not present the native switch guard"
    # Click the semantic Cancel button's bounds directly. Posting Escape can
    # succeed at the CoreGraphics boundary while the hosted confirmation
    # dialog ignores it, and hosted macOS can expose a native dialog button
    # while returning kAXErrorActionUnsupported for AXPress. A bounds click is
    # the same explicit user action and still fails closed unless the stable
    # product-owned identifier resolves to a visible control.
    "$AX_DRIVER" click-center "$APP_PID" ModelSwitchGuard.Cancel \
        > "$OUT/switch-cancelled.json" \
        || die "switch guard did not honor the native Cancel action"

    # Cancellation occurs before /v1/models/load or process teardown. The
    # original stream must complete and the original model remains selected.
    for ((i=0; i<240; i++)); do
        grep -q '"event": "chat_finished"' "$OUT/fake-events.jsonl" 2>/dev/null && break
        sleep 0.25
    done
    grep -q '"event": "chat_finished"' "$OUT/fake-events.jsonl" \
        || die "Cancel did not preserve the active stream through completion"
    [[ "$(jq -s '[.[] | select(.event == "server_started")] | length' "$OUT/fake-events.jsonl")" == 1 ]] \
        || die "Cancel replaced the running sidecar"
    ! grep -q '"event": "model_load"' "$OUT/fake-events.jsonl" \
        || die "Cancel reached the resident model-load endpoint"
    see_main "$OUT/settled.json"
    [[ "$(element_field "$OUT/settled.json" ModelPickerBar.ModelMenu value)" == "fake-alias" ]] \
        || die "Cancel changed the selected model"

    log "  active-request confirmation and cancellation preservation OK"
    cleanup_persona
}


flow_dictation() {
    log "flow: dictation"
    start_persona dictation \
        RAPID_GUI_GOLDEN_MODE=1 \
        RAPID_GUI_DICTATION_READINESS_FIXTURE=1 \
        FAKE_CACHED_DICTATION=1 \
        FAKE_AUDIO_TRANSCRIPTION_DELAY_MS=1800
    dismiss_first_run
    see_main "$OUT/chat.json"

    # Dictation is an input method for the active conversation. Start that
    # conversation first so this journey detects the release-blocking failure:
    # enabling speech input must reuse the mounted audio lane, never replace
    # the conversation process with a transcription-only server.
    start_model
    send_prompt "Conversation state before dictation" "dictation-before"
    local i conversation_id=""
    for ((i=0; i<40; i++)); do
        see_main "$OUT/chat-before-dictation.json"
        conversation_id="$(jq -r '.data.ui_elements[] | (.identifier // "")
            | select(test("^Sidebar\\.Conversation\\.[0-9A-Fa-f-]{36}$"))' \
            "$OUT/chat-before-dictation.json" | head -1)"
        [[ -n "$conversation_id" ]] && break
        sleep 0.1
    done
    [[ -n "$conversation_id" ]] \
        || die "Dictation precondition produced no conversation row"
    press "$OUT/chat.json" Sidebar.Audio "$OUT/dictation-open.json" \
        || die "Sidebar.Audio is not pressable"

    local controls_ready=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/dictation.json"
        if jq -e '[.data.ui_elements[]?
                   | .identifier // ""
                   | select(. == "Dictation.Model"
                            or . == "Dictation.Hotkey"
                            or . == "Dictation.Enable"
                            or . == "Dictation.NewTerm"
                            or . == "Dictation.AddTerm"
                            or . == "Dictation.ArchiveAudio")]
                  | unique | length == 6' "$OUT/dictation.json" >/dev/null; then
            controls_ready=1; break
        fi
        sleep 0.25
    done
    [[ "$controls_ready" == 1 ]] \
        || die "Dictation did not expose its complete setup surface"
    [[ "$(element_field "$OUT/dictation.json" Dictation.Model value)" == "fake-whisper-small" ]] \
        || die "Dictation did not select the available transcription model"
    [[ "$(element_field "$OUT/dictation.json" Dictation.Hotkey value)" == "Right ⌘" ]] \
        || die "Dictation did not expose the safe right-hand default hotkey"
    [[ "$(element_field "$OUT/dictation.json" Dictation.ArchiveAudio value)" != "1" ]] \
        || die "Dictation retained raw microphone recordings without opt-in"

    press "$OUT/dictation.json" Dictation.ArchiveAudio "$OUT/archive-on-press.json" \
        || die "Keep recordings is not pressable"
    see_main "$OUT/archive-on.json"
    [[ "$(element_field "$OUT/archive-on.json" Dictation.ArchiveAudio value)" == "1" ]] \
        || die "Keep recordings accepted a press but did not turn on"
    press "$OUT/archive-on.json" Dictation.ArchiveAudio "$OUT/archive-off-press.json" \
        || die "Keep recordings is not pressable after enabling"
    see_main "$OUT/archive-off.json"
    [[ "$(element_field "$OUT/archive-off.json" Dictation.ArchiveAudio value)" != "1" ]] \
        || die "Keep recordings did not return to its privacy-safe default"

    "$AX_DRIVER" set-value "$APP_PID" Dictation.NewTerm "GoldenTerm2049" \
        > "$OUT/vocabulary-type.json" \
        || die "Dictation vocabulary field rejected input"
    see_main "$OUT/vocabulary-ready.json"
    press "$OUT/vocabulary-ready.json" Dictation.AddTerm "$OUT/vocabulary-add.json" \
        || die "Dictation Add term is not pressable after input"
    wait_identifier Dictation.RemoveTerm.GoldenTerm2049 "$OUT/vocabulary-added.json" \
        || die "Dictation Add term produced no removable vocabulary chip"
    press "$OUT/vocabulary-added.json" Dictation.RemoveTerm.GoldenTerm2049 \
        "$OUT/vocabulary-remove.json" \
        || die "Dictation vocabulary remove is not pressable"
    see_main "$OUT/vocabulary-removed.json"
    if jq -e '.data.ui_elements[]?
              | select(.identifier == "Dictation.RemoveTerm.GoldenTerm2049")' \
             "$OUT/vocabulary-removed.json" >/dev/null; then
        die "Dictation vocabulary term remained after Remove"
    fi

    press "$OUT/vocabulary-removed.json" Audio.Mode.Speech "$OUT/speech.json" \
        || die "Speech mode is not pressable from Dictation"
    wait_identifier Audio.Speech.ModelPicker "$OUT/speech-ready.json"
    press "$OUT/speech-ready.json" Audio.Mode.Dictation "$OUT/dictation-return.json" \
        || die "Dictation mode is not pressable after visiting Speech"
    wait_identifier Dictation.Enable "$OUT/dictation-restored.json"
    [[ "$(jq -rs 'map(select(.event == "server_started")) | length' \
            "$OUT/fake-events.jsonl")" == 1 ]] \
        || die "Opening and configuring Dictation started another server"

    # Enabling is a readiness operation, not merely a preference flip. Hold
    # the fake STT probe open long enough to observe the contract in AX: the
    # pane says Loading while the model's lazy weights are cold, and only
    # changes to Ready after the probe returns and the hotkey can be armed.
    press "$OUT/dictation-restored.json" Dictation.Enable "$OUT/dictation-enable.json" \
        || die "Dictation Enable is not pressable with complete readiness"
    local loading_seen=0
    for ((i=0; i<80; i++)); do
        see_main "$OUT/dictation-loading.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Status"
                           and (((.description // .value // .label // "") | tostring)
                                | contains("Loading fake-whisper-small into memory")))' \
                 "$OUT/dictation-loading.json" >/dev/null; then
            loading_seen=1; break
        fi
        sleep 0.1
    done
    require_observed_phase "$loading_seen" loading
    [[ "$(element_field "$OUT/dictation-loading.json" Dictation.Enable value)" == "1" ]] \
        || die "Dictation lost the user's Enabled intent while loading"

    wait_fake_event '.event == "audio_transcription"' \
        "Dictation never sent the lazy-weight warmup probe"
    local ready_seen=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/dictation-ready.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Status"
                           and (((.description // .value // .label // "") | tostring)
                                | startswith("Listening — press")))' \
                 "$OUT/dictation-ready.json" >/dev/null; then
            ready_seen=1; break
        fi
        sleep 0.1
    done
    require_observed_phase "$ready_seen" listening

    # The warmup request must have stayed on the conversation server. A
    # transcription-only start here is the exact silent-eviction regression.
    [[ "$(jq -rs 'map(select(.event == "server_started")) | length' \
            "$OUT/fake-events.jsonl")" == 1 ]] \
        || die "Dictation replaced the active conversation server"
    jq -e -s 'any(.[]; .event == "server_started" and .alias == "fake-alias")
              and (any(.[]; .event == "server_started" and .alias == "fake-whisper-small") | not)' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "Dictation did not preserve the active conversation alias"

    press "$OUT/dictation-ready.json" "$conversation_id" "$OUT/chat-after-dictation.json" \
        || die "The existing conversation is not reachable after Dictation warmup"
    wait_send_idle "$OUT/chat-after-dictation-ready.json"
    assert_tree_text "$OUT/chat-after-dictation-ready.json" \
        "Conversation state before dictation"
    send_prompt "Conversation survives dictation" "dictation-chat"

    # Relaunch is the state-ownership contract: Dictation remains enabled, but
    # its audio selection must never become the restored chat model. Session
    # restore starts the catalog-proven chat alias first, then re-arms the
    # audio lane on that process; no transcription-only child may appear.
    relaunch_persona
    dismiss_first_run
    local restored_chat=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/dictation-relaunch-chat.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "ModelPickerBar.ModelMenu"
                           and .value == "fake-alias")' \
                 "$OUT/dictation-relaunch-chat.json" >/dev/null; then
            restored_chat=1; break
        fi
        sleep 0.1
    done
    [[ "$restored_chat" == 1 ]] \
        || die "Relaunch did not restore the conversation model after enabling Dictation"
    wait_send_idle "$OUT/dictation-relaunch-send-ready.json"
    jq -e -s '(map(select(.event == "server_started" and .alias == "fake-alias")) | length) == 2
              and (any(.[]; .event == "server_started" and .alias == "fake-whisper-small") | not)' \
        "$OUT/fake-events.jsonl" >/dev/null \
        || die "Relaunch restored the transcription alias as the process-owning chat model"
    press "$OUT/dictation-relaunch-send-ready.json" Sidebar.Audio \
        "$OUT/dictation-relaunch-audio-open.json" \
        || die "Audio is not reachable after Dictation relaunch"
    wait_identifier Dictation.Enable "$OUT/dictation-relaunch-audio.json"
    [[ "$(element_field "$OUT/dictation-relaunch-audio.json" Dictation.Enable value)" == "1" ]] \
        || die "Relaunch disabled the user's persisted Dictation intent"

    # A crashed process releases its mounted speech lane. Moving away from
    # Rapid and returning must repair only the global input tap; app activation
    # must not race the server's bounded crash recovery by launching an
    # audio-only sidecar. The pane stays honest about the next user action.
    local listening_after_relaunch=0
    for ((i=0; i<120; i++)); do
        see_main "$OUT/dictation-relaunch-ready.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Status"
                           and (((.description // .value // .label // "") | tostring)
                                | startswith("Listening — press")))' \
                 "$OUT/dictation-relaunch-ready.json" >/dev/null; then
            listening_after_relaunch=1; break
        fi
        sleep 0.1
    done
    [[ "$listening_after_relaunch" == 1 ]] \
        || die "Dictation did not finish restoring after relaunch"
    local crash_pid crash_command audio_starts_before_foreground
    crash_pid="$(jq -rs 'map(select(.event == "server_started" and .alias == "fake-alias"))
                         | last | .pid // empty' "$OUT/fake-events.jsonl")"
    [[ "$crash_pid" =~ ^[0-9]+$ ]] || die "Dictation crash fixture has no owned sidecar pid"
    crash_command="$(ps -p "$crash_pid" -o command= 2>/dev/null || true)"
    [[ "$crash_command" == *"serve fake-alias"* ]] \
        || die "Dictation crash fixture refused to signal an unowned process"
    local terminal_seen=0 paused_seen=0
    audio_starts_before_foreground="$(jq -rs \
        'map(select(.event == "server_started" and .alias == "fake-whisper-small")) | length' \
        "$OUT/fake-events.jsonl")"
    osascript - "$APP_PID" > "$OUT/dictation-background.json" <<'APPLESCRIPT'
on run argv
    tell application "Finder" to activate
    delay 0.2
    return "{\"success\":true,\"method\":\"finder\"}"
end run
APPLESCRIPT
    kill "$crash_pid"
    for ((i=0; i<40; i++)); do
        see_main "$OUT/dictation-crashed-background.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Status"
                           and (((.description // .value // .label // "") | tostring)
                                | startswith("Listening paused — press")))' \
                 "$OUT/dictation-crashed-background.json" >/dev/null; then
            terminal_seen=1; break
        fi
        sleep 0.05
    done
    [[ "$terminal_seen" == 1 ]] \
        || die "Dictation did not observe the crashed sidecar before foreground activation"
    osascript - "$APP_PID" > "$OUT/dictation-foreground.json" <<'APPLESCRIPT'
on run argv
    set targetPID to (item 1 of argv) as integer
    tell application "System Events"
        set frontmost of first application process whose unix id is targetPID to true
    end tell
    delay 0.5
    return "{\"success\":true,\"method\":\"frontmost-after-crash\"}"
end run
APPLESCRIPT
    for ((i=0; i<10; i++)); do
        see_main "$OUT/dictation-after-foreground.json"
        if jq -e '.data.ui_elements[]?
                  | select(.identifier == "Dictation.Status"
                           and (((.description // .value // .label // "") | tostring)
                                | startswith("Listening paused — press")))' \
                 "$OUT/dictation-after-foreground.json" >/dev/null; then
            paused_seen=1; break
        fi
        sleep 0.05
    done
    [[ "$paused_seen" == 1 ]] \
        || die "Foreground activation hid the explicit Dictation reconnect action"
    # Observe beyond the activation itself instead of taking one instant
    # sample. The expected chat auto-respawn may start fake-alias during this
    # window; only a transcription-owned process proves the foreground path
    # restarted Dictation behind the user's back.
    for ((i=0; i<60; i++)); do
        [[ "$(jq -rs \
                'map(select(.event == "server_started" and .alias == "fake-whisper-small")) | length' \
                "$OUT/fake-events.jsonl")" == "$audio_starts_before_foreground" ]] \
            || die "Foreground activation silently restarted a Dictation sidecar"
        sleep 0.05
    done

    log "  setup controls, privacy toggle, vocabulary, co-loaded warmup, relaunch, preserved chat, and foreground-after-crash produced effects"
    log "  dictation OK"
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
    cached-curated-tradeup) flow_cached_curated_tradeup ;;
    cached-variant-collapse) flow_cached_variant_collapse ;;
    download-progress) flow_download_progress ;;
    settings-persistence) flow_settings_persistence ;;
    settings-mtp) flow_settings_mtp ;;
    chat-restore) flow_chat_restore ;;
    chat-depth) flow_chat_depth ;;
    model-switch-active-request) flow_model_switch_active_request ;;
    model-crash-recovery) flow_model_crash_recovery ;;
    low-memory-choice) flow_low_memory_choice ;;
    update-state) flow_update_state ;;
    update-busy) flow_update_busy ;;
    campaign-banner) flow_campaign_banner ;;
    window-close-prompt) flow_window_close_prompt ;;
    no-dead-controls) flow_no_dead_controls ;;
    catalog-integrity) flow_catalog_integrity ;;
    browse-all-destination) flow_browse_all_destination ;;
    chat-document-attachment) flow_chat_document_attachment ;;
    chat-multimodal-attachments) flow_chat_multimodal_attachments ;;
    image-generation) flow_image_generation ;;
    dictation) flow_dictation ;;
    dictation-rc2-upgrade) flow_dictation_rc2_upgrade ;;
    audio-readiness) flow_audio_readiness ;;
    resident-load-rejected) flow_resident_load_rejected ;;
    launch-integrations) flow_launch_integrations ;;
    all)
        flow_fresh_install
        flow_cached_quickstart
        flow_cached_curated_tradeup
        flow_cached_variant_collapse
        flow_download_progress
        flow_settings_persistence
        flow_settings_mtp
        flow_chat_restore
        flow_chat_depth
        flow_model_switch_active_request
        flow_model_crash_recovery
        flow_low_memory_choice
        flow_update_state
        flow_update_busy
        flow_campaign_banner
        flow_window_close_prompt
        flow_no_dead_controls
        flow_catalog_integrity
        flow_browse_all_destination
        flow_chat_document_attachment
        flow_chat_multimodal_attachments
        flow_image_generation
        flow_dictation
        flow_audio_readiness
        flow_resident_load_rejected
        flow_launch_integrations
        ;;
    *) die "unknown flow: $FLOW" ;;
esac

write_result pass 0
RESULT_WRITTEN=1
log "PASS — $FLOW"
log "artifacts: $OUT_ROOT"
