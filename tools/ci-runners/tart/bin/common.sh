# Shared settings and helpers for the Rapid-MLX tart CI runners.
# Sourced by supervisor.sh, slot.sh and build-golden.sh. bash 3.2 compatible.

CI_ROOT="${CI_ROOT:-$HOME/ci-runners}"
# Per-host settings (MACHINE_ID, SLOTS, VM_CPU, VM_MEM_MB, VM_DISK_GB, MC_DIRECT, MIN_FREE_GB).
[ -f "$CI_ROOT/host.env" ] && . "$CI_ROOT/host.env"
MACHINE_ID="${MACHINE_ID:?set MACHINE_ID in host.env}"
export TART_HOME="${TART_HOME:-$CI_ROOT/tart-home}"
export PATH="$CI_ROOT/bin:/usr/bin:/bin:/usr/sbin:/sbin"
TART="$CI_ROOT/bin/tart"

REPO="${REPO:-raullenchai/Rapid-MLX}"
GOLDEN="${GOLDEN:-rmx-golden}"
SLOTS="${SLOTS:-2}"
VM_CPU="${VM_CPU:-4}"
VM_MEM_MB="${VM_MEM_MB:-8192}"
VM_DISK_GB="${VM_DISK_GB:-30}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"   # do not start a new VM below this free space
VM_PREFIX="rmx-slot"
RUNNER_PREFIX="rmx-tart-$MACHINE_ID"
LABELS_JSON='["self-hosted","Linux","ARM64","rapidmlx-tart"]'
GH_TOKEN_FILE="$CI_ROOT/secrets/gh-token"
PAUSE_FILE="$CI_ROOT/PAUSE"
# Machine Control lease owner for this host. MC_DIRECT=1: query the API with
# $CI_ROOT/secrets/machinecontrol.token. Otherwise a relay on the Studio writes
# owner=<lease owner or empty> to this file every ~20s (hosts off the tailnet).
MC_STATE_FILE="$CI_ROOT/state/mc-lease"
MC_URL="${MC_URL:-}"   # set in host.env when MC_DIRECT=1
MC_TOKEN_FILE="$CI_ROOT/secrets/machinecontrol.token"
MC_STATE_STALE_SECS="${MC_STATE_STALE_SECS:-600}"
OUR_OWNER="rapid-mlx/harbor-ci-runners"
MAX_JOB_SECS="${MAX_JOB_SECS:-25200}"   # 7h hard cap per VM (GitHub job limit is 6h)
LOG_DIR="$CI_ROOT/logs"

log() { printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOG_TAG:-ci}" "$*"; }

# GitHub REST call; token passed via curl config on stdin so it never appears in argv.
# usage: gh_api METHOD PATH [JSON_BODY]  -> prints body, returns 0 on 2xx
gh_api() {
  local method="$1" api_path="$2" body="${3:-}" out code
  out=$(mktemp "${TMPDIR:-/tmp}/ghapi.XXXXXX")
  code=$( { printf 'header = "Authorization: Bearer %s"\n' "$(cat "$GH_TOKEN_FILE")"; } |
    curl -sS -K - -o "$out" -w '%{http_code}' -X "$method" \
      -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
      ${body:+-H "Content-Type: application/json" --data "$body"} \
      "https://api.github.com$api_path" ) || code=000
  cat "$out"; rm -f "$out"
  case "$code" in 2*) return 0;; *) echo "gh_api $method $api_path -> HTTP $code" >&2; return 1;; esac
}

# 0 = draining (do not start new VMs), 1 = clear. Reason on stdout.
refresh_mc_state() {  # MC_DIRECT hosts: refresh the state file at most every 20s
  [ "${MC_DIRECT:-0}" = 1 ] && [ -n "$MC_URL" ] || return 0
  if [ -f "$MC_STATE_FILE" ] && [ $(( $(date +%s) - $(stat -f %m "$MC_STATE_FILE") )) -lt 20 ]; then return 0; fi
  local owner
  owner=$( { printf 'header = "Authorization: Bearer %s"\n' "$(cat "$MC_TOKEN_FILE")"; } |
    curl -sS -K - --max-time 10 "$MC_URL/api/state" |
    jq -er --arg m "$MACHINE_ID" '.machines[] | select(.id==$m) | (.lease // {}) | if .ended then "" else (.owner // "") end' ) || return 0
  printf 'owner=%s\n' "$owner" > "$MC_STATE_FILE.tmp" && mv "$MC_STATE_FILE.tmp" "$MC_STATE_FILE"
}

should_drain() {
  if [ -f "$PAUSE_FILE" ]; then echo "pause file present"; return 0; fi
  refresh_mc_state
  if [ ! -f "$MC_STATE_FILE" ]; then echo "no machine-control state yet (fail-open)"; return 1; fi
  local age owner
  age=$(( $(date +%s) - $(stat -f %m "$MC_STATE_FILE") ))
  if [ "$age" -gt "$MC_STATE_STALE_SECS" ]; then echo "machine-control state stale ${age}s (fail-open)"; return 1; fi
  owner=$(sed -n 's/^owner=//p' "$MC_STATE_FILE" | head -1)
  if [ -n "$owner" ] && [ "$owner" != "$OUR_OWNER" ]; then echo "leased by $owner"; return 0; fi
  return 1
}

# Remove a registered runner if it is idle. Busy runners are refused by GitHub (422).
delete_runner_if_idle() {
  local id="$1"
  [ -n "$id" ] || return 0
  gh_api DELETE "/repos/$REPO/actions/runners/$id" >/dev/null 2>&1
}

runner_busy() {  # prints true/false, gone (HTTP 404), or unknown (other errors)
  local id="$1" r err
  err=$(mktemp "${TMPDIR:-/tmp}/ghapi-err.XXXXXX")
  if r=$(gh_api GET "/repos/$REPO/actions/runners/$id" 2>"$err"); then
    printf '%s' "$r" | jq -r '.busy'
  elif grep -q 'HTTP 404$' "$err"; then
    echo gone
  else
    echo unknown
  fi
  rm -f "$err"
}

# Interruptible sleep so TERM traps run promptly.
nap() { sleep "$1" & wait $!; }

free_gb() { df -g "$TART_HOME" | awk 'NR==2 {print $4}'; }
