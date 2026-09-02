#!/bin/bash
# Post-deployment smoke test for the documented launchd service. This script is
# read-only: it neither loads nor restarts the daemon. A bearer token is read
# from RAPID_MLX_API_KEY and stored in a mode-600 temporary curl config so it
# never appears in argv or test output.
set -euo pipefail

LABEL="${RAPID_MLX_SERVICE_LABEL:-com.rapidmlx.server}"
DOMAIN="${RAPID_MLX_SERVICE_DOMAIN:-system}"
BASE_URL="${RAPID_MLX_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${RAPID_MLX_SMOKE_MODEL:-default}"
EXPECTED_USER="${RAPID_MLX_SERVICE_USER:-}"

case "$LABEL" in
    *[!A-Za-z0-9._-]*|'') echo "invalid service label: $LABEL" >&2; exit 2 ;;
esac
case "$DOMAIN" in
    system|user/[0-9]*|gui/[0-9]*) ;;
    *) echo "invalid launchd domain: $DOMAIN" >&2; exit 2 ;;
esac
case "$MODEL" in
    *[!A-Za-z0-9._/-]*|'') echo "invalid model name: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$BASE_URL" =~ ^http://(127\.0\.0\.1|localhost):[0-9]{1,5}$ ]] &&
   [[ ! "$BASE_URL" =~ ^https://[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?$ ]]; then
    echo "invalid or unsafe base URL: $BASE_URL" >&2
    echo "use an exact loopback HTTP origin or an HTTPS origin without userinfo or a path" >&2
    exit 2
fi
case "${RAPID_MLX_API_KEY:-}" in
    *$'\n'*|*$'\r'*|*\"*|*\\*)
        echo "RAPID_MLX_API_KEY contains characters unsafe for a curl config" >&2
        exit 2
        ;;
esac

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }

CURL_CONFIG="$(mktemp "${TMPDIR:-/tmp}/rapidmlx-smoke.XXXXXX")"
RESPONSE="$(mktemp "${TMPDIR:-/tmp}/rapidmlx-response.XXXXXX")"
RESPONSE_NEXT="$(mktemp "${TMPDIR:-/tmp}/rapidmlx-response-next.XXXXXX")"
cleanup() { rm -f "$CURL_CONFIG" "$RESPONSE" "$RESPONSE_NEXT"; }
trap cleanup EXIT
chmod 600 "$CURL_CONFIG" "$RESPONSE" "$RESPONSE_NEXT"
printf '%s\n' 'silent' 'show-error' 'fail-with-body' 'connect-timeout = 5' 'max-time = 120' > "$CURL_CONFIG"
if [ -n "${RAPID_MLX_API_KEY:-}" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$RAPID_MLX_API_KEY" >> "$CURL_CONFIG"
fi

echo "[1/4] launchd registration"
SERVICE_PRINT="$(launchctl print "$DOMAIN/$LABEL" 2>&1)" || {
    echo "service is not registered in the $DOMAIN domain: $LABEL" >&2
    echo "run: sudo launchctl bootstrap system /Library/LaunchDaemons/$LABEL.plist" >&2
    exit 1
}
PID="$(printf '%s\n' "$SERVICE_PRINT" | awk -F'= ' '/^[[:space:]]*pid = [0-9]+/ {gsub(/[^0-9]/, "", $2); print $2; exit}')"
if [ -z "$PID" ]; then
    echo "service is registered but has no live process" >&2
    exit 1
fi
ACTUAL_USER="$(ps -o user= -p "$PID" | awk '{$1=$1; print}')"
if [ -z "$ACTUAL_USER" ]; then
    echo "service reports pid $PID, but that process is not visible" >&2
    exit 1
fi
if [ -n "$EXPECTED_USER" ]; then
    [ "$ACTUAL_USER" = "$EXPECTED_USER" ] || {
        echo "service runs as $ACTUAL_USER, expected $EXPECTED_USER" >&2
        exit 1
    }
fi
echo "  running (pid $PID${EXPECTED_USER:+, user $EXPECTED_USER})"

echo "[2/4] liveness"
echo "[3/4] readiness and model inventory"
READY=false
READY_DEADLINE=$((SECONDS + 120))
while ((SECONDS < READY_DEADLINE)); do
    if curl -q --config "$CURL_CONFIG" --max-time 1 "$BASE_URL/livez" > "$RESPONSE" 2>/dev/null &&
       grep -Eq '"status"[[:space:]]*:[[:space:]]*"(ok|alive|healthy)"|^OK$' "$RESPONSE" &&
       curl -q --config "$CURL_CONFIG" --max-time 1 "$BASE_URL/readyz" > "$RESPONSE" 2>/dev/null &&
       grep -Eq '"ready"[[:space:]]*:[[:space:]]*true|"status"[[:space:]]*:[[:space:]]*"(ok|ready|healthy)"|^OK$' "$RESPONSE"; then
        READY=true
        break
    fi
    sleep 1
done
if [ "$READY" != true ]; then
    echo "service did not become ready within 120 seconds" >&2
    exit 1
fi
curl -q --config "$CURL_CONFIG" "$BASE_URL/v1/models" > "$RESPONSE"
grep -q '"data"' "$RESPONSE" || { echo "unexpected /v1/models response" >&2; exit 1; }

echo "[4/4] one-token completion"
printf '{"model":"%s","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":1,"temperature":0}' "$MODEL" > "$RESPONSE"
curl -q --config "$CURL_CONFIG" \
    -H 'Content-Type: application/json' \
    --data-binary "@$RESPONSE" \
    "$BASE_URL/v1/chat/completions" > "$RESPONSE_NEXT"
mv "$RESPONSE_NEXT" "$RESPONSE"
grep -q '"choices"' "$RESPONSE" || { echo "completion response has no choices" >&2; exit 1; }

echo "PASS: $LABEL is registered, ready, and serving completions"
