#!/usr/bin/env bash
# Minimal "single-Mac behind Cloudflare" dogfood MVP.
#
# Starts rapid-mlx serving an alias on localhost, then exposes that port
# as a public *.trycloudflare.com endpoint gated by an API key (Bearer
# token). Quick tunnel is ephemeral — URL resets on each run.
#
#   Usage:
#     scripts/run_dogfood_mvp.sh [start]   # default
#     scripts/run_dogfood_mvp.sh stop
#     scripts/run_dogfood_mvp.sh status
#
#   Env vars:
#     MODEL            alias to serve (default: qwen3.5-35b)
#     PORT             local port (default: 8765)
#     API_KEY          bearer token (default: random 24 hex bytes)
#     RAPID_MLX_CMD    serve command (default: auto — editable `python3.12 -m
#                      vllm_mlx.cli` if we're inside the vllm-mlx repo,
#                      else system `rapid-mlx`). The editable path sees
#                      newly-added aliases.json entries; the brew-installed
#                      binary only knows the aliases bundled at install time.
#     TUNNEL_MODE      quick | named (default: auto — named iff ~/.cloudflared/config.yml exists)
#     TUNNEL_NAME      named tunnel name (default: rapid-mlx-mvp)
#     TUNNEL_HOSTNAME  named tunnel hostname (default: parsed from config.yml)
#
# Designed to be re-runnable: `stop` then `start` gives a fresh URL in
# quick mode and the same persistent URL in named mode.

set -euo pipefail

MODEL="${MODEL:-qwen3.5-35b}"
PORT="${PORT:-8765}"
API_KEY="${API_KEY:-$(openssl rand -hex 24)}"
# Extra args appended to `rapid-mlx serve`. Defaults tuned for chat-UX dogfood:
#   --no-thinking         skip <think> reasoning so Open WebUI sees content
#                         immediately (cuts perceived TTFT 10-15s on qwen3.5)
#   --cors-origins "*"    allow browser-based chat UIs (LobeChat, Big-AGI, …)
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:---no-thinking --cors-origins *}"

# Tunnel mode auto-detect: if a cloudflared config exists, default to named.
TUNNEL_NAME="${TUNNEL_NAME:-rapid-mlx-mvp}"
CF_CONFIG="$HOME/.cloudflared/config.yml"
if [ -z "${TUNNEL_MODE:-}" ]; then
  if [ -f "$CF_CONFIG" ]; then TUNNEL_MODE="named"; else TUNNEL_MODE="quick"; fi
fi
if [ -z "${TUNNEL_HOSTNAME:-}" ] && [ -f "$CF_CONFIG" ]; then
  # POSIX bracket class — macOS BSD awk doesn't support \s.
  TUNNEL_HOSTNAME="$(awk '/^[[:space:]]*-[[:space:]]*hostname:/ {print $NF; exit}' "$CF_CONFIG")"
fi

LOG_DIR="$HOME/.cache/rapid-mlx/dogfood"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
TUNNEL_LOG="$LOG_DIR/tunnel.log"
STATE_DIR="$LOG_DIR/state"
mkdir -p "$STATE_DIR"
# State dir holds the API key — keep it owner-only so a shared host doesn't
# leak the bearer token used by the public tunnel.
chmod 700 "$STATE_DIR"
SERVER_PID="$STATE_DIR/server.pid"
TUNNEL_PID="$STATE_DIR/tunnel.pid"
URL_FILE="$STATE_DIR/url"
KEY_FILE="$STATE_DIR/api_key"

cmd="${1:-start}"

is_alive() {
  local f="$1"
  [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null
}

do_stop() {
  for f in "$TUNNEL_PID" "$SERVER_PID"; do
    if is_alive "$f"; then
      kill "$(cat "$f")" 2>/dev/null || true
    fi
    rm -f "$f"
  done
  rm -f "$URL_FILE" "$KEY_FILE"
  echo "stopped."
}

do_status() {
  if is_alive "$SERVER_PID"; then
    echo "server: up (pid $(cat "$SERVER_PID"))"
  else
    echo "server: down"
  fi
  if is_alive "$TUNNEL_PID"; then
    echo "tunnel: up (pid $(cat "$TUNNEL_PID"))"
  else
    echo "tunnel: down"
  fi
  if [ -f "$URL_FILE" ]; then
    echo "url:    $(cat "$URL_FILE")"
  fi
  if [ -f "$KEY_FILE" ]; then
    echo "key:    $(cat "$KEY_FILE")"
  fi
}

case "$cmd" in
  stop)   do_stop;   exit 0 ;;
  status) do_status; exit 0 ;;
  start)  ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac

if is_alive "$SERVER_PID" || is_alive "$TUNNEL_PID"; then
  echo "already running — run '$0 stop' first." >&2
  do_status
  exit 1
fi

# Pick the serve command. Prefer the editable repo CLI when we're inside
# vllm-mlx — the brew-installed `rapid-mlx` ships an older aliases.json
# and won't see recent additions like `minimax-m2.7`.
if [ -z "${RAPID_MLX_CMD:-}" ]; then
  if [ -f "$(git rev-parse --show-toplevel 2>/dev/null)/vllm_mlx/cli.py" ] \
       && python3.12 -c "import vllm_mlx" >/dev/null 2>&1; then
    RAPID_MLX_CMD="python3.12 -m vllm_mlx.cli"
  else
    RAPID_MLX_CMD="rapid-mlx"
  fi
fi
if ! $RAPID_MLX_CMD --help >/dev/null 2>&1; then
  echo "RAPID_MLX_CMD=$RAPID_MLX_CMD not runnable" >&2; exit 1
fi
if ! command -v cloudflared >/dev/null; then
  echo "cloudflared not found — brew install cloudflared" >&2; exit 1
fi

echo "Starting $RAPID_MLX_CMD ($MODEL on :$PORT) [extra: $EXTRA_SERVE_ARGS] …"
# shellcheck disable=SC2086  # intentional word splitting on CMD + EXTRA args
# `set -f` disables pathname expansion so a literal `*` in EXTRA_SERVE_ARGS
# (e.g. `--cors-origins *`) survives the unquoted expansion instead of
# globbing against $PWD. Restore right after.
set -f
nohup $RAPID_MLX_CMD serve "$MODEL" --port "$PORT" --api-key "$API_KEY" \
  --log-level INFO $EXTRA_SERVE_ARGS >"$SERVER_LOG" 2>&1 &
set +f
echo $! > "$SERVER_PID"

# Wait for /healthz — model load can take 60-90s for a 30B MoE from cache.
echo -n "Waiting for /healthz "
ok=0
for _ in $(seq 1 180); do
  if curl -s -f "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
    echo " ok"
    ok=1; break
  fi
  echo -n "."
  sleep 2
done
if [ "$ok" -ne 1 ]; then
  echo " FAIL — see $SERVER_LOG"
  do_stop
  exit 1
fi

if [ "$TUNNEL_MODE" = "named" ]; then
  if [ -z "${TUNNEL_HOSTNAME:-}" ]; then
    echo "TUNNEL_MODE=named but no hostname (set TUNNEL_HOSTNAME or add ingress to $CF_CONFIG)" >&2
    do_stop
    exit 1
  fi
  echo "Starting cloudflared named tunnel ($TUNNEL_NAME → $TUNNEL_HOSTNAME) …"
  nohup cloudflared tunnel run "$TUNNEL_NAME" >"$TUNNEL_LOG" 2>&1 &
  echo $! > "$TUNNEL_PID"
  echo -n "Waiting for tunnel registration "
  ok=0
  for _ in $(seq 1 60); do
    if grep -q "Registered tunnel connection" "$TUNNEL_LOG" 2>/dev/null; then
      echo " ok"; ok=1; break
    fi
    echo -n "."
    sleep 1
  done
  if [ "$ok" -ne 1 ]; then
    echo " FAIL — see $TUNNEL_LOG"
    do_stop
    exit 1
  fi
  URL="https://$TUNNEL_HOSTNAME"
else
  echo "Starting cloudflared quick tunnel …"
  nohup cloudflared tunnel --url "http://localhost:$PORT" >"$TUNNEL_LOG" 2>&1 &
  echo $! > "$TUNNEL_PID"
  echo -n "Waiting for tunnel URL "
  URL=""
  for _ in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
    if [ -n "$URL" ]; then echo " $URL"; break; fi
    echo -n "."
    sleep 1
  done
  if [ -z "$URL" ]; then
    echo " FAIL — see $TUNNEL_LOG"
    do_stop
    exit 1
  fi
fi

echo "$URL" > "$URL_FILE"
echo "$API_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"

# Probe the tunnel itself — quick mode often surfaces the URL before the
# tunnel is actually forwarding traffic. We allow up to 30s; that's well
# below the cloudflared registration timeout (60s loop above) but enough
# to weed out "URL was printed but 502s for 10 seconds" races.
echo -n "Probing $URL/healthz "
probe_ok=0
for _ in $(seq 1 30); do
  if curl -sSf -m 3 "$URL/healthz" >/dev/null 2>&1; then
    echo " ok"
    probe_ok=1
    break
  fi
  echo -n "."
  sleep 1
done
if [ "$probe_ok" -ne 1 ]; then
  echo " FAIL — tunnel did not become reachable; see $TUNNEL_LOG"
  do_stop
  exit 1
fi

cat <<EOF

==================================================================
  Dogfood MVP up
==================================================================
  Endpoint:  $URL
  Model:     $MODEL
  API key:   $API_KEY

  Probe:
    curl $URL/healthz

  Chat:
    curl -sS $URL/v1/chat/completions \\
      -H "Authorization: Bearer $API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"model":"$MODEL","stream":false,"messages":[{"role":"user","content":"hi"}]}'

  Logs:
    tail -f $SERVER_LOG
    tail -f $TUNNEL_LOG

  Stop:
    $0 stop
==================================================================
EOF
