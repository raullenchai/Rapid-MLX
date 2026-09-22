#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One-command production deployment of the Marvin's Garden decision API
# on a Mac (Apple Silicon). Idempotent: stops any previous instance first.
#
#   bash serve/deploy_local.sh                 # foreground (tmux/screen recommended)
#   TOKEN=... bash serve/deploy_local.sh       # custom bearer token
#   PORT=8443 bash serve/deploy_local.sh
#
# The server binds 0.0.0.0 so a Cloudflare Tunnel (cloudflared) can expose
# it: cloudflared tunnel --url http://localhost:${PORT:-8123}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8123}"
TOKEN="${TOKEN:-$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
ADAPTER="${MARVIN_ADAPTER:-$REPO/adapters/release/marvins-garden-v15c}"

pkill -f "demo_server.py --port $PORT" 2>/dev/null || true
sleep 1

export MARVIN_ADAPTER="$ADAPTER"
export MARVIN_SERVE_TOKEN="$TOKEN"
export MARVIN_RATE_PER_MIN="${MARVIN_RATE_PER_MIN:-30}"
export MARVIN_QUEUE_CAP="${MARVIN_QUEUE_CAP:-8}"

echo "== Marvin's Garden decision API =="
echo "port: $PORT (playground http://localhost:$PORT/)"
echo "bearer token: $TOKEN"
echo "adapter: $ADAPTER"
echo "stop with: pkill -f 'demo_server.py --port $PORT'"
echo "expose: cloudflared tunnel --url http://localhost:$PORT"
exec /tmp/rm-mg-venv/bin/python "$HERE/demo_server.py" --port "$PORT"
