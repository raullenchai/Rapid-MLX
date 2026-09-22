#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One-command initialization of a Vast.ai NVIDIA instance as a serving target.
# Run INSIDE the instance (ssh root@<vast-ip>, instance image: CUDA 12.1+ / Ubuntu 22.04).
#
#   bash setup_vast.sh <marvins-garden-repo-tar-or-git-url>
#
# Steps: deps -> llama.cpp at a LOCKED commit (ternary-GGUF custom format,
# upstream risk #29058/#27127 — do not bump without re-running the parity
# gate) -> convert GGUF -> llama-server (systemd, port 8080) -> verify.
# Public exposure is NOT done here: run cloudflared on the instance AFTER the
# parity gate passes (see docs/engineering/operations/vast-ai-serving-runbook.md).
set -euo pipefail
SRC="${1:?usage: setup_vast.sh <repo tarball or git url>}"
LLAMA_COMMIT="${LLAMA_COMMIT:-b4736}"
WORK=/opt/marvin
mkdir -p "$WORK"

echo "== 1/5 system deps =="
apt-get update -qq && apt-get install -y -qq build-essential cmake git curl python3-venv aria2 >/dev/null

echo "== 2/5 llama.cpp @ $LLAMA_COMMIT (locked) =="
[ -d "$WORK/llama.cpp" ] || git clone -q https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
cd "$WORK/llama.cpp" && git fetch -q origin && git checkout -q "$LLAMA_COMMIT"
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >/dev/null
cmake --build build --config Release -j"$(nproc)" --target llama-server llama-cli >/dev/null

echo "== 3/5 fetch model + converted ternary GGUF =="
# The ternary-GGUF conversion is produced by the packaging job in
# bench/marvins_garden (see runbook). Pull from the project release bucket:
[ -f "$WORK/marvin-v15c.gguf" ] || { echo "place marvin-v15c.gguf in $WORK (see runbook packaging section)"; exit 3; }

echo "== 4/5 llama-server (systemd) =="
cat > /etc/systemd/system/llama-marvin.service <<UNIT
[Unit]
Description=llama-server for Marvin's Garden (NVIDIA serving path)
After=network-online.target
[Service]
ExecStart=$WORK/llama.cpp/build/bin/llama-server -m $WORK/marvin-v15c.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 --parallel 1
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now llama-marvin

echo "== 5/5 selfcheck (proxy contract) =="
python3 -m venv "$WORK/venv" 2>/dev/null || true
. "$WORK/venv/bin/activate" && pip install -q requests
sleep 10
curl -s http://127.0.0.1:8080/health && echo " <- llama-server up"

cat <<'NEXT'

NEXT STEP (do not skip): parity gate from the Mac —
  python scripts/parity_gate_vast.py --base http://<vast-ip>:8080
PASS required before any public exposure. Then:
  cloudflared tunnel --url http://127.0.0.1:8123   # after starting classify_proxy --serve
NEXT
