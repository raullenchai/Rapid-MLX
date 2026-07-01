#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/gemma4_mtp_mvp_smoke.sh — end-to-end smoke for `--spec-decode mtp`
# on Gemma 4, after stacking PR-1 (draft-k auto-tune), PR-2 (detect
# allowlist), and PR-3 (this branch: inject + GGUF conversion).
#
# ── DO NOT RUN AS PART OF PR-3 CI ──────────────────────────────────────
# This script is the OPERATOR-run integration test the task doc
# specifies. PR-3 lands the file; PR-3 does NOT run it. The Gemma 4
# MTP inject in this PR is scaffold-only (see the
# ``gemma4-assistant`` architecture note in
# ``vllm_mlx/spec_decode/mtp/gemma4_inject.py``), so running this
# smoke against the current inject WILL either:
#   • Fail during boot (default fail-closed sidecar refusal), or
#   • Boot but emit zero-speedup drafts under ``allow_random_init``.
#
# The operator runs this AFTER the follow-up AssistantModel PR lands,
# which wires the actual gemma4-assistant sidecar consumer.
# ────────────────────────────────────────────────────────────────────────
#
# Usage:
#   scripts/gemma4_mtp_mvp_smoke.sh /path/to/gemma-4-12b-mtp-4bit
#
# The path argument is the local staging dir produced by
# scripts/convert_gemma4_mtp_gguf.py. It must contain
# model-mtp.safetensors + config.json.
#
# What this script does:
#   1. Boots ``rapid-mlx serve --model gemma-4-12b-4bit --enable-mtp
#      --spec-decode mtp --mtp-sidecar $1 --mtp-num-draft-tokens 4``
#      and waits for the "Server ready" health probe.
#   2. Sends a fixed prompt to /v1/chat/completions with temperature 0
#      and captures the raw token stream.
#   3. Second run: same but WITHOUT ``--enable-mtp`` — the baseline.
#   4. Diffs the two token streams byte-for-byte. A mismatch fails the
#      smoke — the MTP lossless contract requires bit-identical output
#      at temp=0.
#   5. Reports accept rate + tok/s from Prometheus
#      ``rapid_mlx_spec_decode_*``.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
SIDECAR_PATH="${1:-}"
if [[ -z "${SIDECAR_PATH}" ]]; then
  echo "Usage: $0 /path/to/gemma-4-12b-mtp-4bit" >&2
  exit 2
fi
if [[ ! -d "${SIDECAR_PATH}" ]]; then
  echo "Sidecar path does not exist: ${SIDECAR_PATH}" >&2
  exit 2
fi

MODEL_ALIAS="${RAPID_MLX_MODEL:-gemma-4-12b-4bit}"
PORT="${RAPID_MLX_PORT:-8990}"
PROMPT='{"model":"'"${MODEL_ALIAS}"'","messages":[{"role":"user","content":"Explain in exactly 40 words why the sky appears blue at midday."}],"temperature":0,"max_tokens":80}'
BASE_URL="http://127.0.0.1:${PORT}"

WORKDIR="$(mktemp -d -t gemma4-mtp-smoke.XXXXXX)"
echo "[smoke] Workdir: ${WORKDIR}"

# Locate rapid-mlx binary.
RAPID_MLX_BIN="${RAPID_MLX_BIN:-$(command -v rapid-mlx || true)}"
if [[ -z "${RAPID_MLX_BIN}" ]]; then
  echo "[smoke] rapid-mlx not on PATH; falling back to .venv/bin/rapid-mlx"
  RAPID_MLX_BIN="$(pwd)/.venv/bin/rapid-mlx"
fi
if [[ ! -x "${RAPID_MLX_BIN}" ]]; then
  echo "[smoke] rapid-mlx binary not found. Set RAPID_MLX_BIN or activate the .venv." >&2
  exit 3
fi

# ── boot_and_prompt ──────────────────────────────────────────────────
# Boot rapid-mlx, wait for ready, send the prompt, dump the completion,
# then kill the server. Args:
#   $1 = "mtp" | "baseline"  — determines the MTP flags
#   $2 = output file for the completion body
#   $3 = output file for the /metrics scrape after generation
boot_and_prompt() {
  local mode="$1"
  local out_body="$2"
  local out_metrics="$3"
  local extra_args=()

  if [[ "${mode}" == "mtp" ]]; then
    extra_args=(
      --enable-mtp
      --spec-decode mtp
      --mtp-sidecar "${SIDECAR_PATH}"
      --mtp-num-draft-tokens 4
    )
  fi

  local server_log="${WORKDIR}/server_${mode}.log"
  echo "[smoke:${mode}] Starting server on :${PORT}..."
  "${RAPID_MLX_BIN}" serve \
    --model "${MODEL_ALIAS}" \
    --port "${PORT}" \
    "${extra_args[@]}" \
    > "${server_log}" 2>&1 &
  local pid=$!
  echo "[smoke:${mode}] server pid=${pid}"

  # Wait up to 180s for /healthz.
  local start_ts=$SECONDS
  while :; do
    if curl -fsS "${BASE_URL}/healthz" > /dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[smoke:${mode}] server died before ready. Last 40 lines:" >&2
      tail -40 "${server_log}" >&2
      exit 4
    fi
    if (( SECONDS - start_ts > 180 )); then
      echo "[smoke:${mode}] server did not become ready in 180s. Last 40 lines:" >&2
      tail -40 "${server_log}" >&2
      kill -TERM "${pid}" 2>/dev/null || true
      exit 4
    fi
    sleep 1
  done
  echo "[smoke:${mode}] Server ready in $((SECONDS - start_ts))s"

  # Fixed prompt, temp=0, max_tokens=80. Timeout 120s.
  curl -fsS \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "${PROMPT}" \
    --max-time 120 \
    "${BASE_URL}/v1/chat/completions" \
    > "${out_body}"

  # Scrape metrics after generation.
  curl -fsS "${BASE_URL}/metrics" > "${out_metrics}" || true

  kill -TERM "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  echo "[smoke:${mode}] Server stopped."
}

# ── Extract completion token stream ──────────────────────────────────
extract_content() {
  # Portable JSON extraction — prefer jq if present, otherwise python.
  if command -v jq > /dev/null 2>&1; then
    jq -r '.choices[0].message.content // .choices[0].delta.content // ""' "$1"
  else
    python3 -c "import json, sys; d=json.load(open(sys.argv[1])); \
print((d.get('choices') or [{}])[0].get('message', {}).get('content', ''))" "$1"
  fi
}

# ── Extract accept rate + tok/s from Prometheus ──────────────────────
extract_mtp_stats() {
  local metrics_file="$1"
  echo "─── MTP telemetry (${metrics_file##*/}) ───"
  grep -E '^rapid_mlx_spec_decode_(attempts|accepts|accept_ratio|tokens_saved)' "${metrics_file}" || \
    echo "  (no rapid_mlx_spec_decode_* series found)"
  echo "─── decode tok/s ───"
  grep -E '^rapid_mlx_.*(decode|tok).*_seconds|^rapid_mlx_tokens_per_second' "${metrics_file}" || \
    echo "  (no throughput series found)"
}

# ── Run both configurations ──────────────────────────────────────────
BODY_MTP="${WORKDIR}/completion_mtp.json"
BODY_BASELINE="${WORKDIR}/completion_baseline.json"
METRICS_MTP="${WORKDIR}/metrics_mtp.txt"
METRICS_BASELINE="${WORKDIR}/metrics_baseline.txt"

boot_and_prompt "mtp"      "${BODY_MTP}"      "${METRICS_MTP}"
boot_and_prompt "baseline" "${BODY_BASELINE}" "${METRICS_BASELINE}"

# ── Lossless diff ────────────────────────────────────────────────────
CONTENT_MTP="${WORKDIR}/content_mtp.txt"
CONTENT_BASELINE="${WORKDIR}/content_baseline.txt"
extract_content "${BODY_MTP}"      > "${CONTENT_MTP}"
extract_content "${BODY_BASELINE}" > "${CONTENT_BASELINE}"

echo ""
echo "═════ LOSSLESS DIFF ═════"
if diff -u "${CONTENT_BASELINE}" "${CONTENT_MTP}"; then
  echo "PASS: mtp completion == baseline completion (byte-equal)."
else
  echo "FAIL: mtp completion differs from baseline — MTP lossless contract broken." >&2
  # Do not exit — still emit telemetry so operator sees the accept rate.
fi

echo ""
extract_mtp_stats "${METRICS_MTP}"
echo ""
extract_mtp_stats "${METRICS_BASELINE}"

echo ""
echo "[smoke] Workdir preserved: ${WORKDIR}"
echo "[smoke] Done."
