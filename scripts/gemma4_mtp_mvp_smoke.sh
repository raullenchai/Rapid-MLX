#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/gemma4_mtp_mvp_smoke.sh — end-to-end smoke for `--spec-decode mtp`
# on Gemma 4, after stacking PR-1 (draft-k auto-tune), PR-2 (detect
# allowlist), and PR-3 (this branch: Google-official assistant inject).
#
# ── DO NOT RUN AS PART OF PR-3 CI ──────────────────────────────────────
# This script is the OPERATOR-run integration test. PR-3 lands the file;
# PR-3 does NOT run it. Running requires:
#   1. All three PRs (#987 draft-k tune, #988 detect allowlist, this
#      replacement PR-3) stacked locally.
#   2. Google's official assistant checkpoint downloaded:
#        huggingface-cli download google/gemma-4-12B-it-assistant \
#          --local-dir ~/rapid-mlx-staging/gemma-4-12B-it-assistant
#   3. The paired target ``mlx-community/gemma-4-12B-it-4bit`` cached.
#   4. Server-side wiring for ``--mtp-sidecar`` — the CLI flag does
#      NOT exist in main today. Adding the CLI arg + scheduler hookup
#      is a post-MVP follow-up scoped OUT of this PR (see PR body).
#      Until that lands, the script demonstrates the intended flow but
#      argparse will reject ``--mtp-sidecar``.
#
# The Gemma 4 inject in THIS PR is a real AssistantModel consumer:
# it instantiates the 4-layer drafter matching Google's safetensors
# layout, loads real weights, and computes forward using target's
# tail K/V. MVP caveats (chained multi-token drafts, centroid embedder,
# final-logit softcapping) are called out in the PR body.
# ────────────────────────────────────────────────────────────────────────
#
# Usage:
#   scripts/gemma4_mtp_mvp_smoke.sh /path/to/gemma-4-12B-it-assistant
#
# The path argument is a local dir carrying Google's assistant checkpoint
# (config.json + model.safetensors). Passing the HF repo id
# ``google/gemma-4-12B-it-assistant`` also works — inject_mtp_support
# will snapshot_download it on demand.
#
# What this script does:
#   1. Boots ``rapid-mlx serve --model gemma-4-12b-4bit --enable-mtp
#      --spec-decode mtp --mtp-sidecar $1 --mtp-num-draft-tokens 4``
#      and waits for the "Server ready" health probe.
#   2. Sends a fixed prompt to /v1/chat/completions with temperature 0
#      and captures the FINAL completion text (choices[0].message.content).
#   3. Second run: same but WITHOUT ``--enable-mtp`` — the baseline.
#   4. Diffs the two FINAL completion strings byte-for-byte. A mismatch
#      fails the smoke — the MTP lossless contract requires bit-
#      identical FINAL TEXT at temp=0. This smoke does NOT compare
#      per-token streaming order (see codex round-13 note): a
#      tokenizer / streaming-order regression that produces the same
#      final text will pass this smoke; catching those regressions
#      requires a separate SSE / event-stream contract test.
#   5. Reports accept rate + tok/s from Prometheus
#      ``rapid_mlx_spec_decode_*``.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
SIDECAR_PATH="${1:-}"
if [[ -z "${SIDECAR_PATH}" ]]; then
  echo "Usage: $0 /path/to/gemma-4-12B-it-assistant" >&2
  echo "       $0 google/gemma-4-12B-it-assistant     # HF repo id also accepted" >&2
  exit 2
fi
# Accept either a local directory OR a local safetensors file OR an
# HF repo id ('owner/name'). Existing local paths (with or without
# a slash) are accepted first so bare names like
# 'gemma-4-12B-it-assistant' in the working directory don't get
# misclassified. Only if the argument is neither an existing path
# nor an HF-repo-id-shape do we fail fast; anything else is passed
# to inject_mtp_support which surfaces a family-specific error.
if [[ -d "${SIDECAR_PATH}" ]] || [[ -f "${SIDECAR_PATH}" ]]; then
  : # local path — accept
elif [[ "${SIDECAR_PATH}" == */* ]]; then
  : # looks like HF repo id — accept, snapshot_download will handle it
else
  echo "Sidecar '${SIDECAR_PATH}' is neither an existing local path nor an HF repo id (owner/name)." >&2
  exit 2
fi

MODEL_ALIAS="${RAPID_MLX_MODEL:-gemma-4-12b-4bit}"
PORT="${RAPID_MLX_PORT:-8990}"
PROMPT='{"model":"'"${MODEL_ALIAS}"'","messages":[{"role":"user","content":"Explain in exactly 40 words why the sky appears blue at midday."}],"temperature":0,"max_tokens":80}'
BASE_URL="http://127.0.0.1:${PORT}"

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

# ── Preflight: --mtp-sidecar must be a recognized CLI arg ─────────────
# Codex round-6 nit fix: this script wires ``--mtp-sidecar`` into the
# rapid-mlx serve invocation, but that flag does NOT exist on main as
# of this PR (see header docstring). Fail fast with a clear message
# instead of letting argparse eat the flag and print an unrelated
# usage error deep in ``WORKDIR/server_mtp.log``. The operator can
# skip this gate by exporting ``SKIP_MTP_SIDECAR_PREFLIGHT=1`` on a
# build that DOES ship the flag.
#
# Codex round-8 nit fix: defer WORKDIR creation until AFTER preflight,
# so a build that lacks the flag doesn't leak a temp dir on every
# expected-failure invocation.
if [[ "${SKIP_MTP_SIDECAR_PREFLIGHT:-0}" != "1" ]]; then
  if ! "${RAPID_MLX_BIN}" serve --help 2>&1 | grep -q -- '--mtp-sidecar'; then
    cat <<EOF >&2
[smoke] Preflight failed: this rapid-mlx build does not advertise --mtp-sidecar.
[smoke] The smoke script depends on the server-side wiring PR that lands the
[smoke] CLI arg + scheduler hook (post-MVP TODO in this PR's body). Once that
[smoke] PR merges, re-run this smoke; or export SKIP_MTP_SIDECAR_PREFLIGHT=1
[smoke] to force it on a hand-patched build.
EOF
    exit 4
  fi
fi

# Deferred until after preflight so a rejected-preflight run doesn't
# leak a stale temp directory.
WORKDIR="$(mktemp -d -t gemma4-mtp-smoke.XXXXXX)"
echo "[smoke] Workdir: ${WORKDIR}"

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

  # ── Codex round-3 fix: install a RETURN-time trap so ANY exit
  # path (curl failure, timeout, set -e trip, etc.) tears down the
  # background server. Without this, `set -e` short-circuits before
  # `kill` at the bottom and leaves rapid-mlx serve running on the
  # port for the NEXT boot_and_prompt call to collide with.
  #
  # Round-5 nit: use a cleanup function that reads the pid from a
  # module-scope variable instead of interpolating pid into the trap
  # string. Safer if pid is unset/malformed.
  # ─────────────────────────────────────────────────────────────
  _boot_current_pid=${pid}
  _cleanup_boot() {
    if [[ -n "${_boot_current_pid:-}" ]]; then
      kill -TERM "${_boot_current_pid}" 2>/dev/null || true
      wait "${_boot_current_pid}" 2>/dev/null || true
    fi
    trap - RETURN
  }
  trap _cleanup_boot RETURN

  # Wait up to 180s for /healthz.
  local start_ts=$SECONDS
  while :; do
    if curl -fsS "${BASE_URL}/healthz" > /dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[smoke:${mode}] server died before ready. Last 40 lines:" >&2
      tail -40 "${server_log}" >&2
      return 4  # trap fires — server pid is already gone, kill is a no-op
    fi
    if (( SECONDS - start_ts > 180 )); then
      echo "[smoke:${mode}] server did not become ready in 180s. Last 40 lines:" >&2
      tail -40 "${server_log}" >&2
      return 4  # trap tears down the still-running server
    fi
    sleep 1
  done
  echo "[smoke:${mode}] Server ready in $((SECONDS - start_ts))s"

  # Fixed prompt, temp=0, max_tokens=80. Timeout 120s.
  # curl -f exits non-zero on 5xx — with `set -e` at file scope, a
  # failure here would blow up the function. The RETURN trap will
  # tear down the server before the outer script exits.
  curl -fsS \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "${PROMPT}" \
    --max-time 120 \
    "${BASE_URL}/v1/chat/completions" \
    > "${out_body}"

  # Scrape metrics after generation.
  curl -fsS "${BASE_URL}/metrics" > "${out_metrics}" || true

  # Explicit teardown at happy path — the RETURN trap will also fire
  # here (harmless — kill on a dead pid is a no-op).
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
echo "═════ LOSSLESS DIFF (FINAL CONTENT ONLY) ═════"
echo "Note (codex round-13): this diff compares FINAL completion text,"
echo "not per-token stream. A tokenizer / streaming-order regression"
echo "that yields the same final text passes here. A separate SSE"
echo "contract smoke is the right tool for streaming-order equality."
LOSSLESS_OK=1
if diff -u "${CONTENT_BASELINE}" "${CONTENT_MTP}"; then
  echo "PASS: mtp completion == baseline completion (byte-equal final text)."
else
  echo "FAIL: mtp completion differs from baseline — MTP lossless contract broken." >&2
  LOSSLESS_OK=0
fi

echo ""
extract_mtp_stats "${METRICS_MTP}"
echo ""
extract_mtp_stats "${METRICS_BASELINE}"

echo ""

# ── Cleanup / preserve decision ──────────────────────────────────────
# Codex round-7 nit fix: previously we ALWAYS preserved WORKDIR, which
# accretes multi-megabyte server logs + response bodies across repeated
# operator smokes. Default is now: preserve on failure (so the operator
# can eyeball logs), clean on success. Opt out with
# ``PRESERVE_WORKDIR=1`` to keep the dir even after a green run
# (useful when comparing back-to-back baseline vs mtp telemetry).
if (( LOSSLESS_OK == 0 )); then
  echo "[smoke] Workdir PRESERVED at ${WORKDIR} (lossless diff failed)."
  echo "[smoke] FAIL: lossless diff mismatched — see diff above." >&2
  exit 1
fi

if [[ "${PRESERVE_WORKDIR:-0}" == "1" ]]; then
  echo "[smoke] Workdir preserved (PRESERVE_WORKDIR=1): ${WORKDIR}"
else
  # rm -rf under mktemp -d output: safe because $WORKDIR was produced
  # by mktemp -t under the OS temp dir; guard anyway.
  if [[ -n "${WORKDIR}" && "${WORKDIR}" == */gemma4-mtp-smoke.* && -d "${WORKDIR}" ]]; then
    rm -rf "${WORKDIR}"
    echo "[smoke] Workdir cleaned (PRESERVE_WORKDIR=1 to keep): ${WORKDIR}"
  else
    echo "[smoke] Workdir untouched (unexpected path shape): ${WORKDIR}"
  fi
fi
echo "[smoke] Done."
