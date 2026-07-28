#!/bin/bash
# Layer-B Tier-1 agent re-verification smoke.
#
# Run ON the Studio (real Apple Silicon + models + agent binaries). GitHub CI
# cannot do this — it has no Metal, no weights, and release-preflight skips
# every gate that needs a live `rapid-mlx serve`.
#
# Boots `rapid-mlx serve`, then drives the FOUR Tier-1 (flagship) agents —
# Claude Code, Codex, Hermes, Aider — through a real multi-step bug-fix task
# against the local model, and asserts each one actually made the test pass.
#
#   Usage:   RAPID_MLX_VENV=~/rapid-mlx-audit-venv ./agent_smoke.sh [model-alias]
#   Default: model-alias = qwen3.6-35b-8bit   (strong 8-bit — never 4-bit,
#            which confounds "weak model" with "broken integration")
#
# Exit code: 0 iff all four Tier-1 agents PASS. Non-zero blocks the release.
# It is non-destructive: it backs up ~/.codex and ~/.hermes config and restores
# them on exit (the Studio is a shared, in-use machine).
set -uo pipefail

export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
VENV="${RAPID_MLX_VENV:-$HOME/rapid-mlx-audit-venv}"
RMLX="$VENV/bin/rapid-mlx"
ALIAS="${1:-qwen3.6-35b-8bit}"
PORT="${RAPID_MLX_PORT:-8000}"
B="http://localhost:$PORT"
WORK="$HOME/agent-smoke-work"
LOG="$HOME/agent-smoke-serve.log"

CODEX_CFG="$HOME/.codex/config.toml"
HERMES_CFG="$HOME/.hermes/config.yaml"

cleanup() {
  pkill -9 -f "$VENV/bin/rapid-mlx serve" 2>/dev/null
  [ -f "$CODEX_CFG.smokebak" ]  && mv -f "$CODEX_CFG.smokebak"  "$CODEX_CFG"  2>/dev/null
  [ -f "$HERMES_CFG.smokebak" ] && mv -f "$HERMES_CFG.smokebak" "$HERMES_CFG" 2>/dev/null
  rm -rf "$WORK" "$LOG"
}
trap cleanup EXIT

fail() { echo "SMOKE-ABORT: $*" >&2; exit 3; }
[ -x "$RMLX" ] || fail "no rapid-mlx at $RMLX (set RAPID_MLX_VENV)"

echo "== versions =="
printf "  rapid-mlx  %s\n" "$($RMLX --version 2>&1 | head -1)"
for b in claude codex aider hermes; do
  command -v "$b" >/dev/null 2>&1 && printf "  %-9s  %s\n" "$b" "$($b --version 2>&1 | head -1)" \
                                  || printf "  %-9s  MISSING\n" "$b"
done
echo "== model: $ALIAS =="

# ---- boot serve ----------------------------------------------------------
pkill -f "$VENV/bin/rapid-mlx serve" 2>/dev/null; sleep 2
nohup "$RMLX" serve "$ALIAS" --port "$PORT" > "$LOG" 2>&1 &
for i in $(seq 1 48); do
  curl -s -m 3 "$B/v1/models" 2>/dev/null | grep -q '"id"' && { echo "serve READY (~$((i*5))s)"; break; }
  sleep 5
  [ "$i" = 48 ] && { tail -20 "$LOG"; fail "serve not ready in 240s"; }
done

# ---- the task: a buggy factorial + a failing test the agent must fix -----
seed_repo() {
  local d="$WORK/$1"; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 1
  git init -q; git config user.email a@b.c; git config user.name smoke
  printf 'def factorial(n):\n    result = 1\n    for i in range(1, n):   # off-by-one bug\n        result *= i\n    return result\n' > calc.py
  printf 'from calc import factorial\nassert factorial(5) == 120, factorial(5)\nassert factorial(6) == 720\nassert factorial(0) == 1\nprint("ALL PASS")\n' > test_calc.py
  git add -A; git commit -qm seed
}
# PASS iff the test now exits 0 (agent fixed the bug and it verifies)
verify() { cd "$WORK/$1" 2>/dev/null && python3 test_calc.py >/dev/null 2>&1 && echo PASS || echo FAIL; }

# Plain phrasing (no backticks / em-dash — keep the prompt boring so the
# check measures the integration, not prompt parsing).
TASK='Run python3 test_calc.py, it fails. Fix the bug in calc.py so all assertions pass, then re-run to confirm. Only edit calc.py.'

# The agents (codex especially) are non-deterministic, so give each up to 2
# attempts — a single miss must not flap the release gate.

# ---- Claude Code (env var, /v1/messages) ---------------------------------
R_CLAUDE=FAIL
for _try in 1 2; do
  seed_repo claude
  ANTHROPIC_BASE_URL="$B" ANTHROPIC_API_KEY=not-needed \
    timeout 260 claude -p "$TASK" --model "$ALIAS" --dangerously-skip-permissions >/dev/null 2>&1
  [ "$(verify claude)" = PASS ] && { R_CLAUDE=PASS; break; }
done

# ---- Codex (agents codex --setup writes ~/.codex/config.toml) ------------
[ -f "$CODEX_CFG" ] && cp "$CODEX_CFG" "$CODEX_CFG.smokebak"
"$RMLX" agents codex --setup >/dev/null 2>&1
R_CODEX=FAIL
for _try in 1 2; do
  seed_repo codex
  timeout 260 codex exec "$TASK" --model "$ALIAS" \
    --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check >/dev/null 2>&1
  [ "$(verify codex)" = PASS ] && { R_CODEX=PASS; break; }
done
[ -f "$CODEX_CFG.smokebak" ] && mv -f "$CODEX_CFG.smokebak" "$CODEX_CFG"

# ---- Hermes (agents hermes --setup; auto-writes context_length >= 64K) ----
[ -f "$HERMES_CFG" ] && cp "$HERMES_CFG" "$HERMES_CFG.smokebak"
"$RMLX" agents hermes --setup >/dev/null 2>&1
R_HERMES=FAIL
for _try in 1 2; do
  seed_repo hermes
  timeout 300 hermes chat -q "$TASK" -Q -m "$ALIAS" >/dev/null 2>&1
  [ "$(verify hermes)" = PASS ] && { R_HERMES=PASS; break; }
done
[ -f "$HERMES_CFG.smokebak" ] && mv -f "$HERMES_CFG.smokebak" "$HERMES_CFG"

# ---- Aider (env vars, LiteLLM openai/ prefix) ----------------------------
R_AIDER=FAIL
for _try in 1 2; do
  seed_repo aider
  OPENAI_API_BASE="$B/v1" OPENAI_API_KEY=not-needed \
    timeout 260 aider --model "openai/$ALIAS" --message "$TASK" \
    --yes-always --no-auto-commits --no-show-model-warnings calc.py >/dev/null 2>&1
  [ "$(verify aider)" = PASS ] && { R_AIDER=PASS; break; }
done

# ---- report --------------------------------------------------------------
echo
echo "== Tier-1 re-verification ($ALIAS) =="
printf "  claude-code  %s\n" "$R_CLAUDE"
printf "  codex-cli    %s\n" "$R_CODEX"
printf "  hermes       %s\n" "$R_HERMES"
printf "  aider        %s\n" "$R_AIDER"

for r in "$R_CLAUDE" "$R_CODEX" "$R_HERMES" "$R_AIDER"; do
  [ "$r" = PASS ] || { echo "RESULT: FAIL — a Tier-1 agent regressed; this blocks the release."; exit 1; }
done
echo "RESULT: PASS — all four Tier-1 agents verified on $ALIAS."
