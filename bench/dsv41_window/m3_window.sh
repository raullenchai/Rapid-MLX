#!/bin/zsh
# DSV4.1 M3 Ultra GPU window: full-model A/B per branch vs main, token parity,
# 16k decode, prefill 1k/4k, batched lanes 1/4/8, draft, server 1/8-concurrent,
# then stress pr_validate for #4982/#4987/#4986.
#
# usage: m3_window.sh [STEP...]      (default: all steps, in order)
# steps: preflight gates server ab long prefill draft validate report
# Resumable: a (label,phase) already present in results.jsonl is skipped.
# Holds /var/tmp/rapid-mlx-large-model.lock for the whole window (re-execs under lockf).
# Mirror: M5 /private/tmp/rapid-mlx-dsv41-speed-m5/studio-scripts/ and branch vector/dsv41-bench-harness.
set -u
B=/private/tmp/rapid-mlx-dsv41-speed-m3
W=/Users/raullenstudio/orca/workspaces/rapid-mlx
M=$HOME/.cache/huggingface/hub/models--rapid-mlx--DeepSeek-V4.1-Flash-1017-MLX/snapshots/c95bb8ad2bf9b2bf65fcab9b9c3dbbdec24dcb39
PY=$W/dsv41-batched-decode/.venv/bin/python     # one interpreter (mlx 0.32.4, py3.12) for every tree
OUT=$B/results.jsonl
SRV=$B/server.jsonl
PORT=18461
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TARGET_REV=c95bb8ad MTP_REV=c95bb8ad FULL=1
LOCK=/var/tmp/rapid-mlx-large-model.lock

if [[ -z "${M3_WINDOW_LOCKED:-}" ]]; then
  echo "[window] waiting for $LOCK"
  exec env M3_WINDOW_LOCKED=1 /usr/bin/lockf -k $LOCK /bin/zsh $0 "$@"
fi
mkdir -p $B/logs
cd $B

typeset -A T
T=(main dsv41-main-ref pr4982 dsv41-batched-decode ring dsv41-ring-rollback head dsv41-head-gemv
   fusion dsv41-decode-fusion overhead dsv41-decode-overhead gate dsv41-gate-gemv gather dsv41-sparse-gather
   fmoe dsv41-fused-moe fhc dsv41-fused-hc mma dsv41-mma-attn rq dsv41-rope-quant
   prefill dsv41-prefill draft dsv41-draft-numerics stackam dsv41-stack-am combo dsv41-combo)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $B/logs/window.log; }
have() { local ph=$2; [[ $ph == prefill ]] && ph=prefill_raw; [[ -f $OUT ]] && grep -q "\"label\": \"$1\", .*\"phase\": \"$ph\"" $OUT; }
rev() { echo "$(git -C $W/${T[$1]} rev-parse --short HEAD)$(git -C $W/${T[$1]} diff --quiet HEAD || echo -dirty)"; }

# ab NAME LABEL PHASE... [env VAR=..]: one model load per call
ab() {
  local name=$1 label=$2; shift 2
  local todo=() p
  for p in "$@"; do have $label $p || todo+=$p; done
  (( ${#todo} )) || { log "skip $label (done)"; return 0; }
  log "ab $label tree=$name@$(rev $name) phases=${todo[*]} ${ABENV:-}"
  env ${=ABENV:-} PYTHONPATH=$W/${T[$name]} nice -n 0 $PY $B/harness/ab.py $M $label $OUT "${todo[@]}" >> $B/logs/$label.log 2>&1 \
    || log "FAIL $label (see logs/$label.log)"
}

wait_ready() {
  local i
  for i in {1..240}; do
    curl -fs -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0
    kill -0 $1 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# serve NAME LABEL LANES CONC...: start server from the tree, run srvbench per concurrency, stop it
serve() {
  local name=$1 label=$2 lanes=$3; shift 3
  local c todo=()
  for c in "$@"; do grep -q "\"label\": \"$label\", \"concurrency\": $c," $SRV 2>/dev/null || todo+=$c; done
  (( ${#todo} )) || { log "skip server $label (done)"; return 0; }
  log "serve $label tree=$name@$(rev $name) lanes=$lanes conc=${todo[*]}"
  (cd $W/${T[$name]} && RAPID_MLX_DSV41_LANES=$lanes PYTHONPATH=$W/${T[$name]} exec $PY -m rapid_mlx.cli serve deepseek-v41-flash-1017 \
     --host 127.0.0.1 --port $PORT) > $B/logs/serve-$label.log 2>&1 &
  local pid=$!
  if wait_ready $pid; then
    for c in "${todo[@]}"; do
      $PY $B/harness/srvbench.py http://127.0.0.1:$PORT deepseek-v41-flash-1017 $label $SRV \
        --prompt-file $B/prompt1008.txt --concurrency $c --max-tokens 256 --rounds 2 >> $B/logs/srv-$label.log 2>&1 \
        || log "FAIL srvbench $label c=$c"
    done
  else
    log "FAIL server $label did not become ready (logs/serve-$label.log)"
  fi
  kill -INT $pid 2>/dev/null; sleep 10; kill -0 $pid 2>/dev/null && kill -TERM $pid; wait $pid 2>/dev/null
  sleep 5
}

step_preflight() {
  log "preflight"
  { date; uname -a; sysctl -n machdep.cpu.brand_string hw.memsize; vm_stat | head -5; df -h / | tail -1
    pmset -g therm 2>/dev/null | tail -2
    echo "--- big processes (rss GB > 4)"; ps -axo pid,rss,comm | awk '$2>4194304 {printf "%s %.1fG %s\n",$1,$2/1048576,$3}'
    echo "--- port $PORT"; lsof -nP -iTCP:$PORT -sTCP:LISTEN
    for n in ${(k)T}; do echo "$n ${T[$n]} $(rev $n)"; done
    $PY -c "import mlx.core as mx; print('mlx', mx.__version__)"; } >> $B/logs/preflight.log 2>&1
  [[ -s $B/prompt1008.txt ]] || $PY $B/harness/mkprompt.py $M $B/prompt1008.txt 1008 >> $B/logs/preflight.log 2>&1
  tail -40 $B/logs/preflight.log
}

# PR gates first: #4986 head GEMV, #4987 ring, #4982 lanes (decode + parity)
step_gates() {
  ab main main-r1 decode tf tokens
  ab head head-r1 decode tf tokens
  ab ring ring-r1 decode tf tokens
  ab pr4982 pr4982-r1 decode tf tokens batched
  ab main main-r2 decode
  ab head head-r2 decode
  $PY $B/harness/tfcmp.py $B main-r1 head-r1 ring-r1 pr4982-r1 | tee -a $B/logs/window.log
}

# Server-level, same 1008-token prompt as the reference: single stream (draft on = product lane),
# single stream draft off (lanes>1, one client), 8 concurrent.
step_server() {
  serve main srv-main 1 1 8
  serve pr4982 srv-4982-l1 1 1
  serve pr4982 srv-4982-l8 8 1 8
}

step_ab() {
  local r n
  for r in 1 2; do
    for n in fusion overhead gate gather fmoe fhc mma rq stackam combo; do ab $n $n-r$r decode tf; done
    [[ $r == 1 ]] || ab main main-r3 decode
  done
  $PY $B/harness/tfcmp.py $B main-r1 fusion-r1 overhead-r1 gate-r1 gather-r1 fmoe-r1 fhc-r1 mma-r1 rq-r1 stackam-r1 combo-r1 | tee -a $B/logs/window.log
}

step_long() { local n; for n in main head stackam; do ABENV="DECODE_CTX=16384" ab $n $n-16k decode; done; }
step_prefill() { local n; for n in main prefill stackam; do ab $n $n-pf prefill stream_prefill; done; }
step_draft() { local n; for n in main draft prefill head stackam; do ab $n $n-draft draft; done; }

# stress pr_validate (no PR_VALIDATE_NO_STRESS); each in its own PR worktree + 3.12 venv, sequential
step_validate() {
  local pr tree
  for pr tree in 4982 dsv41-batched-decode 4987 dsv41-ring-rollback 4986 dsv41-head-gemv; do
    if [[ -f $B/logs/validate-$pr.done ]]; then log "skip validate $pr"; continue; fi
    while pgrep -f "pr_validate" >/dev/null || lsof -nP -iTCP:8451 -sTCP:LISTEN >/dev/null 2>&1; do log "pr_validate/8451 busy; waiting"; sleep 60; done
    log "validate #$pr in $tree@$(git -C $W/$tree rev-parse --short HEAD)"
    (cd $W/$tree && unset PR_VALIDATE_NO_STRESS && .venv/bin/python -m scripts.pr_validate.pr_validate $pr) > $B/logs/validate-$pr.log 2>&1
    log "validate #$pr rc=$? verdict: $(grep -E 'MERGE-SAFE|NOT MERGE|BLOCK' $B/logs/validate-$pr.log | tail -1)"
    touch $B/logs/validate-$pr.done
  done
}

step_report() { $PY $B/harness/summarize.py $OUT $SRV | tee $B/logs/summary.txt; }

steps=("$@"); (( ${#steps} )) || steps=(preflight gates server ab long prefill draft validate report)
for s in "${steps[@]}"; do log "== step $s"; step_$s; done
log "WINDOW DONE"
