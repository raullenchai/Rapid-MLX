#!/bin/bash
# Batch eval runner — runs ALL suites for all text LLMs
# Usage: bash evals/run_all_models.sh [suite1 suite2 ...]
# Examples:
#   bash evals/run_all_models.sh                    # all suites
#   bash evals/run_all_models.sh speed tool_calling # specific suites
# NOTE: Model paths below are machine-specific. Update them to match your
#       local model directory before running.
# No set -e: server kill/wait returns non-zero which is expected

PYTHON=python3.12
CLI_CMD="from vllm_mlx.cli import main; import sys; sys.argv = ['vllm-mlx'] + sys.argv[1:]; main()"
PORT=8000
EVAL_CMD="$PYTHON evals/run_eval.py"

# Suites to run (all by default, or from command line)
if [ $# -gt 0 ]; then
  SUITES="$*"
else
  SUITES="speed tool_calling coding reasoning general"
fi

# Model configs: name|path|parser|quantization
declare -a MODELS=(
  "Qwen3-0.6B-4bit|/Users/raullenstudio/.lmstudio/models/mlx-community/Qwen3-0.6B-MLX-4bit|hermes|4bit"
  "GLM-4.7-4bit|/Users/raullenstudio/.lmstudio/models/mlx-community/GLM-4.7-4bit|glm47|4bit"
  "GPT-OSS-20B-mxfp4-q8|/Users/raullenstudio/.lmstudio/models/mlx-community/gpt-oss-20b-MXFP4-Q8|seed_oss|mxfp4-q8"
  "MiniMax-M2.5-4bit|/Users/raullenstudio/.lmstudio/models/lmstudio-community/MiniMax-M2.5-MLX-4bit|minimax|4bit"
  "Qwen3.5-35B-A3B-4bit|/Users/raullenstudio/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit|hermes|4bit"
  "Qwen3.5-35B-A3B-8bit|/Users/raullenstudio/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-8bit|hermes|8bit"
  "Qwen3-Coder-Next-4bit|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-4bit|hermes|4bit"
  "Qwen3-Coder-Next-6bit|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-6bit|hermes|6bit"
  "Qwen3.5-122B-A10B-mxfp4|/Users/raullenstudio/.lmstudio/models/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx|hermes|mxfp4"
  "Qwen3.5-122B-A10B-8bit|/Users/raullenstudio/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-8bit|hermes|8bit"
  # Requested by community — download and uncomment to eval:
  # "Mistral-Small-3.2-4bit|<path>|hermes|4bit"
  # "Devstral-Small-4bit|<path>|hermes|4bit"
  # "GLM-4.5-Air-4bit|<path>|glm47|4bit"
  # "Nemotron-Nano-30B-4bit|<path>|hermes|4bit"
  # "Qwen3.5-4B-4bit|<path>|hermes|4bit"
  # "Qwen3.5-9B-4bit|<path>|hermes|4bit"
)

start_server() {
  local model_path="$1"
  local parser="$2"
  echo "  Starting server: $(basename "$model_path") (parser=$parser)..."
  $PYTHON -c "$CLI_CMD" serve "$model_path" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser "$parser" &
  SERVER_PID=$!

  for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" | grep -q "healthy"; then
      echo "  Server ready (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "  ERROR: Server failed to start within 240s"
  kill $SERVER_PID 2>/dev/null
  return 1
}

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    SERVER_PID=""
  fi
  lsof -ti:$PORT | xargs kill 2>/dev/null || true
  sleep 3
}

echo "========================================"
echo "vllm-mlx Full Model Evaluation"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "Suites: $SUITES"
echo ""

TOTAL_START=$(date +%s)

for model_config in "${MODELS[@]}"; do
  IFS='|' read -r name path parser quant <<< "$model_config"

  # Skip if model path doesn't exist
  if [ ! -d "$path" ]; then
    echo "SKIP: $name (path not found: $path)"
    echo ""
    continue
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Model: $name ($quant, parser=$parser)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  stop_server

  if start_server "$path" "$parser"; then
    $EVAL_CMD \
      --model "$name" \
      --parser "$parser" \
      --quantization "$quant" \
      --suite $SUITES \
      --server-flags "--enable-auto-tool-choice --tool-call-parser $parser"
    echo ""
  else
    echo "  SKIPPED: $name (server failed to start)"
    echo ""
  fi

  stop_server
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
MINUTES=$((TOTAL_ELAPSED / 60))

echo "========================================"
echo "All evals complete in ${MINUTES}m ${TOTAL_ELAPSED}s"
echo "========================================"
echo ""
echo "Results:"
ls -la evals/results/*.json
echo ""
echo "Regenerate scorecard with: python3.12 evals/generate_scorecard.py"
