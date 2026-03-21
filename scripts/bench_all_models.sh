#!/bin/bash
# Benchmark all locally available ROADMAP models with Rapid-MLX
# Usage: bash scripts/bench_all_models.sh

set -e

PORT=8100
RESULTS_DIR="/Users/raullenstudio/work/vllm-mlx/reports/benchmarks"
BENCH_SCRIPT="/Users/raullenstudio/work/vllm-mlx/scripts/benchmark_engines.py"
PYTHON="python3.12"
SERVER_CMD="$PYTHON -m vllm_mlx.server"

mkdir -p "$RESULTS_DIR"

# Model configs: "short_name|model_path|tool_parser|reasoning_parser"
MODELS=(
  "phi4-mini-14b|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-MLX-4bit|hermes|"
  "mistral-small-24b|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit|hermes|"
  "gemma3-12b|/Users/raullenstudio/.lmstudio/models/mlx-community/gemma-3-12b-it-qat-4bit|hermes|"
  "gpt-oss-20b|/Users/raullenstudio/.lmstudio/models/mlx-community/gpt-oss-20b-MXFP4-Q8|seed_oss|"
  "glm47-9b|/Users/raullenstudio/.lmstudio/models/mlx-community/GLM-4.7-4bit|glm47|"
  "qwen35-122b-a10b|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Qwen3.5-122B-A10B-Text-mxfp4-mlx|hermes|qwen3"
  "qwen3-coder-next-80b|/Users/raullenstudio/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-6bit|hermes|"
)

kill_server() {
  local pids
  pids=$(lsof -ti :$PORT 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  Killing server on port $PORT (PIDs: $pids)"
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 2
  fi
}

wait_for_server() {
  local max_wait=300  # 5 min max for large models
  local waited=0
  echo "  Waiting for server on port $PORT..."
  while [ $waited -lt $max_wait ]; do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
      echo "  Server ready after ${waited}s"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    if [ $((waited % 30)) -eq 0 ]; then
      echo "  Still waiting... (${waited}s)"
    fi
  done
  echo "  ERROR: Server did not start within ${max_wait}s"
  return 1
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name model_path tool_parser reasoning_parser <<< "$entry"

  output_file="$RESULTS_DIR/${name}.json"

  # Skip if already benchmarked
  if [ -f "$output_file" ]; then
    echo ">>> SKIP $name (already exists: $output_file)"
    continue
  fi

  echo ""
  echo "================================================================"
  echo ">>> BENCHMARKING: $name"
  echo ">>> Model: $model_path"
  echo ">>> Tool parser: $tool_parser, Reasoning parser: ${reasoning_parser:-none}"
  echo "================================================================"

  # Kill any existing server
  kill_server

  # Build server command
  SERVER_ARGS="--model $model_path --port $PORT --tool-call-parser $tool_parser"
  if [ -n "$reasoning_parser" ]; then
    SERVER_ARGS="$SERVER_ARGS --reasoning-parser $reasoning_parser"
  fi

  echo "  Starting: $SERVER_CMD $SERVER_ARGS"
  $SERVER_CMD $SERVER_ARGS > "/tmp/bench_server_${name}.log" 2>&1 &
  SERVER_PID=$!

  # Wait for server to be ready
  if ! wait_for_server; then
    echo "  FAILED to start server for $name. Skipping."
    kill_server
    continue
  fi

  # Run benchmark
  echo "  Running benchmark..."
  $PYTHON "$BENCH_SCRIPT" \
    --engine rapid-mlx \
    --rapid-mlx-port $PORT \
    --runs 3 \
    --output "$output_file" \
    2>&1 | tee "/tmp/bench_output_${name}.log" || {
      echo "  Benchmark failed for $name"
      kill_server
      continue
    }

  echo "  Results saved to $output_file"

  # Kill server
  kill_server

  echo ">>> DONE: $name"
done

echo ""
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "Results in: $RESULTS_DIR/"
echo "================================================================"
ls -la "$RESULTS_DIR/"
