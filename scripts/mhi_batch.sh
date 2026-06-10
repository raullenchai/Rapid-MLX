#!/bin/bash
# ACI Batch Runner — run HumanEval + MMLU across all matrix models
# TAU-bench skipped (too slow with local user sim, needs GPT-4o for proper runs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Model definitions: name|path|tool_parser
MODELS=(
    "qwopus-27b-4bit|/Users/raullenstudio/.cache/huggingface/hub/models--Jackrong--MLX-Qwopus3.5-27B-v3-4bit/snapshots/d399209470abffa6b45678c53a910f869b18b2f2|hermes"
    "deepseek-r1-32b-4bit|/Users/raullenstudio/.cache/huggingface/hub/models--mlx-community--DeepSeek-R1-Distill-Qwen-32B-4bit/snapshots/4e0d3848a0ad8f9fb54638891e4928f04fcca978|hermes"
    "llama-70b|/Volumes/Extreme SSD/Models/Llama-3.3-70B-Instruct-4bit|llama"
)

PORT=8000
RESULTS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r label model_path parser <<< "$entry"

    echo ""
    echo "================================================================"
    echo "  Loading: $label"
    echo "  Path: $model_path"
    echo "  Parser: $parser"
    echo "================================================================"

    # Kill any existing server
    pkill -f "vllm_mlx.server" 2>/dev/null || true
    sleep 3

    # Start server
    python3.12 -m vllm_mlx.server \
        --model "$model_path" \
        --tool-call-parser "$parser" \
        --port $PORT \
        2>/dev/null &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "  Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            echo "  Server ready!"
            break
        fi
        sleep 2
    done

    if ! curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  ERROR: Server failed to start for $label"
        kill $SERVER_PID 2>/dev/null || true
        continue
    fi

    # Run ACI eval (humaneval + mmlu only, skip tau)
    echo "  Running ACI eval..."
    python3.12 scripts/aci_eval.py \
        --base-url "http://localhost:$PORT/v1" \
        --label "$label" \
        --suite humaneval 2>&1 | grep -E "^\[|Score:|PASS|FAIL"

    python3.12 scripts/aci_eval.py \
        --base-url "http://localhost:$PORT/v1" \
        --label "$label" \
        --suite mmlu 2>&1 | grep -E "^\[|Score:|PASS|FAIL"

    echo "  Done: $label"

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
done

echo ""
echo "================================================================"
echo "  ACI Batch Complete"
echo "  Results in reports/aci/"
echo "================================================================"
ls -la reports/aci/*.json 2>/dev/null
