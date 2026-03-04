# vllm-mlx Model Evaluations

Standardized evaluation framework for comparing LLM performance on Apple Silicon via vllm-mlx.

**Results**: See [SCORECARD.md](SCORECARD.md) for the comparison table.

## Quick Start

```bash
# 1. Start your model (see Server Flags below for model-specific flags)
vllm-mlx serve <model-path> --port 8000

# 2. Run all eval suites (~5 min)
python evals/run_eval.py --model "My-Model-Name" --quantization 4bit

# 3. View results
cat evals/results/*.json
python evals/generate_scorecard.py   # regenerate SCORECARD.md
```

## Server Flags by Model

Different models require different server flags for tool calling. Use the correct flags when starting the server:

| Model Family | Server Flags |
|-------------|-------------|
| **Qwen / Hermes** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **GPT-OSS (Harmony)** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser harmony` |
| **MiniMax** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser minimax` |
| **DeepSeek V3.1 / R1-0528** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser deepseek_v31` |
| **GLM-4** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser glm47` |
| **Qwen3-Coder (XML)** | `vllm-mlx serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml` |
| **Other / No tools** | `vllm-mlx serve <model> --port 8000` |

Then pass the matching `--parser` to the eval script:
```bash
python evals/run_eval.py --model "X" --parser hermes    # for Qwen/Hermes models
python evals/run_eval.py --model "X" --parser harmony    # for GPT-OSS (Harmony) models
python evals/run_eval.py --model "X" --parser minimax   # for MiniMax models
python evals/run_eval.py --model "X" --parser deepseek_v31  # for DeepSeek V3.1 / R1-0528
python evals/run_eval.py --model "X" --parser glm47     # for GLM-4 models
python evals/run_eval.py --model "X" --parser qwen3_coder_xml  # for Qwen3-Coder (XML)
```

## Eval Suites

| Suite | Items | What it tests | Scoring |
|-------|-------|---------------|---------|
| **Speed** | 6 metrics | TTFT cold/warm, decode tok/s short/long, RAM active/peak | Absolute numbers |
| **Tool Calling** | 30 scenarios | Tool detection, parallel calls, irrelevance, error recovery | % fully correct |
| **Coding** | 10 tasks | HumanEval+ problems (medium-hard) | % tests pass |
| **Reasoning** | 10 problems | MATH-500 competition math (levels 2-5, fractions + integers) | % correct answer |
| **General** | 10 questions | MMLU-Pro multiple choice (CS, Physics, Math, Chemistry, etc.) | % correct letter |

### Thinking Mode Disabled

**All eval suites send `enable_thinking: false` to every model.** This is critical for fair comparison:

- Models like Qwen3.5 default to extended internal deliberation ("Thinking Process") that can produce thousands of tokens before the actual answer
- With thinking enabled, models hit `max_tokens` limits and get truncated before answering, causing artificially low scores (e.g. 30-40% instead of 90% on reasoning)
- Disabling thinking ensures every model uses its full token budget for the actual response
- This applies to **all suites** (tool calling, coding, reasoning, general), not just reasoning

## Tool Calling Categories (30 scenarios)

| Category | IDs | Count | What it tests |
|----------|-----|-------|---------------|
| **A. Single Tool** | tc01-tc04 | 4 | Basic tool invocation with simple/explicit args |
| **B. Function Selection** | tc05-tc09 | 5 | Picking the right tool from 14 available |
| **C. Complex Args** | tc10-tc13 | 4 | Multi-line content, natural dates, piped commands |
| **D. Parallel Calls** | tc14-tc17 | 4 | Multiple tool calls in one response |
| **E. Irrelevance Detection** | tc18-tc20 | 3 | NOT calling tools when none needed |
| **F. Sequential Chains** | tc21-tc24 | 4 | Multi-step chains (2-3 tools in sequence) |
| **G. Missing Parameters** | tc25-tc26 | 2 | Asking for clarification instead of hallucinating |
| **H. Error Recovery** | tc27-tc28 | 2 | Adapting when a tool returns an error |
| **I. Nested Dependencies** | tc29-tc30 | 2 | Using output of one tool as input to another |

See [TOOL_CALLING_TESTS.md](TOOL_CALLING_TESTS.md) for detailed provenance and design rationale.

## Options

```bash
# Run specific suites only
python evals/run_eval.py --model "X" --suite speed tool_calling

# Specify tool parser
python evals/run_eval.py --model "X" --parser hermes   # or: glm47, minimax, auto

# Custom hardware label
python evals/run_eval.py --model "X" --hardware "MacBook Pro M4 Max 128GB"

# Verbose output (show error details)
python evals/run_eval.py --model "X" -v
```

## Contributing Results

We welcome community benchmarks! Different hardware + different models = better data for everyone.

1. Run the eval on your machine (any Apple Silicon Mac)
2. Results are saved to `evals/results/<model-name>.json`
3. Submit a PR with your JSON file
4. The scorecard table auto-regenerates

### Tips
- Use `temperature=0` (default) for reproducible results
- Run with a fresh server (restart before eval) for clean TTFT cold measurement
- Include `--quantization` and `--parser` flags for accurate metadata
- The eval takes ~5 minutes for all suites

## File Structure

```
evals/
├── README.md                 # This file
├── SCORECARD.md              # Auto-generated comparison table
├── TOOL_CALLING_TESTS.md     # Provenance & design rationale for 30 tool tests
├── run_eval.py               # Unified eval runner
├── generate_scorecard.py     # Reads results/*.json → SCORECARD.md
├── prompts/
│   ├── tool_calling.json     # 30 tool-calling scenarios (9 categories, L1-L5)
│   ├── coding.json           # 10 code generation tasks
│   ├── reasoning.json        # 10 MATH-500 competition math problems
│   └── general.json          # 10 MMLU-Pro multiple choice questions
└── results/
    └── <model-name>.json     # One file per model evaluation
```
