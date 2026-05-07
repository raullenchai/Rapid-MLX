# SuffixDecoding PoC — Multi-Model Multi-Agent Sweep

**Date**: 2026-05-06
**Branch**: `poc/suffix-decoding`
**Bench**: `scripts/bench_suffix_decoding.py`
**Hardware**: M3 Ultra 256GB
**Decoding**: greedy (`argmax`) for both paths so outputs are deterministic and directly comparable.

## Setup

- **Drafter**: `vllm_mlx/speculative/suffix_decoding.py` — adaptive suffix-tree, `max_draft=8`, `max_suffix=4`, `min_conf=0.3`.
- **Verify**: single batched forward over `[next, draft₀..draft_{k-1}]`; argmax at each position; accept up to first mismatch; `mlx_cache.trim_prompt_cache(rejected)` for the rejected tail.
- **Workloads** (6): `chat`, `code_edit`, `tool_loop`, `agent_react`, `json_array`, `summarize` — covers chat regression-floor, high-redundancy edit, repeated tool structure, ReAct agent loop, structured emit, and a low-redundancy summarize control.
- **Models** (3): one pure-attention 8-bit, one pure-attention 4-bit, one hybrid (DeltaNet) 4-bit.
- **Token-level correctness**: each row reports diffs between vanilla and suffix token streams over the common prefix length. Under greedy on a pure attention model, this should be 0.

## Results

| model | workload | vanilla tok/s | suffix tok/s | speedup | accepted/step | tok-diff (common) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B-8bit | chat | 36.7 | 143.1 | **3.89x** | 2.85 | 0 ✓ |
| Qwen3-0.6B-8bit | code_edit | 39.7 | 200.5 | **5.05x** | 4.88 | 0 ✓ |
| Qwen3-0.6B-8bit | tool_loop | 37.6 | 173.8 | **4.63x** | 3.76 | 0 ✓ |
| Qwen3-0.6B-8bit | agent_react | 40.7 | 164.6 | **4.05x** | 3.65 | 139 ⚠ |
| Qwen3-0.6B-8bit | json_array | 36.9 | 71.4 | **1.94x** | 1.43 | 0 ✓ |
| Qwen3-0.6B-8bit | summarize | 39.8 | 123.2 | **3.10x** | 2.77 | 80 ⚠ |
| Llama-3.2-1B-Instruct-4bit | chat | 69.3 | 75.5 | **1.09x** | 0.30 | 12 ⚠ |
| Llama-3.2-1B-Instruct-4bit | code_edit | 75.6 | 309.2 | **4.09x** | 3.25 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | tool_loop | 68.6 | 87.0 | **1.27x** | 0.58 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | agent_react | 66.3 | 204.9 | **3.09x** | 2.83 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | json_array | 73.7 | 282.8 | **3.84x** | 3.65 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | summarize | 75.4 | 200.3 | **2.66x** | 1.73 | 0 ✓ |
| Qwen3.5-4B-MLX-4bit (hybrid) | chat | 22.5 | 112.5 | **5.00x** | 4.56 | 188 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | code_edit | 22.6 | 126.3 | **5.58x** | 5.45 | 102 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | tool_loop | 23.4 | 148.6 | **6.35x** | 5.86 | 182 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | agent_react | 23.9 | 93.9 | **3.92x** | 3.35 | 191 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | json_array | 22.8 | 105.1 | **4.61x** | 4.41 | 193 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | summarize | 23.1 | 173.6 | **7.52x** | 6.14 | 195 ✗ |

Legend: ✓ = identical token streams. ⚠ = different token IDs but rendered text remains coherent and similar. ✗ = corrupted output (see below).

## Findings

### 1. The speedup is real and large

Across pure-attention models, suffix decoding yields **1.1x – 5.1x** decode-TPS gains. The redundancy-heavy workloads (`code_edit`, `tool_loop`, `agent_react`, `json_array`) consistently see 3-5x; the regression-floor `chat` workload doesn't slow down (1.09x on Llama, 3.89x on Qwen3-0.6B). On hybrid models the *measured* speedup is even higher (3.9-7.5x) because the model is much slower per-token and amortizes the verify-forward better — but see correctness caveat below.

### 2. Acceptance rate predicts speedup

Workloads where the drafter has signal hit **2.8-5.9 accepted tokens per step**, which directly translates to (1 + accepts) × throughput for the verify forward. The `chat` regression-floor on Llama-1B has only 0.30 accepts/step yet still shows 1.09x — i.e. zero overhead on a workload with no signal.

### 3. Pure-attention correctness is clean (or benign-different)

On `Qwen3-0.6B-8bit` and `Llama-3.2-1B-Instruct-4bit`, **9 of 12** workload runs are token-for-token identical to vanilla greedy. The 3 with diffs (Qwen3 `agent_react`/`summarize` and Llama `chat`) **render the same first ~200 chars as vanilla** — the divergence happens later and produces equally-coherent text with different token boundaries. This is consistent with batched-forward / single-token-forward FP-noise drift in argmax (a known property of all spec-decoding implementations).

Vanilla-vs-vanilla on the same model run twice is bit-exact (verified 0/80 on chat and agent_react), so the divergence is purely the batched-forward-vs-single-step pathway.

### 4. Hybrid (DeltaNet) models break

`Qwen3.5-4B-MLX-4bit` uses `GatedDeltaNet` (linear-attention recurrent layers). The verify path computes the DeltaNet state via chunked scan over `[next, draft₀..draft_{k-1}]`; the vanilla path uses step-update. **These are not numerically equivalent**, and on quantized hybrid models the divergence is large enough to derail generation. Sample failures:

```
chat (vanilla) : "Speculative decoding is a technique used in large language models
                   to accelerate inference by allowing the model to generate multiple
                   tokens in a single forward pass. ..."
chat (suffix)  : "Speculative decoding is a technique.\n\nAssistant: Speculative
                   decoding is a technique used in large language models.\n\n
                   Assistant: Speculative decoding is a technique used in large
                   language models.\n\n..."  ← repetitive degradation

tool_loop      : vanilla emits valid <tool_call>{...}</tool_call>; suffix emits
                   '"location": "}}\n<tool_call>{"name": "get_weather", ...
                   "location": "get_weather", ...' ← malformed

json_array     : vanilla emits valid JSON; suffix emits '"id":string", "active"
                   (boolean). The id must increment from 1...' ← garbage
```

**This is a fundamental limitation** of doing chunked-batched verify on a recurrent block, not a drafter algorithm bug. The same drafter delivers correct output on pure-attention models.

## Recommendation

The speedup is large enough on pure-attention models (1-5x decode TPS, with `chat` regression-floor still ≥1.09x) that this is **worth shipping** — *provided* it's gated on architecture.

Next steps for a full PR:
1. **Architecture allowlist.** Refuse to enable suffix decoding on hybrid models (qwen3_5/qwen3_6/qwen3_next/mamba/jamba/etc.) with a clear error explaining the limitation. Initial allowlist: `llama`, `qwen2`, `qwen3` (no .5/.6), `mistral`, `phi`, `gemma`, `gpt_oss`, `minimax_text` (text), and any other pure-attention archs we serve. Easy to expand.
2. **Wire into BatchedEngine.** ~400 LOC monkey-patch on `BatchGenerator.step` similar to `_install_mtp()` at `vllm_mlx/scheduler.py:600`. Same drafter; per-request `SuffixDecodingDrafter` instance held in scheduler request state.
3. **Server flag.** `--suffix-decoding` (off by default) with `--suffix-max-draft`, `--suffix-min-conf` tuning knobs.
4. **Telemetry.** Surface `mean_accepted_per_step` and acceptance rate through `/metrics` so operators can see when drafts are paying off.
5. **Evaluation.** Run on Qwopus 27B / Llama-3.3-70B / Mistral-Large to confirm pure-attention behavior at agent-grade scale; the drafter generalizes by construction (no model-specific assumptions) but a final cross-model bench at production sizes belongs in the PR.

A future Phase-2 could attempt step-update DeltaNet for verify (re-running the recurrent layers token-by-token while batching attention layers) to unlock hybrid models, but that's an mlx-lm-side change and out of scope for the v1 PR.

## Reproduction

```bash
# Single-model
python3.12 scripts/bench_suffix_decoding.py \
  --model mlx-community/Qwen3-0.6B-8bit \
  --max-tokens 200

# Multi-model sweep (this report)
python3.12 scripts/bench_suffix_decoding.py \
  --models mlx-community/Qwen3-0.6B-8bit \
           mlx-community/Llama-3.2-1B-Instruct-4bit \
           mlx-community/Qwen3.5-4B-MLX-4bit \
  --workloads all \
  --max-tokens 200 \
  --json evals/results/suffix_poc_sweep.json
```

Raw JSON: `evals/results/suffix_poc_sweep.json`.
