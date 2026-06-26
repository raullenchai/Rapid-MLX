# SOTA quant evaluation — SpinQuant / QuaRot spike plan

Branch: `spike/spinquant-eval` (off `main` @ 9b629025)
Mission: **Evaluate whether integrating SOTA quantization rotation methods into rapid-mlx is actually useful for our users.** Not "make SpinQuant work" — "is it worth shipping?"

## Phase 1+2 findings — synthesis

### Pivot recommendation: **SpinQuant → QuaRot**

The task description for #314 promised "no new runtime kernel" and "1-2 weeks LOW risk". Phase 1 research (Meta's reference impl + paper deep-read) shows:

- The "no-kernel" claim **holds only for W4A16 R1+R2 rotations** (weight-only, absorbed into adjacent linear weights).
- W4A4 / KV4 — the configurations with the **headline 1.3-2× decode gains** — require online Hadamard kernels in the activation path. That's months of MLX kernel work and likely still fails on Qwen3 (activation outliers worse than Llama).
- **QuaRot** (arxiv 2404.00456) uses random Hadamard instead of learned Cayley rotation. Same R1+R2 placement. Within ~0.05 PPL of learned SpinQuant on Llama-2-7B, **beats** SpinQuant on Llama-3-70B (where SpinQuant degrades catastrophically). No PyTorch dependency, no GPU calibration loop.

**Decision**: pivot to QuaRot R1+R2 at W4A16 as the actual spike target. Drop "learned" and "W4A4" from scope. If even QuaRot W4A16 doesn't beat existing mlx-community 4bit by ≥1.0 PPL, learned SpinQuant won't help.

### Integration landscape (Phase 2)

**rapid-mlx is a consumer, not a producer**, of quantized weights:

- `vllm_mlx/aliases.json` is a metadata-only registry — points at pre-quantized `mlx-community/*` repos
- No quant pipeline in the repo. `mlx_lm.convert` does the actual quantization, externally.
- 6 verified 4-bit aliases potentially affected: `qwen3.5-{4b,9b,27b,35b}-4bit`, `qwen3.6-{27b,35b}-4bit`

**Integration cost for rapid-mlx itself: trivial.** Add aliases.json entries pointing to new `mlx-community/Qwen3.5-4B-MLX-4bit-QuaRot` repos. Engine code unchanged.

**The actual work lives OFF rapid-mlx**:
1. QuaRot R1+R2 rotation precompute on FP16 Qwen3 weights (offline script, no kernel)
2. `mlx_lm.convert` re-quantize rotated weights to 4-bit
3. Upload rotated 4-bit weights to mlx-community (HF write token `tmax` available)

**PPL eval harness**: Phase 2 audit said ABSENT — **wrong, only checked the rapid-mlx repo**. Upstream `mlx-lm` 0.31.3 ships `mlx_lm.perplexity` as a CLI (`python -m mlx_lm.perplexity --model X --data-path wikitext`). 192-LOC module, takes any HF model path or local mlx model, returns PPL + standard error. **No custom harness needed.** mlx-lm also ships `mlx_lm.evaluate` for lm-evaluation-harness task evals (HumanEval/MMLU/GSM8K) — requires installing `lm_eval` pkg.

**Bench harness**: EXISTS and is reliable. `vllm_mlx/community_bench/runner.py` gives apples-to-apples decode tok/s, locked schema (5 measured rounds, greedy decode, 2 buckets). Just swap model path baseline vs rotated.

**Re-quantize**: upstream `mlx_lm.convert --hf-path <rotated-fp16> --mlx-path <out> -q --q-bits 4`. Standard tool.

**Net custom code for Phase 3**: `scripts/quarot_rotate.py` ONLY. Everything else is shell orchestration around upstream tools.

## Pareto question — the real test

The user-value question is **NOT** "does QuaRot work?" It's:

> **At equal memory footprint, does QuaRot-rotated 4-bit beat existing non-rotated 6-bit / 8-bit?**

If a user has 32GB of RAM budget:
- Option A: ship them `qwen3.5-4b-4bit-quarot` (QuaRot W4A16, +1-3% PPL claimed)
- Option B: ship them `qwen3.5-4b-6bit` (existing, no SOTA work needed)

If Option B is just as good or better per GB, QuaRot adds maintenance burden with no user benefit. **This comparison is the kill criterion.**

## Phase 3 plan — 4B experiment

### Compute budget on operator's machine
- Rotation precompute (QuaRot R1+R2, CPU/MLX): ~30 min
- `mlx_lm.convert` re-quantize: ~10-30 min
- PPL eval (Wikitext-2, 4B model): ~10-20 min × 3 variants (baseline 4bit, rotated 4bit, baseline 6bit if exists)
- Community bench (decode tok/s): ~10 min × 3 variants

**Total Phase 3 compute: ~2-3 hours.**

### Deliverable
A 4-column table:

| Variant | Memory (GB) | Wikitext-2 PPL | Decode tok/s (M-series) |
|---------|-------------|----------------|--------------------------|
| qwen3.5-4b-bf16 (FP baseline) | — | — | — |
| qwen3.5-4b-8bit (mlx-community) | — | — | — |
| qwen3.5-4b-6bit (if exists) | — | — | — |
| **qwen3.5-4b-4bit** (mlx-community, current verified) | — | — | — |
| **qwen3.5-4b-4bit-quarot** (this experiment) | — | — | — |

Plus 1-2 task evals (HumanEval-pass@1 or GSM8K-pass@1) to confirm PPL gain translates to user-visible quality.

### Phase 3 go/no-go criterion
- If `4bit-quarot` shows ≥1.0 PPL improvement over `4bit` AND ≥0% gain over `6bit` (i.e., not Pareto-dominated by just shipping 6bit) → proceed to Phase 4 (27B)
- Else → write up findings, close #314, recommend kill or radical redesign

## Phase 4 plan — 27B experiment

**Held until Phase 3 ack-positive.** Heavier compute commitment (~4-6 hours), and **Hadamard size risk**:

- Qwen3.6-27B `intermediate_size=17408 = 2^10 × 17`. Not cleanly factorizable into small Hadamard primes (no exact Hadamard of size 17).
- QuaRot's random Hadamard tolerates any size via random sign matrix construction — but quality at large arbitrary sizes is empirically less validated than small-prime Kronecker construction.
- This risk doesn't apply to 4B (clean dims).

## Phase 5 plan — user-value report

Honest 1-page recommendation: **integrate / kill / redesign**, with:
- Headline number: "QuaRot R1+R2 W4A16 delivers X PPL gain at Y% decode cost on Qwen3.6-27B vs current 4-bit baseline"
- Pareto comparison vs 6-bit existing aliases
- Per-release maintenance cost estimate (Phase 2 says ~30 min hands-on per Qwen wave with automation; ~8 hours wall-clock serial)
- Decision: ship `mlx-community/Qwen3.5-4B-MLX-4bit-QuaRot` family + add aliases.json entries, or close #314 with findings archived

## Decision points awaiting operator

1. **Confirm pivot SpinQuant → QuaRot.** Same scientific question, much cheaper to validate, strictly safer on 27B.
2. **Ack Phase 3 compute use.** ~2-3 hours on operator's M-series machine. Mostly background-able (rotation precompute and PPL eval don't need the machine interactive).
3. **Hold or proceed on Phase 4 27B** until Phase 3 result is in hand.

## What's already done

- Phase 1 lit + impl deep-read (#324 ✓)
- Phase 2 rapid-mlx quant pipeline + eval harness audit (#325 ✓)
- Architecture-dim Hadamard check for Qwen3.5-4B and Qwen3.6-27B (this doc)
- Spike branch `spike/spinquant-eval` set up

## What's next (ack-positive)

- ~~Build minimal Wikitext-2 PPL harness~~ — **upstream `mlx_lm.perplexity` already provides this**, no custom code
- Clone QuaRot reference impl (github.com/spcl/QuaRot) and adapt R1+R2 rotation precompute to Qwen3 architecture as `scripts/quarot_rotate.py` (~100-300 LOC, the only real custom code in this spike)
- Phase 3 commands (ack required for compute use, ~2-3h on M-series):
  ```bash
  # baseline PPL on existing 4-bit + 6-bit + 8-bit aliases
  mlx_lm.perplexity --model mlx-community/Qwen3.5-4B-MLX-4bit --data-path wikitext --num-samples 200
  mlx_lm.perplexity --model mlx-community/Qwen3.5-4B-MLX-6bit --data-path wikitext --num-samples 200  # if exists
  mlx_lm.perplexity --model mlx-community/Qwen3.5-4B-MLX-8bit --data-path wikitext --num-samples 200
  # QuaRot rotation + re-quant + rotated PPL
  python scripts/quarot_rotate.py --input Qwen/Qwen3.5-4B --output /tmp/qwen3.5-4b-rotated
  mlx_lm.convert --hf-path /tmp/qwen3.5-4b-rotated --mlx-path /tmp/qwen3.5-4b-rotated-4bit -q --q-bits 4
  mlx_lm.perplexity --model /tmp/qwen3.5-4b-rotated-4bit --data-path wikitext --num-samples 200
  # bench
  rapid-mlx bench /tmp/qwen3.5-4b-rotated-4bit --tier speed
  ```
- Fill the Pareto table; decide go/no-go for Phase 4 27B
