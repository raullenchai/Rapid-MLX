# Phase 5 user-value report — QuaRot/SpinQuant evaluation

Branch: `spike/spinquant-eval`
Mission: Evaluate whether integrating SOTA rotation-based quantization (SpinQuant/QuaRot) into rapid-mlx is actually useful for our users.

## Verdict: **KILL.** Do not integrate.

Three independent reasons compound to a clear no:

1. **Architecture incompatibility with hero targets** — Qwen3.5/3.6 use hybrid attention. 3 of every 4 layers are `GatedDeltaNet` linear-attention (different weight matrices, gated non-linearities). QuaRot's rotation-absorption analysis only applies to full-attention transformer blocks. Porting to hybrid is a multi-week analysis effort, not a 1-2 day spike.
2. **Negative quality result on plain Qwen3** — on the one architecture where the rotation IS mathematically clean (Qwen3-0.6B full-attention), QuaRot R1+R2 + RTN 4-bit makes PPL WORSE by +1.39 vs RTN baseline.
3. **The paper claims depend on GPTQ-calibrated quantization** — which MLX does not provide. Porting GPTQ to MLX is itself a multi-month track.

## Measured data

### Qwen3.5-4B Tulu-3 PPL baselines (Pareto reference)

| Variant | Memory | Tulu-3 PPL | Decode tok/s | Weight size |
|---------|--------|------------|--------------|-------------|
| qwen3.5-4b-4bit | 15.4 GB | **3.719** ± 0.013 | 1732 | ~2.3 GB |
| qwen3.5-4b-6bit | 16.4 GB | **3.604** ± 0.013 | 1642 | ~3.3 GB |
| qwen3.5-4b-8bit | 17.5 GB | **3.600** ± 0.013 | 1666 | ~4.3 GB |

Knee at 6bit: 4bit→6bit improves 0.115 PPL; 6bit→8bit improves 0.004 PPL (noise). 6bit is the Pareto sweet spot.

### QuaRot R1+R2 sanity test on Qwen3-0.6B

| Variant | PPL | Notes |
|---------|-----|-------|
| original Qwen3-0.6B-bf16 | 6.104 ± 0.031 | FP baseline |
| rotated Qwen3-0.6B-bf16 (R1+R2) | **6.103** ± 0.031 | **Rotation is mathematically lossless** ✓ |
| mlx-community Qwen3-0.6B-4bit | 6.683 ± 0.033 | Published 4-bit (RTN-equivalent) |
| RTN 4-bit (from bf16, no rotation) | 6.680 ± 0.033 | Apples-to-apples baseline |
| **QuaRot R1+R2 + RTN 4-bit** | **8.067** ± 0.042 | **+1.39 PPL WORSE** |

### Architectural details surfaced during spike

- **R1 (random Hadamard on hidden_size)** — absorbable into embed/q/k/v/o/up/gate/down/lm_head weights. Pure weight transform. ✓
- **R2 (per-head Hadamard on head_dim, V-O path)** — absorbable into v_proj and o_proj for W4A16. No runtime kernel needed (online Hadamard in QuaRot e2e is only for W4A4 outlier suppression, not correctness).
- **R3 (Q-K rotation)** — requires runtime kernel. Only needed when KV cache is quantized.
- **R4 (intermediate_size Hadamard on down_proj)** — **cannot be applied as weight-only**. Initial spike attempt assumed it could; smoke test on Qwen3-0.6B produced garbage output. Reading QuaRot's `e2e/quantized_llama/modeling_llama.py` confirmed `OnlineHadamard(intermediate_size)` is part of the runtime forward pass (line 112) — without that runtime kernel, `down_proj @ H @ x ≠ down_proj @ x`. R4 is intentionally removed from the final script.

This corrects the Phase 1 lit report which conflated SpinQuant's analytical claim that R2 is "weight-only" (true) with R4 being "partially weight-only" (false — R4 needs runtime support even with weight-side prep).

### Hero alias architecture audit

All 18 verified 4-bit/8-bit hero aliases in `vllm_mlx/aliases.json` resolve to Qwen3.5 or Qwen3.6 models. Reading `mlx_lm/models/qwen3_5.py`:

- `class DecoderLayer`: `is_linear = (layer_idx + 1) % args.full_attention_interval != 0`
- Default `full_attention_interval = 4`
- For 32-layer Qwen3.5-4B: layers 4, 8, 12, 16, 20, 24, 28, 32 are full attention; the other 24 are GatedDeltaNet
- GatedDeltaNet weights: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` (5 linears with gated state-space mechanism)
- Standard rotation absorption analysis assumes transformer attention block structure; doesn't apply to GatedDeltaNet

For 75% of layers per hero alias, the rotation scheme would need re-derivation from first principles. This is research work, not engineering.

## Pareto comparison — why this matters for user value

The implicit promise of QuaRot was: "save memory by shipping rotated 4-bit instead of 6-bit, with equivalent quality."

At equal-quality target (PPL ≈ 6-bit baseline of 3.604):
- 6-bit baseline: 3.3 GB weights, 1642 tok/s. **Already works today, zero SOTA effort.**
- Hypothetical QuaRot 4-bit: claimed 2.3 GB weights at similar quality. Would have saved 1.0 GB.
- Measured QuaRot 4-bit on Qwen3-0.6B: makes 4-bit quality WORSE, not equivalent to 6-bit.

Even if hypothetical QuaRot+GPTQ on hybrid Qwen3.5 worked (it doesn't, both due to lack of GPTQ and architectural incompatibility), the user-visible win is 1 GB of memory at the cost of weeks of engineering and ongoing per-release re-quant. Marginal.

**Concrete user-value-positive alternative**: ship 6-bit variants of all 4-bit hero aliases. The data shows the 4bit→6bit jump is the only meaningful quality gain (0.115 PPL); 6bit→8bit is noise. Cost: just upload mlx_lm.convert outputs to mlx-community and add aliases.json entries.

## Recommendation

1. **Close #314 (SpinQuant spike)** with this report archived. Tag as evaluated-not-shipped.
2. **Don't open similar spikes for QuIP#/VPTQ (#316) until the GPTQ-on-MLX question is settled.** Same dependency on calibration; same risk of negative result on RTN-only.
3. **#315 (W8A8 INT8 with fused Metal kernel)** is a separate scientific question — depends on whether MLX/Apple Silicon hardware exposes int8 dot product. Phase 7 NAX audit prereq stands independent of the QuaRot finding.
4. **For 0.9.x or 0.10**: open issue "ship 6-bit variants for hero aliases" — concrete, proven win, ~1 week of work for the full 6-model batch.

## Time spent

- Phase 1 (lit + impl): ~2.5 hours background agent
- Phase 2 (pipeline audit): ~2 hours background agent
- Phase 3 (4B experiment, blocked → pivoted to 0.6B sanity test): ~1.5 hours operator's compute + ~2 hours engineering on the rotation script
- Phase 5 (this report): ~30 min

Total spike investment: ~8 hours. Killed at the right size; the negative finding is well-supported and the recommendation is concrete.

## Code artifacts (for archive)

- `spike/scripts/quarot_rotate.py` — MLX R1+R2 weight-only rotation script for plain Qwen3 (not Qwen3.5/3.6 hybrid)
- `spike/baselines/qwen3.5-4b-4bit.log` — 4-bit baseline run
- `spike/baselines/qwen3.5-4b-multi.log` — 6-bit + 8-bit baselines
- `spike/results/0.6b-quarot-comparison.log` — the negative QuaRot result on 0.6B
- `spike/results/4b-quarot-pipeline.log` — failed 4B pipeline (architecture incompatibility surfaced here)
