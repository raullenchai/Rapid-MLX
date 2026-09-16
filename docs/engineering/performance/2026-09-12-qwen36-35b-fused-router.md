# Qwen3.6-35B-A3B fused MoE router qualification

Date: 2026-09-12

Target: `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`

Host: Mac Studio, Apple M3 Ultra, 256 GB unified memory

OS: macOS 26.5.2 (25F84)

Runtime: MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17

Code base: `e28f68c41fdb8f71c618f7c26109e82bd920cf73`

## Decision

Ship a default-on, fail-closed Metal fast path for Qwen3.5-family MoE routing
at decode and small verification widths. The model's router projection and
precise softmax remain unchanged. The following composed tail is collapsed
into one launch:

1. top-k `argpartition`;
2. selected-score gather;
3. selected-score reduction;
4. score normalization.

Only rank-three, unsharded, normalized MoE blocks with 32-aligned expert counts
between 32 and 512, top-k at most 16, BF16/FP16 activations, and at most eight
rows are eligible. Wider prefill, unknown layouts, other model instances, and
failed install-time probes retain the stock path. Set
`RAPID_MLX_QWEN35_MOE_ROUTER=0` to disable.

## Correctness

The kernel deliberately reproduces the stock `argpartition` order, including
equal-probability ties, rather than returning only the same expert set. It also
uses the input dtype and stock reduction order for score normalization.

The test campaign compared both indices and scores with `mx.array_equal` for:

- BF16 and FP16;
- one, four, and eight rows;
- 50 randomized samples per dtype/width pair;
- an all-equal tie case across eight rows.

All comparisons were bit-exact. The real-checkpoint alternating generation
campaign emitted one token hash across both paths.

## Performance

The primary experiment loaded the checkpoint once, applied the existing MoE
gate/up and GDN projection fusions, compiled both router paths, then alternated
stock and fused 128-token greedy generations in eight adjacent pairs at
concurrency one. The raw prompt contained 26 tokens. Two A/B/B/A warmup pairs
were excluded. This avoids model-load differences and reduces sensitivity to
thermal drift.

| Metric | Result |
| --- | ---: |
| Paired speedup, median | **1.051x (+5.1%)** |
| Paired speedup, mean | **1.049x (+4.9%)** |
| Positive pairs | **8 / 8** |
| Pair range | +1.9% to +7.1% |
| Output token hashes | 1 across all runs |

The eight measured speedup ratios were `1.0451`, `1.0572`, `1.0596`, `1.0711`,
`1.0283`, `1.0658`, `1.0194`, and `1.0432`.

The ordinary text-serving baseline with all previously shipped fusions was
about 103 tok/s on the same host before this change. Absolute throughput moved
during the campaign because another resident workload intermittently used the
GPU; the within-process adjacent ratio is therefore the landing metric.

The multimodal server lane previously did not apply the already-qualified MoE
gate/up fusion to this architecture. Enrolling both that fusion and the router
fast path raised repeated 256-token greedy decode from a median 63.7 tok/s to
66.6 tok/s (**+4.5%**) in separate process runs. Repeated steady outputs had
the same content hash in both conditions.

## Reproduction

Resolve the target only through the configured Hugging Face cache. The primary
paired command is:

```bash
PYTHONPATH=. python3.12 scripts/large-model-run.py \
  --working-set-gb 32 --reserve-gb 12 -- \
  python3.12 scripts/benchmark_qwen35_moe_router.py \
    --model /path/to/immutable/snapshot \
    --pairs 8 --warmup-pairs 2 --max-tokens 128
```

For a separate server comparison, start text-only routing with speculative
decode and PFlash off, temperature zero, thinking disabled, and a 512-token
output. The stock-router comparison adds:

```bash
RAPID_MLX_QWEN35_MOE_ROUTER=0 python3.12 -m vllm_mlx.server \
  --model /path/to/immutable/snapshot \
  --no-mllm --no-spec-decode --pflash off
```

For the multimodal lane, compare the default with both new enrollments
disabled:

```bash
RAPID_MLX_MOE_GATE_UP_FUSION=0 \
RAPID_MLX_QWEN35_MOE_ROUTER=0 \
python3.12 -m vllm_mlx.server \
  --model /path/to/immutable/snapshot --mllm --pflash off
```
