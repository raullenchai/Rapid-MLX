# Qwen3.5/3.6 eager layer-dispatch qualification

Date: 2026-09-13

## Decision

Enable eager decoder-layer submission for narrow, qualified Qwen3.5-family
35B-A3B inference slabs. MLX still returns the same lazy array, but each layer
is submitted as soon as its graph is ready so Metal can execute while Python
constructs the next layer. Other family shapes and slabs wider than 64 batch-
sequence rows keep the dependency's default scheduling. This avoids projecting
one large-model measurement onto smaller architectures where submission cost
could dominate, and avoids extra submissions during substantial prefill.

The optimization is process-wide because every production model load passes
through the shared tokenizer loader. It therefore reaches CLI, server, and
Desktop-launched servers without a separate UI control. Set
`RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH=0` before process start to restore the
dependency's scheduling for diagnosis.

## Environment

- Mac Studio, Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13
- MLX 0.32.2; mlx-lm 0.31.3
- Target: `mlx-community/Qwen3.6-35B-A3B-4bit@38740b847e4cb78f352aba30aa41c76e08e6eb46`
- MTP sidecar where noted: `mlx-community/Qwen3.6-35B-A3B-MTP-4bit@0295b81421bf4d0fccca9a7c0fcfb1418dda3516`
- Greedy decoding with thinking disabled
- Existing Hugging Face cache only; no model download or cache relocation

## Ordinary decode result

A same-model-load, alternating A/B used 96 output tokens for six runs per arm.
Every one of the 12 token streams was byte-identical. The implementation in
this change measured:

| Scheduling | Median decode |
|---|---:|
| Dependency default | 84.32 tok/s |
| Eager layer submission | 88.05 tok/s |

That is a **4.42%** median throughput improvement with no weight, cache,
sampling, or token-selection change.

The broader qualification prompt set covers coding, arithmetic reasoning,
creative writing, strict JSON, and tool arguments. The reproducible harness
warms both modes, alternates their order in one loaded model, compares SHA-256
digests for each greedy pair, and aborts on any output mismatch.

## MTP and concurrent-server boundary

This gain must not be multiplied by the separately measured MTP gain. Native
MTP paired output remained exact, but after excluding the first cold-page
outlier the eager scheduling effect was neutral (paired median approximately
-0.3%, within run variance).

A real standard server then ran three four-request concurrent waves with the
shipping continuous-MTP default. Stable aggregate throughput was 144.83 tok/s
with eager submission and 144.84 tok/s without it. Stable response digests,
completion lengths, and finish reasons matched. This is the required no-
regression result for CLI and Desktop's default accelerated lane, not an
additional performance claim.

## Failed and bounded experiments

- A fused attention-kernel candidate compiled and ran faster in isolation, but
  disagreed materially with a dense numerical oracle. A zero-input compile
  probe was insufficient to detect the error, so the kernel was rejected.
- Eager submission did not improve continuous MTP because that execution path
  already materializes target verification at its own round boundaries.
- Measurements taken immediately after the 38 GB checkpoint was demoted to the
  warm storage tier were dominated by first-read I/O (roughly 7-9 tok/s) and
  were not used for the hot inference qualification.

## Reproduction

Install the MTP extra so the text-only benchmark runtime is available, ensure
the exact target revision is already in the default Hugging Face cache, and
run:

```bash
python scripts/benchmark_qwen35_eager_dispatch.py \
  --rounds 6 --max-tokens 192
```

The script prints each paired A/B row and a final JSON summary. It never
overrides or relocates the Hugging Face cache.
