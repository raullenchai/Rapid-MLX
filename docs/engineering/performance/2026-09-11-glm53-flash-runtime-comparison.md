# GLM-5.3-Flash runtime comparison and first Rapid optimization

- Date: 2026-09-11
- Rapid baseline: `origin/main@cae3412e4e45b07ee6fbcb9426fb9dba64a2791b`
- Host: Mac Studio (`Mac15,14`), Apple M3 Ultra, 256 GB, macOS 26.5.2
- Rapid runtime: MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17

## What can and cannot be compared

Rapid's production alias is
`Vontra/GLM-5.3-Flash-MLX-4bit-MTP@06d6c7530e8290e20fabdc37a825ce07bdfc490c`:
uniform affine q4-g64, 43 shards, 181,709,451,790 bytes. The checked-in
[same-machine campaign](../../benchmarks/recent-large-models-m3-ultra.md)
recorded:

| Prompt tokens | Rapid prompt tok/s | Rapid decode tok/s | Rapid TTFT |
| ---: | ---: | ---: | ---: |
| 128 | 156.3 | 32.38 | 0.819 s |
| 2,048 | 372.8 | 28.36 | 5.493 s |
| 8,192 | 359.6 | 27.75 | 22.779 s |
| 32,768 | 276.9 | 27.20 | 118.341 s |

Those are medians of three 256-token completions, with the prefix cache cleared
between requests. A separate sustained native-MTP experiment recorded 32.00
tok/s without MTP and 31.65 tok/s with fixed K=1, so MTP remains correctly
disabled.

[oMLX 0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) publishes a
different strict oQ4e conversion (169 GiB) on an M3 Ultra with 512 GB. Its
uncached 128-token rows are:

| Prompt | Prompt tok/s | Decode tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 4,096 | 482.3 | 24.1 | 8.49 s |
| 16,384 | 443.2 | 23.7 | 36.97 s |
| 32,768 | 449.6 | 23.5 | 72.89 s |

These rows are useful targets, not an apples-to-apples ranking. At the published
rows, Rapid's decode is numerically higher (27.20-32.38 versus 23.5-24.1 tok/s),
while oMLX's long-context prefill is numerically higher (443-482 versus
277-373 tok/s). The checkpoint, quantizer, prompts, output length, memory
capacity, and runtime differ, so neither difference supports a product-level
speed claim. [mlx-vlm 0.7.0](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.7.0)
has GLM-5.3 and native MTP support but does not publish a comparable
official-size Apple-silicon benchmark.

The Studio Hugging Face quota volume had only 6.0 GiB free and no GLM-5.3
snapshot or warm-tier symlink. Storage policy forbids deleting other models or
redirecting the cache, so a new 181.7 GB identical-checkpoint campaign was not
run in this task.

## Gap found: the MLLM lane skipped Rapid's MoE fusion

Rapid already fuses an eligible `SwitchGLU` gate/up pair after text-model load.
Affine quantization packs every output row independently, so concatenating the
two projections removes one `gather_qmm` launch while preserving the exact
arithmetic. The text path previously measured 86.4 -> 92.5 tok/s (+7.0%) on
Qwen3.6-35B-A3B-8bit.

GLM-5.3 is forced through `MLXMultimodalLM`, and mlx-vlm owns a separate copy of
the `SwitchGLU`, `SwitchLinear`, and `QuantizedSwitchLinear` classes. Therefore:

1. the MLLM loader never called the existing fusion; and
2. calling it unchanged still found zero layers because it recognized only the
   mlx-lm class family.

The fix enrolls the already-loaded mlx-vlm class family in the same structural
and bit-exact rewrite, then invokes it only for `model_type == "glm5_next"` at
the model-owning MLLM load boundary. Other VLM families remain unchanged until
they receive their own real-model qualification. The environment kill switch
`RAPID_MLX_MOE_GATE_UP_FUSION=0` remains authoritative.

GLM-5.3 has 42 sparse-MoE layers (layers 3 through 44). The production geometry
is hidden 4096, expert intermediate 2048, 288 routed experts, top-8, affine
q4-g64.

## Weight-free performance and correctness evidence

The harness `scripts/benchmark_glm53_moe_fusion.py` constructs exactly one
production-shape mlx-vlm `SwitchGLU`, quantizes it q4-g64, and compares 31 warm
decode calls before and after the rewrite. Five fresh-process runs produced a
median paired speedup of 1.037x (range 1.000x-1.119x). Every run was byte-equal;
the representative output SHA-256 was
`7b8413896b503763c97c71b8990106c7bf1739da5eec8817b40b533635c51fc6`.

This is a layer microbenchmark, not a whole-model claim. Attention, routing,
shared experts, normalization, sampling, and server overhead are absent, so the
expected whole-model win is smaller and must be measured once the exact
checkpoint is available again.

Reproduce with an MLX 0.32.2 / mlx-vlm 0.6.17 environment:

```bash
PYTHONPATH=. python3 scripts/benchmark_glm53_moe_fusion.py
```

The 189 MB
`inference-optimization/GLM-5.3-Flash-0.1B-A0.1B@8311399447eba9c9b215e3209ab6f25e59c7d21e`
fixture provided a real architecture smoke. Rapid found and fused both of its
sparse MoE layers, the server reached ready, `/health` returned healthy, and an
OpenAI-compatible non-streaming request completed. Stock mlx-vlm, the Rapid
correctness overlays, and oMLX produced the same deterministic 128-token text
hash on the direct harness. Tiny-model throughput is deliberately not reported
as a proxy for the 321B checkpoint.

## Next optimization assessed

oMLX also routes long affine projections through a native q4 prefill tile. A
shape-faithful test of GLM's largest fused KDA input projection
(`4096 -> 24,896`) on the same host and oMLX's MLX 0.32.0 environment measured:

| Tokens | Stock MLX | Native tile | Native / stock | Max abs error |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.885 ms | 2.184 ms | 0.863x | 0 |
| 512 | 6.484 ms | 6.211 ms | 1.044x | 0 |
| 2,048 | 24.380 ms | 23.015 ms | 1.059x | 0 |

That kernel is a modest win only at larger chunks and a regression at 128
tokens. It also requires shipping and maintaining a native extension. It is not
the next low-risk Rapid change without an official-checkpoint end-to-end
campaign proving a material net gain.

## Required full-model follow-up

When storage capacity exposes the exact alias snapshot again, run fresh-process
stock/fused pairs with identical prompt bytes and seeds at 128, 4K, 16K, and
32K context, 128 and 512 output tokens, prefix cache off, MTP off, and both text
and image requests. Record TTFT, prompt tok/s, decode tok/s, peak MLX/process
memory, output token hashes, and whether all 42 layers fused. The change should
remain gated if the median whole-model decode improvement is not positive or
any output hash differs.
