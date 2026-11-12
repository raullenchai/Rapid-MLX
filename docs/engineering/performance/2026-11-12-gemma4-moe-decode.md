# Gemma 4 26B-A4B single-token MoE decode

Gemma 4 26B-A4B (`gemma4` / `gemma4_text`; 30 MoE layers, 128 experts,
top-8, a shared dense GeGLU MLP beside the routed experts) spent decode time
on many small launches per layer: separate expert gate and up projections, a
materialized weighted expert sum, and a chain of norms, residual adds and
router operations around them. The `gemma4_moe_decode_fusion` knob addresses
that path without changing the arithmetic:

- **Expert gate/up fusion.** The existing gate/up concatenation now enrolls
  Gemma's experts on both SwitchGLU call contracts. On the multimodal route,
  single-token decode previously ran two selected projections.
- **Expert reduction.** The selected expert outputs are weighted and summed
  in one launch (single row, Metal/GPU, supported dtypes, K ≤ 8).
- **Compiled post-attention half.** For single-token decode, each MoE layer's
  residual add, branch norms, dense MLP, router, experts and layer scalar run
  as one compiled graph. It is installed after dense decode fusion and wraps
  whatever decoder call that installer left. Every layer must pass a
  bit-exact probe against the uncompiled half; any decline restores every
  layer. Prefill, multi-row batches, verification widths, per-layer inputs,
  shared caches and other dtypes keep the previous call.

The knob is default-off and resolves once at install. Builtin cells enable it
on the three measured generation/tier classes below. Set
`RAPID_MLX_GEMMA4_MOE_DECODE_FUSION=0` before server startup to restore the
previous path. `RAPID_MLX_MOE_DECODE_FUSION=0` and
`RAPID_MLX_MOE_GATE_UP_FUSION=0` still disable their parts.

## Measurements

Real `rapid-mlx serve mlx-community/gemma-4-26b-a4b-it-4bit` (revision
`0d77464e`), MLX 0.32.4, default serving flags. Each arm starts a fresh
server, and arms are interleaved off/on for three cycles. Each server answers
three greedy chat prompts with 256 completion tokens each (one ~540-token
passage, two short prompts). Results are decode tok/s for the short prompts,
median across cycles. All machines were leased with no other engine running,
and no virtual machine was active during the M4 and M5 runs.

| Machine | Off | On | Change |
| --- | ---: | ---: | ---: |
| M3 Ultra, 60-core GPU, 256 GB | 121.5 | 133.9 | +10.2% |
| M5 Max, 32-core GPU, 36 GB | 117.2 | 124.6 | +6.3% |
| M4 Pro, 16-core GPU, 48 GB | 77.7 | 81.6 | +5.0% |

Cycle spread was below 1% on every machine. Greedy text hashes for all three
prompts match between arms. The off arm's own first-server, first-prompt text
differs once per machine; that cold-start variance is pre-existing and also
appears without the knob.

An earlier M4 run was discarded because a CI virtual machine started on the
host during it (off arm 75.3 tok/s vs 77.7 clean). The harness now records
running VMs and waits for none before measuring.

Component prototypes on the M4 Pro (direct decode loop, same model) showed
the parts only add up together:

| Variant | tok/s |
| --- | ---: |
| Base | 76.8 |
| Expert reduction | 78.5 |
| Compiled post-attention half alone | 77.0 |
| Reduction + compiled half | 78.3 |
| Reduction + compiled half + gate/up fusion | 79.7–80.4 |

A fused router top-k/softmax kernel was also tried and dropped. It produced
no measurable gain and changed greedy tokens.

## Not measured

M2 Pro (generation 14) is not measured and keeps the default (off). Batched
decode with more than one active row and speculative verification take the
previous path by construction and were not benchmarked here.
