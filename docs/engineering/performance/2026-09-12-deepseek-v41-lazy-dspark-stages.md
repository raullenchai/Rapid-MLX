# DeepSeek V4.1 lazy DSpark stages

Date: 2026-09-12

## Scope

This experiment removes the forced MLX materialization between the three
DSpark proposal stages. The final proposal logits and confidence output remain
the synchronization boundary. Model weights, fixed K4 policy, target
verification, and generated tokens are unchanged.

## Environment

- Apple M3 Ultra, 256 GB unified memory
- DeepSeek V4.1 Flash native REAP 2-bit target
- Mixed 4-bit dense / 2-bit routed-expert DSpark sidecar
- Engram SSD offload and fast hyper-connections enabled
- Four 64-token prompts: code, reasoning, JSON-only structured output, Chinese
- Both variants warmed for every prompt before measurement
- Four paired repeats per prompt, alternating lazy/eager and eager/lazy order

## Result

Across 16 paired measurements, weighted decode throughput increased from
18.883 to 19.006 tok/s, a 0.65% improvement. All paired runs emitted identical
tokens and had identical accepted-draft counts. Peak MLX memory was 157.26 GB.

The gain was larger on low-acceptance prompts that execute more speculative
blocks, but remained small: most individual pairs improved by roughly 0.5% to
1.4%. This is a safe reduction in fixed synchronization overhead, not the main
path to the 40 tok/s research target; draft acceptance remains dominant.

## Benchmark caution

An earlier activation-kernel probe appeared to improve throughput by 35% when
the baseline always ran first. After warming every prompt shape and alternating
the paired order, it instead regressed weighted throughput by 0.9%. Kernel and
graph performance claims for this runtime must therefore use per-shape warmup
and alternating paired order.
