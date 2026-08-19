# Architecture

The canonical architecture document lives at
[docs/architecture.md](../architecture.md).

It covers:

- System overview (API layer → BatchedEngine + Scheduler → MLX backends)
- Module map of `vllm_mlx/`
- Request flow for streaming chat completions
- Paged KV cache architecture (block structure, prefix caching, COW)
- Hardware detection (`vllm_mlx.optimizations.detect_hardware`,
  `vllm_mlx.chip_tier.detect_chip_tier`)
- Performance architecture and bottleneck analysis

This page previously held a parallel copy that drifted out of date
(it still documented the deleted `SimpleEngine` and a nonexistent
`vllm_mlx.hardware` module). To keep a single source of truth, all
content now lives in the root document above.
