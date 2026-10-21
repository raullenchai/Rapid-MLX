# Architecture

The canonical architecture document lives at
[docs/architecture.md](../architecture.md).

It covers:

- System overview (API layer → BatchedEngine + Scheduler → MLX backends)
- Module map of `rapid_mlx/`
- Request flow for streaming chat completions
- Paged KV cache architecture (block structure, prefix caching, COW)
- Hardware detection (`rapid_mlx.optimizations.detect_hardware`,
  `rapid_mlx.chip_tier.detect_chip_tier`)
- Performance architecture and bottleneck analysis

Read the root document for the current implementation and module paths.
