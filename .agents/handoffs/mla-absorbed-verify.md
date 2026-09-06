# MLA absorbed verification handoff

Receiving role: Atlas

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB

Branch: `vector/mla-absorbed-verify`

PR: #3110

## Verified facts

- The exact mlx-lm 0.31.3 MLA implementations for DeepSeek V3, GLM-4 MoE
  Lite, Kimi Linear, and LongCat Flash can use absorbed attention for qualified
  multi-token forwards. Unknown source bodies fail closed.
- Default-off is a literal no-wrap path. `L=1` and caches shorter than 1024
  tokens retain mlx-lm behavior. An upstream provider disables the local copy.
- Tiny real MLX layers cover cold and warm behavior for all four architectures.
- GLM-4.7-Flash-4bit M=3 forwards measured 2.500x at 1K, 6.594x at 4K, and
  16.180x at 16K. Outputs are numerically close but not bit-exact.
- A deterministic 4457-token suffix workload improved 4.054x; the two arms
  produced identical 128-token output, although both diverged from vanilla.
- Full commands, environment, and exclusions are recorded in
  `docs/engineering/performance/2026-09-05-mla-absorbed-verify.md`.

## Unresolved questions and risks

- GLM-4.7 is not in Rapid's native MTP allowlist, so current direct product
  reach is narrow.
- Multi-token numerical accumulation can change a near-tied greedy choice.
- Source hashes require deliberate requalification after mlx-lm changes.
- DeepSeek V3.2 is deliberately excluded to avoid colliding with the existing
  indexed-attention patch.

## Next concrete action

The repository owner accepted this as a default-off release compatibility
lever after final Vector review on 2026-09-06. Land it through the Mac merge
queue, but do not promote it to a default without a supported speculative-route
trajectory battery. Retire the local patch when upstream mlx-lm PR #1817 is
released and Rapid adopts that provider.
