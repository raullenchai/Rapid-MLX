# Vector handoff

- Status: implementation and controlled mini verification complete; prepare PR
- Active task: long-context prefill, HTTP prefix reuse, and mixed long/short
  service contention
- Branch: `raullenchai/vector-prefill-service-perf`
- Host: M2 Pro 32 GB Mac mini (`mini`); local worktree for code/tests
- Verified facts:
  - Qwen3.5 4B direct 16K--96K throughput is within 0.7% at prefill chunks 512
    and 2,048, while 512 reduces peak MLX memory by 27--29%.
  - In the controlled service A/B, 512 reduces contended short-request TTFT
    51.1% and total latency 64.4%; standalone latency is unchanged.
  - Exact recurrent prefix reuse reduced a 22.5K prompt's TTFT from 62.48 s to
    0.408 s and server counters confirmed 22,537 cached tokens.
  - The candidate wheel auto-selected 512 without an explicit CLI flag and
    reported that value in `/v1/status`.
  - A single-repeat Gemma 4 12B mlx-vlm scout favored 512 by 3.6--3.8% and used
    less memory, but it is not yet sufficient evidence to change dense models.
  - Focused tests pass; raw artifacts and hashes are recorded in
    `docs/engineering/performance/2026-08-22-long-context-service-prefill.md`.
- Risks:
  - Service A/B and Gemma scout are one repeat; treat them as scoped engineering
    evidence, not publication-grade competitor claims.
  - Bounded recurrent prefix snapshots reached 3.73 GB at 32K; larger cache
    entry counts should not be enabled without multi-prefix pressure testing.
- Receiving role: Atlas for compatibility/integration review.
- Next action: run the full proportional test set, review the diff, commit,
  push, open the PR, and complete the repository's review/CI flow.
