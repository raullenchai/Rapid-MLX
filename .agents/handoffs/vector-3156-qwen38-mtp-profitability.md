# Vector handoff: #3156 Qwen3.8 MTP profitability

Owner: Vector

Branch: `vector/3156-qwen38-mtp-profitability`

Worktree: `/private/tmp/vector-3156-qwen38-mtp-profitability`

Base: `d09f18adaaf0435d07960bcf229e85a512b57616`

## Goal and boundary

Reduce the fixed-K=3 Qwen3.8 native-MTP target verification cost reported in
#3156. The task is limited to the GatedDeltaNet verify hot path, regression
coverage, reproducible measurements, and PR validation. It does not change
model aliases, model artifacts, downloads, speculative defaults, DFlash,
continuous batching, server contracts, or Desktop UI.

## Verified diagnosis and change

Qwen3.8-27B has 48 GatedDeltaNet layers. The K=3 rollback implementation in
`cache_patch.py` processed each four-position verify block through four
separate `gated_delta_update` calls per GDN layer. Rapid already had a generic
Metal gated-delta verify recurrence that returns the output, final state, and
every intermediate rollback boundary in one pass. The task reuses that path
for eligible Qwen3.8 inference and preserves the position-wise implementation
as the fallback.

Reference-first check:

- vLLM's current recurrent-state speculative path carries accepted-token
  metadata and specialized state checkpoint/replay support rather than
  treating a recurrent cache like a KV trim.
- SGLang's current linear-attention path uses a dedicated target-verify kernel
  and exact accepted-prefix state commit/replay, confirming that verify and
  recurrent-state preservation should be fused rather than launched once per
  speculative position.
- mlx-lm 0.31.3's Qwen3.5 implementation and gated-delta Metal recurrence were
  checked instruction-by-instruction. Rapid's existing boundary kernel uses
  the same decay, beta, state update, head mapping, masking, output, and FP32
  recurrent-state order for the production BF16 shape.

## Evidence

- A diagnostic using the real Qwen3.8 GDN shape (five warmups, 40 synchronized
  samples per lane) measured 0.445 ms -> 0.308 ms median (1.45x); BF16 output,
  final state, and all boundary states were byte-identical. The integrated
  measurements below are the landing evidence.
- Three alternating exact-checkpoint runs: median 45.87 -> 52.28 decode tok/s
  (+14.0%); median verify sync 4.954 -> 4.644 seconds (-6.3%).
- Every base/candidate MTP run had the same 256-token SHA-256, 53.74%
  acceptance, and 98 fixed-K=3 verify rounds.
- `tests/test_mtp_spec_decode.py`: 134 passed.

Full commands and environment are in
`docs/engineering/performance/2026-09-07-qwen38-mtp-gdn-verify.md`.

## Known risk and next action

A separate AR-versus-MTP diagnostic was 1.31x faster but produced different
token hashes. Base and candidate MTP output hashes are identical, so this is
not caused by the fusion and remains outside this PR's performance boundary.
Before closing #3156, rerun the fixed-K=3 benchmark on the reporter's clean M2
Max to determine whether the original 0.87x result now crosses break-even.

Next: complete scoped adversarial review, run repository PR validation, commit,
push, open the PR with quantitative evidence, and queue only if every required
gate passes.
