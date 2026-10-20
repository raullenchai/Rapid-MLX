# Batched draft verification follow-up

Status: proposed. Owner: Vector. Current continuous batching is draft-off and
supports one target decode token per row. Default serving remains serial K4.

A future draft-enabled scheduler could propose K tokens independently for each
row, then verify all proposals with one target forward of `q_len = K + 1` per
row (seed plus proposals). Each row compares target greedy tokens against its
own proposals, commits its accepted prefix and mismatch replacement, and rolls
back independently to the corresponding absolute position. EOS and output
budgets cap each row independently; shorter proposal windows need explicit masks
and cannot advance the cache through padded tokens.

The batched cache needs per-row rollback windows: saved window-ring slots,
compressed KV/index committed counts, partial compression KV/score states,
Engram token-history ranges and original offsets. Rollback must preserve shared
source/consumer ownership, invalidate rejected suffixes, and restore each row's
draft observation state to exactly its committed prefix. Admission, compaction
and cache extraction must carry those windows without aliasing. The existing
single-row rollback implementation supplies the correctness contract; the current
batch explicitly rejects rollback before mutating state.

Expected gain comes from amortizing target weight reads across both requests
and accepted proposal tokens, reducing target forwards per committed token.
Acceptance, divergent row lengths, draft cost, cache snapshot bandwidth and
longer kernel execution can erase that gain. No speedup is claimed without a
checkpoint-bound benchmark including per-request latency and aggregate throughput.

This remains a follow-up because the current forward only accepts `q_len = 1`;
chunked per-row causal masks, multi-token compression boundaries and independent
rollback need correctness tests before a scheduler can use them. Qualification
must cover mismatch at every proposal position, mixed accept lengths, EOS,
truncation, cancellation, joins/leaves, Engram prefetch and exact serial greedy
equivalence, then measure Metal memory and latency on the intended checkpoint.
