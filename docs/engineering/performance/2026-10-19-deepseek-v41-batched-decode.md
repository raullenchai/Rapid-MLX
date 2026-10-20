# Native V4.1 per-row decode cache (Phase A)

Owner: Vector. Branch: `vector/dsv41-batched-decode`. Validation host: Studio,
arm64 macOS 26.5.2. Scope: draft-off, one decode token per request, independent
positions in a single model forward. Scheduler integration, speculative rollback,
checkpoint qualification and production throughput claims are outside this phase.

## Layout and execution

- `cache.py`: `BatchModelCache.from_caches` merges single-row prefilled caches.
  It retains host NumPy int64 `offsets` and an int32 device `positions` mirror.
  The default capacity is the largest source capacity; `max_seq_len=` can
  explicitly grow or shrink it, provided every current offset fits. Mixed
  configurations/dtypes and multi-row source caches are rejected.
- Layer arrays have a leading request axis. Each layer has its own window ring;
  compressed KV and index-key arrays remain owned by their original source
  layers. Each source has per-row committed counts and, for pooled compression,
  per-row float32 partial KV/score groups. Engram histories are NumPy int64
  `[rows, capacity]`. Forward-local sharing still hands source buffers and sparse
  selections down the existing layer stack.
- `filter(indices)` mutates the batch, supports reordering and an empty result,
  and compacts every device/host field. `extend(batch)` appends rows and rebuilds
  at the larger capacity. `extract(row)` returns an independent `ModelCache`
  that can resume on the original path. These operations copy logical state;
  joining is not an allocation-free scheduling operation.
- `batched_decode.py`: all query, KV, compressor, indexer and output projections
  operate on the entire `[rows, 1, dim]` tensor. Row-specific RoPE gathers the
  existing frequency table at each position. The window is gathered in each
  row's chronological order; short rows get masked indices. Ring writes use
  `position[row] % window_size`.
- Every row computes a prospective compressor latent. Device masks commit only
  rows closing their group, at `position[row] // ratio`. Completing a group
  leaves its saved partial buffer unchanged, including unused stale slots, to
  match the single-stream cache. Committed counts advance independently.
- Index scores beyond each row's committed count become negative infinity
  before top-k. Insufficient valid entries become `-1` sparse indices. Candidate
  selection pins each row's own newest block and honors its own valid length.
  Sparse attention retains the existing sink and inverse-RoPE behavior.
- `attention.py` dispatches on cache type. `model.py` validates decode shape and
  capacity, hashes each row against its own history, then uses the existing
  block/MoE/Hyper-Connection loop and evaluation boundaries. Batched
  `last_logit_only=True` returns `[rows, vocab]`; otherwise it returns
  `[rows, 1, vocab]`. The original cache retains its original output contract.
- Rollback entry points raise before a batched forward changes cache state.
  Existing single-row and lockstep arithmetic, compressor/indexer code, MoE
  flattening and `ExactDirectDownSwitchGLU` are unchanged. The QMV gate remains
  `rows * 6 <= 36`; larger batches use the existing fallback.

There are no new device-to-host reads inside an attention layer. Shape decisions
use host offsets; scatter indices and commit/visibility masks stay on device.
The existing Engram token transfer and `model.eval_interval` boundaries remain.

## Verification

The sandbox's installed Metal runtime could not import without GPU access. A
CPU-only build of version 0.32.1 (source revision `3a6219917e45`) was used for CPU validation; no installed runtime was replaced.
All weights were synthetic. No checkpoint was downloaded or loaded.

```sh
.venv/bin/python -m pytest \
  tests/test_deepseek_v41_batched_decode.py -q

.venv/bin/python \
  scripts/bench_deepseek_v41_batched_decode.py --tiny --steps 4 --warmup 2
```

The new test module passes 16 tests. The float32 synthetic fixture has six
layers, compression ratios 0/2/1, KV sources and consumers, a non-owner index
source, candidate filtering, two active Engram layers and learned Hyper-Connection
mixes. Six prompts of lengths 1/2/3/4/5/8 decode 64 steps at evaluation intervals
0/1/3. Logits match within `rtol=3e-5, atol=5e-5`, with identical greedy tokens. The absolute tolerance allows batch-dependent
fp32 GEMM reduction rounding on GPU (observed maximum difference `6.6e-6`);
greedy token comparisons remain exact.
Cache comparisons cover all window slots, committed KV/index prefixes, complete
partial-state buffers and active history. Tests additionally cover empty history,
capacity growth/caps, joining, reordered compaction, extraction/resumption,
poisoned uncommitted entries, absent compression/Engram, one projection per
layer for all requests, and QMV dispatch at R=1/2/4/6/7/8 (kernel stubbed on CPU).

A read-only comparison against original `HEAD` module sources produced bitwise
identical logits for single-row and three-row lockstep prefill plus 64 decode
steps. The existing V4.1-related suite had 231 passes and 12 skips, with two CPU
environment failures also reproduced using the original module sources:

- `test_target_qmv_install_and_small_route_execution`: requires the Metal kernel.
- `test_disk_engram_bypass_paths_handle_empty_and_uncached_rows`: the CPU backend
  rejects empty affine dequantization with an inferred reshape dimension.

Metal kernel execution and real-model correctness/performance remain unqualified.
No existing test was changed to suppress these failures.

## Synthetic benchmark smoke result

The command above uses a 64-wide float32 model, CPU, two warmup steps and four
measured steps, `eval_interval=1`. Mixed start positions are `16 + row`;
lockstep uses the largest position. Prefill and merging are outside timing.
Timing includes forward, greedy token selection and evaluation.

| Rows | Mixed median ms | Lockstep median ms |
| ---: | ---: | ---: |
| 1 | 6.27 | 5.87 |
| 2 | 5.86 | 5.54 |
| 4 | 6.30 | 5.49 |
| 8 | 6.38 | 5.02 |

These numbers only demonstrate that the script runs. CPU overhead on synthetic
weights does not predict checkpoint bandwidth or serving throughput. The script
accepts an explicit existing checkpoint directory through `--model`; that mode
was not run.

## Numerical limits

The seeded fixture (`seed=18`) passes the tight comparisons above. Batch-size
changes can select different reduction implementations. Tiny pre-quantization
rounding differences can cross a fake-quantization boundary or reorder tied
sparse scores, and close vocabulary logits can then change greedy selection.
During investigation, synthetic seed 17 crossed an FP8 boundary at decode step
27 (zero-based 26): sparse selections agreed, but one row's maximum logit
difference was about `1.4e-3`. The default fixture avoids that boundary; this is
not a claim of universal bitwise equality between batch sizes. New backend,
dtype or checkpoint qualification must account for these discontinuities.

## Phase B scheduler validation

The draft-off scheduler adds FIFO admission, 512-token single-row prefill chunks,
cache joining/compaction and greedy token delivery through independent queues.
One model thread owns loading and model execution; the HTTP executor only
consumes request queues. `RAPID_MLX_DSV41_LANES=2/4/8` opts in. The default `1`
retains serial K4 decoding. See [model settings](../../reference/models.md) and
[the batched draft verification proposal](../decisions/2026-10-19-deepseek-v41-batched-draft-verification.md).

CPU validation uses the same 0.32.1 development build described above. Its
thread-local compiled-function cache segfaults on worker-thread teardown with
compilation enabled; the native greedy comparisons complete before that crash.
The validation commands therefore disable compilation explicitly, without
changing runtime compilation settings in product code. With a CPU-enabled MLX
installation, run:

```sh
MLX_DISABLE_COMPILE=1 .venv/bin/python -m pytest \
  tests/test_deepseek_v41_batch_scheduler.py \
  tests/test_deepseek_v41_batched_decode.py -q
```

The scheduler tests cover staggered admission at 2/4/8 lanes, FIFO overflow,
independent prompt/output lengths, exact greedy agreement with single-request
execution, batch Engram prefetch hooks, cancellation during prefill/decode,
iterator close, failed prefill and shared decode recovery, cache release while
idle, boundary validation, EOS/ignore-EOS, text finalization, stream/non-stream
agreement, loading/execution thread identity and shutdown with pending work.
An in-process HTTP harness checks the serial selection, opt-in capacity,
draft-off metadata, streaming/non-streaming adapters and shutdown registration.

All weights remain tiny and synthetic. No checkpoint was loaded. The two
pre-existing CPU failures listed above also reproduce against unchanged module
sources. Metal was unavailable in the sandbox: compilation-enabled scheduler
teardown, actual SSD prefetch I/O under batching, real-model numerical stability,
GPU memory headroom and throughput remain unqualified. The batch cache still
rejects speculative rollback; drafting needs the separate follow-up proposal.

Final CPU regression result: **948 passed, 13 skipped, 2 failed** across all
native V4.1 tests and the shared-server import suites. Both failures are the
baseline CPU limitations listed above. The focused scheduler/cache/concurrency
run passed **182 tests**, including **33 scheduler/server tests**. Ruff check and
format checks passed for all changed Python files.
The environment-policy guard plus scheduler rerun passed **47 tests**.
