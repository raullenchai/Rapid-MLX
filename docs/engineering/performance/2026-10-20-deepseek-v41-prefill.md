# Native V4.1 prefill: CPU-verifiable acceleration

Owner: Vector. GPU qualification remains with Vector; Apple test registration is
included for Harbor. No release is authorized.

## Implementation and compatibility

`stream_generate(..., optimize_prefill=True)` keeps the original bounded 512-token
chunks. Set `optimize_prefill=False` to reproduce the original path. Decode,
rollback, cache semantics, model layers, fake quantization, and intra-forward
`eval_interval` are unchanged. Scalar-only drafter adapters retain the old path.

Intermediate chunk logits are unused. Instead of evaluating them, prefill evaluates
all window rings, compressed KV, index keys, compressor partial KV/scores, and the
complete captured hidden tensor. This flush includes caches in layers after the
last captured hidden state and bounds the lazy graph at every chunk. Engram history
continues to be updated by the existing forward. The final chunk still evaluates
the original full-sequence head and returns its last logits.

The only change to `dspark.py` is `observe_many` (with `observe` delegating to it). Observation projections
are independent of previous observations. It projects only rows that survive in
the sliding window, retains the original scalar projection/rotation/quantization
operations, and evaluates the resulting rows together. Existing window list shape,
chronological order, position validation, and decode observations remain intact.
This is batching of evaluation and live-row selection, not a change to draft numerics.
`observe` now delegates to `observe_many` (bit-identical: same per-row operations).
The draft-numerics change carries the textually identical `observe`/`observe_many`
block, so the two branches compose without a merge conflict in either order.

Experiments rejected both a one-row vocabulary projection and multi-row drafter
projection: CPU matrix kernels change their reduction rounding when batch shapes
change. Head differences reached approximately 2.4e-7 in the tiny model; draft
rounding could cross an FP8 midpoint. Even projecting two head rows did not preserve
full-chunk rounding. The shipped approach keeps the original matrix shapes and
scalar observation operations. No relaxed equivalence threshold is used.

## Expected sources of speedup

At 4096 prompt tokens, with a 128-row draft window:

- Only the final 512-row vocabulary head is evaluated: 87.5% less head work.
  For width 5120 and vocabulary 129280, about 4.74 trillion multiply/add FLOPs
  are avoided (counting two FLOPs per multiply/add). Unused final normalization
  and final-layer FFN work can also remain lazy on intermediate chunks.
- Observation flushes fall from 4096 to 8. Including target chunk flushes,
  explicit outer evaluations fall from 4104 to 16. The existing intra-forward
  evaluations remain additional and unchanged (8 with 40 layers/interval 40).
- Only 1024 observation rows are projected, versus 4096: 75% less drafter
  observation projection, norm, rotation and quantization work.
- Target chunk memory remains bounded at 512 tokens. The additive observation
  graph is bounded by the draft window and number of stages.

These are work-count estimates, not a measured GPU throughput claim. The requested
1.5–2x checkpoint throughput improvement remains a GPU qualification target.
For prompts no longer than one chunk, head work and observation projections may
be unchanged; fewer evaluation calls alone need not improve CPU throughput.

## CPU evidence

Environment: ARM64 macOS 26.5.2, Python 3.12.13, CPU MLX
`0.32.1.dev20261019+3a62199`. The installed runtime cannot initialize without Metal,
so tests used the provided CPU build through a test-only import override. No GPU
execution or model download was performed. Timing ran through a pytest harness
invoking the benchmark's tiny-mode entry point.

The seeded synthetic model has three layers, shared ratio-2 and ratio-1 compression,
a 128-entry ring, index keys, nonzero Engram embeddings, and three target captures.
Observation tests use production scalar math with dense and affine-quantized
synthetic projections. Measurements use one warmup and the median of three timed
runs per path; graph profiling and state fingerprints run outside the timer.

| Tokens | Baseline tok/s | Optimized tok/s | Ratio | Outer evals old/new | Graph operations old/new | Evaluated head rows old/new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2504 | 2449 | 0.98x | 129 / 2 | 35046 / 35101 | 128 / 128 |
| 513 | 2519 | 8152 | 3.24x | 515 / 4 | 136323 / 36803 | 513 / 1 |
| 1024 | 2544 | 8279 | 3.25x | 1026 / 4 | 268585 / 69989 | 1024 / 512 |

Graph operations are the summed labeled operation nodes in unevaluated DOT graphs
at explicit `mx.eval` boundaries, not hardware kernel counts. Head-row counts are
analytical. Tiny timings measure Python and CPU behavior and do not predict
checkpoint GPU performance. Cache allocation is included; tokenization and model
loading are excluded. JSON includes schema version and a benchmark methodology hash.

## Correctness and checks

- 23 new tests pass. At lengths 1, 2, 127, 128, 129, 511, 512, 513, 1100 and 2049,
  last logits, all captured hidden states and every cache field are bit-exact
  against the original 512-chunk path, including inactive ring/partial slots and
  Engram IDs. Each case also checks 32 target-only greedy decode steps.
- Dense and quantized synthetic draft projections, three seeds, and sequential
  observations across window boundaries produce bit-exact retained draft windows.
  Production streaming with a synthetic proposal policy checks 32 generated tokens,
  rejection/rollback continuation, all observed hidden states and retained windows.
- All `test_deepseek_v41_*.py`: 193 passed, 12 skipped, 2 existing CPU-environment
  failures. Both failures reproduce using unchanged committed sources:
  `test_target_qmv_install_and_small_route_execution` requires a Metal kernel;
  `test_disk_engram_bypass_paths_handle_empty_and_uncached_rows` encounters an
  empty-array reshape in CPU dequantization. These remain outside this change.
- Changed production executable lines: 32/32 covered (serving 13, additive drafter
  method 19). Apple CI includes the new test file.
- Ruff check/format passes. Benchmark metadata and Apple coverage registration
  checks pass. The pinned Python 3.11 mypy budget remains at 821 grandfathered errors across 158 files, with no growth.

## Pending GPU measurement

Use local checkpoint and optional local draft paths in `MODEL` and `DRAFTER`.
Run from the repository root on the GPU host:

```sh
.venv/bin/python -m scripts.bench_deepseek_v41_prefill \
  --model "$MODEL" --drafter "$DRAFTER" --repeats 5 > prefill-gpu.json
```

Without `--drafter`, the benchmark detects a colocated `dspark` directory. If no
drafter is available, it measures target-only prefill and emits null TTFT. Real
checkpoint mode measures 1024 and 4096 tokens, old versus new, with independent
fresh caches and reset draft state. It verifies exact final logits, full hidden
captures, cache fingerprints and draft-window fingerprints. TTFT is measured by
advancing `stream_generate` to its first token with a 32-token generation budget,
including the first speculative verification forward. No tokenizer roundtrip
changes the deterministic token-ID workload.

Record checkpoint identity, host/chip, MLX version, precision, memory headroom and
JSON methodology hash with the results. GPU throughput, real-drafter TTFT, memory
peak, watchdog behavior and checkpoint-scale greedy continuation remain unverified.
If GPU arithmetic differs, qualify the difference before claiming equivalence or
speedup; do not weaken the CPU bit-exact regression contract.
