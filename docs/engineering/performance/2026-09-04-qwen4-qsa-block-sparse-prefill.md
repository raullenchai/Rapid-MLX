# Qwen4 block-sparse QSA prefill qualification

## Outcome

Rapid-MLX now has an opt-in Metal path that consumes QSA's compact selected
blocks directly during long-context prefill. It avoids rebuilding a dense mask
and avoids visiting every physical K/V row. The route remains off by default
behind `RAPID_MLX_QSA_BLOCK_SPARSE=1`; decode, training, short queries, short
contexts, non-Metal systems, and unsupported layouts retain dense attention.

This is deliberately not an automatic promotion. The sparse kernel changes the
softmax reduction order relative to dense masked attention. A small fp64 probe
is favorable, but model-scale correctness and performance must be requalified
for each MLX build before changing the default.

## Production-path contract

Each QSA attention layer records cumulative `route_constructions`, `declines`,
and a complete `decline_reasons` histogram.
`qwen4_qsa_block_sparse_stats(model)` aggregates the receipt without evaluating
MLX arrays. A route construction proves selection of the sparse graph, not its
successful execution; qualification must pair a positive construction delta
with successful downstream request completion. Eligibility declines fail closed
to dense attention. MLX kernel execution is lazy, so construction and execution
errors surface through the engine's normal generation-error path; the route
deliberately does not synchronize every layer to simulate an execution-time
fallback.

The Metal kernel clamps compact counts to their buffer capacities and skips
out-of-range block starts and tail indices before reading K/V. Production inputs
come from the internal indexer, but the device-side guards prevent malformed
internal state from turning into an out-of-bounds GPU read without imposing a
host synchronization. Before count-based dispatch, invalid compact entries are
replaced with an out-of-range sentinel and sorted behind the valid prefix, so a
future validity hole cannot substitute a padded index for a selected block.
Block validity is represented structurally as one bit per complete block, while
the incomplete tail retains per-token validity; a partially valid complete
block cannot be represented at this boundary.

The existing fused-GDN receipt now also retains every fallback reason instead
of only the most recent reason per layer. Its end-to-end gate subtracts the
cumulative histograms and requires the exact expected prefill fallback. A fast
path is therefore not qualified merely because an engine-direct microbenchmark
can call it.

## Reproducible isolated measurement

- Candidate: worktree diff on `origin/main@d6c50526a85d50346af4126c1dca9f149aaa9fbe`
- Hardware: Apple M3 Ultra, 256 GiB unified memory
- OS: macOS 26.5.2 (25F84)
- Python: 3.12.13 from `.venv`
- MLX / mlx-lm / NumPy: 0.32.1 / 0.31.3 / 2.5.2
- Precision: bf16 performance geometry; fp16 inputs with an fp64 NumPy oracle
  for the numerical probe; TF32 disabled before importing MLX
- Warmup / samples: two warmups before each of five interleaved sparse/dense
  observations; three warmups before 100 synchronized dispatch-floor observations
- Artifact: `/tmp/qsa-qualification-production.json`, SHA-256
  `3eedcab197d6609a83279cac40b2b4e0796a6c94a2b8d487c720d79eff2fd821`

```bash
MLX_ENABLE_TF32=0 .venv/bin/python \
  scripts/bench_qwen4_qsa_block_sparse.py \
  --query-length 64 --key-length 16384 \
  --block-topk 512 --block-size 4 \
  --query-heads 24 --kv-heads 2 --head-dim 256 \
  --warmup 2 --repeats 5 --dispatch-repeats 100 \
  --output /tmp/qsa-qualification-production.json
```

At 64 queries over 16,384 physical keys, selecting 2,048 tokens:

| Measurement | Result |
| --- | ---: |
| Dense masked attention median | 2.828 ms |
| Direct block-sparse median | 1.468 ms |
| Isolated speedup | 1.93x |
| Exact binary synchronized dispatch floor | 177.4 us median |
| Sparse max absolute error vs fp64 | 0.000336 |
| Dense max absolute error vs fp64 | 0.001031 |
| Sparse RMS error vs fp64 | 0.000118 |
| Dense RMS error vs fp64 | 0.000303 |
| Sparse max absolute delta vs dense | 0.000977 |

The dispatch floor is about 12% of the sparse call at this smallest admitted
production geometry. This supports a coarse long-context gate and rejects a
design that would split the same work into many launches.

## Served dogfood

A follow-up campaign exercised the OpenAI-compatible streaming route against the
immutable 98 GB `rapid-mlx/Qwen3.8-Flash-Next-4bit` snapshot at revision
`dcf657e4acda2aae72da99cde65b6c491cd96998`. The baseline was
`d6c50526a85d50346af4126c1dca9f149aaa9fbe`; the candidate was the reviewed
kernel head `c7371139ad4aa0edf5eacc3922243e9c0db91e03`. Both arms used the same M3
Ultra, dependency environment, prompt bytes, three-run order, cold prefix
cache, 16-token decode budget, and Metal command-buffer settings.

The 98 GB parameter materialization intermittently tripped the macOS Metal
watchdog, including on some launches with conservative command-buffer limits.
The retained pair set both `MLX_MAX_OPS_PER_BUFFER=10` and
`MLX_MAX_MB_PER_BUFFER=10`; these are qualification controls applied equally to
both arms, not product defaults or a proven general watchdog workaround.
Contended attempts and one run invalidated by an unrelated concurrent MLX job
and GPU recovery were discarded before the clean pair below. Each retained arm
completed without a new GPU recovery.

| Prompt target | Baseline TTFT | Sparse TTFT | TTFT change | Baseline prefill | Sparse prefill | Prefill change | Decode change | Peak RSS baseline / sparse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 20.125 s | 20.321 s | -0.98% | 812.3 tok/s | 804.5 tok/s | -0.97% | +0.58% | 54.080 / 54.085 GiB |
| 32K | 45.684 s | 39.159 s | +14.28% | 716.5 tok/s | 835.9 tok/s | +16.66% | -0.56% | 54.091 / 54.096 GiB |
| 64K | 116.040 s | 79.950 s | +31.10% | 564.5 tok/s | 819.3 tok/s | +45.14% | +0.86% | 54.113 / 54.111 GiB |

Values are medians except peak RSS, which is the maximum. The 32K and 64K
TTFT improvements are large and settled across all three observations. The 16K
result is effectively flat but slightly negative, supporting the conservative
opt-in status rather than automatic promotion at the current 16K boundary. Any
future default-on policy should re-measure and select a higher crossover if the
16K result remains negative. Decode stayed within one percent and peak RSS
within 0.005 GiB.

The result artifacts were written outside the repository and hashed before the
server was shut down:

- baseline: `/private/tmp/qsa-served-base-d6c50526-caps10-clean.json`, SHA-256
  `bf76f01400241dabdb3a658963cb0e257af52c7ac77f2171e5cffcdc19df14c0`
- candidate: `/private/tmp/qsa-served-candidate-c7371139-caps10.json`, SHA-256
  `2af75b3c2b7389705ecbe945fa3fa0b81c1d0b0b6e71eb5de465ed1f066d99a9`

The served harness uses exact tokenizer-counted prompts, clears the prefix
cache before each request, records first-visible-delta TTFT and server-reported
token counts, and samples the recursive server-process RSS every 50 ms. Run
each arm from its own worktree with the environment above, then invoke:

```bash
.venv/bin/python .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8465/v1 \
  --model rapid-mlx/Qwen3.8-Flash-Next-4bit \
  --tokenizer-path "$SNAPSHOT" --server-pid "$SERVER_PID" \
  --label "$ARM" --runs 3 --decode-tokens 16 \
  --prompt-tokens 16384,32768,65536 \
  --artifact-revision dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --rapid-sha "$RAPID_SHA" --output "$RESULT_JSON" --timeout 240
```

This campaign is a performance and memory receipt, not a new model-quality
gate. The earlier five-path output comparison and the focused fp64/Metal tests
remain the correctness evidence for this default-off path.

The subsequent adversarial review added only device-side malformed-input
guards and removed a synchronous exception fallback that could not catch MLX's
lazy execution errors. On the resulting code head
`f1ec82bbf3ad0f478e2ad59d0960c3a07f8f9b6e`, the same isolated production
geometry measured 1.487 ms sparse versus 3.348 ms dense (2.25x), with the exact
same numerical errors reported above. The artifact is
`/private/tmp/qsa-qualification-safety-guards.json`, SHA-256
`15d6354b039801c39c1e155c72a16ef9e43da6b4f6f793a2bd7d74a4c0fbeab5`.
After block-validity structuring, strict int32 input enforcement, and receipt
renaming, code head `20df4f239c97ba107d6f180a0386f7cf396855f1`
remeasured the geometry at 1.577 ms sparse versus 2.755 ms dense (1.75x), with
the same numerical errors. That artifact is
`/private/tmp/qsa-qualification-final-contract.json`, SHA-256
`c5e7353e86eee32f683d0b6a1c52b51306350c5209ce7836b87022b9c77e2664`.
A final served rerun completed three 16K samples and one 32K sample at the
expected rates before a Metal recovery invalidated the remaining process state;
those partial samples are not merged into the complete table.

## Earlier end-to-end evidence and remaining limit

The repository's earlier M3 Ultra full-model campaign measured a 32K prompt at
523.4 to 579.2 prefill tokens/s (+10.7%) and 62.539 to 56.516 seconds TTFT,
with unchanged decode and byte-identical normalized outputs over five user
paths. That campaign used the same kernel design but predates this rebased
integration and its complete path receipts, so it is supporting evidence rather
than an exact-head promotion gate.

The served campaign now supplies settled wall-time and peak-memory
evidence at 16K, 32K, and 64K. Before automatic enablement, still require a
fresh model-scale correctness comparison on the pinned release dependency
build, the expected nonzero route-construction delta paired with successful
request completion, and no unexpected decline reasons. Keep the feature opt-in
while any of those requirements is missing.
