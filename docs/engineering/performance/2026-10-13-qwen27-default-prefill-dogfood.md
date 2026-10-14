# Qwen 27B default-sized prefill dogfood and shutdown review (2026-10-13)

Owner: Vector. This completes the default-sized 27B evidence gap for PR #4584
and checks the final inference source after adversarial lifecycle fixes.

## User impact and scope

Median seconds; no pooling across source campaigns.

| Candidate / metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| original, overlapping short TTFT | 22.600548 | 13.606063 | -39.798% |
| original, short singleton TTFT | 0.620635 | 0.620728 | +0.015% |
| original, overlapping long total | 22.524110 | 22.752680 | +1.015% |
| original, long singleton total | 21.938271 | 21.934363 | -0.018% |
| first-shutdown-fix, overlapping short TTFT | 22.602775 | 14.972610 | -33.758% |
| first-shutdown-fix, short singleton TTFT | 0.620988 | 0.621886 | +0.145% |
| first-shutdown-fix, overlapping long total | 22.528975 | 22.759662 | +1.024% |
| first-shutdown-fix, long singleton total | 21.940010 | 21.932080 | -0.036% |
| final, overlapping short TTFT | 22.598196 | 13.603800 | -39.801% |
| final, short singleton TTFT | 0.621183 | 0.620596 | -0.095% |
| final, overlapping long total | 22.524652 | 22.754148 | +1.019% |
| final, long singleton total | 21.939462 | 21.973058 | +0.153% |

These are ordinary dense Qwen requests with speculation disabled, not the
TensorFold runtime. The tested checkpoint is
`mlx-community/Qwen3.6-27B-4bit`, snapshot
`c000ac2c2057d94be3fa931000c31723aac53282`. A short request arrives while a long
request is still pre-filling. Yielding existing materialized chunks reduces
shared-executor waiting; the model forward itself is unchanged. Singleton
latency is controlled separately. No universal 85% claim, other Qwen version,
TensorFold/MTP acceleration, media performance or architecture-wide benefit is
established. TensorFold's dedicated serving entrypoint bypasses the modified
engine path; its backend/HTTP regression tests pass, but this campaign does not
claim real-model TensorFold qualification.

## Frozen source and experiment

Baseline `bec47abba980d48432c8b5917ca682dd0a5758de`. Three separate campaigns:
original candidate `32fdd92c880a8fc1341e37e604c8a568ef558ebb`, first shutdown fix
`2a30a17d2bf5310cc432cf60f9243a7422c3668d`, and final production source
`9d72bb8fcb34cf5357f38757e832f07c81acb21d`. The first fix was superseded by a
second lifecycle fix before integration; retain its measurements as intermediate
evidence, not final qualification. All campaigns use the original frozen HTTP
harness SHA-256
`7ff73a02c9c3b790a70d127cbbf996dfa8d24592e5b6b6d3be155313392d34e7`.

M5 Max, 36 GiB, macOS 27.0.1; Python 3.12.13, MLX 0.32.3,
mlx-lm 0.31.3, mlx-vlm 0.7.2, transformers 5.15.1, NumPy 2.5.4,
httpx 0.28.1. Matched before/after comparisons stay within one host and
environment. Explicit 2048-token text prefill, batch limit 8, default HTTP route,
no speculation, no disk cache, no downloads, temperature zero, thinking off,
maximum 32 output tokens. The existing MLLM scheduling budget resolves to 8192;
per-forward text cap remains 2048. No alias profile is silently assumed.

Each campaign alternates baseline/candidate then candidate/baseline across four
fresh server children. Two repetitions per child yield eight overlap waves,
28 distinct measured streams and four warmups. Long singleton receipts are
referenced twice but counted once. Across all three campaigns: 24 waves,
84 measured streams, 12 warmups; do not pool them into one latency estimate.
Long/short raw prompt budgets are 16,384/256; actual templated usage is recorded.
Require exact retrieval answers, cold caches, valid usage, stop/[DONE] and
short arrival before long first content. No correctness gate is relaxed.

Each campaign also passes separate exact/shared-prefix cache reuse, suspended
prefill cancellation before any content, survivor/subsequent answers, final
idle state and clean shutdown. Source/model/method manifests and traces confirm
actual candidate activation and bind each arm to its checkout. The extracted
validator recomputes every reported median and verifies the original ≥30%
short-TTFT/≤10% singleton and long-completion gates for each 27B/2048 campaign.

The worker was leased before access. Final campaign has 36 process/memory samples;
the preexisting idle Python service remained present (maximum sampled CPU
0.1%, RSS 74.3 MiB). It was not stopped or modified.
Sampled swap use was 921.00M→921.00M, maximum 921.00M;
this is not a zero-swap claim. Full samples retain page/swap counters and ordinary
system activity. Existing model cache and read-only environment were reused;
no model download or private cache was created. Own source/scratch/processes
were cleaned before releasing the lease; existing caches/services were kept.

## Adversarial findings and fixes

Direct scheduler shutdown with a borrowed executor could close an executing
cooperative generator, raise `ValueError` and skip continuation/wired-limit
cleanup. The fix requests closure without blocking an active chunk and services
it after that chunk exits. A second review found a close request could arrive
between the worker's final check and lock release; checking after unlock closes
that window. Both findings have deterministic regressions. Successful and
raising chunks verify exactly-once cleanup on the worker and a still-usable
borrowed executor. No further prompt work runs after closure.

A preliminary automated review questioned the MLX stream context. Public
`next()` already wraps each `_next()` invocation in its configured stream;
continuation resumes run inside that outer context, which exits between steps.
A regression verifies enter/forward/exit on every public call. No extra nested
stream context was introduced to satisfy the rejected finding.

The final focused backend/lifecycle/TensorFold suite passes 365 tests,
with three deselected by the repository's normal marker configuration; Ruff
passes. Independent adversarial review confirms both lifecycle findings fixed
and no remaining actionable code finding. Full PR validation and exact-head CI
are recorded separately on the PR; this report does not predeclare their result.

## Integration disposition and limits

The [earlier matrix](2026-10-13-prefill-hardware-model-matrix.md) retains the
M4/4B/2048 original performance gate **FAIL**: observed improvement 27.74%, below
the preregistered 30%, with correctness/lifecycle and regression checks passing.
Nothing here retroactively makes that matrix fully qualified. Integration
scope is the demonstrated concurrency fix, with the ≥30% benefit claim limited
to cells that actually pass it. Larger chunks remain enabled and tested; they
are not promised a minimum reduction on every machine/model. This is an explicit
narrowing of the benefit claim, not a lower experimental threshold. The owner
requested progression to the merge queue after the failed cell and TensorFold
boundary were disclosed. No chunk defaults or model eligibility are changed.

Gemma/Nemotron [architecture controls](2026-10-13-prefill-architecture-controls.md)
remain off the acceleration path and establish no speedup. Their noisy-host
limits are unchanged. No release is authorized by this report.

## Evidence and reproduction

[Raw evidence archive](fixtures/default-prefill-yield/2026-10-13-qwen27-default-dogfood.tar.gz) and
[outer manifest](fixtures/default-prefill-yield/2026-10-13-qwen27-default-dogfood.manifest.json):
49 members, 132,846 bytes, SHA-256
`66ede114e3c851dad89d8f9d4f62b994f38250199d8abf0568336a259d20f346`. Each campaign retains HTTP/lifecycle receipts, logs/traces,
model manifests and sanitized host samples. Frozen scripts and an inner manifest
allow source/receipt/hash verification without inference. No weights,
credentials, private endpoints or user-home paths are included.

From a repository containing the four source commits, extract the archive into
`/private/tmp/rapid-mlx-default-dogfood-validation/` and run
`python3 /private/tmp/rapid-mlx-default-dogfood-validation/validate_default_27b.py /private/tmp/rapid-mlx-default-dogfood-validation`.
Validation does not load a model. To repeat inference on a leased Metal worker,
reuse the frozen harness with pinned dependencies, an existing named snapshot
and clean source checkouts at `/private/tmp/rapid-mlx-default-prefill/base` and
`/private/tmp/rapid-mlx-default-prefill/worktree`:

```sh
python3 benchmark_default_prefill.py --model /path/to/Qwen3.6-27B-4bit/c000ac2c2057d94be3fa931000c31723aac53282 --output /private/tmp/rapid-mlx-default-dogfood/results.json --port 18560 --repeat 2 --cycles 2 --workloads retrieval --arms baseline2048 candidate2048
```

Keep `bench_metadata.py` beside the harness, run from the candidate checkout,
and retain process/memory samples and model digests. Archived lifecycle scripts
record their own commands. No external weights are required for receipt
validation; inference requires the existing checkpoint and compatible hardware.
