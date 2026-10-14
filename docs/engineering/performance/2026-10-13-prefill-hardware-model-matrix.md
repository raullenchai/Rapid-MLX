# Default-route prefill continuation across hardware and models (2026-10-13)

Owner: Vector. PR #4584 remains a draft until the evidence and intended support
scope have been reviewed. This extends the earlier shared-host observation;
it does not replace the original receipts or turn 84.62% into a universal claim.

The change lets an existing chunked text prefill return between materialized
chunks so a native text peer can use the shared executor. It does not accelerate
the underlying model forward. Current production activation follows existing
qualified Qwen hybrid shared-native startup. The affected ordinary text workload
uses the dense singleton route, currently qualified for 4B, 9B and 27B geometries.
MoE pure text uses the native scheduler already; ordinary models outside this
bridge also bypass the changed continuation. No model eligibility or route was
changed to create a favorable benchmark.

Additional [Gemma 4 and Nemotron architecture controls](2026-10-13-prefill-architecture-controls.md) confirm that the acceleration path is disabled for both; they establish no speedup.

The subsequent [27B/default2048 dogfood and shutdown review](2026-10-13-qwen27-default-prefill-dogfood.md) closes the previously untested 27B cell on final production source. This report retains its original per-cell gates, including the M4/2048 failure.

## Frozen experiment

Baseline `bec47abba980d48432c8b5917ca682dd0a5758de`; candidate
`32fdd92c880a8fc1341e37e604c8a568ef558ebb`. The same original HTTP benchmark,
SHA-256 `7ff73a02c9c3b790a70d127cbbf996dfa8d24592e5b6b6d3be155313392d34e7`,
executes both clean production checkouts. It observes scheduler events without
changing decisions. Default route selection, speculation off, disk caches off,
downloads off, temperature zero, thinking off, maximum 32 output tokens.

Each cell has four fresh server children in baseline/candidate then
candidate/baseline order. Each child warms up, runs a cold 16K-token long
singleton and two repetitions of cold short singleton plus long/short overlap.
Short retrieval requests are 294 tokens after templating. Require the expected
answer verbatim, terminal stop/[DONE], correct positive usage, cold caches and
short arrival before long first content. Long singleton receipts are referenced
by both repeats and are not counted as independent samples.

The intended commits, model revisions, chunk setting, four children/eight unique
waves, dependency versions, frozen method hash, source hashes and route activity
are checked separately. Each active cell must show suspended materialized MLLM
prefixes and native peer progress before long first content. Off-path controls
must show no cooperative MLLM work. Performance gates remain at least 30% lower
median concurrent short TTFT and at most 10% long/singleton regression, with
correctness, cache and applicable cancellation checks passing. Report each cell;
no pooled speedup is calculated.

## Environments and observed load

M4 Pro: 48 GiB, macOS 26.5.1, Python 3.12.13, NumPy 2.4.4.
M5 Max: 36 GiB, macOS 27.0.1, Python 3.12.13, NumPy 2.5.4.
Both use MLX 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2,
transformers 5.15.1 and httpx 0.28.1. Every baseline/candidate comparison uses
the same environment on the same hardware; absolute differences between hosts
are not attributed solely to chip differences.

Workers were claimed through the shared scheduler. Process samples were
inspected for unrelated inference, rather than treating a lease as proof of
quietness. Normal desktop services remained running. The M5 had a preexisting
idle serving process (about 60 MiB RSS, 0–0.1% sampled CPU); a read-only status
check returned idle with zero running/waiting requests. It was left untouched.
The 27B run also records memory and swap samples. The earlier M3 Ultra results
remain labeled shared-host observations. The M2 Pro was occupied by another task
and was not borrowed, so no M2 claim is made.

## Results

| Hardware | Model | Text chunk | Short TTFT, baseline → candidate (s) | Short TTFT Δ | Short solo Δ | Long completion Δ | Long solo Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro 48GiB | Qwen3.5-4B-MLX-4bit | 512 | 27.697 → 4.271 | -84.58% | -0.06% | +0.32% | +0.04% |
| M4 Pro 48GiB | Qwen3.5-9B-4bit | 512 | 47.949 → 8.445 | -82.39% | -0.06% | +0.25% | -0.16% |
| M4 Pro 48GiB | Qwen3.5-4B-MLX-4bit | 2048 | 27.778 → 20.073 | -27.74% | +0.03% | +0.28% | +0.00% |
| M5 Max 36GiB | Qwen3.5-4B-MLX-4bit | 512 | 4.588 → 0.747 | -83.71% | -0.26% | +0.64% | +0.05% |
| M5 Max 36GiB | Qwen3.6-27B-4bit | 512 | 23.674 → 3.860 | -83.69% | +0.09% | +0.98% | +0.05% |
| M5 Max 36GiB | Qwen3.5-4B-MLX-4bit | 2048 | 4.177 → 2.403 | -42.47% | +8.13% | +1.08% | -0.14% |
| M5 Max 36GiB | Qwen3.6-35B-A3B-4bit | 512 | 1.635 → 1.636 | +0.05% | +0.20% | -0.70% | +5.25% |

56 overlap waves across 28 fresh children produced 196 distinct measured
streams plus 28 warmups. Negative deltas mean lower latency. These are small
controlled samples, not population guarantees. Native-only MoE is an off-path
control; its timing variation is not attributed to this change.

Both 4B workers used identical weight, config and tokenizer hashes. The 512-token
dense cells restore peer progress across two chip generations and three weight
sets/geometries. The 2048-token cases demonstrate that the size of the gain
depends on scheduling granularity; a fixed 85% claim is unsupported.

## Applicability and qualification

5 of 6 active dense cells meet the preregistered latency gates; all
6 completed dense cells pass answer/usage/cold-cache/terminal/provenance,
peer-progress and their separate cache/cancellation lifecycle checks. The M4
4B/2048 cell improves median short TTFT by 27.74%, below the required 30%, so it
is **unqualified** despite passing correctness and regression limits. No gate
was relaxed. The MoE pure-text cell passes HTTP correctness as a native-only
control; a suspended-MLLM lifecycle pass is neither expected nor claimed.

This is a Qwen dense hybrid concurrency fix with broader hardware and size
evidence, not a model-independent TTFT accelerator. Single-request forward
speed remains essentially unchanged in these samples. The ordinary-text
controls below do not establish compatibility qualification for their families.

Configuration matters: the catalog's 4B/9B four-bit aliases recommend a 512-token
prefill setting, while the 27B alias has no such recommendation. A local snapshot
path does not itself select a catalog profile. The benchmark explicitly sets
512 or 2048; it does not claim all launch settings are defaults. When configured
as 2048, the existing MLLM scheduler budget resolves to 8192, but its existing
text forward cap is 2048; startup logs and frozen source confirm the effective
text chunk. The native lane retains the configured value. No default/profile
policy was changed by this PR.

The 27B memory run retained preexisting swap: sampled use started at 1691.88M,
ended at 1659.88M and never exceeded its starting value. The sampled swap-out
counter did not increase. This is not a zero-swap or peak-memory guarantee.
No OOM or timeout occurred in that completed cell.

The 27B/default-sized-2048 cell is still unqualified: after the first M5 lease
was released, a different task claimed the worker. A FIFO follow-up request
waited ten minutes and was cancelled on timeout; no lease was granted and no
follow-up inference ran. Do not infer the 27B's
default-sized performance from its explicitly configured 512 result.

**Keep draft.** Review the failed 30% cell and these support boundaries before
promotion. Additional machines (including M2), more load shapes, media and
speculative decoding remain outside this campaign. Revert is sufficient for
rollback; no cache or configuration migration is needed.

## Failures and controls

The initial M5 4B cache directory contained metadata without weights, so that
startup failed before inference; a task-local copy of the exact Studio snapshot
was used for the formal campaign. An initial orchestration attempt reused a
recently closed port; the benchmark's port preflight rejected it before model
startup. Extension cells were rerun with separate ports. Neither attempt is
latency evidence.

The 0.6B ordinary-text control failed the baseline's strict answer-format check
(`23 + 19 = 42` instead of `42`). Its campaign is incomplete and unqualified;
request/answer criteria were not relaxed. The MoE suspension-specific probe
observed no suspended MLLM prefill, as expected for its native pure-text route;
that probe is inapplicable and is not represented as passing cancellation.
Raw failure records are retained alongside the completed matrix.

The replacement Granite 8B ordinary-text control also failed the baseline's
strict four-digit answer requirement with a verbose/truncated answer. Both
ordinary control attempts remain incomplete and provide no qualified latency
or before/after correctness result. The completed MoE control supplies the
measured off-path comparison; no activation was forced for any control.

## Evidence and reproduction

- [Frozen runners, receipts, traces, logs, model hashes and host evidence](fixtures/default-prefill-yield/2026-10-13-hardware-model-matrix.tar.gz).
- [Archive SHA-256, size and normalization](fixtures/default-prefill-yield/2026-10-13-hardware-model-matrix.manifest.json).
- [Earlier M3 Ultra shared-host campaign](2026-10-12-default-prefill-yield.md).

The archive's `cells.json` records each completed cell and its explicit gate
status. Incomplete startup/control attempts remain in separate raw directories
and runner logs. Generated logs/JSON normalize HF/user home paths; host process
samples retain timing, CPU/RSS and command basenames. Hardware serials, UUIDs,
credentials and private machine addresses are excluded. Runner source bytes and
all methodology/source hashes remain unchanged. Model manifests hash weights,
configuration and tokenizers; weights themselves are not bundled.

Extract to scratch and validate without model execution, from a repository that
contains the exact source commits:

```sh
mkdir -p /private/tmp/rapid-mlx-prefill-matrix-evidence
tar -xzf docs/engineering/performance/fixtures/default-prefill-yield/2026-10-13-hardware-model-matrix.tar.gz -C /private/tmp/rapid-mlx-prefill-matrix-evidence
python /private/tmp/rapid-mlx-prefill-matrix-evidence/validate_matrix.py
```

To repeat a cell on a worker, prepare clean detached source checkouts at the
frozen runner's fixed paths, using the baseline/candidate commits above as in
the earlier reproduction guide. Use that host's listed dependency versions,
an existing cached snapshot with matching model-manifest hashes, and the same
512 or 2048 arm pair, for example:

```sh
PREFILL_EVIDENCE=/private/tmp/rapid-mlx-prefill-matrix-evidence
PREFILL_MODEL=/path/to/existing/snapshot
python "$PREFILL_EVIDENCE/benchmark_default_prefill.py" --model "$PREFILL_MODEL" --output "$PREFILL_EVIDENCE/rerun/default-routes-final/results.json" --port 18600 --repeat 2 --cycles 2 --workloads retrieval --arms baseline512 candidate512
python "$PREFILL_EVIDENCE/check_http_lifecycle.py" --model "$PREFILL_MODEL" --output "$PREFILL_EVIDENCE/rerun/http-lifecycle-paused" --port 18601
python "$PREFILL_EVIDENCE/validate_matrix_cell.py" "$PREFILL_EVIDENCE/rerun/default-routes-final/results.json" 8
```

For a 2048 cell, use `baseline2048 candidate2048` and
`check_http_lifecycle_2048.py`. Use distinct ports for separate campaigns to
avoid the initial port-preflight failure. For a native-only control, run the
benchmark receipt validator with `--off-path --skip-lifecycle`; this explicitly
makes no suspended-continuation lifecycle claim. Always run the additional
binding validator against the intended snapshot and chunk setting.


The production diff is unchanged from the earlier 641-pass/6-skip focused
regression suite and independent code review. This extension adds measurement
and support-scope evidence only. No merge or release action is included.
