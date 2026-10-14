# Default-route shared-executor prefill continuation (2026-10-12)

Status at this initial campaign: draft, with quiet-host qualification pending.
See the [cross-hardware/model follow-up](2026-10-13-prefill-hardware-model-matrix.md)
for subsequent evidence, failed gates and the current support boundary.
Owner: Vector, Studio. Independent of the native scheduler contention candidate.

A long text request using the serialized multimodal singleton lane previously ran
all of its bounded prefill chunks in one executor submission. Concurrent requests
using the qualified native text lane share that same executor and could not run
until the full prompt returned. The candidate returns after each materialized
text-prefix chunk, retains the same request/cache and checkpoint cursor, and
resumes it on the next scheduler tick. The final one-token forward and sampling
still happen once; cache state is not migrated between lanes.

Activation follows successful startup of the existing qualified shared native
text engine and is passed through lazy batch-generator initialization. It is
restricted to a single-request completion lane. Media, informative masks,
sequence-aligned processor kwargs preserve their indivisible forward behavior.
Short prompts retain their existing forward sequence, including cache-boundary
splits, within one executor turn. Removal, exceptions and close release a
suspended continuation.
Active prefill intervals are accumulated across resumes, excluding executor
waits while preserving the existing reporting convention. The public
request/response schema and routing policy do not change.

## Experiment contract

Use the default HTTP route, without forcing text-only engine selection. Both arms
execute production code from clean checkouts. The harness only observes native
scheduler decisions and multimodal next-call boundaries; it does not override a
scheduling decision. Alternate baseline/candidate then candidate/baseline across
two cycles. Each fresh child warms up, runs one cold long singleton, then repeats
cold short singleton plus cold long/short overlap twice. Clear both prefix stores
between waves. Observe a running long request before starting its short peer and
reject any pair where the short request arrives after the long first token.

Both arms explicitly use a 512-token prefill step, prefill batch limit 8,
speculative decoding off, disk caches off and downloads disabled. Default refers
to route selection: the benchmark does not force the native text-only route.
Other chunk sizes and media workloads are not qualified by this campaign.

The retrieval workload places different four-digit markers in long and short
prompts. Require exact answers, thinking disabled, temperature zero, terminal
stop and [DONE], positive usage, a 32-output-token limit and zero cache reuse in
all measured streams. Long raw template budget is 16,384 tokens; short is 256
(294 tokens after templating). The experiment validates answer integrity rather
than assuming that a request completing is sufficient correctness evidence.

Environment: Apple M3 Ultra, 256 GiB unified memory; Python 3.12.14; MLX 0.32.3,
mlx-lm 0.31.3, mlx-vlm 0.7.2, transformers 5.15.1, numpy 2.5.4, httpx 0.28.1.
Cached model: mlx-community/Qwen3.5-4B-MLX-4bit, snapshot
`32f3e8ecf65426fc3306969496342d504bfa13f3`. No model downloads. Other owners'
inference tasks were active; these are descriptive shared-host observations.

## Evidence and qualification

| Median metric | Baseline (s) | Candidate (s) | Descriptive change |
| --- | ---: | ---: | ---: |
| Short TTFT under contention | 13.295 | 2.044 | -84.62% |
| Short singleton TTFT | 0.349 | 0.330 | -5.37% |
| Long completion under contention | 13.170 | 13.470 | +2.27% |
| Long singleton completion | 12.810 | 12.816 | +0.05% |

Eight paired waves across four fresh children produced 28 distinct measured
streams plus four warmups. Each arm has four overlap waves, four short singleton
samples and two long singleton samples (each long singleton receipt is referenced
by both repeats; it is not two independent samples).

Both baseline traces have zero native ticks between short request start and long
first-token delivery. Both candidate traces have eight such native ticks per
wave, and all four candidate short streams produce their first content before
the long stream's first content. The candidate records 102 materialized prefix
returns per child; baseline records none. The unchanged native scheduler source
hash and identical 16 native ticks per child confirm this campaign does not
exercise a native scheduler budget change.

The short-TTFT median falls in both cycle orders: 13.334 to 2.079 seconds in
cycle 0 and 13.171 to 2.000 seconds in cycle 1. Other owners' inference tasks were
active. Execution order gives direct
evidence of restored peer progress; long/singleton timing shifts are not
attributed to this patch, and shared-host timings do not qualify quiet-host gates.

Acceptance remains at least 30% lower median short TTFT under contention and no
more than 10% long-completion or singleton regression, with correctness/cache/
lifecycle gates passing, on a matched quiet-host campaign. Shared-host medians
alone cannot authorize promotion. No broader model-family or media performance
claim is made. Rollback is to revert this candidate; neither cached request state
nor public configuration requires migration.

## Verification and reproduction

641 focused tests passed, 6 skipped. The focused set covers generator/sampling,
qualified native startup, lazy initialization, media/prefix-cache isolation,
executor cancellation, cross-thread streams, usage/logprobs, hybrid checkpoints,
disk checkpoints and request/protocol cancellation. Ruff and diff checks passed.

The separate default-route HTTP probe returned five correct non-streaming
answers. Cold prompt usage was 5,469 tokens; exact and shared-prefix requests
reused 5,454 tokens. An observational worker trace identified the long request
while its prefill continuation was suspended. It was cancelled before any
content token, while a native peer was active. The cancelled stream ended with
[DONE], the peer completed correctly, status returned zero running/waiting,
a repeated cancel returned 404, and the next singleton answer was correct.

The evidence archive contains both exact runners, the observational lifecycle
server, metadata helper, receipt validator, final raw JSON, traces and logs,
a host-load sample, and a manifest with hashes for each member. Only the local
HF cache root in generated JSON/logs is replaced by `${HF_CACHE}`; source bytes,
methodology hashes, receipt times and outputs are unchanged. No exploratory or
incomplete campaign is included.

- Baseline: `bec47abba980d48432c8b5917ca682dd0a5758de`.
- Candidate: `32fdd92c880a8fc1341e37e604c8a568ef558ebb`.
- [Evidence archive](fixtures/default-prefill-yield/2026-10-12-default-route.tar.gz).
- [Archive hash and size](fixtures/default-prefill-yield/2026-10-12-default-route.manifest.json).

On the same dependency versions and cached snapshot, from this repository,
prepare clean checkouts in a fresh scratch namespace matching the runner's
fixed `CODE_ROOTS` and extract its frozen harness:

```sh
git worktree add --detach /private/tmp/rapid-mlx-default-prefill/base bec47abba980d48432c8b5917ca682dd0a5758de
git worktree add --detach /private/tmp/rapid-mlx-default-prefill/worktree 32fdd92c880a8fc1341e37e604c8a568ef558ebb
mkdir -p /private/tmp/rapid-mlx-default-prefill/evidence
tar -xzf docs/engineering/performance/fixtures/default-prefill-yield/2026-10-12-default-route.tar.gz -C /private/tmp/rapid-mlx-default-prefill/evidence
PREFILL_EVIDENCE=/private/tmp/rapid-mlx-default-prefill/evidence
PREFILL_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3"
python "$PREFILL_EVIDENCE/benchmark_default_prefill.py" --model "$PREFILL_MODEL" --output "$PREFILL_EVIDENCE/rerun/default-routes-final/results.json" --port 18367 --repeat 2 --cycles 2 --workloads retrieval --arms baseline512 candidate512
python "$PREFILL_EVIDENCE/check_http_lifecycle.py" --model "$PREFILL_MODEL" --output "$PREFILL_EVIDENCE/rerun/http-lifecycle-paused" --port 18369
python "$PREFILL_EVIDENCE/validate_evidence.py" "$PREFILL_EVIDENCE/rerun/default-routes-final/results.json" 8
```

Validate the archived receipts without rerunning models with:

```sh
python /private/tmp/rapid-mlx-default-prefill/evidence/validate_evidence.py /private/tmp/rapid-mlx-default-prefill/evidence/default-routes-final/results.json 8
```

Changing the fixed source roots or harness bytes changes the methodology hash
and requires new receipts. Do not combine campaigns with different hashes.
The next checkpoint belongs to Vector/Harbor: rerun the same matrix without
other inference jobs and retain all correctness/cache/lifecycle gates before
asking Atlas to promote the draft. Larger chunk sizes and additional qualified
model geometries need separate evidence; no release or merge action is included.
