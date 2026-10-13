# Reproducing prefill interference before the first token

Owner: Vector. Scope: a developer benchmark and exploratory evidence. Serving
library code and operator defaults are unchanged.

A short request sharing a prompt batch with a long request can wait through
several long prompt forwards before its first output token. Existing decode
protection activates once a generation row exists; it does not cover all of
this interval. A short prompt can also have several checkpoint/template
segments, each requiring a tick. The prompt runtime right-pads rows to the
longest input in each tick.

`scripts/benchmark_prefill_first_token.py` compares fixed 2048/512 chunks with
benchmark-only candidates in owned loopback child servers. The cost candidate
first applies existing memory/decode bounds, then limits concurrent prompt work
using recent measured seconds per prompt token and a 500 ms total budget. It
preserves checkpoint grid boundaries. A single prompt retains its configured
size; a row awaiting promotion remains protected. The target is an estimate,
not a deadline, and cannot interrupt an already-running chunk.

Two optional arms test a 1000 ms budget and clipping work to the shortest current
segment to reduce padding. The latter ignores final one-token promotion rows
and queued rows without admission slots. All patches are confined to benchmark
children and depend on private prompt metadata from **mlx-lm 0.31.3**. Use that
version for reproduction; this is not a portable scheduler implementation.

## Recorded experiments

Apple M3 Ultra, 256 GiB, macOS 26.5.2; Python 3.12.14, MLX 0.32.3,
mlx-lm 0.31.3, Transformers 5.15.1. Existing Qwen3.5-4B 4-bit snapshot
`32f3e8ecf65426fc3306969496342d504bfa13f3`. Other jobs remained active:
these shared-host measurements do not qualify a production default.

Each arm gets a fresh warmed process. Forward/reverse cycles vary arm order.
Each process measures a long request alone, then a short request alone before
each paired wave. The long request starts first; status observation triggers
the short request. Prefix caches are cleared between waves; compilation and
online cost estimates remain warm. Prompts are approximately 16K/256 input
tokens, with exact rendered lengths recorded in usage. Thinking, speculation
and disk caches are off. Retrieval outputs must match distinct codes inserted
halfway through the long/short prompts (7319/9283), stop normally and use at most
32 output tokens. Prose uses a 96-token budget and checks protocol completion,
not semantic quality. Every measured stream requires output, usage, `[DONE]`
and zero cached tokens (the server omits the details object for zero). Archived
final server cache statistics also report zero hits/tokens saved. TTFT excludes
role-only frames.

Medians in seconds; TTFT is for the short request and completion for the long.
Each October 11/12-budget arm has four paired samples, the tail pilot two.
Long-alone medians have two independent process samples per arm/workload.

| Campaign / workload | Arm | Short TTFT | Long completion | Long alone |
| --- | --- | ---: | ---: | ---: |
| Oct 11 prose | Fixed 2048 | 14.472 | 41.743 | 23.942 |
| | Fixed 512 | 3.868 | 40.528 | 23.609 |
| | Cost 500 ms | 3.911 | 41.520 | 19.811 |
| Oct 11 retrieval | Fixed 2048 | 10.947 | 21.166 | 17.251 |
| | Fixed 512 | 4.031 | 28.866 | 18.523 |
| | Cost 500 ms | 4.011 | 23.952 | 21.290 |
| Oct 12 budget / retrieval | Fixed 2048 | 8.978 | 17.218 | 11.778 |
| | Fixed 512 | 2.622 | 16.354 | 16.363 |
| | Cost 500 ms | 2.888 | 14.793 | 13.163 |
| | Cost 1000 ms | 3.508 | 14.597 | 12.115 |
| Oct 12 tail pilot / retrieval | Fixed 2048 | 9.431 | 18.153 | 12.142 |
| | Fixed 512 | 2.847 | 16.858 | 13.741 |
| | Cost 500 ms | 2.663 | 14.442 | 12.013 |
| | Cost 500 ms + tail cap | 2.933 | 15.327 | 14.072 |

The 500 ms candidate lowers short TTFT by 63–73% against fixed 2048 in these
campaigns. October 11 retrieval long completion **regresses 13.2%**; varying
solo baselines and host load prevent isolating the cause. October 12 does not
erase that regression. Widening to 1000 ms worsens short TTFT 21.5% for only
1.3% lower long completion. Tail clipping demonstrably dispatches 14-token
segments, but its latency direction reverses between cycles; median short TTFT
and long completion worsen 10.1%/6.1% versus the cost control. Neither optional
candidate has evidence for adoption.

All 48 paired waves across the three campaigns completed: 172 distinct measured
streams plus 22 warmups and 22 clean child shutdowns. All 130 distinct retrieval
streams returned the expected code. Prose outputs differ from singleton hashes
in every arm (0/4 matches for both long and short), so exact retrieval agreement
does not establish arbitrary answer or logits equivalence. Traces show cost
bounds engaging before decode; initial uncontended chunks still cause waiting.

## Running and inspecting the benchmark

Use an existing local snapshot and an unused port. Use a fresh output directory:
results, traces and server logs must not overwrite previous evidence. The tool
starts and terminates only its own children and disables model downloads.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1 \
python scripts/benchmark_prefill_first_token.py \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
  --output /private/tmp/rapid-mlx-first-token-run/results.json \
  --port 18348 --repeat 2 --cycles 2 --workloads retrieval \
  --arms fixed2048 fixed512 adaptive2048
```

Add `adaptive2048-1000ms` or `adaptive2048-tail` to compare those candidates.
For the original October 11 matrix use `--workloads prose retrieval`; the budget
matrix adds the 1000 ms arm, and the tail pilot adds the tail arm with `--repeat 1`.
New results record schema/methodology hash, source revision, runtime versions,
timestamps, client receipts and per-dispatch chunk/row/cost metadata. The PR
also rejects malformed explicit cache usage and short arrivals after the long first token;
archived timings were collected with the earlier source hashes below.

The [evidence directory](fixtures/first-token-prefill/) contains two compressed
archives and an outer SHA-256 manifest. Archives preserve full normalized
results, unchanged traces, validators, environment and original manifests;
compression avoids thousands of raw-data lines in the review diff. Measured
harness revisions are `1b8f137ca` (October 11), `58cb67370` (budget) and
`0449ead1d` (tail). Each results file records the exact full revision and source
hash. These commits remain available in the research history. Absolute command
paths were normalized, with original command/result hashes retained.

Extract into a fresh scratch directory to inspect all samples and validate:

```bash
mkdir -p /private/tmp/rapid-mlx-prefill-evidence
for archive in docs/engineering/performance/fixtures/first-token-prefill/*.tar.gz; do
  tar -xzf "$archive" -C /private/tmp/rapid-mlx-prefill-evidence
done
PYTHONPATH=. python /private/tmp/rapid-mlx-prefill-evidence/2026-10-11-first-token-prefill/validate.py \
  /private/tmp/rapid-mlx-prefill-evidence/2026-10-11-first-token-prefill
PYTHONPATH=. python /private/tmp/rapid-mlx-prefill-evidence/2026-10-12-first-token-budget/validate.py \
  /private/tmp/rapid-mlx-prefill-evidence/2026-10-12-first-token-budget/budget \
  fixed2048 fixed512 adaptive2048 adaptive2048-1000ms
PYTHONPATH=. python /private/tmp/rapid-mlx-prefill-evidence/2026-10-12-first-token-budget/validate.py \
  /private/tmp/rapid-mlx-prefill-evidence/2026-10-12-first-token-budget/tail \
  fixed2048 fixed512 adaptive2048 adaptive2048-tail
```

Vector owns exclusive-host qualification, long completion, cache histories,
cancellation and more model families before any production proposal. Harbor
owns exclusive-run scheduling; Atlas owns later compatibility/default decisions.
This two-request, text-streaming study does not cover saturated admission,
high-concurrency coalescing, tool calls, multimodal or speculative generation.
