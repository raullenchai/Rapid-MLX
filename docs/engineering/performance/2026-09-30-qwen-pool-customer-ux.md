# Qwen 27B pool customer experience, 2026-09-30

This is a small live diagnostic, not a model-quality or throughput benchmark.
An independent Docker client called the production QuickSilver
`qwen3.8-27b` chat endpoint with a separate consumer account. QuickSilver
mapped every request ID to its actual route after the run.

## Reproduce the client workload

- Client: `python:3.12-alpine` on an M3 Ultra Studio, using Python `urllib` and
  `ThreadPoolExecutor`. Mount a consumer key from a local mode-600 file; do
  not put it in the command line or repository.
- Endpoint: `POST /v1/chat/completions`, model `qwen3.8-27b`, one user message,
  `temperature=0`, `max_tokens=160`, `stream=true`, and
  `stream_options.include_usage=true`.
- Prompt A: “A shop sells a notebook and pen for $1.10 total. The notebook
  costs $1.00 more than the pen. Give the pen price and one short equation;
  answer in English.”
- Prompt B: “Write a Python function `dedupe(items)` that removes duplicates
  from a list while preserving order, including unhashable values. Keep the
  answer under 90 words.”
- Synchronize eight client threads with a barrier, alternating A/B. Record
  monotonic time from send to first nonempty SSE `delta.content` and to
  `[DONE]`, plus response ID, status, finish reason, and final usage. Repeat
  two calls at low concurrency. The measured times include client-to-gateway
  network and queueing.

The exact client is [`scripts/benchmarks/qwen_pool_customer_ux.py`](../../../scripts/benchmarks/qwen_pool_customer_ux.py).
With a separate consumer-account inference key stored in a mode-600 local
file, run from the repository root on a Docker-capable Mac:

```bash
QSP_CONSUMER_KEY_FILE=/absolute/path/to/consumer.key  # mode 600, outside Git
mkdir -p /private/tmp/rapid-mlx-qwen-ux
docker run --rm \
  -e QWEN_UX_CONCURRENCY=8 \
  -e QWEN_UX_OUT=/work/out/results.json \
  -v "$PWD/scripts/benchmarks/qwen_pool_customer_ux.py:/work/measure.py:ro" \
  -v "$QSP_CONSUMER_KEY_FILE:/run/secrets/qsp_key:ro" \
  -v /private/tmp/rapid-mlx-qwen-ux:/work/out \
  python:3.12-alpine python /work/measure.py
```

Set `QWEN_UX_CONCURRENCY=2` and `QWEN_UX_OUT=/work/out/results-idle.json`
for the low-concurrency follow-up. The script records response IDs; ask the
gateway operator to map each ID to its actual pool node or cloud route. An ID
prefix alone is insufficient route proof. The key file and raw answers stay
outside Git.

The eight requests began at 19:58:59–19:59:01 UTC. QuickSilver traced six to
pool nodes: MZR-1, MZR-2, and a third Mac Studio each took two slots. The
remaining two fell back to `qwen3.8-27b-cloud` (OpenRouter/Alibaba). All ten
client calls, including the two later low-load calls, returned HTTP 200,
continuous SSE, `finish_reason=stop`, and correct answers to these simple
prompts.

The measured client observations below retain every sample without response
IDs, answer text, or credentials. Prompt A/B correspond to indices 0/1 in the
script; route/node mapping came from QuickSilver's gateway trace of the
response IDs. First token means first nonempty `delta.content`, not first SSE
event. Times are seconds from immediately before `urlopen`; totals end at
`[DONE]`. The pool rows below used the gateway's pre-fix 14-token usage, so
their input-token figures are excluded from the latency table.

| Run | Prompt | Route | First text | Total | Output tokens |
| --- | --- | --- | ---: | ---: | ---: |
| load-1 | A | MZR-2 | 2.380 | 10.816 | 110 |
| load-2 | B | Mac Studio | 3.580 | 12.594 | 40 |
| load-3 | A | Cloud fallback | 2.153 | 2.611 | 29 |
| load-4 | B | Mac Studio | 3.582 | 12.600 | 40 |
| load-5 | A | MZR-2 | 2.375 | 10.799 | 110 |
| load-6 | B | MZR-1 | 2.917 | 6.184 | 40 |
| load-7 | A | MZR-1 | 2.913 | 9.054 | 110 |
| load-8 | B | Cloud fallback | 3.184 | 4.037 | 57 |
| idle-1 | A | Pool | 1.206 | 6.005 | 110 |
| idle-2 | B | Pool | 1.771 | 6.256 | 49 |

The reported pool medians are `statistics.median` of the six load rows marked
MZR-1, MZR-2, or Mac Studio: first text 2.915 s and total 10.8075 s (rounded
to 10.808). The two cloud and two idle rows are shown individually because
each group has only two samples.

| Client path | Samples | First text token | Total response time |
| --- | ---: | ---: | ---: |
| Pool under six-slot load | 6 | median 2.915 s | median 10.808 s |
| Cloud fallback during that load | 2 | 2.153 / 3.184 s | 2.611 / 4.037 s |
| Pool at low concurrency | 2 | 1.206 / 1.771 s | 6.005 / 6.256 s |

The pool's longer total time largely reflects answer length in this sample:
for prompt A, pool responses used 110 output tokens while the cloud fallback
used 29. The pool answer included a worked derivation; both gave the correct
$0.05 result. Prompt B responses also differed in length. This sample cannot
separate model checkpoint, generation policy, network path, and decode speed.

QuickSilver separately ran the same prompts directly from its OVH EU gateway
host to the cloud upstream with streaming, reasoning off, and Alibaba-only
routing. Prompt A took 0.64–0.87 s to first token and 1.06–2.61 s total;
prompt B took 0.44–0.45 s to first token and 1.18–1.63 s total. These times
exclude the Docker client's trip to the gateway, so they are not directly
comparable to the client-path rows above.

The first eight calls preceded QuickSilver's prompt-usage fix: pool responses
reported a constant 14 input tokens because its gateway counted a redacted
message placeholder. Post-fix ordinary and streaming replays of a longer
prompt both reported 576 input tokens in responses and SpendLogs, versus 654
from a native Rapid Qwen tokenizer including the chat template. QuickSilver
confirmed its gateway intentionally bills with a LiteLLM `cl100k_base`
fallback rather than the node's tokenizer; whether to use a per-model
tokenizer is a QuickSilver policy decision.
