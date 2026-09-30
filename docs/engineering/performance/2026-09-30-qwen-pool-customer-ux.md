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

The eight requests began at 19:58:59–19:59:01 UTC. QuickSilver traced six to
pool nodes: MZR-1, MZR-2, and a third Mac Studio each took two slots. The
remaining two fell back to `qwen3.8-27b-cloud` (OpenRouter/Alibaba). All ten
client calls, including the two later low-load calls, returned HTTP 200,
continuous SSE, `finish_reason=stop`, and correct answers to these simple
prompts.

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
