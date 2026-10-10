# Agent-serving cache, concurrency, and prefill experiments

Date: 2026-10-10. Owner: Vector scope. These are exploratory measurements on
one shared Mac, with three repetitions per configuration. No serving code or
defaults were changed.

## Environment and scope

- Apple M3 Ultra Mac Studio, 256 GiB unified memory, macOS 26.5.2.
- Serving source: `dfaa9cc407a0abb80dd0b5aa0afd46c6d249c47b`.
- Python 3.12.13, MLX 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2,
  Transformers 5.15.1, HTTPX 0.28.1.
- Cached `mlx-community/Qwen3.5-4B-MLX-4bit`, revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3`. No model downloads.
- Text-only batched engine, prefix cache enabled, optional disk caches disabled,
  no drafter, temperature zero, thinking disabled explicitly in each request.
- An isolated server bound to `127.0.0.1:18347`; existing unrelated test and
  service processes were left running. Host load was not controlled.

The synthetic prompts contain deterministic numbered arithmetic code records,
with different session identifiers at their beginnings. Cold trials clear only
this experiment server's prefix cache and verify that it is idle. Reported API
cache usage confirms whether a request reused tokens. Model/GPU compilation and
allocator state are warm: "cold" here means an uncached prompt, not process
startup.

TTFT starts at HTTP dispatch and ends at the first non-empty content or reasoning
delta, excluding role-only frames. Each measured stream must have `[DONE]`, a
finish reason, visible output, and positive completion-token usage. Throughput
is the sum of server-reported output tokens divided by the complete wave's wall
time, including client setup, prefill and decode. It is not decode-only speed.
All tables use medians, not statistically established tail-latency estimates.

## Cache reuse

Each repetition sends a fresh approximately 4K-token prompt, repeats it exactly,
then appends one instruction to the user message. Requests allow 96 output
tokens. The rendered initial prompt contains 4,124 tokens; repeat and append
requests reuse 4,096 tokens.

| Configured prefill chunk | Cold TTFT | Exact-repeat TTFT | Appended-instruction TTFT |
| --- | ---: | ---: | ---: |
| 2,048 | 1,942 ms | 70 ms | 100 ms |
| 512 | 2,066 ms | 68 ms | 102 ms |

All three exact repeats in each arm have the same output hash as their paired
cold request. The 2,048 arm's exact-repeat TTFT is approximately 27.7 times lower
than its uncached TTFT. This is a first-token comparison, not an end-to-end
speedup.

## Concurrent sessions

Cold waves contain independent approximately 2K-token prompts (2,076 rendered
tokens), with a barrier before HTTP dispatch. Warm waves independently populate
each session's approximately 4K-token prefix before the measured barrier wave.
Every measured warm request reports 4,096 cached tokens. Population time is
excluded from warm-wave throughput. Each request allows 96 output tokens.

| Chunk | Sessions | Cold TTFT | Cold output token/s | Warm TTFT | Warm output token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 968 ms | 60.4 | 67 ms | 135.2 |
| 2,048 | 2 | 1,989 ms | 69.1 | 102 ms | 212.9 |
| 2,048 | 4 | 3,903 ms | 77.5 | 168 ms | 296.8 |
| 512 | 1 | 1,044 ms | 57.0 | 68 ms | 133.1 |
| 512 | 2 | 2,052 ms | 67.6 | 103 ms | 212.3 |
| 512 | 4 | 3,893 ms | 77.3 | 164 ms | 298.6 |

Warm-session batching scales substantially better than these short-output cold
waves. Four warm sessions give roughly 2.2 times the aggregate output throughput
of one warm session. Smaller chunks do not show a material warm-throughput gain.
Cold and warm waves deliberately have different prompt sizes; their throughput
ratio should not be presented as a matched-prompt cache A/B result.

## Long prefill competing with a short request

Each repetition first measures an independent short prompt alone, then clears
the cache. It dispatches a 16,412-token long prompt, waits until `/v1/status`
confirms an actively scheduled request, and dispatches the same 284-token short
prompt. Both allow 96 output tokens. The default 500 ms decode-stall target stays
enabled in both arms. The status observation does not guarantee an identical
prefill progress offset at short-request arrival.

| Chunk | Short alone TTFT | Short contended TTFT | Long contended TTFT | Median short maximum SSE gap | Wave output token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 190 ms | 6,870 ms | 12,596 ms | 532 ms | 13.6 |
| 512 | 186 ms | 1,897 ms | 11,508 ms | 376 ms | 15.0 |

The short contended TTFT samples were 7,225 / 6,860 / 6,870 ms at 2,048 and
2,343 / 1,896 / 1,897 ms at 512. The smaller chunk lowers the median by 72.4%
in this workload. Its approximately 4K uncached TTFT is 6.4% higher in the earlier
cache experiment. Small single-digit differences remain susceptible to shared
host load and ordering effects.

Maximum SSE gaps describe client-observed streaming events, which may contain
multiple tokens; they are not per-token latency percentiles. The main-matrix
server-reported Metal peaks were 9.67 GB at 2,048 and 7.57 GB at 512. These are
process allocator measurements, not whole-machine memory or context-capacity
limits.

## Recheck and growing conversations

After the 512 arm, a freshly started 2,048 server repeated the contention trial.
Short contended TTFT was 6,934 / 6,867 / 6,856 ms, with median 6,867 ms. This
returns to the original 6,870 ms median, supporting the chunk-size effect despite
the lack of exclusive host control. It is an A/B/A check of contention only,
not a randomized trial of every matrix cell.

Both settings also ran three independent five-turn conversations, replaying
actual assistant text and appending another user message after every turn.
Each request allowed 64 output tokens.

| Chunk | Warm-turn cache hits | Cold first-turn TTFT | Warm-turn TTFT | Median cached fraction |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 12/12 | 1,927 ms | 134 ms | 97.49% |
| 512 | 12/12 | 2,112 ms | 138 ms | 97.50% |

This verifies growing plain-text conversation reuse on the tested OpenAI route;
it does not validate tool-call replay or cross-route cache identity.

## Limits and next action

Same-prompt single-request versus contended output hashes match in zero of three
initial 2,048 trials and one of three 512 trials. Captured 512 texts share a long
initial explanation before wording diverges. Batch-dependent numerics are one
possible explanation; these experiments do not establish the cause or validate
answer quality. Protocol completion passed; output equivalence across batch
shapes did not. These prompts are descriptive, allow truncation at the output
budget, and do not test arithmetic correctness or completed coding tasks.

The measurements do not qualify a large MoE model, sparse-attention kernel,
speculative decoding, other chips, production traces, tool calling, or the
Anthropic route. No speedup from a different accelerator can be inferred.

Vector's next action is a scoped experiment that reduces long-prefill chunks
when an interactive request is waiting, before that neighbour emits its first
token. Existing memory-aware sizing and decode-stall protection should remain
the starting mechanisms. Compare against the existing fixed 2,048 and 512
controls, track long-request completion as well as short-request TTFT, and
investigate output variation with matched batch shapes and deterministic tasks.
Leave global defaults unchanged until that evidence exists.

## Reproduction and evidence

Raw API measurements, environment versions, summaries and the small probe
scripts are in [the fixture directory](fixtures/2026-10-10-agent-serving/).
Use the serving source revision above and the recorded dependency versions.
Copy the scripts to scratch so rerunning does not overwrite committed evidence:

```bash
experiment_dir=/private/tmp/rapid-mlx-agent-serving
mkdir -p "$experiment_dir"
cp docs/engineering/performance/fixtures/2026-10-10-agent-serving/*.py "$experiment_dir/"
export AGENT_PROBE_OUTPUT_DIR="$experiment_dir"
export AGENT_PROBE_URL=http://127.0.0.1:18347
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
probe_model="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3"
PYTHONPATH="$PWD" rapid-mlx serve "$probe_model" \
  --host 127.0.0.1 --port 18347 --served-model-name agent-probe \
  --no-mllm --disable-disk-caches --disable-model-downloads \
  --prefill-step-size 2048
```

After server warmup completes, in another terminal with the same environment:

```bash
python "$experiment_dir/probe.py" --label prefill-2048 --repeat 3
python "$experiment_dir/warm_probe.py" prefill-2048
python "$experiment_dir/multiturn_probe.py" prefill-2048
```

Stop only the experiment server, restart with `--prefill-step-size 512`, and
repeat using label `prefill-512`. Finally restart with 2,048 and run:

```bash
python "$experiment_dir/probe.py" --label prefill-2048-recheck --repeat 3 --contention-only
python "$experiment_dir/summarize.py"
```

The recorded order was the 2,048 main and warm matrices, the 512 main and warm
matrices plus growing conversations, then a fresh 2,048 contention recheck and
growing conversations. Five-turn conversations replay actual assistant text
and append a user message; they contain no tool invocations. Output text capture
was added after the initial 2,048 matrix revealed differing hashes, without
changing its request payloads or timing logic. The distributed scripts also add
output-directory and URL environment overrides and formatting changes.
