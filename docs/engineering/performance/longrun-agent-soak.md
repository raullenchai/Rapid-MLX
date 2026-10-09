# Long-running agent soak

`scripts/longrun_agent_soak.py` is an opt-in live-server harness. It is outside
unit test discovery and needs a running model. It exercises three concurrent
clients with streaming and non-streaming chat, required tool calls followed by
mock tool results, repeated prompt prefixes, long prompts, stream disconnects,
and task cancellation. It probes `/health`, `/v1/models`, and `/v1/status` once
per minute. All connections should target a loopback listener.

```sh
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit \
  --host 127.0.0.1 --port 18130 --served-model-name qwen3.5-4b-4bit \
  --max-num-seqs 3 --cache-memory-mb 1024 \
  --gpu-memory-utilization 0.35 --enable-prefix-cache --no-thinking --no-mllm
python scripts/longrun_agent_soak.py \
  --url http://127.0.0.1:18130 --model qwen3.5-4b-4bit \
  --pid SERVER_PID --duration 10800 --concurrency 3 \
  --max-rss-mb 11444 --max-metal-gb 12 --output /path/to/run
```

The harness writes `minutes.csv` (per-minute request counts, errors, p50/p95/p99
latency, process RSS, thread and open-file counts, Metal active/cache/peak
memory, scheduler running/waiting counts, and probe results), `errors.jsonl`
(per-request failure details), and `summary.json` (including the pass/fail
outcome). It exits nonzero on request, probe, telemetry, or budget failure.
A successful health probe
alone does not establish liveness: a ghost scheduler slot can leave HTTP
healthy while chat is stuck. Check that completions continue, timeout counts
stay flat, and `running`/`waiting` recover after load. RSS and active Metal
memory overlap on unified-memory Macs; do not add them to estimate total use.

The default guards stop the harness above 12 GB of process RSS or 12 GB of
active plus cached Metal allocations. They do not stop the server; the
operator owns that process and should record its PID before the run. These
are process limits,
not a host-wide memory admission controller. A run that reaches either guard
is a failed soak.

## Background service mode

`--daemon` gives each worker its own HTTP client and growing chat history. The
history rotates after `--session-max-turns` successful attempts to bound a
24-hour run; each new session starts with its own system prompt. The client
reconnects after `--reconnect-every` requests. All workers send during each
`--burst-seconds` window and stop during `--idle-seconds`. Use the primary-model
`--idle-unload-seconds` serve flag and pass the same value to the harness as
`--expected-ttl-seconds`. The idle window must also cover the request deadline,
the server's TTL monitor interval, and the ten-second early probe, so a request
started at the end of a burst cannot consume the whole idle gap.
The verdict requires one observed unload per complete idle cycle and a reload
before each subsequent burst, plus reconnects, session rotations, completed
answers, correct tool arguments, and the existing cancellation, disconnect,
probe, and resource checks. It takes an extra probe ten seconds before each
new burst to prove that cycle unloaded, and checks the reported server TTL.
During standby the server has no Metal telemetry; the harness checks RSS,
threads, and open files against the first settled unloaded sample instead.

```sh
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit \
  --host 127.0.0.1 --port 18620 --served-model-name qwen3.5-4b-4bit \
  --max-num-seqs 6 --cache-memory-mb 1024 \
  --gpu-memory-utilization 0.35 --enable-prefix-cache --no-thinking --no-mllm \
  --idle-unload-seconds 30
python scripts/longrun_agent_soak.py \
  --daemon --url http://127.0.0.1:18620 --model qwen3.5-4b-4bit \
  --pid SERVER_PID --duration 86400 --concurrency 6 \
  --burst-seconds 120 --idle-seconds 180 --expected-ttl-seconds 30 \
  --output /path/to/run-24h
```

Qwen3.5-4B's default multimodal lane serializes ArraysCache requests, so this
command uses `--no-mllm` to exercise concurrent text scheduling. It cannot
validate the Gemma 4 MLLM failure in [#3303](https://github.com/raullenchai/Rapid-MLX/issues/3303)
or the large hybrid Qwen3.6 handle ceiling in
[#2836](https://github.com/raullenchai/Rapid-MLX/issues/2836). It is a
long-running regression signal for adjacent cancellation, prefix-cache, and
resource-retention failure modes.

## 2026-10-07 mac-mini result

Three-hour run on an Apple M2 Pro Mac mini (32 GB, macOS 26.5.2), Python
3.13.11, `mlx-community/Qwen3.5-4B-MLX-4bit`, using the server flags above and
base server commit `e77bbeb03`. The run used the earlier 12,288 MiB RSS guard;
the current replay command tightens it to 11,444 MiB (under 12 GB). Observed
RSS stayed below 2,613 MiB. The server had already warmed up during
short harness pilots. The [minute-by-minute CSV](longrun-agent-soak-2026-10-07.csv)
contains all 182 samples, including the initial and final samples. The final
sample was taken after the last in-flight requests completed. A separate
[post-load idle sample](longrun-agent-soak-2026-10-07-idle.json) was taken
30 seconds later.

| Measure | Start | End / total |
| --- | ---: | ---: |
| Duration | 0 | 10,803.7 s |
| Completed agent turns | 0 | 5,533; 0 errors |
| Disconnects / cancellations | 0 / 0 | 407 / 240 |
| `/health` and `/v1/models` probes | — | 182 / 182 successful each |
| Process RSS | 2,603.6 MB | 2,606.2 MB |
| Metal active / cache, idle | 3.40 / 0.00 GB | 3.60 / 0.00 GB |
| Threads / open files, idle | 25 / 140 | 23 / 139 |
| Scheduler running / waiting, idle | 0 / 0 | 0 / 0 |

The highest sampled active Metal allocation was 6.86 GB; the highest sampled
active plus cached allocation was 7.61 GB. The process-reported Metal peak
was 7.06 GB (6.94 GB had already been reached during pilot warmup).
The server log, including pilots, recorded 2,773 prefix-cache hits and no
Metal resource-limit, admission-cap, generation-recovery, or traceback
signatures. Completions
continued throughout the final hour, with no rising RSS, thread, or open-file
trend. The average of each full minute's p95 latency was 11.37 s in minutes
1–10 and 8.28 s in minutes 171–180; individual long-prompt minutes were higher.

No hang or leak reproduced in this Qwen3.5 text-lane workload. The tested
symptoms look resolved on current `main` for this workload; the original
hybrid Qwen3.6 and Gemma 4 MLLM issue configurations still need their own
model- and hardware-matched qualification. No server fix was needed here.
