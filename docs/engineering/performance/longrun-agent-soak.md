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
  --max-rss-mb 12288 --output /path/to/run
```

The harness writes `minutes.csv` (per-minute request counts, errors, p50/p95/p99
latency, process RSS, thread and open-file counts, Metal active/cache/peak
memory, scheduler running/waiting counts, and probe results), `errors.jsonl`
(per-request failure details), and `summary.json`. A successful health probe
alone does not establish liveness: a ghost scheduler slot can leave HTTP
healthy while chat is stuck. Check that completions continue, timeout counts
stay flat, and `running`/`waiting` recover after load. RSS and active Metal
memory overlap on unified-memory Macs; do not add them to estimate total use.

The default `--max-rss-mb` stops the harness above 12 GiB of process RSS. It
does not stop the server; the operator owns that process and should record its
PID before the run. The 12 GiB guard is a process RSS guard, not a host-wide
memory admission controller. A run that reaches the guard is a failed soak.

Qwen3.5-4B's default multimodal lane serializes ArraysCache requests, so this
command uses `--no-mllm` to exercise concurrent text scheduling. It cannot
validate the Gemma 4 MLLM failure in [#3303](https://github.com/raullenchai/Rapid-MLX/issues/3303)
or the large hybrid Qwen3.6 handle ceiling in
[#2836](https://github.com/raullenchai/Rapid-MLX/issues/2836). It is a
long-running regression signal for adjacent cancellation, prefix-cache, and
resource-retention failure modes.
