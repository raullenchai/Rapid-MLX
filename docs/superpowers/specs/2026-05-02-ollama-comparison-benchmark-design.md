# Ollama Comparison Benchmark Design

## Context

Issue #149 asks for a public, reproducible benchmark that answers how much faster Rapid-MLX is than Ollama. The current repository already has broad benchmark scripts, especially `scripts/benchmark_engines.py`, but those scripts are oriented toward internal engine and capability comparisons. The requested feature is narrower: a standalone Ollama comparison script that launches servers with explicit settings, runs identical workloads, and produces README-ready Markdown plus raw audit data.

The benchmark must own server startup so both engines run with comparable configuration. It must run engines sequentially to avoid Apple Silicon unified-memory and GPU contention.

## Goals

- Add `scripts/bench_vs_ollama.py`.
- Benchmark Rapid-MLX against Ollama on identical prompts and model pairs.
- Launch and stop each server from the script.
- Use sequential engine execution, never simultaneous engine execution.
- Default to the issue's Qwen3.5 model pairs:
  - `qwen3.5-4b` for Rapid-MLX and `qwen3.5:4b` for Ollama.
  - `qwen3.5-9b` for Rapid-MLX and `qwen3.5:9b` for Ollama.
- Auto-fetch missing models by default:
  - Run `ollama pull <model>` for Ollama.
  - Let `rapid-mlx serve <model>` use the normal Hugging Face download path.
- Pick free temporary localhost ports so existing local servers do not affect results.
- Save both raw JSON and Markdown under `reports/benchmarks/ollama-comparison/`.
- Print the final Markdown summary to stdout.

## Non-Goals

- Do not replace `scripts/benchmark_engines.py`.
- Do not add capability tests such as tool calling, reasoning separation, vision, or audio.
- Do not start Rapid-MLX and Ollama at the same time.
- Do not update README benchmark numbers as part of this feature.
- Do not require tests to download models or start real inference servers.

## CLI

The script should expose a focused CLI:

```bash
python scripts/bench_vs_ollama.py
```

Important options:

- `--model-pair RAPID=OLLAMA`: replace the default model matrix with the supplied pair. This flag can be repeated to define a multi-model matrix.
- `--runs N`: measured runs per streaming metric. Default: `3`.
- `--warmups N`: warmup requests excluded from metrics. Default: `1`.
- `--max-tokens N`: completion token limit for primary decode tests. Default: `256`.
- `--concurrency 1 2 4`: concurrency levels for throughput tests. Default: `1 2 4`.
- `--output-dir PATH`: default `reports/benchmarks/ollama-comparison`.
- `--no-pull`: skip `ollama pull`.
- `--no-download`: launch the Rapid-MLX child process with offline Hugging Face environment variables such as `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. If the model is not already cached, server startup fails and the failure is recorded.
- `--startup-timeout SECONDS`: server readiness timeout.
- `--request-timeout SECONDS`: per-request timeout.
- `--rapid-mlx-arg ARG`: append an advanced argument to the Rapid-MLX launch command. Repeatable.
- `--ollama-env KEY=VALUE`: add an environment variable when launching `ollama serve`. Repeatable.

The CLI should make defaults useful for public benchmark reproduction while keeping advanced configuration explicit.

## Server Lifecycle

For every model pair, the script benchmarks Rapid-MLX first and Ollama second:

1. Prepare the engine model.
2. Allocate a free localhost port.
3. Launch the engine server.
4. Wait until the server is ready.
5. Run warmups.
6. Run measured workloads.
7. Stop the server.
8. Continue with the next engine.

The Rapid-MLX command should be explicit:

```bash
rapid-mlx serve <rapid_model> \
  --host 127.0.0.1 \
  --port <port> \
  --no-thinking \
  --default-temperature 0
```

The Ollama command should launch a dedicated daemon on the selected port:

```bash
OLLAMA_HOST=127.0.0.1:<port> ollama serve
```

The script should terminate child processes even when a benchmark request fails or the user interrupts the run. Cleanup belongs in a lifecycle helper that uses `try/finally`.

## Request APIs

Rapid-MLX should use the OpenAI-compatible endpoint:

```text
POST http://127.0.0.1:<port>/v1/chat/completions
```

Ollama should use the native endpoint:

```text
POST http://127.0.0.1:<port>/api/chat
```

The native Ollama API is preferred because it supports `think: false` directly. That keeps the benchmark focused on engine performance rather than reasoning policy.

## Deterministic No-Thinking Mode

All default requests should run in no-thinking deterministic mode:

- Prompts include `/no_think` where appropriate.
- Rapid-MLX requests include:
  - `temperature: 0`
  - `enable_thinking: false`
  - `stream_options: {"include_usage": true}` for streaming metrics.
- Ollama requests include:
  - `think: false`
  - `options.temperature: 0`
  - `options.num_predict: <max_tokens>`

This choice makes TTFT and decode throughput comparable between engines.

## Workloads And Metrics

### Warmup

Run a short request before measured runs. Warmups are excluded from output metrics.

### TTFT And Decode Throughput

Use a streaming chat request with a fixed prompt and `max_tokens`. Repeat it `--runs` times.

Metrics:

- `ttft_ms`: time from request start to first visible content token.
- `decode_tok_s`: completion tokens divided by time after first visible token.
- `completion_tokens`: tokens produced for the run.
- `total_ms`: total request duration.

For Rapid-MLX, prefer server-reported `usage.completion_tokens` when available. For Ollama, prefer final native stream metadata such as `eval_count` and `eval_duration` for decode throughput. If metadata is unavailable, fall back to content chunk counting and wall-clock decode time.

### Multi-Turn Latency

Use a fixed four-turn chat conversation. Each turn sends the accumulated conversation and records end-to-end latency.

Metrics:

- `avg_turn_ms`
- `turn_latencies_ms`

### Concurrent Throughput

Run the same fixed request at concurrency levels `1`, `2`, and `4`.

Metrics:

- `aggregate_tok_s`: total completion tokens divided by wall-clock batch duration.
- `avg_latency_ms`
- `p95_latency_ms`
- `requests`
- `completion_tokens`

Concurrency should use Python async HTTP clients or a thread pool. It must only exercise one engine server at a time.

## Data Model

The raw JSON should include:

- benchmark metadata:
  - timestamp
  - git commit
  - command-line arguments
  - hardware summary when available
  - Python version
  - Rapid-MLX version when available
  - Ollama version when available
- model-pair configuration
- launch commands with sensitive environment values omitted
- per-engine raw runs
- per-engine summaries
- speedup calculations
- error details if an engine fails

Speedup semantics:

- For throughput metrics, speedup is `rapid_mlx / ollama`.
- For latency metrics, speedup is `ollama / rapid_mlx`.
- If the denominator is zero or missing, render speedup as unavailable.

## Markdown Output

The Markdown file should include:

1. Title and timestamp.
2. Hardware and version metadata.
3. Benchmark configuration summary.
4. One table per model pair.
5. A short note explaining that engines were launched sequentially and no-thinking deterministic mode was used.

Each model-pair table should include rows for:

- TTFT.
- Decode tok/s.
- Multi-turn latency.
- Concurrent throughput at 1 user.
- Concurrent throughput at 2 users.
- Concurrent throughput at 4 users.

Columns:

- Metric.
- Rapid-MLX.
- Ollama.
- Speedup.

The same Markdown should print to stdout after files are written.

## Error Handling

- If a required executable is missing, fail with a clear message naming the executable.
- If `ollama pull` fails, stop before benchmarking that Ollama model unless `--no-pull` was set.
- If a server does not become ready before `--startup-timeout`, terminate it and record the failure.
- If one engine fails for a model pair, still record the failure in JSON and continue with remaining model pairs when possible.
- Always terminate child server processes in `finally` blocks.

## Tests

Tests should be fast and offline. They should not download models or launch real inference servers.

Add unit tests for:

- repeated `--model-pair RAPID=OLLAMA` parsing.
- invalid model-pair parsing.
- speedup math for throughput and latency metrics.
- Markdown rendering with complete and partial data.
- Rapid-MLX SSE stream parsing, including final `usage` chunks.
- Ollama native stream parsing, including final metadata chunks.
- lifecycle cleanup with mocked subprocesses.
- free-port allocation returns an integer port that can be rebound after the helper releases it.

Integration testing against real servers can be documented as manual usage and left out of default pytest.

## Open Questions Resolved

- The benchmark launches servers itself.
- Engines run sequentially.
- The default model matrix is Qwen3.5 4B and 9B.
- Missing models are fetched by default.
- Free temporary ports are selected automatically.
- Both raw JSON and Markdown are saved.
- Default requests use deterministic no-thinking mode.
