# CLI Reference

## Commands Overview

| Command | Description |
|---------|-------------|
| `rapid-mlx serve` | Start OpenAI-compatible server |
| `rapid-mlx chat` | Interactive chat REPL with a model |
| `rapid-mlx bench` | Run performance benchmarks |
| `rapid-mlx models` | List available model aliases |
| `rapid-mlx ls` | List models in the local HuggingFace cache (alias for `models --cached`) |
| `rapid-mlx recipe` | Recommend the Smart and Fast models for this Mac's RAM |
| `rapid-mlx info` | Show the per-model profile for an alias or repo |
| `rapid-mlx pull` | Download a model into the HuggingFace cache |
| `rapid-mlx rm` | Remove a cached model |
| `rapid-mlx alias` | Manage user-owned model aliases (`set` / `remove` / `list`) |
| `rapid-mlx ps` | List running rapid-mlx servers |
| `rapid-mlx share` | Expose a local model behind a public URL via rapidmlx.com |
| `rapid-mlx launch` | One-shot bootstrap: patch an IDE/agent client config to use rapid-mlx |
| `rapid-mlx connect` | Show the server's connection info and wire up a tool |
| `rapid-mlx agents` | List, configure, and test agent integrations |
| `rapid-mlx doctor` | Run self-diagnostic / regression harness |
| `rapid-mlx telemetry` | Manage anonymous usage telemetry (opt-in) |
| `rapid-mlx upgrade` | Upgrade rapid-mlx (brew / pip / install.sh) |
| `rapid-mlx version` | Show version number |
| `rapid-mlx help <cmd>` | Show help for a subcommand |

Run `rapid-mlx <cmd> --help` for the full flag list of any subcommand.

Agent integrations can be configured without hand-editing dotfiles:

```bash
# Preview the exact change (never writes)
rapid-mlx agents claude-code --setup --dry-run
rapid-mlx agents continue --setup --dry-run

# Confirm interactively, back up existing config, write atomically, and verify
rapid-mlx agents claude-code --setup
rapid-mlx agents continue --setup
```

Use `--yes` for an explicitly non-interactive apply. `--no-check` skips only
the post-write server health/model check; preview, merge, backup, and atomic
write behavior are unchanged.

## `rapid-mlx serve`

Start the OpenAI-compatible API server.

### Usage

```bash
rapid-mlx serve <model> [options]
```

### Options

Every flag visible in `rapid-mlx serve --help`, grouped by category. Defaults
are the argparse defaults from `vllm_mlx/cli.py`.

#### Network and process

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host (loopback-only by default; pass `0.0.0.0` to expose on LAN — review the auth posture first) | 127.0.0.1 |
| `--listen-fd` | File descriptor of a pre-bound listening socket (3-1023) for socket activation (launchd/systemd/parent-process supervision). When set, `--host`/`--port` are ignored for binding. | None |
| `--log-level` | Log level for Python logging and uvicorn (`DEBUG`, `INFO`, `WARNING`, `ERROR`; case-insensitive) | INFO |
| `--served-model-name` | Model name reported by the API; when unset the `model` argument is used | None |
| `--watchdog-ppid` | Self-terminate when the parent process with this PID dies (defeats orphaned sidecars). Falls back to `RAPID_MLX_WATCHDOG_PPID`; 0 / unset disables. | None (disabled) |

#### Security and limits

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | API key for authentication; falls back to `RAPID_MLX_API_KEY` (inline value wins when both are set); if neither is set, no auth is required | None |
| `--cors-origins` | Allowed CORS origins (space-separated); also settable via `RAPID_MLX_CORS_ALLOW_ORIGINS` (comma-separated) | `*` (all origins) |
| `--trusted-hosts` | Opt-in Host-header allowlist (DNS-rebinding hardening): non-matching requests get HTTP 400. Space- or comma-separated; also settable via `RAPID_MLX_TRUSTED_HOSTS`. | None (not enforced) |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--max-request-bytes` | Maximum HTTP request body size in bytes; larger requests are rejected with HTTP 413 before JSON parsing. 0 disables. Falls back to `RAPID_MLX_MAX_REQUEST_BYTES`. | 8 MiB (8388608) |
| `--timeout` | Request timeout in seconds | 1800 |

#### Admission and batching

| Option | Description | Default |
|--------|-------------|---------|
| `--max-num-seqs` | Max concurrent sequences | 256 |
| `--max-concurrent-requests` | Admission cap on in-flight requests (queued + running); when exceeded, new requests get HTTP 503 with `Retry-After` | 256 |
| `--prefill-batch-size` | Max prompts prefilled together in one cold wave; lower it to cut first-token latency under concurrent cold load, at an aggregate-throughput cost on large MoE models | 8 |
| `--completion-batch-size` | Completion batch size | 32 |
| `--prefill-step-size` | Chunk size for prompt prefill processing; larger values use more memory but can improve prefill throughput | 2048 |
| `--stream-interval` | Tokens to batch before streaming (1 = smooth, higher = throughput) | 1 |
| `--gpu-memory-utilization` | Fraction of device memory for the Metal allocation limit and emergency cache-clear threshold (0.0-1.0); increase to 0.95 for very large (200GB+) models | 0.90 |

#### Prefix, KV, and response caching

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-prefix-cache` | Enable prefix caching for repeated prompts | enabled |
| `--disable-prefix-cache` | Disable prefix caching | off |
| `--prefix-cache-index` | Prefix-cache lookup index: `radix` (token trie, surfaces dedup-bytes-saved on `/metrics`) or `hash` (legacy bisect path) | radix |
| `--prefix-cache-size` | Max entries in the prefix cache (legacy entry-count mode only) | 100 |
| `--cache-memory-mb` | Cache memory limit in MB | Auto (~20% of RAM) |
| `--cache-memory-percent` | Fraction of available RAM for cache when auto-detecting | 0.20 |
| `--idle-cache-clear-seconds` | Clear reusable prefix/KV cache after this many idle seconds; model weights remain loaded. 0 disables. Falls back to `RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS`. | Disabled |
| `--no-memory-aware-cache` | Use the legacy entry-count cache instead of the memory-aware cache | off |
| `--hybrid-cache-entries` | Opt-in trim-free prefix reuse: retain N non-trimmable prefix-cache entries (stable prefix + new suffix each turn) for hybrid (GatedDeltaNet/Mamba) and sliding-window (Gemma 4, GPT-OSS) models. 0 disables. | 0 |
| `--response-cache-entries` | Opt-in response cache: retain N fully-computed greedy (`temperature 0` / `top_k 1`) chat completions; a completely repeated request returns the stored completion with zero GPU decode. 0 disables. | 0 |
| `--pin-system-prompt` | Auto-pin the system prompt in the prefix cache so memory pressure cannot evict it | off |
| `--use-paged-cache` | Enable paged KV cache (experimental) | off |
| `--paged-cache-block-size` | Tokens per cache block | 64 |
| `--max-cache-blocks` | Maximum cache blocks | 1000 |

#### KV cache dtype and quantization

| Option | Description | Default |
|--------|-------------|---------|
| `--kv-cache-dtype` | KV cache dtype (`bf16`, `int8`, `int4`). int8/int4 shrink the KV cache 2x/4x for memory-constrained hosts, but dequant-on-read costs decode throughput at long context (measured -27% int4 / -36% int8 at 16k). Sliding-window (Gemma 3, GPT-OSS) and MLA (DeepSeek V3+, Kimi K2.5) models auto-downgrade to bf16. | bf16 |
| `--reasoning` | Reasoning profile: pins `--kv-cache-dtype` to int8 regardless of the dtype flag (sub-4-bit KV drops accuracy on AIME-class math) | off |
| `--kv-cache-quantization` | Deprecated alias of `--kv-cache-dtype int8`; wins when both flags are passed (backwards compatibility) | off |
| `--kv-cache-quantization-bits` | Bit width for KV cache quantization (4 or 8) | 8 |
| `--kv-cache-quantization-group-size` | Group size for KV cache quantization | 64 |
| `--kv-cache-min-quantize-tokens` | Minimum tokens for quantization to apply | 256 |
| `--kv-cache-turboquant` | TurboQuant KV-cache compression (experimental). Bare flag = `v4` (V-only 3-4 bit Lloyd-Max, K in FP16); `k8v4` = K 8-bit Walsh-Hadamard + V 4-bit mix (~4.6x KV compression on dense models); `none` = explicit off-switch overriding alias-driven auto-resolution. Mutually exclusive with `--kv-cache-quantization`. | None (alias-driven) |
| `--kv-cache-turboquant-bits` | V-side bit width for TurboQuant (3 or 4); ignored in `k8v4` mode (V pinned to 4-bit) | Auto by head_dim (3-bit for >=96, 4-bit for 64) |
| `--kv-cache-turboquant-group-size` | Group size for TurboQuant V-side quantization | 32 |
| `--kv-disk-checkpoint-interval` | Token interval at which the scheduler snapshots KV state to `~/.cache/rapid-mlx/kv_checkpoints/`. 0 disables. Write-only today and each snapshot blocks decode for O(context) — enable only for external tooling that consumes the files. Disk cap via `RAPID_MLX_KV_CHECKPOINT_MAX_BYTES`. | 0 (disabled) |
| `--metal-cap-kv-bytes-per-token` | Override the per-token KV-cache size (bytes) the admission gate projects. Set when running a quantized KV cache so long prompts are not spuriously 503'd; under-setting risks the OOM cliff the gate prevents. 0 auto-derives an architecture-aware fp16 figure. | 0 (auto) |

#### Model loading, residency, and modalities

| Option | Description | Default |
|--------|-------------|---------|
| `--force-disk-check` | Proceed even when the pre-flight disk-space check fails — the check still runs and prints its numbers, but a shortfall becomes a warning instead of an abort (the download will likely fail mid-way) | off |
| `--disk-stream` | Stream MoE routed-expert weights from disk instead of holding them resident (opt-in; only architectures registered in `vllm_mlx.registry`) | off |
| `--disk-stream-cache-gb` | Byte budget (GB) for the disk-stream expert LRU cache; only used with `--disk-stream` | 1.0 |
| `--resident-memory-limit-gb` | Process-wide resident model ceiling in GiB; loading another model evicts the least-recently-used idle unpinned model first. 0 disables. | 0 (disabled) |
| `--resident-model-idle-ttl` | Evict idle unpinned secondary models after this many seconds. 0 disables. | 0 (disabled) |
| `--mllm` | Force-load as multimodal (vision) even if the name doesn't match auto-detection, and hard-fail instead of silently auto-degrading to text-only when the checkpoint ships no usable vision tower | off |
| `--no-mllm` / `--text-only` | Force-load as text-only even when auto-detection would route to the multimodal path (escape hatch for incomplete vision-tower checkpoints) | off |
| `--vision-min-pixels` | Minimum pixels for dynamic-resolution VLM image processors; 0 keeps the model default | 0 |
| `--vision-max-pixels` | Maximum pixels for dynamic-resolution VLM image processors (lower trades image detail for lower TTFT and memory); 0 keeps the model default | 0 |
| `--enable-audio` | Mount the `/v1/audio/*` routes even when the loaded model is text-only (side-car deployments). Audio-capable models auto-mount the routes. | off |

#### Generation defaults

| Option | Description | Default |
|--------|-------------|---------|
| `--max-tokens` | Default max tokens for generation | 32768 |
| `--default-temperature` | Default temperature when not specified in request | None (model default) |
| `--default-top-p` | Default top_p when not specified in request | None (model default) |
| `--default-top-k` | Default top_k when not specified in request | None (model default) |
| `--default-min-p` | Default min_p when not specified in request | None (model default) |
| `--default-repetition-penalty` | Default repetition_penalty when not specified in request | None (model default) |
| `--default-presence-penalty` | Default presence_penalty when not specified in request | None (model default) |
| `--default-frequency-penalty` | Default frequency_penalty when not specified in request | None (model default) |

#### Tool calling and reasoning

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | Enable automatic tool calling | off |
| `--tool-call-parser` | Tool call parser (e.g. `hermes`, `llama`, `deepseek`, `deepseek_v31`, `glm47`, `gemma4`, `minimax`, `kimi`, `harmony`, `qwen3_coder_xml`). Auto-detected from the model name; explicit flag overrides (a literal `auto` is also accepted). | None (auto-detected) |
| `--no-tool-call-parser` | Force-disable tool-call parser auto-detection from the alias profile; mutually exclusive with `--tool-call-parser` | off |
| `--enable-tool-logits-bias` | Bias logits toward structural tool-call tokens for faster generation; only active when `--tool-call-parser` is set (currently supports minimax) | off |
| `--reasoning-parser` | Reasoning parser (`qwen3`, `deepseek_r1`, `deepseek_r1_distill`, `deepseek_v4`, `gemma4`, `glm4`, `gpt_oss`, `harmony`, `hy3`/`hy_v3`, `minimax`, `muse`, `ui_tars`, `vibethinker`). Auto-detected from the alias profile; explicit flag overrides. There is no literal `auto` value — omit the flag for auto-detection. | None (auto-detected) |
| `--no-reasoning-parser` | Force-disable reasoning-parser auto-detection from the alias profile (unlike `--no-thinking`, only skips the auto-config step); mutually exclusive with `--reasoning-parser` | off |
| `--no-thinking` | Disable the reasoning/thinking parser even if auto-detected; thinking tokens appear as regular content | off |
| `--relocate-mid-conversation-system` | Keep a mid-conversation system message at its position (folded into the next user turn) instead of hoisting it into the leading system block; preserves the prefix cache for clients that inject reminders mid-session | off |

#### Profile overrides and engine toggles

Escape hatches for when the per-alias profile's auto-detection misfires; each
binary auto-routing decision has a force-on and force-off pair.

| Option | Description | Default |
|--------|-------------|---------|
| `--force-hybrid` | Force-treat the model as a hybrid (linear-attention / Mamba) architecture; disables spec/suffix decode paths that are unsound on hybrids. Mutually exclusive with `--no-hybrid`. | off |
| `--no-hybrid` | Force-treat the model as non-hybrid (full attention) so spec/suffix decode stays available. Mutually exclusive with `--force-hybrid`. | off |
| `--force-spec-decode` | Force-enable speculative-decode eligibility even when the profile says unsupported (risky on hybrids). Mutually exclusive with `--no-spec-decode`. | off |
| `--no-spec-decode` | Force-disable speculative-decode eligibility (suffix / MTP / DFlash / DDTree). Mutually exclusive with `--force-spec-decode`. | off |
| `--force-openai-harmony-streaming` | Force-on the HarmonyStreamingRouter upgrade even when the compat gate would reject (debug only). Mutually exclusive with `--no-openai-harmony-streaming`. | off |
| `--no-openai-harmony-streaming` | Skip the HarmonyStreamingRouter upgrade and use the legacy harmony state machine on matched-vocab gpt-oss tokenizers. Mutually exclusive with `--force-openai-harmony-streaming`. | off |
| `--gc-control` | Pause Python GC during generation to avoid latency spikes | enabled |
| `--no-gc-control` | Disable GC control (allow normal Python GC during generation) | off |

#### Speculative decoding

| Option | Description | Default |
|--------|-------------|---------|
| `--speculative-config` | vLLM-style speculative decoding JSON config (`dflash`, `ddtree`, `mtp`, `suffix`, ...); see [Speculative Decoding Options](configuration.md#speculative-decoding-options) | None |

#### Embeddings

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-model` | Pre-load an embedding model at startup (requires `pip install 'rapid-mlx[embeddings]'`) | None |
| `--embedding-max-length` | Max input length (tokens) for the embedding model; `auto` derives it from the model's declared maximum, or pass a positive integer for a lower operational ceiling. Inputs above the limit follow `--embedding-overflow-policy` (never truncated silently). | auto |
| `--embedding-overflow-policy` | Overflow handling: `truncate` (discards the tail, logs a warning, increments the `rapid_mlx_embedding_truncations_total` metric) or `error` (HTTP 400 with observed and allowed token counts) | truncate |

#### PFlash long-prompt compression

| Option | Description | Default |
|--------|-------------|---------|
| `--pflash` | PFlash long-prompt prefill compression: `off`, `auto`, `always` | `always` for verified aliases (Qwen3.5 / Qwen3.6 family), `off` otherwise |
| `--pflash-threshold` | Minimum prompt tokens before `--pflash auto` compresses | 32768 |
| `--pflash-keep-ratio` | Fraction of prompt tokens to keep when compressing; unset resolves a per-alias override if pinned, else 0.20 | None (per-alias or 0.20) |
| `--pflash-min-keep-tokens` | Minimum tokens to keep when compressing | 2048 |
| `--pflash-sink-tokens` | Leading prompt tokens always kept | 256 |
| `--pflash-tail-tokens` | Trailing prompt tokens always kept | 2048 |
| `--pflash-block-size` | Middle-token scoring block size | 128 |
| `--pflash-query-window` | Trailing query window used to score middle blocks | 512 |
| `--pflash-stride-blocks` | Keep every Nth middle block as an anchor during scoring (0 disables anchors) | 8 |
| `--pflash-include-tools` | Allow compression on prompts with tool definitions (skipped by default for tool-call reliability) | off |

#### MCP

| Option | Description | Default |
|--------|-------------|---------|
| `--mcp-config` | Path to MCP configuration file (JSON/YAML) for tool integration | None |

#### Deprecated (no-op) flags

These flags once controlled engine paths that have since been removed. They are
**accepted-but-ignored** for backward compatibility — an old launch script that
still passes them keeps booting instead of failing with `unrecognized
arguments` — but they do nothing, are hidden from `--help`, and are slated for
removal in a future release. Drop them from new commands.

| Flag | Replacement |
|------|-------------|
| `--continuous-batching` | none — batching is always on |
| `--simple-engine` | none — `BatchedEngine` is the sole engine |
| `--kv-bits N` | `--kv-cache-quantization --kv-cache-quantization-bits N` (preserves the bit width) |
| `--kv-group-size N` | `--kv-cache-quantization --kv-cache-quantization-group-size N` (preserves the group size) |
| `--draft-model`, `--num-draft-tokens` | `--speculative-config` |
| `--specprefill`, `--specprefill-threshold`, `--specprefill-keep-pct`, `--specprefill-draft-model` | none — prototype removed |
| `--chunked-prefill-tokens` | none — native `prefill_step_size` is used |

Separately, the legacy per-method speculative-decoding flags (`--enable-dflash`,
`--enable-ddtree`, `--enable-mtp`, `--suffix-decoding`, and their companions)
are hidden deprecated aliases that still work — they normalize into the same
config path as `--speculative-config`. Prefer `--speculative-config` in new
commands.

### Examples

```bash
# Default — continuous batching is on by default; short aliases work
rapid-mlx serve qwen3.5-4b-4bit

# A larger general-purpose model (5 GB)
rapid-mlx serve qwen3.5-9b-4bit --port 8000

# Paged KV cache (memory-efficient prefix sharing)
rapid-mlx serve qwen3.5-9b-4bit --use-paged-cache --port 8000

# With MCP tools
rapid-mlx serve qwen3.5-9b-4bit --mcp-config mcp.json

# Multimodal (vision) model — requires the [vision] extra
rapid-mlx serve gemma-4-26b-4bit --mllm

# Reasoning model — parser is auto-detected, but you can pin it
rapid-mlx serve qwen3.5-9b-4bit --reasoning-parser qwen3

# DeepSeek reasoning model
rapid-mlx serve deepseek-r1-8b-4bit --reasoning-parser deepseek_r1

# Tool calling with Mistral/Devstral (parser auto-detected; pin shown for clarity)
rapid-mlx serve devstral-24b-4bit --enable-auto-tool-choice --tool-call-parser mistral

# DFlash speculative decoding (single-user, single supported alias).
# Requires rapid-mlx[dflash]; OpenAI tools and opt-in thinking are supported.
rapid-mlx serve qwen3.5-27b-8bit --speculative-config '{"method":"dflash"}' --port 8000

# DDTree speculative decoding (experimental, single-user)
rapid-mlx serve qwen3.5-9b-8bit --speculative-config '{"method":"ddtree"}' --port 8000

# DeepSeek V4 Flash checkpoint-native DSpark (block size is checkpoint-defined)
rapid-mlx serve /path/to/DeepSeek-V4-Flash-0731-MLX \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' --port 8000

# MTP fixed-K parity bench mode
rapid-mlx serve <mtp-eligible-qwen-checkpoint> \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"disable_auto_k":true}'

# MTP with a sidecar head: serve a FULL base checkpoint and pass the head
# repo in the `model` field. The `*-mtp-4bit` aliases are sidecar HEADS
# (~246 MB, model_type qwen3_5_mtp) — do NOT serve them directly.
# See docs/reference/configuration.md#mtp-sidecar-heads-are-not-standalone-models
rapid-mlx serve qwen3.6-27b-8bit \
  --speculative-config '{"method":"mtp","model":"mlx-community/Qwen3.6-27B-MTP-4bit","num_speculative_tokens":3}'

# SuffixDecoding for explicit high-overlap workloads
rapid-mlx serve gemma-4-12b-4bit \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8}'

# API key authentication
rapid-mlx serve qwen3.5-9b-4bit --api-key your-secret-key

# Production setup with security options
rapid-mlx serve qwen3.5-9b-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120

# Audio models (requires the [audio] extra) — see docs/guides/audio.md
rapid-mlx serve kokoro                    # TTS via /v1/audio/speech
rapid-mlx serve whisper-large-v3          # STT via /v1/audio/transcriptions
rapid-mlx serve parakeet                  # English STT (NVIDIA Parakeet)
rapid-mlx serve mlx-community/Kokoro-82M-bf16   # Full HF id also routes to audio
```

#### Audio aliases (R10-C1)

Pass any of the audio aliases listed in `rapid-mlx models` (the "Audio models" section) to serve the audio-only `/v1/audio/*` endpoints. The audio path skips the text-LM loader entirely — engines load lazily on the first request. See the [audio guide](../guides/audio.md) for the full TTS / STT alias matrix and quickstart examples.

### Security

When `--api-key` is set, protected API routes require the
`Authorization: Bearer <api-key>` header. Anthropic-compatible routes
(`/v1/messages` and `/v1/messages/count_tokens`) also accept
`x-api-key: <api-key>` for SDK compatibility; if both headers are sent, both
must match.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

Or with curl:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `rapid-mlx bench`

Run a built-in performance benchmark against a model.

### Usage

```bash
rapid-mlx bench <model> [options]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `<model>` | Model alias (e.g. `qwen3.5-4b-4bit`) or HF repo (positional) | *(required)* |
| `--num-prompts` | Number of prompts | 10 |
| `--max-tokens` | Max tokens per prompt | 100 |
| `--enable-prefix-cache` / `--disable-prefix-cache` | Toggle prefix caching | enabled |
| `--use-paged-cache` | Use paged KV cache layout | off |
| `--kv-cache-quantization` | Quantize prefix cache entries | off |

Run `rapid-mlx bench --help` for the full list (memory limits, batch sizes, etc.).

### Examples

```bash
# Quick LLM benchmark using a short alias
rapid-mlx bench qwen3.5-4b-4bit

# Bench a vision-language model by full HF repo
rapid-mlx bench mlx-community/Qwen3-VL-8B-Instruct-4bit
```

## `rapid-mlx chat`

Spawn (or attach to) a server and start an interactive REPL with a model. This
is a terminal chat — not a web UI. (For the Gradio web UI, install the optional
`[chat]` extra: `pip install 'rapid-mlx[chat]'`.)

### Usage

```bash
rapid-mlx chat [model] [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `model` | Model alias or HF repo (positional, optional) | `qwen3.5-4b-4bit` |
| `--system` | System prompt prepended to the conversation | *(none)* |
| `--think` / `--no-think` | Enable / disable reasoning output in the REPL | off |
| `--max-tokens` | Max tokens per assistant response | 2048 (raised to 4096 when `--think` is set, so reasoning + answer fit the budget; an explicit `--max-tokens` always wins) |
| `--temperature` | Sampling temperature | 0.7 |
| `--port` | Connect to an existing server on `127.0.0.1:<port>` instead of spawning | *(spawn)* |
| `--base-url` | Connect to an existing server URL (overrides `--port`) | *(spawn)* |
| `--ready-timeout` | Seconds to wait for the spawned server to become ready | 600 |
| `--response-timeout` | Seconds to wait for a single response | 600 |
| `--mcp-config` | Load MCP tools into this chat agent | *(none)* |
| `--mcp-max-rounds` | Maximum tool-call rounds per turn when `--mcp-config` is set; multi-step tasks may need more | 8 |
| `--disable-prefix-cache` | Disable reusable on-disk prefix caching for a server spawned by `chat` | off |

> The REPL defaults to `--no-think` because reasoning models (Qwen3.5, etc.)
> otherwise leak raw chain-of-thought and can loop until `max-tokens`. Pass
> `--think` to surface reasoning.

### Examples

```bash
# Fastest path — defaults to qwen3.5-4b-4bit, spawns its own server
rapid-mlx chat

# A reasoning model with thinking surfaced
rapid-mlx chat qwen3.5-9b-4bit --think

# Attach to a server you're already running on :8000
rapid-mlx serve qwen3.5-27b-4bit --port 8000 &
rapid-mlx chat --port 8000

# Pin a system prompt
rapid-mlx chat qwen3.5-4b-4bit --system "You are a terse, friendly Mac shell tutor."

# Give the built-in chat agent tools from one or more MCP servers
rapid-mlx chat qwen3.5-4b-4bit --mcp-config mcp.json

# Keep sensitive prompts out of the reusable on-disk prefix cache
rapid-mlx chat qwen3.5-4b-4bit --disable-prefix-cache
```

`--disable-prefix-cache` applies only when `chat` spawns its own server. When
using `--port` or `--base-url`, start that server with the corresponding flag.

When attaching with `--port` or `--base-url` and no model argument, `chat`
discovers the model from the server's `/v1/models` response. An explicit model
argument always wins.

In-REPL slash commands: `/help` (alias `/?`), `/reset` (alias `/clear`),
`/model <alias>`, `/save <path>` (write conversation to markdown), `/exit`
(aliases `/quit`, `/bye`).
Type `"""` on its own line to start/end a multi-line block (pasting code).

MCP belongs to the chat process: it discovers and executes the configured
tools, while the spawned or remote Rapid-MLX server receives only standard
OpenAI function tools and tool-result messages. `serve` and `share` do not
need the chat's MCP configuration.

## Environment Variables

Operator-facing `RAPID_MLX_*` variables read by the server and CLI. A CLI
flag always wins over its env-var fallback when both are set.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAPID_MLX_API_KEY` | unset (no auth) | Bearer API key fallback for `serve --api-key`; the inline flag value wins. `rapid-mlx share` uses the env form so the key never lands in `argv`. |
| `RAPID_MLX_TRUSTED_HOSTS` | unset (not enforced) | Comma-separated Host-header allowlist fallback for `--trusted-hosts`; non-matching requests get HTTP 400 |
| `RAPID_MLX_CORS_ALLOW_ORIGINS` | unset (wildcard `*`) | Comma-separated CORS origin allowlist; `--cors-origins` wins when both are set. Unset = friendly wildcard default with a startup notice; set-but-empty parses fail closed (no CORS middleware). |
| `RAPID_MLX_CORS_ALLOW_METHODS` / `_HEADERS` / `_MAX_AGE` / `_CREDENTIALS` | `POST,GET,OPTIONS` / `Content-Type,Authorization,X-Rapid-MLX-Internal` / `3600` / off | Fine-tune the CORS policy when origins come from the env var (the CLI `--cors-origins` path keeps legacy wide-open methods/headers). Credentials are always forced off with a wildcard origin, per the Fetch spec. |
| `RAPID_MLX_MAX_REQUEST_BYTES` | 8388608 (8 MiB) | Request-body size cap fallback for `--max-request-bytes`; oversized bodies get HTTP 413 before parsing. 0 disables. |
| `RAPID_MLX_SSE_KEEPALIVE_SECONDS` | 20 | Interval for SSE keepalive comment lines during silent prefill (defeats proxy idle timeouts). 0 disables the heartbeat. |
| `RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS` | 15 | Max idle seconds between request-body chunks (slowloris defense); exceeded connections get HTTP 408. 0 disables. |
| `RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS` | 0 (disabled) | Fallback for `--idle-cache-clear-seconds`: clear reusable KV state after this many idle seconds, keeping model weights loaded. An explicit CLI value (including 0) wins. |
| `RAPID_MLX_WATCHDOG_PPID` | unset (disabled) | Fallback for `--watchdog-ppid`: self-terminate when the parent with this PID dies |
| `RAPID_MLX_TELEMETRY` | unset | Telemetry kill switch: `0` / `false` / `no` / `off` / empty force-disables telemetry regardless of stored consent. Truthy values do NOT force-enable (consent is interactive-only). |
| `RAPID_MLX_KV_CHECKPOINT_MAX_BYTES` | 21474836480 (20 GiB) | Disk cap for `~/.cache/rapid-mlx/kv_checkpoints/` when `--kv-disk-checkpoint-interval` is enabled; oldest files evicted first. Read at scan time, so it can change without a restart. |
| `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` | unset (heuristic) | Hard byte cap on prefix-cache memory (positive integer). Unset, blank, or invalid values fall back to the heuristic limit (logged once). |
| `RAPID_MLX_MAX_GENERATION_TOKENS` | unset (no ceiling) | Opt-in hard ceiling on per-request `max_tokens`; requests above it are rejected at validation. Invalid or non-positive values are treated as unset. Read per request. |
| `RAPID_MLX_STRICT_JSON_SCHEMA` | enabled | Strict post-generate `json_schema` enforcement; set `0`/`off`/`false`/`no`/`disable`/`disabled` to fall back to legacy prompt-injection-only behavior |
| `RAPID_MLX_CONSTRAIN_TOOLS` | on (`1`) | Grammar-constrained tool calling; see the [configuration reference](configuration.md#environment-variables) for the full contract |
| `RAPID_MLX_MCP_CONFIG` | unset | Path to the MCP config file; `--mcp-config` sets it for the server process, and it is honored at boot when the flag is absent |
| `RAPID_MLX_AUTO_PULL` | unset | Set `1`/`true`/`yes` to auto-confirm model downloads (skips the interactive size prompt) |
| `RAPID_MLX_MODEL_MIRROR` | `https://models.rapidmlx.com` | Model download mirror base URL; set to an empty string to force downloads from Hugging Face |
| `RAPID_MLX_EXTRA_MODEL_ROOTS` | unset | Extra local directories to resolve models from, separated by `os.pathsep` (`:` on macOS/Linux), or a JSON array of paths |
| `RAPID_MLX_DEFAULT_MODEL` | `qwen3.5-4b-4bit` | Default model alias used by `rapid-mlx launch` when `--model` is not given |
| `RAPID_MLX_DISABLE_VERSION_CHECK` | unset | Set to any non-empty value to skip the interactive new-version check |
| `RAPID_MLX_TRUST_REMOTE_CODE` | unset | Set `0`/`false`/`no`/`off` to force `trust_remote_code=False` process-wide for tokenizer loading |
| `VLLM_MLX_TEST_MODEL` | unset | Model for tests |
| `HF_TOKEN` | unset | HuggingFace token for gated/private repos |
