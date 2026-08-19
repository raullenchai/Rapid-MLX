# Configuration Reference

## Server Configuration

The tables below cover the consequential `rapid-mlx serve` options by
category. The exhaustive flag list (every flag visible in
`rapid-mlx serve --help`) lives in the [CLI reference](cli.md#rapid-mlx-serve).

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host address (loopback-only by default; pass `0.0.0.0` to expose on LAN) | `127.0.0.1` |
| `--port` | Server port | `8000` |
| `--listen-fd` | File descriptor of a pre-bound listening socket (3-1023) for socket activation; when set, `--host`/`--port` are ignored for binding | None |
| `--log-level` | Log level for Python logging and uvicorn (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `--served-model-name` | Model name reported by the API; when unset the `model` argument is used | None |
| `--max-tokens` | Default max tokens | `32768` |
| `--default-temperature` | Default temperature when not specified in request | None |
| `--default-top-p` | Default top_p when not specified in request | None |
| `--default-top-k` | Default top_k when not specified in request | None |
| `--default-min-p` | Default min_p when not specified in request | None |
| `--default-repetition-penalty` | Default repetition_penalty when not specified in request | None |
| `--default-presence-penalty` | Default presence_penalty when not specified in request | None |
| `--default-frequency-penalty` | Default frequency_penalty when not specified in request | None |

### Security Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | API key for authentication (falls back to `RAPID_MLX_API_KEY`) | None |
| `--cors-origins` | Allowed CORS origins (space-separated; also via `RAPID_MLX_CORS_ALLOW_ORIGINS`) | `*` (all origins) |
| `--trusted-hosts` | Opt-in Host-header allowlist; non-matching requests get HTTP 400 (also via `RAPID_MLX_TRUSTED_HOSTS`) | None (not enforced) |
| `--rate-limit` | Requests per minute per client (0 = disabled) | `0` |
| `--max-request-bytes` | Max HTTP request body size in bytes; oversized requests get HTTP 413 before parsing. 0 disables. (also via `RAPID_MLX_MAX_REQUEST_BYTES`) | 8 MiB (8388608) |
| `--timeout` | Request timeout in seconds | `1800` |

### Admission and Batching Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stream-interval` | Tokens per stream chunk | `1` |
| `--max-num-seqs` | Max concurrent sequences | `256` |
| `--max-concurrent-requests` | Admission cap on in-flight requests (queued + running); excess requests get HTTP 503 with `Retry-After` | `256` |
| `--prefill-batch-size` | Max prompts prefilled together in one cold wave; lower for better first-token latency under concurrent cold load | `8` |
| `--completion-batch-size` | Completion batch size | `32` |
| `--prefill-step-size` | Chunk size for prompt prefill processing | `2048` |
| `--gpu-memory-utilization` | Fraction of device memory for the Metal allocation limit (0.0-1.0) | `0.90` |

### Cache Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-prefix-cache` / `--disable-prefix-cache` | Toggle prefix caching for repeated prompts | enabled |
| `--prefix-cache-index` | Prefix-cache lookup index: `radix` (token trie) or `hash` (legacy bisect) | `radix` |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | `0.20` |
| `--idle-cache-clear-seconds` | Clear reusable KV cache after idle time; model weights remain loaded | Disabled |
| `--no-memory-aware-cache` | Use legacy entry-count cache | `false` |
| `--pin-system-prompt` | Auto-pin the system prompt in the prefix cache to prevent eviction under memory pressure | `false` |
| `--use-paged-cache` | Enable paged KV cache | `false` |
| `--paged-cache-block-size` | Tokens per block | `64` |
| `--max-cache-blocks` | Maximum blocks | `1000` |
| `--hybrid-cache-entries` | Opt-in: retain N non-trimmable prefix-cache entries for prefix-extension reuse (stable prefix + new suffix each turn). Covers hybrid recurrent-state (GatedDeltaNet/Mamba) and sliding-window (Gemma 4, GPT-OSS) models. `0` disables. | `0` |
| `--response-cache-entries` | Opt-in: retain N fully-computed greedy (`temperature 0` / `top_k 1`) chat completions; a completely repeated request returns the stored completion with zero GPU decode. Sampled requests are never cached. `0` disables. | `0` |

### KV Cache Quantization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--kv-cache-dtype` | KV cache dtype (`bf16`, `int8`, `int4`). int8/int4 shrink the KV cache 2x/4x for memory-constrained hosts at a decode-throughput cost at long context; sliding-window and MLA models auto-downgrade to bf16. | `bf16` |
| `--reasoning` | Pins `--kv-cache-dtype` to int8 (reasoning-accuracy profile) | `false` |
| `--kv-cache-quantization` | Deprecated alias of `--kv-cache-dtype int8`; wins when both are passed | `false` |
| `--kv-cache-quantization-bits` | Bit width for KV cache quantization (4 or 8) | `8` |
| `--kv-cache-quantization-group-size` | Group size for KV cache quantization | `64` |
| `--kv-cache-min-quantize-tokens` | Minimum tokens for quantization to apply | `256` |
| `--kv-cache-turboquant` | TurboQuant KV compression (experimental): bare = `v4` (V-only), `k8v4` (K 8-bit + V 4-bit mix), `none` (explicit off overriding alias auto-resolution). Mutually exclusive with `--kv-cache-quantization`. | None (alias-driven) |
| `--kv-cache-turboquant-bits` | V-side bit width (3 or 4); ignored in `k8v4` mode | Auto by head_dim |
| `--kv-cache-turboquant-group-size` | Group size for TurboQuant V-side quantization | `32` |
| `--kv-disk-checkpoint-interval` | Token interval for KV snapshots to `~/.cache/rapid-mlx/kv_checkpoints/`; write-only, blocks decode per snapshot — external tooling only. `0` disables. | `0` |
| `--metal-cap-kv-bytes-per-token` | Override the projected per-token KV size (bytes) in the admission gate; set when running a quantized KV cache. `0` auto-derives an fp16 figure. | `0` (auto) |

### Model Loading and Residency Options

| Option | Description | Default |
|--------|-------------|---------|
| `--disk-stream` | Stream MoE routed-expert weights from disk instead of holding them resident (opt-in; registered architectures only) | `false` |
| `--disk-stream-cache-gb` | Byte budget (GB) for the disk-stream expert LRU cache | `1.0` |
| `--resident-memory-limit-gb` | Process-wide resident model ceiling in GiB; LRU idle unpinned models are evicted first. `0` disables. | `0` |
| `--resident-model-idle-ttl` | Evict idle unpinned secondary models after this many seconds. `0` disables. | `0` |
| `--mllm` / `--no-mllm` | Force multimodal (vision) loading / force text-only loading, overriding auto-detection | auto-detect |
| `--enable-audio` | Mount `/v1/audio/*` routes on a text-only server (audio-capable models auto-mount them) | `false` |

### PFlash Options

PFlash long-prompt prefill compression. Defaults to `always` for verified
aliases (Qwen3.5 / Qwen3.6 family) and `off` otherwise; tune with
`--pflash off|auto|always`, `--pflash-threshold` (32768), and the
`--pflash-*` keep/scoring knobs — see the
[CLI reference](cli.md#pflash-long-prompt-compression) for the full table.

### Tool Calling Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | Enable automatic tool calling | `false` |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](../guides/tool-calling.md)) | None |

### Reasoning Options

| Option | Description | Default |
|--------|-------------|---------|
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`, `deepseek_r1_distill`, `deepseek_v4`, `gemma4`, `glm4`, `gpt_oss`, `harmony`, `hy3`/`hy_v3`, `minimax`, `muse`, `ui_tars`, `vibethinker`); auto-detected from the alias profile when omitted | None (auto-detected) |

### Embedding Options

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-model` | Pre-load an embedding model at startup (requires `pip install 'rapid-mlx[embeddings]'`) | None |
| `--embedding-max-length` | Max input length (tokens); `auto` derives it from the model's declared maximum, or pass a positive integer for a lower ceiling | `auto` |
| `--embedding-overflow-policy` | Handling for over-length inputs: `truncate` (logged + metric, never silent) or `error` (HTTP 400 with observed/allowed counts) | `truncate` |

### Speculative Decoding Options

Use `--speculative-config` for speculative decoding usage. Legacy
spec-decoder flags are hidden deprecated compatibility aliases that normalize
into the same config path.

| Config | Description |
|--------|-------------|
| `{"method":"dflash"}` | Enable the DFlash single-user bridge on validated aliases. |
| `{"method":"ddtree"}` | Enable experimental DDTree verification on validated aliases. |
| `{"method":"dspark","num_speculative_tokens":5}` | Enable checkpoint-native DSpark for a local DeepSeek V4 Flash checkpoint. The token count must match the checkpoint's complete DSpark block. Greedy single-request decoding is accelerated; unsupported request shapes safely use baseline decoding. |
| `{"method":"mtp"}` | Enable MTP speculative decoding for checkpoints accepted by the existing MTP eligibility gate. |
| `{"method":"mtp","model":"<sidecar-head-repo>"}` | Attach a standalone MTP **sidecar head** (e.g. `mlx-community/Qwen3.6-27B-MTP-4bit`) to a full base checkpoint. The base must be MTP-eligible; the head repo goes in the `model` field — **not** in the `serve` positional. See [MTP sidecar heads are not standalone models](#mtp-sidecar-heads-are-not-standalone-models) below. Gemma 4 sidecar MTP remains disabled after its greedy-lossless A/B failed. |
| `{"method":"mtp","num_speculative_tokens":3}` | Set the MTP max-K controller ceiling. |
| `{"method":"mtp","disable_auto_k":true}` | Disable the MTP EV depth controller for fixed-K parity benches. |
| `{"method":"suffix","num_speculative_tokens":8}` | Enable explicit SuffixDecoding for high-overlap workloads. |

#### MTP is not free — measure before you enable it

MTP is **opt-in and should stay opt-in**. A high draft acceptance rate is
not sufficient for it to pay, and on at least one otherwise-recommended
model it is a large net loss.

The speedup ceiling for chain-of-K MTP is

```
speedup <= (1 + K * accept) / cost_ratio(K + 1)
```

where `cost_ratio(N)` is what an `N`-position forward costs relative to a
1-position decode forward on **your** hardware. The numerator is what
speculation wins; the denominator is what verifying costs. Acceptance is
a property of the drafter, `cost_ratio` is a property of the chip and the
architecture, and only their ratio decides the outcome. Neither number
transfers between machines: the same model and byte-identical acceptance
measured +118% on one host and +13% on another.

Worked example — `qwen3.6-35b-4bit` with `Qwen3.6-35B-A3B-MTP-4bit`,
Mac mini M2 Pro / 32GB, `temperature=0`, medians over 4 repetitions:

| | |
|---|---|
| draft acceptance | **0.857** — the drafter is excellent |
| `cost_ratio(2)` | **1.70** — a 2-position forward costs 1.70x a decode forward |
| predicted ceiling | `1.857 / 1.70` = **1.09x** |
| measured, MTP on, fixed K=1 | 47.4 tok/s |
| measured, MTP on, auto-K | 47.6 tok/s — the controller cannot rescue it |
| measured, MTP off | 59.0 tok/s — **MTP is 20% slower** |

(Acceptance and the fixed-K throughput are from the same run, so they
describe the same work; the auto-K row is a separate run of the same
protocol.)

An 86% acceptance rate buys a 9% ceiling here, and per-round overhead
consumes it. The architecture is why: this is a linear-attention hybrid,
and its forward does not amortize a second position the way a
pure-attention model's does. Enabling MTP on it costs a fifth of your
throughput.

To check your own combination, pin the depth and run a representative
workload:

```bash
rapid-mlx serve <your-model> \
  --speculative-config '{"method":"mtp","model":"<head>","disable_auto_k":true}'
```

then read off `/metrics`:

```
rapid_mlx_spec_decode_k_cost_ms{k="0",model_id="..."}   # park round, no drafter
rapid_mlx_spec_decode_k_cost_ms{k="1",model_id="..."}   # drafting + verify round
rapid_mlx_spec_decode_accept_ratio                      # acceptance
```

`disable_auto_k` is what makes the acceptance term meaningful:
`accept_ratio` is a single number pooled over every depth the controller
sampled, and deeper positions accept less often than shallow ones, so
under auto-K it belongs to no particular K. Pinning the depth makes it
belong to the K you are evaluating. The controller keeps measuring in
this mode — it just stops choosing.

The `k="0"` sample in fixed-K mode comes from the cold-start round each
request opens with, so send enough requests (a dozen is plenty) for it to
settle before reading the ratio.

Each `k_cost_ms` bucket is a whole round — the target forward plus the
drafting that produced the drafts that round consumed — so the two are
directly comparable. A K=1 round emits `1 + accept` tokens on average
against a park round's 1, which makes the decision rule:

```
enable MTP only if   (1 + K * accept)  >  k_cost_ms{k=K} / k_cost_ms{k=0}
```

The left side is what depth K wins you; the right side is what it costs.
On `qwen3.6-35b-4bit` the left side is 1.857 and the right side is larger,
which is the whole story. Note that both buckets carry MTP's own
per-round overhead, so clearing this bar is necessary but not sufficient
— a marginal result should be confirmed against a real MTP-off A/B before
you turn it on.

> **Run this on a single model per process.** The two sides of the rule
> come from series with *different identity scopes*: `k_cost_ms` is keyed
> per checkpoint (`model_id`, from the per-model controller registry that
> survives a model swap), while `accept_ratio` is a **process-global**
> counter — it pools acceptance across every model *and* every depth the
> process ever ran (the `family` label only splits a dashboard panel; it
> is not a separate counter). So after a model swap, or with concurrent
> models in one process, the cost curve and the acceptance term describe
> different model/head combinations and the rule can read wrong. The
> diagnostic above already assumes one model and `disable_auto_k`; keep it
> that way — a fresh process per checkpoint — when you read the ratio.

#### MTP sidecar heads are not standalone models

The `*-mtp-4bit` aliases — `qwen3.6-27b-mtp-4bit`
(`mlx-community/Qwen3.6-27B-MTP-4bit`) and `qwen3.6-35b-mtp-4bit`
(`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`) — resolve to **MTP sidecar
heads**, not servable checkpoints. Each repo (~246 MB) ships only the
multi-token-prediction module — an `fc.*` fusion projection, a single
`layers.0.*` predictor layer, the `pre_fc_norm_embedding` /
`pre_fc_norm_hidden` norms, and a final `norm` — with no full transformer
to generate from. Their `config.json` `model_type` is `qwen3_5_mtp`,
which is intentionally **not** in the MTP eligibility allowlist.

Serving a head directly is rejected by design — the alias name contains
`mtp`, but the repo is a draft head, not a model:

```bash
# Rejected: qwen3.6-27b-mtp-4bit is a sidecar head, not a servable checkpoint
rapid-mlx serve qwen3.6-27b-mtp-4bit --speculative-config '{"method":"mtp"}'
```

Instead, serve a **full base checkpoint** and pass the head repo in the
`model` field of `--speculative-config`, so MTP drafts against the
attached head:

```bash
# Correct: full base checkpoint + head repo in the `model` field
rapid-mlx serve qwen3.6-27b-8bit \
  --speculative-config '{"method":"mtp","model":"mlx-community/Qwen3.6-27B-MTP-4bit","num_speculative_tokens":3}'
```

Pair each head with a base of the same size class: `Qwen3.6-27B-MTP-4bit`
with a `qwen3.6-27b-*` base, and `Qwen3.6-35B-A3B-MTP-4bit` with a
`qwen3.6-35b-*` base.

Sidecar/base precision pairing effects are model-dependent: benchmark your
pairing. On Qwen3.6
([#1258](https://github.com/raullenchai/Rapid-MLX/issues/1258)), matched 4/4
improved throughput by 14% while mixed 8/4 reduced it by 8% versus no
speculation. On Qwen3.8-27B (M5 Max measurements in
[#1216](https://github.com/raullenchai/Rapid-MLX/pull/1216)) the economics
invert: mixed 8/4 gained 35-65% while matched 4/4 was roughly break-even,
because an expensive base leaves more room for a cheap drafter.

### MCP Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mcp-config` | Path to MCP config file | None |

## MCP Configuration

Create `mcp.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### MCP Server Options

| Field | Description | Required |
|-------|-------------|----------|
| `command` | Executable command | Yes |
| `args` | Command arguments | Yes |
| `env` | Environment variables | No |

## API Request Options

### Chat Completions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model name | Required |
| `messages` | Chat messages | Required |
| `max_tokens` | Max tokens to generate | None → server default (32768, `--max-tokens`) |
| `temperature` | Sampling temperature | Model default |
| `top_p` | Nucleus sampling | Model default |
| `stream` | Enable streaming | `false` |
| `stop` | Stop sequences | None |
| `tools` | Tool definitions | None |
| `response_format` | Output format (`json_object`, `json_schema`) | None |

### Multimodal Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_fps` | Frames per second | 2.0 |
| `video_max_frames` | Max frames | 128 |

## Environment Variables

### Server-side variables

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
| `RAPID_MLX_CONSTRAIN_TOOLS` | on (`1`) | Grammar-constrained tool calling (best-effort). Set to `0`/`off`/`false` to opt out. When enabled AND a `--tool-call-parser` is set AND a request sends `tools` with `tool_choice="required"` or a named function, the server compiles a grammar that constrains generation so a completed tool call names a real tool with schema-valid arguments in the family wire format. Requests without tools, or with `tool_choice="auto"`/`"none"`, are always unaffected. **Best-effort fallback:** the request silently falls back to the free-form-then-parse path (no hard error, no structural guarantee) when the `[guided]` extra (`llguidance`) is not installed, the model's tokenizer cannot back an `LLTokenizer`, the grammar fails to compile, or the parser family declares no structural info. `parallel_tool_calls=false` narrows the grammar to exactly one call. Note: the structural guarantee holds only for a call the model runs to a grammar-accepted completion — a `max_tokens` cutoff mid-call can still truncate the arguments and yield invalid JSON. |
| `RAPID_MLX_MCP_CONFIG` | unset | Path to the MCP config file; `--mcp-config` sets it for the server process, and it is honored at boot when the flag is absent |
| `RAPID_MLX_AUTO_PULL` | unset | Set `1`/`true`/`yes` to auto-confirm model downloads (skips the interactive size prompt) |
| `RAPID_MLX_MODEL_MIRROR` | `https://models.rapidmlx.com` | Model download mirror base URL; set to an empty string to force downloads from Hugging Face |
| `RAPID_MLX_EXTRA_MODEL_ROOTS` | unset | Extra local directories to resolve models from, separated by `os.pathsep` (`:` on macOS/Linux), or a JSON array of paths |
| `RAPID_MLX_DEFAULT_MODEL` | `qwen3.5-4b-4bit` | Default model alias used by `rapid-mlx launch` when `--model` is not given |
| `RAPID_MLX_DISABLE_VERSION_CHECK` | unset | Set to any non-empty value to skip the interactive new-version check |
| `RAPID_MLX_TRUST_REMOTE_CODE` | unset | Set `0`/`false`/`no`/`off` to force `trust_remote_code=False` process-wide for tokenizer loading |
| `VLLM_MLX_TEST_MODEL` | unset | Default model for tests |
| `HF_TOKEN` | unset | HuggingFace authentication token |

### Client-side (SDK) variables

These are read by client SDKs, not by the rapid-mlx server:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Read by the OpenAI SDK, not the server. Set to any value when the server runs without `--api-key`; must match the server key when one is set. |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` | Read by Anthropic SDK clients (e.g. Claude Code) to point at the rapid-mlx `/v1/messages` endpoint |

## Example Configurations

### Development (Single User)

```bash
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Production (Multiple Users)

```bash
rapid-mlx serve qwen3.5-27b-4bit \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --port 8000
```

### With Tool Calling

```bash
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

### With MCP Tools

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --enable-auto-tool-choice \
  --tool-call-parser qwen
```

### Reasoning Model

```bash
rapid-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3
```

### With Embeddings

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --embedding-model mlx-community/multilingual-e5-small-mlx
```

### High Throughput

```bash
rapid-mlx serve qwen3.5-27b-4bit \
  --stream-interval 5 \
  --max-num-seqs 256
```
