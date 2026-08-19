# Configuration Reference

## Server Configuration

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host address (loopback-only by default; pass `0.0.0.0` to expose on LAN) | `127.0.0.1` |
| `--port` | Server port | `8000` |
| `--max-tokens` | Default max tokens | `32768` |
| `--default-temperature` | Default temperature when not specified in request | None |
| `--default-top-p` | Default top_p when not specified in request | None |

### Security Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | `0` |
| `--timeout` | Request timeout in seconds | `1800` |

### Batching Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stream-interval` | Tokens per stream chunk | `1` |
| `--max-num-seqs` | Max concurrent sequences | `256` |

### Cache Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | `0.20` |
| `--idle-cache-clear-seconds` | Clear reusable KV cache after idle time; model weights remain loaded | Disabled |
| `--no-memory-aware-cache` | Use legacy entry-count cache | `false` |
| `--use-paged-cache` | Enable paged KV cache | `false` |
| `--paged-cache-block-size` | Tokens per block | `64` |
| `--max-cache-blocks` | Maximum blocks | `1000` |
| `--hybrid-cache-entries` | Opt-in: retain N non-trimmable prefix-cache entries for prefix-extension reuse (stable prefix + new suffix each turn). Covers hybrid recurrent-state (GatedDeltaNet/Mamba) and sliding-window (Gemma 4, GPT-OSS) models. `0` disables. | `0` |
| `--response-cache-entries` | Opt-in: retain N fully-computed greedy (`temperature 0` / `top_k 1`) chat completions; a completely repeated request returns the stored completion with zero GPU decode. Sampled requests are never cached. `0` disables. | `0` |

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

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Default model for tests |
| `HF_TOKEN` | HuggingFace authentication token |
| `OPENAI_API_KEY` | Set to any value for SDK compatibility |
| `RAPID_MLX_CONSTRAIN_TOOLS` | Grammar-constrained tool calling (best-effort). **On by default** (`1`); set to `0`/`off`/`false` to opt out. When enabled AND a `--tool-call-parser` is set AND a request sends `tools` with `tool_choice="required"` or a named function, the server compiles a grammar that constrains generation so a completed tool call names a real tool with schema-valid arguments in the family wire format. Requests without tools, or with `tool_choice="auto"`/`"none"`, are always unaffected. **Best-effort fallback:** the request silently falls back to the free-form-then-parse path (no hard error, no structural guarantee) when the `[guided]` extra (`llguidance`) is not installed, the model's tokenizer cannot back an `LLTokenizer`, the grammar fails to compile, or the parser family declares no structural info. `parallel_tool_calls=false` narrows the grammar to exactly one call. Note: the structural guarantee holds only for a call the model runs to a grammar-accepted completion — a `max_tokens` cutoff mid-call can still truncate the arguments and yield invalid JSON. |

## Example Configurations

### Development (Single User)

```bash
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Production (Multiple Users)

```bash
rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
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
rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --stream-interval 5 \
  --max-num-seqs 256
```
