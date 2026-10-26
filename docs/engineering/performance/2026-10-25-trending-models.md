# October 25 trending-model qualification

Environment: Leased Apple M4 Pro (48 GiB unified memory), macOS 26.5.1, Python 3.12.15; fresh virtual environment from this branch
with `[vision]` (`mlx` 0.32.4, `mlx-lm` 0.31.3, `mlx-vlm` 0.7.2). A separate
temporary `HOME` and Hugging Face cache were used. Model files came through
the Machine Control model store at immutable revisions; the service bound only
to `127.0.0.1`. No benchmark inference ran on the laptop. The worker's model
copies, cache, environment, and temporary credential were removed before the
lease was released.

The qualification command for each catalog entry was:

```sh
rapid-mlx serve ALIAS --host 127.0.0.1 --port 18761 --max-tokens 256
```

The environment also set `RAPID_MLX_TELEMETRY=0`, `HF_HUB_OFFLINE=1`,
`RAPID_MLX_DISABLE_MODEL_DOWNLOADS=1`, and temporary `HOME`/`HF_HUB_CACHE`.
Each model received English and Chinese factual chat requests (Paris/Beijing),
a streamed multi-sentence explanation, a `get_weather(city=Paris)` request
with an OpenAI `tools` array, and 17×23 with template thinking both on and off.
Temperature was zero. Successful tools required a structured `tool_calls`
entry with JSON arguments and `finish_reason=tool_calls`; raw tool text did not
count. Streaming required multiple content deltas where the runtime supplied
them, or a valid SSE completion for a coalescing backend. TTFT and post-first-
token decode rate came from the server's `/metrics` request ledger for the
streamed prompt. Peak memory is the MLX Metal allocator peak from `/metrics` after the
requests, not total system memory. This shared host also transferred other
weights during some runs, so these are qualification measurements, not
comparative throughput claims.

| Model | Result | Pinned MLX repository | Safetensors bytes | Stream TTFT | Decode tok/s | Peak Metal | Tool |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| MiniMax M3.1 Flash | Not runnable on this fleet | `mlx-community/MiniMax-M3.1-Flash-Text-4bit@a7fbd2db43edd596c63f848dc0c4afbfc612d51e` | 248,369,471,329 | — | — | — | Untested |
| Nanbeige4.3 33B | Pass | `pratikgx/Nanbeige4.3-33B-MLX-4bit@883c8bda53b2062ff7e37f178ccf503c79674797` | 18,630,644,776 | 0.539 s | 13.42 | 19.28 GB | Pass |
| Spark-X2.6 4B | Pass | `abenzerps/Spark-X2.6-4B-MLX-4bit@253181296bd0c643880b0f64daa0049a42148d4a` | 2,315,507,460 | 0.126 s | 89.44 | 2.83 GB | Pass |
| Sage-A1.5 | Pass, image | `mlx-community/Sage-A1.5-OptiQ-4bit@efad5741d354e38b3429433c5144331f8fb86cce` | 23,516,214,400 | 0.129 s | 71.89 | 24.13 GB | Pass |

Sage answered `red` for a 128×128 solid-red PNG in an OpenAI image-content
request. Its thinking-on arithmetic used `max_tokens=1024` to reach a final
answer; the first 256-token attempt truncated inside reasoning. The VLM
response stream for Sage and Spark was coalesced into one content delta plus
usage/end events; the server ledger supplied
first-token and decode timing independently.

The alternative MiniMax `oQ4e-mtp` checkpoint is 248,771,603,692
safetensors bytes. Both are close to the largest 256 GiB machine's physical
memory before runtime and KV overhead; neither fits the 48/36 GiB test boxes.
A responsible alias floor would be at least 320 GiB, so no MiniMax alias was
added without a compatible qualification host.

No model was added to recommendation tiers from this small smoke set. The
mirror catalog is unchanged; normal on-demand mirror autopilot handles any
new repo after catalog publication, with no direct object-store write here.

## October 26 landing verification

Repeated the real CLI journey on an Apple M3 Ultra (256 GiB), macOS 26.5.2,
with a fresh Python 3.12.13 `[vision]` environment and a fresh HOME per alias.
Dependencies were mlx 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2 and transformers
5.15.1. The existing standard Hugging Face cache contained all three pinned
snapshots; no model downloads or cache removals were needed. Each server ran
under the shared GPU lock on loopback port 8745, with disk caches disabled.

For each of `spark-x2.6-4b-4bit`, `sage-a1.5-4bit`, and
`nanbeige4.3-33b-4bit`, plain `rapid-mlx serve ALIAS` resolved and loaded the
pinned model. HTTP chat returned Paris, Chinese SSE returned the correct
capital, and a weather request returned structured `get_weather` arguments
with `city=Paris` and `finish_reason=tool_calls`. Template thinking off omitted
reasoning; thinking on returned a separate reasoning field and a final answer.
Sage identified a 224×224 solid-red PNG as `red`. No visible content contained
reasoning or tool-wire delimiters. Spark and Sage coalesced their short SSE
answers; Nanbeige emitted multiple content deltas. These are functional smoke
checks, not latency or sustained-load benchmarks.

### Supply-chain review

All three catalog revisions were checked against Hub metadata at their exact
40-character SHAs. Weight-plus-tokenizer footprints are 2,330,273,898 bytes
(Spark), 23,536,584,172 bytes (Sage), and 18,649,200,689 bytes (Nanbeige), exactly
matching the size manifest. The 16/48/32 GiB memory floors exceed the earlier
measured 2.83/24.13/19.28 GB allocator peaks. All uploads declare Apache-2.0.
The two personal uploads receive the same immutable-pin and code-policy checks.

Code reading followed the serving paths into the installed dependencies:

- `MLXMultimodalLM.load` resolves a pinned snapshot, applies the default-false
  remote-code policy, rejects `model_file` without explicit consent, and passes
  `trust_remote_code=False` to mlx-vlm. None of these snapshots declares
  `model_file`. Spark's `auto_map` names repository Python, but its tokenizer
  uses installed `TokenizersBackend`; its model loads through the installed
  `spark2_5` registry. Sage uses installed `qwen3_5_moe` and `Qwen3VLProcessor`.
- mlx-vlm 0.7.2 forwards the trust flag to `AutoProcessor`. Its no-kwargs
  `load_image_processor` fallback passes true only to `load_config`, which
  reads JSON; the serving path supplies kwargs and does not take that fallback.
  Its detokenizer selection reads tokenizer JSON without constructing a remote
  tokenizer. Transformers gates dynamic processor imports on remote-code trust.
- Nanbeige uses the installed mlx-lm Llama model and tokenizer through
  `load_mlx_lm_checked`, with a nonempty, false-trust tokenizer configuration.
  The separate distributed-loader true default is not this serving path.
  These text/image aliases do not invoke an audio loader.
- An audit hook installed before CLI imports recorded `exec`, `compile`, and
  import events during startup and all HTTP probes. It checked snapshot paths,
  the Hub cache (including blobs), and dynamic Transformers modules. All three
  runs recorded zero repository-code events and zero repository modules in
  the shutdown `sys.modules` inventory. Model module filenames pointed into
  the installed dependencies. The main server processes observed thousands
  of execution events, confirming the hook was active throughout loading.

The OptiQ exception is limited to the exact root sidecar name, a readable VLM
config, and vision-only tensor mappings. Existing containment, nonempty-file,
primary-root-shard, and complete numbered-shard checks remain in force. Tests
reject missing sidecars, language tensors assigned to the sidecar, unknown
sidecars, and partial checkpoints; stale-index fallback is retained only when
none of its indexed shards is present. No parser code changed.

Focused download-gate and catalog tests: 230 passed. Changed-line coverage:
16/16 executable lines (100%).
