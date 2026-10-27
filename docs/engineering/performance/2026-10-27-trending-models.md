# October 27 trending-model qualification

Owner: Vector. Scope: immutable catalog entries only for checkpoints that pass
native-loader, serving, supply-chain, and correctness qualification. Dependency
pins, recommendation tiers, and version are unchanged.

## Qualification environment and method

Ornith qualification uses a leased Apple M3 Ultra with 256 GiB unified memory,
macOS 26.5.2, and a fresh Python 3.12.14 virtual environment installed from this
branch with `[vision]`. Dependency versions are MLX 0.32.4, mlx-lm 0.31.3,
mlx-vlm 0.7.2, and Transformers 5.15.1. Inference runs under the shared large-model
lock on loopback port 18951, with a temporary HOME, telemetry disabled, and
network model downloads disabled after the pinned snapshot is complete.
The standard Hugging Face cache entry is a symlink to cold USB storage; the
69.6 GB checkpoint never occupies the internal SSD. During model download and serving qualification, internal free
space stayed above 100 GiB. The cached model is retained after qualification.

The serving command is:

```sh
lockf -k /var/tmp/rapid-mlx-large-model.lock \
  rapid-mlx serve ornith-1.5-122b-a10b-4bit \
  --host 127.0.0.1 --port 18951 --max-tokens 1024 --disable-disk-caches
```

Requests use temperature zero: English/Chinese factual chat; a streamed
multi-sentence explanation; structured `get_weather(city=Paris)`; 17 × 23 with
template thinking on/off; and a two-turn code-recall conversation. Tools require
structured OpenAI calls and JSON arguments, with `finish_reason=tool_calls`.
Reasoning must remain separate from visible answer content. Stream qualification
records the number of content deltas, plus the server request ledger's TTFT
and decode throughput. Cold means the first model load and request after the
pinned download, with no existing server or prompt cache; warm means a fresh
server load using the same cached files. No system-wide filesystem cache purge
is performed, so these are cold/warm application-load measurements, not claims
of physically uncached USB throughput.

## Qualified catalog entry

`ornith-ai/Ornith-1.5-122B-MLX-4bit@06a57aba4fbae218efc9716bcdf7d6b3035331c8`
is the publisher's MIT-licensed MLX checkpoint. The download-size manifest uses
69,642,922,462 bytes (69,612,579,840 safetensors bytes), matching the existing weight/tokenizer size estimator;
the entire repository, including model-card artwork, is 69,645,012,324 bytes.
The runtime snapshot download explicitly selects pinned weights, JSON,
tokenizer text, and chat-template data. It contains all fourteen language
shards and the separate `model-vision.safetensors` tower. Image input remains
unqualified and is not advertised by this alias.

Its config declares `qwen3_5_moe`, 48 layers with full attention every fourth
layer, 256 experts and eight routed experts per token, plus a shared expert.
This establishes hybrid and MoE routing; speculative decode stays disabled.
The template's `<tool_call><function=...><parameter=...>` format selects the
existing `qwen3_coder_xml` parser, while `<think>` and `enable_thinking` select
the existing `qwen3` reasoning parser. No parser implementation changes.
The 96 GiB floor reserves headroom beyond the serialized weights and the
70.9 GB observed Metal allocator peak. The qualification host has 256 GiB;
the floor is conservative short-context sizing, not a measurement on a 96 GiB
host or a guarantee that the full 262,144-token context fits. Context and
concurrency require additional memory. Existing
launch fit selection respects this profile floor; direct serve uses the
existing memory warning behavior. No recommendation-tier promotion is made.

## Serving results

| Alias | Memory floor | EN / ZH chat | SSE | Tools | Thinking | OCR / image |
| --- | ---: | --- | --- | --- | --- | --- |
| `ornith-1.5-122b-a10b-4bit` | 96 GiB | Paris / 北京 | Pass, 103 content deltas on both runs | Structured `get_weather({"city":"Paris"})`, `tool_calls` finish | On/off pass, 391; reasoning separate | Unqualified |

Two-turn recall returned the exact code `maple-527`. All ten cold/warm requests
passed visible-content checks for leaked thinking or function-wire delimiters.
Desktop tool confidence stays `.unknown`: these CLI probes do not substitute
for the Desktop tool-loop benchmark.

| Measurement | First application load | Warm application reload |
| --- | ---: | ---: |
| Command start to `/health/ready` | 81.775 s | 21.625 s |
| First chat server TTFT | 1.581920 s | 1.384276 s |
| Stream server TTFT | 0.158669 s | 0.158932 s |
| Stream client first content | 0.273562 s | 0.250342 s |
| Stream post-first-token decode | 54.145 tok/s | 55.750 tok/s |
| Peak Metal allocation | 70.9 GB | 70.9 GB |

TTFT comes from the difference between consecutive per-model
`rapid_mlx_model_ttft_seconds_sum` values at `/metrics`; the request count
increased by one each time. Decode uses
`rapid_mlx_model_decode_tokens_per_second_last`, which excludes prefill and
first-token latency. Startup is reported separately and is not included in
request TTFT. These short, greedy probes are functional qualification, not a
sustained-load or comparative performance claim.

## Supply-chain and code-path evidence

The exact Hub revision declares MIT and ships no Python files. Config,
tokenizer config and processor config declare neither `model_file` nor
`auto_map`. A 26-file pinned data allowlist fetched 69,642,920,578 bytes;
`verify_snapshot(..., revision=PIN, filenames=ALLOWLIST, local_files_only=True)`
verified every selected SHA256 or Git object digest against the packaged
immutable inventory. The size-estimator manifest also counts small diagnostic
JSON/text and omits Jinja, explaining its 1,884-byte difference from the
qualification download. No other model cache entries were removed.

`resolve_vlm_checkpoint` resolves the catalog pin to the complete local
snapshot. `MLXMultimodalLM.load` applies `apply_remote_code_policy`, validates
`model_file`, and forwards `trust_remote_code=False` into the installed loader.
No remote-code opt-in environment variable was set. The native `qwen3_5_moe`
model and its registered `Qwen3VLProcessor` came from installed package files.
The processor reaches `AutoProcessor` with the same false trust flag. The
image-processor fallback's true flag only reaches JSON `load_config`, not a
repository Python importer; this serving path supplies loader kwargs anyway.
No audio loader runs on this path.

A `sitecustomize` audit hook was installed before CLI imports and remained
active throughout startup and every HTTP probe. It monitored `exec`, `compile`
and import events for snapshot/cache/blob and dynamic-module paths, blocked
repository-code events, and inventoried `sys.modules` at shutdown. A separate
synthetic repository-source compile was blocked before execution, proving the
hook's deny path. The qualification processes recorded:

| Run | Compile events | Exec events | Import events | Repository events / modules |
| --- | ---: | ---: | ---: | --- |
| Cold main server | 7,906 | 9,686 | 4,062 | 0 / 0 |
| Warm main server | 7,704 | 9,685 | 4,061 | 0 / 0 |

Both small helper processes also recorded zero repository events/modules.
Model and processor module filenames pointed into installed dependency
`site-packages`, including `qwen3_5_moe/language.py` and
`qwen3_vl/processing_qwen3_vl.py`. Both servers shut down cleanly, with disk
cache persistence disabled. The Studio lease is retained through landing
validation and released after its final process cleanup; the cold-storage
model cache remains available for subsequent use.

## Regression evidence

- Catalog/alias/size contracts: 4,044 passed.
- Packaged integrity plus new catalog tests: 21 passed.
- New Python test file: 56/56 executable lines covered (100%); no production
  Python implementation was added or changed.
- `swift test --filter ToolUseCapabilityCatalogCoverageTests`: four tests
  passed, including 42 parameter cases and this exact alias's conservative
  tool-confidence expectation.
- Ruff check/format and `git diff --check` passed.
- Self-review checked immutable selection, complete language/vision shards,
  parser/template agreement, memory-floor semantics, existing-entry stability,
  and the two exclusions before exact-head landing validation.

## Excluded: Unlimited-OCR-2

Unlimited-OCR-2 was excluded before runtime qualification. The pinned MIT
upload `mlx-community/Unlimited-OCR-2-4bit@ac04b808479927cc42ea82e06f55ebecd2b0a623`
contains a 2,459,096,904-byte safetensors file. Its config requires trained DASH
block-sparse attention, page/token 2-axis RoPE and page state; its weight index
contains 60 learned `dash_router` tensors. The installed 0.7.2 decoder always
selects the v1 ring/sliding attention model and drops the sparse-attention
config. Its processor also lacks the trained `<PAGE><image>` layout. Version
0.7.10 retains those same decoder/config files. Correct support requires a
full decoder, cache and processor backport rather than a naming shim; the
dependency pin remains unchanged. Repository Python was never executed. No
weights download or runtime audit was performed, and no OCR accuracy, chat,
SSE, tools, thinking, TTFT or throughput result is claimed.

Follow-up owner: Vector, after a separately scoped native v2 integration or
signed dependency update. That work must qualify printed English, Chinese,
and tables under the same default-false repository-code policy.

## Excluded: LFM3-1.7B

Both official quantizations fail the requested ≥99% teacher-forced top-1
agreement against bf16, even before introducing a vendored implementation.
An isolated reference environment on a leased M5 Max (36 GiB, macOS 27.0.1,
Python 3.12.13) used the upstream architecture installed from
[`mlx-vlm` at `230fc308efb577d87446306bfb8a3e9f1a129558`](https://github.com/Blaizzy/mlx-vlm/tree/230fc308efb577d87446306bfb8a3e9f1a129558/mlx_vlm/models/lfm3). It installed MLX 0.32.4,
Transformers 5.21.0, NumPy 2.5.4 and Tokenizers 0.23.3. This environment was
separate from the product environment; no product dependency pin changed.

| Publication | Immutable revision | Safetensors bytes | Top-1 agreement with bf16 | Mean KL, nats |
| --- | --- | ---: | ---: | ---: |
| `LiquidAI/LFM3-1.7B-MLX-4bit` | `d0ca73dcfa70ce18733754a492cdc1fb0123262d` | 973,083,386 | 867/1,024 = 84.67% | 0.13104454 |
| `LiquidAI/LFM3-1.7B-MLX-8bit` | `ee7667f3cfccb2a41725f91820531823fcabc6b7` | 1,837,290,805 | 999/1,024 = 97.56% | 0.00397543 |
| `LiquidAI/LFM3-1.7B` (bf16 reference) | `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` | 3,457,635,448 | Reference | Reference |

All use LFM Open License v1.0 (`lfm1.0`), recorded here consistently with the
existing LFM aliases, whose schema has no license field. JSON/Jinja/weight
download footprints were 991,059,551 / 1,855,266,971 / 3,475,559,631 bytes,
respectively, excluding the license document. Weight-file SHA256 values were
verified for each snapshot:

- 4-bit: `1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815`
- 8-bit: `2b06cea98d75527c248a0b04383f7cf0a1c72fb4ed7b45096618fc4e59d015ad`
- bf16: `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`

Reproduction: read the task-base (`acf8efca3`) `AGENTS.md` as UTF-8, extract
from `### Atlas —` up to `## Working model`, tokenize with each snapshot's
`tokenizer.json`, and take the first 1,024 IDs. The complete extracted prose
has SHA256 `21da713e04f3661f78e208b37226b0c5651eb5148d2dcc97a601ca371b0742fb`.
Tokenizer JSON semantic objects, vocabularies, and input token IDs were equal
across all three variants. Load each checkpoint with the pinned reference's
`load_model(Path(snapshot), trust_remote_code=False)` and compute one full
causal forward on `[ids]`, without cache, covering positions beyond the
256-token sliding window. Convert its 1,024 × 128,000 logits to float32.
Compare argmax on all 1,024 rows; compute float32 log-softmax after subtracting
the row maximum and average `sum(p_bf16 * (log(p_bf16) - log(p_quant)))` over
all rows. Maximum row KL was 3.49631190 (4-bit) and 0.58992839 (8-bit).
An exploratory README sample also failed: 79.10% / 97.75% agreement and
0.25675827 / 0.00303519 mean KL for 4-bit / 8-bit.

This is a checkpoint/reference baseline rejection. No Rapid vendored model
was added, and no Rapid-bf16 near-exact comparison, product chat/SSE/tools,
thinking, prefix-cache, memory floor, TTFT, or decode-rate result is claimed.
Only installed reference architecture code ran; no model-repository Python
executed. The remote model copies and throwaway reference environment were
deleted before the lease was released and its heartbeat stopped.

Follow-up owner: Vector. Revisit a qualifying publication before vendoring;
then require near-exact bf16 agreement, long-window cache/prefix reuse tests,
and serving qualification. The template uses named XML function calls, so the
LFM2 parser must not be inherited without checking that format.
