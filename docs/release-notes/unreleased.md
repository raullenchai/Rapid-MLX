<!-- Draft for the next release. Rename in the version-bump PR. -->

Rapid-MLX 1.0.3 adds HunyuanVideo 1.5 text-to-video, faster decode on qualified models, and more reliable tool calling and structured answers. Desktop can share its API with devices on your network, answer questions about a selected indexed folder, and show where memory is going. Downloads, speech, model selection and first-run CLI chat also get fixes.

**Download:** [Rapid-MLX Desktop for macOS](https://github.com/raullenchai/Rapid-MLX/releases/download/rapid-mac-v1.0.3/rapid-mlx-desktop.dmg) | CLI: `rapid-mlx upgrade` or `curl -fsSL https://rapidmlx.com/install.sh | bash`

## [Unreleased]

- On base and Pro M-series Macs with 48 GB or more RAM, recommendations now lead with Qwen3.6 35B MoE; model labels show total and active parameters so its speed is easier to understand.

## Video: HunyuanVideo 1.5

HunyuanVideo 1.5 generates 480p video from text on Apple Silicon Macs with 64 GB or more unified memory. Both `hunyuan-video-1.5-480p` and `hunyuan-video-1.5-480p-distilled` use the existing video jobs, progress, cancellation and MP4 output. The base checkpoint downloads about 53 GB directly from Hugging Face ([#5679](https://github.com/raullenchai/Rapid-MLX/pull/5679), [#5687](https://github.com/raullenchai/Rapid-MLX/pull/5687)).

The CFG-distilled variant runs one transformer pass per step instead of two; it still uses 50 steps. On an M3 Ultra with 512 GB, at 848×480, 33 frames, 50 steps and seed 42, the measured times were ([#5687](https://github.com/raullenchai/Rapid-MLX/pull/5687)):

| Variant | Load | Denoise | Video decode | Peak memory |
|---|---:|---:|---:|---:|
| Base | 16 s | 1,331 s | 61 s | 20.2 GB |
| CFG-distilled | 9 s | 657 s | 63 s | 20.2 GB |

This release supports text-to-video only for this family. The Tencent Hunyuan Community License excludes the European Union, United Kingdom and South Korea ([#5679](https://github.com/raullenchai/Rapid-MLX/pull/5679)).

## Faster Decode on Qualified Models

Dense Qwen3.5 models gain guarded decode fusions when speculative decoding is off. Qwen3.5/3.6 35B-A3B MoE models gain measured, device-specific tuning for plain decode. These are separate measurements, each against its own baseline, with a 564-token prompt, 256 greedy output tokens and a fresh server per repetition ([#5690](https://github.com/raullenchai/Rapid-MLX/pull/5690), [#5728](https://github.com/raullenchai/Rapid-MLX/pull/5728)).

| Model / change | Chip | Before tok/s | After tok/s | Change |
|---|---|---:|---:|---:|
| Qwen3.5 4B 4-bit, dense fusions | M3 Ultra | 166.89 | 187.86 | +12.6% |
| Qwen3.5 4B 4-bit, dense fusions | M4 Pro | 81.76 | 85.68 | +4.8% |
| 35B-A3B 4-bit, singleton tuning | M3 Ultra | 157.19 | 177.00 | +12.6% |
| 35B-A3B 4-bit, singleton tuning | M5 Max | 167.27 | 171.81 | +2.71% |

The singleton tuning left MTP throughput within spread on M3 Ultra and essentially unchanged on M5 Max; its small M4 Pro gain stayed below the enablement threshold. These gains are not additive ([#5728](https://github.com/raullenchai/Rapid-MLX/pull/5728)).

Compatible dense Qwen3.5, LFM2.5, LFM3 and Gemma 4 decode also combines residual normalization and supported projections, with smaller measured gains on some chips. MTP avoids unused draft-head work while parked, uses continuously measured verification costs, and enables guarded fused 4-bit MoE experts by default. The expert path uses approximate arithmetic, so output can differ slightly from the unfused path ([#5695](https://github.com/raullenchai/Rapid-MLX/pull/5695), [#5674](https://github.com/raullenchai/Rapid-MLX/pull/5674), [#5689](https://github.com/raullenchai/Rapid-MLX/pull/5689), [#5694](https://github.com/raullenchai/Rapid-MLX/pull/5694)).

## More Reliable Agent Tool Calls

Gemma 3 12B and 27B gain structured tool calls, including streamed calls, parallel calls and tool-result history. The smaller 1B and 4B catalog variants stop advertising tool support they could not reliably produce. The new parser keeps quoted brackets inside arguments and fenced examples in prose consistent between streaming and non-streaming ([#5715](https://github.com/raullenchai/Rapid-MLX/pull/5715), [#5750](https://github.com/raullenchai/Rapid-MLX/pull/5750)).

Forced and required tool choices on the Anthropic-compatible API now generate arguments instead of trying to invent an empty call. Automatic tool choice also works when parallel calls are disabled. Existing Qwen and GLM tool parsers pass undeclared tool names to the client so it can return a tool error and let the model recover; the built-in agent still refuses to run unknown tools ([#5724](https://github.com/raullenchai/Rapid-MLX/pull/5724), [#5723](https://github.com/raullenchai/Rapid-MLX/pull/5723), [#5700](https://github.com/raullenchai/Rapid-MLX/pull/5700)).

## Desktop: LAN Access, Folder Chat and Memory

**LAN access** is an opt-in setting under **Settings → Tools**. It restarts the server with a persistent bearer key saved in Keychain and shows addresses and copyable connection details for other devices on your network. Network clients can use inference, while model management and other administrative routes stay on this Mac. Network access is off by default and uses plain HTTP ([#5676](https://github.com/raullenchai/Rapid-MLX/pull/5676)). Local search, including grounded chat, stays available only on this Mac ([#5749](https://github.com/raullenchai/Rapid-MLX/pull/5749)).

**Ask this folder…** starts a chat scoped to one registered Local Search folder, with a removable folder chip and saved passage previews below each answer. It requires **Built-in Agent Harness (Experimental)** in Settings. Folder questions are text-only with tools disabled; no matching evidence produces an abstention without running inference. Saved excerpts survive changes to the original file or index, so you can inspect the evidence used for that answer; passages do not guarantee that the model's answer is correct ([#5654](https://github.com/raullenchai/Rapid-MLX/pull/5654)).

API clients can also request grounded chat from explicitly selected registered folders, with citation IDs and source locators in JSON and streaming responses. Local Search can return full matched chunks on request, without reopening the original files ([#5652](https://github.com/raullenchai/Rapid-MLX/pull/5652), [#5638](https://github.com/raullenchai/Rapid-MLX/pull/5638)).

**Performance settings** now combines the Mac's app, wired, compressed and unused memory with engine Metal allocations, prompt-prefix cache usage, running and waiting requests, and a five-minute trend. Engine allocations overlap the Mac totals; they are not extra memory to add to them. Missing measurements show a dash, and the panel supports VoiceOver and Simplified Chinese ([#5651](https://github.com/raullenchai/Rapid-MLX/pull/5651), [#5677](https://github.com/raullenchai/Rapid-MLX/pull/5677)).

## New Features and Improvements

- **Benchmark is now a regular Desktop tab:** measure models on this Mac and, if you choose, publish results to the Community Benchmark. Nothing runs until you start a benchmark.

- **Open an existing checkpoint:** **Settings → Models → Open existing checkpoint…** registers a complete local MLX chat checkpoint without copying weights or changing the managed download folder. **Remove reference** leaves its files intact; missing or incomplete checkpoints report a problem instead of starting a download ([#5660](https://github.com/raullenchai/Rapid-MLX/pull/5660)).
- **Measured speed references:** Desktop recommendations and CLI `models` and `doctor` show offline references matched to chip, RAM and model configuration. After chat, **Generation details** saves supported execution facts and timing with the answer. Sparse or unmatched data stays unavailable; the ranges are references, not performance guarantees ([#5647](https://github.com/raullenchai/Rapid-MLX/pull/5647), [#5655](https://github.com/raullenchai/Rapid-MLX/pull/5655), [#5659](https://github.com/raullenchai/Rapid-MLX/pull/5659)).
- **Idle speech release:** transcription, alignment and speech-synthesis models can unload while the API stays online, then reload on the next request. Desktop uses this with its idle setting on supported runtimes; CLI users opt in with `--audio-lane-idle-ttl` ([#5635](https://github.com/raullenchai/Rapid-MLX/pull/5635)).
- **Consistent CLI first run:** the installer and cold-cache chat now use the LFM2.5 2.6B starter, about a 1.6 GB download. Installer recommendations use the smaller starter below 8 GB or when RAM is unknown; existing downloaded chat models can still be reused ([#5631](https://github.com/raullenchai/Rapid-MLX/pull/5631), [#5632](https://github.com/raullenchai/Rapid-MLX/pull/5632)).
- **Download checks and recovery:** `pull --check` includes free disk space, corrupt cached files get one targeted repair attempt, and Desktop automatically falls back from an unforced failing mirror to Hugging Face within its retry budget ([#5725](https://github.com/raullenchai/Rapid-MLX/pull/5725), [#5736](https://github.com/raullenchai/Rapid-MLX/pull/5736), [#5666](https://github.com/raullenchai/Rapid-MLX/pull/5666)).
- **Speech compatibility:** standard TTS names `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` resolve to the default local speech model. Audio-enabled installer runs prepare Kokoro's English prerequisites; missing prerequisites get actionable repair instructions ([#5712](https://github.com/raullenchai/Rapid-MLX/pull/5712), [#5713](https://github.com/raullenchai/Rapid-MLX/pull/5713)).
- **Privacy:** consented anonymous metrics now include Desktop launches and daily foreground use, plus coarse startup and agent-readiness failure categories. Diagnostic categories exclude raw messages, paths and credentials; concurrent first runs share one install identity ([#5697](https://github.com/raullenchai/Rapid-MLX/pull/5697), [#5634](https://github.com/raullenchai/Rapid-MLX/pull/5634), [#5698](https://github.com/raullenchai/Rapid-MLX/pull/5698), [#5668](https://github.com/raullenchai/Rapid-MLX/pull/5668), [#5709](https://github.com/raullenchai/Rapid-MLX/pull/5709)).

## Bug Fixes

- **Ask this folder:** small folders now decline unrelated questions instead of answering with citations to irrelevant files ([#5767](https://github.com/raullenchai/Rapid-MLX/issues/5767)).
- Desktop recognizes models in a custom models folder after unloading and lets the next chat start them again ([#5768](https://github.com/raullenchai/Rapid-MLX/issues/5768)).

- JSON output requests now use the same thinking preference as plain chat, avoiding unexpected reasoning that can leave the answer empty.
- `pull` and `chat` now accept the suggested `--force-disk-check` option to proceed past insufficient-disk warnings.
- Structured output preserves reasoning and the user's system-prompt prefix, including on vision models. JSON constraints start after reasoning ends and accept the separator before the answer; fallback guidance is reported in a warning header on multimodal requests ([#5675](https://github.com/raullenchai/Rapid-MLX/pull/5675), [#5678](https://github.com/raullenchai/Rapid-MLX/pull/5678), [#5682](https://github.com/raullenchai/Rapid-MLX/pull/5682), [#5707](https://github.com/raullenchai/Rapid-MLX/pull/5707)).
- Strict Responses streams no longer duplicate JSON or append reasoning to it. Strict schema violations return HTTP 422 `json_schema_violation` instead of a retryable gateway error ([#5706](https://github.com/raullenchai/Rapid-MLX/pull/5706), [#5727](https://github.com/raullenchai/Rapid-MLX/pull/5727)).
- Vision requests enforce the reasoning budget during generation, leaving room for the answer. Casual-chat thinking decisions follow the requested model on multi-model servers, and CLI chat hides reasoning when thinking is off. Models that always reason get a larger default CLI token budget ([#5730](https://github.com/raullenchai/Rapid-MLX/pull/5730), [#5739](https://github.com/raullenchai/Rapid-MLX/pull/5739), [#5716](https://github.com/raullenchai/Rapid-MLX/pull/5716), [#5747](https://github.com/raullenchai/Rapid-MLX/pull/5747)).
- Anthropic-compatible sessions resume with `redacted_thinking` blocks in assistant history ([#5743](https://github.com/raullenchai/Rapid-MLX/pull/5743)).
- Qwen XML tool parsing rejects ambiguous parameters instead of rewriting arguments, and accepts exactly integral decimal values for integer parameters ([#5636](https://github.com/raullenchai/Rapid-MLX/pull/5636), [#5699](https://github.com/raullenchai/Rapid-MLX/pull/5699)).
- Agent setup skips servers that need a missing key or serve non-chat models and gives a working next step. Process discovery recognizes actual Rapid-MLX servers and preserves repository identities when matching models ([#5594](https://github.com/raullenchai/Rapid-MLX/pull/5594), [#5605](https://github.com/raullenchai/Rapid-MLX/pull/5605)).
- Desktop model search results appear directly below the search box. Long repository names keep their distinguishing text, raw repositories retain their identity, and colliding menu titles get distinct labels ([#5633](https://github.com/raullenchai/Rapid-MLX/pull/5633), [#5669](https://github.com/raullenchai/Rapid-MLX/pull/5669), [#5733](https://github.com/raullenchai/Rapid-MLX/pull/5733)).
- Pending or denied external-volume access shows an explanation and **Retry** instead of hanging model discovery or treating unreadable models as missing ([#5670](https://github.com/raullenchai/Rapid-MLX/pull/5670)).
- Desktop downloads and onboarding now explain denied or pending cache-volume access before pulling and after a stall, with permission remedies and a fresh access check on **Retry** ([#5765](https://github.com/raullenchai/Rapid-MLX/issues/5765)).
- Deleting a Desktop server-backed conversation also requests server deletion and revokes saved-passage access once confirmed. Offline or unconfirmed deletion stays pending with **Retry** across relaunch; original files and the search index remain intact, and retained database and attachment bytes are not physically purged ([#5658](https://github.com/raullenchai/Rapid-MLX/pull/5658)).
- Mirror downloads report disk-full errors clearly and retain partial files for retry. Removing a subfolder model variant preserves other cached variants and shared files ([#5746](https://github.com/raullenchai/Rapid-MLX/pull/5746), [#5735](https://github.com/raullenchai/Rapid-MLX/pull/5735)).
- Dynamic model loads use a default memory budget, evicting idle unpinned models or returning HTTP 507 when admission fails instead of risking an overcommit ([#5701](https://github.com/raullenchai/Rapid-MLX/pull/5701)).
- Whisper large-v3 turbo rejects translation requests with a clear remedy: use Whisper large-v3 for translation or the transcription endpoint for source-language text ([#5719](https://github.com/raullenchai/Rapid-MLX/pull/5719)).
- Media errors explain the allowed local-media root and direct audio users to transcribe before chatting ([#5740](https://github.com/raullenchai/Rapid-MLX/pull/5740)).
- CJK emphasis renders correctly next to symbols, with less repeated work on long streamed messages ([#5708](https://github.com/raullenchai/Rapid-MLX/pull/5708)).
- Qwen4 MTP restores both attention and index caches after rejected drafts ([#5646](https://github.com/raullenchai/Rapid-MLX/pull/5646)).
- `doctor` accepts valid editable contributor installs, and successful routine health probes no longer flood INFO access logs; failed probes remain visible ([#5673](https://github.com/raullenchai/Rapid-MLX/pull/5673), [#5648](https://github.com/raullenchai/Rapid-MLX/pull/5648)).

## Upgrade Notes

- **Catalog names:** 44 niche or superseded aliases leave discovery; their repository IDs retain the same execution profiles. Old names warn and forward for one release ([#5684](https://github.com/raullenchai/Rapid-MLX/pull/5684)).
- **GLM 4.7 Flash:** `glm4.7-flash-4bit` replaces the misleading `glm4.7-9b-4bit` name and is correctly marked as MoE. It is the same 31.2B-parameter checkpoint, about 16.9 GB at 4-bit; existing downloads are reused and the old name forwards for one release ([#5748](https://github.com/raullenchai/Rapid-MLX/pull/5748)).
- **Always-reasoning chat models:** when no explicit token limit is given, CLI chat raises its default from 2,048 to 4,096 tokens if the served model cannot turn thinking off. An explicit `--max-tokens` still wins ([#5747](https://github.com/raullenchai/Rapid-MLX/pull/5747)).
- **Model residency:** without an explicit limit, dynamic loading uses the smaller of 80% of physical RAM and 90% of Metal's recommended working set. Explicit memory limits retain their existing behavior ([#5701](https://github.com/raullenchai/Rapid-MLX/pull/5701)).

[Full changelog](https://github.com/raullenchai/Rapid-MLX/compare/v1.0.2...v1.0.3)
