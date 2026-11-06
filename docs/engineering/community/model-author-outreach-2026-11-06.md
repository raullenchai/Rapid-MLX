# Model-author outreach coverage — 2026-11-06

Owner: Echo. Host: Studio. Branch: `echo/local-ai-submissions-20261106`.
User direction: work systematically from supported model families back to their
original publishers, following the LiquidAI/OpenBMB pattern. Exclude
mlx-community from promotional submissions. Conversion repositories remain
valid provenance inputs, not outreach destinations.

## Inventory scope and counting

Source commit: `4d5ac49d9a5f82574db78df48d6965f05b2d1c51` (upstream main at audit).

- `rapid_mlx/aliases.json`: **251 aliases / 239 distinct target repositories**.
- `rapid_mlx/audio/aliases.json`: **53 model aliases / 34 additional target repositories**.
  Four underscore-prefixed metadata/assets fields were excluded from model counts.
- Combined: **304 aliases / 273 distinct target repositories**.
- Grouped into **51 publisher/model-family owner candidates**, not 51 independently
  verified corporations and not 304 distinct base models.
- These counts cover these two checked-in catalogs. They do not enumerate every
  dynamically loadable HF model, auxiliary drafter/processor, embedding option
  or architecture. Catalog presence does not establish end-to-end capability.

[Full per-alias inventory](model-owner-inventory-2026-11-06.csv) contains every
entry, pinned weight revision when supplied, original conversion repository,
current model-card metadata revision, candidate author and attribution basis.
The CSV's submission status is at publisher level: a submission concerning one
checkpoint does **not** establish coverage or qualification of every row in
that group. No weights were downloaded for this audit.

## Attribution method and limitations

Queried Hugging Face metadata for all 273 catalog repositories; all resolved.
Preserved `base_model` exactly as published, rather than overwriting it with
an inferred answer. Checked that each of the 51 candidate publisher namespaces
has publicly listed models. Namespace existence is not proof of provenance for
every mapped checkpoint.

Grouped with explicit model-family rules and checked the selected submission
targets against their official README/model cards. The CSV distinguishes
publisher-namespace matches, base-model metadata matches, and family inferences
that still need original-card verification. Fine-tune/research model authors
remain separate from foundation-model authors: Bonsai maps to Prism ML, Hermes
to NousResearch, Qwopus to Jackrong, and Pika to Pika-Labs, even where their
metadata names another publisher's base architecture.

Do not blindly follow `base_model`: observed cards point Hermes to Meta's base,
Kokoro to StyleTTS2, VibeVoice to Qwen, and some Gemma QAT conversions to
InternVL. Missing, self-referential, and nested conversion references also
occur. These are reasons to verify original authors before outreach, not
claims that the runtime's model catalog is incorrect.

## Publisher coverage and next action

Submitted/published is not accepted or merged. Existing threads are reused;
quantization variants do not each receive a duplicate request. Held means this pass reviewed the candidate but left a concrete verification or route prerequisite; no future compatibility promise has been made.

| Publisher / family owner candidate | Aliases | Weight repos | Status / next action |
| --- | ---: | ---: | --- |
| [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) | 9 | 9 | Held: UI-TARS: screenshot preprocessing, coordinate mapping and action execution require qualification; catalog loading is insufficient for a computer-use integration. |
| [CohereLabs](https://huggingface.co/CohereLabs) | 2 | 2 | Held: North Mini Code: verify the cohere_command4 chat/reasoning/tool template against the pinned MLX builds; no model-specific runtime receipt found. |
| [FireRedTeam](https://huggingface.co/FireRedTeam) | 2 | 1 | Held: FireRedASR2 AED: verify the pinned AED path with 16 kHz mono audio and transcription output; do not conflate it with the broader streaming ASR2S system. |
| [FunAudioLLM](https://huggingface.co/FunAudioLLM) | 2 | 1 | Held: SenseVoice: verify /v1/audio/transcriptions, language/task fields and transcript output; do not imply that emotion/event tags are exposed by the API. |
| [Hcompany](https://huggingface.co/Hcompany) | 2 | 2 | Held: Holo3.1: official computer-use model verified. Need screenshot/coordinate/action-loop evidence for the exact MLX build. |
| [HiDream-ai](https://huggingface.co/HiDream-ai) | 1 | 1 | [Evidence report published](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev/discussions/2) |
| [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) | 1 | 1 | [SmolLM documentation proposal open](https://github.com/huggingface/smollm/issues/155) |
| [IFM](https://huggingface.co/IFM) | 1 | 1 | [Evidence report published](https://huggingface.co/IFM/K2-Horizon-7B/discussions/13) |
| [IndexTeam](https://huggingface.co/IndexTeam) | 2 | 1 | Held: IndexTTS 1.5: qualify reference-audio-only conditioning separately from F5; no exact audio result receipt found. |
| [InternScience](https://huggingface.co/InternScience) | 1 | 1 | [Evidence report published](https://huggingface.co/InternScience/Sage-A1.5/discussions/10) |
| [Jackrong](https://huggingface.co/Jackrong) | 2 | 2 | Held: Qwopus: official MLX card found, but no exact 9B/27B Rapid serving receipt. Validate the selected build and tool/thinking behavior before adding commands. |
| [Lightricks](https://huggingface.co/Lightricks) | 2 | 2 | Held: LTX-2.5: authenticated card read succeeded. Existing MLX request thread reviewed; no Rapid duplicate found. Need exact distilled Q8 video/audio receipt and 8n+1-frame constraints before claiming parity. |
| [LiquidAI](https://huggingface.co/LiquidAI) | 7 | 7 | [Submitted; awaiting review](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit/discussions/2) |
| [MiniMaxAI](https://huggingface.co/MiniMaxAI) | 3 | 3 | Held: M2.5/M2.7 text and H3 video require separate evidence. The October 25 M3.1 memory exclusion is not evidence for these catalog entries; no family-wide compatibility claim. |
| [Nanbeige](https://huggingface.co/Nanbeige) | 2 | 2 | [Evidence report published](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/discussions/36) |
| [NousResearch](https://huggingface.co/NousResearch) | 2 | 2 | Held: Existing Hermes Agent integration PR 136903 already covers downstream routing. Hermes 4 70B model-specific reasoning/tool receipt still missing; do not duplicate the provider request. |
| [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | 1 | 1 | Held: MOSS-TTS-Nano: verify audio tokenizer dependency, reference-audio workflow and default voice mapping through /v1/audio/speech. |
| [Pika-Labs](https://huggingface.co/Pika-Labs) | 1 | 1 | [Evidence report published](https://huggingface.co/Pika-Labs/Pika-H3-Flex/discussions/4) |
| [Qwen](https://huggingface.co/Qwen) | 80 | 72 | [Community discussion published](https://github.com/QwenLM/Qwen3/discussions/1929) |
| [RecursiveSI](https://huggingface.co/RecursiveSI) | 1 | 1 | Held: Flint INT4: official card prioritizes Rec; existing discussions report uint8 packing and reasoning-budget problems. Resolve compatibility with a pinned Rapid smoke before another runtime listing. |
| [ResembleAI](https://huggingface.co/ResembleAI) | 2 | 2 | Held: Chatterbox: resolve whether the catalog conversion is original or Turbo, then verify reference-audio input and Perth watermark behavior in the mlx-audio path. Do not transfer the Turbo card claims blindly. |
| [SWivid](https://huggingface.co/SWivid) | 1 | 1 | Held: F5-TTS: verify the Chinese conversion provenance plus both reference audio and reference transcript through the speech endpoint. |
| [TokenRhythm](https://huggingface.co/TokenRhythm) | 1 | 1 | [Evidence report published](https://huggingface.co/TokenRhythm/NeoHorse-1-9B/discussions/3) |
| [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) | 1 | 1 | Held: Z-Image-Turbo: original card and existing MLX thread reviewed. Need a pinned real image request and steps/guidance check; no dedicated report found. |
| [Wan-AI](https://huggingface.co/Wan-AI) | 5 | 5 | Held: Wan: catalog spans 2.1 and 2.2. Select a specific checkpoint and qualify its frame-count, text/image conditioning and video result; do not reuse the 2.2 guide as evidence for 2.1. |
| [WeiboAI](https://huggingface.co/WeiboAI) | 2 | 2 | Held: VibeThinker: official guidance recommends a large output budget; validate complete answers and resource sizing, not only model loading. |
| [XHToken](https://huggingface.co/XHToken) | 1 | 1 | [Evidence report published](https://huggingface.co/XHToken/Spark-X2.6-4B/discussions/31) |
| [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) | 1 | 1 | Held: MiMo V2.6: distinguish text, vision, audio and speculative-decoder capabilities; no exact Rapid endpoint qualification found. |
| [ai9stars](https://huggingface.co/ai9stars) | 1 | 1 | Held: G9v3: existing discussion asks for parser guidance. Verify the concrete tool/reasoning format and end-to-end response before proposing a recipe. |
| [alibaba-pai](https://huggingface.co/alibaba-pai) | 3 | 3 | Held: CogVideoX-Fun V1.5 InP: Rapid video guide exists, but no dedicated pinned qualification found. Verify actual inpainting/conditioning support and async video output before a card contribution. |
| [allenai](https://huggingface.co/allenai) | 7 | 7 | [Existing Tmax community report; no duplicate](https://huggingface.co/allenai/tmax-9b/discussions/2) |
| [black-forest-labs](https://huggingface.co/black-forest-labs) | 3 | 3 | [Evidence report published](https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/234) |
| [deepseek-ai](https://huggingface.co/deepseek-ai) | 10 | 9 | [Evidence report published](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/78) |
| [google](https://huggingface.co/google) | 40 | 39 | [Cookbook proposal open](https://github.com/google-gemma/cookbook/issues/446) |
| [hexgrad](https://huggingface.co/hexgrad) | 7 | 3 | Held: Kokoro: official HF discussion endpoint returns 403 (discussions disabled). No post attempted; review a permitted contribution route and qualify audio/voice selection first. |
| [ibm-granite](https://huggingface.co/ibm-granite) | 10 | 9 | [Official model discussion published](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx/discussions/1) |
| [inclusionAI](https://huggingface.co/inclusionAI) | 1 | 1 | [Evidence report published](https://huggingface.co/inclusionAI/Ling-3.0-tiny/discussions/19) |
| [meta-llama](https://huggingface.co/meta-llama) | 4 | 4 | Held: Official Llama 3.2 card fetch denied (403) with existing credentials; no access terms accepted. Obtain authorized access and a pinned serving receipt before a card patch. |
| [meta-models](https://huggingface.co/meta-models) | 3 | 3 | [Evidence report published](https://huggingface.co/meta-models/Muse-Glimmer-30B/discussions/73) |
| [microsoft](https://huggingface.co/microsoft) | 6 | 5 | [Phi documentation proposal open](https://github.com/microsoft/PhiCookBook/issues/606) |
| [mistralai](https://huggingface.co/mistralai) | 6 | 5 | Held: Devstral Small 2: verify Mistral tool formatting and screenshot input on the exact conversion before extending the existing local deployment options. |
| [moonshotai](https://huggingface.co/moonshotai) | 1 | 1 | Held: Kimi K2.6: verify actual DQ3 artifact fit, preserved reasoning and multi-step tools on available hardware before a Mac deployment claim. |
| [nari-labs](https://huggingface.co/nari-labs) | 1 | 1 | Held: Dia: community contributions are welcomed, but need a recorded mlx-audio run with speaker tags, reference conditioning and valid audio output. |
| [nvidia](https://huggingface.co/nvidia) | 17 | 13 | [Cookbook proposal open](https://github.com/NVIDIA-NeMo/Nemotron/issues/394) |
| [openai](https://huggingface.co/openai) | 18 | 14 | Held: GPT-OSS: September Personal Intelligence receipt failed 14/15; that feature was retired October 18. Requalify current Harmony serving before advertising an agent recipe. Whisper needs a separate audio receipt. |
| [openbmb](https://huggingface.co/openbmb) | 8 | 7 | [Official model-card PR open](https://huggingface.co/openbmb/MiniCPM5-2B-MLX/discussions/3) |
| [ornith-ai](https://huggingface.co/ornith-ai) | 3 | 3 | [Official model-card PR open](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit/discussions/1) |
| [prism-ml](https://huggingface.co/prism-ml) | 9 | 8 | [Official model-card PR open](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/discussions/1) |
| [stabilityai](https://huggingface.co/stabilityai) | 2 | 2 | [Evidence report published](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/261) |
| [tencent](https://huggingface.co/tencent) | 1 | 1 | Held: Hy3: historical matrix mention is insufficient for the current preview pin. Verify hunyuan/hy_v3 parsing, memory and the publisher license attribution. |
| [zai-org](https://huggingface.co/zai-org) | 5 | 4 | [Evidence report published](https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/70) |

## Contribution depth

1. **Verified checkpoint and serving evidence:** prepare a concrete model-card or
   cookbook patch with pinned model/runtime versions and scoped observations.
2. **Catalog-only evidence:** submit an explicitly labeled scope proposal only
   where the official community welcomes it. Do not claim a verified recipe.
3. **Unresolved author, missing tests or unsuitable contribution route:** retain
   the candidate and resolve those prerequisites before posting. Do not send
   template issues across all 51 namespaces.
4. **Already contacted owner:** extend the existing thread when there is new
   evidence or a specific model-family request. Do not repeat a listing request
   for each quantization or use a different repository to evade a refusal.
5. **Audio/image/video models:** first establish their actual endpoint and
   workload. Do not reuse text chat-completion recipes for other modalities.

## New submissions from this audit

- Prism ML: README-only PR on the official Bonsai 2 2B MLX pack. It retains
  Prism's source-of-truth demo instructions, includes the non-thinking sampler,
  and cites the 2026-10-15 native Hadamard qualification. Exact model revision
  `1c6cbae21f8c38830ce337972b46fdae7b18eaaa`; historical M3 Ultra 256 GiB
  results only. No broad agent/vision, long-context or speed claim.
- IBM Granite: one discussion on the official 3B q4 model, covering the six
  4.2 q4/q8 aliases as a proposed recipe family. Specifically calls out the
  card's BOS/re-encoding and thinking-mode caveats as future checks. Its GitHub
  CONTRIBUTING accepts bug/inconsistency issues only, so no feature request was
  posted there. No fresh Granite execution or qualification claimed.
- Microsoft Phi: documentation request using the PhiCookBook issue template,
  proposing an API-serving addition to its existing MLX inference chapter.
  Third-party weight conversions are explicitly distinguished from official
  Microsoft models. No CLA signed and no Phi runtime test claimed.
- Hugging Face SmolLM: documentation proposal for SmolLM3 text API serving in
  local-inference materials. Thinking, context and streaming need qualification;
  no SmolVLM or tool reliability claim. Kept separate from existing smolagents
  outreach because the repository and contribution concern model deployment.

All four disclose maintainer affiliation, AI assistance and telemetry opt-out.
Bonsai shell/request JSON syntax checked; original model-card content and
metadata preserved. This audit did not run fresh model inference. Re-read
submission status after creation. The outreach ledger contains the links.

## Follow-up batch: official model communities

Reviewed all 42 previously pending candidate groups against original cards,
existing discussions and checked-in qualification records. Published **13 new
HF discussions and one HF model-card PR**, and recovered an existing AllenAI
Tmax post without duplicating it. The 14 new submissions were read back after
creation and were open; publication does not imply acceptance or endorsement.

| Owner | Contribution | Evidence scope |
| --- | --- | --- |
| TokenRhythm | [discussion](https://huggingface.co/TokenRhythm/NeoHorse-1-9B/discussions/3) | [2026-09-08-neohorse-9b-chat-qualification.md](../performance/2026-09-08-neohorse-9b-chat-qualification.md) |
| IFM | [discussion](https://huggingface.co/IFM/K2-Horizon-7B/discussions/13) | [2026-09-14-k2-horizon-7b-qualification.md](../performance/2026-09-14-k2-horizon-7b-qualification.md) |
| meta-models | [discussion](https://huggingface.co/meta-models/Muse-Glimmer-30B/discussions/73) | [muse-glimmer-30b-dflash-qualification.md](../performance/muse-glimmer-30b-dflash-qualification.md) |
| Nanbeige | [discussion](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/discussions/36) | [2026-10-25-trending-models.md](../performance/2026-10-25-trending-models.md) |
| XHToken | [discussion](https://huggingface.co/XHToken/Spark-X2.6-4B/discussions/31) | [2026-10-25-trending-models.md](../performance/2026-10-25-trending-models.md) |
| InternScience | [discussion](https://huggingface.co/InternScience/Sage-A1.5/discussions/10) | [2026-10-25-trending-models.md](../performance/2026-10-25-trending-models.md) |
| HiDream-ai | [discussion](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev/discussions/2) | [2026-09-04-hidream-o1-dev-dogfood.md](../performance/2026-09-04-hidream-o1-dev-dogfood.md) |
| stabilityai | [discussion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/261) | [2026-09-05-sdxl-base-dogfood.md](../performance/2026-09-05-sdxl-base-dogfood.md) |
| black-forest-labs | [discussion](https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/234) | [2026-09-04-flux1-schnell-dogfood.md](../performance/2026-09-04-flux1-schnell-dogfood.md) |
| deepseek-ai | [discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/78) | [2026-10-21-dsv41-0167-same-machine-ab.md](../performance/2026-10-21-dsv41-0167-same-machine-ab.md) |
| inclusionAI | [discussion](https://huggingface.co/inclusionAI/Ling-3.0-tiny/discussions/19) | [2026-09-16-personal-intelligence-remaining-builds-qualification.md](../performance/2026-09-16-personal-intelligence-remaining-builds-qualification.md) |
| zai-org | [discussion](https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/70) | [2026-09-12-glm53-real-task-mtp.md](../performance/2026-09-12-glm53-real-task-mtp.md) |
| Pika-Labs | [discussion](https://huggingface.co/Pika-Labs/Pika-H3-Flex/discussions/4) | [2026-11-03-pika-ref2va-reference-video-memory.md](../performance/2026-11-03-pika-ref2va-reference-video-memory.md) |
| ornith-ai | [model-card PR](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit/discussions/1) | [2026-10-27-trending-models.md](../performance/2026-10-27-trending-models.md) |

Ornith's PR only adds a community serving section; original model-card metadata
and content remain intact. Its shell snippets and request JSON parse; no new
model execution was performed. Other posts are specific historical evidence
reports, not asserted new benchmarks. NeoHorse/K2 experimental status and changed
catalog pins, Muse's 8-bit-only DFlash qualification, Spark/Sage coalesced SSE,
DeepSeek's memory requirements, Ling's retired product feature, and Pika's
incomplete immutable reproduction bundle are explicitly disclosed. Image posts
do not imply editing or refiner parity. All new posts disclose affiliation,
AI assistance and telemetry opt-out.

AllenAI already has [the Tmax model post](https://huggingface.co/allenai/tmax-9b/discussions/2)
and [the original project issue](https://github.com/hamishivi/tmax/issues/3).
NousResearch has an existing [Hermes Agent provider PR](https://github.com/NousResearch/hermes-agent/pull/136903);
that does not qualify Hermes 4 model behavior and is not counted as a new model
submission. The broad GitHub PR search reached its 1,000-result limit, so it is
not an exhaustive historical audit; official HF targets were checked directly.

## Handoff

Echo owns model-author replies and route verification. Atlas/Vector own the
checkpoint-specific qualifications described in the table; Harbor owns the
resulting deployment documentation. Branch: `echo/local-ai-submissions-20261106`.

**24 of 51 publisher candidate groups have a model-oriented submission or
existing published report; 27 are held with the specific next action above.**
Coverage is at publisher level, not verification of every alias. No weights
were downloaded and no fresh inference was run in this outreach pass.

Priority: qualify North Mini Code and Devstral tool paths, the computer-use
models' screenshot/action behavior, and one concrete checkpoint per remaining
audio/video family. Obtain authorized Llama card access; respect Kokoro's
disabled discussions. Respond to the submitted threads before creating new ones.
No scheduled monitoring, automatic re-posting or acceptance guarantee.
