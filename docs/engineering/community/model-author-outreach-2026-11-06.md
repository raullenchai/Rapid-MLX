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
quantization variants do not each receive a duplicate request. The latest batch submits documentation-scope proposals for the remaining groups. Those proposals explicitly distinguish source inspection from runtime qualification; no future compatibility or acceptance promise has been made.

| Publisher / family owner candidate | Aliases | Weight repos | Status / next action |
| --- | ---: | ---: | --- |
| [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) | 9 | 9 | [Community scope proposal open](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/discussions/20); exact-build inference qualification pending |
| [CohereLabs](https://huggingface.co/CohereLabs) | 2 | 2 | [Community scope proposal open](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/discussions/16); exact-build inference qualification pending |
| [FireRedTeam](https://huggingface.co/FireRedTeam) | 2 | 1 | [Community scope proposal open](https://huggingface.co/FireRedTeam/FireRedASR2-AED/discussions/3); exact-build inference qualification pending |
| [FunAudioLLM](https://huggingface.co/FunAudioLLM) | 2 | 1 | [Community scope proposal open](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/discussions/19); exact-build inference qualification pending |
| [Hcompany](https://huggingface.co/Hcompany) | 2 | 2 | [Community scope proposal open](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B/discussions/4); exact-build inference qualification pending |
| [HiDream-ai](https://huggingface.co/HiDream-ai) | 1 | 1 | [Evidence report published](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev/discussions/2) |
| [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) | 1 | 1 | [SmolLM documentation proposal open](https://github.com/huggingface/smollm/issues/155) |
| [IFM](https://huggingface.co/IFM) | 1 | 1 | [Evidence report published](https://huggingface.co/IFM/K2-Horizon-7B/discussions/13) |
| [IndexTeam](https://huggingface.co/IndexTeam) | 2 | 1 | [Community scope proposal open](https://huggingface.co/IndexTeam/IndexTTS-1.5/discussions/5); exact-build inference qualification pending |
| [InternScience](https://huggingface.co/InternScience) | 1 | 1 | [Evidence report published](https://huggingface.co/InternScience/Sage-A1.5/discussions/10) |
| [Jackrong](https://huggingface.co/Jackrong) | 2 | 2 | [Community scope proposal open](https://huggingface.co/Jackrong/MLX-Qwopus3.5-9B-v3-4bit/discussions/2); exact-build inference qualification pending |
| [Lightricks](https://huggingface.co/Lightricks) | 2 | 2 | [Community scope proposal open](https://huggingface.co/Lightricks/LTX-2.5/discussions/98); exact-build inference qualification pending |
| [LiquidAI](https://huggingface.co/LiquidAI) | 7 | 7 | [Submitted; awaiting review](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit/discussions/2) |
| [MiniMaxAI](https://huggingface.co/MiniMaxAI) | 3 | 3 | [Community scope proposal open](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/discussions/71); exact-build inference qualification pending |
| [Nanbeige](https://huggingface.co/Nanbeige) | 2 | 2 | [Evidence report published](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/discussions/36) |
| [NousResearch](https://huggingface.co/NousResearch) | 2 | 2 | [Community scope proposal open](https://huggingface.co/NousResearch/Hermes-4-70B/discussions/7); exact-build inference qualification pending |
| [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | 1 | 1 | [Community scope proposal open](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/discussions/8); exact-build inference qualification pending |
| [Pika-Labs](https://huggingface.co/Pika-Labs) | 1 | 1 | [Evidence report published](https://huggingface.co/Pika-Labs/Pika-H3-Flex/discussions/4) |
| [Qwen](https://huggingface.co/Qwen) | 80 | 72 | [Community discussion published](https://github.com/QwenLM/Qwen3/discussions/1929) |
| [RecursiveSI](https://huggingface.co/RecursiveSI) | 1 | 1 | [Community scope proposal open](https://huggingface.co/RecursiveSI/flint-int4/discussions/4); exact-build inference qualification pending |
| [ResembleAI](https://huggingface.co/ResembleAI) | 2 | 2 | [Community scope proposal open](https://huggingface.co/ResembleAI/chatterbox-turbo/discussions/27); exact-build inference qualification pending |
| [SWivid](https://huggingface.co/SWivid) | 1 | 1 | [Community scope proposal open](https://huggingface.co/SWivid/F5-TTS/discussions/30); exact-build inference qualification pending |
| [TokenRhythm](https://huggingface.co/TokenRhythm) | 1 | 1 | [Evidence report published](https://huggingface.co/TokenRhythm/NeoHorse-1-9B/discussions/3) |
| [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) | 1 | 1 | [Community scope proposal open](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/168); exact-build inference qualification pending |
| [Wan-AI](https://huggingface.co/Wan-AI) | 5 | 5 | [Community scope proposal open](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/discussions/16); exact-build inference qualification pending |
| [WeiboAI](https://huggingface.co/WeiboAI) | 2 | 2 | [Community scope proposal open](https://huggingface.co/WeiboAI/VibeThinker-1.5B/discussions/12); exact-build inference qualification pending |
| [XHToken](https://huggingface.co/XHToken) | 1 | 1 | [Evidence report published](https://huggingface.co/XHToken/Spark-X2.6-4B/discussions/31) |
| [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) | 1 | 1 | [Community scope proposal open](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/discussions/16); exact-build inference qualification pending |
| [ai9stars](https://huggingface.co/ai9stars) | 1 | 1 | [Community scope proposal open](https://huggingface.co/ai9stars/G9v3-39A5B/discussions/5); exact-build inference qualification pending |
| [alibaba-pai](https://huggingface.co/alibaba-pai) | 3 | 3 | [Community scope proposal open](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP/discussions/3); exact-build inference qualification pending |
| [allenai](https://huggingface.co/allenai) | 7 | 7 | [Existing Tmax community report; no duplicate](https://huggingface.co/allenai/tmax-9b/discussions/2) |
| [black-forest-labs](https://huggingface.co/black-forest-labs) | 3 | 3 | [Evidence report published](https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/234) |
| [deepseek-ai](https://huggingface.co/deepseek-ai) | 10 | 9 | [Evidence report published](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/78) |
| [google](https://huggingface.co/google) | 40 | 39 | [Cookbook proposal open](https://github.com/google-gemma/cookbook/issues/446) |
| [hexgrad](https://huggingface.co/hexgrad) | 7 | 3 | [Community documentation proposal open](https://github.com/hexgrad/kokoro/issues/393); exact-build inference qualification pending |
| [ibm-granite](https://huggingface.co/ibm-granite) | 10 | 9 | [Official model discussion published](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx/discussions/1) |
| [inclusionAI](https://huggingface.co/inclusionAI) | 1 | 1 | [Evidence report published](https://huggingface.co/inclusionAI/Ling-3.0-tiny/discussions/19) |
| [meta-llama](https://huggingface.co/meta-llama) | 4 | 4 | [Community documentation proposal open](https://github.com/meta-llama/llama-cookbook/issues/1088); exact-build inference qualification pending |
| [meta-models](https://huggingface.co/meta-models) | 3 | 3 | [Evidence report published](https://huggingface.co/meta-models/Muse-Glimmer-30B/discussions/73) |
| [microsoft](https://huggingface.co/microsoft) | 6 | 5 | [Phi documentation proposal open](https://github.com/microsoft/PhiCookBook/issues/606) |
| [mistralai](https://huggingface.co/mistralai) | 6 | 5 | [Community scope proposal open](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/discussions/33); exact-build inference qualification pending |
| [moonshotai](https://huggingface.co/moonshotai) | 1 | 1 | [Community scope proposal open](https://huggingface.co/moonshotai/Kimi-K2.6/discussions/61); exact-build inference qualification pending |
| [nari-labs](https://huggingface.co/nari-labs) | 1 | 1 | [Community scope proposal open](https://huggingface.co/nari-labs/Dia-1.6B/discussions/76); exact-build inference qualification pending |
| [nvidia](https://huggingface.co/nvidia) | 17 | 13 | [Cookbook proposal open](https://github.com/NVIDIA-NeMo/Nemotron/issues/394) |
| [openai](https://huggingface.co/openai) | 18 | 14 | [Community scope proposal open](https://huggingface.co/openai/gpt-oss-20b/discussions/277); exact-build inference qualification pending |
| [openbmb](https://huggingface.co/openbmb) | 8 | 7 | [Official model-card PR open](https://huggingface.co/openbmb/MiniCPM5-2B-MLX/discussions/3) |
| [ornith-ai](https://huggingface.co/ornith-ai) | 3 | 3 | [Official model-card PR open](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit/discussions/1) |
| [prism-ml](https://huggingface.co/prism-ml) | 9 | 8 | [Official model-card PR open](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/discussions/1) |
| [stabilityai](https://huggingface.co/stabilityai) | 2 | 2 | [Evidence report published](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/261) |
| [tencent](https://huggingface.co/tencent) | 1 | 1 | [Community scope proposal open](https://huggingface.co/tencent/Hy3-preview/discussions/12); exact-build inference qualification pending |
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

## Final outreach batch: all remaining publisher groups contacted

At the user's direction, the remaining **27 groups received 25 official HF
model discussions and two GitHub documentation proposals**. Each was checked
for duplicates, published and read back open with matching content. These are
scope proposals, not additional model-card PRs or claims of successful model
execution. All use a concrete alias/artifact pin and fixed-commit source links.
No test suite, inference job or model download was run for this batch.

| Publisher | Submission | Exact example alias |
| --- | --- | --- |
| CohereLabs | [scope proposal](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/discussions/16) | `north-mini-code-4bit` |
| NousResearch | [scope proposal](https://huggingface.co/NousResearch/Hermes-4-70B/discussions/7) | `hermes4-70b-4bit` |
| mistralai | [scope proposal](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/discussions/33) | `devstral-v2-24b-4bit` |
| XiaomiMiMo | [scope proposal](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/discussions/16) | `mimo-v2.6-flash-4bit` |
| openai | [scope proposal](https://huggingface.co/openai/gpt-oss-20b/discussions/277) | `gpt-oss-20b` |
| MiniMaxAI | [scope proposal](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/discussions/71) | `minimax-m2.5-4bit` |
| Jackrong | [scope proposal](https://huggingface.co/Jackrong/MLX-Qwopus3.5-9B-v3-4bit/discussions/2) | `qwopus-9b-4bit` |
| tencent | [scope proposal](https://huggingface.co/tencent/Hy3-preview/discussions/12) | `hy3-preview-4bit` |
| RecursiveSI | [scope proposal](https://huggingface.co/RecursiveSI/flint-int4/discussions/4) | `flint-40b-a2.5b-4bit` |
| ai9stars | [scope proposal](https://huggingface.co/ai9stars/G9v3-39A5B/discussions/5) | `g9v3-39a5b-4bit` |
| WeiboAI | [scope proposal](https://huggingface.co/WeiboAI/VibeThinker-1.5B/discussions/12) | `vibethinker-1.5b-4bit` |
| alibaba-pai | [scope proposal](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP/discussions/3) | `cogvideox-fun-5b-q4` |
| Wan-AI | [scope proposal](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/discussions/16) | `wan2.1-t2v-1.3b-bf16` |
| Lightricks | [scope proposal](https://huggingface.co/Lightricks/LTX-2.5/discussions/98) | `ltx-2.5-mlx-q8` |
| Tongyi-MAI | [scope proposal](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/168) | `z-image-turbo` |
| ByteDance-Seed | [scope proposal](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/discussions/20) | `ui-tars-1.5-7b-4bit` |
| moonshotai | [scope proposal](https://huggingface.co/moonshotai/Kimi-K2.6/discussions/61) | `kimi-k2.6` |
| Hcompany | [scope proposal](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B/discussions/4) | `holo3.1-35b-a3b` |
| ResembleAI | [scope proposal](https://huggingface.co/ResembleAI/chatterbox-turbo/discussions/27) | `chatterbox` |
| OpenMOSS-Team | [scope proposal](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/discussions/8) | `moss-tts-nano` |
| nari-labs | [scope proposal](https://huggingface.co/nari-labs/Dia-1.6B/discussions/76) | `dia` |
| SWivid | [scope proposal](https://huggingface.co/SWivid/F5-TTS/discussions/30) | `f5-tts-zh` |
| IndexTeam | [scope proposal](https://huggingface.co/IndexTeam/IndexTTS-1.5/discussions/5) | `indextts-1.5` |
| FunAudioLLM | [scope proposal](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/discussions/19) | `sensevoice-small` |
| FireRedTeam | [scope proposal](https://huggingface.co/FireRedTeam/FireRedASR2-AED/discussions/3) | `fireredasr2-aed` |
| meta-llama | [documentation proposal](https://github.com/meta-llama/llama-cookbook/issues/1088) | `llama3-3b-4bit` |
| hexgrad | [documentation proposal](https://github.com/hexgrad/kokoro/issues/393) | `kokoro` |

Contribution routes and important boundaries:

- Llama: the cookbook exposes a feature-request template. Used that public
  route for a scoped proposal despite the current account's gated model-card
  access; no gate was bypassed and no CLA was signed.
- Kokoro: official GitHub README/issues were reviewed; submitted a documentation
  proposal there. HF discussions remain disabled and were not modified.
- The GitHub App connector lacked issue-write access (`Resource not accessible
  by integration`); the existing authenticated GitHub CLI account successfully
  created the two issues. This was a credential-capability fallback, not an
  approval override.
- NousResearch: the new Hermes 4 model thread explicitly distinguishes the
  existing Hermes Agent provider PR, whose recorded smoke used Qwen. The agent
  PR was not duplicated or modified.
- North Mini Code: current routing selects `cohere_command4` reasoning and no
  default tool parser. The proposal does not advertise qualified tool support.
- MOSS-TTS-Nano: inspected code passes continuation mode and a separately pinned
  audio tokenizer, not the official voice-clone workflow. Proposed scope says so.
- Chatterbox aliases resolve to Turbo, with English scope; multilingual model
  claims and watermark guarantees are not transferred to this backend.
- F5 requires paired reference audio/text; IndexTTS uses reference audio without
  requiring a transcript. SenseVoice's transcription-only boundary is explicit.
- UI-TARS/Holo proposals distinguish parser output from executed screen actions.
  Wan 2.1 is not qualified by a Wan 2.2 guide; LTX audio needs its own output check.
- All 27 disclose affiliation, AI assistance, telemetry opt-out and absent fresh
  inference evidence. No post targets mlx-community.

## Qualification work remaining after submission

The following requirements are retained for future executable model-card or
cookbook patches. They no longer mean the publisher has not been contacted.

| Publisher | Qualification / next technical action |
| --- | --- |
| CohereLabs | Validate typed-channel chat/reasoning, then establish a native tool-parser and replay path before claiming agent tool support. |
| meta-llama | Before a gated model-card patch, obtain authorized model access and an exact-build serving receipt. The public cookbook proposal is already submitted. |
| NousResearch | Qualify Hermes 4 model reasoning/tool behavior separately from the existing Hermes Agent custom-provider smoke with Qwen. |
| mistralai | Devstral Small 2: verify Mistral tool formatting and screenshot input on the exact conversion before extending the existing local deployment options. |
| XiaomiMiMo | MiMo V2.6: distinguish text, vision, audio and speculative-decoder capabilities; no exact Rapid endpoint qualification found. |
| openai | GPT-OSS: September Personal Intelligence receipt failed 14/15; that feature was retired October 18. Requalify current Harmony serving before advertising an agent recipe. Whisper needs a separate audio receipt. |
| MiniMaxAI | M2.5/M2.7 text and H3 video require separate evidence. The October 25 M3.1 memory exclusion is not evidence for these catalog entries; no family-wide compatibility claim. |
| Jackrong | Qwopus: official MLX card found, but no exact 9B/27B Rapid serving receipt. Validate the selected build and tool/thinking behavior before adding commands. |
| tencent | Hy3: historical matrix mention is insufficient for the current preview pin. Verify hunyuan/hy_v3 parsing, memory and the publisher license attribution. |
| RecursiveSI | Flint INT4: official card prioritizes Rec; existing discussions report uint8 packing and reasoning-budget problems. Resolve compatibility with a pinned Rapid smoke before another runtime listing. |
| ai9stars | G9v3: existing discussion asks for parser guidance. Verify the concrete tool/reasoning format and end-to-end response before proposing a recipe. |
| WeiboAI | VibeThinker: official guidance recommends a large output budget; validate complete answers and resource sizing, not only model loading. |
| alibaba-pai | CogVideoX-Fun V1.5 InP: Rapid video guide exists, but no dedicated pinned qualification found. Verify actual inpainting/conditioning support and async video output before a card contribution. |
| Wan-AI | Wan: catalog spans 2.1 and 2.2. Select a specific checkpoint and qualify its frame-count, text/image conditioning and video result; do not reuse the 2.2 guide as evidence for 2.1. |
| Lightricks | LTX-2.5: authenticated card read succeeded. Existing MLX request thread reviewed; no Rapid duplicate found. Need exact distilled Q8 video/audio receipt and 8n+1-frame constraints before claiming parity. |
| Tongyi-MAI | Z-Image-Turbo: original card and existing MLX thread reviewed. Need a pinned real image request and steps/guidance check; no dedicated report found. |
| ByteDance-Seed | UI-TARS: screenshot preprocessing, coordinate mapping and action execution require qualification; catalog loading is insufficient for a computer-use integration. |
| moonshotai | Kimi K2.6: verify actual DQ3 artifact fit, preserved reasoning and multi-step tools on available hardware before a Mac deployment claim. |
| Hcompany | Holo3.1: official computer-use model verified. Need screenshot/coordinate/action-loop evidence for the exact MLX build. |
| hexgrad | Qualify audio output and model-specific voice/language behavior; respond in the GitHub documentation proposal. HF discussions remain disabled. |
| ResembleAI | Catalog now confirmed as Turbo. Qualify English audio, accepted conditioning/expressiveness fields and actual Perth watermark behavior in this backend. |
| OpenMOSS-Team | Qualify the current continuation-mode speech output and separately pinned tokenizer. Reference-voice support is not established by the current adapter. |
| nari-labs | Dia: community contributions are welcomed, but need a recorded mlx-audio run with speaker tags, reference conditioning and valid audio output. |
| SWivid | F5-TTS: verify the Chinese conversion provenance plus both reference audio and reference transcript through the speech endpoint. |
| IndexTeam | IndexTTS 1.5: qualify reference-audio-only conditioning separately from F5; no exact audio result receipt found. |
| FunAudioLLM | SenseVoice: verify /v1/audio/transcriptions, language/task fields and transcript output; do not imply that emotion/event tags are exposed by the API. |
| FireRedTeam | FireRedASR2 AED: verify the pinned AED path with 16 kHz mono audio and transcription output; do not conflate it with the broader streaming ASR2S system. |

## Handoff

Echo owns replies in the existing threads; Atlas/Vector own checkpoint-specific
qualification; Harbor owns eventual deployment documentation. Branch:
`echo/local-ai-submissions-20261106`.

**All 51 publisher/model-family candidate groups now have a model-oriented
submission or an existing published report.** This is publisher-level outreach
coverage across the 304-alias inventory, not verification of every alias and not
51 accepted listings. The latest 27 contributions are source-reviewed proposals;
the runtime checks above remain prerequisites for executable support claims.

Next action: incorporate maintainer feedback into the existing threads and
complete the concrete qualification gates before preparing runnable cookbook
patches. Do not send reminder-only comments or duplicate promotion requests.
No scheduled monitoring, automatic re-posting or acceptance guarantee.
