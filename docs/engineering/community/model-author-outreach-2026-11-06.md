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
quantization variants do not each receive a duplicate request. Pending means
no submission has been made in this campaign and no future compatibility
promise has been made.

| Publisher / family owner candidate | Aliases | Weight repos | Status / next action |
| --- | ---: | ---: | --- |
| [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) | 9 | 9 | Pending: verify original model card, qualification and contribution route |
| [CohereLabs](https://huggingface.co/CohereLabs) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [FireRedTeam](https://huggingface.co/FireRedTeam) | 2 | 1 | Pending: verify original model card, qualification and contribution route |
| [FunAudioLLM](https://huggingface.co/FunAudioLLM) | 2 | 1 | Pending: verify original model card, qualification and contribution route |
| [Hcompany](https://huggingface.co/Hcompany) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [HiDream-ai](https://huggingface.co/HiDream-ai) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) | 1 | 1 | [SmolLM documentation proposal open](https://github.com/huggingface/smollm/issues/155) |
| [IFM](https://huggingface.co/IFM) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [IndexTeam](https://huggingface.co/IndexTeam) | 2 | 1 | Pending: verify original model card, qualification and contribution route |
| [InternScience](https://huggingface.co/InternScience) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [Jackrong](https://huggingface.co/Jackrong) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [Lightricks](https://huggingface.co/Lightricks) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [LiquidAI](https://huggingface.co/LiquidAI) | 7 | 7 | [Submitted; awaiting review](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit/discussions/2) |
| [MiniMaxAI](https://huggingface.co/MiniMaxAI) | 3 | 3 | Pending: verify original model card, qualification and contribution route |
| [Nanbeige](https://huggingface.co/Nanbeige) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [NousResearch](https://huggingface.co/NousResearch) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [Pika-Labs](https://huggingface.co/Pika-Labs) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [Qwen](https://huggingface.co/Qwen) | 80 | 72 | [Community discussion published](https://github.com/QwenLM/Qwen3/discussions/1929) |
| [RecursiveSI](https://huggingface.co/RecursiveSI) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [ResembleAI](https://huggingface.co/ResembleAI) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [SWivid](https://huggingface.co/SWivid) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [TokenRhythm](https://huggingface.co/TokenRhythm) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [Wan-AI](https://huggingface.co/Wan-AI) | 5 | 5 | Pending: verify original model card, qualification and contribution route |
| [WeiboAI](https://huggingface.co/WeiboAI) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [XHToken](https://huggingface.co/XHToken) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [ai9stars](https://huggingface.co/ai9stars) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [alibaba-pai](https://huggingface.co/alibaba-pai) | 3 | 3 | Pending: verify original model card, qualification and contribution route |
| [allenai](https://huggingface.co/allenai) | 7 | 7 | Pending: verify original model card, qualification and contribution route |
| [black-forest-labs](https://huggingface.co/black-forest-labs) | 3 | 3 | Pending: verify original model card, qualification and contribution route |
| [deepseek-ai](https://huggingface.co/deepseek-ai) | 10 | 9 | Pending: verify original model card, qualification and contribution route |
| [google](https://huggingface.co/google) | 40 | 39 | [Cookbook proposal open](https://github.com/google-gemma/cookbook/issues/446) |
| [hexgrad](https://huggingface.co/hexgrad) | 7 | 3 | Pending: verify original model card, qualification and contribution route |
| [ibm-granite](https://huggingface.co/ibm-granite) | 10 | 9 | [Official model discussion published](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx/discussions/1) |
| [inclusionAI](https://huggingface.co/inclusionAI) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [meta-llama](https://huggingface.co/meta-llama) | 4 | 4 | Pending: verify original model card, qualification and contribution route |
| [meta-models](https://huggingface.co/meta-models) | 3 | 3 | Pending: verify original model card, qualification and contribution route |
| [microsoft](https://huggingface.co/microsoft) | 6 | 5 | [Phi documentation proposal open](https://github.com/microsoft/PhiCookBook/issues/606) |
| [mistralai](https://huggingface.co/mistralai) | 6 | 5 | Pending: verify original model card, qualification and contribution route |
| [moonshotai](https://huggingface.co/moonshotai) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [nari-labs](https://huggingface.co/nari-labs) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [nvidia](https://huggingface.co/nvidia) | 17 | 13 | [Cookbook proposal open](https://github.com/NVIDIA-NeMo/Nemotron/issues/394) |
| [openai](https://huggingface.co/openai) | 18 | 14 | Pending: verify original model card, qualification and contribution route |
| [openbmb](https://huggingface.co/openbmb) | 8 | 7 | [Official model-card PR open](https://huggingface.co/openbmb/MiniCPM5-2B-MLX/discussions/3) |
| [ornith-ai](https://huggingface.co/ornith-ai) | 3 | 3 | Pending: verify original model card, qualification and contribution route |
| [prism-ml](https://huggingface.co/prism-ml) | 9 | 8 | [Official model-card PR open](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/discussions/1) |
| [stabilityai](https://huggingface.co/stabilityai) | 2 | 2 | Pending: verify original model card, qualification and contribution route |
| [tencent](https://huggingface.co/tencent) | 1 | 1 | Pending: verify original model card, qualification and contribution route |
| [zai-org](https://huggingface.co/zai-org) | 5 | 4 | Pending: verify original model card, qualification and contribution route |

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

## Handoff

Echo owns original-author verification and replies. Atlas owns model/endpoint
qualification; Harbor owns deployment-documentation integration. Nine publisher
candidate groups currently have a submission or published discussion in this
campaign; **42 remain pending**, and additional models within contacted groups
may still need coverage. This is a complete catalog inventory and an initial
outreach batch, not completion of all-model outreach.

Next concrete work: review pending groups against checked-in qualification,
starting with those with dedicated reports (for example Muse and DeepSeek),
then verify their official contribution routes and duplicates. For catalog-only
families, obtain a reproducible run before proposing executable model-card
instructions. Track outcomes at checkpoint level before expanding claims.
No scheduled monitoring or acceptance guarantee.
