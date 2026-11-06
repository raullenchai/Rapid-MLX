# Local AI directory submissions, 2026-11-06

Owner: Echo (community outreach). Execution host: Studio.

Current outreach exclusion (owner instruction, 2026-11-06): do not submit
promotional listings or model-card recipes to `mlx-community`. The three
fourth-pass model-card PRs were withdrawn and closed; historical submission
records below are retained for deduplication. This does not prohibit using
their model weights or change technical compatibility.

The owner requested discovery and submission to relevant local-AI directories.
All three new submissions disclose the maintainer affiliation and are open
listing requests, not accepted listings. Their repositories explicitly accept
suggestions through issues; no external README was edited.

## Submitted

| Directory | Submission | Suggested category | Verified state |
| --- | --- | --- | --- |
| glincker/awesome-selfhosted-ai | [#25](https://github.com/glincker/awesome-selfhosted-ai/issues/25) | LLM Runtimes and Inference | Open |
| pedramnj/awesome-local-ai | [#18](https://github.com/pedramnj/awesome-local-ai/issues/18) | Local LLM Runtimes | Open |
| bokiko/localist | [#10](https://github.com/bokiko/localist/issues/10) | runner, Mac-specific alternative | Open |

Read each repository's README, CONTRIBUTING.md and submission template before
posting. Re-read each created issue through GitHub REST and confirmed its title,
open state and author (`raullenchai`). Source and description checks used the
current upstream Rapid-MLX README, LICENSE and Desktop privacy policy, rather
than the older website worktree's README. No model or runtime tests were run:
this was an outreach-only task, with no performance claims submitted.

## Deduplication and exclusions

- [CastelDazur/awesome-local-ai](https://github.com/CastelDazur/awesome-local-ai)
  already lists Rapid-MLX under Inference Engines. Its historical PR #27 is
  closed with no merge timestamp; the current README is the evidence of listing.
- [akdeb/awesome-mlx #2](https://github.com/akdeb/awesome-mlx/pull/2) already
  proposes RapidMLX. A search for only the hyphenated spelling misses it.
- Existing open submissions include
  [janhq/awesome-local-ai #151](https://github.com/janhq/awesome-local-ai/pull/151),
  [rafska/awesome-local-llm #31](https://github.com/rafska/awesome-local-llm/pull/31),
  and [Hannibal046/Awesome-LLM #369](https://github.com/Hannibal046/Awesome-LLM/pull/369).
  Do not create duplicate requests.
- BrethofAI/awesome-local-ai advertises no telemetry. Rapid-MLX's current
  anonymous metadata telemetry is enabled by default, so it was excluded.
  The three submitted requests disclose the default and the opt-out controls.
- GitHub reports NOASSERTION for Rapid-MLX's license metadata, while its
  LICENSE expressly identifies Apache-2.0. The glincker submission links the
  actual license so a metadata check does not obscure the evidence.

## Follow-up

Receiving owner: Echo. Branch: `echo/local-ai-submissions-20261106`.
The next action is to review the three issue links for maintainer responses
and answer concrete requests in the existing threads. Approval and timing
are unknown. There is no scheduled monitoring or promise of acceptance.

For future outreach, search both `Rapid-MLX` and `RapidMLX`, inspect the live
README/data file, and check both open and closed issues/PRs before posting.
Describe the Apple Silicon platform limit and telemetry accurately; do not
claim telemetry-free defaults or unconditional benchmark superiority.

## Second pass: agent providers and additional directories

The owner expanded outreach to agent projects that can use Rapid-MLX as their
inference provider. Four additional issues were submitted:

| Project | Submission | Scope |
| --- | --- | --- |
| Agno | [#11511](https://github.com/agno-agi/agno/issues/11511) | Cookbook using existing OpenAILike class |
| CrewAI | [#9197](https://github.com/crewAIInc/crewAI/issues/9197) | Existing OpenAI-compatible LLM connection guide |
| Mastra | [#29641](https://github.com/mastra-ai/mastra/issues/29641) | Custom local model id/url example |
| Best of Open-Source AI | [#131](https://github.com/archestack/best-of-ai/issues/131) | model-serving directory submission |

All disclose affiliation, AI assistance, platform limits, and default telemetry.
Agent examples were checked against the projects' current source/docs, but were
not run end to end. Each proposal says so explicitly and specifies basic response,
streaming where applicable, and deterministic local tool-call checks for future
validation. No new native provider implementation or tested compatibility is
claimed. All four issue URLs were re-read through GitHub REST after submission.

### Contribution rules and follow-up constraints

- Agno requires a maintainer-applied `ready` label before implementation; wait
  for triage on #11511 before opening a cookbook PR.
- Mastra currently disables external PRs and requests detailed issues. Its model
  docs are generated: a future example belongs in the appropriate source/generator.
- CrewAI requires `llm-generated`. The creation request included it, but GitHub
  omitted the label. A subsequent AddLabelsToLabelable request was denied for
  insufficient permissions. The issue body prominently discloses AI authorship
  and asks maintainers to apply the label. This remains an upstream triage action.
- Goose's current CONTRIBUTING.md recommends provider-owned custom-provider JSON
  and documentation, and only exceptionally accepts custom-provider issues. No
  Goose issue or PR was submitted. Any future Goose work should first inspect
  existing Rapid-MLX Goose documentation/configuration and publish a verified
  provider-owned example if needed.
- Pydantic AI already has historical Rapid-MLX PR #5008; vince-lam/awesome-local-llms
  has historical issues/PRs #62, #33, and #27. No duplicate submission was created.
- antranapp/awesome-mlx and tensorchord/Awesome-LLMOps already list Rapid-MLX.
- Computer-Science-Resources/Awesome-Local-Llms is an inactive fork with issues
  disabled; its canonical parent is vince-lam/awesome-local-llms.
- 0xmsu/awesome-inference-engineering requires human checking of AI-assisted
  contributions and rejects automated submissions; no automated submission made.
- tdi/awesome-private-ai is a remaining candidate: accepts local deployment,
  requires 100+ stars and recent maintenance, and expects a focused README PR.
  No submission made in this pass.

Receiving owner remains Echo. Next action: read the four issue threads for
maintainer responses, obtain Agno triage readiness and CrewAI's label, then run
real-model integration checks before describing any agent recipe as verified.

## Ecosystem strategy: models to agents

Rapid-MLX's position is the local inference-serving layer between model artifacts
and agent/application clients. Outreach should make a specific path executable:

```text
Model author / model family
  -> MLX conversion, quantization and model card
  -> Rapid-MLX on Apple Silicon (runtime, cache, parsing, local API)
  -> Agent framework / coding agent / application
  -> User task and verified result
```

Apple Silicon/Metal/MLX enable the runtime underneath this path. Package managers
and desktop distribution install it. Benchmarks and resource directories help
people discover and select it. Those supporting layers are not interchangeable
with a model-provider or agent-provider integration.

| Layer | Reader's problem | Useful contribution | Evidence threshold |
| --- | --- | --- | --- |
| Model authors and family cookbooks | How do I run this model on my Mac? | Model-specific local serving guide or community recipe | Exact checkpoint/revision, package versions, host/RAM, observed capabilities and limitations |
| Quantizers and model hubs | I downloaded these weights; what runs them? | Model-card installation, server and request example | Actual format/loader compatibility, provenance, original model terms, immutable model identity |
| Rapid-MLX runtime | Can this artifact serve the requested workload? | Loader/parser/cache fixes and qualifications | Model-correctness and serving tests, not just an alias entry |
| Agents and SDKs | How do I connect a local inference provider? | Existing custom-endpoint recipe or cookbook | Endpoint/auth/model-ID handling plus real streaming/tool-loop checks before claiming verification |
| Applications and integrations | Can the user complete a task locally? | Provider settings, tested user walkthrough | End-to-end task result, model limits, local/remote tool boundary |
| Distribution and discovery | How do I install or compare options? | Package metadata, release/install docs, curated entry, measured comparison | Working install links and scoped reproducible claims |

Prioritize relevance, an accepted contribution route, verified model capability,
and maintenance cost over the number of issues opened. A native provider class
is unnecessary when a project's existing compatible-endpoint mechanism suffices.
A model alias alone is not proof of tool-call reliability. Static cloud-provider
registries are not automatically a fit for a server whose model inventory is local
and user-selected. Framework/runtime repositories should receive concrete fixes
or documented ecosystem entries, not unrelated promotion.

## Third pass: model-side submissions

| Layer / owner | Submission | Concrete contribution | State at verification |
| --- | --- | --- | --- |
| Liquid AI community ecosystem | [Liquid4All/cookbook PR #144](https://github.com/Liquid4All/cookbook/pull/144) | One row under Community Projects → Deployment for LFM3-1.7B serving | Open, not merged |
| Liquid AI official MLX model card | [Hugging Face PR #2](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit/discussions/2) | README-only local API serving instructions for the exact 8-bit checkpoint | Open PR; main unchanged |
| Qwen model community | [Qwen3 discussion #1929](https://github.com/QwenLM/Qwen3/discussions/1929) | Model-to-local-server-to-agent walkthrough in Show and tell | Published discussion, not documentation acceptance |
| NVIDIA model cookbook | [Nemotron issue #394](https://github.com/NVIDIA-NeMo/Nemotron/issues/394) | Narrow Bolt E4B text-serving recipe proposal with conversion and HTTP evidence | Open proposal |
| Google model cookbook | [Gemma issue #446](https://github.com/google-gemma/cookbook/issues/446) | Gemma 4 local API guide proposal with unified-memory/context lessons | Open proposal |

### Evidence and submission boundaries

- LFM3: `LiquidAI/LFM3-1.7B-MLX-8bit` at
  `ee7667f3cfccb2a41725f91820531823fcabc6b7`, alias `lfm3-1.7b-8bit`.
  The 2026-10-28 native qualification records text, SSE, one structured tool
  call and cache reuse. Claims do not extend to all LFM variants or quantizations.
- Nemotron: the existing 2026-10-23 Bolt E4B qualification covers the published
  Rapid-MLX 4/8-bit text conversions, HTTP tool calls and continuations. It does
  not qualify vision, audio, streaming or DFlash. Community conversions are
  explicitly distinguished from NVIDIA-published artifacts.
- Gemma: the 32GB M2 Pro guide and 2026-08-21 matrix are historical evidence.
  A future cookbook requires fresh pinned-version validation; old tuning flags
  and defaults are not represented as universally current. The old
  google-gemini/gemma-cookbook directs users to google-gemma/cookbook. The latter
  requires discussing new guides before implementation, and a future code/notebook
  contribution has a CLA step. No agreement was signed or CLA status asserted.
- Qwen disables blank issues and directs discussion to Discussions. Used Show
  and tell for a community project rather than filing a model bug. Searched
  existing issues and discussions before posting.
- The new model-side proposals cite existing checked-in qualification, not fresh
  runtime tests performed during outreach. Most evidence links pin Rapid-MLX
  source commit `4d5ac49d9a5f82574db78df48d6965f05b2d1c51`.
- Hugging Face PR modifies only README.md, preserving all prior model-card content,
  metadata and model files. It includes the alias's pinned checkpoint revision,
  model discovery, a streaming curl example, model/runtime license distinction,
  and telemetry disclosure. Verified the discussion is a PR and main is unchanged.

### Review and handoff

Liquid cookbook PR head `06640b4` passed `git diff --check` and independent
read-only review (LGTM). The configured spark2 pr-review-loop runner failed with
expired authentication before producing a verdict. A separate reviewer performed
the fallback review, and that limitation was recorded in the PR comment; do not
report the remote review loop as passed. No upstream merge or acceptance claimed.

Receiving owner: Echo, with Atlas for runtime/compatibility changes and Harbor
for published deployment documentation. Current outreach branch remains
`echo/local-ai-submissions-20261106`; external Liquid PR branch is
`raullenchai/cookbook:add-rapid-mlx-community`.

Next concrete actions: respond in the existing five threads, address model-owner
placement requests, and run fresh model-specific serving/tool-loop checks before
turning Nemotron or Gemma proposals into full recipes. Extend model-card outreach
only to individually verified checkpoints. Restore the remote reviewer's login
before expecting its runner to work; no credential changes were made in this task.

## Fourth pass: application connectors and MLX model distribution

The owner authorized the next batch: Open WebUI, AnythingLLM, LiteLLM and
three to five individually evidenced MLX model cards. Repository inspection
changed the application-side contribution route:

- Open WebUI docs already has unmerged closed PRs
  [#1149](https://github.com/open-webui/docs/pull/1149) and
  [#1491](https://github.com/open-webui/docs/pull/1491). A maintainer explicitly
  objected to broad repetitive promotion in #1149. No third submission or
  follow-up was posted there.
- AnythingLLM's current CONTRIBUTING.md rejects dedicated standard-compatible
  provider adapters and promotional integration requests, directing those users
  to Generic OpenAI. No promotional issue or adapter PR was opened.
- LiteLLM's previous native-provider
  [PR #24325](https://github.com/BerriAI/litellm/pull/24325) is closed, unmerged,
  after a stale notice. Posted one substantive
  [follow-up in that original thread](https://github.com/BerriAI/litellm/pull/24325#issuecomment-6696580771),
  linking the concrete generic-connector guide and asking whether a scoped docs
  example is useful. No new provider issue/PR, reopening, or CLA action.

A [provider-maintained connection guide](../../integrations/local-openai-clients.md)
now covers all three applications using existing connectors. It includes
explicit ports, model discovery, placeholder-versus-real keys, Docker/remote
reachability, independent embedding configuration for RAG, and LiteLLM SDK/proxy
examples. Published on this task branch at commit `f1a2242ef`; not merged to
main or deployed to the website. The previous plan to seek a dedicated
AnythingLLM entry is superseded by its explicit contribution rules.

### New model-card PRs

| Checkpoint | PR | Evidence boundary |
| --- | --- | --- |
| mlx-community/Qwen3.5-4B-MLX-4bit | [#1](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit/discussions/1) | Historical M2 Pro text-decode record; that run does not pin a revision, while the current alias does |
| mlx-community/gemma-4-26b-a4b-it-4bit | [#4](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/discussions/4) | Historical text decode of exact revision on 32 GB M2 Pro; not HTTP/tools/multimodal qualification |
| mlx-community/Qwen3.6-35B-A3B-8bit | [#2](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit/discussions/2) | Historical six-prompt HTTP coherence gate of exact revision on 256 GB M3 Ultra; not vision/MTP/tool qualification |

All three are README-only append proposals. Original frontmatter, instructions,
licenses and model files remain unchanged. Each specifies text-only launch,
loopback binding/port, model discovery and a simple non-streaming chat request.
Affiliation, AI assistance, telemetry opt-out and lack of a fresh runtime run
are disclosed. Gemma has an existing broader model-card PR #3 from another
contributor; #4 explicitly acknowledges it and offers incorporation if preferred.
No existing Rapid-MLX card section or duplicate Rapid discussion was found.

Verified all three HF discussions have `is_pull_request=True`, `status=open`,
and unchanged main revision after creation. These are submissions, not accepted
model-owner endorsements. Checked aliases against catalog commit
`4d5ac49d9a5f82574db78df48d6965f05b2d1c51` and evidence against the 2026-08-21
matrix. CLI flags were checked using installed `rapid-mlx serve --help`;
Python, shell, YAML and request JSON syntax were checked for the connection
guide. No models were downloaded and no model/application inference ran.

Receiving roles: Echo for replies in the four existing threads; Harbor for
reviewing/integrating the provider-owned connection guide; Atlas for runtime
compatibility questions. Branch remains `echo/local-ai-submissions-20261106`.
Next action: run pinned-version end-to-end app checks before calling the recipes
verified integrations, and incorporate maintainer placement feedback. Respect
Open WebUI/AnythingLLM's contribution boundaries; do not reopen promotional
requests to increase submission counts. No monitoring has been scheduled.


## Owner correction: exclude mlx-community from outreach

The owner identified mlx-community as an unsuitable distribution venue because
of its connection to the competing mlx-vlm ecosystem. Honor that channel
preference without claiming exclusive ownership of the public organization.

Evidence checked on 2026-11-06:

- https://huggingface.co/mlx-community describes a community organization for
  mlx-lm, mlx-swift-examples, mlx-vlm and mlx-audio compatible weights. It does
  not establish exclusive mlx-vlm ownership.
- https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit/blob/main/README.md
  explicitly describes mlx-vlm conversion and points to Prince / Blaizzy.
- https://github.com/Blaizzy/mlx-vlm now redirects to
  https://github.com/Neywa-Labs/mlx-vlm.

Actions: closed our Qwen3.5 #1, Gemma #4 and Qwen3.6 #2 model-card PRs linked
above without merging or editing model files. Re-read each status as closed.
The public PR history remains visible. Do not reopen or replace these
submissions unless the owner explicitly changes this preference.

Echo handoff: prioritize original model authors, downstream agents/apps, and
our own rapid-mlx model organization; vet independent quantizers for ownership
and ecosystem relationships before further outreach. The LiquidAI-owned card
and other model-author submissions are outside this exclusion.

## Fifth pass: original model authors and downstream applications

The owner requested further outreach after excluding mlx-community. Echo
continued on the existing outreach branch, with no runtime code changes.

| Venue | Submission | Verified status |
| --- | --- | --- |
| LlamaIndex | [Documentation issue #24043](https://github.com/run-llama/llama_index/issues/24043) | Open; proposes an existing OpenAILike example, no new package |
| OpenBMB official MiniCPM5-2B MLX model | [Model-card PR #3](https://huggingface.co/openbmb/MiniCPM5-2B-MLX/discussions/3) | Open PR; main unchanged |
| Flowise Ideas | [Prepared proposal](flowise-local-chatflow-proposal.md) | Not published: GitHub CreateDiscussion returned FORBIDDEN for raullenchai |

LlamaIndex's current CONTRIBUTING accepts documentation but no new integration
packages. The issue follows its documentation template and uses existing
`OpenAILike` fields checked in source: `is_chat_model`, explicit context budget,
function-calling disabled initially, and structured output disabled initially.
It calls out independent embedding configuration for RAG. AI assistance and
lack of a fresh end-to-end run are explicit; no human review is falsely claimed.

OpenBMB's README links the official `openbmb/MiniCPM5-2B-MLX` model and an
existing MLX deployment cookbook. The card has no prior Rapid-MLX section or
submission. The new README-only PR preserves metadata and prior content,
adding the pinned `minicpm5-2b-4bit` path, loopback server, model discovery and
basic request. Evidence is the historical 2026-09-13 small-agent harness
report at Rapid commit `4d5ac49d9a5f82574db78df48d6965f05b2d1c51`:
exact checkpoint `8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3`, Rapid 0.14.1,
MLX 0.32.2 / mlx-lm 0.31.3 on M2 Pro 32 GB, enhanced arm 25/36 completions.
It explicitly does not certify physical 8 GB Macs or universal agent success.
This is an original model-author repository, not mlx-community.

Flowise CONTRIBUTING directs integration ideas to Discussions/Ideas; Show and
Tell expects an exported JSON flow plus screenshot. Inspected the existing
OpenAI Custom Model node: free-text model name, Base Path, credentials,
streaming and `checkDenyList` handling. The proposal is intentionally an idea,
not a tested/exported flow. GitHub rejected CreateDiscussion for permissions;
no discussion was created and no issue was opened to bypass that restriction.
The exact draft is retained for a permitted future submission.

Deduplication found smolagents issue #2918 and open PR #2175, and closed
Continue PR #11692; did not submit duplicates or add nudges. Searches checked
both Rapid-MLX and RapidMLX on the selected new targets. Mistral cookbook was
inspected but not submitted: a runnable, version-pinned Mistral-specific
example should precede that contribution.

Validation: Python AST, shell syntax and JSON payload checks passed. Model
alias and immutable evidence were read; no new model downloads or live-model
application tests. Both completed submissions disclose affiliation, telemetry
and opt-out, and AI assistance. Re-read LlamaIndex's open issue and HF's open
PR/main revision after submission.

Handoff to Echo: answer in the existing LlamaIndex/OpenBMB threads; wait for
placement feedback before implementation. Flowise submission requires an
account with permission to create discussions; do not repeat the rejected call
or switch to its bug tracker. Atlas owns any compatibility work. No automated
monitoring or acceptance commitment.

## Sixth pass: whole-catalog model-author coverage

The owner expanded outreach to official publishers of the entire supported
model catalog. Completed a [catalog-to-author inventory](model-author-outreach-2026-11-06.md)
and [304-row provenance CSV](model-owner-inventory-2026-11-06.csv) at source
commit `4d5ac49d9a5f82574db78df48d6965f05b2d1c51`. Main aliases: 251;
audio aliases: 53; combined distinct weight repositories: 273. Metadata for
all 273 resolved. Grouped into 51 publisher/model-family candidates with
explicit attribution uncertainty. This does not certify 304 independent models.

| Official destination | New submission | Verified status |
| --- | --- | --- |
| Prism ML Bonsai 2 2B model | [HF PR #1](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/discussions/1) | Open PR; main unchanged |
| IBM Granite 4.2 official MLX model | [HF discussion #1](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx/discussions/1) | Open discussion, not a PR |
| Microsoft PhiCookBook | [Issue #606](https://github.com/microsoft/PhiCookBook/issues/606) | Open documentation proposal |
| Hugging Face SmolLM | [Issue #155](https://github.com/huggingface/smollm/issues/155) | Open documentation proposal |

Prism's exact checkpoint has an existing bounded text/tool/image serving
qualification, so it received an executable README-only proposal. The other
three received explicitly unqualified documentation scope proposals. IBM's
GitHub repository permits only bugs/inconsistencies, so the proposal went to
its official model discussion rather than a feature issue in that tracker.
Phi's issue template was followed; no CLA was signed. SmolLM's model-deployment
proposal is separate from prior smolagents framework submissions.

All four disclose affiliation and AI assistance, default telemetry and opt-out.
No fresh inference, benchmark, model download or broad support promise. Re-read
all four statuses; verified Prism main remained at its previous revision.
Shell and JSON examples checked; catalog mapping counted/validated separately.

Receiving owner: Echo, with Atlas for model qualification and Harbor for docs.
Nine publisher groups now have campaign submissions/discussions; 42 candidate
groups remain pending owner-route/qualification review, and additional models
within contacted groups are not automatically covered. The inventory is
complete for the two catalog files, but all-publisher outreach is not complete.
Do not bulk-post identical proposals to remaining namespaces. Next action:
resolve each candidate's original author and accepted contribution route,
check existing submissions, then attach checkpoint-specific evidence before
sending a recipe. Retain the mlx-community exclusion.

## Additional original-model communities

Echo follow-up: 13 evidence discussions plus one README-only HF PR. All were
read back open after submission; no fresh inference or acceptance claimed.

- TokenRhythm: [discussion](https://huggingface.co/TokenRhythm/NeoHorse-1-9B/discussions/3).
- IFM: [discussion](https://huggingface.co/IFM/K2-Horizon-7B/discussions/13).
- meta-models: [discussion](https://huggingface.co/meta-models/Muse-Glimmer-30B/discussions/73).
- Nanbeige: [discussion](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/discussions/36).
- XHToken: [discussion](https://huggingface.co/XHToken/Spark-X2.6-4B/discussions/31).
- InternScience: [discussion](https://huggingface.co/InternScience/Sage-A1.5/discussions/10).
- HiDream-ai: [discussion](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev/discussions/2).
- stabilityai: [discussion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/261).
- black-forest-labs: [discussion](https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/234).
- deepseek-ai: [discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/78).
- inclusionAI: [discussion](https://huggingface.co/inclusionAI/Ling-3.0-tiny/discussions/19).
- zai-org: [discussion](https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/70).
- Pika-Labs: [discussion](https://huggingface.co/Pika-Labs/Pika-H3-Flex/discussions/4).
- ornith-ai: [model-card PR](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit/discussions/1).

Recovered existing AllenAI [Tmax report](https://huggingface.co/allenai/tmax-9b/discussions/2); no duplicate posted. Full owner table now records 24 contacted groups and 27 specific verification/route holds. No submission targeted mlx-community.

## Remaining 27 publisher groups: documentation-scope proposals

25 official HF model discussions and two GitHub documentation proposals were
published and read back open. All explicitly disclose source-only verification;
no new inference or test run was claimed. These are proposals, not 27 PRs.

- CohereLabs: [scope proposal](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/discussions/16).
- NousResearch: [scope proposal](https://huggingface.co/NousResearch/Hermes-4-70B/discussions/7).
- mistralai: [scope proposal](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/discussions/33).
- XiaomiMiMo: [scope proposal](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/discussions/16).
- openai: [scope proposal](https://huggingface.co/openai/gpt-oss-20b/discussions/277).
- MiniMaxAI: [scope proposal](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/discussions/71).
- Jackrong: [scope proposal](https://huggingface.co/Jackrong/MLX-Qwopus3.5-9B-v3-4bit/discussions/2).
- tencent: [scope proposal](https://huggingface.co/tencent/Hy3-preview/discussions/12).
- RecursiveSI: [scope proposal](https://huggingface.co/RecursiveSI/flint-int4/discussions/4).
- ai9stars: [scope proposal](https://huggingface.co/ai9stars/G9v3-39A5B/discussions/5).
- WeiboAI: [scope proposal](https://huggingface.co/WeiboAI/VibeThinker-1.5B/discussions/12).
- alibaba-pai: [scope proposal](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP/discussions/3).
- Wan-AI: [scope proposal](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/discussions/16).
- Lightricks: [scope proposal](https://huggingface.co/Lightricks/LTX-2.5/discussions/98).
- Tongyi-MAI: [scope proposal](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/168).
- ByteDance-Seed: [scope proposal](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/discussions/20).
- moonshotai: [scope proposal](https://huggingface.co/moonshotai/Kimi-K2.6/discussions/61).
- Hcompany: [scope proposal](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B/discussions/4).
- ResembleAI: [scope proposal](https://huggingface.co/ResembleAI/chatterbox-turbo/discussions/27).
- OpenMOSS-Team: [scope proposal](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/discussions/8).
- nari-labs: [scope proposal](https://huggingface.co/nari-labs/Dia-1.6B/discussions/76).
- SWivid: [scope proposal](https://huggingface.co/SWivid/F5-TTS/discussions/30).
- IndexTeam: [scope proposal](https://huggingface.co/IndexTeam/IndexTTS-1.5/discussions/5).
- FunAudioLLM: [scope proposal](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/discussions/19).
- FireRedTeam: [scope proposal](https://huggingface.co/FireRedTeam/FireRedASR2-AED/discussions/3).
- meta-llama: [documentation proposal](https://github.com/meta-llama/llama-cookbook/issues/1088).
- hexgrad: [documentation proposal](https://github.com/hexgrad/kokoro/issues/393).

Publisher-level outreach coverage is now 51/51 candidate groups. Checkpoint
qualification remains separate and is retained in the owner report. GitHub
issues used the authenticated CLI after the App connector lacked write scope.
No access gate, CLA or disabled HF discussion setting was changed.
