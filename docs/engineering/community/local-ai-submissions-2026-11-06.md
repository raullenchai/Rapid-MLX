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
