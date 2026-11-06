# Local AI directory submissions, 2026-11-06

Owner: Echo (community outreach). Execution host: Studio.

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
