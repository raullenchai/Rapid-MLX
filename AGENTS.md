# Contributing with coding agents

Rapid-MLX is a fast local LLM inference engine for Apple Silicon, with an
OpenAI- and Anthropic-compatible server, a CLI, and a native Swift Mac app.
AI-assisted and AI-written PRs are welcome: see
[Contributing: AI-written PRs welcome](README.md#contributing-ai-written-prs-welcome)
and [CONTRIBUTING.md](CONTRIBUTING.md). The same review standard applies to
all contributions: a worthwhile change, a focused diff, and evidence it works.

## Start here

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the relevant source and tests.
2. Check existing open PRs and issues before starting; avoid duplicate work.
3. Start a task branch from `main`, preferably in your own worktree. Keep one
   task per PR and preserve any existing local changes.
4. Before editing any non-test code, write down a short plan: problem, scope
   (files you expect to touch), non-goals, acceptance criteria, and the tests
   you will run. If you are working with a human, show it to them first. If
   the plan grows past the stated scope, or touches public APIs, defaults, or
   architecture, stop and ask a maintainer in the issue instead of expanding it.
5. Reproduce bugs first; implement the smallest coherent fix, verify it, and
   open a PR against `main` using the PR body skeleton at the end of this file.

## Repository map

| Path | What lives here |
| --- | --- |
| `rapid_mlx/` | Python engine, server, CLI, and supporting runtime code. |
| `rapid_mlx/cli.py`, `rapid_mlx/cli_parser.py` | CLI dispatch and argument definitions. |
| `rapid_mlx/server.py`, `rapid_mlx/routes/` | Server setup and HTTP endpoints, including chat and Anthropic routes. |
| `rapid_mlx/engine/`, `rapid_mlx/engine_core.py` | Engine interfaces, batching, and core execution. |
| `rapid_mlx/models/`, `rapid_mlx/patches/` | Model implementations and compatibility patches. |
| `rapid_mlx/tool_parsers/`, `rapid_mlx/reasoning/` | Tool-call and reasoning parsers. |
| `rapid_mlx/aliases.json`, `rapid_mlx/model_auto_config.py` | Model aliases/profiles and parser/capability auto-detection. |
| `rapid_mlx/cache/`, `rapid_mlx/kernels/`, `rapid_mlx/spec_decode/` | Cache support, optimized kernels, and speculative decoding. |
| `rapid_mlx/audio/`, `rapid_mlx/image/`, `rapid_mlx/video/` | Media inference integrations. |
| `rapid_mlx/telemetry/` | Telemetry event registry, validation, consent, and transport. |
| `apps/rapid-mac/` | Native Swift Desktop app, Swift package, tests, and packaging scripts. |
| `tests/` | Python regression, contract, integration, and model-dependent tests. |
| `docs/` | Guides, development docs, architecture, operations, and performance evidence. |
| `scripts/` | Developer checks, PR validation, qualification, benchmarks, and release tooling. |
| `bench/` | Development-only microbenchmarks and diagnostic reproductions; see its README. |
| `harness/` | Reviewed performance baselines and their schema; see its README. |
| `config/` | CI requirements, type-check budgets, and other shared configuration. |
| `examples/` | Client and integration examples. |
| `proto/` | Versioned schemas and wire contracts. |
| `vllm_mlx/` | Deprecated import compatibility shim; add new code under `rapid_mlx/`. |

## Setup

Inference requires macOS with Apple Silicon. Use Python 3.12 for the documented
contributor setup (`pyproject.toml` requires Python 3.12+). Desktop builds need
a recent Swift toolchain and macOS 14+. Some no-MLX tests run on other hosts;
that does not validate Metal inference.

From the repository root, with `uv` on your `PATH`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e ".[test]" ruff
```

Optional server smoke (downloads weights if unavailable; allow enough RAM/disk):

```bash
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

## Verify your change

Run commands from the root with the virtual environment active unless noted.
Choose checks for the affected behavior; report hardware or tool limitations
and skipped checks honestly.

```bash
# Quick local check: lint + CLI/config audit + selected unit suite.
make smoke

# Target a regression file; substitute the file for your change.
python3 -m pytest tests/test_tool_calling.py -v

# Broader unit run from CONTRIBUTING.md.
python3 -m pytest tests/ -x -q

# Lint and format checks.
ruff check .
ruff format --check .
```

`make smoke` delegates to `scripts/dev_test.py`; it is a useful first check,
but excludes some test files. Run affected tests explicitly as well.
`pytest.ini` excludes `slow`, `integration`, and `needle` tests by default;
MLX-dependent tests can skip without MLX. Live-server and real-weight checks
need their documented environment (notably `tests/test_event_loop.py`).
A passing default suite is not evidence of model qualification.

After opening a PR, replace `<PR#>` with its numeric GitHub PR number:

```bash
PR_VALIDATE_NO_CODEX=1 PR_VALIDATE_NO_STRESS=1 \
    python3 -m scripts.pr_validate.pr_validate <PR#>
```

This is the contributor fast path from CONTRIBUTING.md. It needs the `gh` CLI
authenticated to GitHub. It reads the PR's metadata and diff from GitHub but
runs tests and lint against your current checkout, so push first and run it
from a clean checkout of the exact PR head.
It omits automated Codex review and live-model stress; disclose those omissions.
Read the scorecard and fix failures; skipping a required check is not a pass.
The canonical stage/gating reference is
[scripts/pr_validate/README.md](scripts/pr_validate/README.md).

For Desktop changes, run inside `apps/rapid-mac/`:

```bash
swift build
swift test --no-parallel
```

These match the Desktop CI build/test commands. See the
[Desktop README](apps/rapid-mac/README.md) for app bundling and UI smoke checks.
Consider keyboard navigation, accessibility, cancellation, loading/error states,
and relaunch behavior when changing a user journey.

## Pick a task

Start with [good first issues](https://github.com/raullenchai/Rapid-MLX/labels/good%20first%20issue),
then [other open issues](https://github.com/raullenchai/Rapid-MLX/issues).
Check [open PRs](https://github.com/raullenchai/Rapid-MLX/pulls) and related issues
before implementing. Prefer a reported user problem or concrete maintenance
value over speculative cleanup. Useful recurring contributions include:

- **Model aliases and qualification.** Start at `rapid_mlx/aliases.json` and
  [How to Add a Model Alias](CONTRIBUTING.md#how-to-add-a-model-alias).
  Follow existing profile shapes; use `rapid_mlx/model_auto_config.py` for
  non-default capabilities. Add appropriate regression coverage and run
  `tests/test_aliases_contract.py`, `tests/test_model_aliases.py`, and
  `tests/test_model_auto_config.py` with targeted pytest. Alias bookkeeping
  alone needs no download; claiming model support/performance needs evidence.
  Follow [Adding a new model](docs/development/releasing.md#adding-a-new-model)
  and [model selection guidance](docs/development/contributing.md#choosing-a-model-for-a-test).
  For a SuffixDecoding tier, follow
  [eligibility qualification](docs/suffix_decoding_eligibility.md) and run
  `python3 scripts/bench_suffix_decoding_integrated.py --model <id>`.
  Record the measured tier/speedup and evidence; do not invent capability or
  performance metadata, change defaults without evidence, or bump versions.
- **Tool-call/reasoning parsers.** Start in `rapid_mlx/tool_parsers/` and
  `rapid_mlx/reasoning/`, with detection in `rapid_mlx/model_auto_config.py`.
  Port/adapt established vLLM or SGLang reference implementations rather than
  adding ad hoc output filters. Cover complete and incremental streaming
  output, partial delimiters, malformed input, ordinary prose, and tool-call
  boundaries. Look at `tests/test_tool_parsers.py`,
  `tests/test_tool_parser_wire_formats.py`, and `tests/test_reasoning_parser.py`.
- **Bug reproductions.** Turn a report into a minimal failing test under
  `tests/`, then fix the root cause. Record the commit, model/configuration,
  expected behavior, actual behavior, and reproduction command. A reproducible
  failing test is useful even if the eventual fix needs maintainer help.
- **Docs and translations.** Clarify setup or behavior in `docs/`; Desktop
  translations live in
  `apps/rapid-mac/Sources/Rapid/Resources/Localizable.xcstrings`.
  Follow existing keys/placeholders and check the rendered text or affected UI.
- **Benchmarks.** Start with [benchmark commands](docs/benchmarks/README.md),
  [microbenchmarks](bench/README.md), and [baseline methodology](harness/README.md).
  For example, run `rapid-mlx bench qwen3.5-4b-4bit` on Apple Silicon.
  Report the exact command, Rapid-MLX commit, chip/RAM, macOS/Python/MLX versions,
  model ID/revision/quantization, prompt/output lengths, concurrency, cache
  state, repetitions, and competing workloads. Compare before/after on the
  same machine and workload; retain raw results and summarize variability.
  Do not replace a reviewed baseline merely to make a regression pass.

## Engineering rules

- Search existing code and dependencies before inventing a mechanism. Reuse
  established patterns and inspect analogous implementations for edge cases.
- For engine/server work, inspect both vLLM and SGLang first; then MLX-LM,
  MLX-VLM, and other MLX-native implementations for platform constraints.
  Document concrete reasons when an established approach cannot fit.
  Respect licenses, provenance, and required notices when adapting code.
- Keep one task per branch/PR. Add regression tests that prove the failure and
  important boundaries. For server changes, consider stream/non-stream parity,
  cancellation, truncation, cleanup, and compatibility across affected routes.
- For import-time (monkey)patches: install from a module on the real
  `rapid-mlx serve` load path, and add a test that runs a subprocess which
  imports that production module first, then asserts the patch's
  `is_installed` state and its marker on the patched upstream object. Copy
  `test_install_fires_on_real_serve_import_path` in
  `tests/test_qwen3_5_norm_shift.py`. Importing the patch directly inside the
  pytest worker can install it as a side effect and hide broken wiring.
- Review against the stated PR intention and exact diff. Resolve or explicitly
  disposition findings before requesting another round. File unrelated work
  separately; agree on material scope changes with maintainers.
- Put durable behavior/setup knowledge in docs and regression knowledge in
  tests. Distill findings instead of committing raw agent transcripts.
- Never commit credentials, tokens, private URLs, user data, or machine secrets.
  Production releases/deploys and destructive operations need explicit human
  authorization and a recovery plan for destructive changes.
- Preserve telemetry privacy: follow the
  [privacy policy](apps/rapid-mac/PRIVACY.md), consent/opt-out controls, and
  `rapid_mlx/telemetry/events.json` registry. Keep telemetry metadata-only;
  do not add prompts, responses, attachments, secrets, or arbitrary paths.
  Retain strict validation and run affected telemetry tests when changing it.

## Avoid these changes

- Do not hand-edit generated artifacts. Model-management diagrams and generated
  README sections come from `docs/engineering/architecture/model-management/architecture.yaml`
  via `scripts/render_model_management_architecture.py`.
  Regenerate the Desktop `apps/rapid-mac/scripts/sidecar-requirements.lock`
  using [the dependency procedure](docs/engineering/operations/sidecar-dependencies.md).
  Preserve vendored source shape and license/provenance files.
- Do not bump versions or edit release notes in ordinary contribution PRs.
  Releases use dedicated maintainer-coordinated PRs.
- Do not add dependencies without prior discussion.
- Do not change CI workflows or secrets unless the task is explicitly CI work.
- Do not bundle unrelated formatting, generated output, models, caches, or
  benchmark artifacts into a fix.

## Before opening or updating a PR

- Do not name other projects as the inspiration or source of a change in
  commit messages, code comments, or PR titles/bodies. Describe Rapid-MLX's
  requirement, behavior, evidence, and user impact instead. The only
  exception is attribution that a license requires for adapted code; keep
  those notices and dependency disclosures intact.
- Use this PR body skeleton (it mirrors
  [the PR template](.github/PULL_REQUEST_TEMPLATE.md)). Why, Non-goals, and AI
  assistance disclosure are required; reviews wait until they are filled.

  ```markdown
  ## Why
  Fixes #<issue> — <the user-visible problem or concrete maintenance value>.

  ## Scope
  <what changes: behavior, files, surfaces>

  ## Non-goals
  <what this PR deliberately does not do; "none" is fine for tiny changes>

  ## Acceptance
  <observable outcomes a reviewer can check>

  ## Verification
  - `<command>` → <result>
  - Skipped: <check> (<reason, e.g. no Apple Silicon>)

  ## Behaviour delta
  <before → after for a changed default or policy; delete this section otherwise>

  ## AI assistance disclosure
  <which files AI touched, what it did, how you verified it; no transcripts>
  ```
- Paste commands and concise verification output/scorecards; identify failures,
  skipped checks, and environment limits instead of claiming unrun checks.
- Review the final diff for regressions, secrets, generated churn, and unrelated
  changes. Keep the issue link and acceptance criteria current after fixes.
- Be ready to explain the intent, risks, and behavior of every non-generated
  change, and how generated sections were produced and verified.
