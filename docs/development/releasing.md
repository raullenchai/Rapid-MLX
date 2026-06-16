# Releasing rapid-mlx

This page documents the **end-to-end release flow** and the **safety nets** that catch the common failure modes.

The historical pain point: between v0.6.14 (2026-05-05) and v0.6.16, several PRs added 30+ new model aliases (`granite4-tiny-4bit`, `smollm3-3b-4bit`, `deepseek-v4-flash-8bit`, `qwen3.6-*`, etc), but no version was bumped — leaving brew/PyPI users with a stale `rapid-mlx models` list. The safety nets below are designed to make that exact failure impossible to repeat without explicit human override.

## Quick reference

| Trigger | What happens automatically |
|---|---|
| Push commit `chore: bump version to X.Y.Z` to `main` | `auto-release.yml` creates tag `vX.Y.Z` + GitHub Release |
| GitHub Release published | `publish.yml` builds → PyPI publish → dispatches Homebrew tap to bump formula |
| PR touches `aliases.json` / `model_auto_config.py` / `cli.py` / dep changes | `version-check.yml` requires same PR to bump `pyproject.toml` version (or set the `skip-version-bump` label) |

## Cutting a release

The full path from "I want to release" to "users on `brew upgrade` see the new version":

1. **Run the clean-room install smoke** (mandatory, ~30s):

   ```bash
   make release-smoke
   ```

   Builds the wheel from the working tree and installs it into a fresh
   venv with only PyPI deps, then imports every module the published
   entrypoints would import (`vllm_mlx`, `vllm_mlx.scheduler`,
   `vllm_mlx.server`, `vllm_mlx.cli`). Catches the failure mode that
   shipped in v0.6.53 (#408): code that imports cleanly on the dev
   machine because the dev mlx has a symbol that hasn't appeared in any
   released wheel yet. Every other gate (`make smoke/check/full`,
   `pr_validate`, codex review) runs against the dev mlx and is blind
   to this class of bug. **Do not push a version bump commit if this
   fails** — the failure indicates every `pip install` user will crash
   on import.

   Post-tag verification: `python3 scripts/release_smoke.py --version X.Y.Z`
   re-runs the gate against the wheel actually published to PyPI.

2. **Bump `pyproject.toml`** — change `version = "X.Y.Z"` to `X.Y.(Z+1)` (or minor / major as appropriate). Keep the change in its own commit:

   ```bash
   git checkout main
   git pull
   sed -i '' 's/^version = "0.6.15"/version = "0.6.16"/' pyproject.toml
   git add pyproject.toml
   git commit -m "chore: bump version to 0.6.16"
   git push raullenchai main
   ```

   The commit subject **must** match `chore: bump version to X.Y.Z` exactly — `auto-release.yml` parses it.

3. **`auto-release.yml` fires** (~30s) — verifies the commit, checks the tag doesn't already exist, builds a CHANGELOG from `git log <prev-tag>..HEAD`, creates the GitHub Release.

4. **`publish.yml` fires on `release: published`** (~3min) — builds sdist + wheel, uploads to PyPI (via the `pypi` deployment environment), polls PyPI until the version is queryable, computes the tarball SHA256, dispatches an `update-formula` event to `raullenchai/homebrew-rapid-mlx`.

5. **The tap repo's workflow** (in `homebrew-rapid-mlx`) updates `Formula/rapid-mlx.rb` `url` + `sha256` + commits.

6. **Verify**: `brew update && brew upgrade rapid-mlx` should pull in the new version.

The sequence is hands-off after step 2.

## Safety nets

### `version-check.yml` — block stale releases at PR time

Runs on PRs that modify any of:
- `vllm_mlx/aliases.json` — new model alias entries
- `vllm_mlx/model_auto_config.py` — new model profiles or capability flags
- `vllm_mlx/cli.py` — new flags or entrypoints
- `pyproject.toml` — new dependencies (matched by grep on `dependencies` / `optional-dependencies` / `requires`)

If those files changed but `pyproject.toml`'s `version` field didn't, the check fails with:

```
❌ User-facing change detected but pyproject.toml version is unchanged.
Files that triggered this check: ...
To fix: bump pyproject.toml — e.g. 0.6.15 → next patch.
To bypass (pure refactor, no user-visible change): add the
``skip-version-bump`` label to this PR.
```

**Bypass**: add the `skip-version-bump` label. Use this **only** for refactors that touch a watched file but don't change observable behaviour (e.g. moving a function inside `cli.py` without adding flags).

### `_version_check.py` — warn end users on stale local installs

`rapid-mlx models` (and any other entrypoint that calls `print_staleness_warning_if_any()`) prints a one-line warning when:
- installed version is `>= 2 patch` versions behind the latest GitHub release
- and the same major.minor (no cross-minor nag)
- and stderr is a TTY (no nag in pipes / CI)
- and `RAPID_MLX_DISABLE_VERSION_CHECK` isn't set

Cache: `~/.cache/rapid-mlx/version_check.json` (24h TTL). Network timeout: 2s. **Fail-silent on every error path** — staleness warnings must never break the CLI. See `tests/test_version_check.py` for the contract.

## Adding a new model

If your PR adds a model alias or profile, the version-check guard will require a version bump. The flow:

1. Add the entry to `vllm_mlx/aliases.json` and (if it has non-default capabilities) to `vllm_mlx/model_auto_config.py`.
2. Add tests as appropriate.
3. **Bump `pyproject.toml` version** in the same PR.
4. Optional but recommended: run the eligibility bench (see [issue #269](https://github.com/raullenchai/Rapid-MLX/issues/269)) and paste tier classification into the `ModelConfig` entry.
5. After merge, your bump-version commit triggers the auto-release pipeline.

## Manual override paths

Sometimes the auto pipeline isn't right. Escape hatches:

- **Skip the version-check guard for one PR**: add the `skip-version-bump` label.
- **Disable the staleness warning system-wide**: set `RAPID_MLX_DISABLE_VERSION_CHECK=1` in your shell profile.
- **Re-trigger a release** (e.g. PyPI publish failed mid-pipeline): create the GitHub Release manually from the existing tag — `publish.yml` will re-fire.
- **Skip auto-release entirely** (e.g. you want to bump version but not publish yet): use a different commit subject (`chore: prep 0.6.17` instead of `chore: bump version to 0.6.17`). `auto-release.yml` only matches the strict subject.

## Release commit message format

`auto-release.yml` is intentionally strict. Only this exact form triggers a release:

```
chore: bump version to X.Y.Z
```

— where `X.Y.Z` is three numeric components matching the new `pyproject.toml` version. Anything else (extra words, different prefix, dev suffixes) is silently ignored.

> **Squash-suffix trap.** GitHub's default squash-merge appends `(#NN)` to the subject. That suffix breaks the regex match and strands the version between commit-on-main and PyPI/Homebrew publish (recurring footgun — see `release_squash_subject` memory). Always pass `--subject` to `gh pr merge`:
>
> ```bash
> gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash \
>   --subject "chore: bump version to X.Y.Z" --delete-branch
> ```
>
> The `release-preflight.yml` workflow checks bump-PR titles against the same regex up-front; `scripts/validate_release_subject.py` is the structural belt-and-suspenders.

## Pre-release validation gauntlet

### The boundary

Every gate falls on one side of a single hard rule: **does the gate require running model inference (`rapid-mlx serve` + a real model load)?**

- **No** → CI runs it automatically (every PR or every bump PR)
- **Yes** → M3 local, manually, before pushing the bump commit

This is the rule. No exceptions. CI doesn't fake-inference with a tiny model on macOS-14's 7GB — the perf numbers would be meaningless and the structural coverage no better than unit tests. M3 doesn't re-run gates CI can run cleanly — those are cheap-and-cheerful in the cloud.

### Gate table

| # | Gate | Side | Where it runs | Catches |
|---|---|---|---|---|
| G1 | `make release-smoke` — clean-room install + import | CI | `release-preflight.yml` (macOS-14) | dev mlx symbol drift (#408) |
| G2 | Codex review × 2 rounds | local | maintainer machine | every PR-author bug class |
| G3 | CLI ↔ Config fidelity audit | CI | `ci.yml` lint (ubuntu) | silent CLI flag drop (#400) |
| G4 | unit suite (≈4500 tests) | CI | `ci.yml` test-matrix (linux) + test-apple-silicon (macOS-14) | parser/router regressions |
| G5 | `make stress` — 8 scenarios | **M3** | `make release-check-m3` | concurrent-batching regressions |
| G6 | Live-server fix-path repro | **M3** | `make release-check-m3` | fix doesn't ship to user-visible path |
| G7 | SDK integration (anthropic / pydantic_ai / smolagents) | **M3** | `make release-check-m3` | router-level breakage unit tests miss |
| G7b | Agent harness layer — Part A: `rapid-mlx bench <model> --tier harness` (single command, sweeps codex/opencode/hermes/aider/langchain Chat Completions); Part B: `/v1/responses` curl + SSE probe | **M3** | `make release-check-m3` | live-server harness regressions on Chat Completions (OpenCode tool-call parser, Hermes 62-tool stress, Codex profile shape, Aider streaming/text-edit format, LangChain 6-test suite incl. structured output) + Codex-only `/v1/responses` route regressions (the `AgentTestRunner` only knows Chat Completions, so the shim needs its own probe) |
| G8a | Parser microbench (×10k iters) | CI | `ci.yml` lint (ubuntu) | >10× parser regression |
| G8b | End-to-end perf bench (tok/s baseline) | **M3** | `make release-check-m3` | KV-cache / hot-path perf regressions |
| G9 | 10-sequential latency | **M3** | `make release-check-m3` | tok/s stability degradation |
| G10 | MLX upstream cross-chip-family audit | CI | `release-preflight.yml` advisory (macOS-14) | M5-style #404 landmines |
| G11 | Auto-routing escape-hatch registry | CI | `release-preflight.yml` (macOS-14) + ci.yml test-apple-silicon | silent auto-detection failures (#393/#400/#404) |
| PF-1 | Auto-release subject regex pre-check | CI | `release-preflight.yml` (ubuntu) | `(#NN)` squash suffix trap |

### CI coverage — what runs without you lifting a finger

**Every PR** → `pr-validate.yml` runs the `pr_validate` pipeline (7 of 9 steps; `stress_e2e_bench` and `full_unit` skipped because both need MLX/a live server which ubuntu-latest can't provide). The scorecard is posted as a PR comment so contributor + maintainer see the verdict without leaving the PR page. The skipped pair is covered on M3 by `make release-check-m3` at release time.

**Every bump PR** (title matches `chore: bump version to X.Y.Z`) → `release-preflight.yml` adds PF-1, G1, G10 (advisory), G11. The `preflight-summary` job aggregates them so the bump PR has a single required check.

**Every PR + push to main** → `ci.yml` runs lint (ruff + audit + GHA-pin advisory + parser microbench) + test-matrix (linux curated) + test-apple-silicon (macOS-14 mlx-importing tests).

### M3 local — one command before pushing the bump commit

```bash
make release-check-m3              # uses MODEL=qwen3.5-9b-4bit (default)
MODEL=qwen3.6-27b-4bit make release-check-m3   # override
```

Wrapped by [`scripts/release_check_m3.sh`](../../scripts/release_check_m3.sh). It boots `rapid-mlx serve` once on port 8000, then runs G5 (stress) + G7 (anthropic + pydantic_ai + smolagents) + G7b (agent harness layer: a single `rapid-mlx bench <model> --tier harness` sweep across codex / opencode / hermes / aider / langchain) + G6 (parallel-tool-call cap repro) + G9 (10-seq latency) + G8b (parser microbench, M3 perf baseline) sequentially. The server is killed on exit.

G7b covers the live-server harness path that `pr-validate`'s unit-level profile tests can't reach. Split in two parts so each is honestly scoped:

- **Part A** — `rapid-mlx agents codex / opencode / hermes / aider / langchain --test`. Smoke-tests `/v1/chat/completions` parser/router behavior for the five first-class harnesses. `AgentTestRunner` (`vllm_mlx/agents/testing.py`) only knows the Chat Completions endpoint today, so this part does **not** exercise `/v1/responses`.
- **Part B** — direct curl probes against `/v1/responses` (one non-stream, one SSE). Verifies the Codex-CLI shim added in v0.7.10 is reachable and emits at minimum `response.created` and `response.completed` in the right order. Part B is the only thing in the entire CI + M3 gauntlet that actually touches the Responses route at request time. If you change the route's event sequence, Part B is what catches it.

The remaining seven profiles (`goose`, `openhands`, `cline`, `openclaude`, `pydanticai`, `smolagents`, `generic`) are intentionally not in Part A:

- `goose` needs the Block Goose CLI on PATH — environmentally flaky for a release gate.
- `cline` is a VSCode extension with no CLI mode (`binary` and `query_cmd` both `null`) — `agents cline --test` would false-positive PASS on the API-level default plan without ever exercising the actual Cline workflow.
- `openhands` and `openclaude` are pure-interactive (`query_cmd: null`); same false-positive concern as `cline`.
- `pydanticai` and `smolagents` are already exercised via the G7 SDK block (`tests/integrations/test_pydantic_ai_full.py`, `test_smolagents_full.py`) which calls the libraries directly; running them again via `agents <name> --test` would duplicate coverage.
- `generic` is a fallback OpenAI-compatible config for any agent not covered by a dedicated profile — it doesn't have its own integration semantics to test.

Add a new profile to Part A when (a) the integration is core to a release, (b) `--test` runs without depending on an external CLI binary, and (c) the profile has a real `query_cmd` or `specific_tests` so the run actually exercises the agent workflow (not just the default API plan). Add a new Part-B probe when a new route surface lands that has no `AgentTestRunner` coverage.

Budget: ~15-20 minutes on M3 Ultra with weights warm-cached. Zero $. Default model is `qwen3.5-9b-4bit` — the practical floor for the multi-turn-tool tests (`pydantic_ai 5_multi_turn`, `opencode multi_turn_tool`). Smaller models (e.g. `qwen3.5-4b-4bit`) flake on those tests because the 2048-token per-test cap collides with thinking budget on a 4B model — fast but unreliable. For a higher-confidence release run, `MODEL=qwen3.6-35b-4bit make release-check-m3` matches the codex-CLI workhorse recommendation.

If any sub-gate fails, the script exits non-zero with the failure pinpointed. Don't push the bump commit until it's all green.

### Performance-only PRs

For PRs that are explicitly about perf changes (a kernel rewrite, a new fast path), the perf-side gates aren't optional — but they're also not standard PR-CI material because perf measurements on M1 ubuntu runners are meaningless. Convention: run `make release-check-m3` manually on the perf branch, paste the before/after numbers into the PR description, and link the run log. Maintainer reviews the numbers as part of Step 6 (pr_validate doesn't auto-fail; the human reads them).

### Gates with known pitfalls

| Pitfall | Memory ref | Mitigation |
|---|---|---|
| `(#NN)` squash suffix breaks regex | `release_squash_subject` | PF-1 |
| `skip-version-bump` label needs PR close+reopen to refire | `gotcha_skip_version_bump_label` | Doc note; close+reopen or push to refire `pull_request` event |
| Mutable GitHub Actions tags as supply-chain vector | `pr_merge_sop` §7 | `scripts/check_gha_pinning.py` (advisory pending pinning cleanup) |
| MLX upstream new module-scope calls (M5 #404) | `release_workflow` G10 | `scripts/check_mlx_upstream_calls.py` in `release-preflight.yml` |
| Codex-skip rationalization on bump PRs ("feels like just a version bump") | `feedback_release_sop_third_offense` | CI/M3 split — most skippable gates are now in CI, not in the human's hands |

### Adding a new gate

Decide first: does the gate require running real inference?

- **No** → CI:
  1. Write a pure-Python script under `scripts/`.
  2. Wire to `release-preflight.yml` (bump-PR-only) or `ci.yml` (every PR).
  3. Add unit tests under `tests/test_<gate>.py`.
  4. Append a row to the gate table above.

- **Yes** → M3:
  1. Add the gate logic to `scripts/release_check_m3.sh`.
  2. Update the gate table above.
  3. If the new gate replaces or subsumes a CI gate, remove the CI entry — duplication causes drift.
