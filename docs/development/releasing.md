# Releasing rapid-mlx

This page documents the **end-to-end release flow** and the **safety nets** that catch the common failure modes.

The historical pain point: between v0.6.14 (2026-05-05) and v0.6.16, several PRs added 30+ new model aliases (`granite4-tiny`, `smollm3-3b`, `deepseek-v4-flash`, `qwen3.6-*`, etc), but no version was bumped — leaving brew/PyPI users with a stale `rapid-mlx models` list. The safety nets below are designed to make that exact failure impossible to repeat without explicit human override.

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

## Pre-release validation gauntlet (11 gates)

The full set of gates that block a release. Some run automatically in CI, others must be run locally on an M-series machine before pushing the bump commit.

| # | Gate | CI runner | M3-only | Catches |
|---|---|---|---|---|
| G1 | `make release-smoke` — clean-room install + import | `release-preflight.yml` (macOS-14) | — | dev mlx symbol drift (#408) |
| G2 | Codex review × 2 rounds on the PR | local | local | every PR-author bug class |
| G3 | CLI ↔ Config fidelity audit | `ci.yml` lint (ubuntu) | — | silent CLI flag drop (#400) |
| G4 | `make smoke` — lint + 4500+ unit tests | `ci.yml` test-matrix (linux subset) + `test-apple-silicon` (macOS-14) | full `make smoke` on local | parser/router regressions |
| G5 | `make stress` — 8 scenarios incl. tool storm | — | **M3** (needs cached model + live server) | concurrent-batching regressions |
| G6 | Live-server fix-path repro | — | **M3** (needs live server) | fix doesn't ship to the user-visible path |
| G7 | SDK integration (anthropic / pydantic_ai / smolagents) | — | **M3** (needs cached weights + PyPI deps) | router-level breakage that unit tests miss |
| G8 | Microbench changed code path | — | **M3** (needs perf baseline DB) | silent perf regressions |
| G9 | 10-sequential latency check | — | **M3** (needs live server) | KV-cache / hot-path regressions |
| G10 | MLX upstream cross-chip-family audit | `release-preflight.yml` advisory (macOS-14) | review the warnings | M5-style #404 landmines |
| G11 | Auto-routing escape-hatch registry | `release-preflight.yml` (macOS-14) + `ci.yml` test-apple-silicon | — | silent auto-detection failures (#393/#400/#404) |

### What CI covers automatically on the bump PR

When you open a PR titled `chore: bump version to X.Y.Z`, `release-preflight.yml` runs:

- **PF-1** — `scripts/validate_release_subject.py` checks the title matches the auto-release regex (catches the `(#NN)` trap).
- **Version coherence** — `pyproject.toml` version line must equal the version in the title.
- **G1** — `make release-smoke` on macOS-14.
- **G10** — `scripts/check_mlx_upstream_calls.py` scans the installed mlx-lm/mlx-vlm tree for module-scope calls into chip-family-sensitive APIs; **advisory only**, the release-er decides per finding.
- **G11** — `tests/test_no_mllm_flag.py` 33-test registry asserts every auto-routing helper exposes both force-on and force-off CLI flags.

The `preflight-summary` job aggregates them so the bump PR has a single required check.

### What you must run locally on M3 before pushing the bump commit

These need cached model weights and live MLX inference — they can't run on GitHub-hosted runners:

```bash
# In one shell — start the server with the alias the fix targets:
python3.12 -m vllm_mlx.cli serve qwen3.5-4b --port 8000

# In another shell, run all four:
make stress                          # G5 — 8-scenario stress test
# G6 — boot the fix-path-specific curl repro; see PR description
python3.12 tests/integrations/test_anthropic_sdk.py    # G7
python3.12 tests/integrations/test_pydantic_ai_full.py # G7
python3.12 tests/integrations/test_smolagents_full.py  # G7
# G8 — microbench the changed code path against the prior version
# G9 — 10 sequential requests, watch tok/s stability
```

Document each result in the bump PR description. If any M3-only gate is skipped, explain why (e.g. "G7: smolagents failures pre-existing on main, see <ref>").

### Gates with known pitfalls

| Pitfall | Memory ref | Mitigation |
|---|---|---|
| `(#NN)` squash suffix breaks regex | `release_squash_subject` | PF-1 + doc note above |
| `skip-version-bump` label needs PR close+reopen to refire | `gotcha_skip_version_bump_label` | Doc note in `version-check.yml`; user must close+reopen or push to refire `pull_request` event |
| Mutable GitHub Actions tags as supply-chain vector | `pr_merge_sop` §7 | `scripts/check_gha_pinning.py` (advisory in `ci.yml` lint pending pinning cleanup) |
| MLX upstream new module-scope calls (M5 #404) | `release_workflow` G10 | `scripts/check_mlx_upstream_calls.py` runs in `release-preflight.yml` G10 |
| Codex-skip rationalization on bump PRs ("feels like just a version bump") | `feedback_release_sop_third_offense` | G2 stays local — the 11-gate table above is the explicit override of "feels small" |

### Adding a new gate

When a release-time bug class recurs:

1. Write the gate as a pure-Python script under `scripts/` if it doesn't need MLX.
2. Add it to `release-preflight.yml` (a new job + the `preflight-summary` aggregator).
3. Add unit tests under `tests/test_<gate>.py` covering pass + fail + edge cases.
4. Append a row to the 11-gate table above with the catches/CI columns.
5. If the gate is M3-only, add it to the "must run locally" list with the exact command.
