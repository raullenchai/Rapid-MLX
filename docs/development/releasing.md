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

1. **Bump `pyproject.toml`** — change `version = "X.Y.Z"` to `X.Y.(Z+1)` (or minor / major as appropriate). Keep the change in its own commit:

   ```bash
   git checkout main
   git pull
   sed -i '' 's/^version = "0.6.15"/version = "0.6.16"/' pyproject.toml
   git add pyproject.toml
   git commit -m "chore: bump version to 0.6.16"
   git push raullenchai main
   ```

   The commit subject **must** match `chore: bump version to X.Y.Z` exactly — `auto-release.yml` parses it.

2. **`auto-release.yml` fires** (~30s) — verifies the commit, checks the tag doesn't already exist, builds a CHANGELOG from `git log <prev-tag>..HEAD`, creates the GitHub Release.

3. **`publish.yml` fires on `release: published`** (~3min) — builds sdist + wheel, uploads to PyPI (via the `pypi` deployment environment), polls PyPI until the version is queryable, computes the tarball SHA256, dispatches an `update-formula` event to `raullenchai/homebrew-rapid-mlx`.

4. **The tap repo's workflow** (in `homebrew-rapid-mlx`) updates `Formula/rapid-mlx.rb` `url` + `sha256` + commits.

5. **Verify**: `brew update && brew upgrade rapid-mlx` should pull in the new version.

The whole sequence is hands-off after step 1.

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
