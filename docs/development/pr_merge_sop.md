# PR Merge SOP

The maintainer-side gauntlet that every PR — internal or external, AI-authored or human — passes through before merge to `main`.

## Why this doc exists

`main` auto-publishes to PyPI + Homebrew on any commit matching `chore: bump version to X.Y.Z` (see [`releasing.md`](releasing.md)). A bad PR landing on `main` is in users' `pip install` paths within minutes. The PR-validation pipeline (`scripts/pr_validate/`) catches the common cases; this SOP captures the judgment calls around it that aren't easily automated.

## Step 0 — Necessity check (before anything else)

**The single most important question, and the cheapest to ask.** Before reading the diff, before running validation, before pulling the branch:

> **What goes wrong for a real user if this PR doesn't merge?**

If you can't answer in one specific sentence — close the PR with thanks, don't merge it. "Increases test coverage" / "makes the code cleaner" / "good practice" / "future-proofs against possible refactor" are NOT answers. Acceptable answers look like:

- "Issue #X is open; this fixes the reported broken behavior for [user/agent doing Y]."
- "Bench shows N% TPS regression on model M; this restores it."
- "External CVE in dep X; this PR pins to the patched release."
- "Maintainer-approved exploration in #X; advances the spike."

This applies equally to **PRs you (the maintainer) authored yourself or via Claude**. Most of the gravity in this rule is on AI-authored PRs — agents over-generate refactor and coverage churn that costs real review time, real CI cycles, and real blast-radius risk while shipping zero user value. Be willing to close your own PR.

If the PR is necessary but small enough that the value is borderline against the cost (CI minutes, your review time, contributor's iteration time, blast radius), prefer:

- A code comment / TODO in the file noting the gap, instead of a separate PR
- Bundling the change into the next inevitable touch of the same area

## Step 1 — Pre-flight

- Read the PR description. If "what" or "why" is unclear, ask before touching anything.
- Confirm `git status` clean; branch rebased on latest `raullenchai/main`. Heavy divergence → ask the contributor to rebase first.
- **Identify blast radius** (this gates which later steps fire):
  - **Inference-touching** (`vllm_mlx/{engine,scheduler,parsers,routes,reasoning,tool_parsers,memory_cache}/`, `vllm_mlx/runtime/`, `vllm_mlx/agents/`) → all gates required, including `make check` and Anthropic-compat round-trip.
  - **Surface-touching** (CLI flags, alias registry, `pyproject.toml`) → version-bump check fires; `make check` skip OK if no behavior change in generation path.
  - **Dev-only** (bench scripts, dev tooling, CI workflows, docs, tests) → `make check` skip OK; full unit + lint still required.

- **Verify required PR-template fields** are filled:
  - Necessity field non-empty and concrete (not "improve quality").
  - AI-assistance disclosure honest (which files, which prompts; "fully human" or "Claude wrote test, I wrote impl" both fine — silence isn't).
  - "I can explain every line on demand" affirmation present.

  Missing fields aren't an auto-block, but they shift the burden — the maintainer reads more carefully. For external contributors, ask them to fill in before review.

## Step 2 — Multi-round adversarial review (codex)

Run codex review **iteratively until convergence**.

- A round produces findings prioritized: P0 (must fix), P1 (should fix), P2 (nit/style).
- **Every finding must be addressed.** Either fix it, or post a one-line dismissal in the PR thread explaining why the codex finding is incorrect / not worth acting on. Do not silently merge with open findings.
- **Convergence** = a round produces zero new P0 findings. Two consecutive convergent rounds is the gold standard; one round suffices for diffs ≤ ~50 lines.
- Typical: 2-4 rounds for a non-trivial PR. If round 5 still finds new P0s, the PR scope is too large — split it.

## Step 3 — Test coverage

- Every new behavior MUST have a new test. If a behavior is genuinely untestable, document why in the PR description (not just "hard to test").
- Diff-aware: every modified `vllm_mlx/foo.py` should have a corresponding modified `tests/test_foo*.py` in the same PR.
- **Test-must-fail-on-broken-code spot check.** For new tests on critical code paths (parsers, scheduler, security boundaries), the contributor or maintainer posts one-line evidence that the test catches a deliberate break:

  > `Mutation: removed line 47 of foo.py → test_foo_bar fails as expected.`

  This is the cheap manual version of mutation testing — closes the gap where Claude-written tests sometimes pass tautologically (assert-true-of-the-mock).

- Run the directly-affected test files first:

  ```bash
  python3.12 -m pytest tests/test_<scope>*.py -q --no-header
  ```

- New contract tests should pin **intent**, not implementation — write them so a refactor doesn't break them but a behavior regression does.

## Step 4 — Lint + format

```bash
ruff check <changed paths>
ruff format --check <changed paths>
```

Both must be clean. Do not use `--no-verify` to skip pre-commit hooks. If a hook fails, fix the underlying issue.

## Step 5 — Broader unit suite

```bash
python3.12 -m pytest tests/ \
  --ignore=tests/integrations \
  --ignore=tests/test_event_loop.py \
  --ignore=tests/test_mllm.py \
  --ignore=tests/test_mllm_cache.py \
  --ignore=tests/test_mllm_continuous_batching.py \
  --ignore=tests/test_video.py \
  -q --no-header --tb=line
```

The MLLM / video files need real Qwen3-VL weights and hang locally — the CI matrix covers them.

**Pre-existing flakes** must be **proven** pre-existing by running the test on clean main:

```bash
git stash && python3.12 -m pytest <flake> -q && git stash pop
```

Never assume — confirm. Document any confirmed pre-existing fails in the PR description.

## Step 6 — pr_validate (recommended for substantive PRs)

```bash
python3.12 -m scripts.pr_validate.runner <PR#> --verbose
```

Multi-step pipeline: `fetch → codex → supply_chain → lint → targeted_tests → full_unit → stress_e2e_bench`. See [`scripts/pr_validate/README.md`](../../scripts/pr_validate/README.md).

## Step 7 — Supply-chain audit

`pr_validate`'s `supply_chain` step covers the foundation: hook-file modifications, dependency CVEs (`pip-audit`), suspicious code patterns. **Review the warnings it surfaces — don't just check the green dot.**

Manual checks for the gaps the automated step doesn't cover today (tracked as follow-ups in #320):

- **License drift** — if any new direct dep was added, verify its license is in our compatible set (Apache-2.0, MIT, BSD-*, ISC, MPL-2.0). AGPL/SSPL into the Apache-2.0 tree forces a relicense and must be refused.
- **GitHub Actions SHA pinning** — if `.github/workflows/` changed, every `uses: x/y@<ref>` must be a 40-char SHA, not a tag. Mutable tags = supply-chain compromise vector (see Trivy 2026 incident).
- **Transitive dep tree** — if `pyproject.toml` deps changed (even a version bump), spot-check the resolved tree for new transitive packages. Release-time `pip-audit` in the bundle is currently the safety net; PR-time visibility is a known gap.

## Step 8 — Doctor harness `make check` / `make full` (gated)

Skip rule:

- **Don't touch inference code** → skip and **explicitly note** in PR description: "make check skipped — no inference-path changes".
- **Touch inference code** → run, even if it takes ~10 min:

  ```bash
  make check --model qwen3.5-4b   # smoke tier, ~3-5 min
  make full                        # broader, ~1-2 hr — only when changes affect generation correctness
  ```

The bar is **0 regressions vs the per-model baseline** (`harness/baselines/`). Pre-existing fails (Test 10 streaming usage, `<|im_end|>` leak, thinking-toggle on qwen3.5-4b) are documented; new fails block merge.

## Step 9 — Anthropic-compat round-trip (gated on parser/router PRs)

If the diff touches `vllm_mlx/parsers/`, `vllm_mlx/reasoning/`, `vllm_mlx/routes/anthropic.py`, or `vllm_mlx/routes/chat.py`:

```bash
# in one shell:
rapid-mlx serve qwen3.5-4b
# in another:
curl -s http://localhost:8000/anthropic/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-4b","max_tokens":64,"messages":[{"role":"user","content":"say hi"}]}'
```

Output must be a non-empty Anthropic-shaped response, no `!!!!!!` token-id-0 corruption, no streaming-think misroute. The `/anthropic` surface shares router-level code with `/v1/chat/completions` but diverges at the streaming-think router; multiple historical regressions (#288, #289) shipped with green OpenAI-compat smoke and broken `/anthropic`.

## Step 10 — CI gate

```bash
gh pr view <PR#> --repo raullenchai/Rapid-MLX --json mergeable,mergeStateStatus,statusCheckRollup
```

Wait for `MERGEABLE (CLEAN)`. All checks must be `SUCCESS`. Never merge `UNSTABLE` (flaky) or `BLOCKED`. If a CI step fails, investigate root cause — never re-run hoping it'll pass.

Required checks: `lint`, `type-check`, `version-check`, `test-matrix (3.10/3.11/3.12)`, `test-apple-silicon`, `tests`.

## Step 11 — Final PR description audit

Before merge, the PR description must accurately reflect actual current state:

- Test count matches `pytest --collect-only | tail -1`.
- Test plan checkboxes are honest (not aspirational).
- Out-of-scope follow-ups documented (so reviewers don't ask "why didn't you do X").
- All `[x]` boxes have evidence in the PR or comments.

## Step 12 — Merge

- **Squash-merge** for clean main history:

  ```bash
  gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash --delete-branch
  ```

- If version was bumped: verify `Auto-release on version bump` workflow triggers post-merge.
- If the squash subject contains `(#NN)` GitHub auto-suffix on a `chore: bump version to X.Y.Z` commit, override with `--subject` — the regex in `auto-release.yml` is strict.
- After merge, verify `git log raullenchai/main --oneline -1` shows your squash commit.

## Common pitfalls

- **"Tests pass on my branch" ≠ "no regression"** — always confirm pre-existing flakes on clean main, never assume.
- **Bench data unreliability** — `scripts/bench_suffix_decoding_integrated.py` needs the reliability gates from PR #284 (decode-time floor, TPS ceiling). Older bench data without `raw_runs` field is suspect.
- **Cache contamination** — disk-persisted prefix cache (`~/.cache/vllm-mlx/prefix_cache/`) can replay cached generations and pin TPS to bogus values. Bench tools must pass `--disable-prefix-cache`.
- **Hybrid models** (`is_hybrid=True`: Qwen3.5/3.6, Qwopus, Nemotron, Granite4) cannot use spec-decode / suffix-decode. Trust the gate.
- **Background processes block GPU** — orphaned `rapid-mlx serve` from prior sessions can hang pytest. `pkill -f "vllm_mlx.cli serve"` before benches.
- **Auto-deploy blast radius** — merging to main with version bump = instant PyPI + Homebrew release. External PR review must include the Step 7 supply-chain audit before merge.

## Tracked SOP improvements

The following items are agreed-good but not yet implemented; tracked in [#320](https://github.com/raullenchai/Rapid-MLX/issues/320):

- License-drift check in `scripts/pr_validate/steps/supply_chain.py` (the docstring claims it; the code doesn't).
- GitHub Actions SHA-pinning enforcement when workflows change.
- PR-time transitive-dep audit (currently only release-time).
- Per-PR install-size delta comment (`du -sh` site-packages diff vs main).
- Per-PR doctor-harness smoke tier as a required CI check, gated by an `inference-touching` auto-label.
- `claude-code-security-review` action on PRs touching auth / parsers / serialization paths.
- Quarterly "review-of-the-review" sampling (re-review 10 random merged PRs to score whether codex missed material issues).
