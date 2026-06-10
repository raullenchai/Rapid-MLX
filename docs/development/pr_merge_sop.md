# PR Merge SOP

The maintainer-side gauntlet that every PR — internal or external, AI-authored or human — passes through before merge to `main`.

## Why this doc exists

`main` auto-publishes to PyPI + Homebrew on any commit matching `chore: bump version to X.Y.Z` (see [`releasing.md`](releasing.md)). A bad PR landing on `main` is in users' `pip install` paths within minutes. The PR-validation pipeline (`scripts/pr_validate/`) catches the common cases; this SOP captures the judgment calls around it that aren't easily automated.

## Step 0 — Necessity check (before anything else)

**The single most important question, and the cheapest to ask.** Before reading the diff, before running validation, before pulling the branch:

> **What goes wrong for a real user — or for the repo's day-to-day maintenance — if this PR doesn't merge?**

If you can't answer in one specific sentence — close the PR with thanks, don't merge it. Acceptable answers fall in two buckets:

**User-visible value** (the strong case):

- "Issue #X is open; this fixes the reported broken behavior for [user/agent doing Y]."
- "Bench shows N% TPS regression on model M; this restores it."
- "External CVE in dep X; this PR pins to the patched release."
- "Maintainer-approved exploration in #X; advances the spike."

**Concrete maintenance value** (the carveout, intentionally narrow):

- Typo / broken link / docs clarification that confuses real readers.
- Alias / metadata bookkeeping (`aliases.json`, `model_auto_config.py`) for a model someone wants to serve.
- Deleting dead code that's been stale ≥ 6 months and demonstrably has no callers.
- CI/tooling fixes whose absence is currently making maintainer toil.

**NOT acceptable on their own:** "increases test coverage", "makes the code cleaner", "good practice", "future-proofs against a possible refactor", "matches the pattern used in file X". If the PR fits one of these and ONLY one of these, close it.

This applies equally to **PRs you (the maintainer) authored yourself or via Claude**. Most of the gravity in this rule is on AI-authored PRs — agents over-generate refactor and coverage churn that costs real review time, real CI cycles, and real blast-radius risk while shipping zero user value. Be willing to close your own PR.

If the PR is necessary but the value is borderline against the cost (CI minutes, your review time, contributor's iteration time, blast radius), prefer:

- A code comment / TODO in the file noting the gap, instead of a separate PR
- Bundling the change into the next inevitable touch of the same area

### Exceptions: bot PRs, reverts, version bumps, hotfixes, embargoed security

Not every PR runs the full gauntlet. Skip rules:

| PR class | Step 0 | Steps 1-6 | Step 7 (supply chain) | Steps 8-9 | Steps 10-12 |
|---|---|---|---|---|---|
| Dependabot / version-bump bot | satisfied by the bump itself | codex single round on the diff | mandatory — read the dep CHANGELOG | unchanged | unchanged |
| `chore: bump version to X.Y.Z` (release) | satisfied by linking the commits being shipped | n/a (just the version bump) | bundle-level audit at release time, see [`releasing.md`](releasing.md) | unchanged | unchanged |
| Revert PR | must name the regression / commit being reverted | targeted tests for the affected area | n/a unless deps reverted | unchanged | unchanged |
| Hotfix to broken main | satisfied if a regression issue is open or being filed | targeted tests + lint only; full unit can be skipped if main itself is broken | unchanged | **only skip gates that are physically blocked by the broken-main condition** — if the touched surface is parser / router / inference, run Step 8 (doctor) and Step 9 (Anthropic-compat) targeted to the affected surface anyway, those are the *highest-value* gates for a hotfix | document each skipped gate inline in the PR; merge once unblocked gates pass; file follow-up issue for any deferred non-blocking gate |
| Embargoed security fix | filed under coordinated-disclosure process; PR opens against private fork | full gauntlet but in private | mandatory | mandatory | merge with disclosure window |

For first-time contributors learning the ropes: relax tone, not standards. Walk them through fixes instead of closing; that builds the contributor base.

## Step 1 — Pre-flight

- Read the PR description. If "what" or "why" is unclear, ask before touching anything.
- Confirm `git status` clean; branch rebased on latest `raullenchai/main`. Heavy divergence → ask the contributor to rebase first.
- **Identify blast radius** (this gates which later steps fire):
  - **Inference-touching** (`vllm_mlx/{engine,scheduler,parsers,routes,reasoning,tool_parsers,memory_cache}/`, `vllm_mlx/runtime/`, `vllm_mlx/agents/`) → all gates required, including `make check` and Anthropic-compat round-trip.
  - **Surface-touching** (CLI flags, alias registry, `pyproject.toml`) → version-bump check fires; `make check` skip OK if no behavior change in generation path.
  - **Dev-only** (bench scripts, dev tooling, CI workflows, docs, tests) → `make check` skip OK; full unit + lint still required.

- **Verify required PR-template fields** are filled. If any are missing, **request fill-in before review begins** — don't silently start reading the diff with the contributor unaware of the gap. Required fields:
  - Necessity field, non-empty and concrete (not "improve quality").
  - AI-assistance disclosure: which files were AI-touched, the AI's role (wrote / reviewed / suggested fix), and how the human verified the output. **Don't ask for prompt transcripts** — too invasive and not useful for review.
  - "I can explain every line on demand" affirmation. The standard is intent + risk + behavior of every non-generated change; for generated/boilerplate sections (lockfile churn, framework scaffold, vendored snippets) the contributor identifies them and explains how they were verified, not how each line works.

## Step 2 — Multi-round adversarial review (codex)

Run codex review **iteratively until convergence**.

- A round produces findings prioritized: P0 (must fix), P1 (should fix), P2 (nit/style).
- **Every finding must be addressed.** Either fix it, or post a dismissal in the PR thread. **Dismissal quality bar**:
  - **P0/P1 dismissals**: must include concrete evidence (a counter-example, a code reference, a test that proves the finding wrong, or a documented design decision link). "This is wrong" without evidence is not a valid dismissal.
  - **P2 dismissals**: a one-line rationale is fine.
  - When in doubt, fix rather than dismiss — the failure mode is dismissed-then-shipped-then-broken.
- **Convergence** = a round produces zero new P0 **and** zero new P1 findings. Open P2s must be either addressed or explicitly dismissed per the rule above; a P2 backlog doesn't block convergence by itself but must be resolved before merge. Two consecutive convergent rounds is the gold standard; one round suffices for diffs ≤ ~50 lines.
- Typical: 2-4 rounds for a non-trivial PR. If round 5 still finds new P0s, the PR scope is too large — split it.

## Step 3 — Test coverage

- Every new behavior MUST have a new test. If a behavior is genuinely untestable, document why in the PR description (not just "hard to test").
- Diff-aware: each behavior-changing production file should map to a named test file in the same PR, OR an explicit "no test because X" rationale. The naming heuristic `vllm_mlx/foo.py` → `tests/test_foo*.py` covers most cases but breaks down for shared fixtures, integration paths, and cross-parser tests — judgment over rule.
- **Test-must-fail-on-broken-code spot check.** For new tests on critical code paths (parsers, scheduler, security boundaries, serialization), evidence the test catches a deliberate break must be in the PR. The bar:
  - **Required content**: the exact mutation (one-line `sed`/`Edit` description, or a small diff hunk), the failing test name, and the failure assertion.
  - **NOT acceptable**: "I broke a return statement and the test failed" — that's an obvious mutation that says little about what the test actually asserts. The mutation should be against the *contract* the test is pinning, not against any random line.
  - **Maintainer reproduction**: for changes touching parsers, scheduler, or security boundaries, the maintainer reproduces the mutation locally before merge — don't trust it on faith for the high-blast-radius surfaces.

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

**Pre-existing flakes** must be **proven** pre-existing by running the test on clean main. The naive `git stash && pytest && git stash pop` pattern leaves work stashed if pytest fails — use a worktree:

```bash
git worktree add /tmp/main-check raullenchai/main
( cd /tmp/main-check && python3.12 -m pytest <flake> -q )
git worktree remove /tmp/main-check
```

The worktree is the only safe pattern — `trap`-based stash recovery in an interactive shell delays the pop until the shell exits, leaving the working tree in an unexpected state for an indeterminate time. Don't use it.

Never assume — confirm. Document any confirmed pre-existing fails in the PR description.

## Step 6 — pr_validate (recommended for substantive PRs)

```bash
python3.12 -m scripts.pr_validate.pr_validate <PR#> --verbose
```

Multi-step pipeline: `fetch → deepseek_review → supply_chain → lint → targeted_tests → full_unit → stress_e2e_bench`. See [`scripts/pr_validate/README.md`](../../scripts/pr_validate/README.md).

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
  # make check runs against the default model (qwen3.5-4b-4bit) — ~10 min
  make check
  # make full runs across multiple models (~1-2 hr) — only when changes affect generation correctness
  make full
  # to override the model, call doctor directly (the make targets don't pass --model through):
  python3 -m vllm_mlx.cli doctor check --model <alias>
  ```

The bar is **0 regressions vs the per-model baseline in `harness/baselines/`** *for models that have committed baselines* (currently `qwen3.5-35b-8bit` and `qwen3.6-35b-4bit`). For models without baselines, document the chosen ad-hoc reference (e.g., "compared against output on commit X", "manual eyeball vs main"). Pre-existing fails (Test 10 streaming usage, `<|im_end|>` leak, thinking-toggle on qwen3.5-4b-4bit) are documented; new fails block merge.

## Step 9 — Anthropic-compat round-trip (gated on parser/router PRs)

If the diff touches `vllm_mlx/parsers/`, `vllm_mlx/reasoning/`, `vllm_mlx/routes/anthropic.py`, or `vllm_mlx/routes/chat.py`:

```bash
# in one shell:
rapid-mlx serve qwen3.5-4b-4bit
# in another:
curl -s http://localhost:8000/anthropic/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-4b-4bit","max_tokens":64,"messages":[{"role":"user","content":"say hi"}]}'
```

Output must be a non-empty Anthropic-shaped response, no `!!!!!!` token-id-0 corruption, no streaming-think misroute. The `/anthropic` surface shares router-level code with `/v1/chat/completions` but diverges at the streaming-think router; multiple historical regressions (#288, #289) shipped with green OpenAI-compat smoke and broken `/anthropic`.

## Step 10 — CI gate

```bash
gh pr view <PR#> --repo raullenchai/Rapid-MLX --json mergeable,mergeStateStatus,statusCheckRollup
```

Wait for `MERGEABLE (CLEAN)`. All checks must be `SUCCESS`. Required checks: `lint`, `type-check`, `version-check`, `test-matrix (3.10/3.11/3.12)`, `test-apple-silicon`, `tests`.

**CI failure taxonomy** — different kinds of red are different problems:

| Failure | Diagnosis signal | Action |
|---|---|---|
| **Code failure** (test asserts, lint errors, type errors, build break) | Failure reproduces locally on the PR branch | Fix the code. Never re-run. |
| **Infra flake** (network timeout, runner crash, "lost connection to controller", external service 5xx) | Re-running same commit on same runner type produces a different result; failure mentions infra/network | OK to re-run **once** after pasting the failure log into the PR thread documenting the cause. Two consecutive infra fails = stop and investigate. |
| **Cancelled** (user / GitHub cancelled, or job killed by another workflow's timeout) | `cancelled` not `failure` in API | Re-run; document the cause if it happens repeatedly. |
| **Broken main** (every PR's CI is failing this check, including merges that already passed) | Same check failing on `main` itself or on multiple unrelated open PRs | Don't merge anything against the broken check until main is fixed. Open a separate hotfix PR for the broken check itself. |
| **Pre-existing flake on PR's affected file** | Failure also reproduces on clean main | Document in PR description with the proof command. Doesn't block merge. |

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

## CI coverage of these steps

The full `pr_validate` pipeline runs on every PR via `.github/workflows/pr-validate.yml` — the scorecard is posted as a PR comment so you can see verdicts without leaving the PR page. The table below maps each SOP step to its CI status:

| Step | CI coverage | Local-only | Notes |
|---|---|---|---|
| 0 — necessity | — | judgment | can't automate |
| 1 — pre-flight | `version-check.yml` blast-radius detection + `pr_validate.fetch` (in pr-validate.yml) | — | PR-template fields still need a human read |
| 2 — codex review | `pr_validate.codex_review` step skips on CI (no `~/.codex/auth.json`); humans run codex locally | maintainer | runs in the human's terminal; conclusions feed the PR thread |
| 3 — test coverage | `ci.yml` (existence of `tests/test_<scope>*.py` files) | mutation spot-check | mutation testing is the cheap manual step |
| 4 — lint + format | `ci.yml` lint job (ruff, ruff format, audit_cli_config_fidelity, gha-pinning advisory, parser microbench) | — | full coverage |
| 5 — broader unit suite | `ci.yml` test-matrix (linux-compat subset) + test-apple-silicon (mlx-dependent) | `pr_validate.full_unit` on M3 | CI covers the two surfaces it can; full tests/ tree runs on M3 |
| 6 — pr_validate | **pipeline** (7 of 9 steps) auto via `pr-validate.yml` | `stress_e2e_bench` + `full_unit` | both skipped steps need MLX / a live server; covered by `make release-check-m3` |
| 7 — supply chain | `pr_validate.supply_chain` (auto) + gha-pinning advisory | license drift + transitive deps still need a human read | partial automation; pip-audit is automated |
| 8 — doctor `make check` | — | **M3** (needs MLX + cached weights) | inference-touching PRs only |
| 9 — Anthropic-compat | — | **M3** (needs MLX + live server) | parser/router PRs only — covered by `make release-check-m3` |
| 10 — CI gate | `ci.yml` aggregation + `pr-validate.yml` scorecard | — | full coverage |
| 11 — PR description audit | `pr_validate.cl_description_quality` (auto in pr-validate.yml) | final read | automated rejects empty bodies and bad titles |
| 12 — merge | `auto-release.yml` regex match + `release-preflight.yml` PF-1 subject pre-check | — | strict subject enforced both PR-time and post-merge |

For release-time gates (the gauntlet that fires on bump PRs and the M3 manual checklist), see [`releasing.md` § "Pre-release validation gauntlet"](releasing.md#pre-release-validation-gauntlet).

### What "local-only" means now

After the CI build-out, the human-only surface on a typical PR is:

- Step 0 (necessity) — judgment call
- Step 2 (codex review) — runs in the maintainer's terminal; results posted to PR
- Step 3 mutation spot-check — quick manual mutation test for critical paths
- Step 7 partial — license + transitive dep eyeball when deps change
- Steps 8 + 9 — only for inference-touching PRs, via `make release-check-m3`

Everything else is automated. The `pr_validate` scorecard comment is the single source of truth for "is this PR mergeable?"

## Common pitfalls

- **"Tests pass on my branch" ≠ "no regression"** — always confirm pre-existing flakes on clean main, never assume.
- **Bench data unreliability** — `scripts/bench_suffix_decoding_integrated.py` needs the reliability gates from PR #284 (decode-time floor, TPS ceiling). Older bench data without `raw_runs` field is suspect.
- **Cache contamination** — disk-persisted prefix cache (`~/.cache/vllm-mlx/prefix_cache/`) can replay cached generations and pin TPS to bogus values. Bench tools must pass `--disable-prefix-cache`.
- **Hybrid models** (`is_hybrid=True`: Qwen3.5/3.6, Qwopus, Nemotron, Granite4) cannot use spec-decode / suffix-decode. Trust the gate.
- **Background processes block GPU** — orphaned `rapid-mlx serve` from prior sessions can hang pytest. `pkill -f "vllm_mlx.cli serve"` before benches.
- **Auto-deploy blast radius** — merging to main with version bump = instant PyPI + Homebrew release. External PR review must include the Step 7 supply-chain audit before merge.
- **Squash-suffix trap** — GitHub's default squash-merge appends `(#NN)` to the subject, breaking `auto-release.yml`'s regex. Always pass `--subject` to `gh pr merge` for bump PRs. `release-preflight.yml` PF-1 catches this pre-merge.
- **`skip-version-bump` label refire** — adding the label after `version-check.yml` has already failed does NOT auto-re-run the workflow (the label-add event isn't a `pull_request` event the bypass step listens to). Either close+reopen the PR or push a commit to refire. Memory: `gotcha_skip_version_bump_label_after_run`.
- **A/B classify validation-surfaced bugs** — when `pr_validate` or codex surfaces a bug, replay against base/main before deciding it's PR-introduced. Pre-existing bugs are follow-up issues, not PR scope creep. Memory: `feedback_pr_scope_ab_classify_first`.
- **Codex+DeepSeek convergence asymmetry** — codex converges in ~9 rounds, DeepSeek is asymptotic. Run codex to convergence, then ONE DeepSeek round. Memory: `codex_deepseek_convergence_asymmetry`.
- **Pre-existing pre-existing flake confirmation** — use a worktree, not `git stash`. The stash pattern leaves work stashed if pytest crashes mid-run.

## Tracked SOP improvements

The following items are agreed-good but not yet implemented; tracked in [#320](https://github.com/raullenchai/Rapid-MLX/issues/320):

- License-drift check in `scripts/pr_validate/steps/supply_chain.py` (the docstring claims it; the code doesn't).
- GitHub Actions SHA-pinning enforcement when workflows change.
- PR-time transitive-dep audit (currently only release-time).
- Per-PR install-size delta comment (`du -sh` site-packages diff vs main).
- Per-PR doctor-harness smoke tier as a required CI check, gated by an `inference-touching` auto-label.
- `claude-code-security-review` action on PRs touching auth / parsers / serialization paths.
- Quarterly "review-of-the-review" sampling (re-review 10 random merged PRs to score whether codex missed material issues).
