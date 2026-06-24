# PR validation pipeline

Single-command merge-readiness gate for incoming PRs (especially
external contributions). Strict mode: any single step failure blocks
merge.

## Usage

```bash
# from the repo root
python3.12 -m scripts.pr_validate <PR#>

# verbose mode (more progress logging on stderr)
python3.12 -m scripts.pr_validate <PR#> -v

# stdout = markdown scorecard (paste into PR comment)
# stderr = progress logs
# exit 0 = MERGE-SAFE, exit 1 = DO NOT MERGE
```

## Pipeline

| # | step | gate | runtime |
|---|---|---|---|
| 0 | `fetch` | always (fail-fast) | ~3s |
| 0.5 | `test_plan_check` | always | <1s |
| 0.7 | `cl_description_quality` | always (skip via `PR_VALIDATE_SKIP_DESC=1`) | <1s |
| 0.75 | `supply_chain` | always | ~5s |
| 0.8 | `test_env_check` | always (auto-install opt-out: `PR_VALIDATE_NO_AUTO_INSTALL=1`) | <1s (≈10s if install runs) |
| 6 | `codex_review` | always (skip if codex CLI missing / not logged in) | 30–180s |
| 2 | `lint` | when diff has .py | ~3s |
| 3 | `targeted_tests` | when diff has .py | 30s–3min |
| 4 | `full_unit` | blast ≥ medium | ~25s |
| 5 | `stress_e2e_bench` | blast == high | 5–10min |

(The codex review goes near the front by design: get cheap critical
thinking *before* spending 10 minutes on tests. The two cheapest
description-quality gates run first so a bad title or empty body
fails in under a second without burning the codex budget.)

## Threat model (#275)

`pr_validate` itself executes inside a venv that has the project
installed. Two of its steps are capable of *executing PR-controlled
code on the validator host*:

1. **`test_env_check`** may run `pip install '.[test]'` from the PR's
   working tree to recover a missing pytest plugin. That install
   evaluates `pyproject.toml` (including build hooks like
   `[build-system].build-backend`) and any `setup.py` shim.
2. **`targeted_tests` / `full_unit`** run `pytest`, which evaluates
   `conftest.py` and any plugin entrypoints registered by deps
   declared in `pyproject.toml`.

The validator can't trust the PR's `pyproject.toml` until the
supply-chain scan has had a chance to flag it. Therefore the step
order is **non-negotiable**:

* `supply_chain` runs at index ≈ 0.75 — BEFORE any
  auto-installing step. Any modification to `pyproject.toml`,
  any repo-root `requirements*.txt` (the prefix matcher catches
  `requirements-dev.txt`, `requirements-test.txt`,
  `requirements-pin.txt`, … without enumeration), `setup.py`,
  `setup.cfg`, `conftest.py`, `.github/workflows/`, `Makefile`,
  `.pre-commit-config.yaml`, or `Formula/` from an external author
  is `[BLOCKING]`. Detection delegates to
  `_test_env.is_dep_declaration_file()` so the supply-chain step
  and `test_env_check` share one source of truth.
* `test_env_check` runs at index ≈ 0.8 — AFTER `supply_chain`.
  Its `pip install '.[test]'` (project-extras) fallback is gated:
  if the PR diff touches any dep-declaration file, the
  project-extras path is REFUSED.
* The `test_env_check` step instead **always tries trusted-pins
  first**: a hardcoded, version-pinned set (`TRUSTED_TEST_PINS` in
  `_test_env.py`) installed from PyPI with `pip install --isolated`.
  That install path is intentionally allowed even on dep-file PRs —
  it does not read the PR's working tree and the install target list
  is grep-able in `_test_env.py`, so a malicious `pyproject.toml`
  cannot influence what gets installed. The opt-out
  `PR_VALIDATE_NO_AUTO_INSTALL=1` disables BOTH stages (trusted-pins
  AND project-extras) for hardened CI sandboxes that must not mutate
  the host Python at all.

The invariant is locked in by
`tests/test_pr_validate_runner.py::test_supply_chain_runs_before_auto_installing_steps`.

If you add a future step that auto-installs anything, put it AFTER
`supply_chain` and gate it on `pr_touches_dep_files(ctx.files_changed)`
the same way `test_env_check` does.

## Verdict

Strict — any single `fail` or `error` blocks merge. `skip` is neutral.

The codex review uses [BLOCKING]/[NIT] tiering (see "Code Review
Philosophy" below): only `[BLOCKING]` findings fail the gate; `[NIT]`
findings surface in the scorecard so the author can decide.

## Code Review Philosophy (Google eng-practices)

The pipeline follows [Google's code-review standard](https://github.com/google/eng-practices/blob/master/review/reviewer/standard.md):

> "Reviewers should favor approving a CL once it is in a state where
> it definitely improves the overall code health, even if the CL
> isn't perfect."

Concretely we encode three principles:

1. **Tiered findings.** The codex prompt requires every finding to be
   prefixed `[BLOCKING]` (concrete bug, security issue, broken
   contract, false-positive test) or `[NIT]` (style, future-proofing,
   alternative naming). Only `[BLOCKING]` fails the gate. Default to
   `[NIT]` if unsure. Caps: at most 5 BLOCKING + 5 NIT per review,
   forcing the model to triage instead of pad. Untagged findings
   default to `[BLOCKING]` so a forgotten prefix can't silently
   downgrade a real bug.

2. **Convergence over perfectionism.** Without tiering, the reviewer
   spirals: every round surfaces new style preferences and the PR
   never merges. PR #467 hit this — 5 rounds, each producing fresh
   "could be more defensive" findings. The tiered prompt converges
   in 2–3 rounds because nits are visible but don't block. See the
   `codex_deepseek_convergence_asymmetry` knowledge note for why we
   replaced the previous DeepSeek backend with codex.

3. **Description quality is enforced, not advised.** The
   `cl_description_quality` step rejects PRs with empty bodies, bad
   titles (`fix bug`, `wip`, `update`, `tweaks`, …), and bodies with
   no rationale signal (no `## Why` heading, no `Closes #NNN`, no
   inline `Why:`). Google: "Should be informative enough that
   future code searchers don't have to read your CL." Override with
   `PR_VALIDATE_SKIP_DESC=1` for trivial dep-bumps; don't normalize.

## Blast radius

Computed from `files_changed`. See `context.py::HIGH_BLAST_PATHS` for
the gating list. The classification chooses which expensive steps run.

* **high** — touches scheduler / engine / cli / server / memory_cache
  / routes / pyproject.toml. Full battery.
* **medium** — touches `vllm_mlx/` or `tests/` but not the high-blast
  list. Skips stress.
* **low** — only docs / examples / README. Skips full_unit + stress.

## Adding a step

1. Write a module under `steps/` with a class extending `base.Step`.
2. Set `name`, `description`, override `run(ctx)` (and `should_run` if
   the step is conditional).
3. Import + insert in the `STEPS` list in `runner.py`.

The runner orders steps explicitly — no auto-discovery — so the
pipeline policy is grep-able from one file.

## Step details

### `fetch` (step 0)

Wraps `gh pr view --json` + `gh pr diff`. Saves the diff to
`<work_dir>/pr.diff`. Refuses CLOSED / MERGED / DIRTY (merge-conflict)
PRs by design — re-open or rebase first.

### `test_plan_check` (step 0.5)

Reads the PR body for a `## Test plan` checklist. If any item is
unchecked (`- [ ]`) the step fails — the author hasn't finished what
they said they'd do. Lesson from #427.

### `cl_description_quality` (step 0.7)

Cheap title + body hygiene gate built from
[Google's CL-descriptions guidance](https://github.com/google/eng-practices/blob/master/review/developer/cl-descriptions.md).
Three checks:

1. **Title**: not empty, ≥3 words after a conventional-commit prefix
   strip (`fix(routes):`, `feat:`, …), and not in the bad-pattern
   blacklist (`fix bug`, `wip`, `update`, `tweaks`, `cleanup`,
   `various changes`, …).
2. **Body exists**: empty body fails.
3. **Body has rationale**: at least one of — a `## Why` /
   `## Summary` / `## Rationale` / `## Motivation` / `## Background`
   heading, an inline `Why:` line, a `Closes #` / `Fixes #` / `Refs #`
   link, or a `because`-clause.

Override: `PR_VALIDATE_SKIP_DESC=1` for two-line dep-bumps where
rationale is genuinely overkill.

### `test_env_check` (step 0.8)

Verifies that the same Python interpreter `targeted_tests` and
`full_unit` will hand to pytest can actually import the plugins the
suite needs (chiefly `pytest_asyncio` — `pytest.ini` sets
`asyncio_mode = auto`, so without the plugin every `async def test_*`
fails at collection with "async def functions are not natively
supported").

When a plugin is missing, the step's recovery path runs in two
attempts:

1. **Trusted pins (always tried first).** Installs
   `TRUSTED_TEST_PINS` (a hardcoded, version-pinned set defined in
   `_test_env.py` — currently `pytest>=7,<9`,
   `pytest-asyncio>=0.21,<1`) from PyPI directly with
   `pip install --isolated`. This bypasses the PR's `pyproject.toml`
   entirely so a malicious PR cannot poison the validator's runtime
   (#275).
2. **Project extras (gated).** If the trusted-pins install doesn't
   resolve everything AND the PR's diff does NOT touch any
   dep-declaration file (`pyproject.toml`, `requirements*.txt`,
   `setup.py`, `setup.cfg`), falls back to `pip install '.[test]'`
   from the repo root. If the PR touches a dep file, the
   project-extras path is REFUSED — the step reports `fail` and the
   operator is asked to review the diff before installing manually.

If both attempts still leave a plugin missing, the step reports
`fail` with the missing-package list and the canonical recovery
command (`<interp> -m pip install '.[test]'`) so the operator can
fix it manually.

Closes #185 — the prior implementation had no env check at all,
which meant a host that had lost `pytest-asyncio` (typically after an
orchestrated `pip install --no-deps --force-reinstall .`) would crash
the `full_unit` step with 124+ "async functions not supported" pytest
errors that looked like regressions. The canonical test-deps live in
`pyproject.toml[project.optional-dependencies].test` — the step does
NOT maintain a hand-edited duplicate list.

Override: `PR_VALIDATE_NO_AUTO_INSTALL=1` disables the auto-recover
`pip install` (the step still detects + reports the missing packages,
it just won't mutate the host Python). Use this in CI sandboxes that
must keep the runner image read-only.

### `codex_review` (step 6, runs early)

Sends the diff to `codex exec` (OpenAI gpt-5.5) with the prompt
embedded as the `PROMPT_TEMPLATE` string constant in
`steps/codex_review.py`. The prompt requires `[BLOCKING]`/`[NIT]`
tiering on every finding (see "Code Review Philosophy" above). Only
`[BLOCKING]` findings fail the gate; `[NIT]`s surface in the
scorecard. Skips if `PR_VALIDATE_NO_CODEX=1` or the `codex` binary is
missing. For non-zero codex exits, stderr is matched against a
transient-backend marker list (network, auth, rate-limit, 5xx) — a
match skips (flaky LLM mustn't block PRs), anything else fails (a
malicious diff shouldn't be able to bypass review by inducing a
crash). See "Failure-mode classification" below for the full table.

Authentication: codex uses its own ChatGPT login at
`~/.codex/auth.json`. No env var is read here; the repo is public and
we explicitly do not want a fallback key in source.

Model is pinned to `gpt-5.5` via the `--model` flag so a change to
the caller's `~/.codex/config.toml` default cannot silently swap the
reviewer underneath the gate.

`PR_VALIDATE_NO_DEEPSEEK=1` is still honored (with a one-line stderr
deprecation notice) so CI/local workflows that pre-date the codex
swap don't unexpectedly re-enable the paid LLM review.

**Failure-mode classification.** A non-zero `codex exec` exit is
discriminated: stderr matching transient-backend patterns (network,
auth, rate-limit, 5xx) → `skip`; anything else → `fail`. This stops
a malicious PR from bypassing the review gate by inducing a content-
side codex crash.

**Sandbox-read residual risk.** Codex's `--sandbox read-only` is the
strictest mode the CLI exposes; absolute-path reads (e.g.
`~/.codex/auth.json`) cannot be defended against here without external
sandboxing. Mitigations in place: untrusted-input prompt fence,
no-tool-use rule with "last word" placement, `cwd=` empty tempdir,
no `--dangerously-bypass-approvals-and-sandbox`. Treating absolute-
path reads as an upstream limit is intentional; the codex review step
is "best effort" against that vector. See the module docstring in
`steps/codex_review.py` for the full threat model.

Replaces the previous DeepSeek V4 Pro step. See the
`codex_deepseek_convergence_asymmetry` knowledge note for why we
switched — DeepSeek is asymptotic across rounds; codex converges in a
small bounded number.

### `supply_chain` (step 0.75)

Runs BEFORE any auto-installing step so a malicious PR can't get its
build-hook executed inside the validator venv before the scan flags
it (see "Threat model" above, #275).

* Flags any modification to install hooks (`pyproject.toml`,
  `requirements*.txt`, `setup.py`, `setup.cfg`, `conftest.py`,
  CI workflows, Makefile, Homebrew tap, pre-commit config):
  BLOCKING for external authors, warning for collaborators.
* Greps added lines for suspicious patterns (`eval`, `exec`,
  `pickle.loads`, `subprocess(... shell=True)`, hardcoded URLs/IPs,
  large hex/base64 blobs).
* Runs `pip-audit` against any new dependencies declared in
  `pyproject.toml` / `requirements.txt`.

### `lint` (step 2)

`ruff check` + `ruff format --check` on the changed `.py` files only.

### `targeted_tests` (step 3)

Maps each changed `.py` to candidate test files (heuristic by
filename) and runs them. **Negative control**: if any fail, re-runs
the same set on a fresh `git worktree` of `main` to filter
pre-existing flakes. Real regressions = fail.

### `full_unit` (step 4)

`pytest tests/` minus integrations + event_loop. Gated on blast ≥
medium (low-blast PRs can't break runtime).

### `stress_e2e_bench` (step 5)

The heaviest. Gated on blast == high. For each model in
`golden_models.yaml` that fits machine RAM (highest-quality candidate
per family):

1. Boot a server on port 8451.
2. Run `scripts/stress_test.py` (8 stress scenarios).
3. Run each agent integration in the registry (matrix: m × n).
4. Run an inline bench (cold TTFT + warm TTFT + speedup).
5. Compare bench to `harness/baselines/bench-<model>.json` —
   regression > 5% = fail.

Skip with `PR_VALIDATE_NO_STRESS=1`.

## Artifacts

Every run writes to `/tmp/pr_validate/pr-<N>/`:

* `pr.diff` — full unified diff
* `lint-{check,format}.log` — ruff output
* `targeted-{pr,main}.log` — pytest output (PR + neg-control on main)
* `full-unit.log` — pytest full output
* `codex-{request,review,usage}.{txt,md,json}` — codex exec input + reply + token usage
* `supply-chain-scan.log` + `pip-audit.log` — supply-chain artifacts
* `server-<model>.log` — boot + lifespan log
* `stress-<model>.log` — stress test output
* `agent-<name>-<model>.log` — per-integration output
* `bench-<model>.json` — bench numbers (latest run)

## Roadmap

* GitHub Action wiring (cheap layers only — codex per-push has a
  rate-limited login token).
* License-drift check via `pip show <pkg>` against an allowlist.
* Diff-aware import-graph for `targeted_tests` (replace stem heuristic).
* Expand `golden_models.yaml` to the full family list once we have RAM
  budget for big-model boots.
