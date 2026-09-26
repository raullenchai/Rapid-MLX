# Local verification: from a diff or a Desktop report to evidence

`scripts/dev_verify.py` is the one documented command a coding agent runs
before opening a PR (or while repairing one locally). It turns either a
worktree diff or a Desktop issue-form area into a reviewable verification
plan, executes the existing checks, and writes commit-bound, machine-readable
evidence. It replaces command rediscovery; it does not replace the checks
themselves — every command it runs is an existing dev/CI verification.

## Usage

```bash
# Plan only — what would run for a Desktop report area?
uv run --extra dev python scripts/dev_verify.py --area "Settings" --plan

# Plan only — what would run for this worktree's diff against main?
uv run --extra dev python scripts/dev_verify.py --diff-base origin/main --plan

# Execute the plan and record evidence (default mode)
uv run --extra dev python scripts/dev_verify.py --area "Settings"
uv run --extra dev python scripts/dev_verify.py --diff-base origin/main

# From a changed-path file (same format the CI router accepts)
uv run --extra dev python scripts/dev_verify.py --paths-file /tmp/changed-paths.txt

# Pin a single journey (validates against journeys.yaml)
uv run --extra dev python scripts/dev_verify.py --journey settings-persistence

# Let the command build the Desktop app first (uses apps/rapid-mac/scripts/build.sh)
uv run --extra dev python scripts/dev_verify.py --area "Settings" --build
```

Run under `--extra dev` (the standard dev environment): the desktop contract
gates and engine unit tier invoke pytest, which the base environment does not
install.

Exactly one primary input is required: `--diff-base`, `--paths-file`,
`--area`, or `--journey`. `--journey` may additionally refine an area or diff.
`--plan` prints the plan and runs nothing; `--json` emits the same payload the
result artifact carries.

## What the plan contains

For every selected check: the command, why it was selected, its
prerequisites, and the expected evidence location. Selection is delegated to
the production routers, so local and CI agreement is structural:

- Lanes come from `scripts/classify_ci_changes.py` (engine / desktop /
  docs-only, fail-closed for unknown paths).
- Desktop GUI journeys come from `scripts/select_gui_flows.py` and
  `apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml`.
- Engine checks mirror the documented dev entry point (`scripts/dev_test.py`
  lint/unit tiers), which mirror the CI lint/unit jobs.
- Desktop checks mirror the mac CI contract gates: the GUI contract pytest
  files, the in-process `swift test` suites (full suite under the hang
  backstop for broad selections, targeted `--filter "Golden journey: …"`
  otherwise), the standalone Server/Chat contract scripts, and the bash GUI
  harness itself (`gui-golden-flows.sh --flow <name>`).

Issue areas yield **candidate** journeys — an issue-form choice localizes the
report, it does not prove a cause. Areas without full automated coverage
declare `coverage gaps` in the plan (for example, DMG code-signature
validation and latency measurement have no automated journey). The documented
handling for such an area: reproduce manually against the built app (the GUI
harness's fake-sidecar personas in `apps/rapid-mac/scripts/fake-rapid-mlx.sh`
keep that reproduction deterministic and weight-free), record the manual
evidence path in the PR, and never let the area's candidates imply the
gap is covered.

## Evidence

Each run writes a `verify-result.json` with the HEAD SHA, diff base and base
SHA, selection reasons, per-check command/status/exit code, and artifact
paths, under a git-ignored per-run directory:

```
rapid-dev-verify/<UTC-timestamp>-<head-short-sha>/
  verify-result.json
  logs/<check-id>.log
  journeys/<name>/result.json     # the GUI harness's own verdict
```

The GUI harness runs each journey in an isolated persona with the bundled
fake sidecar, so ordinary verification never downloads model weights; the
pytest selection excludes `slow`, `integration`, and `needle` via
`pytest.ini`.

## Fail-closed rules

- A missing or invalid diff, an unknown area, or an unknown journey is a
  usage error (exit 2) — never an empty plan that could read as green.
- Any check that fails, is blocked by a missing prerequisite (no built app,
  no `jq`, no swift toolchain), or is left unexecuted makes the overall
  result `fail` (exit 1). Blocked reasons name the recovery step (for
  example, rerun with `--build`).
- Plan-only results are recorded with `status: "plan-only"`; a plan is never
  a pass.
- A GUI journey passes only when the harness exits 0 **and** its own
  `result.json` reports `status: pass`.

## Contracts that keep the tables honest

`tests/test_dev_verify_contracts.py` enforces, in both directions:

- `AREA_JOURNEYS` keys equal the `area` dropdown options in
  `.github/ISSUE_TEMPLATE/desktop_bug.yml` exactly.
- Every mapped journey exists in `journeys.yaml` at a triage-usable tier.
- Every PR-tier harness journey is reachable from at least one area.
- The ten representative historical reports in
  `tests/fixtures/dev_verify/desktop_reports.yaml` route exactly as recorded,
  including one explicitly uncovered report.

When adding or renaming a Desktop area or journey, update the form, the
mapping, and the fixture together — the contract test fails on any drift.
