# Source PR CPU preflight

`RAPID_MLX_SOURCE_PREFLIGHT=true` opts ordinary same-repository engine PRs into
CPU preflight. The variable is off unless explicitly enabled after deployment.

## Gate placement

Preflight runs all three ordinary/headless unit shards on Python 3.11, merges
all three Linux coverage artifacts, and retains lint, type, engine contracts and
the MLX dependency guard. It does not select only a mapped subset of unit tests.

Python version compatibility, Apple execution, model smoke contracts and the
100% changed-line coverage requirement are enforced on the actual combined
merge candidate. Source preflight therefore defers source Apple and changed-line
jobs. Passing the source `tests` aggregate admits a PR to normal queue validation;
it does not qualify a full evidence record or permit merging a failed candidate.
The full evidence validator still requires nine CPU and five model identities,
Apple success and candidate coverage. Candidate, main and release policy are
unchanged by this source-only option.

Only diffs confined to `rapid_mlx/` and ordinary top-level `tests/test_*.py`
files qualify. Source controllers, CI tests, collection support, dependencies,
unknown paths, cross-product changes and mixed documentation retain the full
source route. Runtime hot paths can receive this CPU prefilter because their
combined candidate still requires full execution before merge. Forks, promoted
`train/*` or managed queue branches, merge groups and main retain full validation.
The existing mapped source canary takes precedence when its own exact-base
qualification succeeds; the two source routes are mutually exclusive.

## Deployment and rollback

1. Independently review the exact policy head and verify its normal required
   source checks. Policy changes themselves use full validation.
2. Merge through the normal full candidate and verify actual candidate job
   identities and coverage, rather than inferring success from skipped jobs.
3. Confirm current main qualification, then explicitly set the repository
   variable `RAPID_MLX_SOURCE_PREFLIGHT` to `true`.
4. Inspect the classifier output on a real eligible source PR: `source_preflight`
   is true, matrix is Python 3.11 shards 1–3, and aggregate output explicitly
   states that full candidate and changed-line coverage remain required.
5. Verify its actual combined candidate selects the nine CPU and five model
   jobs and passes the required Apple/coverage gates before actual merge.

Set the variable to `false` to restore full source routing on subsequent runs.
Already running preflights may complete, but their merge candidates remain full;
rollback does not require bypassing checks or cancelling unrelated runs.

Measure source creation, execution, queue admission, candidate creation and
actual merge separately. This removes duplicate source GPU/compatibility work;
it does not establish a per-PR time guarantee or remove serial queue waiting.
