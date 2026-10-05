# Source-only CI canary

Tracking: #2252. This canary reduces ordinary source PR checks for exact reviewed
CLI help/banner and telemetry registry/chip paths. The mapping is in
`scripts/classify_ci_changes.py`; tests, shared support, dependencies, controls,
unknown paths and mixed unmapped changes otherwise retain full validation.

`RAPID_MLX_SOURCE_CANARY=true` is an opt-in repository Actions variable. Missing
or any other value broadens source checks to the full matrix. Enable only after
this workflow's full integration candidate has passed and merged. The latest
main-push CI run on the exact PR base must also be completed successfully. A
missing, pending, failed or cancelled qualification, or an API failure, falls
back to full checks without advancing a reduced chain.

Mapped source checks run all mapped CPU regressions, reject empty/skipped/error
JUnit proof, and enforce 100% changed executable-line coverage. Universal lint,
engine contracts, type checks and dependency-bound guards remain required.
Artifacts named `source-canary-unit-<sha>` describe `mapped-cpu-only` scope,
selected suites, source/base SHAs and the qualifying base run. They cannot be
consumed as `queue-tree-evidence` or release qualification.

Every promoted train/queue candidate remains full: nine CPU jobs, Apple checks,
coverage and relevant model checks. Main keeps full validation or authenticated
identical-tree full candidate reuse. Exact release qualification remains
unchanged. There is no reduced candidate route, batching, qualification
coalescing, new model download policy or deferred nightly coverage claim.

Rollback: set `RAPID_MLX_SOURCE_CANARY=false` (or delete it), then rerun only an
incomplete/failed source check when needed. Already-qualified candidates are
unchanged. Unknown or broadened scope never uses mapped evidence as a full pass.

Measure actual source job duration, runner minutes, eligible path sets and
source-to-candidate/merge times after activation. Local mapped-test timing is not
a hosted CI or merge-latency benchmark. Expand the exact map only through scoped
regression tests and independent review; an allowlist never grants permission
to move critical behavior into an otherwise low-risk file.
