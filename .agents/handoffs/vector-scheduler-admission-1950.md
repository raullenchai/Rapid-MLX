# Vector handoff — contention-aware prompt admission (#1950)

## 2026-09-07 — starting

- Start FYI fallback for Atlas, Pixel, Harbor, Echo, and ds0731 (Orca role
  messaging unavailable): Vector owns branch
  `vector/1950-contention-admission`, worktree
  `/private/tmp/vector-scheduler-admission-1950`, based on
  `origin/main@dcad4aa90` on Local Mac.
- Intention: add an opt-in scheduler admission policy that selects the
  shortest validated uncached prompt tail when prompt slots are contended,
  reducing interactive TTFT without changing the default FCFS behavior.
- Scope: scheduler config/CLI plumbing, prompt-slot admission, non-mutating
  validated-tail ranking, live-generator compatibility filtering,
  selection-count anti-starvation, metrics/tests, reproducible contention
  benchmark, and durable performance evidence.
- Non-goals: changing the default policy, preempting running requests,
  chunk-boundary yielding, modifying mlx-lm, activating the dormant external
  priority API, multimodal scheduling changes, release, or deployment.
- Private reference check: vLLM keeps FCFS as the default and isolates policy
  selection in its request queue; SGLang separates policy ranking from token /
  cache admission, computes prefix matches at selection time, and documents
  starvation risk for cache-locality ordering. Rapid will adapt the narrow
  principles: opt-in policy, validate current cost at grant time, keep
  execution/correctness gates authoritative, and bound displacement. It will
  not copy their GPU-specific token-budget or preemption machinery.
- Verification plan: unit/property tests for estimator purity and grant order;
  stop/sampler compatibility; cache invalidation fallback; max-deferrals;
  cancellation/capacity/default parity; matched-shape greedy output parity;
  real same-host contended TTFT and no-contention overhead; three rounds of
  scope-locked adversarial self-review; exact-head PR validation and managed
  queue. No Spark or reviewer sub-agent.

## 2026-09-07 — implementation and dogfood

- Implemented the opt-in `shortest_validated_tail` policy while leaving the
  default FCFS admission branch unchanged. The policy grants only real prompt
  and completion headroom, ranks a non-mutating cache-validated tail estimate,
  skips incompatible live-generator stop groups, and forces the oldest
  compatible request after eight pass-overs by default.
- Added both server-entrypoint flags, config validation, status counters,
  focused tests, and CLI documentation. No MLLM, request priority, running
  preemption, or prompt-chunk-yield behavior was added.
- Local M3 Ultra / 256 GB dogfood with cached Qwen3-0.6B-4bit, prefix cache
  disabled, prefill B=1, three 8,000-word prompts ahead of one short prompt:
  short-request TTFT median fell from 3.067 s to 1.206 s (2.54x faster, 60.7%
  lower; five waves per arm). Client cancellation after first SSE content also
  recovered all slots between waves. Reproduction and raw samples are in
  `docs/engineering/performance/scheduler-admission-1950.md`.
- Review round 1 found compatibility risk for lightweight downstream config /
  scheduler test doubles; non-critical reads now use FCFS/zero fallbacks while
  real `SchedulerConfig` remains strict. Targeted regression suite after the
  fix: 205 passed, 2 deselected.
- Review rounds 2-3 verified prompt-slot cleanup across cancellation, normal
  completion, cache recovery and fatal generation errors; stop-token cohort
  isolation; MTP width; CLI fidelity; default FCFS order; and byte-identical
  B=1 greedy output (`black, white, gray`) across both policies.
- Full non-external suite: 22,973 passed, 138 skipped, 25 deselected, 6 xfailed,
  1 xpassed, and one unrelated missing-optional-`mflux` failure in
  `test_packaged_bf16_uses_model_path_without_onload_quantization`. The exact
  same test fails at clean base `dcad4aa90` with the same
  `ModuleNotFoundError: mflux`; the temporary comparison worktree was removed.
- Repository-wide Ruff lint and format checks pass. The local mypy-budget
  runner is operationally blocked because the repository targets Python 3.10
  while the installed NumPy stub uses Python 3.12 `type` syntax; this produced
  no project diagnostic and will be rechecked by the PR validation environment.
- PR #3216 validation round 1 found one legitimate blocker: older mlx-lm flat
  response shapes do not expose prompt-promotion events, so the mirror could
  stay occupied. Fixed with a one-time, warning-backed fallback to historical
  FCFS when that legacy shape is observed; the configured policy remains
  visible but unsafe ranking is disabled. Focused post-fix regression run:
  94 passed, 2 deselected. The obsolete-head full-unit portion of validation
  was stopped after recording the blocker because the broader suite had already
  completed locally and exact-head validation must be rerun after this fix.
- PR validation round 2 found two in-scope correctness gaps and one test nit.
  The cost probe now validates cache shape, cached-token bounds, the exact
  current prompt suffix, and every observable cache-layer offset before using
  a warm-tail cost. Selection is now read-only; queue removal, deferral updates,
  and forced-grant accounting commit only after `BatchGenerator.insert*`
  returns a UID, so every failure preserves position and starvation history.
  The legacy-runtime regression now feeds a flat response through real
  `Scheduler.step()`. Post-fix cache/scheduler regression run: 226 passed,
  2 deselected.
- PR validation round 3 was MERGE-SAFE with two nits. Both were resolved:
  unusable/negative observable cache offsets now force cold-cost ranking, and
  scheduler stats expose both configured and effective policy so a legacy
  runtime fallback reports effective `fcfs`. Post-fix focused run: 178 passed,
  2 deselected.
- PR validation round 4 remained MERGE-SAFE with one final nit: permissive
  integer coercion could accept malformed boolean/float/string cache offsets.
  Offset validation now requires a non-boolean Python integer and tests each
  rejected shape.
- PR validation round 5 found the reversed batch-size edge: upstream currently
  coerces completion capacity up to prefill capacity, but the opt-in Rapid
  admission contract must honor the operator-configured completion limit.
  Effective capacity now caps directly at `completion_batch_size`, with a
  regression where prefill B=4 and completion B=2.
- GitHub type-check exposed two new dynamic-attribute diagnostics for admission
  deferrals (the local Python 3.12 environment could not run the repository
  Python 3.10-targeted mypy surface). Deferral state is now an explicit typed
  `Request` field with `init=False`, preserving constructor compatibility.
