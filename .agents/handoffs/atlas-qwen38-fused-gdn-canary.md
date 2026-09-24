# Atlas handoff — Qwen3.8 MLLM fused-GDN canary

- **Owner / receiving role:** Atlas
- **Branch:** `atlas/qwen38-fused-gdn-canary`
- **Clean base:** `0e68c4ea7e266159e5c325b3b4e09e8daef3aaee`
- **Production code commits:** `5278e76ff`, `3b6acab1d`, plus the local
  default-safety commit on this branch
- **Status:** implementation complete with zero automatic-qualification rows;
  no alias or public default changed

## Delivered contract

- A single internal opt-in, `RAPID_MLX_QWEN38_MLLM_FUSED_GDN=1`, can enable
  fused GDN decode only for the exact immutable
  `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` revision `aa985c29…`.
- A pure automatic-enrollment resolver reports `operator_enabled`,
  `automatic_qualified`, or `hardware_not_qualified`. Its production receipt
  table is empty. Unknown hardware, 32 GiB, and unqualified 48/64 GiB hosts
  stop before artifact/runtime/real-weight qualification. The environment
  opt-in remains diagnostic/canary behavior, not automatic qualification.
- Admission requires B0's privately minted artifact truth and opaque verified
  target, the exact 64-layer/48-GDN layout and loaded geometry, pinned runtime
  and source identities, plain non-speculative cache state, and a synchronized
  32-step real-weight parity probe. Any mismatch retains stock mlx-vlm.
- The canary uses the already-loaded VLM and its model-owner executor; it does
  not load another target or create another executor. A pre/post topology
  invariant records one MLLM scheduler, no companion, the same loaded model
  and executor, and one retained-cache budget owner. Qwen3.6's existing
  32-head path is isolated.
- Pre-commit Python failures may use stock. Once cache commit starts, failures
  propagate and invalidate the request cache; there is no stock replay.
- Failed startup, scheduler-stop exceptions, stop, and reload restore the
  original class method, including an active-to-disabled reload. A competing
  class-patch owner fails the compare-and-swap without disturbing the first.

## Frozen evidence

- Experiment source: `1c1c6573ab0353aa94bfb2b04e18208d5ddf2466`
- Methodology SHA-256:
  `5eec2e6dbcd67e5b9ef2df856880eef468ea5394c7d43a3fbad62150394bb5b3`
- External raw receipt SHA-256:
  `7fb5e03a8b672cb93943ff33238076b3ebad0814edf04e9059dbb1ecc14de5a2`
- M3 Ultra result: +7.80% median wall throughput and +8.12% median decode
  throughput; all six strata were positive.
- The raw aggregate result remains **false**. Its only failed aggregate was an
  invalid semantic checker. Independent adjudication is
  `vlm_differential_pass_semantic_aggregate_invalid`: stock-before, candidate,
  and stock-after image token/text hashes matched; exact per-layer candidate
  hits and zero stock hits proved engagement. This adjudication does not
  rewrite the raw result.

## Verification

- `244 passed, 1 skipped, 2 deselected`: focused canary, B0 artifact truth and
  planner, Qwen boot status, existing Qwen3.6 native-cache path, batching, and
  Qwen4 experiment artifact parity suites.
- Ruff check and format, Python compilation, and `git diff --check` pass.
- No model was run or downloaded during production implementation.

## Rollback and remaining gates

Unset `RAPID_MLX_QWEN38_MLLM_FUSED_GDN` (or set it to `0`) and restart. With no
qualified automatic row, the engine uses stock mlx-vlm and reports
`hardware_not_qualified`; no persisted model mutation is required to roll
back.

Apple M2 Pro 32 GiB is a measured **NO-GO**: the frozen experiment harness's
only model-loading attempt failed the mandatory real-weight parity setup before
any performance stratum and increased swap use by 8.46 GiB. It was not rerun.
The production canary received static/read-only checks only; no production
engine was started. The sanitized evidence is recorded in
`docs/benchmarks/results/2026-09-23-qwen38-mllm-fused-gdn-32gb-no-go.json`.

M4 Pro 48 GB and M1 Max 64 GB have not been measured. The B3 audit found the
cross-lane ledger/cache registry inapplicable to this current in-place Q38
path: it has one model, one MLLM scheduler, one executor, one retained-cache
budget domain, and no companion. This change records and enforces those facts
locally rather than adding a generic framework. Automatic qualification stays
closed until an exact hardware receipt is added.

## Next concrete action and risks

Atlas should collect the same setup, text, and media qualification receipts on
48 GB and 64 GB hosts. Do not rerun 32 GB without an explicit diagnostic plan,
and do not change alias defaults without reviewed exact receipts.

The main carried risk is runtime drift: any mlx, mlx-lm, mlx-vlm, kernel,
loaded-layout, or cache-ABI change deliberately disables the optimization and
needs a fresh parity/performance qualification. A process hosting more than
one eligible engine also fails closed at the class-patch compare-and-swap; it
must not bypass that ownership guard.
