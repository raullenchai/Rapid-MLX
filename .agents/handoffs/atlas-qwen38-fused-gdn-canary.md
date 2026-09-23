# Atlas handoff — Qwen3.8 MLLM fused-GDN canary

- **Owner / receiving role:** Atlas
- **Branch:** `atlas/qwen38-fused-gdn-canary`
- **Clean base:** `8aaa8da22d26e2d4a01b5bf37b2ce125889ecc25`
- **Production commits:** `6a6669396` and `015d436b7`
- **Status:** implementation complete and default off; no alias or public default
  changed

## Delivered contract

- A single internal opt-in, `RAPID_MLX_QWEN38_MLLM_FUSED_GDN=1`, can enable
  fused GDN decode only for the exact immutable
  `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` revision `aa985c29…`.
- Admission requires B0's privately minted artifact truth and opaque verified
  target, the exact 64-layer/48-GDN layout and loaded geometry, pinned runtime
  and source identities, plain non-speculative cache state, and a synchronized
  32-step real-weight parity probe. Any mismatch retains stock mlx-vlm.
- The canary uses the already-loaded VLM and its model-owner executor; it does
  not load another target or create another executor. Qwen3.6's existing
  32-head path is isolated.
- Pre-commit Python failures may use stock. Once cache commit starts, failures
  propagate and invalidate the request cache; there is no stock replay.
- Stop/reload restores the original class method and atomically resets status,
  including after an active-to-disabled reload.

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

- `148 passed, 2 deselected`: focused canary, B0 artifact truth, existing
  Qwen3.6 fused-GDN, and batching regression suites.
- Ruff check and format, Python compilation, and `git diff --check` pass.
- No model was run or downloaded during production implementation.

## Rollback and remaining gates

Unset `RAPID_MLX_QWEN38_MLLM_FUSED_GDN` (or set it to `0`) and restart. The
engine uses stock mlx-vlm and reports `operator_disabled`; no persisted model
mutation is required to roll back.

M4 Pro 48 GB and M1 Max 64 GB have not been measured. B3 cross-lane memory
admission, cache-persistence parity, and default-on qualification are also not
implemented. Therefore this canary must remain internal and default off.

## Next concrete action and risks

Atlas should integrate and dogfood the explicit canary first, then collect the
same text-plus-media qualification receipts on 48 GB and 64 GB hosts. Do not
change alias defaults until B3 admission and cache-persistence gates pass.

The main carried risk is runtime drift: any mlx, mlx-lm, mlx-vlm, kernel,
loaded-layout, or cache-ABI change deliberately disables the optimization and
needs a fresh parity/performance qualification. A process hosting more than
one eligible engine also fails closed at the class-patch compare-and-swap; it
must not bypass that ownership guard.
