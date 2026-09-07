# Qwen4 fp32-input fast RMSNorm handoff

Receiving role: Atlas

Owner / host: Vector / Studio, with M2 Pro Mini microbenchmark

Branch / PR: `vector/qwen4-fast-rmsnorm`, PR #3151

## Verified facts

- The route is opt-in through `RAPID_MLX_QWEN4_FAST_RMSNORM=1`, limited to
  widths `<= 8`, and confined to the vendored Qwen4-Exp text model.
- M3 Ultra production-shape micro speedups are 1.105x, 1.109x, and 1.117x;
  M2 Pro results are 1.103x, 1.126x, and 1.124x.
- Over 1,310,720 elements, fp64 RMS-error ratio is 1.000000000066x for the
  fp32-input candidate versus 1.413971716665x for the rejected bf16-input arm.
- The QSA synthetic singleton now carries the real parent forward width, so a
  wide prefill cannot enter the narrow route.
- 157 focused tests, ruff check/format, compileall, and diff check passed.
- Full method, raw summary ranges, commands, and limitations are in
  `docs/engineering/performance/2026-09-06-qwen4-fast-rmsnorm.md`.

## Unresolved questions and risks

- The handoff source reports +4.2% decode at 1K and +3.9% at 16K, but Vector
  did not independently rerun the 106 GB artifact because no safe resident
  storage was available. These remain inherited rather than Rapid-owned e2e
  claims.
- Exactness is class 3. The PR must remain opt-in unless a named owner accepts
  the fidelity scope plus a resident-artifact greedy/digest gate.
- Final Codex self-review on the exact PR head found no blocking correctness,
  routing, scope, benchmark, or documentation issue. The repository owner
  explicitly selected this review path instead of the unavailable spark2 loop.

## Next concrete action

1. On a host where revision `dcf657e4acda2aae72da99cde65b6c491cd96998`
   is already resident, run the committed 1K/16K interleaved real-model gate.
2. Atlas decides whether the independently reproduced kernel and accuracy
   evidence is sufficient to merge the default-off route before that e2e rerun.
