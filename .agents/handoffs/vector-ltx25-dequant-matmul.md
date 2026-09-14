# Vector → Atlas: LTX-2.5 dequantized matmul

- Receiving role: Atlas
- Rapid branch: `raullenchai/LTX`
- Upstream runtime branch: `raullenchai:vector/ltx25-dequant-matmul`
- Upstream PR: https://github.com/MrMoferFRAN/ltx-2-mlx/pull/1
- Runtime commit: `15ae0280cb3b2372db9399484ba3944ed0316bb6`
- Rapid integration: exact runtime commit and source SHA pinned on this branch;
  the optimization remains opt-in at `LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024`.

## Verified facts

- `LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024` improves LTX-2.5 Q8 end-to-end
  latency by 4.14% at 121 frames and 4.49% at 241 frames on MZR-3.
- Stage-2 step latency improves 6.91% and 5.46%, respectively.
- Three runs per workload completed with zero swap growth; MLX allocator
  peaks are unchanged from baseline.
- The runtime default remains unchanged and the path is inference-only.
- Output stream contracts are unchanged. Visual/audio comparisons show only
  the expected floating-point trajectory differences; a broader blind test
  remains required for default enablement.

## Unresolved questions and risks

- The qmm-versus-BF16 crossover depends on Apple GPU and MLX version. M3 Ultra
  and a second MLX release have not been measured.
- The 4.1-4.5% end-to-end gain is below the roadmap's 5% default-enablement
  target, although the dominant stage-2 loop exceeds 5%.
- The release pin temporarily uses `raullenchai/ltx-2-mlx`; upstream PR #1 is
  still open, so this should return to the upstream repository after merge.

## Next action

1. Vector benchmarks the PR on the target M3 Ultra and runs the multi-prompt
   visual/audio blind set.
2. If the gain remains positive and quality passes, Atlas decides whether the
   knob stays expert-only or is automatically enabled for qualified chips.
3. After upstream merge, return `LTX25_RUNTIME_REPOSITORY` and the Desktop
   archive URL to upstream, refresh the source hash if GitHub's archive bytes
   differ, rebuild the embedded sidecar provenance, and rerun the Rapid video
   route contract tests.
