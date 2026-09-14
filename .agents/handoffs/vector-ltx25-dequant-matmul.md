# Vector → Atlas: LTX-2.5 dequantized matmul

- Receiving role: Atlas
- Rapid branch: `raullenchai/LTX`
- Upstream runtime branch: `raullenchai:vector/ltx25-dequant-matmul`
- Upstream PR: https://github.com/MrMoferFRAN/ltx-2-mlx/pull/1
- Runtime commit: `905efb2308a05385f4051e1af6ac322147be23ed`
- Rapid integration: exact runtime commit and source SHA pinned on this branch;
  the optimization remains opt-in at `LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024`.

## Verified facts

- `LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024` improves LTX-2.5 Q8 end-to-end
  latency by 4.14% at 121 frames and 4.49% at 241 frames on MZR-3.
- Stage-2 step latency improves 6.91% and 5.46%, respectively.
- Three runs per workload completed with zero swap growth; MLX allocator
  peaks are unchanged from baseline.
- The runtime default remains unchanged and the path is inference-only.
- On the target M3 Ultra, whole-block measurements improve 4.65-4.70% at the
  121-frame shape and 3.93-4.41% at the 241-frame shape across MLX 0.32.0 and
  0.32.2. These use synthetic Q8 weights because the model is not cached on
  Studio; MZR-3 remains the end-to-end evidence.
- Smaller whole-block shapes regress 1.63-1.72% with the current 1,024-token
  threshold, so a global default is not qualified.
- Output stream contracts are unchanged. Visual/audio comparisons show only
  the expected floating-point trajectory differences; a broader blind test
  remains required for default enablement.

## Unresolved questions and risks

- The qmm-versus-BF16 crossover depends on Apple GPU, projection shape, and MLX
  version. Studio was measured on MLX 0.32.0 and 0.32.2, but its approved
  Hugging Face cache has only 12 GiB free and lacks the 67.7 GB model, so a
  full end-to-end Studio generation was not possible.
- The 4.1-4.5% end-to-end gain is below the roadmap's 5% default-enablement
  target, although the dominant stage-2 loop exceeds 5%.
- The release pin temporarily uses `raullenchai/ltx-2-mlx`; upstream PR #1 is
  still open, so this should return to the upstream repository after merge.

## Next action

1. Vector evaluates a conservative shape-aware threshold, then repeats the
   MZR-3 end-to-end matrix and runs the multi-prompt visual/audio blind set.
2. If every affected shape remains positive and quality passes, Atlas decides
   whether the knob stays expert-only or is automatically enabled for
   qualified chips.
3. After upstream merge, return `LTX25_RUNTIME_REPOSITORY` and the Desktop
   archive URL to upstream, refresh the source hash if GitHub's archive bytes
   differ, rebuild the embedded sidecar provenance, and rerun the Rapid video
   route contract tests.
