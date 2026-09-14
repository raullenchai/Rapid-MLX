# Vector to Atlas: LTX-2.5 few-step distillation

Date: 2026-09-13

Owner / host: Vector / MZR-3

Runtime feasibility branch: `vector/ltx25-distillation-feasibility` in the
`ltx-2-mlx` repository. Rapid documentation branch: `raullenchai/LTX`, PR
#3438.

## Verified facts

- The current distilled two-stage product path performs eight stage-1 and
  three stage-2 transformer evaluations.
- Exact kernel-level candidates found so far cannot approach a 2x end-to-end
  improvement.
- Q8 rank-8 Q/K/V LoRA backward works on the 48 GiB MZR-3 host at 3072 video
  tokens: 31.63 GiB MLX peak and 71.59 seconds for one forward/backward.
- A 6144-token backward completed but peaked at 50.10 GiB and took 179.29
  seconds. It is unsafe for sustained training on this host.
- MZR-3 has about 19 GiB free internal storage and no RTL-2T mount. Do not
  accumulate teacher trajectory datasets there.

## Recommendation

Implement and qualify a deterministic stage-2 `3 -> 1` terminal-latent
student first, using a 468/1536/3072-token curriculum. If it is perceptually
non-inferior, progressively distill ancestral stage 1 from eight to four
evaluations. This targets `4 + 1`, roughly 1.9-2.1x by the current timing
model. More aggressive `3 + 1` or `2 + 1` schedules are research tiers.

## Risks and unresolved questions

- LoRA capacity may be insufficient for one-step stage 2; rank and target
  coverage must be expanded based on the pilot, not assumed.
- Stage-1 ancestral noise coupling requires a separate progressive-
  distillation implementation.
- "No quality loss" must be a predeclared perceptual non-inferiority gate;
  pixel identity is not achievable for a changed sampler.
- Formal product exposure and any default switch belong to Atlas.

## Next action

Vector implements teacher trajectory capture, Euler terminal-target tests,
and a `stage2_terminal_distill` training strategy, then runs a 256-item pilot.
Atlas should review the fast-tier/default policy only after the pilot clears
the quality gate.

