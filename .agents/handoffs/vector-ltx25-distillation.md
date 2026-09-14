# Vector to Atlas: LTX-2.5 few-step distillation

Date: 2026-09-13

Owner / host: Vector / MZR-3

Runtime feasibility branch: `raullenchai:vector/ltx25-distillation-feasibility`
at `d22f2b6` in the `ltx-2-mlx` repository. Rapid documentation branch:
`raullenchai/LTX`, PR #3438.

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
- The complete 12-item capacity pilot ran successfully: 50 rank-8 Q/K/V steps
  took 4.6 minutes at 19.80 GiB peak. Held-out video/audio MSE improved by up
  to 27.0%/56.2% versus the unadapted terminal jump.
- The pilot failed decoded quality: a held-out moving bee present in the
  teacher disappeared in step-20 and step-50 students. Aggregate latent
  metrics are therefore insufficient as a gate.
- A rank-32 broad-target control on the same split peaked at 24.22 GiB but
  regressed video MSE 10.6% below the unadapted baseline and produced a
  blurrier decode. Rank alone does not solve the small-data direct-jump issue.
- Progressive `3 -> 2` reverses the semantic failure: the bee and all four
  held-out subjects remain present. At step 50, video/audio MSE improve
  22.2%/61.3% versus the unadapted transition. Visible differences remain.
- A zero-shot 768x512 / 1536-token hummingbird probe also improved and
  preserved its moving subject. The measured stage-2 projection is about
  32.7 seconds to 22 seconds, or 1.5x for stage 2 at 25 frames.
- A frozen 100-trajectory scale run is now capturing 80 train / 20
  prompt-disjoint validation items across 468/1536/3072-token buckets.
- The implementation now includes resumable BF16 trajectory capture, the
  terminal strategy, checkpoint schedule/LoRA-scale metadata, paired latent
  evaluation, and paired MP4 rendering. Full tests pass: 616 passed,
  22 skipped.

## Recommendation

Continue qualifying the deterministic stage-2 `3 -> 2` progressive student
using the frozen 468/1536/3072-token curriculum. Keep rank 8 until scaled
train/validation evidence demonstrates capacity underfit. If it becomes
perceptually non-inferior, progressively distill ancestral stage 1 from eight to four
evaluations. This targets `4 + 1`, roughly 1.9-2.1x by the current timing
model. More aggressive `3 + 1` or `2 + 1` schedules are research tiers.

## Risks and unresolved questions

- LoRA capacity may still be insufficient at production motion diversity;
  expand rank and target coverage only from scaled underfit evidence.
- Stage-1 ancestral noise coupling requires a separate progressive-
  distillation implementation.
- "No quality loss" must be a predeclared perceptual non-inferiority gate;
  pixel identity is not achievable for a changed sampler.
- Formal product exposure and any default switch belong to Atlas.

## Next action

Vector should finish the active 100-trajectory capture, verify all 700
components, archive the source set to the Studio dataset tier, and train the
rank-8 progressive objective. Include decoded small-subject, speech, impact,
and synchronization review. If the scaled adapter still drops semantic
detail, escalate to full-model or
smaller-architecture student distillation. Atlas should review fast-tier and
default policy only after a pilot clears the quality gate.
