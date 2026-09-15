# LTX-2.5 few-step distillation feasibility on MZR-3

Date: 2026-09-13

Owner: Vector

Host: MZR-3, Mac mini `Mac16,11`, M4 Pro 12-core CPU / 16-core GPU,
48 GiB unified memory, macOS 26.5.1, MLX 0.32.0.

## Decision

Pursue model-evaluation reduction, beginning with a one-step stage-2 student.
Kernel work remains useful as a secondary 5-10% multiplier, but it cannot
produce the requested roughly 2x end-to-end improvement on its own.

The current distilled two-stage schedule makes 11 transformer evaluations:
eight half-resolution ancestral stage-1 evaluations followed by three
full-resolution deterministic stage-2 evaluations. The target schedule is
`4 + 1`. The first checkpoint is `8 + 1`, because stage 2 is deterministic,
expensive, and easier to supervise than the ancestral first stage.

"No quality loss" is defined as perceptual non-inferiority to the current
teacher over a held-out prompt/seed suite. A few-step student will not be
pixel-identical to its teacher. Same-seed image metrics alone are not an
adequate quality definition for generated video.

## Measured training boundary

The existing Q8 distilled transformer was loaded non-streaming, Q/K/V rank-8
LoRA adapters were attached, and one real 48-layer forward/backward was run
with per-block gradient checkpointing. No new model was downloaded.

| Video tokens | Audio tokens | Text/register tokens | Rank | MLX peak | Step time | Result |
|---:|---:|---:|---:|---:|---:|---|
| 144 | 26 | 64 | 2 | 20.57 GiB | 3.81 s | pass |
| 468 | 101 | 1024 | 8 | 21.52 GiB | 12.16 s | pass |
| 3072 | 126 | 1024 | 8 | 31.63 GiB | 71.59 s | pass |
| 6144 | 126 | 1024 | 8 | 50.10 GiB | 179.29 s | unsafe for sustained training |

The rank-8 Q/K/V adapter has 40,108,032 trainable parameters. The loaded Q8
base plus adapter occupies about 19.5 GiB; the unsafe 6144-token peak is
activation-driven. Use a 468 -> 1536 -> 3072 token curriculum on MZR-3 and
reserve 6144-token runs for validation inference. Do not run sustained
full-resolution backward on this 48 GiB host.

MZR-3 has only about 19 GiB free internal storage and no RTL-2T mount. Teacher
targets must be generated online or streamed to the Studio dataset tier at
`/Volumes/RTL-2T/datasets/`; they must not accumulate in MZR-3's internal
cache or `/private/tmp`.

## Stage-2 student

For each training item, start with the exact upscaled latent at sigma
`0.909375`. Run the frozen teacher through the current deterministic schedule
`0.909375 -> 0.725 -> 0.421875 -> 0`. Train one student evaluation at
`0.909375` to land on the teacher's terminal video and audio latents.

The first implementation should reuse one Q8 base in memory and toggle a LoRA
student, rather than loading separate teacher and student transformers. Cache
detached terminal targets when storage permits. Start with rank-8 Q/K/V as a
capacity probe; if the quality plateau is capacity-bound, expand to rank
16/32 and include video FFN, video-text attention, and both audio-video cross
attention directions. Fuse and requantize the accepted adapter for inference
so adapter matmuls do not erode the saved model evaluation time.

Use a terminal latent reconstruction loss for video and audio, supplemented by
intermediate velocity supervision. Add temporal/frequency-weighted video loss
and cross-modal timing loss only after the basic teacher-target objective is
stable. Training against ordinary clean-video flow-matching targets alone is
fine-tuning, not few-step distillation, and will not teach the one-step
schedule.

## Quality and speed gates

Use a held-out, stratified prompt suite covering camera motion, fast subject
motion, faces/hands, text, speech, impacts, ambience, and silence. Keep prompts
and seeds fixed across teacher and student.

The stage-2 student advances only if:

- median latency improves by at least 30% end to end at both 121 and 241 frames;
- paired blind preference is non-inferior to the teacher, with no systematic
  regression category;
- temporal consistency, semantic alignment, audio quality, and audio-video
  synchronization stay inside predeclared non-inferiority margins;
- peak memory does not regress beyond the 48 GiB product envelope; and
- no single benchmark prompt hides severe motion collapse, repeated frames,
  speech drift, or transient smearing behind a good aggregate score.

PSNR, SSIM, LPIPS/DINO/CLIP similarity, audio spectral distance, and a sync
model are diagnostics. Human paired review is the release gate because the
student is not expected to reproduce identical pixels.

## Roadmap

1. Implement teacher trajectory capture and a `stage2_terminal_distill`
   training strategy. Unit-test Euler target construction and LoRA-disable
   teacher calls.
2. Run a 256-item rank-8 pilot at 468-1536 tokens. Stop early if held-out
   terminal error and decoded motion do not improve over schedule truncation.
3. Scale successful training to mixed 468/1536/3072-token buckets. On MZR-3,
   2,000 student-only 3072-token steps would take about 40 hours before teacher
   generation and validation, so most optimization steps should occur in the
   smaller buckets with a short 3072-token finish.
4. Qualify `8 + 1`. At 241 frames, removing two measured stage-2 evaluations
   should move the current roughly 515-second run toward roughly 320-340
   seconds, or about 1.5-1.6x, before any kernel gain.
5. Only after stage 2 passes, progressively distill ancestral stage 1 from
   eight to four evaluations with coupled teacher noise. Qualify `4 + 1` at an
   expected roughly 240-270 seconds, or about 1.9-2.1x.
6. Treat `3 + 1` or `2 + 1` as research tiers. More than about 2x with the same
   22B architecture requires these more aggressive schedules or architectural
   distillation, and perceptual non-inferiority is substantially less certain.

## Capacity-pilot result

A 12-trajectory 192x192x25 pilot (8 train, 4 held out) validated the complete
capture, training, adapter-fusion, terminal-evaluation, and decode path. Rank-8
Q/K/V training ran 50 steps in 4.6 minutes at 19.80 GiB peak. On held-out data,
the best video checkpoint reduced one-step MSE by 27.0%; step 50 reduced audio
MSE by 56.2%. One student evaluation took about 2.03 seconds at this shape.

The checkpoint failed the perceptual gate: a small moving bee visible in the
teacher disappeared in both the step-20 and step-50 decoded students. This is
not ready for inference integration. The next experiment expands adapter
data and compares progressive `3 -> 2 -> 1` supervision with the direct
terminal jump. A same-data rank-32 broad adapter regressed held-out video MSE
by 10.6% versus the unadapted baseline and produced a blurrier decode, showing
that capacity alone is not the fix. Detailed metrics and paired MP4s are recorded in
`123/strategy/2026-09-13-ltx-stage2-distillation-pilot-results.md` outside this
repository.

## Progressive 3 -> 2 result

A controlled follow-up kept the same 8/4 split and rank-8 Q/K/V adapter but
changed the target to the teacher state at sigma `0.421875`. The product
candidate performs a learned `0.909375 -> 0.421875` jump and retains the
unchanged base model for `0.421875 -> 0`.

At step 50, held-out video MSE improved from `0.029600` to `0.023035`
(-22.2%) and audio MSE from `0.037557` to `0.014525` (-61.3%). Video/audio
cosine reached `0.981379` / `0.993464`. Paired decodes retained the bee,
fireworks, dog, and locomotive subjects and coarse motion. Visible detail and
trajectory differences remain, so the adapter passes the direction gate but
not the release-quality gate.

The same adapter was tested zero-shot at 768x512, 25 frames (1536 video
tokens) on a new hummingbird prompt. Video MSE improved
`0.025395 -> 0.022698`; audio MSE improved `0.021937 -> 0.010367`, and the
decoded subject and hovering motion were preserved. Three stage-2 evaluations
took about 32.7 seconds at this shape; two project to about 22 seconds. This
demonstrates resolution transfer and a 1.5x stage-2 gain, not yet a
241-frame end-to-end claim.

## Scaled qualification

Runtime-port commit `c475352` freezes a 100-trajectory manifest with 40 train
and 10 prompt-disjoint validation prompts, two seeds each. It allocates 60
items at 468 tokens, 30 at 1536, and 10 at 3072. The checked-in capture path
is resumable by global manifest index, and a split tool hard-links complete
components without duplicating storage.

Capture completed on MZR-3 on 2026-09-14 at
`/private/tmp/LTX-progressive-scale.zv79Uv`. The fail-closed supervisor passed
the exact 420/630/700 component-file milestones. Prompt-disjoint split views
contain 560 train and 140 validation components. The 700-file, 1.3 GiB source
set is archived at
`/Volumes/RTL-2T/datasets/ltx-stage2-distillation/progressive-scale-2026-09-14/`.

An initial 200-step run shuffled all three shapes in one process and was
terminated by the OS before its first checkpoint, without a Python traceback.
Memory returned immediately after exit. The working diagnosis is accumulated
compiled graphs/allocations across changing shapes, not a single-shape OOM;
3072-token backward had already passed independently at 31.63 GiB.

Training therefore uses separate processes as a true curriculum: 100 steps at
468 tokens, 50 at 1536, and 16 at 3072, carrying the rank-8 Q/K/V checkpoint
forward and resetting the optimizer at lower learning rates. The 468 stage is
stable through step 5 at 12.63 seconds/step and loss `0.1746`. A fail-closed
supervisor requires each final checkpoint before launching the next shape.
The validation split remains outside training and will be used only for paired
latent evaluation and decoded review.

## Stage-1 complete-noise correction

The initial compressed Stage-1 prototype selected only the first seeded noise
lane in each coarse span. For `0 -> 3`, teacher lanes 1 and 2 therefore remained
unobserved target randomness, a plausible source of conditional-mean blur and
motion loss.

The `span-v2` prototype instead combines all covered fine lanes with their
ancestral Euler propagation coefficients. Under fixed denoised predictions,
one coarse injection reproduces the exact accumulated stochastic forcing of
the fine path. The model drift is nonlinear, so this is a better coupling, not
an analytical proof of equal samples. It costs no additional transformer
evaluation and preserves the projected `4 + 1` timing.

The implementation uses a distinct package capability with a full reference
sigma table, contiguous noise spans, curriculum provenance, and fail-closed
runtime validation. The v1 path and the standard eight-step sampler remain
unchanged. At upstream `dcc65a5`, all 729 tests pass with 22 skipped. MZR-3 is
running matched v1/v2 first-transition pilots before committing compute to the
full four-transition curriculum.

The coupling arithmetic is not a material runtime cost. On the M3 Ultra Studio
with MLX 0.32.0, 30 materialized calls at video/audio shapes `(1, 3072, 128)`
and `(1, 126, 128)` averaged 0.407 ms for a legacy lane and 0.592 ms for the
three-lane `0 -> 3` span, a 0.185 ms delta. This is a primitive microbenchmark,
not an end-to-end result.

## Reproduction

Model: `MrMofer/ltx-2.5-mlx-q8`, revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`, transformer
`transformer-distilled.safetensors`. The feasibility branch in the MLX port is
`vector/ltx25-distillation-feasibility`; run its
`scripts/distill_backward_smoke.py` with `LTX2_DIT_EVAL_EVERY=0` and the
workspace packages on `PYTHONPATH`.

## References

- [LTX-2 official trainer configuration](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/configuration-reference.md)
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- [DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](https://arxiv.org/abs/2412.15689)
- [SCott: Accelerating Diffusion Models with Stochastic Consistency Distillation](https://arxiv.org/abs/2403.01505)

## Stage-1 schedule attribution (2026-09-15)

The broad 24-prompt `0 -> 3` control improved a new 12-trajectory held-out
split by 31.13% video MSE and 24.76% audio MSE, modestly better than the narrow
adapter's 28.19%/20.65%. Its chef decode recovered only a partial late side
face and still failed front-facing speaker preservation.

Holding the original `[0,3,5,7,8]` schedule fixed while disabling only the
early adapter took 257.90 seconds (2.094x); disabling every adapter took 254.39
seconds (2.123x). Both retained the identity/framing failure, which rejects
the adapters as the primary cause. Four clean-base boundary controls then
measured `[0,1,5,7,8]` at 257.32 seconds (2.098x), `[0,1,3,7,8]` at 258.57
seconds (2.088x), `[0,1,3,5,7,8]` at 276.10 seconds (1.956x), and
`[0,1,2,7,8]` at 255.40 seconds (2.114x). All had zero swap and all failed
the same decoded chef identity gate.

Pure clean-base schedule search is therefore stopped. The next candidate is
`[0,1,3,5,7,8]` with exact base at `0 -> 1` and `7 -> 8`, a newly trained
broad-data student at `1 -> 3`, and the independent students at `3 -> 5` and
`5 -> 7`. It remains diagnostic-only. The 1.956x clean-base timing is 2.27%
above the 2x latency threshold; the separately measured equivalent
dequantized-linear dispatch saved 4.49% on the 241-frame workload, so the
combined budget can exceed 2x if and only if the learned schedule first passes
decoded quality.

The targeted `1 -> 3` run completed 100 steps in 9.1 minutes at 19.79 GB peak;
final loss was 41.4717 and the sole checkpoint is 160,674,992 bytes. On the 12
held-out trajectories it reduced video MSE by 49.16% and audio MSE by 39.54%
relative to clean base. Every paired sample stayed within the 5% regression
limit in both modalities. The combined 241-frame chef decode is now the
authoritative remaining gate; these latent results alone do not qualify it.

That decode and an adapter-isolation rerun both failed the gate. The combined
candidate took 278.28 seconds (1.940x versus 539.94 seconds); applying the
separately measured 4.49% dispatch gain projects 265.8 seconds, or 2.031x.
However, it omitted almost the entire chef rather than preserving the requested
front-facing speaker. Keeping only the new `1 -> 3` adapter and clean base for
the later spans took 281.78 seconds and produced the same failure. The early
student therefore enters a wrong semantic branch by itself; downstream legacy
adapters are not required to trigger it. This artifact is rejected regardless
of its latent MSE and timing.

The next experiment changes the training distribution rather than guessing
more boundaries. A fail-closed rollout materializer hard-links the immutable
teacher dataset and replaces only a student's reached boundary, preserving the
same conditions, seeds, noise lanes, and later teacher targets by inode. It
materialized all 48 train and 12 prompt-disjoint validation `1 -> 3` outputs in
97.32 and 24.39 student seconds. The legacy `3 -> 5` adapter still improved MSE
on these on-policy inputs by 51.40% video and 47.08% audio, so simple exposure
bias is not a complete explanation. A matched on-policy `3 -> 5` control must
substantially exceed that reference and restore the decoded subject before any
additional downstream training or product integration.

The first matched rank-4 control failed that compute gate. It improved
shifted-input video/audio MSE by 47.43%/38.34%, worse than the old adapter's
51.40%/47.08%; it was not decoded and no `5 -> 7` training followed. The
target was also counterfactual: it asked a student step-3 state to reach the
step-5 state produced from a different teacher step-3 state.

A stricter DAgger-style capture now runs the clean teacher's original fine
steps from the actual student state with the original per-step noise lanes.
Starting from a captured teacher step 3, its `3 -> 4 -> 5` smoke reproduced
the stored target to video/audio MSE `3.26e-10`/`7.38e-11`; only five of
27,904 bfloat16 values differed, with maximum absolute error 0.001953125. It
materialized 48 train and 12 validation reachable targets in 193.35/47.58
seconds. This changes only offline supervision; the inference schedule and
portable artifact contract stay unchanged.

That teacher-corrected rank-4 control also failed its compute gate. Against
the reachable target it improved video/audio MSE by 50.71%/47.66%; the old
teacher-forced rank-8 adapter improved 57.71%/55.89% on exactly the same
inputs and targets. Neither on-policy variant was decoded, and ordinary
endpoint-MSE downstream training is stopped.

The next quality control protects the semantic branch with exact clean-base
steps `0 -> 1 -> 2 -> 3`, uses the existing independent students only for
`3 -> 5` and `5 -> 7`, and finishes with exact `7 -> 8`. Boundaries
`[0,1,2,3,5,7,8]` require six Stage-1 evaluations and are expected to be
roughly 1.8x end to end. If decoded quality passes, the remaining gap to 2x
belongs to portable kernel/runtime optimization, not another high-noise skip.

That route completed the 768x512x241 chef workload in 301.40 seconds versus
539.94 seconds for the identity-verified standard: 1.791x end to end and
44.18% lower latency. It used zero swap and peaked at 40,379,845,680 bytes.
The initial ten-frame screen retains the same male chef and front-facing
composition throughout, unlike every schedule that compressed the first three
high-noise evaluations. It is still pending human and broader-suite
non-inferiority; the timing is not a release claim.

The measured 4.49% dequantized-linear dispatch gain projects 287.87 seconds,
or 1.876x, if it composes. A merged-middle `3 -> 7` diagnostic would reduce
Stage 1 to five calls while leaving `0 -> 1 -> 2 -> 3` and `7 -> 8` exact. Its
timing model is about 280 seconds before dispatch and 267 seconds after it,
enough to test the strict 2x boundary without compressing the semantic-sensitive
prefix. It must pass held-out latent, decoded subject/motion/audio, and
cross-generation gates before packaging.
