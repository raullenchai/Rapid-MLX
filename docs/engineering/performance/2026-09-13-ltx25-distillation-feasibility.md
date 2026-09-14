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
capacity and data, and compares progressive `3 -> 2 -> 1` supervision with the
direct terminal jump. Detailed metrics and paired MP4s are recorded in
`123/strategy/2026-09-13-ltx-stage2-distillation-pilot-results.md` outside this
repository.

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
