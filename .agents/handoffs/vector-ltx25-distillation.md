# Vector to Atlas: LTX-2.5 few-step distillation

Date: 2026-09-13

Owner / host: Vector / MZR-3

Runtime feasibility branch: `raullenchai:vector/ltx25-distillation-feasibility`
at `14a0cb9` in the `ltx-2-mlx` repository. Rapid design branch:
`design/ltx25-portable-fast-stage2`. PR #3438 remains the separate marginal
dequantization release work and must not absorb this unqualified model path.

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
- The frozen 100-trajectory scale capture completed: 700/700 components,
  80 train / 20 prompt-disjoint validation items across 468/1536/3072-token
  buckets. The source set is archived on the Studio dataset tier.
- A shuffled mixed-shape training process was OS-terminated before checkpoint
  without traceback, consistent with compiled graph/allocation accumulation.
  Training now uses separate 468/1536/3072 processes (100/50/16 steps) with
  checkpoint handoff and descending learning rates. The 468 stage is stable
  through step 5 at 12.63 seconds/step. A fail-closed supervisor owns stage
  transitions. Validation remains excluded until paired evaluation.
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

Vector should finish the active 200-step rank-8 progressive run and evaluate
every 20-step checkpoint on the frozen validation split. Include decoded
small-subject, speech, impact,
and synchronization review. If the scaled adapter still drops semantic
detail, escalate to full-model or
smaller-architecture student distillation. Atlas should review fast-tier and
default policy only after a pilot clears the quality gate.

## Productization update (2026-09-14)

The scaled run exposed and fixed three generic upstream trainer problems:
lazy gradients were not materialized before clipping/AdamW, Python data
shuffling was not seeded, and diffusers-style saved LoRA tensors were not
converted back to native MLX names/shapes on resume. Upstream feasibility
branch `vector/ltx25-distillation-feasibility` now passes 622 tests with 22
skips. A correct low-learning-rate mixed replay improved held-out video/audio
MSE in all 468/1536/3072-token buckets.

A blind 768x512, 241-frame complex-scene review then passed: the human reviewer
could not tell whether the resume-fixed student or teacher was better. The
mapping was A/left = student and B/right = teacher. Stage 2 measured about
206.7 seconds versus 301.5 seconds (1.46x); projected end-to-end remains only
1.20-1.25x.

The candidate adapter is archived under the Studio cold-model tier with digest
`04ed313c536fae4ad78732f8c403d7ac044614dcb8a95f0f5ae0e896e310855c`.
It is not published or release-qualified. The portable product contract is
proposed in
`docs/engineering/decisions/2026-09-14-ltx25-portable-fast-stage2.md`.

The upstream feasibility branch now includes a hardware-agnostic, fail-closed
checkpoint validator at `396667f`. It binds the adapter to an external artifact
digest, exact base model/revision, transformer/config fingerprint, runtime
contract, schedule, qualification revision, and LoRA tensor shapes. Commit
`c0da6b6` adds the student-first/clean-base-correction lifecycle plus an
experimental manifest-gated CLI entry; `7d4a37d` adds deterministic package
creation. The full upstream suite passes: 644 tests passed and 22 skipped.

A scratch product-format adapter was generated on MZR-3 with SHA-256
`ace51ca8f4d3de9e26099331cac319702bf09bca2b294da8c473c067b43fea05`.
It validated against base revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`; it is not a publishable artifact.
The runtime smoke supervisor waits for the active qualification capture to
release MZR-3 rather than contaminating its timings.

Atlas disposition is required for the proposed public
`generation_mode=standard|fast` control and any default-on policy. Vector's
next backend action is to inspect the runtime smoke result and render the frozen
ten-case, ten-second teacher/student blind suite when capture completes.

## Stage-1 8 -> 4 continuation

The 2x objective requires `4 + 1`, not merely the currently qualified `8 + 2`
candidate. Upstream commit `070eac5` adds exact seeded ancestral stage-1
boundary capture and stops before stage 2; `d2405bb` adds a configurable
one-evaluation transition strategy and a conservative final-pair pilot. The
full suite passes at 647 tests with 22 skipped.

Stage-1 teacher noise is independent between steps, so deterministic endpoint
regression cannot by itself prove a valid ancestral student. Treat the first
pilot as a capacity and failure-mode probe. Do not productize it unless decoded
motion/diversity and prompt-disjoint blind tests pass; if it averages stochastic
targets into blur, move to explicitly noise-coupled or distributional
distillation rather than tuning around the failure.

Commit `a1213c9` adds held-out student-versus-base evaluation, and `e3beea8`
adds resumable deterministic A/B rendering for the ten-case Stage-2 suite. The
full upstream suite passes at 651 tests with 22 skips. MZR-3 is executing the
qualification capture, product-runtime smoke, Stage-1 train/validation capture,
40-step endpoint pilot, held-out evaluation, and blind Stage-2 render as one
fail-closed serialized chain.

The method escalation is now explicit. Endpoint regression remains a cheap
rank/capacity probe only. If decoded motion or multi-seed diversity regresses,
Vector should implement a SCott-inspired noise-controlled stochastic-consistency
transition next. If that still narrows the decoded distribution, escalate to a
DOLLAR-like consistency plus variational-score objective (or DMD-style
distribution matching). Do not compensate for the wrong objective merely by
increasing adapter rank, and do not use latent MSE as a release gate.

## Active 2x experiments (2026-09-14)

Upstream `9976410` adds a fail-closed, resumable `3 -> 1` Stage-2 curriculum
over the frozen 468/1536/3072-token data followed by low-rate mixed replay.
Commit `280acb0` extends the package validator and runtime with a distinct
`ltx_stage2_terminal_v1` capability, so a qualified one-step student avoids the
base correction while remaining isolated from the progressive two-step path.
The complete suite passed at 655 tests with 22 skips.

Upstream `d6084cc` and `164cdcd` add exact original-step ancestral noise
reproduction, inverse ancestral targets, a noise-coupled Stage-1 strategy, and
matching held-out evaluation. The complete suite passes at 659 tests with 22
skips. MZR-3 has serialized the scaled Stage-2 terminal curriculum, latent
evaluation, ten-case blind render, first Stage-1 stochastic transition pilot,
and its held-out evaluation behind the currently running qualification chain.

The upstream work is now tracked as draft PR
`MrMoferFRAN/ltx-2-mlx#3`; its description separates measured gains from the
projected `4 + 1` target and lists all release gates. The latest full local
suite passes 662 tests with 22 skips. The configured independent review runner
on `spark2` could not run: its Codex refresh token is revoked (HTTP 401), and
its basename-keyed clone cache selected an unrelated `ltx-2-mlx` checkout.
It posted no review comment. Harbor/Atlas should repair both authentication and
repo-qualified review worktree selection before treating that review gate as
satisfied.

The manifest-gated product smoke completed on MZR-3. It generated a 768x512,
25-frame, 24 fps H.264 stream plus 48 kHz stereo AAC through the actual CLI in
70.3 seconds. Stage 1 took about 29 seconds, the two fast Stage-2 evaluations
took 12.42 and 11.22 seconds, and decode/mux took 5.9 seconds. Output SHA-256 is
`23d7e0332f9514f4551f1eb3a53a305bea6dd96c422311e1a7bed10cefec02dc`.
This verifies package validation and runtime lifecycle, not the 2x or quality
gate. System swap was 203.25 MiB after the preceding ten-case capture and must
be reported rather than described as zero-swap.

The 40-step deterministic Stage-1 `0.725 -> 0` endpoint probe completed in
11.8 minutes at 19.79 GiB peak. On two prompt-disjoint 3072-token trajectories,
it improved video MSE only 3.62% and audio MSE only 3.18% versus the unadapted
one-evaluation base; evaluation latency was unchanged (6.00 versus 6.05
seconds). This is insufficient evidence to expand or productize terminal
endpoint regression. Keep it as a negative/weak capacity result and continue
with the queued noise-coupled non-terminal transition.

Stage-1 path analysis was run independently on all eight training and both
prompt-disjoint validation trajectories. Both splits ranked boundaries
`[0, 3, 5, 7, 8]` first, i.e. sigmas
`[1.0, 0.98125, 0.909375, 0.421875, 0]` and a `3 + 2 + 2 + 1` compression.
Validation mean video/audio chord error was 0.7099; the second-ranked schedule
was 0.7353. The queued first stochastic pilot was updated from `0 -> 2` to the
selected target segment `0 -> 3`. These scores select an experiment and are
not a release-quality metric.

## Portable Stage-1 implementation (2026-09-14)

Upstream `0c71e5d` adds the runtime primitive needed to preserve the exact
original eight-step ancestral noise lanes under a compressed schedule. Its
default arguments reproduce the existing sampler exactly; malformed explicit
mappings fail closed. `c32306b` adds a resumable shared rank-8 adapter
curriculum for transitions `0 -> 3`, `3 -> 5`, `5 -> 7`, and `7 -> 8`, followed
by a complete low-learning-rate replay pass. `7070141` evaluates each
transition independently against held-out teacher boundaries so catastrophic
forgetting cannot hide behind an aggregate score.

Upstream `8685b9b` implements the opt-in Stage-1 package and runtime contract.
It validates the immutable base revision, transformer/config fingerprint,
adapter SHA-256 and shapes, exact five-sigma schedule, original noise lanes,
runtime major, and qualification revision. A composed Stage-1/Stage-2 pair
must target the same base transformer, additional LoRAs and schedule overrides
are rejected, and model state is released at stage/request boundaries. The
standard eight-step path is unchanged when the manifest flag is absent. Full
upstream tests pass: 694 passed, 22 skipped.

This is product plumbing, not a qualified model claim. The first noise-coupled
segment, complete shared adapter, 241-frame `4 + 1` timing, decoded blind suite,
and cross-generation Apple Silicon run are still pending. Atlas owns the
public `generation_mode` mapping, any default switch, and release integration.

The progressive Stage-2 ten-case blind suite is now complete. Every case is
768x512, 241 frames at 24 fps, with 48 kHz stereo audio. The mapping-safe
anonymous means are SSIM `0.924042`, PSNR `32.504` dB, and audio APSNR
`167.519` dB; cases 03 and 08 are the lowest visual-similarity outliers. The
review bundle is under
`123/strategy/ltx-stage2-qualification-2026-09-14/` and contains only A/B,
muted side-by-side videos, review metadata, and anonymous metrics. It contains
no teacher/student names or hidden mapping. Do not call the ten-case suite a
pass until the human judgments are recorded.

The serialized MZR-3 chain has advanced to the terminal Stage-2 468-token
training phase. A later supervisor will advance the Stage-1 `0 -> 3` pilot to
the complete four-transition curriculum only if both mean modalities improve
at least 10% and no held-out sample regresses more than 5%; that is a compute
gate, not a product-quality gate.

The first exported blind bundle exposed a review-metadata indexing defect:
rendering used `PrecomputedDataset` filesystem discovery order while prompt
labels came from a separately sorted filename list. The A/B videos and
anonymous metrics were valid, but prompt/stress labels were permuted. Upstream
`b2054aa` makes dataset discovery deterministic and derives both render
positions and review metadata from the same dataset instance. The bundle's
index was repaired from the captured original dataset order, the prior index
was retained on MZR-3 as `review-index.pre-index-fix.json`, and the future
terminal blind supervisor is verified to run `b2054aa`. The full upstream
suite passes 711 tests with 22 skips at `14a0cb9`.

Subsequent hardening adds an index-paired Stage-2 student/base diagnostic
report, fail-closed metadata validation for resumed terminal curriculum
checkpoints, and index-paired Stage-1 pilot gating. The latter prevents a
changed dataset export order from silently approving the full curriculum.
The terminal 468-token phase completed 100 steps in 20.4 minutes at 19.8 GB
peak and handed its step-100 adapter to the 1536-token phase successfully.

MZR-3 has 9.5 GiB free during this run. Three completed, inactive pilots were
copied without deletion to
`/Volumes/RTL-2T/scratch-archive/LTX-MZR3-20260914/`; source and archive file
counts and byte totals match for all three directories. The remote originals
remain intact pending explicit cleanup authorization.
