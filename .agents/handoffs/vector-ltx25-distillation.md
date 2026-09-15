# Vector to Atlas: LTX-2.5 few-step distillation

Date: 2026-09-13

Owner / host: Vector / MZR-3

Runtime feasibility branch: `raullenchai:vector/ltx25-distillation-feasibility`
at `6b43046` in the `ltx-2-mlx` repository. Rapid design branch:
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

## Complete-noise Stage-1 coupling (2026-09-14)

The first `lane` coupling was found to leave hidden target randomness: a coarse
`0 -> 3` student re-injected original lane 0, while the teacher path also drew
independent lanes 1 and 2. This can force an MSE student toward conditional
mean outputs even though the transition remains nominally stochastic.

Upstream `abbbe6b` adds a separate `span-v2` coupling. For each coarse
transition it propagates and combines every covered fine-step noise lane using
the ancestral Euler input/noise coefficients, then scales that aggregate for
the coarse step. This exactly preserves accumulated random forcing when
teacher denoised predictions are held fixed; it does not claim equivalence of
the nonlinear teacher drift. Training and runtime use the same coupling and
still perform only four Stage-1 model evaluations.

The v2 artifact contract is explicit and fail closed. It carries the complete
fine sigma schedule and contiguous spans `[[0,3],[3,5],[5,7],[7,8]]`, requires
a durable span-v2 curriculum marker, and cannot be relabeled as the v1
single-lane capability. The standard path and existing v1 package behavior are
unchanged. Upstream `dcc65a5` routes the qualification runner through the
selected coupling; the complete suite passes 729 tests with 22 skips.

MZR-3 is running the terminal Stage-2 curriculum and its queued blind suite.
The Stage-1 queue now runs the v1 `0 -> 3` pilot, then the otherwise identical
span-v2 pilot. Span-v2 advances to the full four-transition curriculum only if
both held-out mean modalities improve at least 10% and no paired sample
regresses more than 5%; a gate rejection falls back to lane v1. This is only a
compute gate. Decoded multi-seed motion, detail, diversity, audio, and sync
remain the product gate.

## Terminal Stage-2 scale result (2026-09-14)

The terminal `3 -> 1` curriculum and 24-step replay completed without OOM;
replay took 10.7 minutes and peaked at 19.79 GiB. The final checkpoint improved
all 20 prompt-disjoint held-out pairs in both modalities versus the unadapted
one-step baseline. Mean video latent MSE fell 23.1% and mean audio latent MSE
fell 74.7%; the smallest paired video improvement was 5.2%. These results
authorize decoded testing only and are not a perceptual pass.

The checkpoint was packaged with capability `ltx_stage2_terminal_v1`, exact
schedule `[0.909375, 0]`, immutable base/config/content identity, runtime
contract major 1, and adapter SHA-256
`6e8de0813b3731e1be5ccb66170e219bc1f6395052702a04f3a141cd98d749f1`.
The production loader reopened and validated the package. Its qualification
revision is deliberately `diagnostic-terminal-20260914`; it must not be
uploaded or exposed as a release artifact.

The ten-case decoded terminal blind suite completed. Without reading its
mapping, mean A/B similarity is video SSIM `0.889767`, video PSNR `30.469` dB,
and audio APSNR `167.780` dB. Cases 00-02 are lowest (`0.779-0.811` SSIM) and
case 08 is next (`0.849`); human review should start there. This is below the
progressive suite's `0.924042` SSIM, so the terminal checkpoint remains a
higher perceptual-risk candidate despite improving every latent pair. The
mapping-safe bundle is at
`123/strategy/ltx-stage2-terminal-qualification-2026-09-14/`. Stage-1 v1 is
now training, followed by its paired evaluation and the equal-budget span-v2
pilot.

## Stage-1 span-v2 gate result (2026-09-14)

Both matched `0 -> 3` pilots completed in 29.4 minutes at 19.79 GiB peak. The
v1 single-lane control improved held-out video/audio MSE only 2.84%/1.69%.
Span-v2, changing only the noise coupling, improved the same metrics
33.42%/31.78%; both held-out samples improved in both modalities. Its final
training loss was 202 versus v1's 5366. This supports complete fine-noise
conditioning as the correct Stage-1 direction, but remains a compute gate, not
a decoded quality pass.

The full four-transition span-v2 shared-adapter curriculum is now running. A
fresh-phase validation bug was fixed upstream at `c30cb68`: newly trained
span-v2 checkpoints now use the selected coupling validator just like resumed
phases, rather than accidentally applying the default v1 rule. The upstream
suite passes `736 passed, 22 skipped`.

## Stage-1 clean-final result (2026-09-14)

The complete four-primary/four-replay curriculum finished and every real phase
handoff passed span-v2 metadata validation. The first evaluator launch exposed
a direct-script import bug; upstream `7e3fc26` adds a reproducing subprocess
test and supports both package and direct entrypoints. The recovered evaluator
completed without retraining.

One shared adapter must not execute the already-uncompressed `7 -> 8` step.
On two prompt-disjoint held-out trajectories, the first three transitions
improved video/audio MSE by 7.9%/14.5%, 43.4%/59.6%, and 8.1%/3.6%, while the
adapter-backed final transition regressed by 51834%/1525% relative to the
nearly exact clean base. That artifact is rejected.

Upstream `6b43046` adds a distinct
`ltx_stage1_compressed_span_v2_clean_final` contract. It packages the replay
`5 -> 7` checkpoint, runs the student for the three genuinely compressed
transitions, materializes both modalities, releases the adapter, and runs the
original `7 -> 8` transition on the immutable clean base. Ordinary v1/v2
package semantics remain unchanged. The full upstream suite passes 749 tests
with 22 skips.

A real 768x512, 25-frame, same-prompt/seed smoke measured standard `8 + 3` at
78.80 seconds and combined clean-final `4 + 1` at 47.95 seconds: 1.64x end to
end. Peak footprints were 17.58 and 17.41 GB with no reported process swap
faults. This validates the runtime lifecycle and short-workload speed
direction, not decoded quality or the required 241-frame result.

The complete qualification, both packages, and smoke outputs are archived at
`/Volumes/RTL-2T/scratch-archive/LTX-MZR3-20260914/clean-final-qualification/`.
It contains 70 files / 5.4 GB; all 35 safetensors hashes match the remote
sources. MZR-3 retains the originals and has about 1 GB free. The next action
is to remove only hash-verified archived scratch checkpoints after explicit
owner authorization, then run the 241-frame randomized standard/fast harness
and export its anonymous review bundle. Atlas must not expose or default-enable
this capability before that quality result and a second Apple GPU generation
pass.

## Segmented Stage-1 result (2026-09-14)

The shared clean-final adapter's 25-frame decoded output was visibly broken,
despite its 1.64x speedup, so it is rejected. Per-transition evaluation showed
strong sigma-region weight interference. Upstream `cc9091b` now binds separate
span-v2 adapters to `0 -> 3`, `3 -> 5`, and `5 -> 7`, followed by the exact
clean-base `7 -> 8`. It validates order, schedules, digests, LoRA shapes,
immutable base identity, and runtime major before model mutation. The same
artifact path applies across supported Apple Silicon; hardware identity does
not alter numerical behavior. Full upstream tests pass: `767 passed, 22
skipped`.

MZR-3 paired 768x512x241 runs measured standard `8 + 3` at 539.94 seconds and
segmented `4 + 1` at 259.65 seconds: 2.079x end to end and 51.91% lower
latency. Both reported zero swap; peak footprint increased 3.05% from 39.65 to
40.86 GB. A newly generated standard output exactly matched the prior teacher
SHA-256. Contact-sheet inspection found no grayscale or structural collapse,
but that first case did not establish non-inferiority.

The broader decoded screen rejected this exact three-adapter set. Five more
241-frame fast cases remained stable at 259.09-262.39 seconds with zero swap,
but the chef speech output omitted the requested front-facing speaker across
20 evenly spaced samples while standard retained the chef's face throughout.
The existing terminal-Stage-2-only A/B retained the face on both sides, so the
failure is attributable to segmented Stage 1 or its downstream interaction,
not terminal Stage 2 alone. Stop qualification work on these adapters. Keep the
portable, fail-closed runtime/package design, but do not add or default the
Rapid `generation_mode` API. Vector's next action is to expand prompt/seed and
semantic coverage in the Stage-1 teacher trajectories, retrain each segment,
and repeat the decoded gates before cross-generation qualification. Atlas must
continue to block product exposure of the rejected artifact.

Upstream `d524c95` implements the clean-base independent control and makes
segmented packaging reject shared/cumulative checkpoint provenance. Full
upstream tests pass (`770 passed, 22 skipped`). MZR-3 is training that control
without deleting any existing artifacts; each span retains only its final
checkpoint to fit the remaining disk budget. This does not remove the need for
broader data: the current Stage-1 train split is only eight trajectories from
four duplicated prompts and contains no human face or speech example.

The serialized follow-up captures 48 low-cost train trajectories from 24
prompts and 12 prompt-disjoint validation trajectories from six prompts, then
retrains only `0 -> 3` and reruns the held-out chef case. Upstream `ce31075`
reuses exact Stage-2 conditions by same-filesystem hard link, avoiding roughly
755 MB of duplicate embeddings while freshly capturing every Stage-1 latent
boundary. Full upstream tests pass (`775 passed, 22 skipped`).

The independent three-span control completed, but its valid held-out chef
decode still omits the front-facing speaker. Bound-span video/audio MSE
improvements were 28.47%/31.54%, 61.04%/70.06%, and 4.81%/6.25%; these latent
gains do not override the decoded failure. A first local A/B accidentally used
corrected case 06 (pottery) as the standard; it was rejected and rebuilt with
case 05, the actual chef entry. The broad 24-prompt early-span control is now
running. Atlas must continue to block public API/default integration.

Upstream `052583b` keeps decoded span ablation behind a research-only script
and private runtime state; the production constructor and CLI are unchanged.
Two held-out chef renders are queued after the broad control: clean base for
only `0 -> 3`, then clean base for every compressed span. They preserve the
same 4+1 schedule and isolate adapter harm from schedule harm. Full upstream
tests pass (`781 passed, 22 skipped`).

Those controls and four alternative clean-base schedules are now complete.
All met 1.956-2.123x timing with zero swap, but every one lost the chef's
identity/front-facing framing. Adapter attribution and pure boundary guessing
are closed. Upstream `23ebfd7` adds private custom schedule/checkpoint binding,
an explicit `1 -> 3` independent training control, and a single-transition
paired evaluator; full tests pass (`783 passed, 22 skipped`). MZR-3 is training
that one 100-step broad-data checkpoint. A fail-closed supervisor will require
at least 10% mean video/audio MSE improvement and no held-out sample regression
above 5% before decoding `[0,1,3,5,7,8]`. Atlas must keep public API/default
integration blocked until decoded qualification and cross-generation testing.

The targeted checkpoint completed at 19.79 GB peak and passed the automatic
held-out gate: video/audio MSE improved 49.16%/39.54%, with no greater-than-5%
paired regression across 12 samples in either modality. The combined
`[0,1,3,5,7,8]` chef decode is running. The artifact remains diagnostic and
must be rejected if identity/frontal framing is not restored regardless of
the latent result.

Both decoded variants were rejected. The combined candidate took 278.28
seconds (1.940x), while an early-only version with clean base after the new
`1 -> 3` adapter took 281.78 seconds. Both omitted almost all of the chef, so
the targeted early student itself selects the wrong semantic branch despite
its strong held-out MSE. The projected 4.49% dispatch gain would make the
combined timing 265.8 seconds (2.031x), but cannot override the quality gate.

Vector is now running a progressive on-policy control. Upstream adds a
fail-closed rollout dataset materializer that hard-links all unchanged teacher
components and rewrites only the reached student boundary. It materialized 48
train and 12 prompt-disjoint validation `1 -> 3` outputs in 97.32/24.39 student
seconds, preserving step 5 and later teacher files by inode. The old independent
`3 -> 5` adapter still improves on-policy video/audio MSE by 51.40%/47.08%, so
input-distribution shift alone is not a sufficient diagnosis. A rank-4 matched
`3 -> 5` control is in progress under MZR-3's 290 MiB free-space constraint.
Stop unless it materially exceeds the old adapter and restores the decoded
chef. Atlas continues to own any public API, default, or release decision.

The first rank-4 on-policy `3 -> 5` control did not beat the reference:
video/audio MSE improved 47.43%/38.34%, versus 51.40%/47.08% for the old
adapter. It was not decoded and downstream training stopped. Its target was
counterfactual because it came from the original teacher step-3 state.

Upstream `233d7b6` adds clean-teacher correction from the actual student
boundary using every original fine step and matching seeded noise lane. A
teacher-state `3 -> 5` smoke reproduced the stored target to `3.26e-10` video
MSE and `7.38e-11` audio MSE; only five of 27,904 bfloat16 values changed, with
maximum absolute error 0.001953125. It generated 48 train/12 validation
reachable targets in 193.35/47.58 seconds. The matched rank-4 control is now
running; decoded work remains gated on beating both prior latent references.

The teacher-corrected control failed that gate: it improved reachable-target
video/audio MSE by 50.71%/47.66%, versus 57.71%/55.89% for the existing
teacher-forced rank-8 adapter on the same data. Neither on-policy control was
decoded and no downstream span training started. Vector has stopped ordinary
endpoint-MSE retraining and is decoding `[0,1,2,3,5,7,8]`: exact high-noise
steps, existing independent middle/late adapters, and exact final step. A
quality pass would establish a roughly 1.8x base for a portable kernel/runtime
push; it is not yet a release claim.

The exact-high-noise control completed at 301.40 seconds versus the
539.94-second standard (1.791x, 44.18% lower latency), with zero swap and a
40,379,845,680-byte peak process footprint. Its ten-frame contact sheet keeps
the male chef, face, upper body, bread, oven, composition, and motion
throughout; it avoids the categorical subject loss seen in every compressed
high-noise route. The synchronized side-by-side remains pending owner human
non-inferiority review, so this is not a product/default qualification.

Composing the separately measured 4.49% dispatch gain projects 287.87 seconds
(1.876x), leaving about 6.22% further latency reduction for strict 2x. Vector
is training a diagnostic merged-middle `3 -> 7` adapter on 48 train and 12
prompt-disjoint validation trajectories on Studio. It retains exact
`0 -> 1 -> 2 -> 3` and `7 -> 8`, yielding a five-call Stage-1 schedule that is
projected near 280 seconds before dispatch and 267 seconds after it. Artifact
capability, shapes, memory admission, and a standard fallback—not Apple chip
names—remain the intended product boundary. Atlas owns any public surface,
default, or release decision after decoded and cross-generation qualification.

The rank-4 merged-middle control completed 100 Studio training steps in 230.25
seconds at a trainer-reported 19.49 GB peak and zero swap. On 12
prompt-disjoint trajectories it improved video/audio MSE 15.22%/16.45% over
the same clean-base `3 -> 7` jump; all 12 samples improved in both modalities.
The MZR-3 768x512x241 chef render then measured 278.38 seconds (1.940x versus
539.94 seconds, 48.44% lower latency), zero swap, and a 40,381,926,520-byte
peak footprint. Ten sampled frames retain the same male chef, face, upper
body, bread, oven, composition, and action sequence. It passes the categorical
subject-preservation screen but still needs owner detail/motion/audio review
and the broader suite. Dispatch composition projects 265.88 seconds/2.031x;
the actual composed run measured 270.01 seconds (1.9997x, 49.985% lower
latency), zero swap, and a 39,464,962,808-byte peak footprint. It misses the
strict 269.97-second 2x threshold by 0.04 seconds and must not be rounded up.
The combined contact sheet still retains the chef and action sequence; SSIM
against the non-dispatch merged-middle output is 0.963892. Human review remains
authoritative.

Upstream now has a default-off product contract for this exact route. The new
`ltx_stage1_exact_prefix_middle_span_v1` capability reuses the existing
segmented-manifest CLI, permits only exact `0 -> 1`, `1 -> 2`, `2 -> 3`, one
bound learned `3 -> 7`, and exact `7 -> 8`, and validates immutable model and
artifact identity, independent training/noise metadata, LoRA shapes, and
runtime major. The dedicated packager copies the adapter for distribution;
symlinks remain diagnostic-only. Existing segmented and standard paths are
unchanged. Focused tests pass 31/31 and the full upstream suite passes 802 with
22 skips. Atlas should review this compatibility surface but must not default
it before the remaining human, suite, and cross-generation gates.

The actual product CLI entry point has now loaded that manifest and completed
the same combined workload in 269.21 seconds (2.0057x), zero swap, with a
40,371,407,824-byte peak footprint. Its MP4 hash exactly matches the prior
270.01-second research-path combined render. The two-run range is
269.21-270.01 seconds and median is 269.61 seconds (about 2.001x); because the
range straddles the strict 269.97-second threshold, release language should say
approximately 2x rather than stably greater than 2x.

Adversarial self-review closed a loader-level provenance bypass: the packager
already rejected cumulative/shared Stage-1 checkpoints, but a hand-crafted
matching manifest could previously reach the loader without that same check.
The loader now independently requires clean-base `independent` provenance for
all segmented capabilities. The full upstream suite passes 803 with 22 skips.
The required independent `spark2` PR reviewer could not start because its Codex
refresh token is revoked; no review comment was posted. Reauthentication is the
only remaining prerequisite for the automated loop to issue LGTM.

Three broader real-product-CLI samples are complete: cafe dialogue/faces/hands
269.76 seconds, mountain-bike fast motion 269.25 seconds, and four-impact sync
268.97 seconds. All reported zero swap and 39.51-40.42 GB peak footprint. The
initial contact sheets retain subjects and coarse actions; impact audio attacks
appear at comparable early timestamps, but that candidate is about 5.3 dB
quieter. The owner must review the synchronized side-by-sides and individual
audio files under `123/strategy/ltx-exact-prefix-suite-2026-09-15/`. Vector
must then run the identical artifact on another Apple Silicon generation.
Atlas continues to own public/default/release disposition, and the standard
path remains fallback. Reauthenticate `spark2` and rerun independent review.

The upstream branch now includes the formerly separate opt-in dequantized-
matmul dispatch, so a single pinned revision contains the complete measured
path. Focused combined-path tests pass 40/40 and the full upstream suite passes
808 with 22 skips.

Rapid branch `design/ltx25-portable-fast-stage2` now implements the default-off
`generation_mode=standard|fast` request contract against that audited upstream
revision. Fast mode is LTX-2.5 T2V only, requires an operator-selected complete
model package, fails closed before queueing, advertises package provenance
through capabilities, and never selects numerical behavior from chip or
hostname. Standard remains the default and I2V fast requests are rejected. The
focused cross-backend, route, capability, and persistence suite passes 187
tests; scoped Ruff and diff checks pass.

The real Rapid engine path was paired on a Mac Studio M3 Ultra at
768x512x241, 24 fps, seed 271007. Standard `8+3` took 175.65 seconds and
combined `5+1` took 90.71 seconds, or 1.936x. Both outputs contain
10.041667-second H.264 plus 48 kHz stereo AAC. This and the M4 Pro's
2.001-2.008x product runs satisfy the cross-generation speed/portability gate.

The M3 bakery case fails decoded quality: combined fast has visible dark/high-
frequency texture artifacts and composition drift. Ablation measured Stage
1-only at 150.23 seconds (1.17x) and Stage 2-only at 108.40 seconds (1.62x);
the latter is visually closer to standard in the contact screen. No artifact
is release-qualified or bundled. Atlas must keep standard as the default and
must not authorize upload/default-on from speed evidence alone. Review media
and hashes are in `123/strategy/ltx-studio-ablation-2026-09-15/`.

Next action: Vector should retain the terminal Stage-2 gain and retrain or
reduce the Stage-1 learned span on broader semantic/motion coverage, then rerun
the prompt-disjoint blind gate. Atlas should review only the experimental API
and rollback plumbing for release integration. The spark2 reviewer remains
unavailable because its Codex refresh token is revoked; no LGTM exists.
