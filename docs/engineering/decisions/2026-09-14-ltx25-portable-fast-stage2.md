# Portable LTX-2.5 fast stage-2 product contract

Date: 2026-09-14

Status: proposed; Atlas owns the public API and default-on disposition

Owner: Vector (runtime and qualification), Atlas (product integration)

## Decision

Productize few-step LTX acceleration as a checkpoint capability, never as a
Mac model capability. Any Apple Silicon system that passes the existing base
LTX-2.5 resource admission may execute the same fast schedule. Chip generation,
GPU-core count, and installed-memory labels must not select weights, schedules,
or numerical behavior.

The initial candidate replaces the first two deterministic stage-2 teacher
transitions with one learned transition, then retains the unchanged final base
correction. It therefore changes stage 2 from three transformer evaluations to
two. One blind 768x512, 241-frame review passed perceptual non-inferiority after
the reviewer had detected brightness and detail regressions in the preceding
checkpoint. Measured stage-2 latency was approximately 301.5 to 206.7 seconds
(31.4% lower, 1.46x); projected end-to-end improvement is approximately
1.20-1.25x because stage 1, text encoding, decoding, and muxing are unchanged.

This one-sample pass authorizes product-contract work and a broader quality
suite. It does not authorize default enablement.

## Relationship to the 2x objective

This `8 + 2` candidate is a productization checkpoint, not the final speed
target. At the measured 241-frame shape, startup/encoding is about 11 seconds,
stage 1 is about 173 seconds, standard stage 2 is about 302 seconds, and decode
plus mux is about 53 seconds. Replacing one full-resolution stage-2 evaluation
therefore projects approximately 439 seconds total, only about 1.23x faster
than the 539-second baseline.

Reaching approximately 2x requires a qualified `4 + 1` schedule: four
half-resolution stage-1 evaluations and one full-resolution stage-2 evaluation.
The timing model projects roughly 252 seconds before any additional kernel
gains, or about 2.14x. This remains a target until both distillation stages pass
the same blind non-inferiority gates.

Stage 1 is ancestral and injects independent noise between evaluations. It
cannot be treated as deterministic schedule truncation. The research branch
now records every seeded teacher boundary and can train a one-evaluation
transition pilot, but deterministic endpoint regression is only a capacity
probe: distributional and decoded blind evaluation must reject blur, motion
collapse, or reduced diversity before any four-step schedule is proposed.

## Stage-1 distillation method decision

The first `8 -> 7` final-pair pilot intentionally minimizes endpoint latent
error. It is cheap enough to answer whether a rank-8 adapter can represent one
compressed transition, but its target includes ancestral noise that is not
recoverable from the input state alone. A model trained only with per-sample
endpoint MSE may therefore learn a conditional mean, reduce diversity, and
blur motion even when held-out MSE improves. That result must not advance to a
product checkpoint on metric evidence alone.

If that pilot fails decoded motion or diversity, the next implementation is a
noise-controlled stochastic-consistency objective inspired by SCott. Training
must condition or couple the student transition to the teacher's stochastic
path and preserve an explicit sampling-strength control. This is the smallest
method change that directly addresses the ancestral mismatch rather than
adding LoRA rank to the wrong objective.

If stochastic consistency preserves motion but still narrows the decoded
distribution, escalate to distribution-level video distillation: a DOLLAR-like
combination of consistency and variational score distillation, or DMD-style
distribution matching. These require additional score-model or adversarial
training machinery and are second-line work, not a reason to weaken the
quality gate.

Primary method references:

- SCott, stochastic consistency distillation for SDE samplers:
  <https://arxiv.org/abs/2403.01505>
- DOLLAR, few-step video distillation with consistency and variational score
  objectives: <https://arxiv.org/abs/2412.15689>
- Distribution Matching Distillation: <https://arxiv.org/abs/2311.18828>

For every method, acceptance requires decoded prompt-disjoint comparisons of
subject identity, fine detail, temporal motion, motion amplitude, diversity
across seeds, audio content, and synchronization. Latent MSE, cosine similarity,
and runtime are diagnostics, not substitutes for blind non-inferiority.

## Artifact contract

The accepted adapter should ship inside an immutable model revision beside the
base transformer. Runtime auto-discovery must not search arbitrary user cache
paths. A fast-stage checkpoint must carry, at minimum:

- a versioned capability identifier such as `ltx_stage2_transition_v1`;
- start and target sigma, plus the complete resulting schedule;
- adapter rank and alpha;
- expected base model identifier, immutable model revision, transformer
  filename, and transformer/config fingerprint;
- supported pipeline family and major runtime contract version;
- qualification manifest revision.

The immutable model manifest must bind the adapter path to its SHA-256 digest.
The digest cannot live inside the file it authenticates without becoming
self-referential; runtime validates the external manifest digest before reading
or fusing the adapter.

The manifest and adapter metadata must also bind the actual transformer content
SHA-256, not only its configuration and a self-declared revision. For Hugging
Face snapshots the runtime may use the immutable LFS `blobs/<sha256>` object
name, avoiding a multi-gigabyte startup read; ordinary local files must be
hashed incrementally. The manifest value, adapter metadata, resolved
transformer content, and a composed Stage-1/Stage-2 pair must all agree.

The current research metadata (`distillation`, `stage2_sigma`,
`stage2_target_sigma`, and `stage2_steps`) is necessary but insufficient for
automatic product activation because it does not bind the adapter to an exact
base checkpoint.

The compressed Stage-1 counterpart uses capability
`ltx_stage1_compressed_v1`. In addition to the same immutable-base and digest
requirements, it carries the complete schedule
`[1.0, 0.98125, 0.909375, 0.421875, 0.0]`, original noise-lane indices
`[0, 3, 5, 7]`, original noise-stream length 8, and the fixed ancestral
`eta=1`, `s_noise=1` semantics. A Stage-1 and Stage-2 package may compose only
when their base model, revision, transformer file, and configuration digest
match exactly.

Metadata, tensor names, tensor shapes, schedule monotonicity, and base identity
must all validate before model mutation. Missing, malformed, or incompatible
metadata fails closed. Standard generation remains available; the runtime must
never silently truncate the teacher schedule as a fallback.

Package creation is also a trust boundary. Before adding product metadata, the
builder must verify that its source is the expected distillation strategy and
transition, names the expected teacher target latents, and (for Stage 1) is the
final `7 -> 8` checkpoint from the complete shared-adapter curriculum. The v1
runtime accepts only exact qualified schedules `[0.909375, 0.421875, 0]` or
`[0.909375, 0]` for Stage 2; merely monotonic arbitrary sigmas are not a v1
capability. This prevents an unrelated LoRA from being relabeled as a qualified
fast artifact.

## Runtime lifecycle

The base transformer runs all of stage 1. At the stage-2 boundary, the runtime
loads and fuses the qualified transition adapter, evaluates the learned
`0.909375 -> 0.421875` transition, releases that model, reloads the clean base
transformer, and evaluates the unchanged `0.421875 -> 0` correction. Video and
audio must switch together.

This lifecycle matches the validated research renderer and prevents the
transition adapter from leaking into stage 1 or the final correction. Model
reload overhead is small relative to a production-shape transformer evaluation
and preserves the low-RAM envelope. Cancellation, timeout, temporary-output,
and process-group cleanup remain owned by Rapid's existing LTX sidecar adapter.

Block-streaming support is a separate compatibility axis. The first product
candidate may reject fast mode when its runtime cannot safely fuse the adapter
into a streamed model, but that rejection must name the unsupported runtime
mode, not the Mac chip.

## Rapid API and discovery

Proposed request control: `generation_mode=standard|fast`, initially defaulting
to `standard`. `fast` is valid only for LTX-2.5 and only when the served model
revision advertises a validated fast-stage capability. An unsupported explicit
request returns a clear 400/409-style capability error rather than silently
running standard mode.

`GET /v1/videos/capabilities` should advertise the available values, current
default, schedule evaluation counts, experimental status, and checkpoint
qualification revision. It must not advertise fast mode merely because the
host is an M3/M4 or has a particular memory size.

After the broader quality and portability gates pass, Atlas may change the
default to `fast` while retaining `standard` as a deterministic rollback. The
same model package and request value must mean the same numerical path on M2,
M3, M4, and later Apple Silicon generations.

The upstream research CLI now exposes separate experimental
`--fast-stage1-manifest` and `--fast-stage2-manifest` inputs. Rapid should map
these internal artifacts to one product-level `generation_mode=fast` only
after both packages pass qualification; users should not have to assemble a
machine-specific schedule themselves.

## Resource admission versus algorithm selection

Portable does not mean every memory configuration can load a 67.7 GB model
package. Existing model-level workload admission and low-RAM execution remain
responsible for determining whether a requested resolution and duration fit.
Once the base LTX-2.5 workload is admitted, hardware identity must not disable
or alter the fast schedule. If empirical memory shows the fast path has a
different envelope, express that as a measured workload budget, not a list of
chips.

Hardware may affect progress estimates and telemetry labels only. No quality,
schedule, sigma, adapter-strength, or fallback branch may inspect `M2`, `M3`,
`M4`, future chip names, GPU core counts, or a machine hostname.

## Qualification and rollout gates

1. Freeze 8-12 prompt-disjoint 10-second cases covering faces, hands, readable
   text, fast subject motion, camera motion, low light, fine texture, speech,
   impacts, ambience, and silence. Use at least two seeds for high-risk classes.
2. Produce blinded teacher/student individual MP4s plus a muted side-by-side
   for every case. Human non-inferiority is the release gate; latent error,
   brightness, detail, temporal, and audio metrics are diagnostics.
3. Verify exact schedule selection, base reload, audio/video coupling, malformed
   metadata rejection, explicit unsupported-mode errors, cancellation, timeout,
   and fallback behavior with automated tests.
4. Run the same model revision, adapter digest, prompt/seed cases, and runtime
   commit on representative M2, M3, and M4-or-later systems that satisfy base
   workload admission. Record wall time, peak memory, swap growth, crashes, and
   output hashes/metrics. Do not tune numerical settings per machine.
5. Ship as explicit experimental opt-in. Promote to default only after the
   cross-prompt quality gate and at least two materially different Apple GPU
   generations pass. Rollback is metadata/default removal; `standard` remains
   intact.

## Current assets and blockers

The resume-fixed candidate is archived outside the repository at
`/Volumes/RTL-2T/models-cold/ltx-stage2-distillation/resume-fixed-2026-09-14/`
with SHA-256
`04ed313c536fae4ad78732f8c403d7ac044614dcb8a95f0f5ae0e896e310855c`.
It is not yet a distributable product artifact.

The upstream base-bound validator landed on the feasibility branch at
`396667f`. It checks the external artifact digest, capability and schedule,
rank/alpha, exact base identity and immutable revision, transformer filename
and config fingerprint, runtime-contract major, qualification revision, and
LoRA pair shapes before mutation. It contains no hardware-name selection.

The stage-2-only runtime lifecycle and experimental CLI entry landed at
`c0da6b6`. The first transition loads the validated student, materializes its
video and audio outputs, releases it, and reloads the clean base for the final
correction. Standard mode is unchanged; fast mode rejects schedule overrides,
teacher capture, and unqualified additional-LoRA composition. Commit `7d4a37d`
adds the deterministic model-package builder. Commit `e3beea8` adds a
resumable, deterministically blinded Stage-2 qualification renderer; it does
not change inference behavior.

The product contract now also has a separate qualified-terminal capability at
upstream commit `280acb0`. A `ltx_stage2_terminal_v1` artifact declares the
two-sigma `[0.909375, 0]` schedule and executes only the learned terminal
evaluation; it cannot be confused with the three-sigma progressive artifact.
The runtime materializes and releases the terminal student without performing
an unnecessary base correction, then forces a clean base reload before a later
request. This path remains unavailable unless a terminal checkpoint passes the
same qualification gates.

Commits `d6084cc` and `164cdcd` add seeded original-step noise reproduction,
an exact ancestral target inverse, noise-coupled Stage-1 training, and paired
evaluation. The first stochastic pilot compresses original boundaries `0 -> 3`
while explicitly re-injecting original noise lane 0. The full upstream suite at
that point passes 659 tests with 22 skips.

The progressive Stage-2 ten-case suite has completed at 768x512, 241 frames,
24 fps, with 48 kHz stereo audio. Its anonymous means are SSIM 0.924042, PSNR
32.504 dB, and audio APSNR 167.519 dB; these are diagnostics while human blind
judgments remain pending. Review caught a prompt-label permutation caused by
mixing filesystem discovery order with a separately sorted filename list.
Upstream `b2054aa` now sorts dataset discovery deterministically and derives
render positions and review metadata from the same dataset instance. The A/B
media and anonymous metrics were unaffected, and the exported index was
corrected from the captured original order. The full upstream suite at
`fee2f51` passes 714 tests with 22 skips. That revision also validates source
checkpoint provenance before packaging and rejects non-qualified Stage-2
schedules at runtime.

Upstream `2a7685b` adds transformer-content binding. A real package build
against the MZR-3 Hugging Face snapshot completed in 0.43 seconds and resolved
the transformer identity to LFS object
`98d4c4d08ecd9e8d6cf1a836240a13bfc9d01e8e9ddc42a238e29e67229cf670`
without scanning the weight file. The complete suite passes 718 tests with 22
skips.

Path analysis over eight training and two prompt-disjoint validation
trajectories independently selected the same four-evaluation boundaries on
both splits: `[0, 3, 5, 7, 8]`, corresponding to sigmas
`[1.0, 0.98125, 0.909375, 0.421875, 0]`. The validation mean video/audio chord
error was 0.7099 versus 0.7353 for the second-ranked schedule. This selects the
`3 + 2 + 2 + 1` research candidate but does not replace decoded qualification.

The accepted adapter still needs completed multi-prompt qualification and an
immutable published model-repository revision before Rapid integration. A
scratch package has been produced for runtime smoke testing only; upload,
release, default changes, and public API disposition require explicit human
and Atlas authorization.
