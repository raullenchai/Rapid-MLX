# LTX-2.5 fast stage-2 qualification

Date frozen: 2026-09-14

Status: capture in progress; no release or default-on claim

This suite qualifies the portable fast stage-2 candidate described in
`docs/engineering/decisions/2026-09-14-ltx25-portable-fast-stage2.md`. It is
prompt-disjoint from the training and development examples and deliberately
stresses faces and hands, readable text, fast motion, camera motion, low light,
fine texture, speech synchronization, impact synchronization, ambience, and
near silence.

## Frozen inputs

- Manifest: `manifest.jsonl`
- Resolution: 768x512
- Duration: 241 frames at 24 fps (approximately 10 seconds)
- Base model: `MrMofer/ltx-2.5-mlx-q8`
- Base revision: `f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`
- Candidate adapter SHA-256:
  `04ed313c536fae4ad78732f8c403d7ac044614dcb8a95f0f5ae0e896e310855c`
- Fast schedule: learned `0.909375 -> 0.421875`, then unchanged base
  `0.421875 -> 0`
- Standard schedule: three base stage-2 transformer evaluations

The first capture host is MZR-3, a 48 GiB Apple Silicon Mac. Host identity is
recorded as benchmark evidence only; it does not select weights, schedules, or
numerical behavior.

## Required evidence

For each manifest row, retain the standard and fast MP4s, a randomized blind
mapping, a muted side-by-side comparison, generation timings, peak memory,
swap growth, exit status, and output digests. The reviewer must grade semantic
content, identity and geometry, motion, brightness and detail, temporal
stability, and audio-event synchronization before the mapping is revealed.

The candidate does not pass on aggregate latent error alone. Every severe
semantic loss, synchronization regression, crash, or incompatible artifact
must fail the gate. Lesser preferences are recorded per case and evaluated as
a predeclared non-inferiority result rather than hidden by averaging.

## Portability matrix

After the prompt suite passes, rerun the same immutable base revision, adapter
digest, runtime commit, manifest rows, and schedule on admitted M2, M3, and
M4-or-later systems. Do not tune numerical settings by chip. Record machine
memory and GPU configuration only to explain latency and resource behavior.

An inability to admit the base 67.7 GB workload is a capacity result, not an
algorithm compatibility failure. A machine that admits the standard workload
must receive the same fast capability and validation rules.

## Reproduction notes

The capture and render commands, runtime commit, randomized mapping, per-case
measurements, and reviewer decisions will be added to `RESULT.md` after the
current run completes. Until then this directory freezes inputs only and must
not be cited as completed qualification evidence.
