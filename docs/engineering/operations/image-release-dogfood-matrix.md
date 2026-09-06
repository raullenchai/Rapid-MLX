# Image release dogfood matrix

This is the acceptance contract for the built-in image aliases shipped by
Rapid-MLX. It is a release-operator checklist, not a claim that an old run
qualifies a new binary. Every release candidate needs evidence from its exact
wheel or Desktop sidecar and exact model revisions.

The automated release build currently exercises one real FLUX.2 Klein model
through the assembled Desktop sidecar. The remaining rows are explicit manual
release checks until equivalent artifact-bound automation exists. A historical
receipt proves that a path has worked; it never replaces the current-candidate
column.

## Candidate matrix

<!-- image-release-matrix:start -->
| Alias | Runtime family | Required operation | Exact-candidate proof | Historical real-weight receipt |
| --- | --- | --- | --- | --- |
| `flux2-klein-4b` | mflux / FLUX.2 Klein q4 | Generate + edit | Required; generation is automated in the release sidecar | [Mini release-readiness dogfood](2026-08-22-mini-release-readiness-dogfood.md) |
| `bonsai-image-4b-2bit` | Native Bonsai | Generate | Required manually | [Bonsai Image 4B 2-bit dogfood](../performance/2026-09-05-bonsai-image-4b-2bit-dogfood.md) |
| `z-image-turbo` | mflux / Z-Image | Generate | Required manually | [Initial real-weight qualification](https://github.com/raullenchai/Rapid-MLX/commit/056ebe79b9904dd432049a550b1bf6c3b298783d) |
| `flux-schnell` | mflux / FLUX.1 Schnell | Generate | Required manually | [FLUX.1 Schnell dogfood](../performance/2026-09-04-flux1-schnell-dogfood.md) |
| `sdxl-base` | Native SDXL | Generate | Required manually | [SDXL Base dogfood](../performance/2026-09-05-sdxl-base-dogfood.md) |
| `flux2-klein-4b-bf16` | mflux / FLUX.2 Klein BF16 | Generate + edit | Required manually | [Klein precision qualification](../performance/2026-09-04-image-weight-precision.md) |
| `hidream-o1-dev` | Native HiDream | Generate | Required manually | [HiDream-O1 Dev dogfood](../performance/2026-09-04-hidream-o1-dev-dogfood.md) |
| `sd35-large-4bit` | Native SD3.5 | Generate | Required manually | [SD3.5 Large dogfood](../performance/2026-09-05-sd35-large-dogfood.md) |
| `qwen-image` | mflux / Qwen Image | Generate | Required manually | [Pinned-checkpoint real server qualification](https://github.com/raullenchai/Rapid-MLX/pull/2157) |
<!-- image-release-matrix:end -->

## Required evidence per row

A row passes only when one evidence bundle records all of the following:

1. Candidate identity: release version, full source SHA, wheel or sidecar
   checksum, Apple chip, physical unified memory, macOS version, Python, MLX,
   and image-runtime versions.
2. Artifact identity: alias, repository, immutable revision, audited download
   bytes, and a successful offline completeness check. Auxiliary repositories
   count as part of the checkpoint.
3. Cold path: start the exact candidate with no other model resident; require
   `/health` to report `ready=true` and `engine_type=image`.
4. Default request: omit `steps`, call `/v1/images/generations`, require HTTP
   200, exactly one base64 payload, a decodable non-uniform RGB/RGBA PNG at the
   requested dimensions, and progress that reaches the catalog default.
5. Warm path: repeat in the same server, then restart offline from the verified
   cache. Record request time, peak process footprint, swap, and any memory
   warning. Both requests and the restart must succeed.
6. Recovery: cancel during denoising, require the request to terminate without
   an orphan process, then generate successfully from the same server.
7. Capability-specific path: aliases marked **Generate + edit** must also call
   `/v1/images/edits` with a real input image and verify that the output follows
   the instruction. Generation-only aliases must remain absent from the edit
   picker and reject the edit route.

Do not substitute a two-step or reduced-resolution import smoke for the
default-request result. Short probes are useful for lifecycle debugging, but
they do not establish product quality or the published memory floor.

## Product-level checks

After every row passes the API contract, validate the release-shaped Desktop
once against the same sidecar:

- the Images picker contains exactly the current atomic image roster;
- RAM fit, download size, operation mode, and default steps agree with
  `rapid-mlx models --json`;
- Download → Start → Generate reaches progress and a visible gallery result;
- changing aspect ratio changes the requested dimensions;
- edit-capable and generation-only picker partitions are correct;
- Stop/cancel returns to an actionable state and a subsequent render works;
- quitting Desktop leaves no candidate-owned server or model process.

The deterministic `image-generation` GUI journey covers these interaction
states with a fake sidecar on every relevant Desktop change. At least one real
model must still traverse the release-shaped Desktop for a release candidate;
the fake journey cannot prove model packaging, Metal execution, or PNG quality.

## Recording a release result

Store the completed evidence under `docs/engineering/operations/` or
`docs/engineering/performance/` and include:

```text
Candidate: <version> @ <full SHA>
Artifact: <wheel/sidecar path and SHA-256>
Host: <chip>, <RAM>, <macOS>, <power mode>
Alias: <alias> -> <repo>@<revision>
Cold default: <size>, <steps selected>, <seconds>, <PNG bytes>, <peak>, <swap>
Warm default: <seconds>, <PNG bytes>
Offline restart: PASS/FAIL
Cancellation recovery: PASS/FAIL
Edit path or generation-only rejection: PASS/FAIL
Desktop: PASS/FAIL/not run with reason
Verdict: PASS/BLOCKED, with issue link for every failure or waiver
```

Keep raw logs and generated images as release artifacts when practical. Do not
commit model weights, machine-specific cache paths, credentials, or raw chat
transcripts.
