<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**Private-by-construction product telemetry** — 0.15.0 moves anonymous,
metadata-only reporting to the v2 PostHog pipeline. The retired engine v1
transport and its per-request wire are gone; `rapid-mlx telemetry status`
shows consent, the live upload gate, build provenance, and install identity,
while `off`, `reset-id`, and the exact v2 `preview` make control auditable.
