# License review — Ternary-Bonsai-27B base (Marvin's Garden)

**Status:** evidence gathered 2026-09-21; conclusion Apache 2.0 (commercial OK).
**Formal sign-off:** pending — legal should confirm the reading below, but the
license text is the standard OSI text and the obligations are mechanical.

## What we ship

Marvin's Garden = prism-ml **Ternary-Bonsai-27B** (2-bit MLX build) + our own
LoRA adapters + our own code/data. Two third-party layers:

1. `models--prism-ml--Ternary-Bonsai-27B-mlx-2bit` — the conversion we run on.
2. Upstream `Qwen3.6-27B` (Alibaba Cloud) — the base the Bonsai weights were
   distilled from (per NOTICE).

## Evidence (files in the snapshot we actually load)

- `LICENSE.txt`: verbatim Apache License 2.0 (OSI text, no additional terms,
  no field-of-use restrictions, no copyleft).
- `NOTICE.txt`: "This software is copyright 2026-present Prism ML, Inc. It is
  available under the Apache 2.0 license. If you publicly deploy or
  redistribute this software, we would appreciate attribution such as:
  'Created using Bonsai by Prism ML.' … built from Qwen3.6-27B, Copyright
  2026 Alibaba Cloud, … Apache 2.0 License."

## Obligations under Apache 2.0 (mechanical, non-blocking)

- Keep `LICENSE.txt` + `NOTICE.txt` with any redistribution of weights.
- Attribution line "Created using Bonsai by Prism ML." in public deploy pages
  (NOTICE says "would appreciate" — courtesy, but we will include it).
- State significant changes (our LoRA adapters are our own work product).
- PATENT grant: Apache 2.0 §3 covers the Work as contributed by Prism ML.

## Residual checks before signing off

1. Confirm the upstream Prism ML repo page carries the same license (the
   converted mirror we use could in theory diverge from the source of truth).
2. Note `models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit` exists (a newer
   generation) — out of scope for this review, re-review if we adopt it.
3. Our own artifacts (adapters, code, data, eval sets) carry
   `SPDX-License-Identifier: Apache-2.0` headers — consistent, no conflict.

## Decision requested

Adopt Apache 2.0 compliance checklist (NOTICE redistribution + attribution)
and treat the base-model license as **no longer a release blocker**, pending
formal legal confirmation.
