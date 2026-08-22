# First-run GUI friction dogfood — 2026-08-22

## Scope and environment

- Source: `origin/main` at `365a29dd`
- Host: Studio, Apple M3 Ultra, 256 GB unified memory
- Install surface: locally assembled release DMG
- Persona: isolated bundle identifier, fresh preferences, fresh Application
  Support, and a cold Hugging Face cache
- Journey: DMG install, telemetry consent, onboarding, first model download,
  chat, tools, documents, Images, Text to Speech, restart, Launch, and Settings

This document records the nine dogfood findings selected for the first-run GUI
friction pass. It intentionally keeps the evidence and acceptance criteria in
the repository instead of relying on a chat transcript.

## Findings and acceptance criteria

### F1 — Images download completion leaves contradictory state

After the 4.3 GiB `flux2-klein-4b` download completed, the download strip said
`Downloaded — ready to load`, while Images still said the model was not
downloaded, kept the `Download` action, and disabled Generate. Waiting and
pressing Download again did not recover; restarting the app did.

Acceptance:

- A successful image-model download immediately invalidates the cached model
  inventory and transitions Images to `Start` without an app restart.
- A completed download cannot coexist with a `not downloaded` readiness state.
- Regression coverage exercises the real completion notification/state path.

### F2 — Images start fails with an opaque resident HTTP 500

After restart recognized the model, `Start` consistently failed with
`The model could not be kept resident (HTTP 500)` on a 256 GB machine. Retry
produced the same result. The message exposed transport detail and offered no
actionable diagnosis.

Acceptance:

- Image models use the image-generation runtime instead of the text-model
  residency endpoint.
- The default image model can start and generate pixels from the bundled
  sidecar in a cold persona.
- Any remaining start failure is translated into an actionable user message;
  raw HTTP status may appear only as secondary detail.

### F9 — Download ETA is unstable during early sampling

The 633 MB onboarding download jumped between approximately 1 and 12 minutes
remaining as the first few speed samples arrived.

Acceptance:

- ETA is hidden until enough time and bytes have been sampled for a stable
  estimate.
- Once shown, short-lived throughput drops do not cause multi-fold ETA jumps.
- Bytes and current speed remain visible while ETA is calibrating.

### F10 — Cancel download looks disabled

The onboarding `Cancel download` action used low-contrast secondary styling
that visually resembled disabled text.

Acceptance:

- Cancel is visibly interactive, keyboard accessible, and destructive without
  visually competing with the primary download progress.
- AX label/help continue to describe the consequence.

### F11 — Completed download strip contradicts resident state

The chat model remained represented as `Downloaded — ready to load` even while
the sidebar and footer reported it Resident/Ready. Completed strips also piled
up after image and TTS downloads.

Acceptance:

- Completed strips disappear when their alias becomes the running model.
- Non-running successful downloads may remain briefly as a useful `Start`
  affordance, then auto-dismiss.
- Multiple completed jobs do not permanently consume the bottom of the app.

### F12 — Onboarding completion prompt blocks the first message

The completion prompt appeared over the composer/model picker immediately
after the first model became ready and asked the user to star the repository.

Acceptance:

- The first chat composer is unobstructed and focused after onboarding.
- Repository-star promotion moves to a non-blocking empty-state or delayed
  moment and is never layered over the composer.

### F19 — Image starter prompts are clipped without discoverability

Four one-line starter pills overflowed horizontally. The final prompts were
clipped, with no visible scroll affordance.

Acceptance:

- Every starter prompt is readable without hidden horizontal scrolling at the
  minimum supported window width.
- Layout wraps or uses a grid while retaining semantic AX identifiers.

### F20 — Images repeats download messaging instead of explaining value

The empty state repeated the model name, 4.3 GiB size, and download requirement
across the center stage, readiness banner, and disabled composer. It did not
explain expected generation time, local/offline behavior, or what the starter
model is good at.

Acceptance:

- One primary readiness surface owns download/start status and size.
- The empty state explains the outcome in plain language and avoids duplicate
  model/runtime jargon.
- The disabled composer gives a short action-oriented reason only.

### F27 — Launch exposes an expert wall at top-level navigation

Launch presented base URLs, API keys, and a long list of shell commands for
Claude Code, Codex, Cline, Continue, Cursor, LangChain, and others in one view.
When the selected chat model was stopped by TTS, every integration action was
disabled and the page became a dense dead end.

Acceptance:

- Launch opens with a beginner-readable purpose statement and a short set of
  common clients.
- Advanced endpoint/key details and the full integration catalog require an
  explicit disclosure action.
- When no chat model is running, the primary action is `Start <model>` and
  disabled command walls are not rendered.

## Verification journey

The post-fix dogfood must repeat the same cold-persona sequence:

1. Mount the release DMG and inspect the install surface.
2. Complete telemetry and onboarding; download the starter model.
3. Verify stable progress, visible cancel, unobstructed first composer, and no
   contradictory completed strip.
4. Download `flux2-klein-4b`; verify immediate `Start`, start it, generate a
   512×512 image, and verify non-empty pixels.
5. Return to chat, switch/start a text model, and verify history survives.
6. Open Launch with and without a running chat model and inspect both beginner
   and advanced disclosure states.
7. Run focused Swift tests plus the relevant GUI golden flows.
