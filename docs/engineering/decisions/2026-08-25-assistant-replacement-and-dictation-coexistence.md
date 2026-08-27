# Assistant replacement and dictation coexistence

## Status

Accepted for the 0.13.1 implementation track. Desktop wiring is separate.

## User contract

Changing the Desktop assistant model does not restart the server. The caller
chooses one explicit policy for work already owned by the current assistant:

- `reject` (default) leaves the current assistant untouched when it is busy;
- `wait` closes new assistant admission, drains admitted/running/queued work,
  and then replaces it;
- `abort` closes admission, terminates admitted/running/queued work, and then
  replaces it. Streaming and non-streaming callers both receive a terminal
  cancellation signal.

Speech-to-text and text-to-speech are auxiliary audio lanes. They do not join
the `assistant` replacement group. A completed assistant replacement changes
the model worker through the existing audio-worker handoff transaction while
preserving the audio lane's loaded model and lifecycle state. If audio work is
active, the handoff fails closed: the assistant remains available, the audio
request reaches its original terminal result, and no worker is stopped.

When dictation is enabled, its speech-to-text lane is process-wide protected
state: every assistant load, replacement, reload, and switch back preserves the
loaded STT engine. The residency response must show the assistant record and
the STT lane together, and dictation must remain usable after each transition.
An assistant lifecycle operation never calls an audio-lane unload path.

## Ownership and state

The inference engine owns assistant admission and scheduler state. Its
lifecycle snapshot exposes `paused`, `pause_mode`, and admitted, queued,
running, and total active request counts. The residency manager serializes the
replacement transaction and publishes that engine-owned truth through
`GET /v1/models/residency`.

The audio-worker dispatcher remains the single source of truth for auxiliary
lane residency and active work. The residency response appends its
`audio_lanes` snapshot without folding audio activity into assistant counters.
This keeps replacement policy scoped to the selected lifecycle group.

## Transaction boundary

1. Materialize the replacement through the existing runtime serving-lane
   resolver, but do not publish it.
2. Close admission on every old assistant engine and reach the selected
   reject/drain/abort boundary.
3. Acquire the existing primary/audio-worker handoff lease.
4. Publish the replacement as primary and retire the old primary first. Until
   that stop succeeds, failure restores the old primary and audio-worker lease.
5. Successful primary retirement is the commit point. Commit the worker
   handoff, remove every quiesced sibling from routing, and treat any later
   sibling stop failure as cleanup rather than rolling routes back to an engine
   that may already be stopped.
6. Before the commit point, cancellation or failure discards the replacement,
   reopens old assistant admission, and rolls back the audio-worker lease.

This decision does not invent capacity, idle-TTL, audio-lane, scheduler, or
Desktop policy. Enforcing the same protection under a configured process memory
ceiling requires the separate shared auxiliary-role budget: it must charge the
resident STT role, mark it ineligible for assistant-driven eviction, and reject
an assistant admission with an actionable capacity conflict when both cannot
fit. Desktop presentation of a downgrade or model choice remains a separate UI
contract. Until that budget lands, this decision guarantees lifecycle
coexistence but does not claim combined LLM+STT capacity enforcement.

## Verification

Contract tests cover admission races, queued/running truth, wait/abort terminal
behavior for streaming and non-streaming callers, replacement rollback, audio
work blocking a worker handoff under every assistant replacement policy, and
speech-to-text serving before and after successful assistant replacement,
reload, and switch-back sequences. Release evidence additionally exercises a
fresh process with cached chat and STT checkpoints; mocked lifecycle tests do
not substitute for that model-worker validation.
