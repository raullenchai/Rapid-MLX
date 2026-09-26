# Atlas handoff — general Computer Use and System One

Date: 2026-09-25
State: active experiment
Branch: `atlas/general-cue-mvp`
Host: Studio

## Verified

- Rapid's pinned mlx-vlm 0.7.2 contains native Muse Glimmer perception.
- The real 4-bit Muse checkpoint loaded in the MLLM lane and processed live
  macOS screenshots.
- The pinned Meta `metacua` loop has Screen Recording and Accessibility
  permission on Studio.
- The adapter fixes dotted tool names, nested screenshot tokenization, and
  polluted Muse message replay.
- Laya CPU candidate ranking scored 7/8 on the checked-in small probe at
  76.2 ms mean HTTP wall time.
- The zero-shot binary gate scored 7/10 and is not ready for active use.
- Live Muse turns ranged from 14 to 32 seconds early in the run and later
  reached 85 to 148 seconds under concurrent Studio load.

## Risks

- No end-to-end desktop task completed in the measured runs.
- The candidate probe is small and hand-written; it is useful for falsification,
  not a product quality claim.
- Meta Muse product parity also needs isolation, connectors, durable tasks,
  confirmations, and cross-device session behavior.
- The active `guard` mode is experimental and must remain off by default.

## Next action

Collect replayable CUA traces with labeled candidate actions, add one-turn
multi-candidate generation, and compare baseline task success with Laya shadow
ranking. Separately benchmark the qualified 8-bit DFlash pair on visual turns
when the Studio GPU is available without the concurrent model evaluation.
