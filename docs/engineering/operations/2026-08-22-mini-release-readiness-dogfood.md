# Mac mini release-readiness dogfood: post-v0.12.18 candidate

Dogfood ran on 2026-08-22 on a Mac mini with an Apple M2 Pro (10 CPU cores),
32 GB unified memory, and macOS 26.5.2. The clean task worktree started at
`00085528`, pulled `dcd6fb81` while testing, and finished on merged main
`9880a705`. The Python environment used Python 3.12.13, MLX/Metal 0.32.1,
mlx-lm 0.31.3, transformers 5.12.1, mlx-vlm 0.6.3, mflux 0.19.0, and
mlx-audio 0.4.3. `uv pip check` reported all 120 packages compatible.

This was a mini-scoped readiness pass, not a public release. It did not create
a tag, publish artifacts, notarize, or run the Studio/M3 release fleet.

## Blocking finding and resolution

The MLX 0.32.1 dependency move exposed a correctness bug in the live quantized
batch KV cache. Six numerical tests failed with errors in the hundreds or
thousands. The stored packed tensors, scales, and biases were correct; MLX
0.32.1's quantized kernels read a non-contiguous capacity-prefix view with the
wrong row stride. Making the compressed view contiguous restored a zero-diff
round trip without reconstructing the full bf16 history.

The fix landed in [PR #2234](https://github.com/raullenchai/Rapid-MLX/pull/2234)
at `9880a705`. Independent Codex review granted LGTM in round 1. The full-ci
label then ran the Python version matrix, Apple Silicon tests, five L1 model
smokes, Rapid Mac build and accessibility gates, and every named GUI golden
flow. All required checks passed. A post-merge mini rerun passed all 47 focused
quantized-cache tests and the clean-room release smoke.

Real-server A/B evidence on `qwen3.5-4b-4bit`:

| Mode | Workload | Result |
| --- | --- | --- |
| bf16 | 256 output tokens, short prompt, median of last two runs | 56.7 wall tok/s |
| int8 | same short workload | 55.1 wall tok/s (-2.8%) |
| bf16 | 9,034-token prompt + 256 output tokens | 27.9 s |
| int8 | same long workload | 30.2 s (+8.2% end-to-end) |
| int8 | four concurrent requests | 4/4 request sentinels preserved; no cross-row corruption |
| int4 | two concurrent requests | 2/2 request sentinels preserved |

Disconnecting a 3,000-token int4 stream incremented both cancellation metrics,
and the next request returned `RECOVERED`. The effective KV dtype metric matched
bf16, int8, and int4 in each server run.

## Automated gates

- Python unit gate excluding live integration clients: 18,401 passed, 79
  skipped, 25 deselected, 6 xfailed, and 1 xpassed in 293.44 seconds.
- Quantized batch cache after the fix and again after merge: 47/47 passed.
- Rapid Mac on final `9880a705`: 2,740 Swift tests passed; the complete app and
  embedded sidecar rebuilt successfully, strict deep codesign validation
  passed, and the release-smoke window launched at 1200x820. The embedded MLX
  0.32.1 quantized-cache round-trip smoke also passed.
- GitHub full CI for #2234: lint, type check, MLX bound guard, Python 3.10/3.11/
  3.12, Apple Silicon, five L1 model smokes, Rapid Mac build, and all named GUI
  golden flows passed.
- Clean-room working-tree install/import smoke passed both before and after the
  blocker merged.

## Live model and server matrix

| Surface | Evidence | Verdict |
| --- | --- | --- |
| Qwen3.5 4B text | Health/models, chat, SSE usage trailer, named tool call, bf16/int8/int4, four-way batching, cancellation recovery, Responses API, and Anthropic Messages API | Pass |
| Qwen3.5 9B + MTP | Real sidecar injection, deterministic generation, named tool call, and non-zero proposal/accept/token-saved metrics | Pass |
| Qwen3.5 9B vision | Forced MLLM lane described the supplied cheetah image correctly; serialized hybrid compatibility activated | Pass |
| Gemma 4 e2b vision | Real MLLM request described the DMG background, including text, arrow, placeholders, and colors | Pass |
| Nemotron-Labs-Diffusion 3B | AR text lane, non-streaming, SSE, four concurrent requests, and `RAPID_MLX_TRUST_REMOTE_CODE=0` vendored load | Pass |
| Whisper Small STT | Cold lazy load, WAV transcription, JSON/text/SRT/VTT, verbose segments, and word timestamps | Pass |
| Qwen3 ASR 0.6B | Cold lazy load and multilingual verbose transcription | Pass |
| Qwen3 TTS 1.7B 4-bit | Cold lazy load and WAV speech generation (24 kHz mono, 5.28 seconds) | Pass |
| FLUX.2 Klein 4B | Real 512x512 generation in 17.35 s; 1024x1024 default generation; input-sized image edit | Pass |
| Ornith 1.5 9B bf16 | Alias/info and safe pre-download disk rejection | Partial: weights require 16.7 GiB; host had 15.5 GiB free |
| Ornith 1.5 35B-A3B bf16 | Alias/info and safe pre-download disk rejection | Partial: 64.6 GiB download and model cannot be meaningfully loaded on this 32 GB mini |
| Qwen3-Coder-Next 80B 4-bit disk stream | Alias and safe pre-download disk rejection | Partial: cache contained metadata only; 41.8 GiB weights could not fit |

The three partial rows are environment limits, not silent passes. Their alias,
profile, size, and graceful error paths were tested; real inference remains for
a host with the weights and sufficient storage/memory.

## GUI dogfood

All 30 named GUI golden flows passed, and their app logs contained no fatal,
segfault, uncaught-exception, or nil-crash marker. The release app and real
sidecar passed fresh-install, cached quickstart and trade-up, download
progress, audio readiness, dictation, image generation,
multimodal attachments, settings persistence/MTP, update states, launch
integrations, model crash recovery, low-memory selection, resident-load
rejection, catalog integrity, and re-run setup flows.

Specific release-candidate checks:

- The STT picker exposed 10 deduplicated rows through accessibility. Every row
  had a display name, family badge, cache/download state, size, and a concrete
  Whisper/Parakeet/Qwen/SenseVoice tradeoff. Selecting Whisper Turbo changed
  the action to Download without starting a server implicitly.
- Image generation defaulted to 512x512 and sent `size=512x512`. Image editing
  intentionally sent no size; the backend derives output dimensions from the
  conditioning image to avoid mflux noise caused by mismatched latent sizes.
- `Run setup again` relaunched the app into Quickstart without resetting the
  existing telemetry choice.
- The DMG background asset presents the app on the left, Applications on the
  right, an orange left-to-right drag arrow, installation copy, aligned icon
  wells, and a finished product background. A manually created and mounted
  UDRW image contained the app plus `Applications -> /Applications`. Final
  Finder-layout validation was not possible because Finder on macOS 26 timed
  out before writing or reading the presentation `.DS_Store`; see #2240.

## Non-blocking follow-ups

- [#2233](https://github.com/raullenchai/Rapid-MLX/issues/2233): the image-route
  recovery error recommends the nonexistent `flux-schnell` alias instead of a
  current alias such as `flux2-klein-4b`.
- [#2238](https://github.com/raullenchai/Rapid-MLX/issues/2238): the release
  guide still documents the old same-minor/two-patch/TTY-only nudge behavior.
- [#2239](https://github.com/raullenchai/Rapid-MLX/issues/2239): the re-run setup
  confirmation title says it erases the Mac's Rapid state even though the
  operation resets onboarding state only.
- [#2240](https://github.com/raullenchai/Rapid-MLX/issues/2240): local DMG
  Finder-layout automation times out waiting for Apple Events on macOS 26;
  macOS 15 CI has not reproduced the failure.

## Release decision boundary

No tested mini or GUI surface has an open release blocker after `9880a705`.
Before publishing, Atlas still needs to run the canonical Studio/M3 release
fleet, choose the next version, open the dedicated version-bump PR, and let the
release artifact/notarization pipeline validate the exact distributables. The
three large-model partial rows above should remain explicit in that handoff.
