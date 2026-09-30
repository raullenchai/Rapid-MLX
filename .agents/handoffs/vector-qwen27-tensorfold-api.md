# Vector: Qwen3.8-27B experimental TensorFold HTTP adapter

Status: bounded Phase 2 proof complete; awaiting Atlas PR disposition
Owner: Vector
Branch/worktree: vector/qwen27-tensorfold-api at /private/tmp/harbor-desk-qwen27-api
Base: fc804cf7f35bcfc7102be5ecb7a593be0f2038a9

Goal: add an explicit opt-in TensorFold provider to Rapid's dedicated B=1 DFlash text chat server, preserving its security, request admission, streaming/nonstream, cancellation and error contracts.

Scope: exact pinned Vontra 27B + DFlash pair; /v1/chat/completions text only; health/models; reasoning/content split; usage and full generated token capture; optional dependency. Tools, grammar, media, batch, and standard BatchedEngine are excluded and rejected.

Default behavior remains unchanged. TensorFold owns its model, cache, scheduler, and lifecycle. Rapid owns HTTP validation, prompt rendering, response mapping, and stream postprocessing.

Verification: focused provider/unit tests; HTTP ASGI integration tests for stream/nonstream, rejection, usage, cancellation/error; existing DFlash tests; exact-pair MZR HTTP smoke with token IDs/hash and clean shutdown.

## Result

- Final code: `1a618ce3a33184197cca715ed3f7b3b982e89a80`.
- Tests: 217 passed, 1 real-model skip across the provider, Phase 1 adapter,
  speculative config, and complete DFlash integration suite.
- MZR HTTP evidence: `/private/tmp/harbor-desk-qwen27-integration/http-smoke/`.
  Stream and non-stream returned `OK`, stop, usage 17/2/19, and the exact token
  IDs `[3793, 248046]` (SHA-256 `9b3560a21927dbfe85843c5ef86796c3b0cd3d80dde20a801eaba58b9caf7a58`).
- Stop string, unsupported media, auth, body limit, admission, disconnect recovery,
  and exact process cleanup passed. A first smoke exposed visible EOS text in the
  stream; commit `1a618ce3a` fixes it and pins the regression test.
- This remains an experimental serial text endpoint. It is not evidence for
  standard BatchedEngine parity, tools/grammar/media support, or a speed claim.
