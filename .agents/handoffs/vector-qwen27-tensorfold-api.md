# Vector: Qwen3.8-27B experimental TensorFold HTTP adapter

Status: in progress
Owner: Vector
Branch/worktree: vector/qwen27-tensorfold-api at /private/tmp/harbor-desk-qwen27-api
Base: fc804cf7f35bcfc7102be5ecb7a593be0f2038a9

Goal: add an explicit opt-in TensorFold provider to Rapid's dedicated B=1 DFlash text chat server, preserving its security, request admission, streaming/nonstream, cancellation and error contracts.

Scope: exact pinned Vontra 27B + DFlash pair; /v1/chat/completions text only; health/models; reasoning/content split; usage and full generated token capture; optional dependency. Tools, grammar, media, batch, and standard BatchedEngine are excluded and rejected.

Default behavior remains unchanged. TensorFold owns its model, cache, scheduler, and lifecycle. Rapid owns HTTP validation, prompt rendering, response mapping, and stream postprocessing.

Verification: focused provider/unit tests; HTTP ASGI integration tests for stream/nonstream, rejection, usage, cancellation/error; existing DFlash tests; exact-pair MZR HTTP smoke with token IDs/hash and clean shutdown.
