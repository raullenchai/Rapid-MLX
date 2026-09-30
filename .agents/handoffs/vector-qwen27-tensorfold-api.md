# Vector: Qwen3.8-27B experimental TensorFold HTTP adapter

Status: product profile implemented on PR #3929; performance and independent review gates pending
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

## Delivery risks and remaining gates

- **P0 — performance qualification is pending.** The HTTP proof establishes
  correctness and lifecycle behavior only. Do not claim a speedup, recommend
  this lane for production, or promote it beyond experimental opt-in until the
  separately owned exact-pair benchmark records reproducible before/after
  throughput, latency, acceptance, memory, and output-quality evidence.
- **P0 — the supported environment is intentionally narrow.** Startup rejects
  anything except Apple Silicon macOS, TensorFold 0.5.0 from commit
  `9cd52ab4daba68ddd09be89be8f23ad43175e821`, and MLX 0.32.3. The target and
  drafter must be the exact cached snapshots qualified by their full revisions;
  broad model-family compatibility is unproved.
- **P1 — installation is source-based and opt-in.** The
  `tensorfold-qwen27` extra installs a revision-pinned Git dependency, so it
  requires Git and network access (or an equivalent pre-populated installer
  cache) and is unsuitable for the normal wheel-only/offline install path.
  TensorFold and the DFlash weights remain separately distributed under their
  own licenses; the repository's NOTICE retains the required attribution.
- **P1 — one serial request lane only.** TensorFold owns the model, cache,
  scheduler, and cancellation lifecycle. Rapid caps admission at one active
  request; tools, grammar, media, general batching, and standard BatchedEngine
  integration remain explicit non-goals for this PR.
- Before merge disposition: complete the independent performance gate, finish
  the scope-locked PR review loop, and retain the exact dependency/model pins.

## Evidence archive

The independent HTTP archive manifest was regenerated after clean shutdown so
`SHA256SUMS` now records the final `server.log` digest
`954790b2768471b42131c971058d23d918a261f9773f9f1876bb836bd14993ef`.

## Productization update (2026-09-30)

- Commit `8f3582696` adds the experimental `qwen3.8-27b-tensorfold` profile.
  It resolves the exact target and DFlash2 snapshots at immutable revisions
  through the default Hugging Face cache and auto-selects the TensorFold
  backend. The measured product floor is 48 GB; the ordinary recovery alias is
  `qwen3.8-27b-4bit`.
- `/v1/models/{id}` publishes `speculative_decoding.backend=tensorfold`, active
  runtime state, `unsupported_features=[tools,media,grammar]`, the recovery
  alias, and the memory floor. `/v1/status` and `/healthz` publish the paired
  identities and readiness/capability detail.
- Packaging metadata now validates and a wheel builds successfully. Ruff is
  clean; focused TensorFold HTTP/backend and DFlash eligibility tests pass
  (49 tests). The prior exact-pair HTTP acceptance remains valid because no
  generation/provider hot-path behavior changed.
- Reference check: the dedicated serial-provider lifecycle remains consistent
  with the primary serving precedents' explicit alternate-engine boundaries;
  existing Rapid paired-artifact download and model-profile patterns were
  reused. PR #3926 is complementary BatchedEngine dense-matmul work and has no
  code overlap; do not copy or stack its kernels into this provider.
- Remaining gates: independent performance owner must qualify the product
  profile; spark2 Codex review cannot currently start because that host's Codex
  refresh token returns HTTP 401. Re-run the scope-locked review loop after the
  host is re-authenticated. Do not merge or release without Atlas/human approval.
