# Vector handoff: Qwen3.8-27B opt-in decoder proof

## PR-start FYI (pending delivery)

Recipients: Atlas, Pixel, Harbor, Echo

- Intention: prove a narrow, explicit Qwen3.8-27B TensorFold decoder adapter while preserving Rapid's API and output parsing.
- Owner/host: Vector; Studio implementation, MZR pinned-model validation.
- Branch/worktree: `vector/qwen27-tensorfold-optin`; `/private/tmp/harbor-desk-qwen27-optin`; base `origin/main@8a9e46fc0`.
- Allowed scope: optional dependency and version/capability gate; B=1 text chat stream/non-stream; cancellation, terminal error and cleanup contracts; fake tests plus pinned Qwen3.8-27B/DFlash2 smoke.
- Non-goals: no default enablement, server replacement, cache sharing, kernel transplant, multimodal/embeddings/audio, additional model families, public performance promise, deployment or release.
- Expected affected areas: speculative backend adapter/config, model load selection, engine request bridge, optional dependency metadata, focused tests and private benchmark notes.
- Verification: default-path regression tests; fake lifecycle/stream/cancel/gate tests; pinned same-backend serial/drafted token exactness; natural EOS, repair, tool parsing, cancellation and cleanup; per-task timing/memory reported with exact configuration.

Agent messaging was unavailable in the current tool set, so this records the required non-blocking start FYI for later delivery.

## Completion FYI (pending delivery)

Recipients: Atlas, Pixel, Harbor, Echo

- Outcome: isolated programmatic adapter proof passed for the one pinned Qwen3.8-27B pair. The proof remains unconnected to CLI/EngineCore and cannot change defaults.
- Exact pair: `Vontra/Qwen3.8-27B-MLX-4bit@70ae7fac63274ff2eac54152031433374cb80f2f` plus `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; TensorFold 0.5.0, MLX 0.32.3, arm64 macOS.
- Unit evidence: `python3 -m unittest -v tests.test_tensorfold_qwen27_backend`, 8/8. Includes exact capability gates, unsupported request rejection, ordered terminal events, cancellation and full-queue disconnect shutdown under one second.
- MZR smoke: exact-width 16 load gate passed. Same-backend greedy serial/drafted natural-EOS outputs matched: four tokens, TensorFold token SHA prefix `e157733beb64`, adapter content SHA `658f8c5e7bfb4cee0fb88d845a557f2ee8e8d9ed75c36ae91c80b59912c88b06`, finish `stop`. Drafted accepted 3/7 rows. Cancellation after first delta ended `RequestCancelled`, left no active adapter request, and the exact smoke process count after exit was zero.
- Timing: serial TTFT/total 0.331/0.524 s; drafted 0.325/0.485 s. This four-token smoke establishes engagement and correctness only, not a performance claim.
- Artifact: MZR `/private/tmp/harbor-desk-qwen27-adapter-smoke/smoke2.log`, SHA-256 `63fba40bcc65c103e078103c2ef4b2977af5213650187d3ab8b2f8b378ae5a10`.
- Known boundary: TensorFold 0.5.0 internal builder calls are concentrated in one exact-version adapter. TensorFold owns all model/scheduler/cache state. Tools, multimodal and unknown sampling fields are rejected. Route/EngineCore wiring, parser parity, concurrent lanes and any alias mapping are later separately reviewed work.
- Rollback: do not install/select the optional adapter; no persisted Rapid state or cache format changed.
