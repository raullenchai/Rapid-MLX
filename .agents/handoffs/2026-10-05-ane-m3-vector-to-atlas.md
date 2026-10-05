# Vector → Atlas: M3 Ultra ANE/GPU Qwen 27B experiment

Branch: `vector/ane-m3-27b-experiment-20261005`. Host: Studio.

Verified: oMLX 0.7.0 at pinned commit, same 4-bit Qwen3.8-27B snapshot, no cache or speculative decoding. On the shared M3 Ultra, ANE/GPU prefill reduced ~9.8K-token TTFT from 28.6 to 22.9–23.2 seconds and ~2.6K-token TTFT from 7.42 to 6.0–6.56 seconds. Built-in profile counted native work on both ANEs. Load-time footprint rose about 12.8 GB; decode speed was essentially unchanged. Two complete quality responses matched byte-for-byte. Full method, data and caveats: `docs/engineering/performance/2026-10-05-m3-ultra-ane-qwen27b.md`.

Unresolved: no Rapid integration; no M4/M5/M6 hardware measurements; no broad quality or agentic corpus; no concurrency or prefix-cache A/B. The candidate uses private Apple interfaces and approximate INT8. Studio was shared and had pre-existing swap, so a quiet-host repetition remains necessary.

Next action: Atlas decides whether a private-API experimental path is acceptable. If yes, Vector can scope a Rapid opt-in prototype gated by model/quantization/chip/RAM and run full quality plus M4/M5/M6 tests. Do not mark production-ready from this experiment.
