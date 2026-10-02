# Performance knowledge

For every durable result, record the commit, hardware, OS, model, precision,
context length, concurrency, warmup, exact command, relevant environment, raw
summary statistics, and limitations.

## Reports

- [Mac mini model matrix: Qwen3.5 4B, Gemma 4 26B, and Qwen3.8 27B](2026-08-21-mac-mini-model-matrix.md)
- [Long-context and service-prefill study (M2 Pro, 2026-08-22)](2026-08-22-long-context-service-prefill.md)
- [FLUX.1 schnell image-generation dogfood](2026-09-04-flux1-schnell-dogfood.md)
- [HiDream-O1 Dev server dogfood](2026-09-04-hidream-o1-dev-dogfood.md)
- [FLUX.2 Klein explicit bf16 execution path](2026-09-04-image-weight-precision.md)
- [Bonsai Image 4B 2-bit dogfood](2026-09-05-bonsai-image-4b-2bit-dogfood.md)
- [Absorbed MLA for multi-token verification](2026-09-05-mla-absorbed-verify.md)
- [Qwen Image Edit q8 release dogfood (2026-09-05)](2026-09-05-qwen-image-edit-dogfood.md)
- [Qwen3.8 Flash-Next K=2 indexed-QSA requalification](2026-09-05-qwen38-k2-indexed-requalification.md)
- [Stable Diffusion 3.5 Large Server and Desktop-path dogfood](2026-09-05-sd35-large-dogfood.md)
- [SDXL Base Server and Desktop-path dogfood](2026-09-05-sdxl-base-dogfood.md)
- [Qwen4 fp32-input fast RMSNorm qualification](2026-09-06-qwen4-fast-rmsnorm.md)
- [Qwen3.8 MTP GDN verify fusion](2026-09-07-qwen38-mtp-gdn-verify.md)
- [NeoHorse 1 9B Chat qualification](2026-09-08-neohorse-9b-chat-qualification.md)
- [Qwen3.8 27B Abliterated qualification](2026-09-09-qwen38-27b-abliterated-qualification.md)
- [DeepSeek V4.1 Flash DSpark mixed-precision experiment](2026-09-11-deepseek-v41-dspark-mixed-precision.md)
- [GLM-5.3-Flash runtime comparison and first Rapid optimization](2026-09-11-glm53-flash-runtime-comparison.md)
- [GLM-5.3 real-task MTP qualification](2026-09-12-glm53-real-task-mtp.md)
- [GLM-5.3 RMQ MVP: target-matched Apple quantization](2026-09-12-glm53-rmq-mvp.md)
- [MiniCPM5 2B small-agent harness A/B — 2026-09-13](2026-09-13-minicpm5-small-agent-harness-ab.md)
- [Qwen3.8 copy-draft for sampled and cache-reusing requests](2026-09-13-qwen38-copy-draft-sampled.md)
- [Qwen4 QSA stage-one selector qualification](2026-09-13-qwen4-qsa-stage1-selector.md)
- [K2 Horizon 7B performance qualification](2026-09-14-k2-horizon-7b-qualification.md)
- [Personal Intelligence top-model qualification](2026-09-15-personal-intelligence-top-model-qualification.md)
- [Personal Intelligence remaining-build qualification](2026-09-16-personal-intelligence-remaining-builds-qualification.md)
- [Gemma 4 26B-A4B assistant-sidecar MTP qualification](2026-09-20-gemma4-assistant-mtp.md)
- [Qwen Image 2.1 source-environment dogfood (2026-09-21)](2026-09-21-qwen-image-2.1-dogfood.md)
- [Qwen Image 2.1 8/16 GB feasibility spike (2026-09-25)](2026-09-25-qwen-image-2.1-low-memory-spike.md)
- [Qwen 27B pool customer experience, 2026-09-30](2026-09-30-qwen-pool-customer-ux.md)
- [Muse-Glimmer 30B 8-bit DFlash qualification](muse-glimmer-30b-dflash-qualification.md)
- [Contention-aware prompt admission benchmark](scheduler-admission-1950.md)
- [Telemetry v2 inference-path overhead](telemetry-v2-inference.md)

Superseded reports are removed once no live code, tests, CI, or docs reference
them; retrieve older ones from git history.

