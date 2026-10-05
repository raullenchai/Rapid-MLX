# Apple Neural Engine inference survey (2026-10-05)

Owner: Atlas (architecture); proposed experiment: Vector. Status: survey only; no Rapid ANE implementation or Rapid benchmark in this document.

## Current Rapid baseline

Rapid's text inference depends on MLX / mlx-lm (`pyproject.toml`). MLX lists CPU and GPU as its supported devices, and its maintainer explicitly confirmed no ANE support. Thus a normal Rapid-MLX LLM request does **not** use the Neural Engine merely because the Mac has one. Unified memory and Metal GPU execution explain the current path. Core ML may choose ANE for a separate model, but selecting `computeUnits = .all` does not prove ANE execution: inspect placement with Xcode's Core ML performance report or `MLComputePlan`.

Sources: [MLX README](https://github.com/ml-explore/mlx), [MLX ANE issue](https://github.com/ml-explore/mlx/issues/1229), [Apple placement guide](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode).

## Relevant implementations

| Project | What has been demonstrated | Fit for Rapid |
| --- | --- | --- |
| [Apple `ml-ane-transformers`](https://github.com/apple-aiml-research/ml-ane-transformers) | Public Core ML conversion and ANE-friendly layouts; Apple's reported up-to-10x speed / 14x peak-memory improvement is for **DistilBERT vs its baseline**, not autoregressive 27B decode. | Reuse layout, conversion and placement-testing ideas for small transformer lanes. No drop-in MLX backend. |
| [ANEMLL](https://github.com/Anemll/Anemll) and [ANEMLL Bench](https://github.com/Anemll/anemll-bench) | End-to-end conversion, Swift/Python inference and pre-converted Llama, Qwen, Gemma examples. Its published recommended contexts are mostly 512–2048, verified up to 4K for some models. Bench measures ANE primitives / models, not Rapid workloads. | Best public-API prototype for a small standalone model or speech/embedding lane. Model conversion, separate artifacts, KV handling and context limits prevent a universal toggle. |
| [coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli) | Core ML Llama 2 7B demo; reported ~7 tok/s M1 Max, ~14 tok/s M3 Max, with ANE power numbers. Shows IOSurface-based KV I/O and chunking. | Useful reference for copy costs and power measurement; not evidence that ANE beats MLX on the same checkpoint. |
| [oMLX hybrid prefill](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md) | Splits Qwen 3.5/3.6/3.8 MLP and selected GDN projections between ANE and GPU while keeping decode on GPU. Its documented 2048-token M3 Ultra paired measurement is 334.9 to 454.3 prefill tok/s (+35.6% throughput), with single-prompt logit checks. | Closest to our Qwen 27B. Experimental, opt-in, uses undocumented AppleNeuralEngine APIs and lossy INT8 re-quantization. Their [release notes](https://github.com/jundot/omlx/releases) describe long-prompt recurrent-state quality fixes; independent Rapid quality testing is mandatory. |
| [mlx-serve hybrid prefill](https://github.com/ddalcu/mlx-serve/releases) | Reports 19–35% faster 16K prefill for Qwen 3.5/3.6/3.8 on M1–M4, unchanged decode speed, and ~11 GB extra RAM for 27B; first compilation 1–2 minutes. Their [architecture note](https://github.com/ddalcu/mlx-serve/blob/main/CLAUDE.md) confirms private ANE API and lossy INT8/FP16 offload. | Independent evidence for the same seam, not a direct code transplant: Zig/C architecture and private API. Their M5 default disables it because GPU-only wins. |
| [ANEForge](https://github.com/sbryngelson/ANEForge), [Espresso](https://github.com/christopherkarani/Espresso) | Direct ANE dispatch and LLM experiments via private symbols. Espresso's published fast microbenchmarks use local artifact families, not our production models. | Research references only for a shipping Mac app; private API compatibility is explicitly not guaranteed. |
| [anemll-profile](https://github.com/Anemll/anemll-profile) | Reports `MLComputePlan` op placement, non-ANE islands, and actual prediction throughput for Core ML models. | Can adopt as an external validation tool immediately. |

Project claims above are authors' measurements, not independently reproduced on Rapid. The experiments differ in model, quantization, context, operating system and power settings, so speedup numbers cannot be ranked against each other.

## Engineering judgment

1. **No global `--npu` switch.** ANE needs converted/compiled graphs, compatible shapes and quantization, and may leave individual ops on CPU/GPU. A blanket setting would silently change quality or do nothing for most models.
2. **Most promising LLM seam: opt-in prefill on Qwen 27B, GPU decode retained.** Repeated single-token decode rereads weights and has synchronization / bandwidth costs; the demonstrated gain is on longer prompt prefill, not decode. First check whether prefix-cache hits already make the relevant prompt work disappear.
3. **Public-API route first for small fixed models.** Core ML/ANEMLL can establish that we actually reach ANE and can measure power/latency without private symbols. For general 27B MLX checkpoints, the demonstrated hybrid routes currently rely on private APIs, conversion or duplicate weights and approximate arithmetic. This is an explicit product/compatibility decision, not just an implementation detail.
4. **Do not infer quality from cosine similarity or one-token top-1.** Recurrent GDN makes tiny prefill errors potentially compound over long contexts. Require end-to-end text/tool/JSON/retrieval evaluation.

## Proposed scoped experiment

Use an isolated M3 Ultra run and one pinned Qwen 27B checkpoint. Do not stop unrelated Studio services for this survey; reserve a benchmark window before full-load measurements. Compare GPU-only Rapid with one candidate hybrid engine on the *same* prompt set: 128, 2K, 8K, 16K and 32K uncached prefill; 128/512-token decode; 1/2/4 concurrent chats; then repeat with prefix cache. Record TTFT, prefill tok/s, decode tok/s, peak RSS/wired memory, swap, compile/load time, thermal/power and ANE placement. Alternate order and collect at least 5 steady-state pairs per condition. Run long-context retrieval, coding, JSON schema/tool calls and multi-turn correctness at deterministic settings; report token/reply differences and task success. Only call it a useful optimization if practical TTFT improves without material quality regressions, memory pressure or decode loss. Gate by chip/model/quantization/available RAM, with GPU fallback and an explicit status reason.

This survey does not establish a production-ready ANE path. Next checkpoint: Vector reproduces one hybrid prefill result and one public Core ML small-model result on Studio, with pinned commands/artifacts and a quality comparison; Atlas then decides whether the private-API compatibility cost is acceptable.
