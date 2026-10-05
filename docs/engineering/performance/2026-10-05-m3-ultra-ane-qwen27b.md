# M3 Ultra ANE/GPU prefill experiment (2026-10-05)

Owner: Vector. Status: external-engine experiment, **no Rapid integration**.

## Question and setup

Can ANE run alongside the GPU to improve local Qwen 27B inference? We measured the same oMLX 0.7.0 runtime, identical cached checkpoint and prompts, toggling only its opt-in ANE prefill. This isolates the candidate feature within one engine; it does **not** measure Rapid's own server against oMLX.

- Host: Mac Studio M3 Ultra, 256 GiB unified memory, macOS 26.5.2; shared host with other workloads. Swap already held 20.5 GiB before testing and stayed at 20.5 GiB. We did not stop other services or reboot.
- Candidate source: `jundot/omlx` commit `a435a3736071286fdb2b74f486b3505f23f3c194`, built with `OMLX_WITH_CUSTOM_KERNEL=1`; `mlx==0.32.2`, `mlx-lm==0.31.4.dev132+g94cdcae13`.
- Pinned local checkpoint: `mlx-community/Qwen3.8-27B-4bit`, snapshot `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, `qwen3_5`, affine 4-bit/group 64. No model download.
- oMLX settings: no cache, one concurrent request, temperature 0, thinking off, 24 maximum output tokens. ANE arm: 2048-token tiles, 53% eligible MLP projection, two ANE instances, GDN z-only (runtime capped requested 50% to 37.5%). GPU arm: ANE toggle off. The server used loopback only.
- Timing: streaming `/v1/chat/completions` `usage.time_to_first_token`, excluding model load. Each request had a different early nonce so no common prefix could be reused. Prompt repeated the same innocuous sentence, then asked for `ZEBRA-4417`. Exact prompt generator and raw results: `scripts/experiments/ane_prefill_paired_probe.py`, `data/2026-10-05-ane-m3-qwen27b.jsonl` in this repository.

## Measurements

| Input tokens | GPU TTFT, warm (s) | ANE/GPU TTFT, warm (s) | Observed effect |
| ---: | ---: | ---: | --- |
| 165 | 0.66 | 0.66, 0.67 | No benefit for short prompt. |
| 2,597 | 7.42, 7.42 | 6.56, 6.00 | Mean TTFT about 15% lower. |
| 9,797 | 28.60, 28.59; GPU return 28.63 | 22.90, 23.17 | Mean TTFT about 19% lower; about 1.24x prompt throughput. |

All 27B probe replies were `ZEBRA-4417`. A separate ~2,600-token arithmetic prompt (`37*48`) and a Python deduplication task returned **byte-identical complete replies** in GPU and ANE arms; this is only a two-task quality smoke test (source: `scripts/experiments/ane_prefill_quality_probe.py`; raw replies: `data/2026-10-05-ane-m3-qwen27b-quality.jsonl`). Warm 27B loads reported 15.73 GB GPU-only versus 28.56 GB with ANE, an extra 12.83 GB. In the built-in 4097-token diagnostic, oMLX logged 126 MLP and 96 GDN native ANE operations, with nonzero evaluation time on both ANE instances. The diagnostic was run with `align_prompt_to_ane=true`, which skips omlx.ai upload.

The initial 9B probe also executed native ANE work (62 MLP and 48 GDN operations at 4097 tokens), but timings on the shared host were noisy: a 9797-token ANE request ranged from 6.8 to 11.7 seconds, versus GPU 8.1 to 10.1 seconds. It does not establish a stable 9B gain.

## Reproduction

Clone the pinned oMLX commit into scratch, create a Python 3.13 virtual environment there, and install with `OMLX_WITH_CUSTOM_KERNEL=1 uv pip install --python <venv>/bin/python -e <omlx-checkout>`. Symlink the existing HF snapshot under a scratch model directory (do not create another model cache). Start oMLX with `serve --model-dir <scratch-model-dir> --base-path <scratch-state-dir> --host 127.0.0.1 --port 18940 --no-hf-cache --no-cache --max-concurrent-requests 1`. For the ANE arm, put a `model_settings.json` in that isolated base path with `version: 1` and the model's `qwen35_ane_prefill_enabled: true`, `qwen35_ane_prefill_sequence_length: 2048`, `qwen35_ane_prefill_fraction: 0.53`, `qwen35_ane_prefill_dual_ane: true`, `qwen35_ane_prefill_gdn: true`, `qwen35_ane_prefill_gdn_fraction: 0.5`; restart the isolated server to apply. Use the probe script with `--model <local-model-id> --rounds 2`, once per arm, and exclude the first cold-load request. Check `/api/status` and `[benchmark-ane-profile]` for real ANE dispatch. To reproduce the diagnostic, use oMLX's local throughput benchmark with `align_prompt_to_ane=true` and 4096 requested prompt tokens.

## Decision boundary

This is a promising **M3 Ultra long-prefill** result; it does not imply faster decode or benefits on every chip. oMLX uses undocumented AppleNeuralEngine APIs and requantizes selected weights to approximate INT8, so Rapid cannot ship it as an unconditional universal NPU path. The ~13 GB duplicate working set is especially material on 32–64 GB Macs. A product candidate needs the Rapid serving path, real agent/JSON/retrieval quality corpus, prefix-cache and concurrency comparisons, repeat runs on quiet hardware, failure/fallback behavior, memory gating and macOS-update compatibility testing.

M4, M5 and M6 were **not tested here**. The current oMLX documentation says M5 NAX-aware kernels resolved an earlier ANE/GPU split regression, while mlx-serve defaults ANE off on M5; these are engine-specific reports, so each chip needs its own A/B. Apple's M6 has both GPU Neural Accelerators and a dual Neural Engine, but hardware peak figures do not predict this workload's gain. See [oMLX ANE implementation notes](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md), [mlx-serve releases](https://github.com/ddalcu/mlx-serve/releases), [Apple's M6 announcement](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/).
