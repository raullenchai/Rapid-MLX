# Qwen3.6 35B-A3B native MTP qualification

Date: 2026-09-13

## Decision

Ship an explicit, greedy-only serial native-MTP backend for the exact
`qwen3.6-35b-4bit` artifact pair. Keep it opt-in. Do not replace the standard
server because the native serial path intentionally exposes fewer capabilities.

## Environment

- Mac Studio, Apple M3 Ultra, 28 CPU cores, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.14
- MLX 0.32.2
- Native runtime 0.6.17
- Target: `mlx-community/Qwen3.6-35B-A3B-4bit@38740b847e4cb78f352aba30aa41c76e08e6eb46`
- Sidecar: `mlx-community/Qwen3.6-35B-A3B-MTP-4bit@0295b81421bf4d0fccca9a7c0fcfb1418dda3516`
- Greedy decoding, thinking disabled, block size 3 (two proposed tokens)
- Existing default Hugging Face cache; no cache override or model copy

## Correctness gate

The paired campaign used coding, arithmetic reasoning, creative writing,
strict JSON, and tool-argument prompts. Each prompt ran six interleaved
target-only/native-MTP pairs at up to 192 output tokens. All 30 pairs produced
byte-identical decoded text. Acceptance varied by workload rather than being
treated as a universal constant:

| Workload | Accepted drafts | Native result |
|---|---:|---:|
| Coding | 88.4% | faster |
| Reasoning | 91.9% | faster |
| Creative writing | 49.4% | faster |
| JSON | 67.9% | faster |
| Tool arguments | 100.0% | faster |

The creative-writing case is the important floor: even with roughly half the
drafts rejected, target verification preserved the exact output and still
improved throughput.

The final HTTP dogfood also exercised user-visible response shaping rather
than only comparing offline token streams. Arithmetic reasoning completed the
requested calculation and exact final-line constraint, creative prose stopped
cleanly, strict JSON parsed with the requested keys and item count, and the
tool request produced the expected structured `weather(city="Tokyo",
unit="celsius")` call.

## Performance

Across all 30 paired offline runs:

| Mode | Median decode | Relative |
|---|---:|---:|
| Target-only native runtime | 83.32 tok/s | 1.00x |
| Native MTP, block 3 | 130.93 tok/s | 1.571x |

Peak active memory increased from about 21.07 GB to 21.24 GB (approximately
0.17 GB). A real non-streaming HTTP coding request with 192 output tokens then
measured 1.52-1.54 seconds after warmup, or about 125 tok/s end to end. The
standard Rapid MTP server completed the same request at 97.4 tok/s, so the
native API path retained roughly 28% higher steady end-to-end throughput.

## Lifecycle and failure checks

- `/healthz` returned the exact target and sidecar revisions.
- A sampled request (`temperature=0.7`) failed closed with HTTP 400.
- A streaming 4,096-token request was disconnected after 350 ms.
- The immediately following request returned HTTP 200 and `RECOVERED` in 65 ms.
- Startup loaded both artifacts through one thread-affine worker; shutdown
  completed without a live model process.

## Reproduction

Install the MTP extra, ensure both artifacts fit in the default cache, and run:

```bash
python scripts/benchmark_qwen36_native_mtp.py \
  --rounds 6 --max-tokens 192

rapid-mlx serve qwen3.6-35b-4bit \
  --speculative-config '{"method":"mtp","backend":"native"}' \
  --host 127.0.0.1 --port 8766
```

The benchmark prints one JSON record per run, including text SHA-256,
generation throughput, token count, and acceptance statistics. It does not
override or relocate the Hugging Face cache.
