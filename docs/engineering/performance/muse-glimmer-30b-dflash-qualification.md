# Muse-Glimmer 30B 8-bit DFlash qualification

## Decision

Enable curated DFlash for `muse-glimmer-30b-8bit`. The immutable target and
assistant pair cleared the 1.30x code-median ship gate and the 1.00x non-code
floor. Keep `muse-glimmer-30b-4bit` disabled because the legacy DFlash runtime
does not qualify 4-bit targets.

## Immutable model identity

- Target: `mlx-community/Muse-Glimmer-30B-8bit` at
  `679c45e1b331a6514a6e38076a353cc5fed21bf6`
- Assistant: `meta-models/Muse-Glimmer-30B-assistant` at
  `e8192f3a8f617f74be2ce220360c89ef4789f39f`
- Runtime algorithm: `dflash`

The target download footprint was 33,404,526,711 bytes (31.1 GiB). After a
generation, cancellation, and recovery request, the server process reported
36.9 GiB RSS. The curated alias therefore requires 48 GB unified memory,
leaving roughly 11 GB for the OS and application overhead on the measured
workload.

## Environment

- Mac Studio, Apple M3 Ultra (28 CPU cores), 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.14
- MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17
- Rapid-MLX commit under test: `origin/main@9113b5d8d` plus the alias change
- Date: 2026-09-07

## Reproduction

Run the repository's qualification harness; it resolves and downloads both
exact revisions before starting either server:

```bash
python3.12 scripts/bench_dflash.py \
  --model muse-glimmer-30b-8bit \
  --expected-algorithm dflash \
  --max-tokens 256 \
  --runs 3 \
  --port 18845 \
  --output /tmp/muse-dflash-bench.json
```

The harness runs the same greedy prompts against the normal text server and
the dedicated single-user DFlash server. Throughput is median decode tokens
per second across three runs per workload.

| Workload | Plain (tok/s) | DFlash (tok/s) | Speedup |
| --- | ---: | ---: | ---: |
| Fibonacci | 22.6 | 43.9 | 1.94x |
| Quicksort | 22.1 | 46.3 | 2.10x |
| Hashtable | 22.6 | 42.8 | 1.89x |
| Sorted list | 22.6 | 46.3 | 2.05x |
| General chat | 22.6 | 41.3 | 1.83x |

- Code-workload median: **2.00x** (ship gate: at least 1.30x)
- General-chat result: **1.83x** (non-code floor: at least 1.00x)
- All-workload median: **1.94x**
- Decision: `SHIP (supports_dflash=true)`

## Server-path checks

The exact cached artifacts booted through the public alias in 5.2 seconds on
an initial warm start. `/healthz` reported the expected algorithm and both
revision pins. Non-streaming generation completed successfully. A streaming
client was then disconnected after 1.5 seconds and 4,537 response bytes; an
immediate follow-up request completed in about 1.0 second, confirming that
cancellation released the serial generation lease and the server recovered.
