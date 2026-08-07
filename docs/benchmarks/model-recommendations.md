# Measured model recommendations

Release recommendations use two slots per RAM tier:

- **Faster:** the fastest model that remains useful for its stated scope.
- **Smarter:** the highest-capability model that clears the interaction and safety gates.

A recommendation must decode at **10 tok/s or faster**, complete the standard
8K prefill at **100 prompt tok/s or faster**, stay below **75% of physical RAM**
at the tier floor, and add **no swap**. A model that misses a gate may remain in
the catalog, but belongs under “Runs, but slow or tight”, not Recommended.

The benchmark is reproducible with:

```bash
python scripts/benchmark_model_recommendations.py \
  --output /tmp/model-recommendations.json
```

Weights may live on a mounted model store when the test Mac is short on local
disk. Pass the concrete Hugging Face Hub cache directory (the directory that
contains `models--…`, not its parent):

```bash
python scripts/benchmark_model_recommendations.py qwen3-1.7b-4bit \
  --hf-cache /Volumes/mac-storage/hf-cache \
  --output /tmp/model-recommendations-jetson-cache.json
```

The cache path is recorded in the raw result. Network storage can affect load
time, so only steady-state prefill/decode and post-load memory are comparable
with local-cache runs.

It starts one cached model at a time through `rapid-mlx serve`, disables the
persisted prefix cache so a rerun cannot reuse the benchmark prompt, uses macOS
`footprint` so Metal unified memory is counted, runs short and ~8K-token
prompts, records `/v1/status` throughput, and stops between models. Raw reviewed
rows live in [`model-recommendation-measurements.json`](model-recommendation-measurements.json).

## Table 1 — chip × RAM × model × engine

First release sweep: Mac mini Mac14,12, Apple M2 Pro (10-core), 32 GB, macOS
26.5.2; Rapid-MLX 0.12.5 at `aba2fdd1`; MLX 0.31.2; mlx-lm 0.31.3. `Peak` is
the process-lifetime `phys_footprint_peak`. Throughput columns show short / 8K.

| Model | Load | Idle | 8K peak | Prefill tok/s | Decode tok/s | New swap | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `lfm2.5-1b-4bit` | 3.1s | 0.78 GB | 1.87 GB | 1,123 | 213 / 127 | 0 MB | Very fast; basic chat only |
| `lfm2.5-2.6b-4bit` | 3.1s | 1.73 GB | 3.03 GB | 488 | 94.5 / 65.4 | 0 MB | Smarter small-model option; not for coding |
| `qwen3-1.7b-4bit`² | 11.1s | 1.31 GB | 5.00 GB | 617 | 133 / 21.2 | 0 MB | Runs well; quality remains untested |
| `lfm2.5-8b-a1b-4bit` | 5.1s | 4.69 GB | 5.89 GB | 634 | 120 / 84.0 | 0 MB | Fast chat specialist |
| `qwen3.5-4b-4bit` | 6.1s | 2.79 GB | 5.86 GB | 314 | 61.8 / 41.6 | 0 MB | Fast general-purpose |
| `qwen3.5-9b-4bit` | 7.1s | 5.40 GB | 8.72 GB | 173 | 36.1 / 31.8 | 0 MB | Strong laptop default |
| `qwen3.5-9b-8bit`² | 75.5s | 9.49 GB | 13.0 GB | 174 | 21.1 / 19.4 | 0 MB | Fits 18 GB narrowly; slow remote load |
| `gemma-4-12b-4bit` | 6.1s | 7.00 GB | 11.0 GB | 52 | 23.5 / 22.2 | 0 MB | Fails 8K prefill gate |
| `deepseek-coder-v2-lite-16b-4bit`² | 67.4s | 8.53 GB | 15.0 GB | 466 | 84.3 / 11.3 | 0 MB | Coding specialist; 32 GB floor |
| `bonsai-27b-2bit` | 8.1s | 7.68 GB | 13.0 GB | 169 | 17.7 / 15.3 | 0 MB | Smart 24 GB candidate |
| `gemma-4-26b-4bit`¹ | 12.1s | 14.0 GB | 17.0 GB | 277 | 50.8 / 39.6 | 0 MB | Floor at 32 GB, not 24 GB |
| `qwen3.5-35b-4bit` | 14.1s | 19.0 GB | — | — | 58.5 / — | **1,120 MB** | Aborted after short prompt; floor above 32 GB |
| `qwen3.6-27b-4bit` | 12.1s | 15.0 GB | 20.0 GB | 48.9 | 11.4 / 10.6 | 0 MB | Fails 8K prefill gate |

¹ Text-only launch flags: `--no-mllm --kv-cache-dtype bf16 --cache-memory-mb 512`.

² Weights were read directly from the Jetson-backed SMB cache. Load time is
not comparable to local-cache rows; steady-state throughput and memory are.

The 32 GB Qwen 35B swap regression is tracked in #1634. The unsafe 24 GB
Gemma 26B floor is tracked in #1636. Prefix-cache contamination of rerun
measurements is tracked in #1641 and prevented by the harness now.

## Table 2 — two choices per RAM tier

“Smarter” is the primary pick. “Faster” deliberately trades capability for
latency. Rows above the measured 32 GB host retain the existing reviewed
large-memory picks and must gain host-specific measurements before their next
release change.

| Physical RAM | Faster | Smarter | Rationale |
|---|---|---|---|
| 8–15 GB | `lfm2.5-1b-4bit` | `lfm2.5-2.6b-4bit` | Smallest safe pair; neither is for serious coding |
| 16–17 GB | `lfm2.5-1b-4bit` | `qwen3.5-4b-4bit` | Instant basic chat vs reliable general use |
| 18–23 GB | `qwen3.5-4b-4bit` | `qwen3.5-9b-4bit` | Both tool-capable and comfortably above 10 tok/s |
| 24–31 GB | `qwen3.5-4b-4bit` | `bonsai-27b-2bit` | 13 GB measured peak; Gemma 26B is too large at 24 GB |
| 32–47 GB | `qwen3.5-4b-4bit` | `gemma-4-26b-4bit` | 20 GB 8K peak with no new swap on the 32 GB floor |
| 48–63 GB | `qwen3.6-35b-4bit` | `gemma-4-26b-4bit` | Retains the existing reviewed fast pick pending a 48 GB host measurement |
| 64–95 GB | `qwen3.6-35b-4bit` | `qwen3.6-35b-8bit` | Same family: speed vs quantization fidelity |
| 96 GB+ | `qwen3.6-35b-4bit` | `qwen3.5-122b-mxfp4` | Workhorse speed vs maximum local capability |

Recommendation data is hardware-specific. A result from one chip/RAM pair must
not be silently copied to another; missing rows are estimates and should be
replaced by measurements before changing a tier.
