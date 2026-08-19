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

Release refresh: Mac mini Mac14,12, Apple M2 Pro (10-core), 32 GB, macOS
26.5.2; Rapid-MLX 0.12.7 at `850f6213`; MLX 0.31.2; mlx-lm 0.31.3. `Peak` is
the process-lifetime `phys_footprint_peak`. Throughput columns show short / 8K.

| Model | Load | Idle | 8K peak | Prefill tok/s | Decode tok/s | New swap | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `lfm2.5-1b-4bit` | 3.1s | 0.78 GB | 1.87 GB | 1,084 | 208 / 124 | 0 MB | Very fast; basic chat only |
| `lfm2.5-2.6b-4bit`² | 19.1s | 1.73 GB | 3.01 GB | 473 | 93.5 / 65.0 | 0 MB | Smarter small-model option; not for coding |
| `lfm2.5-8b-a1b-4bit`² | 39.2s | 4.70 GB | 5.90 GB | 618 | 118 / 82.5 | 0 MB | Fast chat specialist |
| `qwen3.5-4b-4bit` | 6.1s | 2.91 GB | 5.98 GB | 303 | 60.7 / 39.9 | 0 MB | Fast general-purpose |
| `qwen3.5-9b-4bit`² | 43.2s | 5.40 GB | 8.72 GB | 166 | 35.7 / 30.5 | 0 MB | Strong laptop default |
| `gemma-4-12b-4bit`¹² | 56.3s | 7.02 GB | 11.0 GB | 105 | 22.3 / 19.0 | 0 MB | Barely clears prefill gate; 24 GB floor |
| `bonsai-27b-2bit`² | 63.3s | 7.69 GB | 13.0 GB | 164 | 17.5 / 15.0 | 0 MB | Smart 24 GB candidate |
| `gemma-4-26b-4bit`¹ | 11.2s | 14.0 GB | 17.0 GB | 269 | 49.5 / 36.9 | 0 MB | Floor at 32 GB, not 24 GB |
| `qwen3.5-35b-4bit` | 14.1s | 19.0 GB | 22.0 GB | 287 | 58.5 / 21.1 | 4.4 MB | Runs at 32 GB; fails strict zero-new-swap gate |
| `qwen3.6-27b-4bit`² | 119.8s | 15.0 GB | 21.0 GB | 46.7 | 11.0 / 10.1 | 0 MB | Fails 8K prefill gate |
| `qwen3.6-35b-4bit`² | 155.0s | 19.0 GB | — | — | 59.9 / — | **648 MB** | Aborted after short prompt; floor above 32 GB |

¹ Text-only launch flags: Gemma 12B uses `--no-mllm`; Gemma 26B uses
`--no-mllm --kv-cache-dtype bf16 --cache-memory-mb 512`.

² Weights were read directly from the Jetson-backed SMB cache. Load time is
not comparable to local-cache rows; steady-state throughput and memory are.

The earlier Qwen 3.5 35B swap regression (#1634) is fixed on this HEAD; the
different Qwen 3.6 35B alias still crosses the abort threshold on 32 GB
(#1650). The unsafe 24 GB Gemma 26B floor (#1636) remains excluded. Prefix-cache
contamination (#1641) is prevented by the harness. The Gemma 12B base-install
launch omission found by this sweep is tracked in #1648 and covered by a
regression test in this change.

### 2026-08-18 addendum — Qwen3.8-27B (M3 Ultra host)

Measured on the real serve path on Mac15,14 (M3 Ultra, 256 GB), Rapid-MLX
0.12.15 — not the M2 Pro baseline host above; footprint columns are
config-bound, throughput is chip-bound (reads lower on smaller chips). Raw
rows live in `model-recommendation-measurements.json` with the same note.

| Model | Load | Idle | 8K peak | Prefill tok/s | Decode tok/s | New swap | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `qwen3.8-27b-4bit` | 7.1s | 15.0 GB | 20.0 GB | 324 | 40.7 / 37.6 | 0 MB | Smart pick for every tier from 32 GB (fits 24.0 GB budget); misses the 24 GB tier's 18.0 GB budget |
| `qwen3.8-27b-mixed-3.5bpw` | 11.1s | 13.0 GB | 19.0 GB | 324 | 40.8 / 42.3 | 0 MB | Also clears 32 GB+; still misses the 24 GB budget, so it stays a non-default alias |

## Table 2 — two choices per RAM tier

Since 2026-08-18 the Smarter column follows the Artificial Analysis
Intelligence Index: each tier recommends the highest-scoring open-weights
model the engine serves that clears the tier's fit gates (peak < 75 % of the
floor, zero new swap, 8K prefill ≥ 100 tok/s, decode ≥ 10 tok/s). Qwen3.8-27B
scores 52 (GPT-5.6-class) — above every larger model we serve — so every tier
from 32 GB up shares it. Quantization note: the index scores the full-precision
release; our 4-bit build's vendor-published deltas are unmeasured, which is the
standing caveat for every quantized pick in this table. In the SSOT the
`footprint_gb` column stores the measured **8K-prompt peak** (the gate's
number), not the steady post-load footprint; and `capability_pct` is the
same curated 0–100 display scale the existing picks use (ordered by the
AA index; 92 keeps the 27B above the 88 the retired 122B carried), not a
benchmark score.

“Smarter” is the primary pick. “Faster” deliberately trades capability for
latency. Rows above the measured 32 GB host retain the existing reviewed
large-memory picks and must gain host-specific measurements before their next
release change. In particular, the `qwen3.6-35b-4bit` fast cell in the SSOT
(20.0 GB / 60.0 tok/s, marked `"provenance": "estimate"`) is a reviewed
estimate, not a measurement: the only measurement row for that alias (Table 1)
aborted with new swap on the 32 GB host, so those numbers stand until a
measurement on a ≥ 48 GB host (the alias's tier floor) replaces them.

| Physical RAM | Faster | Smarter | Rationale |
|---|---|---|---|
| 8–15 GB | `lfm2.5-1b-4bit` | `lfm2.5-2.6b-4bit` | Smallest safe pair; neither is for serious coding |
| 16–17 GB | `lfm2.5-1b-4bit` | `qwen3.5-4b-4bit` | Instant basic chat vs reliable general use |
| 18–23 GB | `qwen3.5-4b-4bit` | `qwen3.5-9b-4bit` | Both tool-capable and comfortably above 10 tok/s |
| 24–31 GB | `qwen3.5-4b-4bit` | `bonsai-27b-2bit` | 13 GB measured peak; Gemma 26B is too large at 24 GB |
| 32–47 GB | `qwen3.5-4b-4bit` | `qwen3.8-27b-4bit` | 20 GB 8K peak, zero swap; AA Intelligence Index 52 — the highest of any open-weights model we serve |
| 48–63 GB | `qwen3.6-35b-4bit` | `qwen3.8-27b-4bit` | Same smart pick — nothing larger we serve scores higher (122B: 33, 35B: 32) |
| 64–95 GB | `qwen3.6-35b-4bit` | `qwen3.8-27b-4bit` | Same smart pick — parameter count stopped predicting capability at this roster |
| 96 GB+ | `qwen3.6-35b-4bit` | `qwen3.8-27b-4bit` | Same smart pick; the 122B it displaces scores 33 on the same index |

Recommendation data is hardware-specific. A result from one chip/RAM pair must
not be silently copied to another; missing rows are estimates and should be
replaced by measurements before changing a tier.
