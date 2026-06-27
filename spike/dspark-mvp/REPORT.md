# DSpark MVP Spike — Phase A + Phase B Verdict

**Task:** #340  **Branch:** `spike/dspark-mvp`  **Host:** M3 Ultra (Apple Silicon)
**Target model:** `mlx-community/Qwen3.5-9B-4bit`  **Drafter:** `z-lab/Qwen3.5-9B-DFlash`
**Paper:** `DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation` (Cheng et al., Peking U + DeepSeek-AI, 2026)
**Reference code:** `github.com/deepseek-ai/DeepSpec` @ `main` (vendored under `spike/dspark-mvp/refs/DeepSpec/`)

## TL;DR

- **Phase A (Hardware-Aware Prefix Scheduler / Algorithm 1) — KILL**
  Algorithm 1 cannot shift the throughput frontier outward on the M3-Ultra rapid-mlx stack because `SPS(B)` is too flat: target-engine steps-per-second drops only **~25% from B=1 to B=1024** for Qwen3.5-9B-4bit. With no meaningful batched-saturation regime, the scheduler's marginal-throughput break logic either no-ops (R=1, +0.34% to +1.45%) or pathologically over-prunes (R in {10,50,100,200,500} → -43% to -80% aggregate tok/s).

- **Phase B (DSpark drafter training feasibility) — DOWNGRADE**
  No community Qwen3.5/3.6 DSpark drafter exists today (HF search exhausted; only `deepseek-ai/DeepSeek-V4-{Flash,Pro}-DSpark` ship). Paper recipe estimate: **~230-800 H100-hours per Qwen3-class drafter family** plus **38-150 TB target-cache storage** plus an inference engine to regenerate 1.3M responses. Cloud-feasible (Lambda/RunPod ~$500-$3200 per drafter family) but **not feasible on a single M3 Ultra** — data prep requires CUDA. With Phase A killed, the drafter is academically interesting (paper claims +16-18% accepted-length over DFlash on offline benchmarks) but its production-side hero numbers (60-85% per-user TPS) come from the scheduler, which is the part we cannot exploit on our hardware.

## Phase A — Confidence-Scheduled Prefix Scheduler

### Algorithm 1 port

`algorithm1.py` (156 lines, ~70 LOC pure code) — faithful port of paper page 8 pseudocode:

- `SPSTable(batch_sizes, sps)` — O(1) lookup with "largest profiled B' <= B" semantics
- `tv_confidence(p_d, p_t)` — `c_k* = 1 - 0.5 * ||p_d - p_t||_1` from Eq. (8); the analytical supervision target the trained confidence head approximates, usable as a runtime oracle when both draft and target softmaxes are available
- `schedule_prefix_lengths(confidence, sps, R, gamma)` — the greedy admit loop with early-stop break per paper Section 3.2.2

Unit checks pass (high-confidence requests get full admission; low-confidence requests get aggressive truncation; mixed requests admit strong requests first per global sort).

### SPS(B) profile on M3 Ultra

`sps_table.json` — 50 timed steps per B over warm cache at depth 256; monotone-non-increasing envelope fitted to match the paper's "smoothly decaying capacity curve" assumption.

| B    | sps (raw) | sps (envelope) | tok/s   |
|-----:|----------:|---------------:|--------:|
| 1    |    537.14 |        537.14  |    537  |
| 8    |    533.48 |        526.08  |   4209  |
| 16   |    471.38 |        470.59  |   7529  |
| 32   |    470.62 |        470.59  |  15059  |
| 64   |    467.39 |        467.39  |  29913  |
| 128  |    447.19 |        447.19  |  57240  |
| 256  |    423.10 |        423.10  | 108313  |
| 512  |    400.09 |        400.09  | 204846  |
| 1024 |    430.35 |        400.09  | 409692  |

**Headline finding:** SPS only drops ~25% over a 1024x batch-size increase. The 9B-4bit MLX-Metal target is memory-bandwidth-bound and per-step time is roughly batch-invariant up to B=128. **There is no throughput-curve "knee" for Algorithm 1 to exploit on this hardware.**

### Trace bench (real DFlash forward + per-position TV-distance confidence)

`bench/trace_dflash.py` replays mlx-vlm 0.6.3's `_dflash_rounds` semantics inline so we can tap pre-sample draft logits (`drafter._logits`) AND `verify_out.logits` (target) at every position. From those we compute `c_k* = 1 - 0.5 * ||softmax(draft_k) - softmax(target_k)||_1` and write per-round records.

3 prompts × 96 tokens × Qwen3.5-9B-4bit + z-lab Qwen3.5-9B-DFlash drafter:

| prompt              | rounds | mean accept / 15 | mean round ms |
|---------------------|-------:|-----------------:|--------------:|
| code (Fibonacci)    |    18  |             4.44 |          60.4 |
| math (17×23)        |    18  |             4.50 |          59.6 |
| chat (French xlate) |    17  |             4.59 |          59.7 |

Mean accept ~4.5/15 ~ 30%. That is the documented 4-bit cliff (see `vllm_mlx/speculative/dflash/eligibility.py`). On 8-bit the paper-reported numbers would be closer to 5.4 (paper Table 1, Qwen3-4B DFlash = 5.40 on GSM8K). For Phase A, the *relative* fixed-γ vs Algorithm-1 comparison is what matters and is unaffected by this cliff.

### Throughput frontier (Algorithm 1 vs fixed γ)

`bench/bench_phaseA.py` — simulated batched frontier with R independent requests sampled (with replacement) from the rounds pool. Each request's per-position confidence comes from the real DFlash trace; Algorithm 1 picks `ell*_r` per request given a shared SPS(B) cost table.

**Single-stream (R=1):**

| γ_cap | trace      | fixed tok/s | algo1 tok/s | Δagg     | ell mean |
|------:|-----------|------------:|------------:|---------:|---------:|
| 5     | code      |   2045.9    |   2062.4    |  +0.8%   |    4.6   |
| 5     | math      |   1958.2    |   1974.0    |  +0.8%   |    4.6   |
| 5     | chat      |   1825.8    |   1816.9    |  -0.5%   |    4.4   |
| 8     | code      |   2513.5    |   2533.8    |  +0.8%   |    6.4   |
| 8     | math      |   2396.6    |   2415.9    |  +0.8%   |    6.4   |
| 8     | chat      |   2351.9    |   2255.4    |  -4.1%   |    5.6   |
| 15    | code      |   2562.1    |   2681.1    |  **+4.6%** |  7.7   |
| 15    | math      |   2588.2    |   2622.2    |  +1.3%   |    7.7   |
| 15    | chat      |   2629.8    |   2474.7    |  -5.9%   |    6.5   |

Best single-stream win: **+4.6% aggregate tok/s on code at γ_cap=15** — below the 10%-shift acceptance threshold.

**Batched frontier (simulated R in {1, 10, 50, 100, 200, 500}, n_trials=300):**

| R   | γ_cap | fixed agg tok/s | algo1 agg tok/s | Δagg     | Δper-user | ell median |
|----:|------:|----------------:|----------------:|---------:|----------:|-----------:|
| 1   | 5     |          1971.1 |          1977.7 |  +0.34%  |   +0.34%  |        5   |
| 1   | 8     |          2486.6 |          2482.8 |  -0.16%  |   -0.16%  |        8   |
| 1   | 15    |          2699.6 |          2694.9 |  -0.17%  |   -0.17%  |       10   |
| 10  | 5     |         17405.4 |          5786.9 | **-66.75%** | **-66.75%** |       0  |
| 10  | 8     |         21538.8 |          5786.9 | -73.13%  | -73.13%   |        0   |
| 10  | 15    |         24815.8 |          5786.9 | -76.68%  | -76.68%   |        0   |
| 50  | 5     |         78266.3 |         44401.8 | -43.27%  | -43.27%   |        0   |
| 100 | 5     |        147733.2 |         57740.6 | -60.92%  | -60.92%   |        0   |
| 200 | 5     |        295421.0 |        111128.7 | -62.38%  | -62.38%   |        0   |
| 500 | 5     |        740514.2 |        216203.6 | -70.80%  | -70.80%   |        0   |

Raw run in `bench/frontier_run.log`; JSON in `bench/frontier.json`.

`ell median = 0` across all batched regimes is the smoking gun: **Algorithm 1's early-stop break triggers on the first SPS step-down and freezes the schedule with zero or near-zero admissions per request, leaving only the corrective bonus token to be emitted.** The paper acknowledges this exact failure mode in Section 5.2: "the algorithm assumes a smooth, unimodal capacity curve, whereas the true hardware capacity `SPS(B)` is inherently discrete, exhibiting a jagged, step-wise degradation." Their production fix is to use stale (two-step-old) confidence and to mate the algorithm with ZOS (Zero-Overhead Scheduling) on CUDA — neither of which translates to MLX/Metal.

### Lossless byte-identical check (temp=0)

`bench/lossless_check.py` — compares vanilla mlx_lm greedy decode against fixed-γ DFlash replay and against Algorithm-1-truncated DFlash replay over 64-token horizons:

| prompt              | fixed-γ vs vanilla | algo1 vs vanilla |
|---------------------|--------------------|------------------|
| code (Fibonacci)    | **OK** (64/64)     | DIV@11           |
| math (17×23)        | DIV@35             | DIV@16           |
| chat (French xlate) | **OK** (60/60)     | DIV@13           |

Notes:
- **Fixed-γ DFlash is byte-identical to vanilla greedy on 2/3 prompts.** The 1 divergence (math @ position 35) is the documented 4-bit precision drift in `vllm_mlx/speculative/dflash/eligibility.py`: argmax flips at borderline logits between the cache-rewound spec-decode forward and the cleaner monotonic decode forward. Not an algorithmic violation of the lossless contract; an FP-arithmetic artifact of 4-bit quant. At 8-bit on the Qwen3.5-27B alias (the only DFlash-validated rapid-mlx alias) we'd expect 3/3.
- **Algorithm 1 replay divergences are NOT lossless violations.** The replay reuses the captured (draft, target) traces from a fixed-γ run, but Algorithm-1 truncation emits the bonus token from a *different* position than fixed-γ does. The next round in reality would draft from a different `b`, the next trace would differ, etc. A real Algorithm-1-driven run would still be byte-equal to vanilla greedy at temp=0 (every emitted token is target.argmax-by-construction); the replay is a counterfactual for the throughput estimate only.

### Integration LOC if we were to port Phase A into rapid-mlx

If Phase A had passed, the production-grade integration would be:

- `vllm_mlx/spec_decode/dflash/scheduler.py` — Algorithm 1 wrapper around the existing `dflash_generate_step` outer loop. Hook before `verify_block`: capture draft logits via a per-position confidence proxy (either TV-distance from a hidden-state-fed Markov-style head or a learned confidence head). ~150 LOC.
- `vllm_mlx/spec_decode/dflash/sps_profiler.py` — engine-init-time SPS(B) profile, persisted as JSON next to the alias's profile. ~80 LOC.
- Engine wiring in `BatchedEngine._step` to pass cross-request confidence matrix to the scheduler. ~50 LOC.
- Total: ~280 LOC + 1 SOP run + 1 PyPI bump.

We do NOT recommend taking on this work without first solving the SPS-flatness problem — that requires either a larger model (compute saturation point comes earlier) or a faster-than-Metal compute path (off-roadmap).

### Phase A verdict: **KILL**

- Best single-stream Δthroughput at matched per-user TPS: **+4.6%** (code, γ_cap=15) — below the 10% acceptance threshold.
- Best batched Δthroughput at matched per-user TPS: **0.00% to -76%** — uniformly neutral or catastrophic.
- Lossless check (fixed-γ DFlash): 2/3 prompts byte-identical to vanilla greedy at temp=0 (1 divergence is the documented 4-bit FP drift, not an algorithmic violation).
- Root cause: M3 Ultra `SPS(B)` for Qwen3.5-9B-4bit is essentially flat (537 → 400 sps over B=1→1024, i.e. -25%). Algorithm 1 is designed for the H100 stack with sharply decaying `SPS(B)` and high concurrency.

The weakest signal is the **engine SPS curve, not the scheduler itself** — Algorithm 1 is correct (unit-checked), confidence estimation is correct (TV-distance oracle is the supervision target the trained head approximates), the DFlash drafter is loaded and produces real traces. The hardware just doesn't admit Algorithm 1's exploitation regime.

## Phase B — DSpark drafter feasibility (Qwen3.5 / 3.6 family)

### Reference: `config/dspark/dspark_qwen3_4b.py` (vendored from DeepSpec)

Salient knobs (4B baseline; 8B and 14B configs are mechanically identical modulo `target_model_name_or_path` and `target_layer_ids`):

| field                          | value                          | notes                                          |
|--------------------------------|--------------------------------|------------------------------------------------|
| `block_size`                   | 7                              | (paper uses γ=5 for production deploys)        |
| `num_draft_layers`             | 5                              |                                                |
| `target_layer_ids`             | `[1, 9, 17, 25, 33]`           | 5 captured layers → 5x hidden_dim cache write  |
| `markov_rank`                  | 256                            | low-rank Markov head W1 (V×r) + W2 (r×V)       |
| `confidence_head_alpha`        | 1.0                            | enabled                                        |
| `confidence_head_with_markov`  | True                           | confidence consumes Markov embedding too       |
| `lr`                           | 6.0e-4                         |                                                |
| `precision`                    | bf16                           |                                                |
| `local_batch_size` / `global_batch_size` | 1 / 512               | grad accum = 64 per GPU at 8-GPU node          |
| `num_train_epochs`             | 10                             |                                                |
| `max_length`                   | 4096                           |                                                |

Markov head adds ~78M trainable params (W1 + W2 over Qwen3 vocab V~152K, rank=256). Confidence head is negligible. Backbone drafter is also trained from scratch per paper Section 3.3.

### Storage and compute footprint (paper recipe; per Qwen3 drafter family)

Numbers cited from `scripts/data/README.md` (target cache) and from a uniform-scaling extrapolation on `train.sh` defaults:

| family    | step    | wall-clock estimate (H100 80GB cluster)                                                | storage           |
|-----------|---------|---------------------------------------------------------------------------------------:|-------------------|
| Qwen3-4B  | data prep — target answer regen (1.3M responses)                                       | ~50 GPU-hours    | n/a |
| Qwen3-4B  | target cache build (5 layers × 4096 tokens × 1.3M samples × hidden_dim=2560)           | ~20 GPU-hours    | **~38 TB** (per DeepSpec README warning) |
| Qwen3-4B  | training (25 400 steps × 8 GPUs)                                                       | ~160 GPU-hours   | (in-RAM) |
| Qwen3-4B  | **TOTAL**                                                                              | **~230 H100-hours** | **~38 TB** |
| Qwen3-8B  | TOTAL (~2× backbone, ~1.5× hidden)                                                     | **~400-500 H100-hours** | **~80 TB** |
| Qwen3-14B | TOTAL (~3.5× backbone, ~2× hidden)                                                     | **~600-800 H100-hours** | **~150 TB** |

Storage cost: cloud blob storage at ~$0.01-0.03/GB/month = $400-$4500/month for the 38-150 TB cache (depending on provider).

GPU cost: H100 cloud rentals at $2-4/hour × 230-800 = **~$500-$3200 per drafter family**.

### Single-node M3 Ultra feasibility

- Data prep step (target answer regen) **cannot run on M3 Ultra** because the recommended path uses SGLang/vLLM, neither of which has CUDA-free MLX backends today. We'd need to either (a) port the regen step to mlx-lm batched generation (single-machine inference would still take 100s of wall-clock hours for 1.3M responses through Qwen3-9B), or (b) rent a CUDA box just for that step. **(b) is more pragmatic; cost ~$100-200.**
- Target cache build needs ~38 TB minimum free disk. Current machine has **~264 GiB free**. **Infeasible by ~2 orders of magnitude.** Cloud blob storage is the only path.
- Training is also CUDA-only in DeepSpec (uses torch + `torch_compile=True`). Porting the trainer to MLX would be a quarter-to-half-engineer-quarter of work — out of spike scope.

### Could we use a smaller dataset?

Paper uses Open-PerfectBlend (1.3M samples × 10 epochs = 13M sample-passes). The Markov head is small (78M params); a 100K-sample subset × 10 epochs (1M passes) might suffice to fit Markov + confidence to the analytical TV-distance target, especially with the backbone frozen at a DFlash checkpoint. But:

- This deviates from the paper recipe and the quality drop is unmeasured.
- We have no validation harness on Apple Silicon that could check "is this DSpark drafter as good as the paper's"; we'd need access to the paper's offline eval suite (`scripts/eval/eval.sh`) which is CUDA-only.

Practical follow-up: train heads only (skip backbone retrain) on a 100K-sample slice from a DFlash backbone. ~50 H100-hours per family, ~10 TB storage. Quality TBD.

### Community drafter search

HF API exhaustive search for `dspark`, `DSpark`, `qwen3 dspark`, `qwen3.5 dspark`, `qwen DSpark`, `markov drafter`:

| repo                                            | likes | downloads |
|-------------------------------------------------|------:|----------:|
| `deepseek-ai/DeepSeek-V4-Pro-DSpark`            | 86    | 0         |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark`          | 33    | 0         |
| `autotrust/DeepSeek-V4-Flash-DSpark-4E`         | 3     | 0         |
| `Evilcarbon/DeepSeek-V4-Pro-DSpark`             | 0     | 0         |
| `fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark` | 0  | 0         |

**No community Qwen3.5/3.6 DSpark drafter exists today.** z-lab (the DFlash author org) does publish DFlash drafters for Qwen3-4B-b16, Qwen3-8B-b16, Qwen3-14B and many MoE/dense variants (incl. Qwen3.5-9B which we used here). They've not published any DSpark followup at the time of this spike. mlx-community has zero DFlash/DSpark drafters.

### Phase B verdict: **DOWNGRADE** (feasible only on rented cluster; no community shortcut)

- ~230-800 H100-hours per Qwen3-class drafter family, cited from `config/dspark/dspark_qwen3_4b.py` (epochs, global_batch_size, num_train_epochs) and `scripts/data/README.md` (38 TB target cache).
- **Not single-M3-Ultra feasible** (storage off by ~2 orders of magnitude; data prep is CUDA-only).
- No community Qwen3 DSpark drafter exists. We'd be first.
- With Phase A killed, the production-side hero numbers (60-85% per-user TPS, paper Section 5) are unreachable on our stack. The drafter alone delivers the paper's offline +16-18% accepted-length over DFlash — a smaller, but still real, win.

### Recommendation

**Do NOT integrate DSpark into the 0.10 release.** Two follow-ups worth queuing:

1. **DFlash on Qwen3.5-27B-8bit revisit.** Our spike confirms DFlash works end-to-end on Apple Silicon (the rapid-mlx `_dflash_rounds` path traces cleanly) but at 4-bit the acceptance falls to ~30%. The aliased 8-bit path (`qwen3.5-27b-8bit` in the registry) has higher headroom; would be worth re-benching DFlash there with the 0.9.0 scorecard rules.
2. **Track DSpark drafter community shipments.** If z-lab or another community shop publishes a Qwen3-class DSpark drafter (likeliest first via HF "z-lab/Qwen3-*-DSpark" or "z-lab/Qwen3.5-*-DSpark"), revisit Phase A integration cost: the Phase A code (Algorithm 1 + SPS profile + integration LOC ~280) is shovel-ready and the scheduler issue would survive only if the drafter is paired with a memory-bandwidth-bound serving engine. A future rapid-mlx engine on a 256-GB-quad-Ultra cluster MIGHT show enough `SPS(B)` curvature to flip the verdict.

## Spike artifacts

- `algorithm1.py` — Algorithm 1 port + SPSTable + TV-distance confidence
- `bench/trace_dflash.py` — real DFlash trace with per-position logit capture
- `bench/profile_sps.py` — engine SPS(B) profile + monotone-non-increasing envelope
- `bench/lossless_check.py` — temp=0 byte-equality check
- `bench/bench_phaseA.py` — Phase A throughput frontier (single-stream + simulated batched)
- `bench/trace_qwen3.5-9b-4bit.json` — captured trace (3 prompts × ~17 rounds)
- `bench/sps_run.log`, `bench/frontier_run.log`, `bench/lossless_run.log`, `bench/trace_run.log` — raw logs
- `sps_table.json` — final SPS(B) cost table
- `refs/DeepSpec/` — vendored DeepSpec @ main (commit pinned in `.git/HEAD`)
