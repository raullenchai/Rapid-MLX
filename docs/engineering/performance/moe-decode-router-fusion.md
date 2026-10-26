# LFM2 MoE single-token router and combine fusion

The loaded model's `lfm2_moe` blocks receive a local subclass. No upstream
class is modified globally and no weight buffers are duplicated. Existing
gate/up fusion stays enabled. The text engine enrolls blocks after loading.

Fusion requires `[B,L,H]` with `B*L == 1`, Metal as the default device,
32–256 experts in multiples of 32, top-k 1–8, and H divisible by 128.
Supported gates are bias-free standard Linear with f32/bf16 weights.
Other gate layouts, sharded blocks, larger batches and prefill use the prior
block implementation. The scheduler supplies `[1,1,H]` for one active request;
admission and completion can also leave qualifying single-request steps.

Set `RAPID_MLX_MOE_DECODE_FUSION=0` before starting `rapid-mlx serve` to disable
installation. The switch also bypasses already-enrolled blocks.

Numerics preserve the dependency's GEMV lane split, multiply-add and reduction
order, four-value softmax loads, fast exponential with f32 accumulation,
reciprocal multiplication, post-softmax bias and epsilon, and dtype-rounded
products and accumulation. Keeping only f32 accumulation until the final bf16
cast was insufficient for model-level agreement.

Deterministic ties preserve the runtime's stable ascending sort order: the
largest tied indices occupy the top-k tail. Regression coverage compares exact
indices, scores and full bf16 block outputs for 32–256 experts, top-k 1–8,
uniform gates and biased boundary ties (128 cases).

## Measurement method

Checkpoint: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, revision
`667408d494bde191c2cb9dd4f4549827816b9275`.
Runtime: Python 3.12, MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2.

A is an export of origin/main; B is the candidate source tree. One venv serves
both. Order ABAB, separate `rapid-mlx serve` launches, 16-token warmup, three
measured 128-token requests per context per launch, greedy sampling, prefix/disk
caches and speculation disabled. Results are medians of six samples per
variant/context. Decode rate is
`(completion_tokens-1)/(last_text_chunk-first_text_chunk)` from streaming
`POST /v1/completions`; TTFT includes tokenization, prefill and serving overhead.
This measurement assumes incremental text chunks as emitted by this workload.

## Three-generation results

Final LFM-only implementation, rerun on all three generations; M5 numbers
were rerun after removing unused kernel specializations. Baseline:
`0165b054ab0267ce3b6a1ad7d2552ba8243f9d1e` (origin/main when runs began).
The subsequent rebase onto `7dfe272bd13a8adde8c5944ea6d8a7f130b7f564`
changes chat/tool postprocessing; the measured engine, model utilities and
`/v1/completions` route are unchanged. That route streams directly from the
engine without the changed chat postprocessor.

| Machine | Context | Main → fused decode tok/s | Change | Main → fused TTFT ms | Main → fused peak GB |
|---|---:|---:|---:|---:|---:|
| M5 Max, studio-max, 36 GB | 256 | 174.21 → 199.73 | +14.65% | 85.00 → 84.81 | 13.99 → 13.99 |
| M5 Max, studio-max, 36 GB | 4096 | 166.91 → 190.36 | +14.05% | 566.86 → 566.00 | 14.74 → 14.74 |
| M4 Pro, mzr-2, 48 GB | 256 | 116.47 → 129.82 | +11.46% | 306.11 → 305.40 | 14.00 → 14.00 |
| M4 Pro, mzr-2, 48 GB | 4096 | 110.77 → 122.36 | +10.46% | 3779.99 → 3778.66 | 14.75 → 14.75 |
| M3 Ultra, studio-ultra, 256 GB | 256 | 160.74 → 188.62 | +17.34% | 128.81 → 127.71 | 14.00 → 14.00 |
| M3 Ultra, studio-ultra, 256 GB | 4096 | 155.86 → 181.71 | +16.59% | 1217.84 → 1213.94 | 14.75 → 14.75 |

All single-request 128-token greedy outputs matched exactly at both contexts
on all three machines. Peak Metal memory was unchanged at 0.01 GB resolution.
M4 and M3 gains exceed the requested 8% at both contexts.

Final M5 concurrency measurements use ABAB at 256 context; aggregate throughput
includes prefill and completion. Every response completed with 128 tokens and
output sets matched, including two scheduling-dependent variants at concurrency
four.

| Concurrent requests | Main → fused aggregate tok/s | Change | Main → fused peak GB |
|---|---:|---:|---:|
| 2 | 205.38 → 205.70 | +0.16% | 14.01 → 14.01 |
| 4 | 276.07 → 277.49 | +0.52% | 14.04 → 14.04 |

Teacher forcing on M3 and M5 uses 256 prefill tokens from the baseline README
followed by 1,024 forced tokens. Three caches share one set of weights;
main and fused prefill in chunks of 128, the main noise control in chunks of
64. M3 and M5 each enrolled 38 blocks and produced 1,024/1,024 top-1 matches, zero mean KL
and byte-identical logits. Changed-chunking control KL was 0.0654713 on M3 and 0.0633344 on M5.

M4 preflight showed about 76 GiB free and no heavy inference process; after
weights and environment setup 61 GiB remained. The user authorized a task-only
50 GB reserve exception. M5 had about 294 GiB free and no heavy inference
process. M3 had 219 GiB free; existing idle Desktop/operator/other-agent servers
remained untouched. M3 ran under its own FIFO lease plus
`lockf -k /var/tmp/rapid-mlx-large-model.lock`. Remote checkpoint copies and
scratch were removed before returning the corresponding leases.

## Reproduction

Use a leased host with the required disk reserve. Put a main export in
`<scratch>/main`, the candidate in `<scratch>/wt`, checkpoint copies inside
remote scratch, and an editable installation in `<scratch>/venv`. Use the same
venv for both source trees; the harness sets `PYTHONPATH` for each server.
It stops only its own subprocess and refuses occupied ports. On Studio use
the existing HF cache and take the shared GPU lock in addition to the lease.

```sh
python scripts/benchmark_moe_decode_fusion.py <local-model-directory> \
  --workdir <scratch> --contexts 256,4096 --order ABAB --out serving.jsonl
python scripts/benchmark_moe_decode_fusion.py <local-model-directory> \
  --workdir <scratch> --contexts 256 --order ABAB --concurrency 4 --out c4.jsonl
python scripts/quality_moe_decode_fusion.py --model <local-model-directory> \
  --text README.md --output quality.json
```

The serving harness uses `--host 127.0.0.1`, a port in 18950–18999,
`--disable-prefix-cache --disable-disk-caches --no-spec-decode --max-tokens 128`.
Requests set temperature 0, top_p 1, top_k 0, min_p 0 and repetition_penalty 1.
Fixed prompt text is truncated by the checkpoint tokenizer to the requested
context length. Metal peak values come from `/v1/status`, rounded to 0.01 GB.
