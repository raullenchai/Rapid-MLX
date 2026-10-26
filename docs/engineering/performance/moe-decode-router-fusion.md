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

## Prior M5 evidence

On `studio-max` (M5 Max, 36 GB), the original implementation measured
174.17 → 199.41 tok/s (+14.49%) at 256 context and
166.64 → 189.92 tok/s (+13.97%) at 4096 context, against main
`e56b3e4a68d991d7213e566d4a441a6bd939dbd6`. Metal peak memory was
13.99 / 13.99 GB and 14.74 / 14.74 GB respectively. Single-request 128-token
greedy outputs matched exactly at both contexts.

Concurrent runs used the same ABAB method at 256 context with 128-token
responses. Aggregate throughput includes prefill and request completion.

| Requests | Main aggregate tok/s | Fused aggregate tok/s | Change | Main/fused peak GB |
|---|---:|---:|---:|---:|
| 2 | 204.90 | 205.83 | +0.45% | 14.01 / 14.01 |
| 4 | 277.13 | 276.65 | -0.17% | 14.04 / 14.04 |

Every response completed with 128 tokens; each variant produced the same set
of output texts, including scheduling-dependent variants at concurrency four.

Teacher forcing used the repository README at the main revision above: 256
prefill tokens followed by 1,024 forced tokens. Three independent caches share
one set of weights. Main and fused prefill in chunks of 128; the main noise
control uses chunks of 64. Fused vs main produced 1,024/1,024 matching top-1
predictions, zero mean KL and byte-identical logits. Main's chunking control
produced mean KL 0.0633344. No near-tie exception was needed in the greedy runs.

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
