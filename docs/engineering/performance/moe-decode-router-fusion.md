# Single-token MoE router and combine fusion

Status: performance acceptance incomplete. The M5 Max LFM target passes;
Qwen gains about 2.2%, below the requested 6%. M4 measurements are blocked by
the machine's disk reserve. These results do not authorize merging.

## Implementation and serving boundary

The loaded model's supported MoE instances receive a local subclass. No upstream
class is modified globally by this patch and no weight buffers are duplicated.
The existing gate/up fusion remains unchanged. Both the text engine loader and
the multimodal language loader enroll blocks after existing routing patches.
Qwen3.5/3.6 text serving shares the latter loader's language weights.

Fusion requires `[B,L,H]` with `B*L == 1`, Metal as the default device,
32–256 experts in multiples of 32, top-k 1–8, and H divisible by 128. Supported
gates are bias-free standard Linear with f32/bf16 weights and standard
QuantizedLinear with compatible packed dimensions. Quantized gates retain their
existing projection kernel. Other configurations invoke the prior block call.
Sharded blocks, larger batches, and prefill retain that call too.

The scheduler forwards the active tokens as `inputs[:, None]`; one active
request therefore supplies `[1,1,H]`, with no padding to a minimum larger batch.
With more active requests the batch path remains unchanged. Admission and
completion can still leave single-request steps that qualify for fusion.

Set `RAPID_MLX_MOE_DECODE_FUSION=0` before starting `rapid-mlx serve` to disable
installation. The switch also bypasses already-enrolled blocks.

Numerics preserve the dependency's GEMV lane split, multiply-add and reduction
order, four-value softmax loads, fast exponential with f32 accumulation,
reciprocal multiplication, post-softmax LFM bias and epsilon, rounded Qwen
normalization, and dtype-rounded products and accumulation. The shared sigmoid
also retains intermediate dtype rounding. Keeping only f32 accumulation until
the final bf16 cast was insufficient for model-level agreement.

## M5 Max streaming measurements

Machine: `studio-max`, M5 Max, 36 GB. Runtime: Python 3.12, MLX 0.32.4,
mlx-lm 0.31.3, mlx-vlm 0.7.2. Two idle virtualization processes of roughly
1.8 GB RSS each were present and left running. Initial free disk: 294 GiB.

A is an export of main `e56b3e4a68d991d7213e566d4a441a6bd939dbd6`;
B is the candidate source tree. One venv serves both. Order ABAB, separate
server launches, 16-token warmup, three measured 128-token requests per context
per launch, greedy sampling, prefix/disk caches and speculation disabled.
Numbers are medians of six samples per variant/context. Decode rate is
`(completion_tokens-1)/(last_text_chunk-first_text_chunk)` from streaming
`POST /v1/completions`; TTFT includes tokenization, prefill, and serving overhead.

| Model | Context | Main tok/s | Fused tok/s | Change | Main/fused TTFT ms | Main/fused Metal peak GB |
|---|---:|---:|---:|---:|---:|---:|
| LFM2.5-24B-A2B-MLX-4bit | 256 | 174.17 | 199.41 | +14.49% | 84.96 / 84.67 | 13.99 / 13.99 |
| LFM2.5-24B-A2B-MLX-4bit | 4096 | 166.64 | 189.92 | +13.97% | 567.66 / 566.14 | 14.74 / 14.74 |
| Qwen3.6-35B-A3B-4bit | 256 | 150.60 | 153.87 | +2.17% | 128.29 / 127.46 | 20.86 / 20.86 |
| Qwen3.6-35B-A3B-4bit | 4096 | 144.28 | 147.35 | +2.13% | 969.27 / 970.70 | 22.46 / 22.46 |

Checkpoints: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` revision
`667408d494bde191c2cb9dd4f4549827816b9275`, and
`mlx-community/Qwen3.6-35B-A3B-4bit` revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`.
The Qwen checkpoint has 8-bit quantized router and shared-expert gates.
Main already fuses Qwen's routing tail, reducing the incremental opportunity.

All measured single-request 128-token outputs matched between variants at both
contexts. TTFT differences were within observed run scatter. Metal peak values
come from `/v1/status`, rounded to 0.01 GB; no increase was observable at that
resolution, below the 50 MB allowance.

Concurrent runs used the same ABAB method, 256-token prompts and 128-token
responses. Aggregate throughput here includes prefill and request completion.

| Model | Requests | Main aggregate tok/s | Fused aggregate tok/s | Change | Main/fused peak GB |
|---|---:|---:|---:|---:|---:|
| LFM2.5 | 2 | 204.90 | 205.83 | +0.45% | 14.01 / 14.01 |
| LFM2.5 | 4 | 277.13 | 276.65 | -0.17% | 14.04 / 14.04 |
| Qwen3.6 | 2 | 179.02 | 178.95 | -0.04% | 20.94 / 20.94 |
| Qwen3.6 | 4 | 234.60 | 233.96 | -0.27% | 21.10 / 21.10 |

Every response completed with 128 tokens. Each variant produced the same set
of output texts. Scheduling produced two text variants in both main and fused
runs for Qwen and for four-request LFM; two-request LFM produced one.

## Quality

Teacher forcing used the repository README at the main revision above: 256
prefill tokens followed by 1,024 forced tokens. Three independent caches share
one set of weights. Main and fused prefill in chunks of 128; the main noise
control uses chunks of 64. Both models produced 1,024/1,024 matching top-1
predictions, zero mean KL, and zero maximum logit difference for fused vs main.
Main's chunking control produced mean KL 0.0633344 for LFM and 0.00771299 for
Qwen. No near-tie exception was needed in the single-request greedy runs.

## Reproduction

Use a leased host with the required free disk reserve. Put a main export in
`<scratch>/main`, the candidate in `<scratch>/wt`, local checkpoint copies
inside scratch, and the editable installation in `<scratch>/venv`. Use the same
venv for both source trees; the harness sets `PYTHONPATH` for each server.
It stops only its own subprocess and refuses occupied ports.

```sh
python scripts/benchmark_moe_decode_fusion.py <local-model-directory> \
  --workdir <scratch> --contexts 256,4096 --order ABAB --out serving.jsonl
python scripts/benchmark_moe_decode_fusion.py <local-model-directory> \
  --workdir <scratch> --contexts 256 --order ABAB --concurrency 4 --out c4.jsonl
python scripts/quality_moe_decode_fusion.py --model <local-model-directory> \
  --text README.md --output quality.json
# Qwen serving's shared-weight language implementation:
python scripts/quality_moe_decode_fusion.py --model <local-qwen-directory> \
  --loader multimodal --text README.md --output quality-qwen.json
```

The serving harness starts `rapid-mlx serve` with `--host 127.0.0.1`, a port in
18950–18999, `--disable-prefix-cache --disable-disk-caches --no-spec-decode`,
and `--max-tokens 128`. Requests explicitly set temperature 0, top_p 1,
top_k 0, min_p 0 and repetition_penalty 1. Prompt text is fixed in the harness,
truncated by the checkpoint tokenizer to the requested context length.

## Remaining acceptance

`mzr-2` (M4 Pro, 48 GB) was briefly leased for preflight. It had 76 GiB free,
below the required 100 GB reserve before copying either checkpoint. No scratch
or model copies were created there; the lease was released immediately. M4
results require the machine owner to restore the reserve. Qwen still needs a
further in-scope improvement to reach +6% on M5; current data do not meet that
acceptance gate.
