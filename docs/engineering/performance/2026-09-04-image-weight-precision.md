# FLUX.2 Klein explicit bf16 execution path

Issue [#3058](https://github.com/raullenchai/Rapid-MLX/issues/3058)
measured FLUX.2 Klein 4B at 1024×1024, four steps, with the same prompt and
seed on an otherwise-idle 32 GB M2 Pro Mac mini:

The broader issue baseline used Rapid-MLX 0.13.4, mflux 0.19.1, macOS 26.5.2,
six prompts per model, and observed under 3% variance:

| Model and default | M3 Ultra 256 GB | M2 Pro 32 GB |
|---|---:|---:|
| FLUX.2 Klein q4, 1024 square, 4 steps | 9.2 s | 52 s |
| Z-Image Turbo q4, 1024 square, 8 steps | 34 s | 150 s |

Those cross-machine values are user-expectation baselines, not a controlled
hardware comparison: the machines have different GPU resources. The precision
comparison below changes weights on the same M2 Pro workload.

| Weight path | Wall time per image |
|---|---:|
| BFL bf16 | 44.0 / 47.4 s |
| BFL quantized on load to q4 | 52.5 / 52.0 s |
| Pre-quantized q4 alias | 51.5 / 52.7 s |

The mean q4-to-bf16 throughput gain is about 1.14×. The matching on-load and
pre-quantized q4 times isolate execution precision, rather than checkpoint
conversion, as the useful lever for this workload. These figures are supplied
measurements from the issue, not a new run performed by this change.

## Branch dogfood

The implementation branch was then dogfooded through the real `rapid-mlx serve`
and `/v1/images/generations` path on the same class of host:

- Mac mini, Apple M2 Pro, 32 GB unified memory, macOS 26.5.2;
- mflux 0.19.1, MLX 0.32.2, low-power mode off, no recorded thermal warning;
- otherwise-idle host, with its resident QSP service stopped for the A/B;
- identical prompt, seed 1001, 1024×1024 output, and four inference steps;
- two independent server processes per checkpoint, each with one discarded
  warm-up, followed by five measured requests combined across both processes.

| Weight path | Warm-up | Measured requests | Median | p95 (nearest rank) |
|---|---:|---:|---:|---:|
| Pinned pre-quantized q4 alias | 52.431 / 52.085 s | 54.643 / 60.425 / 51.882 / 52.463 / 52.703 s | 52.703 s | 60.425 s |
| Pinned packaged bf16 alias | 43.287 / 44.036 s | 46.065 / 41.820 / 40.538 / 42.805 / 40.844 s | 41.820 s | 46.065 s |

The packaged bf16 path reduced median end-to-end wall time by 20.6%, or 1.26×
throughput. Every repeat within a precision produced the same PNG SHA-256;
q4 and bf16 intentionally produced different bytes because the denoising
weights differ. Visual inspection found both outputs coherent and faithful to
the paper-crane prompt. A direct-engine companion run measured peak process RSS
at approximately 4.57 GB for q4 and 12.49 GB for bf16. Raw timing JSON remains
on the mini under `~/qsp-node/image_weight_dogfood_{exact_q4,exact_bf16}.json`.

The bf16 run used the exact pinned snapshot below. Before generation,
`mflux_missing_weights()` returned `[]`, proving all indexed component shards
were locally present. The QSP LaunchAgent was restored after the run and its
health endpoint passed.

Rapid-MLX therefore exposes an opt-in path while preserving existing behavior:

```bash
rapid-mlx serve flux2-klein-4b --image-weight-precision bf16
# Equivalent explicit alias:
rapid-mlx serve flux2-klein-4b-bf16
```

The bf16 alias uses the pinned, mflux-layout
`mflux-community/flux2-klein-4b-mflux-bf16` snapshot (15,975,684,703 bytes),
loads it through `model_path`, and passes `quantize=None`. The q4 alias and all
other models are unchanged. A range-read of the pinned snapshot's first
transformer shard found 69 tensors, all declared `BF16`, and no `.scales` or
`.biases` quantization auxiliaries. Automatic chip selection is deliberately
not adopted: the one measured M2 Pro working set is not enough to define a safe
M1/M2-family policy, and silently changing the public alias would also change
its download size, memory demand, and output bytes. The explicit BF16 alias is
the stable policy until each additional hardware/model combination has
independent qualification.

## Completion telemetry

The Images generation route logs one completion line per image. mflux exposes
callbacks after prompt encoding and after the final synchronized denoise step,
so this measurement adds no evaluation barrier to the hot path and keeps model
load, prompt encoding, VAE decode, PNG encoding, and base64 work separate from
the denoise-only rate:

```text
Image generation: model=... family=flux2-klein image=1/1 size=1024x1024 steps=4 total=...s denoise=11.20s (2.80 s/step, ~13.6 estimated TFLOPS)
```

The TFLOPS value is an operation-count estimate, not a hardware counter. It is
emitted only for FLUX.2 Klein at 1024 square, using the issue's approximately
38 TFLOP per denoise-step derivation divided by measured seconds per step. Other
families and sizes still report exact total time; backends without both denoise
boundaries say `denoise timing unavailable` rather than re-labelling
end-to-end time or extrapolating an unqualified FLOP count.

## Reproduction checklist

For a qualification run, record `sw_vers`, `sysctl -n machdep.cpu.brand_string`,
physical RAM, Rapid-MLX/mflux/MLX versions, power mode, and whether another model
is resident. Warm each path once, then run at least five timed images with an
identical prompt, seed, dimensions, and step count; report median and p95 plus
peak resident memory.

## Other diffusion lanes

DiffusionGemma is **discrete text diffusion**, not an mflux image model. Its
default canvas is 256 text tokens and each denoising step can likewise present
multi-row matrix multiplies, so the same q4-kernel crossover is a plausible
benchmark target. It is not covered by this switch: the curated checkpoint is
a 26B-total/4B-active MoE, uses mixed 4/8-bit weights, and a full bf16 copy would
exceed the 32 GB target where Klein bf16 fits.

A shape-level probe on the Studio (`Apple M3 Ultra`, macOS 26.5.2, MLX 0.32.2),
using DiffusionGemma's real dense projection shape `M×2816 @ 2816×4096`, group
size 64, five warm-ups and 20 individually synchronized samples, measured:

| Canvas rows | bf16 median | q8 median | q4 median |
|---:|---:|---:|---:|
| 256 | 0.676 ms | 0.591 ms | 0.586 ms |
| 4096 | 6.475 ms | 6.260 ms | 6.183 ms |

On M3 Ultra, dequantizing that projection would therefore be a regression, not
an optimization. This microbenchmark does not cover the model's grouped MoE
expert kernels or prove the result on M1/M2. A useful follow-up is an
end-to-end 4-bit versus 8-bit and selective-dequantization profile on a
high-memory M1/M2 host; until that exists, changing DiffusionGemma's weight
precision would be an unmeasured compatibility and memory-policy change.

Reproduce the dense projection probe with:

```python
import statistics, time
import mlx.core as mx

def bench(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)

for rows in (256, 4096):
    x = mx.random.normal((rows, 2816)).astype(mx.bfloat16)
    weight = mx.random.normal((4096, 2816)).astype(mx.bfloat16)
    mx.eval(x, weight)
    print(rows, "bf16", bench(lambda: x @ weight.T))
    for bits in (8, 4):
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
        mx.eval(packed, scales, biases)
        print(
            rows,
            f"q{bits}",
            bench(
                lambda: mx.quantized_matmul(
                    x, packed, scales, biases, transpose=True,
                    group_size=64, bits=bits,
                )
            ),
        )
```
