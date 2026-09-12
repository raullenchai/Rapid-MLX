# DeepSeek V4.1 Flash fast hyper-connections

Date: 2026-09-11

Status: the release-layout Metal path passes kernel parity, real-model output
stability, and the four-domain throughput gate. It is enabled automatically for
the native V4.1 runtime on supported Apple Silicon layouts; portable and
non-release layouts retain the operation-based implementation.

## Question and boundary

DeepSeek V4.1 evaluates two hyper-connection cycles in each of 40 target layers.
The former path materialized the residual mixing graph and performed stream
collapse and RMS normalization as separate operations. This experiment asks
whether compiling the coefficient graph and fusing the two bandwidth-bound
operations can materially lower target verification time without changing the
fixed-K4 target-authoritative contract.

Only the native V4.1 hyper-connection path changes. Draft weights, target
weights, quantization, the model catalog, other model families, and adaptive
verification remain unchanged.

## Environment

- Apple M3 Ultra, 256 GiB unified memory (`hw.memsize=274877906944`)
- macOS 26.5.2 (25F84)
- Target: local 212.93 GB REAP12.5 native affine 2-bit checkpoint
- Engram tables: affine 2-bit SSD offload
- Draft: pinned 4-bit-dense/2-bit-expert DSpark sidecar
- Verification: fixed K=4, packed MTP, exact affine-2bit target down projection
- Model load: 163.52 seconds; peak qualification memory: 171.13 GB

No model was downloaded or relocated for this experiment. Both artifacts were
already present in the Studio storage hierarchy.

## Correctness

Focused Metal tests cover BF16, FP16, and FP32 post-mix and collapse-normalize
paths, four-way Sinkhorn, empty tensors, and unsupported multiplicities. At the
release width of 5120, BF16 fused and operation-based post-mix and
collapse-normalize outputs are elementwise equal for one-row and five-row
inputs. FP32 maximum absolute error in the focused suite is `2.38e-7`.

The four 128-token prompts are token-for-token stable across two consecutive
K4 runs. As in the existing fixed-shape product contract, K4 is
target-authoritative but is not required to match sequential AR because a
different target batch shape can cross low-margin decisions.

## Reproduction

```shell
python3.11 scripts/qualify_deepseek_v41_dspark_suite.py \
  --target <reap12.5-target> \
  --overlay <pinned-mixed-dspark-sidecar> \
  --engram-ssd-offload \
  --tokens 128 --repeats 2 --verify-k 4 \
  --collect-target-margins
```

The Studio was otherwise idle when the qualification began. The first repeat
includes Metal compilation and warm-tier page warming; the second repeat is the
steady-state result users see after warmup.

## Result

| Domain | First K4 tok/s | Warm K4 tok/s | Accepted tokens/block | Repeat stable |
| --- | ---: | ---: | ---: | --- |
| Code | 26.13 | 34.88 | 2.88 | yes |
| Reasoning | 15.91 | 25.04 | 1.78 | yes |
| Structured | 14.27 | 21.45 | 1.37 | yes |
| Chinese | 19.96 | 27.53 | 2.05 | yes |

Across both repeats, K4 produces 1,016 transitions in 47.348 seconds, or
**21.46 tok/s**. The warm repeat produces 508 transitions in 19.247 seconds,
or **26.39 tok/s**. The prior identically shaped four-domain qualification was
19.39 tok/s across both repeats, so the conservative all-repeat result improves
by **10.7%** and the steady-state result by **36.1%**.

Sequential AR now reaches a weighted **15.07 tok/s**, showing that the target
kernel improvement benefits non-speculative generation as well. A same-process
16-token code probe isolates the hot-path effect: the operation-based path is
9.60 tok/s and the warmed fused path is 34.90 tok/s with identical output and
the same three accepted draft tokens per block.

## Decision and next bottleneck

The fast path is worthwhile as a default for its strict release layout: it is
transparent to unsupported devices and shapes, preserves output stability, and
raises both AR and K4 throughput. High-acceptance code already reaches about 35
tok/s. The remaining gap to a four-domain 40 tok/s is dominated by draft
acceptance: target verification takes roughly 90 ms per K4 block after warmup,
while reasoning and structured prompts need many more blocks because they
accept only 1.78 and 1.37 draft tokens per block. The next experiment should
improve the draft head rather than widen this kernel PR.
