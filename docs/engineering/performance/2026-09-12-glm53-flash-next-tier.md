# GLM-5.3-Flash next-tier performance investigation

Date: 2026-09-12  
Owner: Vector  
Host: Studio (M3 Ultra)  
Target: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`

## Outcome

The next material GLM-5.3-Flash improvement should be native MTP with an exact,
transactional verifier. Quantization-format changes and isolated projection
fusions do not have evidence for a tier-sized gain. The strongest full-model
external result is mlx-vlm's optimized MTP path: 29.48 to 43.67 tok/s at batch
one and 512-token context (1.481x, 90.5% acceptance, exact output parity) on an
M3 Ultra.

Rapid must not enable this path from the external number alone. The optimized
implementation landed after the mlx-vlm v0.7.0 tag, changes a broad runtime
surface, and the exact 181.7 GB Rapid target is not currently present in the
policy-controlled Hugging Face cache. A paired Rapid server benchmark remains
the release gate.

## Current Rapid baseline

Benchmark artifact revision: `06d6a0240420137661ac3c84f845ae5e3513ff2f`.
The repository has since advanced to `76add2a341a1cd90ad0e86bb69839ea9c35827c6`;
the old revision no longer resolves through the Hub. Keep comparisons tied to
the revision that produced them.

| Context | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 128 | 13.81 | 32.38 |
| 2,048 | 348.04 | 28.36 |
| 8,192 | 358.37 | 27.75 |
| 32,768 | 276.90 | 27.20 |

Rapid's first native-head experiment used K=1. It measured 31.94 tok/s for
ordinary decode and 31.59 tok/s for MTP over 512 output tokens (0.989x), despite
72.97% acceptance and 5.71 GiB additional active memory. That implementation
split the recurrent KDA update at every verification boundary. It established
correct loading and rollback, but it did not amortize verification work.

### Current-revision real MTP head probe

The current immutable revision was probed without restoring the 181.7 GB
target. Its 2,641 layer-45 tensors span shards 1 and 2 (2,215 and 426 tensors),
rather than the single shard assumed by the older experiment. Both source
shards were downloaded through the policy-controlled HF cache. The upstream
GLM splitter produced a strict-loadable 3.9 GiB `glm5_next_mtp` drafter with
4-bit, group-size-64 affine weights and 3.896 GiB active memory.

Thirty measured one-token head forwards followed five warmups in each of three
campaigns. The per-campaign medians were 1.598, 1.541, and 1.317 ms; the median
of medians was 1.541 ms. This excludes the shared target embedding/output head
and does not predict E2E speedup by itself. It does show that the real drafter
block is about 5% of Rapid's approximately 31-32 ms ordinary target-token time.
The old K=1 regression is therefore more consistent with verification and
rollback overhead than with an intrinsically too-expensive drafter.

A second campaign added the checkpoint's real q4-g64 `lm_head` and greedy
argmax over its 154,880-token vocabulary. The three 30-sample medians were
2.148, 2.141, and 2.112 ms; the median of medians was 2.141 ms. The output head
is shared with the target in production, so its weights are not incremental MTP
memory. This complete one-token proposal is about 6.8% of an ordinary target
step, reinforcing that target verification/rollback is the dominant lever.

## External full-model evidence

### mlx-vlm GLM rewrite and MTP

The merged GLM work in `Blaizzy/mlx-vlm#2127` reports the following M3 Ultra
results for its final exact verifier:

| Batch | Base tok/s | MTP tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 29.48 | 43.67 | 1.481x |
| 2 | 45.09 | 64.54 | 1.431x |
| 4 | 70.77 | 83.03 | 1.173x |

The batch-one run used 512-token context and reported 90.5% acceptance with
parity. The implementation combines a batch-wide quantized verifier,
batch-grouped expert execution, a fused batch-one hyperconnection ingress, and
a transactional recurrent-cache rollback. Its native block size is three; the
benefit is not evidence that arbitrarily raising K is safe or faster.

Important packaging boundary: commit `bf4b861` is 22 commits after the v0.7.0
tag in the inspected checkout. PyPI `mlx-vlm==0.7.0` therefore does not contain
this optimized GLM implementation even though the source tree still reports
version 0.7.0.

### DwarfStar GLM kernels

The `IngeniousIdiocy/ds4` `glm53-m3ultra` branch reports:

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 62,174-token prefill | 365.7 tok/s | 550.4 tok/s | 1.505x |
| 62K-context decode | 23.75 tok/s | 38.07 tok/s | 1.603x |
| 300K-token prefill | 316.8 tok/s | 473.9 tok/s | 1.496x |
| 300K-context decode | 21.64 tok/s | 37.39 tok/s | 1.728x |
| Short-context decode | 29.2 tok/s | 40.9 tok/s | 1.401x |

Its useful lever map is dispatch-oriented: fold routed gate/up work, fuse
hyperconnection/residual epilogues, use expert-parallel routed down projection,
stage DSA queries once, and use bounded radix selection at long context. The
branch reports roughly 1,000 dispatches per token before this work, about 600
of them small. Its persistent all-in-one MoE megakernel was slower, so Rapid
should not pursue that design without contrary measurements.

### oMLX

oMLX v0.6.4 reports the following oQ4e results on M3 Ultra 512 GB:

| Context | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 4K | 482.3 | 24.1 |
| 16K | 443.2 | 23.7 |
| 32K | 449.6 | 23.5 |

oMLX also has a native DSA scorer/top-k extension that avoids materializing the
full heads-by-context intermediate. The published numbers use a different
169 GiB oQ4e artifact, so they are architecture references rather than a fair
Rapid throughput comparison.

## Local experiments and rejected shortcuts

All timings below are fresh-process MLX measurements on Studio unless noted.

| Experiment | Result | Decision |
| --- | --- | --- |
| Routed MoE gate/up structural fusion | median 1.037x, range 1.000-1.119x; byte-identical | Ship separately in PR #3334 |
| Shared-expert dense gate/up concat, 4096 to 2048 q4-g64, batch 1 | 0.2979 to 0.3080 ms (0.967x) | Reject |
| MLX `argpartition` plus selected sort, P=2K/4K/8K | 0.834x/0.858x/0.873x | Reject for normal context |
| Same selection, P=18,750 (about 300K tokens) | 1.230x | Long-context-only follow-up |
| Naive custom Metal DSA scorer | 7-10% slower than MLX matmul; bf16 top-k membership drift | Reject |

The dedicated `incoai/GLM-5.3-Flash-DFlash2` head is 2.34 GB and uses five
layers with block size eight. Its model card has no published benchmark and its
license is CC-BY-NC-ND-4.0. DwarfStar reports only +4.3% on a real coding agent
and a 2-3% prose regression for its DFlash2 path. Rapid should keep DFlash
disabled for this alias until it passes the existing sustained qualification.

## Compatibility finding

Rapid's released GLM correctness overlay subclasses the mlx-vlm 0.6.x
`Glm5NextSparseAttention` class. The post-0.7 rewrite replaces it with
architecture-owned `Glm5NextAttention`, `Glm5NextMoE`, and `Glm5NextMLP`
classes. Before this investigation, the overlay raised `AttributeError` before
the new runtime could load.

The accompanying compatibility change recognizes only the complete new class
family and retires the old overlay. Verification:

- pinned mlx-vlm runtime: 33 focused GLM tests passed;
- post-v0.7 upstream checkout: the overlay retired and the cached 0.1B GLM
  fixture loaded strictly;
- Rapid `MLXMultimodalLM` wrapper: the same five-layer fixture completed load
  when the dependency-version gate was intentionally bypassed for the probe.
- current-revision real layer-45 head: upstream split and strict load passed;
  1.541 ms median block-only latency, 2.141 ms including the real q4 output
  head and argmax, and 3.896 GiB incremental drafter memory.

This does not authorize a dependency bump. Atlas owns that compatibility
decision after upstream publishes a fixed revision.

## Implementation order and release gate

1. Land the fail-closed compatibility seam.
2. Pin a tagged mlx-vlm revision that contains the GLM rewrite, or vendor only
   the GLM drafter/verifier after an Atlas dependency review.
3. Restore the full target without redirecting the policy-controlled HF cache.
4. Run paired server tests at K=0, K=1, K=2, and K=3 with identical prompts and
   seeds; record decode tok/s, acceptance by position, TTFT, peak/active memory,
   and token/text equality.
5. Enable native MTP only if sustained batch-one decode is at least 1.10x with
   exact greedy parity, no workload bucket below 0.95x, and no server lifecycle
   regression. A default-on decision should require at least 1.20x.

## Primary references

- <https://github.com/Blaizzy/mlx-vlm/pull/2127>
- <https://github.com/IngeniousIdiocy/ds4/blob/glm53-m3ultra/README.md>
- <https://github.com/jundot/omlx/releases/tag/v0.6.4>
- <https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2>
- `docs/benchmarks/recent-large-models-m3-ultra.md`
