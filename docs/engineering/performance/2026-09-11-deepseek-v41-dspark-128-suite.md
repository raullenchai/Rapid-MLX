# DeepSeek V4.1 Flash K4 128-token qualification

Date: 2026-09-11

Status: qualified as a deterministic, target-authoritative experimental path;
not qualified as an unconditional default because low-acceptance prompts can
regress against autoregressive decoding.

## Boundary and environment

This qualification tests the exact affine 2-bit down-QMV candidate together
with packed, vectorized K4 DSpark. It does not change server, catalog, GUI, or
default inference behavior.

- Apple M3 Ultra, 256 GiB unified memory
- Target: 212.93 GB REAP12.5 native affine 2-bit checkpoint
- Draft: 4.46 GB three-stage DSpark sidecar loaded by Rapid-owned runtime code
- Four fixed prompt domains: code, arithmetic reasoning, JSON-only structured
  output, and Chinese explanation
- Maximum output: 128 tokens per prompt
- Two consecutive K4 repeats per prompt in one loaded process
- Comparison: complete token sequence against sequential autoregressive output
  from the same loaded target, prompt, and direct-QMV target kernels

Representative command:

```shell
python scripts/qualify_deepseek_v41_dspark_suite.py \
  --target <reap12.5-target> \
  --overlay <target-plus-dspark-overlay> \
  --tokens 128 \
  --repeats 2 \
  --collect-target-margins
```

## Results

| Domain | AR tok/s | K4 tok/s, two-run weighted | Change | Accepted draft tokens/block | Repeat stable |
| --- | ---: | ---: | ---: | ---: | --- |
| Code | 9.31 | 16.21 | +74.1% | 1.42 | yes |
| Reasoning | 9.34 | 7.69 | -17.7% | 0.44 | yes |
| Structured | 9.33 | 10.59 | +13.6% | 0.57 | yes |
| Chinese | 8.70 | 11.67 | +34.1% | 0.75 | yes |

The sustained A/B run produces 972 K4 transitions in 90.677 seconds, or
**10.72 tok/s**, versus **9.16 tok/s** for AR with identical target kernels
(**+17.1%**). Peak MLX memory is 218.07 GB and does not increase on the second
repeat. The average accepted draft length is 0.79 token/block. All four prompts
produce identical token sequences across both K4 repeats.

A separate K4-only two-repeat run reaches **11.76 tok/s** at the same 218.07 GB
peak. The lower sustained A/B number is the conservative product figure: it
captures the warm, continuous workload after the AR comparison rather than
cherry-picking an isolated pass.

| Domain | First divergent token index | Minimum batched top-2 margin | Rows below 0.05 |
| --- | ---: | ---: | ---: |
| Code | 79 | 0.0183 | 1 |
| Reasoning | 4 | 0.0099 | 11 |
| Structured | 15 | 0.0046 | 16 |
| Chinese | 7 | 0.0052 | 4 |

No prompt preserves the complete sequential-greedy stream. Batched target
logits remain authoritative: draft tokens are emitted only after target
verification, and corrections come from the target. The supported contract is
therefore target-authoritative decoding with deterministic output for a fixed
K4 execution shape, not bitwise identity with a different target batch shape.

An operator trace locates the numerical departure in target execution itself.
For the same five known-correct tokens, batched and serial target predictions
are initially identical, but layer-zero hyper-connection mixing first differs
by 5.96e-8. The first quantized query projection differs by 0.001953125 and the
difference amplifies through 40 layers to a final-logit maximum absolute
difference of 4.7786. Serializing every token-local projection would restore a
single batch shape at the cost of the speculative speedup, so that is not an
appropriate product fix.

The target itself also shows repetition on the reasoning and Chinese probes.
That is a model/artifact quality concern distinct from speculative correctness
and must be included in any later product qualification.

## Product decision and next experiment

A 0.05-margin hybrid that replayed only numerically ambiguous verification
blocks was tested and rejected. It triggered 1/8/9/4 replays across the four
domains, preserved 0/4 exact sequences, and reduced weighted throughput to
11.21 tok/s. The Chinese probe also reached a different early EOS. The replay
implementation was removed rather than adding ineffective controller
complexity.

Fixed K4 is suitable for an explicitly admitted experimental path on machines
with sufficient memory. It must not become an unconditional default: the
reasoning probe regresses because 0.44 accepted draft tokens per block cannot
repay verification overhead. Product integration must retain an AR fallback or
an evidence-backed admission rule, expose the target-authoritative semantics,
and pin the qualified artifact and runtime versions.

The next credible speed path is an adaptive controller that exits speculative
decoding when observed acceptance cannot repay its target-batch overhead.
Blindly increasing K is rejected: K5 is already known to change the short
greedy stream.
