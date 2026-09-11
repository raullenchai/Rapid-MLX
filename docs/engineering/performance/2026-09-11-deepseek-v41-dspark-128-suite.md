# DeepSeek V4.1 Flash K4 128-token qualification

Date: 2026-09-11

Status: rejected for product exposure; short-prompt performance did not
generalize with the required throughput and sequential-greedy equivalence.

## Boundary and environment

This qualification tests the exact affine 2-bit down-QMV candidate together
with packed, vectorized K4 DSpark. It does not change server, catalog, GUI, or
default inference behavior.

- Apple M3 Ultra, 256 GiB unified memory
- Target: 212.93 GB REAP12.5 native affine 2-bit checkpoint
- Draft: 4.46 GB checkpoint-native three-stage DSpark sidecar
- Four fixed prompt domains: code, arithmetic reasoning, JSON-only structured
  output, and Chinese explanation
- Maximum output: 128 tokens per prompt
- Comparison: complete token sequence against sequential autoregressive output
  from the same loaded target and prompt

Representative command:

```shell
python scripts/qualify_deepseek_v41_dspark_suite.py \
  --target <reap12.5-target> \
  --overlay <target-plus-dspark-overlay> \
  --checkpoint-runtime <trusted-checkpoint-runtime> \
  --trust-checkpoint-runtime \
  --tokens 128
```

## Results

| Domain | AR tok/s | K4 tok/s | Accepted draft tokens/block | Sequential-greedy equivalent |
| --- | ---: | ---: | ---: | --- |
| Code | 7.77 | 16.24 | 1.42 | no |
| Reasoning | 7.85 | 9.87 | 0.44 | no |
| Structured | 7.83 | 10.74 | 0.57 | no |
| Chinese | 7.84 | 12.03 | 0.75 | no |

K4 produces 486 measured transitions in 41.242 seconds, or **11.78 tok/s**
weighted across the suite. Peak MLX memory is 218.07 GB. The average accepted
draft length is 0.79 token/block. No prompt preserves the complete sequential
greedy stream.

| Domain | First divergent token index | Minimum batched top-2 margin | Rows below 0.05 |
| --- | ---: | ---: | ---: |
| Code | 79 | 0.0183 | 1 |
| Reasoning | 4 | 0.0099 | 11 |
| Structured | 15 | 0.0046 | 16 |
| Chinese | 7 | 0.0052 | 4 |

The 32-token coding probe reached 12.70 tok/s with an exact sequence, but the
longer and broader suite fails both required gates: weighted throughput is below
12 tok/s and exactness is 0/4. Batched target logits remain authoritative, so
this is not an unverified draft-token escape; nevertheless the resulting stream
does not meet the current sequential-equivalence product contract.

The target itself also shows repetition on the reasoning and Chinese probes.
That is a model/artifact quality concern distinct from speculative correctness
and must be included in any later product qualification.

## Next experiment

A 0.05-margin hybrid that replayed only numerically ambiguous verification
blocks was tested and rejected. It triggered 1/8/9/4 replays across the four
domains, preserved 0/4 exact sequences, and reduced weighted throughput to
11.21 tok/s. The Chinese probe also reached a different early EOS. The replay
implementation was removed rather than adding ineffective controller
complexity.

The next credible speed path must improve draft acceptance or make target batch
numerics stable by construction. Blindly increasing K is rejected: K5 is
already known to change the short greedy stream.
