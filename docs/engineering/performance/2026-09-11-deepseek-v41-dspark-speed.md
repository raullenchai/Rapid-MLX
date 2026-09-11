# DeepSeek V4.1 Flash 2-bit decode experiments

Date: 2026-09-10/11

Status: experimental; the single-prompt K4 performance gate is met, while the
broader quality and long-context gates remain open.

## Goal and boundary

Determine whether the existing 2-bit REAP12.5 build can reach at least 12
generated tokens/s on a 256 GiB Mac by using its checkpoint-native three-stage
DSpark network, batched verification, or a faster MoE kernel. The stretch goal
is 20 tok/s. This work does not expose the model in the catalog or server.

All generation measurements use `(output_tokens - 1) / decode_seconds`; prompt
processing, model loading, and MTP packing are excluded from decode throughput.
Greedy equivalence means exact token equality with the autoregressive run in the
same process.

## Environment

- Apple M3 Ultra, 256 GiB unified memory
- MLX recommended working-set limit: 239.144 GB
- Target: `DeepSeek-V4.1-Flash-native-2bit-reap12_5`
- Target tensor bytes: 212.930 GB; 336 routed experts, six active
- DSpark sidecar: three stages, block size five, 128 routed experts, three active
- DSpark tensor bytes: 4.460 GB
- Prompt: 24 tokens, concise Python interval-merging task
- Output: 32 tokens, 31 measured transitions
- Target graph boundary: 40 layers
- Warm-tier weights were read sequentially before qualification; the reported
  ~240 second load is therefore not a cold-storage first-load claim.

Representative command:

```shell
python scripts/benchmark_deepseek_v41_dspark.py \
  --target <reap12.5-target> \
  --overlay <target-plus-dspark-overlay> \
  --checkpoint-runtime <trusted-checkpoint-runtime> \
  --trust-checkpoint-runtime \
  --packed-mtp-only \
  --tokens 32
```

## Results

| Candidate | tok/s | Change vs 7.90 AR | Greedy equivalent | Peak MLX memory |
| --- | ---: | ---: | --- | ---: |
| Autoregressive baseline | 7.90 | — | reference | 213.54 GB |
| Serial DSpark verification | 7.11 | -10.0% | yes | 218.02 GB |
| Batched K4, deferred correction | 9.76 | +23.5% | yes | 218.02 GB |
| Batched K4 + packed MTP MoE | 10.07 | +27.5% | yes | 218.00 GB |
| Batched K4 + packed/vectorized MTP | **10.55** | **+33.5%** | **yes** | 218.00 GB |
| Batched K5 + packed MTP MoE | 11.03 | +39.6% | no | 218.00 GB |
| Batched K5 + packed/vectorized MTP | **11.53** | **+46.0%** | no | 218.00 GB |
| Native small-route MoE AR | 9.24 | +17.0% | no | 213.54 GB |

K4 accepted 1.00 additional draft token per block. K5 accepted 1.21. The K5
token difference is produced by the target's batched numerical path, not by an
unverified draft token escaping verification. The product gate nevertheless
requires a broader quality qualification before treating it as acceptable.

The packed MTP MoE replaces five-position-by-three-expert Python loops with
three batched gather-QMM projections. Its one-token output is bit-exact with the
checkpoint implementation after preserving the checkpoint's required order:
apply route weights before the BF16 cast and down projection. Packing uses
4.247 GB, releases the equivalent individual resident expert tensors, and adds
only about 0.4 GB to peak memory.

Vectorizing the MTP attention output projection reduces its 5 x 8 grouped QMM
dispatch pattern to one group-and-position-batched call. This changes low-bit
drafter numerics and may change proposals, but target verification remains
authoritative. On the qualification prompt it preserved the same acceptance
length and added 4.8% over packed MTP alone at K5.

## Target-forward ceiling

Oracle runs feed known-correct target tokens in chunks and measure only the
target forward:

| Verify rows | Target rows/s | Greedy rows matching sequential path |
| ---: | ---: | ---: |
| 2 | 13.6–14.0 | 30/31 |
| 3 | 19.0–19.3 | 30/31 |
| 4 | 24.6–24.9 | 30/31 |
| 5 | 27.4–27.6 | 30/31 |

The target is fast enough when rows batch, but observed DSpark acceptance is
only 1.0–1.2 extra tokens/block. K5 spends 2.40 seconds of its 2.77-second
decode in target verification, so further draft-only optimization cannot reach
20 tok/s and is unlikely to clear 12 tok/s by itself.

## Exact affine 2-bit down-projection follow-up

A route-direct QMV kernel now specializes only the expert down projection. It
keeps gate/up, activation, routing, and weighted reduction on their stock paths.
That boundary matters: attempts to specialize gate/up changed BF16 arithmetic,
while the down-only candidate is element-for-element equal to the stock result.

Real layer 20 measurements, using the same target and six routes per token:

| Input tokens | Routed rows | Layer speedup | Maximum absolute difference |
| ---: | ---: | ---: | ---: |
| 1 | 6 | 2.65x | 0 |
| 4 | 24 | 1.94x | 0 |
| 5 | 30 | 1.82x | 0 |
| 6 | 36 | 1.72x | 0 |

Across three full target-only 32-token runs, the specialization reached
**9.49–9.83 tok/s** from 7.81–7.92 tok/s (+21.5% to +24.0%). The complete
generated token sequence was identical and peak MLX memory remained 213.5387
GB. Oracle target throughput was 17.91–18.13, 25.38–25.49, 32.39–33.86, and
34.49–36.32 rows/s at K2 through K5 respectively. K4 is 30–36% faster than the
earlier 24.6–24.9 rows/s range.

Representative target-only command:

```shell
python scripts/benchmark_deepseek_v41_dspark.py \
  --target <reap12.5-target> \
  --direct-down-qmv \
  --target-only \
  --tokens 32
```

The trusted checkpoint runtime was recovered from the already-present model
cache and the combined path was remeasured without downloading or copying model
weights:

| Candidate | tok/s | Change vs same-run 7.81 AR | Greedy equivalent | Peak MLX memory |
| --- | ---: | ---: | --- | ---: |
| K4 packed DSpark + direct down QMV | **12.70** | **+62.5%** | **yes** | 218.00 GB |
| K5 packed DSpark + direct down QMV | 13.96 | +78.8% | no | 218.00 GB |

K4 accepted exactly one additional draft token per block. Its 2.442-second
decode spent 0.297 seconds in the draft, 2.096 seconds in target verification,
and 0.006 seconds in rollback. This is the first exact-greedy configuration to
cross the 12 tok/s performance floor, but it is still one 32-token prompt rather
than the multi-domain, long-context qualification required for product exposure.

## Product-owned draft runtime

The K4 path was then moved behind a Rapid-owned, data-only DSpark runtime. The
default benchmark no longer imports Python from the model directory; the old
runtime remains available only as an explicit trusted A/B oracle. The sidecar
loader validates a basename-only index, rejects symlinked MTP shards, requires
the complete three-stage tensor contract, shares the target embedding and head,
and admits its 4.46 GB allocation against both host and Metal headroom.

On actual sidecar tensors, six observed KV windows, the six-token proposal, and
all five confidence logits were element-for-element equal between the old and
owned implementations (maximum difference 0). A complete default-path run on
the same Studio measured:

| Candidate | tok/s | Change vs same-run 7.82 AR | Greedy equivalent | Peak MLX memory |
| --- | ---: | ---: | --- | ---: |
| K4 owned packed DSpark + direct down QMV | **13.42** | **+71.7%** | **yes** | **218.00 GB** |
| K5 owned packed DSpark + direct down QMV | 14.61 | +86.9% | no | 218.00 GB |

An adversarial full-path run first exposed a 4.25 GB ownership leak: after
packing experts, the loader retained the unpacked arrays as well. The final
implementation explicitly releases all 3,456 replaced expert tensors; a repeat
run reduced peak memory from 222.27 GB back to 218.00 GB without changing K4
output. K5 remains excluded regardless of its higher throughput.

## Rejected paths

- Immediate singleton correction erases batching gains. Exact cache rollback
  plus delayed correction is required; rollback itself costs about 4–6 ms for
  the complete 32-token run.
- Confidence thresholds 0.3, 0.5, and 0.7 did not improve the speed/equivalence
  frontier. The checkpoint confidence scores are not calibrated well enough to
  compensate for shorter target batches on this workload.
- A native affine small-route MoE kernel is 1.76x faster in the isolated
  six-route microbenchmark and improves AR by 17%, but changes greedy output.
  At the 24–36 routes used by verification it is 21–33% slower than the stock
  gather-QMM path.
- Gate/up projection fusion is bit-exact but improves a real K5-shaped MoE layer
  by only 2.8%. Full-model post-load fusion exceeds 256 GiB transient capacity
  and fails with Metal insufficient memory, so an offline fused artifact is not
  justified.
- Moving both Engram tables (61.442 GB) to SSD lowers the unpruned target's
  active memory from about 234 GB to 172.74 GB, but random row reads reduce AR
  to 5.49 tok/s and K5 to 6.42 tok/s. This is a capacity option, not a speed
  option, and was removed from the implementation.
- The unpruned target did not materially improve DSpark acceptance (1.13 extra
  tokens/block versus 1.21 for REAP12.5 on this task), rejecting the hypothesis
  that REAP target/drafter mismatch is the dominant loss.

## Product decision and next experiments

Do not enable DSpark for DeepSeek V4.1 Flash by default yet. Product-owned K4
now reaches 13.42 tok/s with an exact greedy stream on the qualification prompt,
clearing the performance floor. It still needs stable multi-domain 128-token
and long-context cache qualification. K5 reaches 14.61 tok/s but remains
excluded because its batched target numerics change the greedy stream.

Reaching 20 tok/s requires a new capability rather than threshold tuning:

1. Run K4 on a multi-domain 128-token prompt suite and verify exact target
   authority, sustained throughput, and cache rollback at longer contexts.
2. A model-trained MHC-aligned block drafter with materially higher accepted
   length. No compatible V4.1 checkpoint was available during this experiment.
3. After either exists, re-run a multi-domain, 128-token suite and long-context
   cache qualification before server/catalog integration.
