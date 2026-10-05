# MTP under concurrency on an M4 Pro (2026-10-02)

## Decision

A verified MTP alias keeps the singleton MTP verifier for a lone request and
serves concurrent requests on the ordinary batched decode path. The
singleton yields at its next token boundary once another request is waiting
(PR "singleton verifier yields to waiting requests at a token boundary");
the continuous self-MTP cohort remains an explicit opt-in
(`{"method":"mtp","continuous_batching":true}`). Verified artifacts keep the
unquantized BF16 cache contract their qualification measured.

The [continuous MTP qualification](2026-08-31-qwen35-dense-continuous-mtp.md)
compared the continuous cohort with the serialized ordinary-MTP scheduler. It
did not compare either with plain batched decode, which is the relevant
alternative once the singleton no longer serializes the queue.

## Why a batched MTP verify loses at two to four lanes

A continuous cohort verifies `K+1 = 3` rows per lane in one target forward.
On this host the 4-bit quantized matmuls cost roughly linearly per row from
about 3 to 12 rows and only flatten inside the 32-row tile, so a 4-lane verify
(12 rows) costs about as much as 2.8-4.6 single-token steps while it delivers
about 2.4 tokens per lane.

Target forward latency (`return_hidden=True`, median of 12, ms), batch `B`
lanes times `S` positions per lane:

| rows (B x S) | qwen3.5-9b-4bit | qwen3.6-35b-4bit |
| --- | ---: | ---: |
| 1 (1x1) | 23.3 | 13.0 |
| 2 (2x1) | 24.5 | 16.8 |
| 3 (1x3) | 30.4 | 20.7 |
| 4 (4x1) | 37.9 | 25.6 |
| 6 (2x3) | 56.3 | 34.8 |
| 12 (4x3) | 106.5 | 56.3 |
| 24 (8x3) | 115.4 | 86.7 |

MTP drafter step: 3.1 / 5.7 ms (9b, B=1 / B=4) and 1.7 / 2.9 ms (35b).

Offline engine throughput, 256 tokens per lane, greedy (tok/s):

| | B=1 | B=2 | B=4 |
| --- | ---: | ---: | ---: |
| 9b plain batched decode | 47.2 | 85.5 | 106.3 |
| 9b continuous cohort (after the single-sync change) | 62.0 | 74.6 | 81.2 |
| 9b singleton verifier | 67.4 | - | - |
| 35b plain batched decode | 90.5 | 130.9 | 160.8 |
| 35b continuous cohort (after the single-sync change) | 92.4 | 118.0 | 142.5 |
| 35b singleton verifier | 102.7 | - | - |

The cohort only wins once the row count saturates the tile (about 6+ lanes
for the 9b model by the cost table), which is outside the measured local use.

## Served results

M4 Pro (Mac mini, 48 GB), macOS 26.5.1, mlx 0.32.3, mlx-lm 0.31.3, base
`main` at `4e51b93d7`. One server at a time, fresh `HOME`, shared HF cache,
loopback, default serve flags except as listed, GPU idle between servers.
Aggregate decode tok/s = sum of engine-reported completion tokens / wall time
of N simultaneous streaming requests (600-word explainer prompts,
`max_tokens=256`, `temperature=0`, default thinking), median of two reps.

| model | condition | c1 | c2 | c4 |
| --- | --- | ---: | ---: | ---: |
| qwen3.6-35b-4bit | main, MTP default (continuous cohort) | 97.7 | 111.8 | 109.6 |
| qwen3.6-35b-4bit | continuous cohort with dynamic membership | 92.2 | 103.7 | 118.6 |
| qwen3.6-35b-4bit | this decision (singleton MTP + yield + plain batching) | 98.9 | 135.2 | 164.3 |
| qwen3.6-35b-4bit | `--no-spec-decode` | 93.1 | 137.7 | 166.1 |
| qwen3.5-9b-4bit | main, MTP default (continuous cohort) | 58.6 | 73.6 | 68.1 |
| qwen3.5-9b-4bit | continuous cohort with dynamic membership | 60.6 | 62.6 | 66.6 |
| qwen3.5-9b-4bit | this decision | 60.1 | 84.0 | 104.7 |
| qwen3.5-9b-4bit | `--no-spec-decode` | 46.2 | 84.4 | 105.5 |

The main c2 median hides a bimodal result: the second request either
serialized behind the first (35b 86.4 tok/s, TTFT 3.1 s) or ran as plain
batched decode because both arrived in the same scheduler tick (137.2). The
decision removes the serialized mode (35b c2 reps 132.8 / 137.7, both TTFTs
under 0.2 s).

Single-stream rates vary by a few percent between runs because the singleton
verifier's depth controller adapts K from measured round times; a 5-rep c1
run gave medians of 100.3 (main) and 98.8 tok/s (this decision) for the 35b
model with overlapping ranges.

## Correctness

- Concurrent greedy output equals `--no-spec-decode` concurrent output
  token for token (4/4 prompts on both models).
- Mid-stream yield: request A alone versus A with a second request arriving
  1.5 s later (the singleton yielded after 93-175 emitted tokens). Every
  divergence occurred before the yield and reproduced identically on `main`
  without any yield (9b: 2/3 identical; 35b: 1/3 identical, the other two
  diverging at the same character offsets on `main`). The run-to-run
  divergences come from the depth controller choosing different verify
  shapes, which flips near-tied quantized argmaxes.
- Single-request output is unchanged: the singleton verifier path is the
  same code (9b sequential outputs identical to `main`, 4/4).

## Reproduce

```bash
rapid-mlx serve qwen3.6-35b-4bit --host 127.0.0.1 --port 18911
python3 docs/engineering/performance/perf_mtp_concurrency.py \
  --base-url http://127.0.0.1:18911/v1 --model <served model id> --out conc.json
```

`--no-spec-decode` and `--speculative-config` select the comparison
conditions. The client is stdlib-only and refuses non-loopback URLs.

## Limitations

- One host class (M4 Pro). The cost curve has the same shape on other Apple
  GPUs, but the lane count where a cohort would win is not measured here.
- Two reps per cell; c2 and c4 use identical prompts across reps, so prefix
  cache hits are part of the workload, as in normal multi-turn use.
- Seeded sampled requests keep the singleton verifier (they do not yield), so
  their reproducibility does not depend on concurrency.
