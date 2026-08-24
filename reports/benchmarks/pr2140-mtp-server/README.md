# PR #2140 server-level MTP benchmark

This is an exploratory end-to-end positive control for non-greedy MTP through
Rapid-MLX's production OpenAI-compatible streaming route. It complements the
in-process microbenchmark; it does not replace a repeated qualification run.

## Pinned inputs

- Rapid runtime: `7421ce3826d21b67dfa63d590c40ff2e899f489d`
- Target: `mlx-community/Qwen3.5-9B-4bit` at
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`
- MTP sidecar: `mlx-community/Qwen3.5-9B-MTP-4bit` at
  `222dfd2c23fc9518d7b817e4f8e0cb0571787489`
- Dataset: `nvidia/SPEED-Bench` at
  `487aa718444e816458d1a0a52bfce7a454285cf4`
- Sampling: temperature `0.6`, top-p `0.95`, top-k `20`
- Output limit: 256 tokens
- Prefix cache: disabled
- Warm-up: one discarded sample per cell
- Hardware and exact benchmark-file hash: recorded in each arm receipt

Baseline and MTP arms run in separate fresh server processes. Coding runs in
baseline-then-MTP order; the 8K cell runs MTP-then-baseline order. Each cell is
one bounded exploratory trial.

## Results

| Workload | Concurrency | Baseline aggregate tok/s | MTP aggregate tok/s | MTP / baseline | Median TTFT ratio | Median latency ratio | MTP attempts | Cell acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPEED-Bench qualitative/coding, 4 samples / 5 turns | 1 | 94.22 | 114.00 | 1.210x | 0.960x | 0.836x | 686 | 82.65% |
| SPEED-Bench qualitative/coding, 4 samples / 5 turns | 4 | 181.24 | 199.55 | 1.101x | 1.031x | 1.031x | 151 | 88.08% |
| SPEED-Bench throughput_8k, 4 samples | 1 | 40.79 | 41.72 | 1.023x | 1.075x | 1.014x | 578 | 61.94% |

The concurrency-1 coding cell is a valid MTP positive control: speculative
activity is nonzero and the server-level result is net-positive in this run.
Its controller chooses K=1 for 87.97% of measured rounds, K=0 for 2.31%, K=2
for 4.12%, and K=3 for 5.60%.

The concurrency-4 cell does not demonstrate overlapped MTP verification.
Rapid's scheduler routes generation batches with more than one UID through
ordinary AR. The MTP-configured process records only 151 attempts for 1,166
completion tokens, versus 686 attempts for 1,177 completion tokens at
concurrency 1; this is consistent with MTP reappearing when the batch drains
to a singleton tail. Median TTFT and latency are both about 3% slower in this
single cell. The 1.101x aggregate-token-rate ratio also includes a small
stochastic output-length difference and is not a clean claim of concurrent
MTP acceleration.

The 8K cell is approximately neutral: aggregate token rate is 1.023x, while
median TTFT is 1.075x and median latency is 1.014x. Cell-local acceptance falls
to 61.94%; the controller chooses K=0 for 14.74%, K=1 for 83.76%, and K>=2 for
1.50% of rounds. This result does not support a general claim that MTP gains
increase merely because context is longer.

## Interpretation limits

- Non-greedy sampling intentionally produces different valid token streams;
  this is a product-workload comparison, not token-for-token parity evidence.
- These cells contain four samples and one measured trial. They establish
  activity and expose scheduler/context behavior, but do not estimate a stable
  population speedup. A publishable performance claim needs repeated,
  counterbalanced trials and more task categories.
- Aggregate throughput is completion tokens divided by cell wall time. The
  per-request pooled rate, TTFT, latency, raw request results, metric snapshots,
  cell-local counter deltas, and SHA-256 checksums remain in the JSON receipts.
- This benchmark does not resolve the separate forced-K greedy divergence in
  `repro_mtp_forced_k_parity.py`.

## Reproduction

Use `bench/bench_spec_decode_mtp_server.py run` against fresh baseline and MTP
servers, followed by its `compare` subcommand. `bench/README.md` contains the
full command shape. The MTP concurrency-1 arm uses
`--require-mtp-activity`; a zero-attempt receipt fails closed.
