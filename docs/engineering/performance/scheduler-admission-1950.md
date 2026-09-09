# Contention-aware prompt admission benchmark

Date: 2026-09-07

This benchmark measures the interactive request most exposed to prompt-slot
head-of-line blocking: one short request arriving behind three long requests.
It compares the historical `fcfs` default with the opt-in
`shortest_validated_tail` policy. It does not measure or claim preemption of a
prompt that is already running.

## Environment

- Mac Studio, Apple M3 Ultra (28 CPU cores), 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3, Rapid-MLX 0.13.4
- Model: `mlx-community/Qwen3-0.6B-4bit`, already present in the local HF cache
- Base revision: `dcad4aa900f1711d850f9a27c7477a0a417cd411`
- Prefix cache disabled so prompt-tail ordering, not reuse, drives the result

## Server commands

Run each arm in a freshly started process:

```bash
uv run rapid-mlx serve mlx-community/Qwen3-0.6B-4bit \
  --host 127.0.0.1 --port 18124 \
  --prefill-batch-size 1 --completion-batch-size 4 --max-num-seqs 4 \
  --scheduling-policy fcfs --disable-prefix-cache --log-level WARNING
```

For the experimental arm, replace `fcfs` with
`shortest_validated_tail --scheduling-max-deferrals 8`.

Each of five measured waves sent one 8,000-word `contention` prompt, waited
50 ms, then sent two more identical long prompts at 0/10 ms and `Say OK.` at
20 ms. Every request used streaming, `temperature=0`, and `max_tokens=8`.
TTFT was measured from the short request's HTTP dispatch to its first non-empty
SSE content delta. The client closed each stream after that first delta, which
also exercised cancellation cleanup between waves.

## Results

| Policy | Short-request TTFT samples (s) | Median | Change |
| --- | --- | ---: | ---: |
| `fcfs` | 3.166, 3.041, 3.124, 3.043, 3.067 | 3.067 s | baseline |
| `shortest_validated_tail` | 0.262, 1.214, 1.208, 1.206, 1.206 | 1.206 s | 2.54x faster / 60.7% lower |

The first experimental sample selected the short request before the initial
long prompt acquired the sole prefill slot; the other four show the intended
non-preemptive case, where the short request waits for the active prompt and
then bypasses the two unstarted long prompts. The median therefore remains a
conservative non-preemptive result.

## Interpretation

The improvement comes from retaining excess requests in Rapid-MLX's queue
until a real prompt slot is available. That preserves the opportunity to rank
the currently grantable requests by cache-validated remaining prompt work.
Default `fcfs` behavior remains unchanged, and the opt-in policy forces an old
compatible request after eight pass-overs to bound starvation.

A separate matched-shape greedy check used a fresh B=1 server per policy,
`temperature=0`, `seed=42`, and the prompt “Reply with exactly three lowercase
color words.” Both arms returned byte-identical content: `black, white, gray`.
