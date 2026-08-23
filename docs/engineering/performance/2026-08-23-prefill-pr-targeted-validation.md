# Targeted validation of the August prefill PRs

## Result

The recent prefill work is release-ready for the cached models tested, but the
user-visible claim needs to distinguish two different effects:

- PR #2199's MLX 0.32.1 upgrade improved end-to-end cold service TTFT by
  3.7-5.6% on Qwen3.5 4B and Gemma 4 26B-A4B. This service result does not
  reproduce the PR's 20-34% isolated raw-prefill gains because request,
  scheduling, sampling, and streaming overhead remain in the TTFT measurement.
- PRs #2203, #2210, #2211, #2213, and #2215 reduced contended short-request
  TTFT by 54-56% and peak Metal memory by 22-24% on Qwen3.5 4B, Qwen3.5 9B,
  and Gemma 4 12B. Uncontended 2K TTFT regressed by 2-6%, which is the expected
  fairness tradeoff of the smaller prefill chunks.

No release blocker was found. Do not advertise a blanket 10-20% improvement
to decode speed or all service TTFT. The supported claim is substantially
better short-request latency and memory under long-prefill contention, plus a
smaller single-request service-TTFT improvement from MLX 0.32.1.

## PR and model scope

| PR | Change | Affected or benchmarked models |
| --- | --- | --- |
| #2199 | MLX 0.31.2 to 0.32.1 | Qwen3.5 4B, Gemma 4 26B-A4B, Qwen3.8 27B |
| #2203 / #2210 | recurrent prefill auto-default | Qwen3.5 4B initially |
| #2211 | conservative measured allowlist | Qwen3.5 4B/9B 4-bit; negative controls Bonsai 8B, LFM2.5 2.6B, Qwen3.5 35B-A3B |
| #2213 | expanded measured profiles | Qwen3.5 4B/9B 6/8-bit and 27B 4-bit at 1024; Gemma 4 12B 4-bit at 512 |
| #2215 | split language and vision budgets | Gemma 4 12B language prefill 512; vision admission budget remains 8192 |

## Environment

- Date: 2026-08-23
- Host: Mac Studio `Mac15,14`, Apple M3 Ultra, 256 GB memory
- OS: macOS 26.5.2 (25F84)
- Candidate: main `ba5025f1ac4a804e78417157e2e85530c0d3506f`
- Baseline: tag `v0.12.18`, `291ede85b8688e7f2c60601e70c8403f1221ccd4`
- Python: 3.12.13
- Candidate runtime: MLX/Metal 0.32.1, mlx-lm 0.31.3,
  transformers 5.12.1
- Baseline runtime: MLX/Metal 0.31.2, mlx-lm 0.31.3,
  transformers 5.12.1
- Candidate and baseline used isolated virtual environments. All tested
  weights were already cached; no model was downloaded.

## Method

The service benchmark used `scripts/bench_service_prefill.py`, streaming chat
completions, cache clearing before cold requests, `max_tokens=1`, and the first
visible content delta as TTFT. Prefix-cache persistence was disabled. PFlash
and thinking were disabled to keep both arms comparable.

For the runtime A/B, both versions used an explicit 2048-token prefill step so
the alias policy change could not contaminate the MLX comparison. Each cold
length was repeated three times:

```bash
rapid-mlx serve MODEL --enable-prefix-cache --pflash off --no-thinking \
  --prefill-step-size 2048
python scripts/bench_service_prefill.py \
  --url http://127.0.0.1:18080/v1 --model ALIAS --tokenizer HF_PATH \
  --lengths 1024 4096 --repeat 3 --max-tokens 1 \
  --contention-length 4096 --contention-repeat 1 --output RESULT.json
```

For the profile A/B, both arms used main and differed only in automatic profile
versus explicit 2048. A 2K short request was submitted 100 ms after a 16K or
32K long request, with three repetitions. Qwen3.5 4B used 32K; Qwen3.5 9B and
Gemma 4 12B used 16K.

## MLX 0.32.1 service TTFT

| Model | Context | v0.12.18 p50 | main p50 | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 4B 4-bit | 1K | 555.52 ms | 525.99 ms | 5.3% |
| Qwen3.5 4B 4-bit | 4K | 2038.94 ms | 1962.74 ms | 3.7% |
| Gemma 4 26B-A4B 4-bit | 1K | 586.52 ms | 553.40 ms | 5.6% |
| Gemma 4 26B-A4B 4-bit | 4K | 2205.10 ms | 2115.37 ms | 4.1% |

Peak Metal memory was unchanged in these controlled runtime comparisons:
8.26 GB for Qwen3.5 4B and 20.27 GB for Gemma 4 26B-A4B in both arms.

Qwen3.8 27B could not receive a valid version A/B without downloading another
checkpoint. The only complete local checkpoint is an MTP model with
`qwen3_5_mtp` architecture, which MLX 0.31.2 cannot load; the external bf16
cache contains only a ref and no snapshot. Downloading a 27B checkpoint was
intentionally avoided with only about 18 GiB free. This is a coverage gap, not
a failure.

## Automatic prefill profiles under contention

| Model | Auto step | 2K cold TTFT delta | Contended short TTFT, 2048 | Contended short TTFT, auto | Improvement | Peak Metal, 2048 | Peak Metal, auto | Memory reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B 4-bit | 512 | +6.5% | 7.920 s | 3.468 s | 56.2% | 12.83 GB | 9.96 GB | 22.4% |
| Qwen3.5 9B 4-bit | 512 | +5.1% | 13.510 s | 6.199 s | 54.1% | 12.31 GB | 9.48 GB | 23.0% |
| Gemma 4 12B 4-bit | 512 | +2.3% | 23.655 s | 10.441 s | 55.9% | 19.31 GB | 14.67 GB | 24.0% |

The long request also improved modestly with 512 chunks: 2.0% for Qwen3.5 4B,
3.2% for Qwen3.5 9B, and 7.6% for Gemma 4 12B. The result therefore does not
hide a long-request throughput collapse in this workload.

## Profile resolution and negative controls

The targeted contracts passed:

```text
pytest -q tests/test_recurrent_prefill_auto_default.py \
  tests/test_aliases_contract.py
2920 passed in 4.46s
```

They cover all configured variants: Qwen3.5 4B and 9B 4-bit resolve to 512;
their 6/8-bit variants and Qwen3.5 27B 4-bit resolve to 1024; Gemma 4 12B
4-bit resolves to 512. Bonsai 8B, LFM2.5 2.6B, and Qwen3.5 35B-A3B remain
unset and therefore retain 2048. A real cached Qwen3.5 35B-A3B 4-bit server
boot additionally reported `adaptive_prefill.chunk_size=2048` through
`/v1/status`.

The 6/8-bit and 27B quantizations were not cached and were not downloaded.
Their validation is configuration-level, not a fresh model execution.

## Recommendation

Release the profiles with their present measured allowlist and explicit user
override. Describe the improvement as contention responsiveness and lower peak
memory. Keep 2048 for the negative-control families, and do not generalize the
isolated MLX raw-prefill percentage to end-to-end TTFT. Before publishing a
Qwen3.8-specific number, repeat the same isolated-environment A/B with one
compatible non-MTP checkpoint present in both environments.
