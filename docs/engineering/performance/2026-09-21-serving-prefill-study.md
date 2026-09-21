# Serving performance study — M3 Ultra prefill characteristics (2026-09-21)

## Question

Why does latency grow with prompt length, and can this Studio host the
product? (Apple-silicon TTFT vs NVIDIA — compute or bandwidth?)

## Method

Direct instrumented runs of the v15c release readout (single forward, think
enabled) at token lengths 293–4409, tokenizer and forward timed separately,
MLX peak memory recorded. (`marvin_spire._state`, eval_label_readout.)

## Result: strictly linear, compute-bound — throughput is a constant

| tokens | tokenize | forward | tok/s | MLX peak mem |
| ---: | ---: | ---: | ---: | ---: |
| 293 | 0.4 ms | 1050 ms | 279 | 8.8 GB |
| 685 | 1.9 ms | 2247 ms | 305 | 8.9 GB |
| 1126 | 2.7 ms | 3645 ms | 309 | 9.1 GB |
| 1784 | 1.9 ms | 5743 ms | 311 | 9.5 GB |
| 2659 | 2.9 ms | 8508 ms | 313 | 10.1 GB |
| 3534 | 10.3 ms | 11331 ms | 312 | 10.6 GB |
| 4409 | 10.4 ms | 14286 ms | 309 | 11.3 GB |

**~310 tok/s constant.** No superlinear term (an earlier 5.9 s @ ~900 tok
reading was an artifact of how the test prompts were constructed). Latency ≈
tokens / 310 + ~150 ms fixed. Tokenization is negligible (<1% until 3.5k).

## Interpretation

- Prefill FLOPs ≈ 2 × 27B × L ≈ 19.4 TFLOP at 360 tokens; at 1.05 s that is
  ~18 TFLOPS effective on M3 Ultra (~29 TFLOPS FP16 GPU) — kernel efficiency
  ~60%, plausible for 2-bit affine quant. **The bottleneck is GPU COMPUTE
  (Apple ~29 TFLOPS vs RTX 4090 ~82 TFLOPS dense FP16, ≈3×), not memory
  bandwidth.** This is exactly the TTFT gap the external review predicted;
  a 4090 should land ~350–400 ms at 360 tokens (within its 300–800 ms band).
- Peak MLX memory ≈ 11 GB at 4.4 k tokens → **multiple worker processes fit
  easily in 512 GB**: horizontal scaling is available on this machine
  without any GPU purchase.
- Tokenization/CPU never matters at our input limits.

## Concurrency + throughput (demo_server, measured)

- Serial decision ≈ 0.94–0.95 RPS at ~250-token prompts, constant across
  concurrency 1/2/4/8 (single main-thread worker → extra requests queue
  linearly: p50 1.06 s → 2.11 s → 3.16 s → 5.27 s).
- Sustained 60 s @ conc=4: 57 decisions, 0 errors, 0.95 RPS.
- Rate limiter verified (429 on burst) — needs a whitelist knob for load
  tests (`MARVIN_RATE_WHITELIST`, now implemented).

## Hosting capacity math (this Studio, single worker)

| pattern | capacity |
| --- | --- |
| playground-style (1 decision/user/min) | ~57 users active simultaneously |
| API-style bursts | ~3.4k decisions/hour |
| N worker processes (8 × 16 GB) | ~7.6 RPS ≈ 27k decisions/hour (untested but memory-trivial) |
| latency SLA option | cap input at 300 tokens → every request <1.1 s |

## Studio-as-product checklist (remaining)

1. `launchd` service + caffeinate (no sleep) + crash auto-restart.
2. Cloudflare Tunnel + custom domain + HTTPS; per-user tokens, log rotation.
3. Multi-worker deployment (port-per-process + L/B) when >50 concurrent users.
4. Move training/experiments to the M4 Pro (isolation from serving hangs).
5. NOTICE/attribution visible on the public page (Apache-2.0 obligation).
