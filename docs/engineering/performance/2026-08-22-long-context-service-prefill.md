# Long-context and service-prefill study (M2 Pro, 2026-08-22)

## Outcome

Bench-verified text-model profiles should select their own prefill chunk in
`rapid-mlx serve`: Qwen3.5 4B/9B 4-bit use 512; Qwen3.5 4B/9B 6/8-bit and
27B 4-bit use 1,024; Gemma 4 12B 4-bit uses 512 with an independent 8,192-token
vision admission budget. Against the previous universal
2,048-token default, the original Qwen3.5 4B 4-bit result:

- reduced a short request's TTFT under a concurrent long prefill by 51.1%;
- reduced that short request's end-to-end latency by 64.4%;
- kept 16K--96K single-request prefill throughput within 0.7%; and
- reduced peak MLX memory by 27--29% at 16K--96K.

The change is profile-scoped, not architecture-scoped. Repeated follow-up
measurements found that the same 512-token setting regresses Bonsai 8B,
LFM2.5 2.6B, and Qwen3.5 35B-A3B by 6--16%. An explicit
`--prefill-step-size` always wins; unverified profiles retain 2,048.

## Goal and constraints

- Owner: Vector (performance)
- Host: M2 Pro Mac mini, 32 GB unified memory
- OS: macOS 26.5.2
- Base: Rapid-MLX `fc7f1635`; candidate branch
  `raullenchai/vector-prefill-service-perf`
- Runtime: MLX 0.32.1, mlx-lm 0.31.3, Rapid-MLX 0.12.18 candidate wheel
- Primary model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Verification: controlled direct prefill, OpenAI-compatible streaming HTTP,
  exact/partial prefix reuse, mixed long/short concurrency, unit tests, and a
  Gemma 4 12B dense/sliding spot check

Chrome and competing model processes were not running. Raw JSON remains in
`~/mac-model-matrix/results/` on the mini; it is intentionally not committed.

## Reproduction

Direct prefill:

```bash
python bench_prefill.py \
  --engine rapid \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --lengths 16384 32768 65536 98304 \
  --repeat 1 \
  --prefill-step-size 512 \
  --output results/qwen35-4b-prefill-step512-scout.json
```

Service workload:

```bash
rapid-mlx serve qwen3.5-4b-4bit \
  --host 127.0.0.1 --port 18080 \
  --enable-prefix-cache --pflash off --no-thinking

python bench_service_prefill.py \
  --url http://127.0.0.1:18080/v1 \
  --model qwen3.5-4b-4bit \
  --tokenizer mlx-community/Qwen3.5-4B-MLX-4bit \
  --label rapid-auto512-branch \
  --lengths 2048 --repeat 1 --max-tokens 1 \
  --contention-length 32768 --contention-repeat 1 \
  --contention-delay-ms 100 \
  --output results/qwen35-4b-service-auto512-branch.json
```

`scripts/bench_service_prefill.py` clears the service cache before cold trials,
times the first visible content/reasoning/tool delta rather than the initial SSE
role frame, records server-reported prompt/cached tokens, tests exact and
partial prefix reuse, and submits a short request behind an active long prefill.
It polls `/v1/status` until the long request is server-confirmed as running
before it submits the short request, rejects streams with no visible delta, and
rejects streams that omit prompt/completion/cached-token usage rather than
silently treating missing counters as zero. It stamps the result with a
methodology hash. Use at least three repeats for publication claims; the
one-repeat runs here are scoped engineering A/Bs.

The recorded A/B artifacts predate the enforced status poll; their server logs
show the long request scheduled before the short request. The committed harness
turns that observed ordering into a required precondition for future runs, so
its methodology hash intentionally differs from those artifacts.

## Long-context scaling

With the old 2,048 chunk, prompt throughput scaled smoothly rather than falling
off a discrete cliff:

| Prompt | Prompt tok/s | Peak MLX GB |
| ---: | ---: | ---: |
| 2K | 400.47 | 4.21 |
| 8K | 388.26 | 4.82 |
| 16K | 369.72 | 5.62 |
| 32K | 336.96 | 7.23 |
| 64K | 285.90 | 10.45 |
| 96K | 245.04 | 13.83 |

The 96K rate is 38.8% below 2K, consistent with context-length cost. There was
no swap and no abrupt regression boundary.

## 512 versus 2,048 on recurrent prefill

| Prompt | 2,048 tok/s | 512 tok/s | Throughput delta | 2,048 GB | 512 GB | Memory delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 369.72 | 370.02 | +0.1% | 5.62 | 4.10 | -27.1% |
| 32K | 336.96 | 336.29 | -0.2% | 7.23 | 5.26 | -27.3% |
| 64K | 285.90 | 283.96 | -0.7% | 10.45 | 7.49 | -28.3% |
| 96K | 245.04 | 245.65 | +0.2% | 13.83 | 9.81 | -29.1% |

For the controlled service A/B, both variants used identical prompts. The long
request had 45,080 server-reported prompt tokens and the short request 2,840.

| Metric | 2,048 | 512 | Delta |
| --- | ---: | ---: | ---: |
| Short standalone TTFT | 7.11 s | 7.08 s | -0.4% |
| Contended short TTFT | 42.44 s | 20.77 s | **-51.1%** |
| Contended short total | 84.76 s | 30.17 s | **-64.4%** |
| Contended long TTFT | 157.17 s | 150.91 s | -4.0% |

This isolates scheduler monopolization, not a faster math kernel: the smaller
chunk gives waiting work more scheduling opportunities while leaving long
request throughput intact.

The candidate wheel, launched without a prefill override, logged the recurrent
auto-selection and reported `adaptive_prefill.chunk_size=512`. With the fixed
exact-token harness, cold 2K TTFT was 5.20 s, exact prefix reuse was 0.173 s
(2,035 cached tokens), and a 2K request behind a 32K prefill reached its first
visible token in 15.42 s. This is an end-to-end confirmation, not a cross-run
comparison to the earlier differently-sized prompts.

## Prefix-cache finding

Prefix reuse is functioning for the recurrent model. On the 2,048-chunk
baseline's 22,552-token prompt, an exact hit reused 22,537 tokens and reduced
TTFT from 62.48 s to 0.408 s; a partial extension reused the same prefix and
reached its first token in 0.416 s. The service auto-defaulted bounded recurrent
snapshots to eight entries.

The tradeoff is memory: recurrent snapshots can be large. The exact-token 32K
candidate run reached 3.73 GB of prefix-cache memory before pressure eviction.
Future concurrency work should measure cache-entry admission and eviction under
agentic multi-prefix workloads rather than increasing the entry count blindly.

## Dense/sliding spot check

Gemma 4 12B (`mlx-community/gemma-4-12B-it-4bit`) must use the mlx-vlm/Gemma
loader; mlx-lm 0.31.3 rejects its `gemma4_unified` model type. An initial
single-repeat scout through mlx-vlm 0.6.15 found that 512 was promising:

| Prompt | 2,048 tok/s | 512 tok/s | Delta | Peak-memory delta |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 131.79 | 136.48 | +3.6% | -7.8% |
| 16K | 125.38 | 130.12 | +3.8% | -13.3% |

The later repeat-three direct-prefill matrix reproduced this result. The runtime
now keeps the language-model prefill chunk separate from the per-image admission
budget: Gemma can use the measured 512 chunk while vision-bearing requests keep
the safe 8,192-token budget. This matters especially to Desktop users, because
the GUI launches the alias without performance flags and receives both defaults
automatically.

Operator compatibility is source-aware. An explicit
`--vision-prefill-token-budget` always wins; an explicit
`--prefill-step-size` without that new flag preserves the historical shared
limit for memory-constrained deployments. Only the zero-flag/profile path
separates Gemma's 512 computation chunk from the 8,192 vision budget.

A zero-flag service spot check used the same Desktop launch shape on the M2 Pro:
`rapid-mlx serve gemma-4-12b-4bit`. Startup selected the profile's 512 chunk,
and a 64×64 red PNG sent through `/v1/chat/completions` completed normally with
280 prompt tokens and the non-empty response `The image is dark red.` The
admission regression test separately pins a representative 2,292-token image
prompt as accepted by the 8,192 budget while proving it would have failed if
the budget still followed the 512 chunk.

## Recurrent cross-model regression matrix

A repeat-three follow-up tested whether the Qwen3.5 4B result generalized to
other recurrent/linear-attention models. Each cell is the median of three exact
token-count prefills with one generated token. Qwen3.5 9B ran on the M2 Pro
32 GB mini; the other rows ran on an M3 Ultra 256 GB Studio. Both used macOS,
MLX 0.32.1, and mlx-lm 0.31.3. No competing inference process was active; an
idle `mlx_audio.server` remained on the Studio.

| Model | Prompt | 2,048 tok/s | 512 tok/s | Throughput delta | 2,048 GB | 512 GB | Memory delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B 4-bit | 4K | 217.63 | 218.22 | +0.3% | 6.97 | 6.08 | -12.8% |
| Qwen3.5 9B 4-bit | 16K | 209.32 | 209.51 | +0.1% | 8.18 | 6.75 | -17.5% |
| Ternary-Bonsai 8B 2-bit | 4K | 1,153.24 | 1,087.78 | -5.7% | 3.66 | 3.54 | -3.2% |
| Ternary-Bonsai 8B 2-bit | 16K | 929.66 | 853.79 | -8.2% | 5.32 | 5.26 | -1.2% |
| LFM2.5 2.6B 4-bit | 4K | 3,444.45 | 3,168.61 | -8.0% | 2.81 | 2.44 | -13.1% |
| LFM2.5 2.6B 4-bit | 16K | 3,194.98 | 2,912.64 | -8.8% | 2.97 | 2.57 | -13.4% |
| Qwen3.5 35B-A3B 4-bit | 4K | 2,427.65 | 2,037.85 | -16.1% | 21.51 | 20.30 | -5.6% |
| Qwen3.5 35B-A3B 4-bit | 16K | 2,159.47 | 1,806.38 | -16.4% | 22.55 | 20.73 | -8.1% |

The 9B result reproduces the 4B tradeoff: effectively unchanged throughput
with meaningfully lower peak memory. The other architectures save varying
amounts of memory but exceed the predeclared 3% throughput-regression limit.
Therefore recurrent config detection is not a safe default selector. The
runtime uses an explicit `recommended_prefill_step_size` profile field only on
the measured aliases only. Bonsai, LFM, MoE/hybrid, bare local paths, and future
aliases keep the general 2,048 default until measured.

## 512 / 1,024 / 2,048 profile expansion

The follow-up added 1,024 as a candidate and repeated every cell three times.
Qwen rows ran on the M3 Ultra 256 GB Studio with MLX 0.32.1 and mlx-lm 0.31.3;
Gemma ran on the M2 Pro 32 GB mini with MLX 0.32.1 and mlx-vlm 0.6.15. The
decision rule remained: reject any candidate with a median throughput regression
greater than 3% at either 4K or 16K, then prefer the smallest remaining chunk.

| Model | Direct winner | Deployed default | 4K throughput vs 2,048 | 16K throughput vs 2,048 | 4K memory | 16K memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B 6-bit | 1,024 | 1,024 | -1.2% | -2.3% | -10.7% | -7.8% |
| Qwen3.5 4B 8-bit | 1,024 | 1,024 | -1.9% | -2.5% | -9.0% | -6.7% |
| Qwen3.5 9B 6-bit | 1,024 | 1,024 | -0.8% | -1.2% | -7.0% | -10.1% |
| Qwen3.5 9B 8-bit | 1,024 | 1,024 | -1.2% | -1.6% | -5.1% | -7.8% |
| Qwen3.5 27B 4-bit | 1,024 | 1,024 | -0.6% | -0.5% | -6.6% | -8.9% |
| Gemma 4 12B 4-bit | 512 | 512 | +3.3% | +3.8% | -7.8% | -13.3% |

For the four 6/8-bit Qwen aliases, 512 regressed throughput by 3.9--7.5%, so
1,024 is not merely a midpoint: it is the smallest candidate that stays within
the regression budget. Qwen3.5 27B at 512 was acceptable at 4K (-2.5%) but
missed at 16K (-3.3%), so it also uses 1,024. Gemma's repeat-three result
confirmed the earlier scout. It is deployed only after separating the 512
language-model chunk from the 8,192-token vision admission budget.

### Desktop/GUI consumption audit

Rapid Desktop starts the bundled sidecar as `rapid-mlx serve <alias>` and does
not pass `--prefill-step-size` by default. Therefore the CLI resolves these
profile recommendations for GUI launches automatically. `ModelPerfConfig` is
sparse: an untouched GUI passes no performance override, while a future or
explicit user flag remains higher priority than the profile recommendation.

The broader per-model profiling work is only partially visible in the GUI:

- backend profile decisions such as prefill, PFlash/TurboQuant tiers, parsers,
  and hybrid/MoE safety gates apply automatically to GUI-started servers;
- the GUI directly consumes recommended sampling, parser/context metadata,
  speculative presets, and the shared RAM-tier recommendation catalog; but
- benchmark evidence and `recommended_prefill_step_size` are not currently in
  the desktop's `ServerModelProfile` DTO, so the optimization runs but the UI
  cannot yet explain the selected chunk or its evidence.

This is a visibility/SSOT presentation gap, not a runtime-application gap. A
future desktop change should surface the server's effective prefill value and
source (`user`, `profile`, or global default) instead of copying the resolver
or hard-coding alias values in Swift.

## Artifacts

SHA-256 checksums:

- `qwen35-4b-prefill-mlx032-2k-96k-scout.json`:
  `112fdbbe603b6dd87324395b60b4292a53f3eff6df0a8bb391e9a1d43f8370dd`
- `qwen35-4b-prefill-step512-scout.json`:
  `0182a37e30234c4075570ba64c34fff6246f1efca6ea2d25dc418eea8507b0de`
- `qwen35-4b-service-prefill-r1.json`:
  `997f674d95e53971d94a7932721907b1aede487e45f1feaca499c3561558bbd9`
- `qwen35-4b-service-step512-r1.json`:
  `8fff0413575b659a33fc7ad4a3e64d2b08688b7cd0bff9692755254450cbfbae`
- `qwen35-4b-service-auto512-branch.json`:
  `da73c19461ce719d84eb13c4d9338b808d5e6ace549b97897a576f0b97964107`
- Gemma 2,048 / 512 scouts:
  `9fdf433fe29ffd6d6fc6691f1e7081aac864c0fccf2e6c2746a4d45df168a323`,
  `5a39dc7ed674d6c056e2962ddedc75f46f1f16bc331b4c2f7fe973b642b7be51`
- Qwen3.5 9B 2,048 / 512 repeat-three:
  `f25d19a46a8a9aa5260cb15f0e2d3acdaeed21412ce4fa2df016540310cf642b`,
  `7970ee9617db0fb0f211b5c310787987819e4eaa3f1ba22ed26def4646f939d1`
- Bonsai 8B 2,048 / 512 repeat-three:
  `300f379e3902b8d1ec1c4f627f687104ed3ef4e44469e0ac1ee99f9d7b7a5533`,
  `3588e7cb0319eff3efc3ab0a827d323cd210c251875c13c2eaee2eb184b56c80`
- LFM2.5 2.6B 2,048 / 512 repeat-three:
  `5d5143f442bd098b3661acf6d19dd1f13cbed6d3f79a883ebff0795fae08305b`,
  `763a7d3e4c196bafa88eaec706c72762b69bca81590cce3e53c260f7ca4202f5`
- Qwen3.5 35B-A3B 2,048 / 512 repeat-three:
  `d6b2459883bf0ce3a1a52d2cf8624f23315b7c9be5c47218286ba6977a4293a2`,
  `35846b1024bc3ab2c5488c50ed96638b33a32c409fe252f1f44bebc0682a5ce0`
- Profile-expansion artifacts (each list is 2,048 / 1,024 / 512):
  - Qwen3.5 4B 6-bit: `ea6d2144f0bd72e600230427536d4cea231ef84eca10d60fc005d357c9e8bf5d`,
    `9b187251b720569b9f4baa26eb8121082b0cfca3f4465e9635a1eaa1ee5e0323`,
    `aa257c27c642f659171ce30f17355ea1fe66c553b9bcbda1424c5333d83c03ca`
  - Qwen3.5 4B 8-bit: `a1252c300b7c0eef027215263ef4a4175c9237e30f87235a0896f2dc93184714`,
    `a7e8e713e12d655599b4a069389518c87d875dbd6df51db67795fd13852bce55`,
    `ce913140ac990420a1a9aa0bca3de4bd642ae50d9b9df3a92dc4069139a23d5e`
  - Qwen3.5 9B 6-bit: `fbd88aacd479979dada08181caeccb56ec4aa6e449b1d4d53b427b905db49b39`,
    `47f50a708c3dcdd2397e94f3042d9e350705d18754aed102e4fd41b872206090`,
    `b2883c95a149436af905177baae346e35fef0ac262c3ff4e320c9093ad24c231`
  - Qwen3.5 9B 8-bit: `6e139d97b6330143cb2d33f67f51c7b7f93a76b099476359a53dd897531d799a`,
    `25f32641ee0f934b70dd56ba1813aebc67b87a194ca15ef99eb8a3e5c706d494`,
    `88d510221f03163ad02c449668d1cc00179e6d8fd635e8eb247fc8a2b8ed90f9`
  - Qwen3.5 27B 4-bit: `9c8b934669dd96651eeb2b0af3f9a0190b3bc56c811d47c90c91565162f3fd5d`,
    `6cb4a8f248c686ac368941de1b09277d164af6a7df6ba55dce0cd15f8f208ea3`,
    `7bcdc194432e32262a39e9920eb81592d0967d3eab3a3ae39e043918efbd1c37`
  - Gemma 4 12B 4-bit: `0746e72887f8c98ed31aa9ae2ae67cfd1ab9426f9765f39fcd9d5566f4c42c4b`,
    `d785ba5338924836125db5524f4619cd8fc3cd96b187864989d64b3c59694e29`,
    `a587b3b7833f5cc7c6f2ba82e16952f8543c665354c84872f61f7aa84dfa7128`

## Recommendation and next work

Land the profile-scoped 512 auto-default with the benchmark harness. Then:

1. run the same HTTP workload against a calibrated oMLX deployment (none was
   installed on this mini during this run), including equivalent cache policy;
2. repeat the Gemma 4 12B image-service A/B before generalizing to other MLLMs;
3. profile cache admission/eviction for multiple 16K--64K agent prefixes; and
4. add a repeat-three service tier to scheduled Mac performance CI.
