# M4 Pro 48 GB Community Benchmark

This record captures seven public Community Benchmark submissions produced on
an Apple M4 Pro Mac with 48 GB of unified memory. It is intended to answer two
practical questions: what popular model sizes look like on this hardware, and
how another user can reproduce the same fixed workload rather than relying on
an unexplained headline number.

## Environment

| Component | Value |
|---|---|
| Machine | Mac16,11 |
| SoC | Apple M4 Pro |
| CPU / GPU cores | 12 / 16 |
| Unified memory | 48 GiB |
| macOS | 26.5.1 |
| Power / thermal state | AC power, Low Power Mode off, nominal thermals |
| Memory pressure | Normal before and after every run |
| Rapid-MLX | 0.14.1, unmodified PyPI distribution |
| Python | 3.12.13 |
| MLX / mlx-lm | 0.32.2 / 0.31.3 |

No two models were loaded concurrently. Models were measured one at a time,
and no cache transfer or other heavy I/O ran during a measurement.

## Fixed workload

Every row uses registered protocol `rapid-community-speed` v2:

- batch/concurrency: 1;
- deterministic greedy decode;
- prefix cache disabled;
- speculative decoding disabled;
- `pp512-tg128`: 512 prompt tokens, 128 output tokens;
- `pp2048-tg512`: 2,048 prompt tokens, 512 output tokens;
- one warmup and five measured rounds for each case.

The synthetic prompt corpus is fixed by the protocol. It contains no user
content. Decode throughput is `(output_tokens - 1) / decode_duration`; TTFT is
measured separately and is included below.

## Results

| Alias | Short TTFT | Short decode | Long TTFT | Long decode | Peak active memory |
|---|---:|---:|---:|---:|---:|
| `qwen3.5-4b-4bit` | 0.75 s | 83.45 tok/s | 2.95 s | 82.02 tok/s | 4,505 MiB |
| `qwen3.5-9b-4bit` | 1.37 s | 49.76 tok/s | 5.59 s | 48.68 tok/s | 6,821 MiB |
| `gemma-4-12b-4bit` | 2.19 s | 31.87 tok/s | 9.16 s | 30.92 tok/s | 12,191 MiB |
| `gpt-oss-20b-mxfp4-q8` | 0.75 s | 73.24 tok/s | 2.82 s | 70.10 tok/s | 12,733 MiB |
| `gemma-4-26b-4bit` | 0.78 s | 75.59 tok/s | 3.06 s | 71.33 tok/s | 17,337 MiB |
| `qwen3.6-27b-4bit` | 4.87 s | 15.53 tok/s | 19.25 s | 15.30 tok/s | 18,572 MiB |
| `qwen3.8-27b-4bit` | 4.87 s | 15.53 tok/s | 19.31 s | 15.28 tok/s | 18,810 MiB |

These are runtime performance measurements, not model-quality scores. Parameter
count alone does not predict throughput: architecture, active parameter count,
quantization and memory traffic all matter. In particular, the 20B and 26B
mixture-of-experts models activate only part of their total parameters per
token, so their decode rate should not be compared with the dense 27B rows on
parameter count alone.

Normal `rapid-mlx serve` startup can select qualified accelerations from a
model profile. The Community Benchmark protocol intentionally holds
speculative decoding off to preserve cross-machine comparability; its
Qwen3.8 row therefore must not be compared directly with an MTP-enabled
serving result.

## Run a model

Use the short alias from the table. The default server is local-only on port
8000:

```bash
rapid-mlx serve qwen3.5-9b-4bit
```

Choose another port when 8000 is occupied:

```bash
rapid-mlx serve gpt-oss-20b-mxfp4-q8 --port 8100
```

For an endpoint reachable by another machine, bind explicitly and require an
API key:

```bash
RAPID_MLX_API_KEY='replace-with-a-secret' \
  rapid-mlx serve gemma-4-26b-4bit --host 0.0.0.0 --port 8000
```

Then use the OpenAI-compatible API:

```bash
# Run this on the server Mac. From another machine, replace localhost with
# the server Mac's LAN address.
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Do not expose an unauthenticated server outside the Mac. See the
[server guide](../guides/server.md) for API-key, TLS and deployment guidance.

## Reproduce and optionally share

The model-first flow separates planning, measurement and publication:

```bash
# See registered models and whether they fit this machine.
rapid-mlx benchmark catalog

# Inspect the exact cases before loading anything.
rapid-mlx benchmark plan qwen3.5-9b-4bit

# Run locally. The result is archived; nothing is uploaded.
rapid-mlx benchmark run qwen3.5-9b-4bit

# Inspect the complete local record.
rapid-mlx benchmark results
rapid-mlx benchmark inspect <run-id>

# Preview the exact upload and answer y/N. The default is No.
rapid-mlx benchmark share <run-id>
```

To simulate the catalog fit decision for a 48 GB machine without running a
model, use `rapid-mlx benchmark catalog --memory-gib 48`.

## Public records

| Alias | Hugging Face revision | Submission ID |
|---|---|---|
| `qwen3.5-4b-4bit` | `32f3e8ecf65426fc3306969496342d504bfa13f3` | `74ee01b3-3211-4f01-8878-80e636e592e5` |
| `qwen3.5-9b-4bit` | `8b2b98c00a6b4d291155e4890773ca8f769aee53` | `6e0c67d0-9eed-4cd2-aa89-8be6aef2b720` |
| `gemma-4-12b-4bit` | `73bcf09092aa277861d5a191b989b666f7f32e8f` | `f0fd8a0a-c859-4eb1-a869-eaf8e23657d7` |
| `gpt-oss-20b-mxfp4-q8` | `773a7da77e569019bb0fd17a554b263738d669a3` | `f90c0569-cbcb-4964-a385-d03eaffe976e` |
| `gemma-4-26b-4bit` | `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | `e646b8fc-a9b8-432f-9d82-d50858f2eb21` |
| `qwen3.6-27b-4bit` | `c000ac2c2057d94be3fa931000c31723aac53282` | `7401c891-3feb-4b88-9085-eedb7b34d675` |
| `qwen3.8-27b-4bit` | `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456` | `bfa944b3-182d-40ac-8290-b45832c1d174` |

All seven rows were accepted on 2026-09-11 under the anonymous contributor
identity [`jolly-rooted-zebra·edc`](https://rapidmlx.com/leaderboard/contributors/jolly-rooted-zebra-edc).
The public board currently labels atomic Community Benchmark observations as
beta and `unverified_not_ranked`; they are public measurements, not automatic
recommendation-policy evidence.

The privacy-safe public projection is also available as JSON at
`https://rapidmlx.com/api/benchmarks/atomic/public`. It omits the install ID,
payload digest, local paths and other private execution details.
