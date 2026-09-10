# Qwen3.8 27B Abliterated qualification

Date: 2026-09-09

This note records the evidence used to add
`qwen3.8-27b-abliterated-4bit` as an experimental Server and Desktop catalog
entry. It does not support making the model a default or enabling speculative
decoding for it.

## Artifacts and environment

- Machine: Apple M3 Ultra, 256 GB unified memory
- macOS: 26.5.2 (25F84)
- Python: 3.11
- MLX: 0.32.2
- mlx-lm: 0.31.3
- Text-lane mlx-vlm installation: 0.6.16
- Vision qualification mlx-vlm installation: 0.6.17, in a worktree-local
  virtual environment with the system packages inherited
- Source: `windowsxp811203/Qwen3.8-27B-Abliterated-MLX-MTP`
- Immutable source revision: `5b6802378702c89de48c990b29f3e55a3c84a2e3`
- Selected repository subfolder: `oQ4e/`
- Selected 18-file subfolder size: 16,998,765,834 bytes (15.83 GiB)
- Size-manifest weight/config/tokenizer footprint: 16,998,733,375 bytes;
  the 32,459-byte difference is the subfolder's license, README, and chat
  template, which the manifest deliberately excludes
- Rapid-MLX base revision under test:
  `d32277189476be85bd1867b71ef99eab4ff40a29`

The standard Hugging Face cache was checked before download. `rapid-mlx pull`
fetched 18 files under `oQ4e/` and did not fetch the repository's sibling
quantizations or `drafter/` directory.

## Rapid-MLX product-path results

All requests used `temperature: 0`. These are smoke and operational
qualification measurements from one machine, not a general performance
benchmark.

| Journey | Result | Observed decode |
|---|---|---:|
| Text factual/arithmetic | Exact coherent answer: Tokyo and `17 * 23 = 391` | 22 tokens in 1.05 s, 20.9 tok/s |
| Required Qwen tool call | `get_weather` with `{"city":"Tokyo"}` and `finish_reason: tool_calls` | 27 tokens in 2.24 s, 12.1 tok/s |
| Image description | Correctly described the checked-in cartoon cheetah asset | 25 tokens in 1.60 s, 15.6 tok/s |
| Four simultaneous requests | Exact `PARALLEL_1` through `PARALLEL_4`; no errors | 0.827 s wall time; each 5 tokens in 0.79 s |
| Cancelled streaming request | Engine aborted after the client disconnected; server stayed healthy | Recovery returned exact `RECOVERED` |

The loader emitted the expected `qwen3_5_norm_shift` compatibility patch for
161 standard-form gains. This is important for this conversion: the publisher
reports that unpatched mlx-lm 0.31.3 produces invalid output, while the
Rapid-MLX text path produced the coherent results above.

## Reproduction

Use the normal cache location; do not override Hugging Face cache variables or
pass a custom cache directory.

```bash
rapid-mlx pull qwen3.8-27b-abliterated-4bit

# Text path used for the text, tool, cancellation, and concurrency checks.
rapid-mlx serve qwen3.8-27b-abliterated-4bit \
  --host 127.0.0.1 --port 8498 --no-thinking --no-mllm

# Vision path used with the release-pinned mlx-vlm 0.6.17 environment.
rapid-mlx serve qwen3.8-27b-abliterated-4bit \
  --host 127.0.0.1 --port 8499 --no-thinking --mllm
```

The text request was sent through `/v1/chat/completions`; the tool check used a
required OpenAI-compatible function named `get_weather`; and the image check
sent `apps/rapid-mac/Sources/Rapid/Resources/cheetah-sm.png` as an image data
URL through the same endpoint. The cancellation probe terminated a streaming
client after one second and immediately issued a normal recovery request.

## Publisher-reported checkpoint evidence

The model card reports the following measurements for `oQ4e`. These are the
checkpoint author's results and were not independently reproduced here:

| Evaluation | oQ4e | Comparison |
|---|---:|---:|
| Seeded 400-question MMLU sample | 81.75% | 82.50% for the bf16 conversion |
| AdvBench refusals | 0/80 | refusal-oriented evaluation |
| HarmBench-safety refusals | 0/119 | refusal-oriented evaluation |
| oMLX MTP decode | about 34 tok/s | about 21.8 tok/s without MTP |

Abliteration removes refusal behavior; it is not evidence that answers are
more correct or safe. The publisher's oMLX MTP throughput is also not a
Rapid-MLX claim.

## Speculative-decoding gate

This repository contains a matching, modified MTP head under `drafter/`, but
the Rapid-MLX alias schema currently identifies MTP artifacts by repository and
does not identify a drafter subfolder within the target repository. The model
therefore ships with `supports_spec_decode: false` and
`mtp_default_enabled: false`.

Do not substitute the curated base-Qwen DFlash2 or DSpark drafter. Abliteration
changed the target trunk, and the publisher changed the matching MTP head as
well. A follow-up may add pinned drafter-subfolder resolution, but it must
download only that subtree and qualify acceptance, output correctness,
streaming cancellation, concurrency, and end-to-end speed before enabling it.
