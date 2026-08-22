# Mac mini model matrix: Qwen3.5 4B, Gemma 4 26B, and Qwen3.8 27B

Measured 2026-08-21 on a Mac mini (Apple M2 Pro, 10 CPU cores, 32 GB unified
memory) running macOS 26.5.2. Chrome was closed with human authorization. The
host was rebooted before the two large-model comparisons; swap was zero and no
process used more than 20% CPU at the pre-run idle gate.

These are single-stream, short-prompt decode results. They do not establish
HTTP, concurrency, long-context prefill, or multimodal performance.

## Comparable autoregressive decode

Each Rapid-MLX and mlx-vlm cell is the median of 16 samples: eight prompts
repeated twice in one model load. Each oMLX cell combines two independent
eight-prompt model loads by averaging their reported medians. All engines used
the same checkpoint for a model, deterministic sampling, and a maximum of 128
generated tokens.

| Model/checkpoint | Rapid-MLX | mlx-vlm | oMLX | Rapid vs mlx-vlm | Rapid vs oMLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B 4-bit | **64.00** | 62.42 | 59.74 | +2.5% | +7.1% |
| Gemma4-26B-A4B 4-bit | **51.44** | 50.56 | 48.39 | +1.8% | +6.3% |
| Qwen3.8-27B 4-bit | 11.60 | **11.66** | 11.57 | -0.5% | +0.2% |

Units are generated tokens per second. Qwen3.8's differences are noise-level
parity; there is no evidence of an autoregressive decode deficit worth a risky
product change. Rounded claim-ready Rapid results are **64.0**, **51.4**, and
**11.6 tok/s**, scoped to this host and workload.

## Memory and cached model load

| Model | Rapid peak MLX memory | mlx-vlm peak MLX memory | Reduction | Rapid load | mlx-vlm load |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | **2.65 GB** | 3.73 GB | 28.9% | 2.90 s | **2.36 s** |
| Gemma4-26B | **14.40 GB** | 15.55 GB | 7.3% | **2.61 s** | 6.51 s |
| Qwen3.8-27B | **15.53 GB** | 18.73 GB | 17.1% | **5.95 s** | 7.63 s |

Gemma4-26B completed without swap under Rapid-MLX and mlx-vlm. oMLX's second
Gemma run left only 0.25 MB used in a 1 GB swap allocation, which is
operationally negligible. The MLX memory counter does not include every mapped
file or host allocation, so these numbers are comparative rather than total
system resident memory.

## Qwen3.8 MTP result

The combined Qwen3.8 checkpoint carries an MTP draft. Rapid-MLX's adaptive
experiment produced:

| Mode | Median decode | Pooled decode | Acceptance | Token-exact prompts |
| --- | ---: | ---: | ---: | ---: |
| AR | **11.64** | **11.65** | — | 8/8 reference |
| adaptive MTP k=2 | 9.43 | 9.39 | 80.6% | 8/8 |
| adaptive MTP k=3 diagnostic | 9.89 | 9.77 | 88.1% | 8/8 |

AR is the best tested configuration on M2 Pro. The k=3 row is diagnostic only:
the benchmark reused a controller initialized with `max_k=2`, and the runtime
logged that it ignored `max_k=3`. This harness isolation issue must be fixed
before using k=3 as a standalone result, but it cannot overturn the conclusion
that the measured MTP paths were slower than AR.

## Prefill observation and next experiment

The direct harness reported lower Rapid prompt rates than mlx-vlm:

| Model | Rapid prompt tok/s | mlx-vlm prompt tok/s | Rapid delta |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | 116.0 | 144.0 | -19.4% |
| Gemma4-26B | 82.8 | 94.4 | -12.3% |
| Qwen3.8-27B | 19.6 | 24.9 | -21.1% |

Do not publish those rows as long-context prefill claims: the prompts contain
only 14--33 tokens, so fixed setup, cache construction, and synchronization
dominate the rate. The controlled follow-up below establishes the actual cause.

### Prefill scope-down: MLX 0.31.2 is the bottleneck

The follow-up used the Qwen3.5-4B checkpoint, identical decoded text, exactly
128/1,024/4,096 tokens as independently reported by both tokenizers, one output
token, three repeats, and a fresh prompt cache per sample.

| Context | Rapid stack: mlx-lm 0.31.3 + MLX 0.31.2 | mlx-vlm + MLX 0.32.1 | mlx-lm 0.31.3 + MLX 0.32.1 |
| --- | ---: | ---: | ---: |
| 128 | 290.9 | 331.1 | **343.6** |
| 1,024 | 326.9 | 342.2 | **399.9** |
| 4,096 | 324.6 | 395.1 | **397.1** |

Replacing mlx-lm 0.31.3 with upstream main (`dfb5da1`, package version 0.32.0)
on the same MLX 0.32.1 runtime measured 344.0/400.9/397.4 tok/s at
128/1K/4K. That is effectively tied with the final column above: Qwen3.5-4B's
material prefill gain comes from MLX/Metal 0.32.1, not an mlx-lm model change.

Units are prompt tokens per second. The decisive cross-over keeps the same
`mlx-lm` model implementation and changes only `mlx`/`mlx-metal` from 0.31.2 to
0.32.1. It improves 1K and 4K prefill by 22.3%, completely closes the mlx-vlm
gap, and slightly exceeds mlx-vlm at 4K. Therefore:

1. Rapid-MLX's Qwen generation/cache path is not the material bottleneck.
2. The observed prefill deficit is caused by the production dependency ceiling
   `mlx>=0.31.2,<0.32`, not by HTTP/server overhead or the mlx-lm Qwen model.
3. Raising that ceiling is compatibility-sensitive. It is deliberately blocked
   by `scripts/check_mlx_bound_move.py` because an earlier upstream mlx-lm
   heuristic produced incoherent Qwen3.6 output. A full-family output-coherence
   sweep and Atlas approval are required before shipping the faster runtime.

An earlier short-prompt M3 Ultra A/B failed to expose the version effect; those
14--33-token measurements were dominated by fixed overhead and are superseded
by the controlled exact-length result above.

### mlx-lm release versus upstream main

`mlx-lm` 0.31.3 is the latest PyPI release, so the production stack was already
using the newest released mlx-lm. A separate Studio coherence probe compared
that release with official upstream main at commit
`dfb5da1d61f87679b0bc060c0794551e8db0d243`, whose package version is the
unreleased 0.32.0. Both used MLX/Metal 0.32.1 and the same cached
`mlx-community/Qwen3.6-35B-A3B-8bit` checkpoint.

| Stack | Decode median | Prompt median | Cached load | Exact versus 0.31.3 |
| --- | ---: | ---: | ---: | ---: |
| mlx-lm 0.31.3 + MLX 0.32.1 | 88.35 | 277.43 | 7.53 s | reference |
| mlx-lm main (`dfb5da1`) + MLX 0.32.1 | **88.73** | **300.11** | **2.79 s** | 8/8 prompts |

Each prompt generated 64 deterministic tokens. Main was token-identical on all
eight prompts, essentially tied on decode, 8.2% faster on prompt processing,
and materially faster to load. This is positive evidence for testing the next
mlx-lm release as the upgrade target, not proof that every Qwen3.6 checkpoint is
fixed: upstream issue #1197 targets VLM-MTP checkpoint weight layouts and
remains open.

The exact issue #1197 layout was therefore re-run separately on the M3 Ultra
256 GB Studio at Rapid-MLX commit `6acb5306`, using mlx 0.32.1 and released
mlx-lm 0.31.3. The checkpoint was
`mlx-community/Qwen3.6-35B-A3B-8bit` snapshot
`e06a74e6236a60c8367e1a3214e83d8b61b637b0`: its config contains both a
`vision_config` and `mtp_num_hidden_layers=1`, its weight index contains all
333 `vision_tower.*` tensors, and the snapshot includes
`model-mtp.safetensors`. Rapid was forced through the text lane with
`--no-mllm --disable-prefix-cache --no-thinking`; the blocking coherence gate
passed 6/6 (Tokyo, 391, blue, seven, banana, and Paris with no think leak).
This closes the VLM-plus-MTP checkpoint-layout case for the dependency move;
it does not claim that upstream issue #1197 itself is resolved.

```bash
python -m vllm_mlx.cli serve "$SNAPSHOT" --port 8403 \
  --no-mllm --disable-prefix-cache --no-thinking
RAPID_MLX_BASE_URL=http://127.0.0.1:8403/v1 \
  python evals/coherence_gate.py
```

Here `SNAPSHOT` is the local directory for the pinned Hugging Face snapshot
named above.

### Long-context follow-up for issue #2165

Qwen3.5-4B was extended to 8K and 16K exact-token prompts with the default
2,048-token prefill step:

| Context | Production: mlx-lm 0.31.3 + MLX 0.31.2 | Candidate: mlx-lm main + MLX 0.32.1 | Gain |
| --- | ---: | ---: | ---: |
| 8,192 | 318.6 | **388.1** | +21.8% |
| 16,384 | 306.0 | **369.5** | +20.8% |

The runtime upgrade raises the long-context baseline without increasing peak
MLX memory, but does not remove scaling loss: throughput falls 4.0% from 8K to
16K on production and 4.8% on the candidate. This addresses one contributor to
issue #2165, not its full 21K--97K scaling question.

The issue's proposed larger-chunk experiment was also tested on the candidate
stack at 16K:

| `prefill_step_size` | Prompt tok/s | Peak MLX memory |
| ---: | ---: | ---: |
| 2,048 | **369.4** | **5.62 GB** |
| 4,096 | 364.6 | 7.28 GB |
| 8,192 | 355.9 | 11.02 GB |
| 16,384 | 336.2 | 18.63 GB |

For this hybrid/GDN model, larger chunks are both slower and substantially more
memory-hungry. The default 2K step is the best tested choice. A generic policy
of using the largest chunk that fits would regress throughput by up to 9.0% and
raise peak MLX memory by 3.3x. Issue #2165's adaptive chunk policy therefore
must be architecture-specific; dense full-attention models still require their
own sweep and may behave differently.

### Crossover for Gemma 4 26B-A4B and Qwen3.8-27B

The controlled MLX-version crossover was extended to the two other large
hybrid/GDN models measured in this campaign. The method is identical to the
Qwen3.5-4B crossover: the same `mlx_lm` 0.31.3 implementation and the same
checkpoint, only `mlx`/`mlx-metal` changes from 0.31.2 to 0.32.1. Both
environments pin mlx-lm 0.31.3, transformers 5.15.1, numpy 2.4.6, and
tokenizers 0.22.2; the evaluated stacks differ only in the MLX runtime. Each
cell is the median of three samples with a fresh prompt cache and the default
2,048-token prefill step. Exact 1,024- and 4,096-token prompts.

| Model | Context | MLX 0.31.2 | MLX 0.32.1 | Peak memory (both) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4-26B-A4B | 1,024 | 294.4 | 394.9 | 15.05 GB | +34.1% |
| Gemma4-26B-A4B | 4,096 | 288.3 | 379.3 | 15.96 GB | +31.6% |
| Qwen3.8-27B | 1,024 | 49.9 | 61.8 | 17.02 GB | +23.8% |
| Qwen3.8-27B | 4,096 | 49.8 | 61.6 | 18.55 GB | +23.7% |

Units are prompt tokens per second. The runtime upgrade materially improves
prefill on both models with essentially unchanged peak MLX memory. The Gemma row
is the largest prefill gain measured so far in this campaign (+31--34%);
Qwen3.8-27B's +24% gain is in line with the Qwen3.5-4B crossover (+22%). These
results confirm that the MLX 0.32.1 prefill benefit is not specific to small
Qwen checkpoints and broadens the controlled evidence across the lightweight
activation (A4B) and MTP-carrying Qwen3.8 families.

Qwen3.8-27B here is the MTP-carrying `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`
checkpoint; `bench_prefill.py` generates a single token, so this measures the
prompt/cache path under the plain autoregressive mlx-lm implementation, not MTP
draft acceptance. Its prefill peak memory of 18.55 GB at 4K stays within the
32 GB Mac mini budget without swap.

## Shipping recommendation

Ship the runtime improvement with `mlx>=0.32.1,<0.33` while retaining the
released `mlx-lm>=0.31.3,<0.32` line. The controlled crossover attributes the
Qwen3.5 prefill gain to MLX itself: moving only MLX/Metal from 0.31.2 to 0.32.1
improved 1K/4K prefill by 22.3%, whereas moving from released mlx-lm 0.31.3 to
official main on the same MLX 0.32.1 runtime changed Qwen3.5 throughput by less
than 0.3%. This avoids pinning an unreleased mlx-lm commit while taking the
measured runtime gain.

The image extra must move at the same time from mflux 0.18.x to
`mflux>=0.19.0,<0.20`: mflux 0.18.1 declares `mlx<0.32`, while 0.19.0 declares
`mlx>=0.32,<0.33`. Fresh pip resolver probes selected MLX 0.32.1 and mlx-lm
0.31.3 for core, and additionally mflux 0.19.0 for `[image]`. The mflux 0.19.0
candidate passed Rapid-MLX's 88 image-lane tests and 67 image alias/dependency
contract tests.

The blocking coherence evidence for the dependency PR is separate from the
throughput harness. Under mlx-lm 0.31.3 + MLX 0.32.1, all six ordinary release
families passed all 6/6 golden cases: Qwen3.5 4B, Qwen3.5 35B-A3B, Qwen3.6 27B,
Gemma4 12B, DeepSeek-R1-Distill 32B, and GPT-OSS 20B. The sweep now disables
prefix-cache reuse because persisted KV tensors are not keyed by MLX runtime;
before that isolation fix, a stale DeepSeek cache produced a false 4/6 failure,
while the cold run and full cold-cache rerun both passed 6/6.

The toolchain-only Hy3 representative was subsequently staged on a 1.8 TiB
external SSD and run on the M3 Ultra 256 GB Studio with the same candidate and
`--disable-prefix-cache`. The captured run used Rapid-MLX commit `51923343`,
mlx 0.32.1, released mlx-lm 0.31.3, macOS 26.5.2, and Hy3 snapshot
`8e4d56f18efd912b8c7581a8ccfa8b2a79ba3469`. `hy3-preview-4bit` passed all
6/6 blocking cases; swap usage after the run was only 13.69 MB. The captured
sweep artifact is
`/Volumes/Extreme SSD/rapid-mlx-validation/hy3-mlx0321-coherence-r1.txt`, with
SHA-256 `817969d4c78df19594d7c464990fa0b4e16beda3b8346e423161735ab8b9db72`.
Together with the six ordinary release families, the dependency candidate has
now passed the complete seven-family toolchain coherence fleet (42/42 cases).

### Additional regression spot checks

Three extra cached families were checked on the M2 Pro 32 GB mini with the PR
wheel, released mlx-lm 0.31.3, cold prefix caches, and MLX 0.32.1:

| Model | MLX 0.32.1 | MLX 0.31.2 A/B | Regression verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-9B 4-bit | 6/6 | — | Pass |
| LFM2.5-1.2B 4-bit | 5/6 (`17×23` → `4939`) | identical 5/6 | No runtime regression |
| Nemotron-Labs-Diffusion-3B 4-bit | 5/6 (correct blue answer, extra prose) | identical 5/6 | No runtime regression |

The two non-passing rows are stable model capability/instruction-following
baselines, not output corruption introduced by MLX 0.32.1. Separately, the
Qwen3.5-9B AR benchmark was re-run with an explicit `--ar-only` command and
returned 37.4165 median / 37.4258 pooled tok/s with 8/8 exact prompts, matching
the recorded 37.42 / 37.43 result.

## Versions and checkpoints

| Engine | Version | MLX stack |
| --- | --- | --- |
| Rapid-MLX | 0.12.18, source `a3a0d02bbc050c37923b8a1aeb3773f0e3390f94` | mlx 0.31.2, mlx-lm 0.31.3 |
| mlx-vlm | 0.6.15, source `72f37ca46ace7bb8f8b3fd91d1b6c75e20c77b40` | mlx 0.32.1 |
| oMLX | 0.6.3rc2, source `2df39bfcdd9c8fb80847b2869d7f2d62a162f673` | mlx 0.32.0, mlx-vlm 0.6.3, mlx-lm 0.31.3 |

- `mlx-community/Qwen3.5-4B-MLX-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`, snapshot
  `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`

## Reproduction

The benchmark workspace is `~/mac-model-matrix` on the mini. The direct command
shape was:

```bash
python bench_direct.py \
  --engine rapid \
  --model /path/to/checkpoint \
  --max-tokens 128 \
  --repeat 2 \
  --output results/model-rapid.json
```

Use the isolated Rapid or mlx-vlm environment and change `--engine` for the
direct comparison. oMLX used `~/qwen9-perf/bench_omlx_engine.py` or the matching
Gemma workspace script, with cache storage disabled. The eight prompts cover
coding, explanation, JSON, memory efficiency, dialogue, summary, arithmetic,
and translation. One four-token generation warmed each loaded engine.

Raw JSON remains outside Git under `~/mac-model-matrix/results` on the mini and
`~/mac-model-matrix/mini-results` on the Studio. SHA-256 evidence:

| Artifact | SHA-256 |
| --- | --- |
| `qwen35-4b-rapid-r1.json` | `53bb5b4af82a7332d9c9d326f6f34f852e98afaf28de9d0eac6e358d107bdc65` |
| `qwen35-4b-mlxvlm-r1.json` | `da019873486c42b65f4fe82e1f46651bc4166d34d360f4e5c29183c5934b515b` |
| `gemma4-26b-rapid-r1.json` | `1fe0e08f4f62cb13cc37ca7a17b136b590ec83adfc25dfbe0907fab15e6c764d` |
| `gemma4-26b-mlxvlm-r1.json` | `f667a7247edd2aec129b859616f99236de7e15c666608da5586f793b393f40f8` |
| `qwen38-27b-rapid-ar-r1.json` | `112e446bd225a3baaef1af198dddc085651ecb1006a0282f2a86191353ba9ce5` |
| `qwen38-27b-mlxvlm-ar-r1.json` | `85a330dc2a631ff556207976e3e065d5a1a15326831bc46e3faee963b77fdedd` |
| `qwen38-27b-rapid-mtp-adaptive-r1.json` | `961c51cca9cf0fe670ac76417a5f12174aeee8b0f6d921912fee5eea8a537e88` |
| `qwen35-4b-prefill-rapid-r1.json` | `8e3869a4eb3a2e78f2fb3ae471586d7923b3066d24a6dbf8ebd482d4d750bffa` |
| `qwen35-4b-prefill-mlxvlm-r1.json` | `8dadbfc1350dbc4d91b326b4853784ffc2e17dc89bf2d3a5978309059f1f2d00` |
| `qwen35-4b-prefill-mlxlm-mlx032-r1.json` | `6aab6f33389269dbda14839c72e6698831dc9c92c151213c4e38f14fd3eb8c6b` |
| `qwen35-4b-prefill-mlxlm-main-mlx032-r2.json` | `42690fcfd1563aeb90023b395760961164c9a8ef3990f6022935884a5de4b481` |
| `qwen36-35b-mlxlm0313-mlx032-coherence.json` | `ce47d11810815c0860f8b4db6c40720016a138544daeef7c3043693887ee24ac` |
| `qwen36-35b-mlxlm-main-mlx032-coherence.json` | `908db76bb97967e2095f0493c2caebe58c23b9db5bdce09fd941cb2b6319f82a` |
| `qwen36-vlm-mtp-8bit-coherence-r1.txt` | `58aa10fccb3d96de92d5fccb9b9ba084ff4b823211d43e6154977d39d8feaa68` |
| `qwen36-vlm-mtp-8bit-server-r1.log` | `3d3df9d9dc726e8861f82ba59c65bee3b6655789894233f3d33092870937c801` |
| `hy3-mlx0321-coherence-r1.txt` | `817969d4c78df19594d7c464990fa0b4e16beda3b8346e423161735ab8b9db72` |
| `qwen35-4b-prefill-prod-long-r1.json` | `15295614ba2daacbd3f2fba2a748b83deea23489df91e610a6109eab66755321` |
| `qwen35-4b-prefill-main-long-r1.json` | `0f28ac3d75537ecbd33057288f9eec21760446b7b8aa71cdb527df046ab5eb02` |
| `qwen35-4b-prefill-main-16k-step2048.json` | `1eaa5ef167eaeabc2f7ea5ca7239c063c7aee3771823831c6304895c51731ff5` |
| `qwen35-4b-prefill-main-16k-step4096.json` | `35579aabfd32b75d067f761d73d99b9275173b9b02b23c29122abe72fc3b9f13` |
| `qwen35-4b-prefill-main-16k-step8192.json` | `01c1b812138d24b7f439bd4d2fcdf0062afd4bd37c519b69b78b451a2dbbbe06` |
| `qwen35-4b-prefill-main-16k-step16384.json` | `bbbc65863cd192fd5d6174a183bed4b9bb3613a3f0fc0a8d6ff676aefc3b5a80` |
| `qwen38-27b-prefill-mlx0312-r1.json` | `8194339f25b82756cd1c7aa772d37e9aa59385c319cea8b030531a383703833e` |
| `qwen38-27b-prefill-mlx032-r1.json` | `4e383bf60a797e6e1cb03531e3573782136f9ef11c58702caecce7e870e055b7` |
| `gemma4-26b-prefill-mlx0312-r1.json` | `98e7bf7c80ee5ab8c1ab64e3ee4b6f7ec4542895849f8485226bc57cb8fda8b8` |
| `gemma4-26b-prefill-mlx032-r1.json` | `943217db81e6cd89fa5c2880582b7693b423da3574b0bdf63a4dfeba1883fdb2` |

## Limitations

- Direct in-process generation excludes server scheduling and HTTP overhead.
- The engines use their supported dependency stacks, so this is a real-install
  comparison rather than an isolated MLX-version experiment.
- The direct rows generated the same token budget but were not required to be
  token-identical across different engine implementations.
- Qwen3.8 is a native multimodal model, but this report measures text only.
- M2 Pro Qwen3.8 throughput must not be combined with the existing M3 Ultra
  recommendation number; memory bandwidth and hardware differ materially.
