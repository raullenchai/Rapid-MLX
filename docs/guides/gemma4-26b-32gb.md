# Gemma 4 26B-A4B on a 32GB Mac

`gemma-4-26b-4bit` is a 128-expert MoE: 25.2B total parameters, 3.8B active
per token, 30 layers, 256K context, 14.3 GiB of weights. On a 32GB Apple
Silicon machine it is faster than the dense 12B on **both** prefill and
decode, so on any Mac that can hold it there is no throughput reason to
prefer the 12B.

Everything below was measured on a Mac mini M2 Pro / 32GB, macOS 26.5.2,
mlx 0.31.2, single request, `temperature=0`, medians over 4-6 repetitions
with a distinct prompt per repetition.

## Recommended invocation

```bash
rapid-mlx serve gemma-4-26b-4bit \
  --no-mllm \
  --kv-cache-dtype bf16 \
  --cache-memory-mb 512
```

Every one of those three flags is off the default path, and each is worth
measurable throughput. They are explained below.

## Why these flags

### `--kv-cache-dtype bf16` — the default costs you up to 14%

The server defaults to **int4** KV for this model, and says why:

```
KV cache dtype: int4 — Defaulting to int4 (memory-bandwidth-bound on
M-series); model=mlx-community/gemma-4-26b-a4b-it-4bit not in safelist
```

The model gets quantized KV by *failing to appear on a list*, not because
anyone measured it. Measured, quantizing KV is a straight loss here:

| KV dtype | chat | codegen | code edit |
|---|---:|---:|---:|
| int4 (default) | 47.9 | 46.9 | 41.7 |
| int8 | 47.6 | 46.6 | 40.1 |
| **bf16** | **49.5** | **49.8** | **47.4** |

(tok/s decode, higher is better)

The reason is the architecture. Only 5 of the 30 layers use full attention
(indices 5, 11, 17, 23, 29); the other 25 are sliding-window with a 1024
window, so their cache is bounded and only those 5 grow with context. Note
that full-attention layers use `num_global_key_value_heads` (2), **not** the
`num_key_value_heads` (8) that the sliding layers use — see
`gemma4_vendored/language.py`, where `attention_k_eq_v` selects the global
count. So the growing KV is

```
5 layers x 2 global KV heads x 512 global_head_dim x 2 (K+V) x 2 B
    = 20 KB/token
```

which is tiny — a 32K context costs 640MB. KV is nowhere near the bandwidth
bottleneck on this model, so quantizing it buys nothing measurable and the
quantize/dequantize work is pure overhead.

For comparison the dense 12B is **16 KB/token** (8 full-attention layers but
`num_global_key_value_heads = 1`), so the 26B actually carries slightly
*more* growing KV per token, not less. Its advantage is throughput, not
cache footprint.

Note the checkpoint's own `config.json` is the authority here — the model
card on the Hub describes 4 full-attention layers, and there are 5.

### `--no-mllm` — text-only serving, for the prefix cache

The checkpoint carries a `vision_config`, so auto-detection routes it to the
MLLM lane. That lane does **not** use the prefix cache: a repeated
502-token prompt re-prefills at 2.44s every single time, where the text lane
answers in 0.28s once the prefix is cached.

Cold prefill is the same speed in both lanes (~4.8 ms/token) — the entire
advantage is cache reuse, which is exactly what agent traffic generates.
Pass `--mllm` if you actually need image input.

### `--cache-memory-mb 512` — buys headroom for free

The default sizes the prefix cache at 20% of RAM (~3.2GB here). Shrinking it
to 512MB cut peak Metal memory at a 16K-token prompt from 24.17 GB to
**21.51 GB** with no measurable throughput cost at 4K or 8K. On a 32GB
machine the allocation limit is 24.1GB, so 2.7GB of headroom is the
difference between comfortable and one bad request from an abort.

## What you get

With the recommended three flags:

| prompt | TTFT | decode | peak Metal |
|---:|---:|---:|---:|
| ~500 tok (cached prefix) | 0.28s | 47 tok/s | — |
| 1K | 3.8s | 41.7 tok/s | 15.6 GB |
| 4K | 15.0s | 38.0 tok/s | 16.3 GB |
| 8K | 29.4s | 38.0 tok/s | 16.6 GB |

Against `gemma-4-12b-4bit` on the same machine. Both models here run
`--no-mllm --kv-cache-dtype bf16` *without* `--cache-memory-mb 512`, so the
26B column differs slightly from the table above:

| prompt | 26B TTFT | 12B TTFT | 26B decode | 12B decode |
|---:|---:|---:|---:|---:|
| 1K | **3.8s** | 9.9s | **41.7** | 19.6 |
| 4K | **15.0s** | 37.9s | **38.0** | 19.4 |
| 8K | **32.7s** | 81.8s | **32.8** | 17.6 |

2.5x the prefill and 2x the decode from a model with twice the parameters.
That is the MoE trade working in your favour: only 3.8B parameters are
active per token. It costs 8GB more resident weights and slightly more KV
per token; everything else moves the right way.

## Limits worth knowing before you rely on it

**Throughput becomes unreliable past ~12K tokens of context.** Up to 8K,
decode is steady — six consecutive 8K requests measured 36.3-38.5 tok/s
(±3%). At 12K the same six-run protocol gives 16.4-27.1 tok/s, and
individual samples as low as 8.1. Time-to-first-token stays perfectly
linear and reproducible throughout (±0.5%); only decode is affected. The
cause is not prompt length, response content, machine load, or prefix-cache
occupancy — all four were tested and ruled out. **If you need predictable
throughput, keep contexts under ~8K.**

**A 16K prompt drives peak Metal memory to 21.5GB** against a 24.1GB
allocation limit. Metal allocation failure aborts the process rather than
raising, so treat that as the working ceiling on a 32GB machine, not a
target.

**Leave SuffixDecoding off** (it is off by default; the server also says so
at boot). Measured on this model:

| workload | baseline | suffix K=8 |
|---|---:|---:|
| chat | 49.5 | 30.6 (**-38%**) |
| codegen | 49.7 | 26.3 (**-47%**) |
| code edit | 47.1 | 52.6 (+12%) |

The cause is acceptance, not verify cost. Measured on this machine, a
9-wide forward costs **6.13x** a 1-wide one on the 26B and **6.05x** on the
dense 12B — so being MoE does not make speculation structurally more
expensive here, which is worth stating because it is the intuitive guess
and it is wrong. What that cost curve means is that a drafter has to land
about **0.64 acceptance per token at K=8** simply to break even:

| verify width | cost vs 1-wide | acceptance needed to break even |
|---:|---:|---:|
| 2 | 1.74x | 0.741 |
| 4 | 3.12x | 0.707 |
| 5 | 3.82x | 0.704 |
| 9 | 6.13x | 0.641 |
| 13 | 7.29x | 0.524 |

A suffix tree reaches that on text it has already seen — hence the +12% on
code edit — and nowhere near it on a fresh question, which is why ordinary
traffic pays 6x for a verify that emits barely more than one token. A
*trained* drafter is a different proposition against the same table; see
the note on the official assistant models below.

## Agent workloads

Six consecutive `aider` edit rounds against the recommended configuration
completed without a crash, a dropped connection, or a 503, at 14-36s per
round. Server-side stability is not the limitation.

Edit *quality* is. Across seven runs of one simple refactor task ("add type
hints and docstrings"), six produced correct output and one wrote the
model's own deliberation into the source file as comments while inverting a
`random.uniform(0.5, 1.0)` range to `(0.5, 0.1)`. Enabling `--reasoning`
neither caused nor prevented it — three runs with it and three without were
all clean, and the failure came from a run in between. Budget for roughly
one bad edit in seven on a task this size, and keep the tests green between
rounds rather than trusting a batch of edits.

## Things that do not apply

- **MTP / the official assistant drafter.** Google ships a 0.4B, 4-layer
  drafter for every Gemma 4 size, including this one
  (`gemma-4-26b-assistant`), and this repo already implements the injection
  path in `vllm_mlx/spec_decode/mtp/gemma4_inject.py`. It is nonetheless
  unreachable: `_SUPPORTED_MODEL_TYPES` in `vllm_mlx/spec_decode/mtp/detect.py`
  admits only `qwen3_5`, `qwen3_5_moe` and `hy_v3`, and the docstring records
  that "Gemma 4 sidecar promotion remains disabled until it passes
  end-to-end greedy-lossless validation". The drafter's config does line up
  with this target (`backbone_hidden_size` 2816 matches the target's
  `hidden_size`; its four `layer_types` match the target's last four), so the
  validation is worth doing — but it has not been done, and nothing here
  depends on it.
- **DFlash / DDTree.** Both require >=8-bit precision and a declared drafter;
  the 4-bit alias has neither.
- **PFlash / TurboQuant.** Both tiers are still `unknown` for this alias.
  Neither was exercised by the measurements above.
