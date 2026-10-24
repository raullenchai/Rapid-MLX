<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

- **Nemotron 3.5 Bolt E4B** — pinned four-bit and eight-bit text aliases with native tool calls and explicit thinking controls. Gemma architecture detection now takes precedence over the legacy Nemotron tokenizer fallback.

- Catalog aliases support an optional immutable `revision`; Flint now enforces its validated commit across downloads, metadata, cache checks, and loading.

- **Recursive Flint support** — Rapid-MLX serves Recursive's Flint (40B total / ~2.5B active MoE, sliding-window 513 + full no-RoPE layers, learned attention sinks) from the official MLX INT4 checkpoint via `rapid-mlx serve flint-40b-a2.5b-4bit`, with `glm47` tool calls and `qwen3` reasoning. Validated revision: `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`. KV cache stays bf16; quantized KV is refused because the quantized attention path does not support sinks. No speculative decoding or MTP. 48 GB unified memory recommended (weights: 22.6 GB); it runs on 36 GB, but concurrent ~8k prefills are admitted one at a time. ([#5079](https://github.com/raullenchai/Rapid-MLX/pull/5079))

Measured on M5 Max 36 GB, mlx 0.32.4, mlx-lm 0.31.3, greedy decoding,
256 output tokens, streaming; server: `rapid-mlx serve <flint-int4 dir> --host 127.0.0.1 --port 18742`, default settings.

| Context | Mode | TTFT | Decode tok/s | Aggregate tok/s | Peak footprint |
| --- | --- | --- | --- | --- | --- |
| ~1k | single | 0.35–0.38 s | 131 | — | 23.6 GiB |
| ~1k | 4 concurrent (batched) | 1.0–1.7 s | 62–70 per request | 193 | 22.9 GiB |
| ~8k | single | 2.17–2.18 s | 118.5 | — | 24.9 GiB |
| ~8k | 4 concurrent (admitted serially on 36 GB) | 2.2–15.0 s | 119 per request | 59.6 | 23.5 GiB |
