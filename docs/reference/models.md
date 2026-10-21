# Supported Models

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal. Ask the CLI which models fit this Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx recipe
rapid-mlx models
```

Use `rapid-mlx info <alias>` for one model's memory needs, capabilities, and runtime profile. The [model catalog](https://rapidmlx.com/docs/aliases.html) is the current long-form list. An alias in the installed v0.16.6 registry is the source for a runnable model name.

## Text and vision

Run `rapid-mlx models` to browse current text and vision aliases. Start a recommended text model with `rapid-mlx chat` or `rapid-mlx serve <alias>`. For images in a request, follow the [multimodal guide](../guides/multimodal.md).

## DeepSeek V4.1 Flash 1017

`deepseek-v41-flash-1017` is an experimental text-only model for a 256 GB Apple Silicon Mac. The CLI refuses to load it below 224 GiB unified memory. It serves greedy OpenAI chat with an 8,192-token input limit and a 4,096-token output limit; tool calling is not qualified. The alias uses the current pinned GPTQ weights. Check its requirements before downloading:

```bash
rapid-mlx info deepseek-v41-flash-1017
rapid-mlx serve deepseek-v41-flash-1017
```

The default serves one request at a time. For concurrent requests, set `RAPID_MLX_DSV41_LANES=2`, `4`, or `8` before starting the server; this turns drafting off. See the [0.16.6 release notes](../release-notes/v0.16.6.md) for the pinned weights and runtime details.

## Flint

`flint-40b-a2.5b-4bit` serves Recursive's Flint (40B total / ~2.5B active MoE)
from the official MLX INT4 checkpoint with `glm47` tool calls and `qwen3`
reasoning. Validated revision: `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`.
It uses dense and sparse expert layers, sliding-window 513 + full no-RoPE
layers, learned attention sinks, and an 8-bit embedding, BF16 routers, and 4-bit
attention/expert/output projections.
48 GB unified memory is recommended (weights: 22.6 GB); it runs on 36 GB,
but concurrent ~8k prefills are admitted one at a time. KV cache stays bf16:
quantized KV is refused because the quantized attention path does not support sinks.
Speculative decoding and MTP are not enabled for this alias.

Measured single-stream on M5 Max 36 GB (mlx 0.32.4, mlx-lm 0.31.3, greedy, 256 output tokens, streaming, defaults): ~1k TTFT 0.35–0.38 s / 131 tok/s / 23.6 GiB peak; ~8k TTFT 2.17–2.18 s / 118.5 tok/s / 24.9 GiB peak. See the [release notes](../release-notes/unreleased.md#highlights) for the server command and concurrency results.

## Image, video, and audio

The CLI lists these model types separately. Start from the [image-generation guide](https://rapidmlx.com/docs/models/families/image.html), [video guide](../guides/video-generation.md), or [audio guide](../guides/audio.md) for a working request. Check the selected alias with `rapid-mlx info <alias>` before downloading a large model.

## Embeddings and reranking

Use the [embeddings guide](../guides/embeddings.md) or [rerank guide](../guides/rerank.md) for the optional runtime and serving commands.

## TensorFold family profiles

The following aliases are available for qualified hardware and their matching draft paths. Use `rapid-mlx info <alias>` for requirements:

- `bonsai2-27b-tensorfold`
- `nemotron-3.5-lightning-tensorfold`
- `qwen3.8-flash-next-tensorfold`
- `gemma-4-26b-tensorfold`
- `deepseek-v4-flash-tensorfold`
