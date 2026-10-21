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

## Image, video, and audio

The CLI lists these model types separately. Start from the [image and vision guide](../guides/multimodal.md), [video guide](../guides/video-generation.md), or [audio guide](../guides/audio.md) for a working request. Check the selected alias with `rapid-mlx info <alias>` before downloading a large model.

## Embeddings and reranking

Use the [embeddings guide](../guides/embeddings.md) or [rerank guide](../guides/rerank.md) for the optional runtime and serving commands.
