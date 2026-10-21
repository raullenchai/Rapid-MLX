# Audio measurement

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Follow the [audio guide](../guides/audio.md) to run a speech or audio request with your own file, then record the elapsed time and media duration. Use the same file, model, and options when comparing runs. The shipped `rapid-mlx bench` command measures text generation, so its output is not an audio result.

For speech-to-text, a useful measure is media duration divided by elapsed transcription time. Include the model alias, chip, unified memory, audio duration, and Rapid-MLX version with any result.
