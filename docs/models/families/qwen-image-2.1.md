# Qwen-Image 2.1

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start the catalog image-generation alias:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen-image-2.1
```

The server prints its local address and may choose a free port if 8000 is busy.
Send a prompt to that address's `/v1/images/generations` route or select the
model in the Mac app. Run `rapid-mlx info qwen-image-2.1` to inspect the current
checkpoint and download size before starting it; `rapid-mlx recipe` helps choose
a model for your Mac.

## Other catalog choices

- `qwen-image-2.1-bf16` uses the larger checkpoint.
- `qwen-image-2.1-turbo` is another image-generation alias.
- `qwen-image-edit` is the instruction-edit variant.

Run `rapid-mlx models --modality image-gen` for the current catalog list. The [server guide](../../guides/server.md) lists the image API routes, and `rapid-mlx serve --help` lists current runtime options.
