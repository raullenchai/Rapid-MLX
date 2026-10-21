# Configuration Reference

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal. Start a server with its defaults:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Use `rapid-mlx serve --help` for the flags and defaults in your installed version. The [site CLI reference](https://rapidmlx.com/docs/cli.html) explains common choices. Add only the setting your workload needs.

## Server configuration

- **Network:** `--host`, `--port`, and `--api-key` control where the API listens and who can call it. The default host is loopback. See the [server guide](../guides/server.md).
- **Model:** Pass an alias to `serve`; use `rapid-mlx recipe` to pick one that fits this Mac and `rapid-mlx info <alias>` to inspect it.
- **Memory and cache:** Inspect the current `--kv-cache-*`, `--prefix-cache-*`, and `--disable-disk-caches` flags with `rapid-mlx serve --help` before tuning them.
- **Multimodal:** The [image and video guide](../guides/multimodal.md) shows model selection and request formats.

## Speculative decoding options

Speculative modes depend on the model and its matching draft checkpoint. Check the alias profile with `rapid-mlx info <alias>` and the current `--speculative-*` options with `rapid-mlx serve --help` before enabling one.

## Environment variables

Use `rapid-mlx serve --help` for settings with environment equivalents. Keep API keys outside scripts and check the [server guide](../guides/server.md) before exposing a server beyond loopback.
