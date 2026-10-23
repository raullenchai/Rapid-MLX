# CLI Reference

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal. Launch an installed coding agent with a local model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx launch codex
```

Replace `codex` with `claude-code`, `opencode`, `pi`, `hermes`, `crush`, or `goose`. Run `rapid-mlx launch list` for other client integrations. The [site CLI reference](https://rapidmlx.com/docs/cli.html) covers longer workflows. The installed CLI is authoritative for flags and defaults:

```bash
rapid-mlx --help
rapid-mlx <command> --help
```

## Everyday commands

- **Coding agent:** `rapid-mlx launch codex` selects a model, starts a local server if needed, and opens Codex with a private profile. Run it again next time. `rapid-mlx launch codex --remove` removes that profile.
- **Terminal chat:** `rapid-mlx chat` downloads a starter model if needed and opens a chat session.
- **Local API:** `rapid-mlx serve qwen3.5-4b-4bit` prints the server address. Use the printed `/v1` endpoint for compatible clients.
- **Choose a model:** `rapid-mlx recipe` recommends models for this Mac; `rapid-mlx models` lists aliases; `rapid-mlx info <alias>` shows one model profile.
- **Check installation:** `rapid-mlx doctor` reports environment and dependency checks.

## `rapid-mlx serve`

```bash
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx serve --help
```

The server listens on loopback by default and prints its selected port. An explicit `--port` does not fall back to another port. See the [server guide](../guides/server.md) for the API and network setup.

An interactive `serve` prints a one-time [newsletter pointer](https://rapidmlx.com/newsletter) after its ready banner.

### KV cache dtype and quantization

The default BF16 KV cache favors fidelity. `--kv-cache-dtype int8` or `int4` uses less memory and can change output quality or speed. Use `rapid-mlx serve --help` for the current quantization options and defaults; the [performance flags guide](https://rapidmlx.com/docs/perf-flags.html) covers tuning.

### PFlash long-prompt compression

PFlash compresses long prompts before prefill. `--pflash auto` applies it at the configured threshold; `--pflash always` applies it to eligible prompts. It is off by default. Compression discards some prompt content, so validate results on your workload. Use `rapid-mlx serve --help` for its current tuning flags and the [performance flags guide](https://rapidmlx.com/docs/perf-flags.html) for examples.

## `rapid-mlx chat`

```bash
rapid-mlx chat
rapid-mlx chat --help
```

Type `/help` in the chat session for its interactive commands.

## `rapid-mlx launch`

```bash
rapid-mlx launch codex
rapid-mlx launch claude-code
rapid-mlx launch list
rapid-mlx launch --help
```

`launch` uses a private agent profile for supported coding agents. Run the same command after a restart. See the [agent guides](../agents/matrix.md) for each agent's path. Claude Desktop uses its [in-app Gateway](../integrations/claude-desktop.md).

## `rapid-mlx agents` and `rapid-mlx start`

These commands provide explicit agent configuration, connection tests, and foreground server control. Run `rapid-mlx agents --help` or `rapid-mlx start --help` for their current options. For the primary coding agents, use `rapid-mlx launch <agent>`.

## Optional workloads

- **Embeddings:** [embeddings guide](../guides/embeddings.md)
- **Audio:** [audio guide](../guides/audio.md)
- **Video:** [video guide](../guides/video-generation.md)
- **Typed decisions:** [System One guide](../guides/system-one.md)

## Settings and diagnostics

The [configuration reference](configuration.md) gives the setup path for server settings. Run `rapid-mlx doctor --help` and `rapid-mlx telemetry --help` for current diagnostic and privacy controls.
