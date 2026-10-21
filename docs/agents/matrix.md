# Coding agents

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, then launch your installed coding agent with a local model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx launch codex
```

Replace `codex` with `claude-code`, `opencode`, `pi`, `hermes`, `crush`, or `goose`. Run the same command next time. Use `rapid-mlx launch <agent> --remove` to remove Rapid's private profile for that agent. Run `rapid-mlx launch list` for other integrations.

## Guides

- [Codex](codex-cli.md)
- [Claude Code](claude-code.md)
- [OpenCode](opencode.md)
- [Pi](pi.md)
- [Hermes](hermes-agent.md)
- [Crush](crush.md)
- [Goose](goose.md)
- [Claude Desktop Gateway](../integrations/claude-desktop.md)

The [current compatibility guide](https://rapidmlx.com/docs/matrix.html) has detailed API routes and test status. Models vary in memory use and tool-calling ability; run `rapid-mlx recipe` for recommendations on this Mac.
