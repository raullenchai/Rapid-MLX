# Codex local first turn, 20 October 2026

This measurement used Codex CLI 0.163.0 on an Apple M4 Pro with 48 GiB RAM.
Rapid-MLX 0.16.0 served `mlx-community/Qwen3.5-9B-4bit` on a private
loopback port. Each run used a fresh `HOME` and `CODEX_HOME`, a restarted
server, and the same prompt: `Reply exactly READY.` No existing user profile
or server was involved.

| Codex selection | First-turn input tokens | Uncached input tokens | Command duration | Session `turn_context.model` |
| --- | ---: | ---: | ---: | --- |
| Flagship-model selection with the previous private profile | 9,548 | 9,548 | 26.9 s | `gpt-6-sol` |
| Local-model selection with the revised private profile | 5,100 | 5,100 | 16.3 s | `qwen3.5-9b-4bit` |

The first row reproduces the reported wrong-model *selection* by passing
`--model gpt-6-sol` over a profile whose TOML selected the local model. A
clean Codex 0.163.0 profile without that override selected the local model in
our separate CLI probe, so this measurement does not prove why the owner's
interactive session retained the flagship selection. It measures the effect
of that selection on the same 9B weights. The server log and Codex usage
agreed on prompt token counts. The server reported 26.48 s and 14.01 s for
the respective response streams; command durations include startup overhead.

The shorter prompt results from both explicit local-model metadata and the
private profile's app, plugin, memory, and skill-catalog settings. This run
does not apportion the reduction among those settings. A separate clean
profile-only probe used 5,215 input tokens before the configuration change,
which suggests the model-selection error was the dominant cost.

The distilled [run evidence](fixtures/launch-codex-first-turn-2026-10-20.json)
records the tested code commit, Codex thread IDs, effective models, usage, and
server timings. Raw session transcripts are deliberately omitted.

To reproduce, use Codex 0.163.0 and a fresh home for each row. Generate a
private Codex profile with `rapid-mlx launch codex --isolated --model
qwen3.5-9b-4bit --port <free-loopback-port>`. For the baseline, remove the
generated `[features]`, `[memories]`, and `[skills]` sections and invoke Codex
with `--model gpt-6-sol`; for the revised run, leave those sections intact and
invoke Codex with `--model qwen3.5-9b-4bit`. Both commands also pass `exec
--profile rapid --skip-git-repo-check --json` and the prompt above. Start a
fresh server for each run from the same weights with `rapid-mlx serve
<weights-directory> --served-model-name <Codex model> --host 127.0.0.1
--port <free-loopback-port>`. Set `RAPID_MLX_API_KEY` to the same test value
for Codex and server, and point the profile provider URL to the loopback
server. Record Codex's `turn.completed.usage`, the session JSONL
`turn_context.model`, and the server's `Responses (stream): prompt=…` line.
