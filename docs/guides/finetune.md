# LoRA fine-tuning on your Mac

Use `rapid-mlx finetune` (alias `rapid-mlx lora`) for local LoRA fine-tuning of
a text model on Apple Silicon. Quantized MLX checkpoints use QLoRA.
Use checkpoints from sources you trust, since mlx-lm allows custom tokenizer code.

## Which model should I fine-tune?

Choose an mlx-lm text checkpoint that supports LoRA, such as `qwen3-0.6b-4bit`,
or a dense Qwen3.5 checkpoint such as `qwen3.5-9b-4bit` or `qwen3.5-27b-4bit`.
For dense Qwen3.5, `serve --adapter` automatically loads the text backbone.
Image inputs return HTTP 400. Adapter serving disables speculative decoding
and requires a base checkpoint without MTP draft tensors. See the
[serve CLI reference](../reference/cli.md#rapid-mlx-serve).

Qwen3.5 training uses the native gated-delta backward pass when available
(MLX 0.32.4 or newer with a compatible mlx-lm). On older MLX, the stock
training path runs with a warning about speed and memory use.

These measurements were taken on an M3 Ultra with 4-bit weights, 16 adapted
layers, a maximum sequence length of 1,024 tokens and training rows that fill
it. Steps/s is the steady-state rate; the first two stock steps run at about
0.05 while compiling. Method and commands:
[fine-tuning probe](../engineering/performance/2026-11-28-finetune-probe.md).

| Checkpoint | Batch | Path | Peak memory | Steps/s |
| --- | ---: | --- | ---: | ---: |
| Qwen3.5-9B | 1 | Stock mlx-lm 0.31.3 | 85.9 GB | 0.11 |
| Qwen3.5-9B | 1 | Native gated delta | 15.4 GB | 0.45 |
| Qwen3.5-9B | 4 | Native gated delta | 35.2 GB | 0.11 |
| Qwen3.5-27B | 1 | Native gated delta | 26.3 GB | 0.20 |
| Qwen3.5-27B | 2 | Native gated delta | 36.6 GB | 0.10 |

Stock mlx-lm 0.32.2, the latest release, peaked at the same 85.9 GB on this
probe. Shorter rows use less memory on both paths.
The CLI memory estimate assumes the native path for Qwen3.5.

## Prepare data and run

Supply a JSONL file or a directory with `train.jsonl` and `valid.jsonl`.
Use the same format across all splits, including `test.jsonl` if present, with
one JSON object per line:

- Chat: `{"messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello!"}]}`
- Prompt and completion: `{"prompt":"Say hello","completion":"Hello!"}`
- Plain text: `{"text":"Hello from a local model."}`

A single file is split reproducibly, holding out about 10% for validation.
Rapid-MLX holds out at least one batch for validation. Each split needs a full
batch (four rows by default). Lower `--batch-size` for small datasets.

If you omit `--iters`, Rapid-MLX prints its choice of three iterations per
full training batch, for about three epochs with a minimum of 3 iterations and a
cap of 1,000. The learning-rate default is `1e-5`. If the final reported
validation loss equals or exceeds the initial loss, Rapid-MLX suggests trying more
training data or a different `--learning-rate`.

This example checks the workflow with at least two rows in `examples.jsonl`.
For fine-tuning on your task, use representative data.

```bash
rapid-mlx finetune qwen3-0.6b-4bit --data ./examples.jsonl --out ./my-adapter --iters 10 --batch-size 1 --num-layers 4 --max-seq-length 512
rapid-mlx serve qwen3-0.6b-4bit --adapter ./my-adapter --port 8613
# In a second terminal, with Codex installed:
rapid-mlx agents codex --setup --base-url http://127.0.0.1:8613/v1
codex
```

Agent setup writes the local server and adapter model ID to the Codex config.
Use a fresh output directory for each run. The default is a timestamped path under
`~/.rapid-mlx/adapters/`. The memory estimate is advisory. The tuner prints
measured `Peak mem` during training. On success, Rapid-MLX
prints the serve command. To start a new run from saved LoRA weights, pass
`--resume path/to/adapters.safetensors` with the same base model and layer settings.

When telemetry is enabled, the completion event contains the outcome, elapsed
time and row-count buckets, and whether preflight accepted a validation split you
supplied.
