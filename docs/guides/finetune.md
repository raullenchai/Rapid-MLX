# LoRA fine-tuning on your Mac

Use `rapid-mlx finetune` (alias `rapid-mlx lora`) for local LoRA fine-tuning of
a text model on Apple Silicon. Quantized MLX checkpoints use QLoRA. Choose a
checkpoint that supports [serving LoRA adapters](../reference/cli.md#rapid-mlx-serve).
Use checkpoints from sources you trust, since mlx-lm allows custom tokenizer code.

Supply a JSONL file or a directory with `train.jsonl` and `valid.jsonl`.
Use one format within each file, with one JSON object per line:

- Chat: `{"messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello!"}]}`
- Prompt and completion: `{"prompt":"Say hello","completion":"Hello!"}`
- Plain text: `{"text":"Hello from a local model."}`

A single file is split reproducibly, holding out about 10% for validation.
Rapid-MLX holds out at least one batch for validation. Each split needs a full
batch (four rows by default). Lower `--batch-size` for small datasets.

This example checks the workflow with at least two rows in `examples.jsonl`.
For fine-tuning on your task, use representative data and increase `--iters`.

```bash
rapid-mlx finetune qwen3-0.6b-4bit --data ./examples.jsonl --out ./my-adapter --iters 10 --batch-size 1 --num-layers 4 --max-seq-length 512
rapid-mlx serve qwen3-0.6b-4bit --adapter ./my-adapter --port 8613
# In a second terminal, with Codex installed:
rapid-mlx agents codex --setup --base-url http://127.0.0.1:8613/v1
codex
```

Agent setup writes the local server and adapter model ID to the Codex config.
Use a fresh output directory for each run. The default is a timestamped path under
`~/.rapid-mlx/adapters/`. The memory estimate is advisory. On success, Rapid-MLX
prints the serve command. To start a new run from saved LoRA weights, pass
`--resume path/to/adapters.safetensors` with the same base model and layer settings.

When telemetry is enabled, the completion event contains the outcome, elapsed
time and row-count buckets, and whether validation data was prepared.
