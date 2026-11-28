# Qwen3.5 LoRA fine-tuning probe (2026-11-28)

Reproducible numbers behind the table in `docs/guides/finetune.md` and the
README. They replace an earlier six-step probe whose data was not recorded and
whose stock speed (0.05 steps/s) matched only the two warm-up steps.

## Environment

- Mac Studio M3 Ultra, 256 GB, macOS 26.5
- MLX 0.32.5; mlx-lm 0.31.3 (Rapid-MLX pin) and 0.32.2 (latest) for stock runs
- Rapid-MLX `main` at `9921fee2a` for native runs
- Checkpoints: `mlx-community/Qwen3.5-9B-4bit`, `mlx-community/Qwen3.5-27B-4bit`

## Data

16 train and 4 valid rows of `{"text": ...}`, each about 2,200 tokens of
synthetic text, so every row is truncated to the 1,024-token maximum and each
step trains 1,023 tokens per batch element. Generator: Python `random.seed(0)`,
2,200 words drawn from a fixed 28-word vocabulary.

## Commands

Stock (isolated venv with `mlx-lm==0.31.3` or `==0.32.2`):

    python -m mlx_lm lora --model mlx-community/Qwen3.5-9B-4bit --train \
      --data ./data --batch-size 1 --num-layers 16 --max-seq-length 1024 \
      --iters 24 --steps-per-report 1

Native:

    rapid-mlx finetune mlx-community/Qwen3.5-9B-4bit --data ./data \
      --batch-size 1 --num-layers 16 --max-seq-length 1024 --iters 30

## Results

| Checkpoint | Batch | Path | Steps | Peak | Steady steps/s |
| --- | ---: | --- | ---: | ---: | ---: |
| 9B | 1 | stock mlx-lm 0.31.3 | 24 | 85.9 GB | 0.107 (steps 1-2: 0.047) |
| 9B | 1 | stock mlx-lm 0.32.2 | 6 | 85.9 GB | ~0.11 (from tok/s) |
| 9B | 1 | native | 30 | 15.35 GB | 0.449 |
| 9B | 4 | native | 20 | 35.23 GB | 0.111 |
| 27B | 1 | native | 20 | 26.29 GB | 0.198 |
| 27B | 2 | native | 20 | 36.62 GB | 0.099 |

For 9B at batch 1 the native path uses 5.6x less memory and runs 4.2x faster
at steady state. The CLI estimate printed before training (14.4 GB, 36.0 GB,
27.1 GB, 36.9 GB) was within 1 GB of each measured native peak.

Loss is not compared here: runs used different step counts. The gradient
equivalence check lives in `tests/test_finetune_runner.py`.
