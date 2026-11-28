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

16 train and 4 valid rows of `{"text": ...}`, each about 2,200 tokens, so every
row is truncated to the 1,024-token maximum and each step trains 1,023 tokens
per batch element. Exact generator (writes `./data/train.jsonl` and
`./data/valid.jsonl`):

```python
import json, os, random
random.seed(0)
words = ("the model learns a mapping from tokens to actions while the optimizer "
         "adjusts low rank adapters across sixteen layers of the network and "
         "memory grows with sequence length").split()
os.makedirs("data", exist_ok=True)
for name, n in (("train", 16), ("valid", 4)):
    with open(f"data/{name}.jsonl", "w") as f:
        for _ in range(n):
            text = " ".join(random.choice(words) for _ in range(2200))
            f.write(json.dumps({"text": text}) + "\n")
```

## Commands

Stock, mlx-lm 0.31.3 (isolated venv with `mlx-lm==0.31.3`), 24 steps:

```bash
python -m mlx_lm lora --model mlx-community/Qwen3.5-9B-4bit --train \
  --data ./data --batch-size 1 --num-layers 16 --max-seq-length 1024 \
  --iters 24 --steps-per-report 1 --steps-per-eval 100 --val-batches 1
```

Stock, mlx-lm 0.32.2 (isolated venv with `mlx-lm==0.32.2`), 6 steps. Its
progress display does not print peak memory, so the run goes through a wrapper
that prints `mx.get_peak_memory()` at exit. Save as `probe.py`:

```python
import atexit, runpy, sys
import mlx.core as mx
atexit.register(lambda: print(f"PEAK_GB {mx.get_peak_memory() / 1e9:.2f}", flush=True))
sys.argv = ["mlx_lm"] + sys.argv[1:]
runpy.run_module("mlx_lm", run_name="__main__")
```

```bash
python probe.py lora --model mlx-community/Qwen3.5-9B-4bit --train \
  --data ./data --batch-size 1 --num-layers 16 --max-seq-length 1024 \
  --iters 6 --steps-per-report 1 --steps-per-eval 100 --val-batches 1
```

Native, one run per reported row:

```bash
for cfg in "9B 1 30" "9B 4 20" "27B 1 20" "27B 2 20"; do
  set -- $cfg   # bash; in zsh use ${=cfg}
  rapid-mlx finetune "mlx-community/Qwen3.5-$1-4bit" --data ./data \
    --batch-size "$2" --num-layers 16 --max-seq-length 1024 --iters "$3" \
    --out "./adapter-$1-b$2"
done
```

Peak memory is the `Peak mem` value mlx-lm prints; steps/s is its `It/sec`
after the first two steps. For 0.32.2, peak is the `PEAK_GB` line and steps/s is
derived from its tokens/s (1,023 tokens per step).

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
