# Qwen3.8-Flash-Next mixed-q4 streaming converter

**Lane:** script-only helper for Vector's `qwen4_exp` port. Never edits model
math. It is verified against a scaled synthetic checkpoint and by a complete
header-only classification of the pinned real checkpoint before payload reads.

## Why this shape

The real checkpoint (`Qwen/Qwen3.8-Flash-Next`, revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`) has ~113–131 safetensors shards and
a ~51B-param PLE embedding table spread across **128 PLE shards**. Loading the
PLE table into memory (or concatenating it) blows past any reasonable footprint
guard. So this converter:

1. processes one checkpoint tensor at a time, and
2. uses the model's fine-grained quantization contract: PLE is q4-g32, routing
   gates are q8-g64, and ordinary eligible matrices are q4-g64. It never
   concatenates the ~51B PLE table.

## Files

| file | purpose |
|---|---|
| `scripts/qwen38_streaming_convert.py` | the converter |
| `scripts/synthetic_qwen38_fixture.py` | generate a scaled-down synthetic shard set |
| `scripts/verify_synth_conversion.py` | end-to-end round-trip + SHA-256 verifier |
| `scripts/test_fail_closed_guards.py` | fail-closed guard tests |

## Requirements

Python 3.11+, `mlx`, `safetensors`, `numpy`. This worktree has a ready venv:

```sh
source .venv/bin/activate   # mlx 0.32.2, safetensors, numpy
```

## Run the synthetic verification

```sh
# 1. build a tiny stand-in checkpoint (same name/dtype/shape pattern, scaled down)
.venv/bin/python scripts/synthetic_qwen38_fixture.py /tmp/synth-src

# 2. convert it (fail-closed)
.venv/bin/python scripts/qwen38_streaming_convert.py \
    --source /tmp/synth-src \
    --output /tmp/synth-out \
    --max-shard-bytes 2000000

# 3. verify COPY preservation, mixed-quant round-trip, total_size, manifests
.venv/bin/python scripts/verify_synth_conversion.py /tmp/synth-src /tmp/synth-out

# 4. prove the guards abort closed
.venv/bin/python scripts/test_fail_closed_guards.py /tmp/synth-src
```

Expected: all green, with the converter ledger reporting `peak_rss_bytes`
(phys_footprint). On the synthetic set peak RSS is a few hundred MB.

## Converter CLI

```
--source           <snapshot dir>   must contain model.safetensors.index.json
--output           <dir>            must NOT exist (aborts if it does)
--max-shard-bytes  <int>            output shard cap (default 4 GiB)
--inspect-only                       classify all headers; read no payloads
```

Runbook-guard parameters (only via the Python API today): `min_free_bytes`
(default 140 GiB) and `max_rss_bytes` (default 220 GiB). On a real run the
operator confirms ≥140 GiB free at the output root before starting.

## Output contract

* The 128 exact PLE shard embeddings (width 160) use affine q4-g32. The 98
  decoder/MTP routing gates use affine q8-g64. Other eligible matrices use
  affine q4-g64. No padding, slicing, or name-derived model heuristic exists.
  Every quantized tensor emits its weight, `.scales`, and `.biases` keys.
* **Copy** tensors (1-D norms/biases, `A_log`, aux/buffer dtypes, widths not
  divisible by the applicable group size) → carried through value-for-value with their source
  shape/dtype, so no quantisable-unfriendly tensor is dropped or mangled.
* Non-weight metadata is copied at the root. `config.json` additionally records
  the global q4-g64 format and exact per-module PLE/routing overrides consumed
  by the loader.
* `model.safetensors.index.json` maps every emitted weight/scales/biases key and
  `total_size` is the sum of every emitted tensor's bytes.
* Output shards use deterministic `model-{i:05d}-of-{N:05d}.safetensors` with
  the real total `N` (bounded by `--max-shard-bytes`).
* `SHA256SUMS.txt` → byte-sorted `sha256  <relative path>` per output file.
* Execution ledger on stdout: file count, output bytes, shard list, **peak RSS**,
  mixed quant/copy counts, source/output weight bytes, and `status`.

## Fail-closed guarantees

| guard | behavior on violation |
|---|---|
| output dir exists | abort, no publish |
| missing / empty `model.safetensors.index.json` | abort |
| weight_map references a missing source shard | abort |
| non-flat shard path or symlink outside the model cache root | abort |
| source or output under `/Volumes/Extreme SSD` | abort |
| free space at output root < 140 GiB | abort |
| process peak RSS > 220 GiB | abort mid-run |

Copied BF16 tensors remain BF16. There is no upload option; publication is a
separate, explicitly guarded operation after loader and evaluation evidence.

## Reference

* `mlx_lm convert.py` (`load(..., lazy=True)` → `quantize_model` → `save`) — the
  streaming MoE pattern this mirrors.
* `/private/tmp/rapid-qwen38-ops/qwen38_conversion_runbook.md` — operator
  runbook (source revision, output root, guards, command shape) this prototype
  implements.
