#!/usr/bin/env python3
"""Compare serial and TensorFold-style parallel Engram page reads, without loading the model."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapid_mlx.models.deepseek_v41_native.engram import (  # noqa: E402
    DiskQuantizedEngramEmbedding,
)
from rapid_mlx.models.deepseek_v41_native.load import (  # noqa: E402
    resolve_indexed_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rows", type=int, default=12288)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 128 <= args.rows <= 65536 or not 1 <= args.repeats <= 20:
        parser.error("rows must be 128..65536 and repeats must be 1..20")

    snapshot = args.snapshot.resolve()
    config = json.loads((snapshot / "config.json").read_text())
    text = config.get("text_config") or config
    layers = text["engram_layer_ids"]
    position = layers.index(args.layer)
    prefix = f"layers.{args.layer}.engram.embed"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    filenames = {index[f"{prefix}.{part}"] for part in ("weight", "scales", "biases")}
    if len(filenames) != 1:
        parser.error("the selected Engram tensors must share one shard")
    shard = resolve_indexed_shard(str(snapshot), filenames.pop())
    quant = config["quantization"]
    table = DiskQuantizedEngramEmbedding(
        shard,
        weight_key=f"{prefix}.weight",
        scales_key=f"{prefix}.scales",
        biases_key=f"{prefix}.biases",
        num_embeddings=text["engram_num_embeddings"][position],
        dim=text["engram_head_dim"],
        group_size=quant["group_size"],
        bits=quant["engram_bits"],
        cache_rows=0,
    )
    try:
        timings = {"serial": [], "parallel": []}
        for repeat in range(args.repeats):
            order = ("serial", "parallel") if repeat % 2 == 0 else ("parallel", "serial")
            for slot, mode in enumerate(order):
                indices = np.random.default_rng(10007 + 2 * repeat + slot).integers(
                    0, text["engram_num_embeddings"][position], size=args.rows, dtype=np.int64
                )
                table.read_ahead = mode == "parallel"
                started = time.perf_counter()
                values = table._raw_rows(indices)
                elapsed = time.perf_counter() - started
                timings[mode].append(elapsed)
                print(
                    json.dumps(
                        {
                            "repeat": repeat,
                            "mode": mode,
                            "seconds": round(elapsed, 6),
                            "rows": args.rows,
                            "returned_bytes": sum(value.nbytes for value in values),
                        }
                    ),
                    flush=True,
                )

        check = np.random.default_rng(1997).integers(
            0, text["engram_num_embeddings"][position], size=args.rows, dtype=np.int64
        )
        table.read_ahead = False
        reference = table._raw_rows(check)
        table.read_ahead = True
        candidate = table._raw_rows(check)
        equal = all(np.array_equal(a, b) for a, b in zip(reference, candidate))
        summary = {
            "event": "engram_page_read_summary",
            "snapshot": str(snapshot),
            "layer": args.layer,
            "rows": args.rows,
            "serial_median_seconds": statistics.median(timings["serial"]),
            "parallel_median_seconds": statistics.median(timings["parallel"]),
            "raw_rows_equal": equal,
            "scope": "selected-row file I/O only; no model load or token generation",
        }
        print(json.dumps(summary), flush=True)
        if not equal:
            raise SystemExit("parallel read-ahead changed Engram rows")
    finally:
        table.close()


if __name__ == "__main__":
    main()
