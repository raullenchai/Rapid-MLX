# SPDX-License-Identifier: Apache-2.0
"""Convert Marvin's Garden contrastive pairs into mlx-lm chat SFT files.

Output shape (one JSON object per line, mlx_lm.lora chat format)::

    {"messages": [{"role": "user", "content": "<rendered prompt>"},
                  {"role": "assistant", "content": "<single letter>"}]}

The assistant content is EXACTLY one decision letter (Nimble-style: no CoT,
no JSON — the serving path reads label-token logits, it never parses text).
Letters come from ``render.label_letter`` so train/eval share one mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import render

HERE = Path(__file__).resolve().parent


def pair_to_chat(row: dict) -> dict:
    prompt = row["input"]  # committed pairs already carry the base-style render
    letter = render.label_letter(row["candidates"], row["label"])
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": letter},
        ]
    }


def convert(pairs_path: Path, out_path: Path) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = pair_to_chat(json.loads(line))
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    for name, out_name in (("pairs_train.jsonl", "sft/train.jsonl"), ("pairs_heldout.jsonl", "sft/valid.jsonl")):
        src = args.data_dir / name
        dst = args.data_dir / out_name
        count = convert(src, dst)
        print(f"{src} -> {dst} ({count} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
