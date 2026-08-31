#!/usr/bin/env python3
"""Copy public proto contracts into the Python wheel and verify drift."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "vllm_mlx" / "catalog" / "schemas"
SOURCES = (
    ROOT / "proto" / "model-runtime" / "v1" / "model-identity.schema.json",
    ROOT / "proto" / "model-runtime" / "v1" / "machine-observation.schema.json",
    ROOT / "proto" / "model-runtime" / "v1" / "execution-config.schema.json",
    ROOT / "proto" / "model-catalog" / "v1" / "model-alias.schema.json",
    ROOT / "proto" / "model-catalog" / "v1" / "model-registry-record.schema.json",
    ROOT / "proto" / "model-catalog" / "v1" / "recommendation-policy.schema.json",
    ROOT / "proto" / "model-catalog" / "v1" / "catalog-snapshot.schema.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = [
        source
        for source in SOURCES
        if not (DESTINATION / source.name).exists()
        or source.read_bytes() != (DESTINATION / source.name).read_bytes()
    ]
    if args.check:
        if drift:
            print(
                "generated catalog schemas are stale: "
                + ", ".join(path.name for path in drift)
            )
            return 1
        return 0
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        shutil.copyfile(source, DESTINATION / source.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
