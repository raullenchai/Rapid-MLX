#!/usr/bin/env python3
"""Copy public proto contracts into the Python wheel and verify drift."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "vllm_mlx" / "catalog" / "schemas"
SOURCES = (
    (
        ROOT / "proto" / "model-runtime" / "v1" / "model-identity.schema.json",
        "model-identity.schema.json",
    ),
    (
        ROOT / "proto" / "model-runtime" / "v1" / "machine-observation.schema.json",
        "machine-observation.schema.json",
    ),
    (
        ROOT / "proto" / "model-runtime" / "v1" / "execution-config.schema.json",
        "execution-config.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v1" / "model-alias.schema.json",
        "model-alias-v1.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v2" / "model-alias.schema.json",
        "model-alias.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v1" / "model-registry-record.schema.json",
        "model-registry-record.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v1" / "recommendation-policy.schema.json",
        "recommendation-policy.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v1" / "catalog-snapshot.schema.json",
        "catalog-snapshot-v1.schema.json",
    ),
    (
        ROOT / "proto" / "model-catalog" / "v2" / "catalog-snapshot.schema.json",
        "catalog-snapshot.schema.json",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = [
        source
        for source, destination_name in SOURCES
        if not (DESTINATION / destination_name).exists()
        or source.read_bytes() != (DESTINATION / destination_name).read_bytes()
    ]
    if args.check:
        if drift:
            print(
                "generated catalog schemas are stale: "
                + ", ".join(str(path.relative_to(ROOT)) for path in drift)
            )
            return 1
        return 0
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for source, destination_name in SOURCES:
        shutil.copyfile(source, DESTINATION / destination_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
