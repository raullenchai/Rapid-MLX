#!/usr/bin/env python3
"""Validate owner-prepared local exports and emit contrastive roles. No network."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(manifest, calibration, heldout):
    specs = {s["name"]: s for s in manifest["slices"]}
    cal, held = calibration["slices"], heldout["slices"]
    expected = {n for n, s in specs.items() if not s["heldout_only"]}
    if set(cal) != expected or set(held) != set(specs):
        raise ValueError(
            "exports must cover exactly the manifest calibration/held-out slices"
        )
    seen = set()
    for split in (cal, held):
        for name, files in split.items():
            if not files:
                raise ValueError(f"empty slice {name}")
            if (
                sum(Path(file).stat().st_size for file in files)
                > manifest["max_export_bytes_per_slice"]
            ):
                raise ValueError("export slice exceeds byte cap")
            for file in files:
                path = Path(file)
                if (
                    not path.is_absolute()
                    or path.stat().st_size > manifest["max_export_bytes_per_slice"]
                ):
                    raise ValueError("exports require bounded absolute local paths")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    raise ValueError("duplicate file content across slices/splits")
                seen.add(digest)
    return {n: specs[n]["role"] for n in cal}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("deepseek_v41_rd_corpus_manifest.json"),
    )
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--output-roles", type=Path, required=True)
    a = p.parse_args()
    roles = prepare(
        json.loads(a.manifest.read_text()),
        json.loads(a.calibration.read_text()),
        json.loads(a.heldout.read_text()),
    )
    a.output_roles.write_text(json.dumps(roles, indent=2) + "\n")


if __name__ == "__main__":
    main()
