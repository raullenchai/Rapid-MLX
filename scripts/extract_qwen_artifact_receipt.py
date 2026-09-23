#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract a content-free Qwen artifact receipt from an existing HF snapshot.

This script is intentionally offline: it validates the canonical Hub-cache
binding, reads config/index metadata and filesystem receipts, and writes JSON
to stdout. It never downloads a model or opens safetensors content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapid_mlx.runtime.qwen_artifact import (
    probe_qwen_artifact,
    verify_hub_snapshot_binding,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--subfolder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = verify_hub_snapshot_binding(
        args.snapshot_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        subfolder=args.subfolder,
    )
    if binding is None:
        raise SystemExit("snapshot is not a verified canonical HF cache binding")
    truth = probe_qwen_artifact(args.snapshot_dir, binding=binding)
    print(json.dumps(truth.to_receipt_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
