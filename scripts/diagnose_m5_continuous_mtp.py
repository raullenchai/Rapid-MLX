#!/usr/bin/env python3
"""Locate a token mismatch in the existing continuous-MTP benchmark."""

import argparse
import json
import re
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bench import bench_spec_decode_mtp as bench

    original = bench.hashlib.sha256
    observations = []
    for mode in ["none", "mtp", "none"]:
        receipts = []

        def record(data=b"", *, _receipts=receipts, **kwargs):
            if isinstance(data, bytes) and re.fullmatch(
                rb"[0-9]+(?:,[0-9]+){8,}", data
            ):
                _receipts.append([int(t) for t in data.split(b",")])
            return original(data, **kwargs)

        with patch.object(bench.hashlib, "sha256", side_effect=record):
            row = bench._run_once(
                model_alias=args.model,
                condition=mode,
                prompt=bench._BENCH_PROMPTS[0],
                max_tokens=192,
                temp=0,
                mtp_sidecar=args.drafter,
            )
        if not receipts or len(receipts[-1]) != row.n_tokens:
            raise RuntimeError("missing full token audit")
        observations.append(
            {"mode": mode, "tokens": receipts[-1], "sha256": row.token_sha256}
        )
    ar, mtp, repeat = [r["tokens"] for r in observations]
    first = next((i for i, (a, b) in enumerate(zip(ar, mtp)) if a != b), None)
    if first is None and len(ar) != len(mtp):
        first = min(len(ar), len(mtp))
    result = {
        "observations": observations,
        "first_mismatch_index": first,
        "stock_repeat_exact": ar == repeat,
        "stock_vs_mtp_exact": ar == mtp,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}))


if __name__ == "__main__":
    main()
