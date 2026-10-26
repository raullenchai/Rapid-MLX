"""Replay frozen observations against a checkout, without importing its tests.

Usage: python tests/parser_migration/replay.py CHECKOUT baseline|main
Each mismatch is printed; a nonzero exit prohibits an unledgered change.
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in {"baseline", "main"}:
        raise ValueError("expected CHECKOUT baseline|main")
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    from harness import route

    data = json.loads(Path(__file__).with_name("differential.json").read_text())
    failures = []
    count = 0
    for row in data["rows"]:
        if row["revision"] != sys.argv[2]:
            continue
        reasoning, tool, _, thinking, mode, surface, stream, split = row["key"]
        raw = row["raw"]
        chunks = (
            list(raw)
            if split == "chars"
            else [raw]
            if split == "whole"
            else [raw[i : i + 7] for i in range(0, len(raw), 7)]
        )
        actual = route(
            reasoning, tool, raw, surface, stream, chunks, thinking, tools_mode=mode
        )
        count += 1
        if actual != row["result"]:
            failures.append(
                {"key": row["key"], "expected": row["result"], "actual": actual}
            )
    assert count > 0
    print(json.dumps({"cases": count, "differences": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
