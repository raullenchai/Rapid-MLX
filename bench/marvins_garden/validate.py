# SPDX-License-Identifier: Apache-2.0
"""Validate Marvin's Garden contrastive pair files (fail-closed).

Mirrors the community-benchmarks validator contract:

* ``jsonschema`` is MANDATORY. If it is not installed every file FAILs with
  an install hint — the earlier "warn and skip" fallback demoted the
  load-bearing schema gate to a no-op (see community validator, PR #582).
* Layered checks: JSON parse -> schema -> semantic checks (label in
  candidates, pair integrity, flip_key known for the family).
* Exit code is the number of failed files (capped at 125). 0 = clean.

Usage::

    python bench/marvins_garden/validate.py                 # default data dir
    python bench/marvins_garden/validate.py FILE.jsonl ...  # explicit files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FILES = [HERE / "data" / "pairs_train.jsonl", HERE / "data" / "pairs_heldout.jsonl"]
SCHEMA_PATH = HERE / "schema.json"

KNOWN_FAMILIES = {
    "model_routing": {"context_tokens", "host_ram_gb", "needs_vision", "task_type"},
    "tool_gate": {"available_tools", "tool_rounds_left", "request"},
    "injection_guard": {"override_line", "tag_line", "exfil_line", "code_note"},
}


class _IssueError(Exception):
    """json: non-finite number is not permitted (mirrors community validator)."""


def _reject_non_finite(constant: str) -> None:
    raise _IssueError(f"json: non-finite number ({constant}) is not permitted")


def _load_schema():
    import jsonschema

    return jsonschema.Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )


def validate_file(path: Path, schema_validator) -> list[str]:
    """Return a list of problems; empty list means the file is clean."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unreadable: {exc}"]
    seen_groups: dict[str, dict[str, str]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            row = json.loads(line, parse_constant=_reject_non_finite)
        except _IssueError as exc:
            problems.append(f"{where} ERROR {exc}")
            continue
        except json.JSONDecodeError as exc:
            problems.append(f"{where} ERROR not valid JSON: {exc}")
            continue
        for error in sorted(schema_validator.iter_errors(row), key=lambda e: e.json_path):
            problems.append(f"{where} ERROR schema: {error.message}")
        label = row.get("label", "")
        candidates = row.get("candidates", [])
        if label and candidates and label not in candidates:
            problems.append(f"{where} ERROR label {label!r} not in candidates")
        family = row.get("family", "")
        flip_key = row.get("flip_key", "")
        if family and flip_key and flip_key not in KNOWN_FAMILIES.get(family, set()):
            problems.append(f"{where} ERROR flip_key {flip_key!r} not audited for family {family!r}")
        group = row.get("contrast_group", "")
        pair_id = row.get("pair_id", "")
        if group and not pair_id.startswith(group + "-"):
            problems.append(f"{where} ERROR pair_id {pair_id!r} does not sit under contrast_group {group!r}")
        if group:
            slot = seen_groups.setdefault(group, {})
            slot[pair_id.rsplit("-", 1)[-1]] = label
    # Contrast integrity across the whole file.
    for group, members in sorted(seen_groups.items()):
        if sorted(members) != ["a", "b"]:
            problems.append(f"{path.name}: contrast group {group} has members {sorted(members)}, expected [a, b]")
        if len(set(members.values())) != 2:
            problems.append(
                f"{path.name}: contrast group {group} labels do not flip ({sorted(set(members.values()))})"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = [Path(a) for a in argv] or DEFAULT_FILES
    try:
        validator = _load_schema()
    except ImportError:
        print(
            "  ERROR: jsonschema is mandatory (fail-closed). "
            "Install it with: pip install 'jsonschema>=4.0'",
            file=sys.stderr,
        )
        return min(125, len(targets))
    failures = 0
    for path in targets:
        problems = validate_file(path, validator)
        if problems:
            failures += 1
            print(f"FAIL {path}")
            for problem in problems[:20]:
                print(f"  {problem}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
        else:
            print(f"PASS {path}")
    total = len(targets)
    print(f"  {total - failures}/{total} files passed.")
    return min(125, failures)


if __name__ == "__main__":
    sys.exit(main())
