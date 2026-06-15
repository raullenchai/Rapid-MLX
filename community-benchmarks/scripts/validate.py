#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one or more community benchmark submission files.

Run modes:

    # Validate every JSON file under submissions/
    python community-benchmarks/scripts/validate.py

    # Validate specific files (used by the GHA on PR diff)
    python community-benchmarks/scripts/validate.py path/to/sub.json ...

Layered checks, in order:

1. **JSON parse** — file must decode.
2. **Schema** — must match ``community-benchmarks/schema.json``
   (draft 2020-12, ``additionalProperties: false`` everywhere). The
   const fields on ``config.rounds`` etc. are enforced here.
3. **Whitelist** — ``model.alias`` must exist in
   ``vllm_mlx/aliases.json`` and ``model.hf_path`` must match the
   value stored there. We re-check after the CLI's own whitelist
   guard because the JSON file in a PR is the authoritative artifact;
   anything else is just history.
4. **Sanity** — decode_tps > 0, ttft_ms < 30 s, peak_ram_mb <= chip's
   RAM, chip non-empty, etc. These don't catch every fraud, but they
   cut the noise floor so a real outlier still gets reviewer attention.

Exit code is the number of failed files (capped at 125 so it fits in a
shell exit status). 0 = all clean. The GHA fails the job on non-zero.

Designed to run with stdlib only when ``jsonschema`` isn't installed —
in that case schema validation is skipped with a clear warning. The
GHA installs ``jsonschema`` explicitly, so CI always runs the full
check; local invocations stay friction-free.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
ALIASES_PATH = REPO_ROOT / "vllm_mlx" / "aliases.json"
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"

# Sanity bounds. Wider than any realistic Apple Silicon number so a
# legitimate outlier still passes — these only catch obvious tampering
# (1e9 decode_tps, negative TTFT) and unit confusion.
MAX_DECODE_TPS = 2_000.0  # M3 Ultra single-line tops out ~140 in the wild
MAX_PREFILL_TPS = 50_000.0  # llama.cpp pp can hit ~10k on Ultra; cap well above
MAX_TTFT_MS = 30_000.0  # 30 s is well past "the model failed to load"
MAX_RAM_GB = 1024
# Filename pattern: <YYYYMMDD>-<chip-slug>-<alias-slug>-<id>.json. We
# don't enforce the exact slugs (chip names change) — just the shape.
FILENAME_RE = re.compile(r"^[0-9]{8}-[a-z0-9-]+-[a-z0-9.-]+-[0-9a-f]{12}\.json$")


class _IssueError(Exception):
    """Validation failure with a single human-readable line."""


def _load_schema() -> dict | None:
    if not SCHEMA_PATH.exists():
        print(f"  WARN: schema not found at {SCHEMA_PATH}; skipping schema check")
        return None
    return json.loads(SCHEMA_PATH.read_text())


def _load_aliases() -> dict[str, dict]:
    """Read ``vllm_mlx/aliases.json`` directly — no engine import needed."""
    if not ALIASES_PATH.exists():
        return {}
    raw = json.loads(ALIASES_PATH.read_text())
    return raw if isinstance(raw, dict) else {}


def _check_schema(payload: dict, schema: dict | None) -> None:
    """Raise ``_IssueError`` if the payload doesn't match the JSON Schema.

    Falls back to a no-op when ``jsonschema`` isn't installed locally
    (the GHA pins it, so CI always runs the strict path).
    """
    if schema is None:
        return
    try:
        import jsonschema
    except ImportError:
        print("  WARN: jsonschema not installed; skipping schema check")
        return
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as e:
        # Take just the first failure — jsonschema's full path is
        # already in the message, and reviewers don't need a stack.
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise _IssueError(f"schema: {loc}: {e.message}") from None


def _check_alias_whitelist(payload: dict, aliases: dict[str, dict]) -> None:
    alias = payload.get("model", {}).get("alias")
    hf_path = payload.get("model", {}).get("hf_path")
    if not alias:
        raise _IssueError("alias: missing model.alias")
    if alias not in aliases:
        raise _IssueError(
            f"alias: '{alias}' is not on the whitelist "
            f"(vllm_mlx/aliases.json). Register it there first."
        )
    expected_path = aliases[alias].get("hf_path")
    if expected_path and hf_path != expected_path:
        raise _IssueError(
            f"alias: model.hf_path '{hf_path}' does not match registered "
            f"hf_path for '{alias}' (expected '{expected_path}')"
        )


def _check_sanity(payload: dict) -> None:
    """Plausibility bounds. Each violation raises ``_IssueError``."""
    hw = payload.get("hardware", {})
    if not hw.get("chip"):
        raise _IssueError("hardware: chip is empty")
    if not (1 <= hw.get("ram_gb", 0) <= MAX_RAM_GB):
        raise _IssueError(f"hardware: ram_gb out of range: {hw.get('ram_gb')}")

    peak = payload.get("peak_ram_mb")
    if peak is not None:
        # 1 GB = 1024 MiB. peak ≤ total RAM (some slack for shared GPU).
        if peak > hw["ram_gb"] * 1024 * 2:
            raise _IssueError(
                f"hardware: peak_ram_mb={peak} exceeds 2× total RAM ({hw['ram_gb']} GB)"
            )

    for bucket_name in ("short", "long"):
        b = payload["buckets"][bucket_name]
        for stat_field in ("decode_tps", "prefill_tps", "ttft_ms"):
            stat = b[stat_field]
            if stat["median"] < 0:
                raise _IssueError(f"buckets.{bucket_name}.{stat_field}: median < 0")
        if b["decode_tps"]["median"] > MAX_DECODE_TPS:
            raise _IssueError(
                f"buckets.{bucket_name}.decode_tps: median "
                f"{b['decode_tps']['median']:.1f} > {MAX_DECODE_TPS} (unrealistic)"
            )
        if b["prefill_tps"]["median"] > MAX_PREFILL_TPS:
            raise _IssueError(
                f"buckets.{bucket_name}.prefill_tps: median "
                f"{b['prefill_tps']['median']:.1f} > {MAX_PREFILL_TPS}"
            )
        if b["ttft_ms"]["median"] > MAX_TTFT_MS:
            raise _IssueError(
                f"buckets.{bucket_name}.ttft_ms: median "
                f"{b['ttft_ms']['median']:.1f} > {MAX_TTFT_MS} ms (likely a stuck request)"
            )


def _check_filename(path: Path) -> None:
    name = path.name
    if not FILENAME_RE.match(name):
        raise _IssueError(
            f"filename: '{name}' does not match "
            f"<YYYYMMDD>-<chip-slug>-<alias-slug>-<12hex>.json"
        )


def _check_path_in_submissions(path: Path) -> None:
    """Refuse files that aren't inside ``submissions/``.

    PRs editing other files in the same diff are fine — they just
    don't get fed to us. We're called explicitly with each submission
    file path, but a buggy GHA filter could feed us something else;
    this is the cheap belt-and-braces guard.
    """
    try:
        rel = path.relative_to(SUBMISSIONS_DIR.resolve())
    except ValueError:
        raise _IssueError(
            f"path: {path} is not inside community-benchmarks/submissions/"
        )
    if "/" in str(rel) or "\\" in str(rel):
        raise _IssueError(
            f"path: {path} contains a subdirectory; submissions must be flat"
        )


def validate_one(
    path: Path, schema: dict | None, aliases: dict[str, dict]
) -> list[str]:
    """Return the list of issues found for one file. Empty = OK."""
    issues: list[str] = []

    try:
        _check_path_in_submissions(path.resolve())
        _check_filename(path)
        payload = json.loads(path.read_text())
        _check_schema(payload, schema)
        _check_alias_whitelist(payload, aliases)
        _check_sanity(payload)
    except _IssueError as e:
        issues.append(str(e))
    except json.JSONDecodeError as e:
        issues.append(f"json: parse error: {e}")
    except (OSError, KeyError) as e:
        # KeyError surfaces "schema passed but we then asked for a field
        # the schema doesn't require" — that's a validator bug, not user
        # error, but it shouldn't silently pass either.
        issues.append(f"internal: {type(e).__name__}: {e}")
    return issues


def main(argv: list[str]) -> int:
    targets = (
        [Path(p) for p in argv[1:]]
        if len(argv) > 1
        else sorted(SUBMISSIONS_DIR.glob("*.json"))
    )
    if not targets:
        print("  No submission files to validate.")
        return 0

    schema = _load_schema()
    aliases = _load_aliases()
    if not aliases:
        print("  ERROR: aliases.json is empty or missing — every file will fail.")
        return min(125, len(targets))

    failures = 0
    for path in targets:
        issues = validate_one(path, schema, aliases)
        if issues:
            failures += 1
            print(f"  FAIL  {path.name}")
            for issue in issues:
                print(f"        {issue}")
        else:
            print(f"  OK    {path.name}")

    print()
    print(f"  {len(targets) - failures}/{len(targets)} files passed.")
    return min(125, failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
