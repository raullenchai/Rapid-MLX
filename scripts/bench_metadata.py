# SPDX-License-Identifier: Apache-2.0
"""Bench JSON metadata and compatibility helpers.

Every ``scripts/bench_*.py`` JSON artifact carries two top-level keys:

* ``_schema_version`` — an integer schema version for the artifact shape.
* ``_methodology_hash`` — a SHA-256 of the emitting bench script's bytes,
  so a methodology change forces a re-bench instead of silently merging
  old and new numbers.

Consumers that classify or aggregate bench JSON must call
``assert_compatible`` before merging payloads. The explicit backward
compatibility policy is:

* ``SCHEMA_VERSION`` is the current artifact schema. It starts at ``1``.
* A payload with ``_schema_version`` equal to ``SCHEMA_VERSION`` is
  compatible with any other payload of the same version.
* A payload with ``_schema_version`` missing or not an integer is
  rejected: it cannot be trusted to mean anything.
* A payload with ``_schema_version`` different from ``SCHEMA_VERSION``
  is rejected. There is no silent cross-version merge. If a future
  schema is additive-only, bump ``SCHEMA_VERSION`` and update this
  policy explicitly; old artifacts remain readable but are not merged
  into new aggregates.
* ``_methodology_hash`` must be present and equal across all payloads
  being merged. A different hash means the bench script changed, so the
  numbers were produced under a different methodology and must not be
  combined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
JSON_BENCH_SCRIPTS = frozenset(
    {
        "bench_attention.py",
        "bench_decode_tps.py",
        "bench_deepseek_restart_cache.py",
        "bench_dflash.py",
        "bench_diffusion_gemma.py",
        "bench_engine_solo.py",
        "bench_service_prefill.py",
        "bench_readme_refresh.py",
        "bench_suffix_decoding.py",
        "bench_suffix_decoding_integrated.py",
        "bench_vs_ollama.py",
    }
)
NON_JSON_BENCH_SCRIPTS = frozenset(
    {
        "bench_cache_perf.py",
        "bench_detok.py",
        "bench_kv_cache.py",
        "bench_metadata.py",
        "bench_snapshot.py",
    }
)

_MISSING = object()


def methodology_hash_for_script(script_path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``script_path``'s bytes.

    The hash is over the exact file bytes so any edit to the bench
    script — prompt, workload, thresholds, measurement code — changes
    the methodology hash and forces a re-bench.
    """
    return hashlib.sha256(Path(script_path).read_bytes()).hexdigest()


def add_bench_metadata(
    payload: dict[str, Any], script_path: str | Path
) -> dict[str, Any]:
    """Add ``_schema_version`` and ``_methodology_hash`` to ``payload``.

    Returns a stamped copy. Existing keys are overwritten in the copy:
    metadata is authoritative and cannot be spoofed by a payload body.
    """
    stamped = dict(payload)
    stamped["_schema_version"] = SCHEMA_VERSION
    stamped["_methodology_hash"] = methodology_hash_for_script(script_path)
    return stamped


def format_bench_json(
    payload: dict[str, Any],
    script_path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    default: Any = None,
) -> str:
    """Stamp and serialize one benchmark artifact."""
    script_name = Path(script_path).name
    if script_name not in JSON_BENCH_SCRIPTS:
        raise ValueError(
            f"unregistered benchmark JSON emitter: {script_name}; "
            "update JSON_BENCH_SCRIPTS"
        )
    return json.dumps(
        add_bench_metadata(payload, script_path),
        indent=indent,
        sort_keys=sort_keys,
        default=default,
    )


def write_bench_json(
    destination: str | Path,
    payload: dict[str, Any],
    script_path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    default: Any = None,
) -> None:
    """Stamp and write one benchmark artifact with a trailing newline."""
    Path(destination).write_text(
        format_bench_json(
            payload,
            script_path,
            indent=indent,
            sort_keys=sort_keys,
            default=default,
        )
        + "\n",
        encoding="utf-8",
    )


def _schema_version(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _MISSING
    return payload.get("_schema_version", _MISSING)


def _methodology_hash(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _MISSING
    return payload.get("_methodology_hash", _MISSING)


def assert_compatible(*payloads: Any) -> None:
    """Raise ``ValueError`` if any payload is not merge-compatible.

    Compatibility requires every payload to be a dict carrying the
    current ``_schema_version`` and a non-empty ``_methodology_hash``,
    and all payloads to share the same ``_methodology_hash``.
    """
    if not payloads:
        return
    first_hash: str | None = None
    for index, payload in enumerate(payloads):
        label = f"payload[{index}]"
        version = _schema_version(payload)
        if version is _MISSING:
            raise ValueError(f"{label} is missing _schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"{label} has non-integer _schema_version: {version!r}")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{label} has _schema_version {version!r}; expected "
                f"{SCHEMA_VERSION}. Methodology changes force a re-bench."
            )
        digest = _methodology_hash(payload)
        if digest is _MISSING:
            raise ValueError(f"{label} is missing _methodology_hash")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
        ):
            raise ValueError(f"{label} has invalid SHA-256 _methodology_hash")
        digest = digest.lower()
        if first_hash is None:
            first_hash = digest
        elif digest != first_hash:
            raise ValueError(
                f"{label} has _methodology_hash {digest!r}; expected "
                f"{first_hash!r}. Bench scripts differ; refusing to merge."
            )


def load_compatible_bench_json(*paths: str | Path) -> list[dict[str, Any]]:
    """Load artifacts only when their schema and methodology are compatible."""
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    assert_compatible(*payloads)
    return payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSON before aggregation or classification."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    payloads = load_compatible_bench_json(*args.artifacts)
    print(f"compatible benchmark artifacts: {len(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
