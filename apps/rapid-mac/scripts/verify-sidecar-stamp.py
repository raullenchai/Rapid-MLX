#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the provenance stamp inside a built ``Rapid-MLX Desktop.app``.

The stamp is written several layers down (``build.sh`` → ``build-sidecar.sh``
→ ``write-sidecar-stamp.py``) and decides what every benchmark the shipped app
publishes claims about its own origin. An environment variable that silently
failed to reach that depth would produce a plausible-looking app that lies
about where it came from, and nothing downstream would notice.

So the finished bundle is inspected, and the answer is checked against what
the caller said it was building:

* ``official_release=true``  → ``{"distribution": "release"}``, and **no**
  revision (``execution-config.schema.json`` forbids one on a release).
* ``official_release=false`` → ``{"distribution": "source", "revision": <sha>}``
  with a 40-character lowercase revision, optionally ``dirty``.

Exits non-zero with an explanation on any mismatch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

STAMP_RELATIVE = Path(
    "Contents/Resources/rapid-mlx/site-packages/rapid_mlx/_build_stamp.json"
)

#: The closed schema, loaded by path rather than imported as
#: ``rapid_mlx.community_bench.provenance_schema``: importing the package pulls
#: in the catalog and its third-party dependencies, which a build runner has no
#: reason to have. One definition either way.
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "rapid_mlx/community_bench/provenance_schema.py"
)


def _schema():
    spec = importlib.util.spec_from_file_location("provenance_schema", _SCHEMA_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise SystemExit(f"cannot load the provenance schema from {_SCHEMA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(stamp: dict, official: bool) -> list[str]:
    """Problems with this stamp, given what was meant to be built.

    Two independent questions. First: is this a valid provenance document at
    all? That is the closed schema, and it is what rejects a revision or a
    ``dirty`` flag on a release, a non-boolean ``dirty``, and any unknown
    field. Second: does it describe the build the caller said it was making?
    """

    schema = _schema()
    problems: list[str] = []
    try:
        schema.validate(stamp, label="the build stamp")
    except schema.ProvenanceInvalid as exc:
        problems.append(str(exc))

    distribution = stamp.get("distribution")
    expected = "release" if official else "source"
    if distribution != expected:
        problems.append(
            f"distribution is {distribution!r}, expected {expected!r}"
            + (
                " — an official release was requested but the build stamped "
                "itself as a source build"
                if official
                else " — this is not an official release lane, so it must not "
                "claim to be a release"
            )
        )
    if not official and not problems and schema.is_dirty(stamp):
        # Not a build failure: a dirty CI checkout is worth shouting about but
        # the artifact is still honest about itself, and its benchmarks simply
        # cannot be published.
        print(
            "::warning::sidecar was built from a modified working tree; "
            "benchmarks from this app cannot be published",
            file=sys.stderr,
        )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(
            "usage: verify-sidecar-stamp.py <app-path> <official:true|false>"
        )
    app, official_raw = Path(argv[1]), argv[2]
    if official_raw not in ("true", "false"):
        raise SystemExit(f"official must be 'true' or 'false', got {official_raw!r}")
    official = official_raw == "true"

    path = app / STAMP_RELATIVE
    if not path.is_file():
        raise SystemExit(f"no provenance stamp at {path}")
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"provenance stamp is not readable JSON: {exc}") from exc
    if not isinstance(stamp, dict):
        raise SystemExit("provenance stamp is not a JSON object")

    problems = verify(stamp, official)
    if problems:
        for problem in problems:
            print(f"::error::sidecar provenance: {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(f"sidecar provenance verified: {json.dumps(stamp, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main(sys.argv))
