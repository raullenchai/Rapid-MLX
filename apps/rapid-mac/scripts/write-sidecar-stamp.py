#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a packaged sidecar's provenance stamp.

Called by ``build-sidecar.sh``. The stamp is what
``rapid_mlx.community_bench.run_builder.resolve_provenance`` reads instead of
probing Git at run time, so it has to be right: a benchmark record says which
build produced it, and a packaged build cannot work that out for itself on a
user's Mac.

Two kinds of build, and the difference matters:

``release``
    An official build, identified by ``RAPID_MLX_OFFICIAL_RELEASE=1``.
    ``execution-config.schema.json`` forbids ``rapid_mlx_revision`` on a
    release — it is identified by its version — so no revision is written.

``source``
    Anything else, including every local ``bash scripts/build.sh``. It MUST
    carry a valid 40-character revision; without one its results cannot be
    attributed to anything, and calling it a release would attribute them to
    the wrong thing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

#: The closed schema, loaded by path so this script needs no package imports.
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


def flag(name: str, value: str) -> bool:
    """A build flag, accepted only as exactly ``"0"`` or ``"1"``.

    Not truthiness. ``official`` was compared with ``== "1"`` and anything else
    fell through to the source branch, so ``"true"`` — the spelling a shell
    author is most likely to reach for — quietly produced a source build on a
    release lane. ``dirty`` had the mirror-image bug: ``"true"`` and ``"bogus"``
    were both silently clean, which is the dangerous direction.

    A shell caller that cannot produce ``0`` or ``1`` has a bug, and this is
    where it should surface.
    """

    if value not in ("0", "1"):
        raise SystemExit(
            f"ERR: {name} must be exactly '0' or '1', got {value!r}; refusing "
            "to guess what was meant"
        )
    return value == "1"


def is_exact_revision(value: str) -> bool:
    """Exactly 40 lowercase hexadecimal characters — no normalisation.

    This deliberately does not strip or lowercase. The revision reaches the
    stamp verbatim, the readers enforce the same exact form, and a writer that
    quietly repaired ``" ABC…"`` into ``"abc…"`` would be writing a document
    its caller never produced — which is precisely the class of silent repair
    the provenance chain exists to eliminate.
    """

    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def build_stamp(official: str, revision: str, dirty: str) -> dict[str, object]:
    """The stamp for one build, from the raw command-line arguments.

    Every argument is validated *before* a document is constructed. Validating
    only the finished document cannot catch an input that was silently
    coerced on the way in: a malformed ``dirty`` produced a perfectly valid
    clean-source stamp, and the schema had nothing to object to.
    """

    is_official = flag("official", official)
    is_dirty = flag("dirty", dirty)

    # Validated for BOTH branches, before either is chosen. The official branch
    # used to return its fixed document first and never look at the revision,
    # so `""`, `"garbage"` and an uppercase sha all produced a clean release
    # stamp. The release document does not *carry* the revision, but the caller
    # still had to resolve one to get here — `build-sidecar.sh` computes it to
    # decide whether the tree is clean — and a caller that could not produce a
    # real commit has a bug this is the last chance to catch. Accepting
    # nonsense for an argument the output ignores is how "the output looks
    # fine" and "the inputs were meaningless" coexist.
    if not is_exact_revision(revision):
        raise SystemExit(
            f"ERR: refusing to stamp a build whose revision {revision!r} is not "
            "exactly 40 lowercase hexadecimal characters; its benchmark records "
            "could not be attributed"
        )

    if is_official:
        # Defence in depth. ``build-sidecar.sh`` already refuses this, but a
        # release stamp carries no revision and therefore no evidence of the
        # tree it came from — if that check is ever bypassed, reordered, or
        # copied into another script, the lie becomes unrecoverable. Refusing
        # here means the dishonest stamp cannot be written at all.
        if is_dirty:
            raise SystemExit(
                "ERR: refusing to stamp an official release built from a "
                "modified working tree; a release stamp cannot describe a "
                "tree that is not a commit"
            )
        # The validated revision is deliberately NOT carried:
        # `execution-config.schema.json` forbids `rapid_mlx_revision` on a
        # release, and the closed schema requires a release document to be
        # exactly this. Validation and content are separate questions.
        return {"distribution": "release"}

    stamp: dict[str, object] = {"distribution": "source", "revision": revision}
    if is_dirty:
        # Recorded so a reader can tell that the tree differed from the named
        # commit. The record is still attributable to that commit.
        stamp["dirty"] = True
    return stamp


def validated(stamp: dict[str, object]) -> dict[str, object]:
    """Refuse to write a document the readers would reject.

    The writer and the readers are the two halves of the same contract; a
    stamp that only the writer accepts is a file that fails at run time on a
    user's Mac instead of here.
    """

    schema = _schema()
    try:
        return schema.validate(stamp, label="the build stamp")
    except schema.ProvenanceInvalid as exc:  # pragma: no cover - guard
        raise SystemExit(f"ERR: refusing to write an invalid stamp: {exc}") from exc


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        raise SystemExit(
            "usage: write-sidecar-stamp.py <path> <official> <revision> <dirty>"
        )
    path, official, revision, dirty = argv[1:5]
    stamp = validated(build_stamp(official, revision, dirty))
    Path(path).write_text(json.dumps(stamp, sort_keys=True) + "\n", encoding="utf-8")
    print(f"    stamped {json.dumps(stamp, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main(sys.argv))
