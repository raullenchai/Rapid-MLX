# SPDX-License-Identifier: Apache-2.0
"""The one definition of a valid provenance document.

A provenance document says which build produced a benchmark. Four places read
or write one — the packaged build stamp, the stamp verifier that gates a
release, the per-run record stored beside an archived result, and the
publication check — and until now each applied its own partial rules. Partial
rules on a security-relevant document compose badly: ``resolve_provenance``
silently dropped a revision from a release stamp, and
``workspace.validate_provenance`` accepted unknown fields, so a document
rejected by one reader was quietly repaired by another.

So the schema is **closed** and lives here, with no dependencies beyond the
standard library, so the build scripts can load it by path without importing
the package.

Exactly two shapes are valid:

``{"distribution": "release"}``
    Nothing else. A release is identified by its version;
    ``execution-config.schema.json`` forbids ``rapid_mlx_revision`` on one, and
    a release built from a modified tree is refused at build time, so neither
    ``revision`` nor ``dirty`` has any meaning here. A release stamp carrying
    one is not a release stamp with extra detail — it is a document whose
    producer disagrees with this one, and guessing which half to believe is
    how a modified build ships as official.

``{"distribution": "source", "revision": <40 lowercase hex>[, "dirty": bool]}``
    A build from a checkout, naming the commit it was built from.

Nothing is sanitized. An invalid document raises; it is never repaired into a
valid one, because the repaired version would assert something its producer
never said.
"""

from __future__ import annotations

from typing import Any

#: The distributions the execution-config contract defines.
DISTRIBUTIONS = ("release", "source")

#: Exactly the keys each shape may carry.
RELEASE_KEYS = frozenset({"distribution"})
SOURCE_REQUIRED_KEYS = frozenset({"distribution", "revision"})
SOURCE_OPTIONAL_KEYS = frozenset({"dirty"})
SOURCE_KEYS = SOURCE_REQUIRED_KEYS | SOURCE_OPTIONAL_KEYS


class ProvenanceInvalid(ValueError):  # noqa: N818 - stable domain error name
    """A provenance document does not satisfy the closed schema."""


def is_revision(value: Any) -> bool:
    """A 40-character lowercase hex sha, exactly.

    Not normalised here: an uppercase sha in a stored document means the
    producer wrote something this schema does not define, and quietly
    lowercasing it would hide that.
    """

    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate(document: Any, *, label: str = "the build provenance") -> dict[str, Any]:
    """Return ``document`` unchanged, or raise :class:`ProvenanceInvalid`.

    Returning the input rather than a cleaned copy is deliberate: a caller can
    use the result directly and know it is exactly what was on disk.
    """

    if not isinstance(document, dict):
        raise ProvenanceInvalid(f"{label} is not an object")

    distribution = document.get("distribution")
    if distribution not in DISTRIBUTIONS:
        raise ProvenanceInvalid(
            f"{label} names an unknown distribution {distribution!r}"
        )

    keys = set(document)
    if distribution == "release":
        unexpected = sorted(keys - RELEASE_KEYS)
        if unexpected:
            raise ProvenanceInvalid(
                f"{label} is a release but carries {unexpected}; a release "
                "stamp must contain exactly {'distribution': 'release'}"
            )
        return document

    unexpected = sorted(keys - SOURCE_KEYS)
    if unexpected:
        raise ProvenanceInvalid(f"{label} carries unknown fields {unexpected}")
    missing = sorted(SOURCE_REQUIRED_KEYS - keys)
    if missing:
        raise ProvenanceInvalid(f"{label} is a source build but is missing {missing}")
    if not is_revision(document["revision"]):
        raise ProvenanceInvalid(
            f"{label} is a source build whose revision {document['revision']!r} "
            "is not a 40-character lowercase sha"
        )
    # Present or absent, and strictly boolean. The string "true" is the one
    # that matters: read as a flag it is truthy, read as a value it is not
    # `True`, so a lenient reader calls a dirty build clean.
    if "dirty" in document and not isinstance(document["dirty"], bool):
        raise ProvenanceInvalid(
            f"{label} has a non-boolean dirty flag {document['dirty']!r}"
        )
    return document


def is_dirty(document: dict[str, Any]) -> bool:
    """Whether a **validated** document describes a modified tree."""

    return document.get("dirty") is True


__all__ = [
    "DISTRIBUTIONS",
    "ProvenanceInvalid",
    "RELEASE_KEYS",
    "SOURCE_KEYS",
    "SOURCE_OPTIONAL_KEYS",
    "SOURCE_REQUIRED_KEYS",
    "is_dirty",
    "is_revision",
    "validate",
]
