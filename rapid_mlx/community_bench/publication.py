# SPDX-License-Identifier: Apache-2.0
"""What leaves this Mac, as distinct from what the local archive keeps.

A benchmark record and a benchmark *submission* are not the same document, and
conflating them is what made normal runs unpublishable.

``unresolved_model_identity`` fills every fact the local Hugging Face cache can
vouch for: the ``subfolder`` a nested variant lives in, the
``resolved_revision`` the loader actually read, and real ``quantization``
facts (``{"kind": "weights", "method": "affine", "weight_bits_x2": 8, ...}``).
That is the right content for an archive on the user's own disk — it is what
makes a stored result reproducible months later.

The deployed ingestion validator accepts none of it. ``atomicValidateModel``
allowlists ``model.components[0].source`` to exactly ``{kind, repo_id}`` and
requires ``quantization`` to be exactly ``{"kind": "unknown", "base_dtype":
"unknown"}``, rejecting anything else with ``… is not upload-allowlisted``.
Since :func:`atomic_upload.preview_run` sent the archived record verbatim, a
model loaded from a warm cache — the ordinary case — could be measured but
never published.

So the wire document is **projected** from the record rather than copied from
it. Two rules govern the projection:

1. It only ever *removes* facts. It never invents, rounds, or substitutes one,
   so a field that survives means exactly what the archive says it means.
2. Nothing is removed silently. :func:`project_run_for_publication` returns
   what it withheld, the consent prompt prints it, and ``--preview --json``
   carries it so a GUI can show the same thing before the user agrees.

When the service widens its allowlist this module gets smaller, and the local
record does not change at all.
"""

from __future__ import annotations

import copy
from typing import Any

from .workspace import ProvenanceUnreadable, validate_provenance

#: Keys the ingestion validator allows on ``model.components[0].source``.
#: Mirrors ``atomicExactObject(component.source, ["kind", "repo_id"], …)``.
_PUBLIC_SOURCE_KEYS = ("kind", "repo_id")

#: The only quantization block the validator accepts. Mirrors
#: ``component.quantization.kind !== "unknown" || … base_dtype !== "unknown"``.
_PUBLIC_QUANTIZATION: dict[str, str] = {"kind": "unknown", "base_dtype": "unknown"}


class PublicationRefused(Exception):  # noqa: N818 - stable domain error name
    """This result must not be published, and why.

    Distinct from a validation error: the payload could be *made* to validate
    by dropping the objection, and dropping it is exactly what must not
    happen.
    """


def ensure_publishable(provenance: dict[str, Any] | None) -> None:
    """Refuse to publish a result measured by a modified build.

    ``execution.runtime`` on the wire is an exact allowlist —
    ``distribution``, versions, and a 40-character ``rapid_mlx_revision`` —
    so the only thing a submission can say about its own code is "this
    commit". A build whose working tree differed from that commit would be
    published under the clean commit's identity, attributing numbers produced
    by modified benchmark code to code that never produced them.

    The tree cannot be represented, so the result is not published. It is
    still measured, still saved, and still readable locally; only the claim to
    the outside world is refused. Rebuilding cleanly does not rehabilitate an
    older result, because this reads the provenance stored *with that run*.

    ``None`` means no provenance was recorded (a run archived before this
    existed). That is not evidence of modification, so it is not an objection.
    Anything present is validated in full — distribution, revision shape and
    the type of ``dirty`` — because asking only ``dirty is True`` let a
    malformed document through as publishable.
    """

    if provenance is None:
        # Genuinely absent: a run archived before provenance existed. The
        # caller has already distinguished this from an unreadable file.
        return
    try:
        validate_provenance(provenance)
    except ProvenanceUnreadable as exc:
        # A damaged record is not a legacy record. Refusing is the only
        # reading that does not let corruption authorise a publication.
        raise PublicationRefused(
            f"{exc}. This result's build provenance cannot be verified, so it "
            "cannot be published. The result is saved on this Mac and can be "
            "inspected; re-run the benchmark to record it again."
        ) from exc
    if provenance.get("dirty") is not True:
        return
    revision = provenance.get("revision")
    named = f" ({revision[:12]}…)" if isinstance(revision, str) else ""
    raise PublicationRefused(
        "this result was measured by a build whose working tree differed from "
        f"its commit{named}, so publishing it would attribute these numbers to "
        "code that did not produce them. The Community Benchmark can only "
        "name a commit, not a modified tree. The result is saved on this Mac "
        "and can be inspected; to publish, commit your changes, rebuild, and "
        "measure again."
    )


class WithheldFact:
    """One fact the archive holds that the public submission may not carry."""

    __slots__ = ("path", "value", "reason")

    def __init__(self, path: str, value: Any, reason: str) -> None:
        self.path = path
        self.value = value
        self.reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "reason": self.reason}

    def describe(self) -> str:
        return f"{self.path} = {self.value!r} ({self.reason})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WithheldFact({self.path!r}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WithheldFact):
            return NotImplemented
        return (self.path, self.value, self.reason) == (
            other.path,
            other.value,
            other.reason,
        )


_SOURCE_REASON = (
    "the benchmark service records models by repository id only; it rejects "
    "submissions carrying this field"
)
_QUANTIZATION_REASON = (
    "the benchmark service accepts only an unknown quantization block in this "
    "beta, so the measured artifact's quantization is not published"
)


def project_run_for_publication(
    run: dict[str, Any],
) -> tuple[dict[str, Any], list[WithheldFact]]:
    """Return ``(public_run, withheld)`` for one archived record.

    The input is never mutated: the archive on disk keeps every fact, and the
    returned document is a deep copy with the non-allowlisted ones removed.
    """

    public = copy.deepcopy(run)
    withheld: list[WithheldFact] = []
    model = public.get("model")
    if not isinstance(model, dict):
        return public, withheld
    components = model.get("components")
    if not isinstance(components, list):
        return public, withheld

    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        prefix = f"model.components[{index}]"
        source = component.get("source")
        if isinstance(source, dict):
            for key in sorted(set(source) - set(_PUBLIC_SOURCE_KEYS)):
                withheld.append(
                    WithheldFact(f"{prefix}.source.{key}", source[key], _SOURCE_REASON)
                )
                del source[key]
        quantization = component.get("quantization")
        if isinstance(quantization, dict) and quantization != _PUBLIC_QUANTIZATION:
            withheld.append(
                WithheldFact(
                    f"{prefix}.quantization",
                    copy.deepcopy(quantization),
                    _QUANTIZATION_REASON,
                )
            )
            component["quantization"] = dict(_PUBLIC_QUANTIZATION)
    return public, withheld


def describe_withheld(withheld: list[WithheldFact]) -> list[str]:
    """Lines for the consent prompt. Empty when nothing was withheld."""

    if not withheld:
        return []
    lines = [
        "These details stay on this Mac and are NOT in the payload above:",
    ]
    lines.extend(f"  - {fact.describe()}" for fact in withheld)
    lines.append("  Your local copy keeps them; only the submission is narrowed.")
    return lines


__all__ = [
    "PublicationRefused",
    "WithheldFact",
    "describe_withheld",
    "ensure_publishable",
    "project_run_for_publication",
]
