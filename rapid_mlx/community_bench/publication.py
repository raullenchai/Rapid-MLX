# SPDX-License-Identifier: Apache-2.0
"""Project archived runs onto the public ingestion contract.

Hugging Face revision, subfolder and quantization facts survive verbatim.
Local source fields stay on this Mac and are listed in the consent preview.
The archive is never mutated. Execution settings are sent truthfully; a
service that cannot store them rejects the upload.
"""

from __future__ import annotations

import copy
from typing import Any

from .workspace import ProvenanceUnreadable, validate_provenance

#: Optional provenance is public only for Hugging Face sources.
_PUBLIC_SOURCE_KEYS = ("kind", "repo_id")
_HF_SOURCE_KEYS = (*_PUBLIC_SOURCE_KEYS, "resolved_revision", "subfolder")


#: Wall-clock bound for the anonymous public-repository check.
_PUBLIC_PROOF_DEADLINE_S = 5.0


class PublicationRefused(Exception):  # noqa: N818 - stable domain error name
    """This result must not be published, and why.

    Distinct from a validation error: the payload could be *made* to validate
    by dropping the objection, and dropping it is exactly what must not
    happen.
    """


def ensure_public_model_sources(run: dict[str, Any]) -> None:
    """Require public identity for every component, including draft models.

    Cached proof has the telemetry classifier's lease and revocation rules.
    Otherwise query metadata anonymously: readable metadata alone is not
    proof for gated models, whose metadata can also be public.
    """
    from rapid_mlx.model_aliases import catalog_alias_for
    from rapid_mlx.telemetry import model_id

    checked: set[str] = set()
    for index, component in enumerate(run["model"]["components"]):
        source = component["source"]
        repo = source.get("repo_id")
        prefix = f"Cannot publish model.components[{index}]: "
        if (
            source.get("kind") != "huggingface"
            or not isinstance(repo, str)
            or model_id._looks_local(repo)
        ):
            raise PublicationRefused(prefix + "local-path models cannot be published.")
        if repo in checked:
            continue
        if catalog_alias_for(repo) is not None:
            checked.add(repo)
            continue
        if not model_id._HF_REPO_RE.fullmatch(repo):
            raise PublicationRefused(
                prefix + "the model has no public Hugging Face repository id."
            )
        # The telemetry proof store is not enough here: it records any
        # successful anonymous metadata read, and gated repositories expose
        # metadata anonymously. Publication needs ``private is False`` and
        # ``gated is False`` from a fresh anonymous read.
        try:
            if not model_id._hub_connection_can_prove_public():
                raise ValueError("untrusted Hub network configuration")
            from huggingface_hub import model_info

            from rapid_mlx._download_gate import call_with_deadline

            # httpx's timeout bounds each read, not the whole call; only a
            # wall-clock deadline keeps a trickling response from hanging.
            info = call_with_deadline(
                model_info, _PUBLIC_PROOF_DEADLINE_S, repo, token=False, timeout=5
            )
        except Exception as exc:
            model_id._revoke_proof(repo)
            # Do not echo library errors: they may carry URLs or credentials.
            raise PublicationRefused(
                prefix + "anonymous public access could not be verified. "
                "The repository may be private, gated, or unreachable. "
                "Nothing was uploaded; check your connection and try again."
            ) from exc
        if (
            getattr(info, "private", None) is not False
            or getattr(info, "gated", None) is not False
        ):
            model_id._revoke_proof(repo)
            raise PublicationRefused(
                prefix
                + "private, gated, or unproven repositories cannot be published. "
                "Nothing was uploaded; the result remains available locally."
            )
        model_id.note_hub_fetch(repo)
        checked.add(repo)


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
    "this source field is local or is not accepted by the benchmark service"
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
    components = model.get("components") if isinstance(model, dict) else None
    if not isinstance(components, list):
        components = []

    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        prefix = f"model.components[{index}]"
        source = component.get("source")
        if isinstance(source, dict):
            allowed = (
                _HF_SOURCE_KEYS
                if source.get("kind") == "huggingface"
                else _PUBLIC_SOURCE_KEYS
            )
            for key in sorted(set(source) - set(allowed)):
                withheld.append(
                    WithheldFact(f"{prefix}.source.{key}", source[key], _SOURCE_REASON)
                )
                del source[key]
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
    "ensure_public_model_sources",
    "ensure_publishable",
    "project_run_for_publication",
]
