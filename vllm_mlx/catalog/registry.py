# SPDX-License-Identifier: Apache-2.0
"""Small content-addressed store for validated atomic model records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from .canonical import canonical_json_bytes, rcj_digest
from .validation import CatalogValidationError, ContractValidator

RegistryKind = Literal[
    "model_identity",
    "machine_observation",
    "execution_config",
    "recommendation_policy",
    "catalog_snapshot",
]


def _projection(
    kind: RegistryKind, document: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if kind == "model_identity":
        return (
            {
                key: document[key]
                for key in ("schema_version", "pipeline_kind", "components")
            },
            "identity_digest",
        )
    if kind == "machine_observation":
        return document["profile"], "profile_digest"
    if kind == "execution_config":
        return (
            {key: document[key] for key in ("task_type", "resources", "task")},
            "config_digest",
        )
    if kind == "recommendation_policy":
        return (
            {key: value for key, value in document.items() if key != "policy_digest"},
            "policy_digest",
        )
    return (
        {
            key: document[key]
            for key in (
                "schema_version",
                "models",
                "aliases",
                "recommendation_policy_digests",
            )
        },
        "catalog_digest",
    )


class AtomicRegistry:
    """Persist immutable records under ``<root>/<kind>/<sha256>.json``."""

    def __init__(
        self, root: str | os.PathLike[str], validator: ContractValidator | None = None
    ) -> None:
        self.root = Path(root)
        self.validator = validator or ContractValidator()

    def _validate_document(
        self,
        kind: RegistryKind,
        document: dict[str, Any],
        *,
        catalog_snapshot: dict[str, Any] | None,
    ) -> None:
        if kind == "catalog_snapshot":
            self.validator.validate_catalog_snapshot(document)
        elif kind == "recommendation_policy":
            if catalog_snapshot is None:
                raise ValueError(
                    "recommendation_policy validation requires catalog_snapshot"
                )
            self.validator.validate_catalog_snapshot(catalog_snapshot)
            aliases = {item["alias"]: item for item in catalog_snapshot["aliases"]}
            self.validator.validate_recommendation_policy(document, aliases=aliases)
            if (
                document["policy_digest"]
                not in catalog_snapshot["recommendation_policy_digests"]
            ):
                raise CatalogValidationError(
                    "recommendation_policy",
                    "policy_digest",
                    "is not referenced by the catalog snapshot",
                )
        elif kind == "model_identity":
            self.validator.validate_model_identity(document)
        else:
            self.validator.validate(kind, document)

    @staticmethod
    def _validate_declared_digest(
        kind: RegistryKind,
        document: dict[str, Any],
        digest: str,
        digest_field: str,
    ) -> None:
        declared = document.get(digest_field)
        digest_is_optional = (
            kind == "model_identity"
            and document.get("identity_strength") == "unresolved"
        )
        if (not digest_is_optional and declared != digest) or (
            digest_is_optional and declared is not None
        ):
            raise CatalogValidationError(
                kind, digest_field, "declared digest does not match RCJ-1"
            )

    def put(
        self,
        kind: RegistryKind,
        document: dict[str, Any],
        *,
        catalog_snapshot: dict[str, Any] | None = None,
    ) -> str:
        self._validate_document(kind, document, catalog_snapshot=catalog_snapshot)
        projection, digest_field = _projection(kind, document)
        digest = rcj_digest(projection)
        self._validate_declared_digest(kind, document, digest, digest_field)

        target_dir = self.root / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest.removeprefix('sha256:')}.json"
        payload = canonical_json_bytes(document) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=target_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Same-directory hard-link publication is create-if-absent:
                # unlike exists()+replace(), it can never overwrite a winner
                # from another process between observation and commit.
                os.link(temporary, target)
                directory_descriptor = os.open(target_dir, os.O_RDONLY)
                try:
                    # Persist the newly created directory entry before
                    # returning its address. The file fsync above protects
                    # bytes, not the name that makes them reachable.
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise CatalogValidationError(
                        kind, digest_field, "content-address collision"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def get(
        self,
        kind: RegistryKind,
        digest: str,
        *,
        catalog_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("digest must be sha256:<64 lowercase hex characters>")
        suffix = digest.removeprefix("sha256:")
        if any(character not in "0123456789abcdef" for character in suffix):
            raise ValueError("digest must be sha256:<64 lowercase hex characters>")
        path = self.root / kind / f"{suffix}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self._validate_document(kind, document, catalog_snapshot=catalog_snapshot)
        projection, digest_field = _projection(kind, document)
        computed = rcj_digest(projection)
        if computed != digest:
            raise CatalogValidationError(
                kind, "", "stored object failed content-address verification"
            )
        self._validate_declared_digest(kind, document, computed, digest_field)
        return document
