# SPDX-License-Identifier: Apache-2.0
"""Syntactic and cross-record validation for atomic catalog contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import jsonschema
import referencing

from .canonical import rcj_digest

_SCHEMA_FILES = {
    "model_identity": "model-identity.schema.json",
    "machine_observation": "machine-observation.schema.json",
    "execution_config": "execution-config.schema.json",
    "model_alias": "model-alias.schema.json",
    "model_registry_record": "model-registry-record.schema.json",
    "recommendation_policy": "recommendation-policy.schema.json",
    "catalog_snapshot": "catalog-snapshot.schema.json",
}


@dataclass(frozen=True)
class CatalogValidationError(ValueError):
    """One stable, caller-renderable contract failure."""

    contract: str
    path: str
    message: str

    def __str__(self) -> str:
        location = f" at {self.path}" if self.path else ""
        return f"{self.contract}{location}: {self.message}"


class ContractValidator:
    """Validate packaged schemas plus relationships JSON Schema cannot express."""

    def __init__(self) -> None:
        schema_root = resources.files("vllm_mlx.catalog.schemas")
        schemas = {
            kind: json.loads(schema_root.joinpath(filename).read_text(encoding="utf-8"))
            for kind, filename in _SCHEMA_FILES.items()
        }
        registry = referencing.Registry().with_resources(
            (
                schema["$id"],
                referencing.Resource.from_contents(schema),
            )
            for schema in schemas.values()
        )
        self._validators = {
            kind: jsonschema.Draft202012Validator(
                schema,
                registry=registry,
                format_checker=jsonschema.FormatChecker(),
            )
            for kind, schema in schemas.items()
        }

    def errors(
        self, contract: str, document: dict[str, Any]
    ) -> list[CatalogValidationError]:
        try:
            validator = self._validators[contract]
        except KeyError as exc:
            raise KeyError(f"unknown contract {contract!r}") from exc
        return [
            CatalogValidationError(
                contract,
                "/".join(str(part) for part in error.absolute_path),
                error.message,
            )
            for error in sorted(
                validator.iter_errors(document),
                key=lambda error: tuple(str(part) for part in error.path),
            )
        ]

    def validate(self, contract: str, document: dict[str, Any]) -> None:
        failures = self.errors(contract, document)
        if failures:
            raise failures[0]

    def validate_catalog_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.validate("catalog_snapshot", snapshot)
        model_by_id: dict[str, dict[str, Any]] = {}
        for model in snapshot["models"]:
            model_id = model["registry_model_id"]
            if model_id in model_by_id:
                raise CatalogValidationError(
                    "catalog_snapshot",
                    "models",
                    f"duplicate registry_model_id {model_id!r}",
                )
            model_by_id[model_id] = model

        seen_aliases: set[str] = set()
        for alias in snapshot["aliases"]:
            alias_name = alias["alias"]
            if alias_name in seen_aliases:
                raise CatalogValidationError(
                    "catalog_snapshot", "aliases", f"duplicate alias {alias_name!r}"
                )
            seen_aliases.add(alias_name)
            preset_ids: set[str] = set()
            for preset_index, preset in enumerate(alias["execution_presets"]):
                preset_id = preset["preset_id"]
                if preset_id in preset_ids:
                    raise CatalogValidationError(
                        "catalog_snapshot",
                        f"aliases/{alias_name}/execution_presets/{preset_index}/preset_id",
                        f"duplicate preset_id {preset_id!r}",
                    )
                preset_ids.add(preset_id)
            default_preset = alias["default_execution_preset_id"]
            if default_preset is not None and default_preset not in preset_ids:
                raise CatalogValidationError(
                    "catalog_snapshot",
                    f"aliases/{alias_name}/default_execution_preset_id",
                    "does not resolve to exactly one execution preset",
                )
            target = alias["target"]
            model = model_by_id.get(target["registry_model_id"])
            if model is None:
                raise CatalogValidationError(
                    "catalog_snapshot",
                    f"aliases/{alias_name}/target",
                    "registry_model_id does not resolve",
                )
            for field in ("resolution_status", "model_identity_digest"):
                if target.get(field) != model.get(field):
                    raise CatalogValidationError(
                        "catalog_snapshot",
                        f"aliases/{alias_name}/target/{field}",
                        "does not match the registry model record",
                    )

        projection = {
            key: snapshot[key]
            for key in (
                "schema_version",
                "models",
                "aliases",
                "recommendation_policy_digests",
            )
        }
        if rcj_digest(projection) != snapshot["catalog_digest"]:
            raise CatalogValidationError(
                "catalog_snapshot",
                "catalog_digest",
                "does not match the RCJ-1 projection",
            )

    def validate_recommendation_policy(
        self, policy: dict[str, Any], *, aliases: dict[str, dict[str, Any]]
    ) -> None:
        self.validate("recommendation_policy", policy)
        previous_floor = -1
        for tier_index, tier in enumerate(policy["tiers"]):
            floor = tier["minimum_memory_mib"]
            if floor <= previous_floor:
                raise CatalogValidationError(
                    "recommendation_policy",
                    f"tiers/{tier_index}/minimum_memory_mib",
                    "tiers must be strictly increasing",
                )
            previous_floor = floor
            roles: set[str] = set()
            for pick_index, pick in enumerate(tier["picks"]):
                if pick["role"] in roles:
                    raise CatalogValidationError(
                        "recommendation_policy",
                        f"tiers/{tier_index}/picks/{pick_index}/role",
                        "roles must be unique within a tier",
                    )
                roles.add(pick["role"])
                alias = aliases.get(pick["alias"])
                if alias is None:
                    raise CatalogValidationError(
                        "recommendation_policy",
                        f"tiers/{tier_index}/picks/{pick_index}/alias",
                        "does not resolve in the catalog",
                    )
                if policy["task_type"] not in alias["capabilities"]["task_types"]:
                    raise CatalogValidationError(
                        "recommendation_policy",
                        f"tiers/{tier_index}/picks/{pick_index}/alias",
                        "alias does not advertise the policy task_type",
                    )
        projection = {
            key: value for key, value in policy.items() if key != "policy_digest"
        }
        if rcj_digest(projection) != policy["policy_digest"]:
            raise CatalogValidationError(
                "recommendation_policy",
                "policy_digest",
                "does not match the RCJ-1 projection",
            )
