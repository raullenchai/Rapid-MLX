# SPDX-License-Identifier: Apache-2.0
"""Product-wide model catalog built on Rapid-MLX atomic contracts."""

from .canonical import canonical_json_bytes, rcj_digest
from .legacy import (
    build_catalog_bundle,
    build_legacy_catalog_snapshot,
    build_legacy_recommendation_policy,
    build_shadow_report,
    load_product_recommendation_policy,
)
from .registry import AtomicRegistry
from .validation import CatalogValidationError, ContractValidator

__all__ = [
    "AtomicRegistry",
    "CatalogValidationError",
    "ContractValidator",
    "build_catalog_bundle",
    "build_legacy_catalog_snapshot",
    "build_legacy_recommendation_policy",
    "build_shadow_report",
    "canonical_json_bytes",
    "rcj_digest",
    "load_product_recommendation_policy",
]
