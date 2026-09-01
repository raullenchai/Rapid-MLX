# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contract for composing prefix-cache hits with persistent MTP.

Automatic prefix caching owns target-cache storage.  Persistent self-MTP owns
an additional draft cache and the trunk hidden state that seeds the first pair
after a restored prefix.  Those three objects are reusable only when they were
captured together at one exact token boundary::

    target tokens == covered tokens
    MTP pairs     == covered tokens - 1
    seed hidden   == hidden state of the final covered token

This module deliberately does not import MLX or modify the prefix cache.  It is
the validation/policy seam an eventual APC integration can call before it
mutates either cache.  Producer mistakes raise at capture time; persisted or
lookup-time mismatches return a refusal so serving can fall back safely.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeGuard

PREPARED_STATE_SCHEMA_VERSION = 1
DEFAULT_MIN_USEFUL_PREFIX_TOKENS = 64


class RestoreReason(str, Enum):
    """Stable result codes for prepared-state admission."""

    ELIGIBLE = "eligible"
    TRIVIAL_HIT = "trivial_hit"
    STALE = "stale"
    MODEL_MISMATCH = "model_mismatch"
    CONFIG_MISMATCH = "config_mismatch"
    BOUNDARY_MISMATCH = "boundary_mismatch"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class PreparedStateIdentity:
    """Identity of the exact runtime contract that produced a sidecar.

    ``model_id`` alone is not sufficient: an alias can be repointed, an
    adapter can change the target distribution, or a speculative configuration
    can change the draft-cache meaning.  Cache and seed-hidden layout strings
    are caller-provided stable fingerprints (for example, topology + dtype +
    hidden width), keeping this module independent of MLX objects.
    """

    model_id: str
    model_revision: str
    speculative_config_fingerprint: str
    target_cache_layout: str
    mtp_cache_layout: str
    seed_hidden_layout: str
    adapter_id: str | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        required = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "speculative_config_fingerprint": (self.speculative_config_fingerprint),
            "target_cache_layout": self.target_cache_layout,
            "mtp_cache_layout": self.mtp_cache_layout,
            "seed_hidden_layout": self.seed_hidden_layout,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name, optional_value in (
            ("adapter_id", self.adapter_id),
            ("tokenizer_fingerprint", self.tokenizer_fingerprint),
        ):
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise ValueError(f"{name} must be None or a non-empty string")

    @classmethod
    def from_config(
        cls,
        *,
        model_id: str,
        model_revision: str,
        speculative_config: Mapping[str, Any],
        target_cache_layout: str,
        mtp_cache_layout: str,
        seed_hidden_layout: str,
        adapter_id: str | None = None,
        tokenizer_fingerprint: str | None = None,
    ) -> PreparedStateIdentity:
        """Build an identity with a canonical speculative-config digest."""

        return cls(
            model_id=model_id,
            model_revision=model_revision,
            speculative_config_fingerprint=fingerprint_config(speculative_config),
            target_cache_layout=target_cache_layout,
            mtp_cache_layout=mtp_cache_layout,
            seed_hidden_layout=seed_hidden_layout,
            adapter_id=adapter_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )


@dataclass(frozen=True)
class PreparedStateMetadata:
    """Serializable metadata paired with the three opaque runtime objects."""

    identity: PreparedStateIdentity
    covered_tokens: int
    mtp_covered_pairs: int
    boundary_fingerprint: str
    captured_at: float
    schema_version: int = PREPARED_STATE_SCHEMA_VERSION


@dataclass(frozen=True)
class PreparedMTPState:
    """Target cache, draft cache, and seed hidden captured as one unit."""

    metadata: PreparedStateMetadata
    target_cache: Any
    mtp_cache: Any
    seed_hidden: Any


@dataclass(frozen=True)
class RestoreEligibility:
    """Non-throwing lookup decision for a prepared state.

    ``bypass_hit`` is true for a valid but too-small prefix.  That is the
    trivial-hit fail-open rule: the caller should ignore the APC hit and start
    the normal MTP path instead of letting a few cached template tokens disable
    speculation.  Other refusals leave fallback selection to the caller.
    """

    eligible: bool
    reason: RestoreReason
    covered_tokens: int = 0
    resume_at: int | None = None
    bypass_hit: bool = False


def fingerprint_config(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON config data."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    try:
        encoded = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("config must contain canonical JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_tokens(tokens: Sequence[int]) -> str:
    """Fingerprint an ordered token boundary without retaining prompt text."""

    digest = hashlib.sha256()
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("tokens must contain non-negative integers")
        if token >= 1 << 64:
            raise ValueError("token ids must fit in an unsigned 64-bit integer")
        digest.update(token.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def prepare_mtp_state(
    *,
    identity: PreparedStateIdentity,
    prefix_tokens: Sequence[int],
    target_cache: Any,
    target_cache_tokens: int,
    mtp_cache: Any,
    mtp_cache_pairs: int,
    seed_hidden: Any,
    captured_at: float | None = None,
) -> PreparedMTPState:
    """Validate and capture one exact target/MTP/hidden boundary.

    This strict producer-side function raises for a programmer error.  Lookup
    uses :func:`evaluate_restore`, which converts corrupted, stale, or mismatched
    state into a refusal instead of raising on the serving path.
    """

    prefix = tuple(prefix_tokens)
    covered = _validated_count(target_cache_tokens, "target_cache_tokens")
    mtp_pairs = _validated_count(mtp_cache_pairs, "mtp_cache_pairs")
    if not prefix or covered == 0:
        raise ValueError("prepared MTP state requires a non-empty prefix")
    if covered != len(prefix):
        raise ValueError(
            "target_cache_tokens must equal the exact covered prefix length"
        )
    if mtp_pairs != covered - 1:
        raise ValueError("mtp_cache_pairs must equal target_cache_tokens - 1")
    if target_cache is None:
        raise ValueError("target_cache must be present")
    if mtp_cache is None:
        raise ValueError("mtp_cache must be present")
    if seed_hidden is None:
        raise ValueError("seed_hidden must cover the final prefix token")
    timestamp = time.time() if captured_at is None else float(captured_at)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("captured_at must be a finite non-negative timestamp")

    metadata = PreparedStateMetadata(
        identity=identity,
        covered_tokens=covered,
        mtp_covered_pairs=mtp_pairs,
        boundary_fingerprint=fingerprint_tokens(prefix),
        captured_at=timestamp,
    )
    return PreparedMTPState(metadata, target_cache, mtp_cache, seed_hidden)


def evaluate_restore(
    state: PreparedMTPState,
    *,
    expected_identity: PreparedStateIdentity,
    request_tokens: Sequence[int],
    target_cache_tokens: int,
    mtp_cache_pairs: int,
    now: float | None = None,
    max_age_seconds: float | None = None,
    min_useful_prefix_tokens: int = DEFAULT_MIN_USEFUL_PREFIX_TOKENS,
) -> RestoreEligibility:
    """Return whether ``state`` may be restored for ``request_tokens``.

    Validation order intentionally reports identity and staleness before the
    performance-only trivial-hit rule.  A malformed or foreign state is never
    disguised as a benign small hit.
    """

    if (
        isinstance(min_useful_prefix_tokens, bool)
        or not isinstance(min_useful_prefix_tokens, int)
        or min_useful_prefix_tokens < 1
    ):
        raise ValueError("min_useful_prefix_tokens must be a positive integer")
    if max_age_seconds is not None:
        max_age_seconds = float(max_age_seconds)
        if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be finite and non-negative")

    metadata = getattr(state, "metadata", None)
    if not _metadata_is_well_formed(metadata):
        return _refusal(RestoreReason.MALFORMED)
    covered = metadata.covered_tokens
    if not isinstance(expected_identity, PreparedStateIdentity):
        return _refusal(RestoreReason.MALFORMED, covered)

    identity_reason = _identity_mismatch(metadata.identity, expected_identity)
    if identity_reason is not None:
        return _refusal(identity_reason, covered)

    try:
        current_time = time.time() if now is None else float(now)
    except (TypeError, ValueError):
        return _refusal(RestoreReason.MALFORMED, covered)
    if not math.isfinite(current_time) or current_time < metadata.captured_at:
        return _refusal(RestoreReason.STALE, covered)
    if (
        max_age_seconds is not None
        and current_time - metadata.captured_at > max_age_seconds
    ):
        return _refusal(RestoreReason.STALE, covered)

    try:
        live_target = _validated_count(target_cache_tokens, "target_cache_tokens")
        live_mtp = _validated_count(mtp_cache_pairs, "mtp_cache_pairs")
        request = tuple(request_tokens)
        boundary_matches = (
            len(request) > covered
            and fingerprint_tokens(request[:covered]) == metadata.boundary_fingerprint
        )
    except (TypeError, ValueError):
        return _refusal(RestoreReason.MALFORMED, covered)

    exact_boundary = (
        getattr(state, "target_cache", None) is not None
        and getattr(state, "mtp_cache", None) is not None
        and getattr(state, "seed_hidden", None) is not None
        and metadata.mtp_covered_pairs == covered - 1
        and live_target == covered
        and live_mtp == covered - 1
        and boundary_matches
    )
    if not exact_boundary:
        return _refusal(RestoreReason.BOUNDARY_MISMATCH, covered)

    if covered < min_useful_prefix_tokens:
        return RestoreEligibility(
            eligible=False,
            reason=RestoreReason.TRIVIAL_HIT,
            covered_tokens=covered,
            bypass_hit=True,
        )

    return RestoreEligibility(
        eligible=True,
        reason=RestoreReason.ELIGIBLE,
        covered_tokens=covered,
        resume_at=covered,
    )


def _validated_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _metadata_is_well_formed(metadata: Any) -> TypeGuard[PreparedStateMetadata]:
    if not isinstance(metadata, PreparedStateMetadata):
        return False
    if metadata.schema_version != PREPARED_STATE_SCHEMA_VERSION:
        return False
    try:
        covered = _validated_count(metadata.covered_tokens, "covered_tokens")
        pairs = _validated_count(metadata.mtp_covered_pairs, "mtp_covered_pairs")
    except ValueError:
        return False
    try:
        valid_timestamp = (
            math.isfinite(metadata.captured_at) and metadata.captured_at >= 0
        )
        valid_fingerprint = (
            isinstance(metadata.boundary_fingerprint, str)
            and len(metadata.boundary_fingerprint) == 64
            and all(
                character in "0123456789abcdef"
                for character in metadata.boundary_fingerprint
            )
        )
    except TypeError:
        return False
    return (
        covered > 0
        and pairs == covered - 1
        and valid_fingerprint
        and valid_timestamp
        and isinstance(metadata.identity, PreparedStateIdentity)
    )


def _identity_mismatch(
    actual: PreparedStateIdentity,
    expected: PreparedStateIdentity,
) -> RestoreReason | None:
    model_actual = (
        actual.model_id,
        actual.model_revision,
        actual.adapter_id,
        actual.tokenizer_fingerprint,
    )
    model_expected = (
        expected.model_id,
        expected.model_revision,
        expected.adapter_id,
        expected.tokenizer_fingerprint,
    )
    if model_actual != model_expected:
        return RestoreReason.MODEL_MISMATCH
    config_actual = (
        actual.speculative_config_fingerprint,
        actual.target_cache_layout,
        actual.mtp_cache_layout,
        actual.seed_hidden_layout,
    )
    config_expected = (
        expected.speculative_config_fingerprint,
        expected.target_cache_layout,
        expected.mtp_cache_layout,
        expected.seed_hidden_layout,
    )
    if config_actual != config_expected:
        return RestoreReason.CONFIG_MISMATCH
    return None


def _refusal(
    reason: RestoreReason,
    covered_tokens: int = 0,
) -> RestoreEligibility:
    return RestoreEligibility(False, reason, covered_tokens=max(0, covered_tokens))


__all__ = [
    "DEFAULT_MIN_USEFUL_PREFIX_TOKENS",
    "PREPARED_STATE_SCHEMA_VERSION",
    "PreparedMTPState",
    "PreparedStateIdentity",
    "PreparedStateMetadata",
    "RestoreEligibility",
    "RestoreReason",
    "evaluate_restore",
    "fingerprint_config",
    "fingerprint_tokens",
    "prepare_mtp_state",
]
