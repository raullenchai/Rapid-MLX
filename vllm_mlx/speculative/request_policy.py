# SPDX-License-Identifier: Apache-2.0
"""Request-level compatibility policy for speculative decoding.

Speculative decoding is configured once when an engine starts, but some
request features can require the scheduler to use ordinary decoding for that
request.  Keep that distinction in the engine package so API clients can
report the effective request path without reproducing scheduler internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpeculativeRequestFallbackFeature = Literal["tools"]


@dataclass(frozen=True)
class SpeculativeRequestPolicy:
    """Configured method and request features that trigger safe fallback."""

    method: str
    request_fallback_features: tuple[SpeculativeRequestFallbackFeature, ...] = ()


def resolve_speculative_request_policy(
    method: object,
) -> SpeculativeRequestPolicy | None:
    """Return the policy for one configured method, or ``None`` when off.

    MTP verifies tokens through the target model, but its current request-local
    sampler contract admits only the scheduler's standard penalty processors.
    Tool requests add grammar and agent-loop processors, so the scheduler
    deliberately keeps those requests on ordinary decoding.  Advertising that
    fact is additive; it does not weaken the fail-closed processor identity
    check in the decode loop.
    """

    if not isinstance(method, str):
        return None
    normalized = method.strip().lower()
    if normalized in ("", "none"):
        return None
    fallback_features: tuple[SpeculativeRequestFallbackFeature, ...] = (
        ("tools",) if normalized == "mtp" else ()
    )
    return SpeculativeRequestPolicy(
        method=normalized,
        request_fallback_features=fallback_features,
    )
