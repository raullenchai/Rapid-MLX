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
    *,
    default_tools_verified: bool = False,
) -> SpeculativeRequestPolicy | None:
    """Return the policy for one configured method, or ``None`` when off.

    The scheduler remains the final authority: it admits only the exact built-in
    processors with a verified target-row transaction contract.  A caller may
    remove the categorical tools fallback only after proving that the live
    server configuration uses the default constrained grammar path. Optional or
    unknown processors still fail closed at the decode-loop identity gate.
    """

    if not isinstance(method, str):
        return None
    normalized = method.strip().lower()
    if normalized in ("", "none"):
        return None
    fallback_features: tuple[SpeculativeRequestFallbackFeature, ...] = (
        ("tools",) if normalized == "mtp" and not default_tools_verified else ()
    )
    return SpeculativeRequestPolicy(
        method=normalized,
        request_fallback_features=fallback_features,
    )
