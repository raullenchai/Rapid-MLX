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

    The scheduler remains the final authority: it admits only the exact built-in
    processors with a verified target-row transaction contract.  MTP supports
    the default tool grammar and agent-loop guard, so tools are no longer a
    categorical request fallback. Optional or unknown processors still fail
    closed at the decode-loop identity gate.
    """

    if not isinstance(method, str):
        return None
    normalized = method.strip().lower()
    if normalized in ("", "none"):
        return None
    return SpeculativeRequestPolicy(
        method=normalized,
        request_fallback_features=(),
    )
