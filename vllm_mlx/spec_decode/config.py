# SPDX-License-Identifier: Apache-2.0
"""vLLM-style speculative decoding config parsing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .registry import get_spec_decoder, iter_spec_decoders


class SpeculativeConfigError(ValueError):
    """Raised when ``--speculative-config`` is malformed or unsupported."""


@dataclass(frozen=True)
class SpeculativeConfig:
    """Parsed ``--speculative-config`` payload.

    Field names intentionally match vLLM where they overlap. Backend-specific
    keys remain in ``raw`` until a method is migrated to this surface.
    """

    method: str
    model: str | None = None
    num_speculative_tokens: int | None = None
    tree_budget: int | None = None
    max_suffix_len: int | None = None
    min_confidence: float | None = None
    min_draft_len: int | None = None
    raw: dict[str, Any] | None = None


_COMMON_KEYS = frozenset(
    {
        "method",
    }
)

_METHOD_KEYS = {
    "ddtree": frozenset({"model", "num_speculative_tokens", "tree_budget"}),
    "dflash": frozenset({"model"}),
    "mtp": frozenset({"model", "num_speculative_tokens"}),
    "suffix": frozenset(
        {
            "num_speculative_tokens",
            "max_suffix_len",
            "min_confidence",
            "min_draft_len",
        }
    ),
}


def _positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeConfigError(f"{key} must be a positive integer")
    if value <= 0:
        raise SpeculativeConfigError(f"{key} must be a positive integer")
    return value


def _positive_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeculativeConfigError(f"{key} must be a positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise SpeculativeConfigError(f"{key} must be a positive number")
    return numeric


def _confidence(value: Any, key: str) -> float | None:
    numeric = _positive_float(value, key)
    if numeric is None:
        return None
    if numeric > 1:
        raise SpeculativeConfigError(f"{key} must be between 0 and 1")
    return numeric


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpeculativeConfigError(f"{key} must be a non-empty string")
    return value.strip()


def parse_speculative_config(value: str | None) -> SpeculativeConfig | None:
    """Parse the JSON value passed to ``--speculative-config``."""

    if value is None:
        return None
    if not value.strip():
        raise SpeculativeConfigError("--speculative-config cannot be empty")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SpeculativeConfigError(
            f"--speculative-config must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise SpeculativeConfigError("--speculative-config must be a JSON object")

    raw_method = payload.get("method")
    if not isinstance(raw_method, str) or not raw_method.strip():
        raise SpeculativeConfigError(
            "--speculative-config requires string key 'method'"
        )
    method = raw_method.strip().lower()
    plugin = get_spec_decoder(method)
    known = ", ".join(plugin.method for plugin in iter_spec_decoders())
    if plugin is None:
        raise SpeculativeConfigError(
            f"unsupported speculative decoding method {method!r}; "
            f"known methods: {known}"
        )
    method = plugin.method

    allowed_keys = _COMMON_KEYS | _METHOD_KEYS.get(method, frozenset())
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise SpeculativeConfigError(
            f"unsupported speculative-config key(s) for {method!r}: {joined}"
        )

    return SpeculativeConfig(
        method=method,
        model=_optional_string(payload.get("model"), "model"),
        num_speculative_tokens=_positive_int(
            payload.get("num_speculative_tokens"), "num_speculative_tokens"
        ),
        tree_budget=_positive_int(payload.get("tree_budget"), "tree_budget"),
        max_suffix_len=_positive_int(payload.get("max_suffix_len"), "max_suffix_len"),
        min_confidence=_confidence(payload.get("min_confidence"), "min_confidence"),
        min_draft_len=_positive_int(payload.get("min_draft_len"), "min_draft_len"),
        raw=dict(payload),
    )


def legacy_ddtree_config() -> SpeculativeConfig:
    """Return the compatibility config represented by ``--enable-ddtree``."""

    return SpeculativeConfig(method="ddtree", raw={"method": "ddtree"})


def legacy_dflash_config(model: str | None = None) -> SpeculativeConfig:
    """Return the compatibility config represented by DFlash legacy flags."""

    raw = {"method": "dflash"}
    drafter = _optional_string(model, "model")
    if drafter is not None:
        raw["model"] = drafter
    return SpeculativeConfig(method="dflash", model=drafter, raw=raw)


def legacy_mtp_config(
    *,
    model: str | None = None,
    num_speculative_tokens: int | None = None,
) -> SpeculativeConfig:
    """Return the compatibility config represented by MTP legacy flags."""

    raw: dict[str, Any] = {"method": "mtp"}
    sidecar = _optional_string(model, "model")
    if sidecar is not None:
        raw["model"] = sidecar
    tokens = _positive_int(num_speculative_tokens, "num_speculative_tokens")
    if tokens is not None:
        raw["num_speculative_tokens"] = tokens
    return SpeculativeConfig(
        method="mtp",
        model=sidecar,
        num_speculative_tokens=tokens,
        raw=raw,
    )


def legacy_suffix_config(
    *,
    num_speculative_tokens: int | None = None,
    max_suffix_len: int | None = None,
    min_confidence: float | None = None,
    min_draft_len: int | None = None,
) -> SpeculativeConfig:
    """Return the compatibility config represented by ``--suffix-decoding``."""

    raw: dict[str, Any] = {"method": "suffix"}
    tokens = _positive_int(num_speculative_tokens, "num_speculative_tokens")
    if tokens is not None:
        raw["num_speculative_tokens"] = tokens
    suffix_len = _positive_int(max_suffix_len, "max_suffix_len")
    if suffix_len is not None:
        raw["max_suffix_len"] = suffix_len
    confidence = _confidence(min_confidence, "min_confidence")
    if confidence is not None:
        raw["min_confidence"] = confidence
    draft_len = _positive_int(min_draft_len, "min_draft_len")
    if draft_len is not None:
        raw["min_draft_len"] = draft_len
    return SpeculativeConfig(
        method="suffix",
        num_speculative_tokens=tokens,
        max_suffix_len=suffix_len,
        min_confidence=confidence,
        min_draft_len=draft_len,
        raw=raw,
    )


def require_migrated_speculative_config(config: SpeculativeConfig) -> None:
    """Fail until ``config.method`` is wired to a backend runner."""

    plugin = get_spec_decoder(config.method)
    if plugin is None:
        raise SpeculativeConfigError(
            f"unsupported speculative decoding method {config.method!r}"
        )
    if not plugin.config_enabled:
        hint = f"; {plugin.legacy_hint}" if plugin.legacy_hint else ""
        raise SpeculativeConfigError(
            f"--speculative-config method {plugin.method!r} is not wired yet{hint}."
        )


__all__ = [
    "SpeculativeConfig",
    "SpeculativeConfigError",
    "legacy_ddtree_config",
    "legacy_dflash_config",
    "legacy_mtp_config",
    "legacy_suffix_config",
    "parse_speculative_config",
    "require_migrated_speculative_config",
]
