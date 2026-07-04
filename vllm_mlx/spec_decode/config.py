# SPDX-License-Identifier: Apache-2.0
"""vLLM-style speculative decoding config parsing."""

from __future__ import annotations

import json
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
    raw: dict[str, Any] | None = None


_COMMON_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
    }
)


def _positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeConfigError(f"{key} must be a positive integer")
    if value <= 0:
        raise SpeculativeConfigError(f"{key} must be a positive integer")
    return value


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

    unknown = sorted(set(payload) - _COMMON_KEYS)
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
        raw=dict(payload),
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
    "parse_speculative_config",
    "require_migrated_speculative_config",
]
