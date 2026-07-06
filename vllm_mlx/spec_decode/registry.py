# SPDX-License-Identifier: Apache-2.0
"""Registry metadata for speculative decoder frontends.

The user-facing shape follows vLLM's ``--speculative-config`` JSON. The
runtime contracts are still MLX-specific, so this registry starts as the
stable place to name/configure methods before each backend is migrated
behind a common runner interface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecDecoderPlugin:
    """Metadata for a speculative decoding method exposed to the CLI."""

    method: str
    description: str
    config_enabled: bool = False
    legacy_hint: str | None = None
    aliases: tuple[str, ...] = ()


_PLUGINS: dict[str, SpecDecoderPlugin] = {}


def register_spec_decoder(plugin: SpecDecoderPlugin) -> None:
    """Register ``plugin`` under its method and aliases."""

    method = plugin.method.strip().lower()
    if not method:
        raise ValueError("spec decoder method cannot be empty")
    normalized = SpecDecoderPlugin(
        method=method,
        description=plugin.description,
        config_enabled=plugin.config_enabled,
        legacy_hint=plugin.legacy_hint,
        aliases=tuple(alias.strip().lower() for alias in plugin.aliases if alias),
    )
    for name in (normalized.method, *normalized.aliases):
        if name in _PLUGINS and _PLUGINS[name] != normalized:
            raise ValueError(f"spec decoder method {name!r} is already registered")
        _PLUGINS[name] = normalized


def get_spec_decoder(method: str) -> SpecDecoderPlugin | None:
    """Return the plugin registered for ``method``, if any."""

    return _PLUGINS.get(method.strip().lower())


def iter_spec_decoders() -> tuple[SpecDecoderPlugin, ...]:
    """Return each registered plugin once, excluding alias duplicates."""

    seen: set[str] = set()
    out: list[SpecDecoderPlugin] = []
    for plugin in _PLUGINS.values():
        if plugin.method in seen:
            continue
        seen.add(plugin.method)
        out.append(plugin)
    return tuple(out)


register_spec_decoder(
    SpecDecoderPlugin(
        method="ddtree",
        description="DDTree / DFlash draft-tree verifier",
        config_enabled=True,
    )
)
register_spec_decoder(
    SpecDecoderPlugin(
        method="dflash",
        description="Block-diffusion drafter via the existing single-user bridge",
        config_enabled=True,
        legacy_hint="use --enable-dflash or --spec-decode dflash",
    )
)
register_spec_decoder(
    SpecDecoderPlugin(
        method="mtp",
        description="Model-side multi-token prediction head",
        config_enabled=True,
        legacy_hint="use --spec-decode mtp",
    )
)
register_spec_decoder(
    SpecDecoderPlugin(
        method="suffix",
        description=("Explicit suffix / n-gram speculation for high-overlap workloads"),
        config_enabled=True,
        legacy_hint="use --suffix-decoding",
        aliases=("ngram",),
    )
)


__all__ = [
    "SpecDecoderPlugin",
    "get_spec_decoder",
    "iter_spec_decoders",
    "register_spec_decoder",
]
