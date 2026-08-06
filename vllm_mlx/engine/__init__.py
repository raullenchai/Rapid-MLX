# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for rapid-mlx inference.

BatchedEngine is the sole engine — continuous batching for all workloads.

``base`` is pure stdlib (abc/dataclasses/typing) and is imported eagerly.
``engine_core`` and ``batched`` reach ``import mlx.core`` at module scope,
so they are imported on first attribute access (PEP 562) instead.

That split is what lets a name like ``BaseEngine`` be used in an
annotation without dragging the engine onto the import path.
``config.server_config`` does exactly that, and while this package
imported everything eagerly, ``import vllm_mlx.config`` transitively
required MLX — which killed the whole ``TestAnthropicToOpenai`` suite on
Linux CI, since a wire adapter has no business needing Metal.

The alternative was to hide the import behind ``TYPE_CHECKING`` on the
consumer side, but a name that exists only for the type checker is a name
``typing.get_type_hints`` cannot resolve, and binding a runtime stand-in
makes introspection answer with the WRONG type. Making the import cheap
is the fix that leaves both the runtime and the type checker correct.
"""

from .base import BaseEngine, GenerationOutput

__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "BatchedEngine",
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]

_LAZY = {
    "EngineCore": ("..engine_core", "EngineCore"),
    "AsyncEngineCore": ("..engine_core", "AsyncEngineCore"),
    "EngineConfig": ("..engine_core", "EngineConfig"),
    "BatchedEngine": (".batched", "BatchedEngine"),
}


def __getattr__(name: str):
    """Resolve the MLX-dependent members on first use (PEP 562).

    Caches into the module globals so the import cost is paid once and
    later lookups skip this hook entirely.
    """
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Real module contents PLUS the names that are still deferred.

    Returning only ``__all__`` would hide ordinary module attributes and
    anything already imported, which breaks the introspection and
    completion that ``dir()`` exists to serve. The union is what a reader
    expects: everything that is here, and everything that would be if
    asked for.
    """
    return sorted(set(globals()) | set(__all__))
