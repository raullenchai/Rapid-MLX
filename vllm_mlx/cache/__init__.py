# SPDX-License-Identifier: Apache-2.0
"""KV cache export/import wire protocol (issue #476).

The engine-level save/load primitives live in ``vllm_mlx.memory_cache``
and ``vllm_mlx.runtime.cache``; this package defines only the HTTP-
layer contract that ``vllm_mlx.routes.cache`` exposes: protocol version,
manifest schema, and path-whitelist helpers shared between routes and
the engine integration follow-up.
"""

from .protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    Manifest,
    ManifestMismatchError,
    ManifestNotFoundError,
    read_manifest,
    resolve_cache_dir,
    write_manifest,
)

__all__ = [
    "PROTOCOL_VERSION",
    "InvalidExportPathError",
    "MalformedManifestError",
    "Manifest",
    "ManifestMismatchError",
    "ManifestNotFoundError",
    "read_manifest",
    "resolve_cache_dir",
    "write_manifest",
]
