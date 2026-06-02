# SPDX-License-Identifier: Apache-2.0
"""Pinned versions + integrity digests for the bundled frpc binary.

frp ships pre-built static binaries on every GitHub release. We pin one
known-good version so users get a deterministic download — bumping it
means re-running ``scripts/share_bump_frpc.sh`` to recompute these
hashes and re-test the share flow end-to-end.
"""

from __future__ import annotations

FRPC_VERSION = "0.69.1"

# Computed via `shasum -a 256 frp_${VER}_${plat}.tar.gz` against the
# assets at https://github.com/fatedier/frp/releases/download/v${VER}/
FRPC_SHA256: dict[str, str] = {
    "darwin_arm64": "310012e2f1dcf3cdde2605d29b95340b686c94d1680a23711d58efeffc02f64e",
    "darwin_amd64": "2bc26d02100ef333f2712149ea5997dc530dc0eefac64f4be41cb0f49d032f40",
    "linux_arm64": "bbc0c75e896af3f292fb46ba09c844a04fa9b5ea3530c039c7af20637f836355",
    "linux_amd64": "7be257b72dbbc60bcb3e0e25a5afd1dfac7b63f897084864d3c956dd3d5674e1",
}

DEFAULT_RELAY_URL = "https://api.rapidmlx.com"
"""Control-plane base URL. Override with ``RAPID_MLX_RELAY_URL`` for local dev."""
