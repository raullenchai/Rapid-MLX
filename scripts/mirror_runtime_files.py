"""Runtime file allowlists for repositories whose loader fetches a fixed subset.

Most catalog repositories are fetched whole, so "the repository" and "what a
pull downloads" are the same set of files. Vendored image backends are the
exception: ``rapid_mlx._download_gate.IMAGE_MODEL_DATA_FILES`` lists the exact
files such a pull requests. SDXL's repository also carries fp32, ONNX, Flax
and OpenVINO copies of the same weights (about 70 GB) that the runtime never
reads. The mirror uploader and the drift audit both scope to this list, so the
mirror holds exactly what a pull fetches.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Running ``python scripts/<tool>.py`` puts ``scripts/`` first on sys.path, so a
# bare ``import rapid_mlx`` could resolve to an older installed copy and read a
# stale allowlist. Make the checkout these scripts ship with win.
if sys.path[:1] != [str(ROOT)]:
    sys.path.insert(0, str(ROOT))


def runtime_files(repo_id: str) -> frozenset[str] | None:
    """Return the files a pull of ``repo_id`` fetches, or ``None`` for all."""
    from rapid_mlx._download_gate import IMAGE_MODEL_DATA_FILES

    files = IMAGE_MODEL_DATA_FILES.get(repo_id)
    return frozenset(files) if files is not None else None
