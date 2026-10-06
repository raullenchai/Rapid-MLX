"""Pinned runtime file lists for repositories whose loader fetches a fixed set.

Most catalog repositories are fetched whole at the default branch. Vendored
image backends are the exception: ``rapid_mlx._download_gate`` pins each to an
exact commit (``IMAGE_MODEL_REVISIONS``) and lists the exact files a download
requests (``IMAGE_MODEL_DATA_FILES``). SDXL's repository also carries fp32,
ONNX, Flax and OpenVINO copies (about 70 GB) that are never requested. The
drift audit judges the mirror for such a repository against exactly that list
at exactly that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Running ``python scripts/<tool>.py`` puts ``scripts/`` first on sys.path, so a
# bare ``import rapid_mlx`` could resolve to an older installed copy and read a
# stale list. Make the checkout these scripts ship with win.
if sys.path[:1] != [str(ROOT)]:
    sys.path.insert(0, str(ROOT))


def pinned_runtime_files(repo_id: str) -> tuple[str, frozenset[str]] | None:
    """``(revision, files)`` a pull of ``repo_id`` fetches, or ``None``."""
    from rapid_mlx._download_gate import IMAGE_MODEL_DATA_FILES, IMAGE_MODEL_REVISIONS

    files = IMAGE_MODEL_DATA_FILES.get(repo_id)
    revision = IMAGE_MODEL_REVISIONS.get(repo_id)
    if files is None or revision is None:
        return None
    return revision, frozenset(files)
