"""Hugging Face Hub helpers (optional dependency, imported lazily).

We never import ``huggingface_hub`` at module load. A local path is always used
as-is; only an unresolved repo id triggers a download, and only then do we require
the optional dependency.
"""

from __future__ import annotations

from pathlib import Path

_INSTALL_HINT = (
    "Resolving a Hub repo id requires `huggingface_hub`. "
    "Install it with `pip install mlx-diffuser[hub]` (or `pip install huggingface_hub`)."
)


def is_local(path_or_repo_id: str | Path) -> bool:
    """True if the argument points at an existing local file or directory."""
    return Path(path_or_repo_id).exists()


def resolve(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Return a local directory for a model/pipeline.

    If ``path_or_repo_id`` exists locally it is returned unchanged. Otherwise it is
    treated as a Hub repo id and downloaded with ``snapshot_download``.
    """
    if is_local(path_or_repo_id):
        return Path(path_or_repo_id)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_INSTALL_HINT) from exc

    local = snapshot_download(
        repo_id=str(path_or_repo_id),
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
    )
    return Path(local)


def push_folder(
    folder: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload with mlx-diffuser",
) -> str:
    """Upload a saved model/pipeline folder to the Hub. Returns the repo URL."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_INSTALL_HINT) from exc

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
