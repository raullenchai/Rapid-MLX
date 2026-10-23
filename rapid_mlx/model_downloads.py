# SPDX-License-Identifier: Apache-2.0
"""``--disable-model-downloads``: never fetch a model that is not on disk.

Every path that would download a model calls :func:`check` once it knows the
model is missing locally, or :func:`require_local` before it hands a model id
to a loader that downloads on a cache miss. ``rapid-mlx pull`` is the explicit
way to provision models and is not affected.
"""

from __future__ import annotations

import os
from typing import Any

from ._env import env_truthy

ENV_VAR = "RAPID_MLX_DISABLE_MODEL_DOWNLOADS"
FLAG = "--disable-model-downloads"

_cli_disabled = False


class ModelDownloadsDisabledError(RuntimeError):
    """A model is not available locally and model downloads are disabled."""

    def __init__(self, model: str, source: str):
        self.model = model
        self.source = source
        super().__init__(
            f"{model} is not available locally and model downloads are "
            f"disabled by {source}"
        )


# Record the ``--disable-model-downloads`` flag for this process.
def configure(cli_flag: bool) -> None:
    global _cli_disabled
    _cli_disabled = bool(cli_flag)


# The setting that disables model downloads, or None when they are allowed.
def source() -> str | None:
    if _cli_disabled:
        return FLAG
    if env_truthy(ENV_VAR):
        return ENV_VAR
    return None


def disabled() -> bool:
    return source() is not None


# Call right before downloading ``model``.
def check(model: str) -> None:
    disabled_by = source()
    if disabled_by is not None:
        raise ModelDownloadsDisabledError(model, disabled_by)


# Call before handing ``model`` to a loader that downloads on a cache miss.
# Only a local path or a snapshot the cache inventory reports as runnable
# passes; an inconclusive cache probe counts as missing.
def require_local(model: str) -> None:
    if not disabled():
        return
    if os.path.exists(os.path.expanduser(model)):
        return
    from . import cli

    if cli._cache_runnability(model) is True:
        return
    check(model)


def snapshot_download(repo_id: str, **kwargs: Any) -> str:
    """Resolve a snapshot without permitting a network fallback when disabled."""
    from huggingface_hub import snapshot_download as hub_snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from .model_catalog import resolve_repository_identity

    resolved_repo, revision = resolve_repository_identity(repo_id)
    if revision is not None:
        requested = kwargs.get("revision")
        if requested is not None and requested != revision:
            raise ValueError(
                f"{repo_id} is pinned to revision {revision}; "
                f"cannot download conflicting revision {requested}"
            )
        kwargs["revision"] = revision
    repo_id = resolved_repo
    _pin_local_destination(repo_id, kwargs)
    if not disabled():
        result = str(hub_snapshot_download(repo_id, **kwargs))
        _verify_download(repo_id, result, kwargs)
        return result
    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        result = str(hub_snapshot_download(repo_id, **local_kwargs))
        _verify_download(repo_id, result, local_kwargs)
        return result
    except LocalEntryNotFoundError:
        raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None


def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
    """Resolve one Hub file locally when downloads are disabled."""
    from huggingface_hub import hf_hub_download as hub_file_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from .model_catalog import resolve_repository_identity

    resolved_repo, revision = resolve_repository_identity(repo_id)
    if revision is not None:
        requested = kwargs.get("revision")
        if requested is not None and requested != revision:
            raise ValueError(
                f"{repo_id} is pinned to revision {revision}; "
                f"cannot download conflicting revision {requested}"
            )
        kwargs["revision"] = revision
    repo_id = resolved_repo
    _pin_local_destination(repo_id, kwargs)
    if not disabled():
        result = str(hub_file_download(repo_id, filename, **kwargs))
        _verify_download(repo_id, result, kwargs, filename=filename)
        return result
    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        result = str(hub_file_download(repo_id, filename, **local_kwargs))
        _verify_download(repo_id, result, local_kwargs, filename=filename)
        return result
    except LocalEntryNotFoundError:
        raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None


def _pin_local_destination(repo_id, kwargs):
    """Local destinations lack a snapshot directory identifying the commit."""
    import re

    if kwargs.get("local_dir") is None or re.fullmatch(
        r"[0-9a-f]{40}", str(kwargs.get("revision", ""))
    ):
        return
    from huggingface_hub import HfApi

    from ._download_gate import call_with_deadline

    if disabled() or kwargs.get("local_files_only"):
        raise ValueError("Offline local_dir downloads require an immutable revision")
    info = call_with_deadline(
        lambda: HfApi().model_info(
            repo_id, revision=kwargs.get("revision"), token=kwargs.get("token")
        ),
        30,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", str(info.sha)):
        raise ValueError("Hub did not return an immutable commit for local_dir")
    kwargs["revision"] = info.sha


def _verify_download(repo_id, result, kwargs, *, filename=None):
    import re
    from pathlib import Path

    from .model_integrity import verify_snapshot

    root = Path(result)
    exact = None
    if filename is not None:
        name = str(Path(kwargs.get("subfolder") or "") / filename)
        exact = [name]
        for _ in Path(name).parts:
            root = root.parent
    local = kwargs.get("local_dir") is not None
    revision = kwargs.get("revision")
    verify_snapshot(
        repo_id,
        root,
        revision=revision if re.fullmatch(r"[0-9a-f]{40}", str(revision)) else None,
        resolved_revision=revision if local else None,
        allow_patterns=kwargs.get("allow_patterns"),
        ignore_patterns=kwargs.get("ignore_patterns"),
        filenames=exact,
        local_files_only=bool(kwargs.get("local_files_only")),
        token=kwargs.get("token"),
    )
