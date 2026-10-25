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


def _download_identity(repo_id: str, kwargs: dict) -> str:
    """Resolve only reviewed identities; a missing pin cannot select main."""
    import re

    from .model_aliases import resolve_hub_model, resolve_revision
    from .model_integrity import ModelIntegrityError
    from .runtime_model_pins import AUXILIARY_REVISIONS

    resolved_repo = resolve_hub_model(repo_id)
    revision = resolve_revision(repo_id) or AUXILIARY_REVISIONS.get(resolved_repo)
    requested = kwargs.get("revision")
    if revision is not None and requested is not None and requested != revision:
        raise ValueError(
            f"{repo_id} is pinned to revision {revision}; "
            f"cannot download conflicting revision {requested}"
        )
    revision = revision or requested
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelIntegrityError(
            f"Implicit download of {repo_id} requires a full immutable revision; "
            "provision a verified local checkpoint with rapid-mlx pull first"
        )
    kwargs["revision"] = revision
    return resolved_repo


def _verify(repo_id: str, path: str, kwargs: dict, *, filename=None) -> None:
    from pathlib import Path, PurePosixPath

    from huggingface_hub.constants import HF_HUB_OFFLINE

    from .model_integrity import verify_snapshot

    root = Path(path)
    if filename is not None:
        # Hub file results retain the snapshot path (do not resolve blob symlinks).
        for _ in PurePosixPath(filename).parts:
            root = root.parent
    verify_snapshot(
        repo_id,
        root,
        revision=kwargs["revision"],
        resolved_revision=kwargs["revision"] if kwargs.get("local_dir") else None,
        allow_patterns=kwargs.get("allow_patterns"),
        ignore_patterns=kwargs.get("ignore_patterns"),
        filenames=[filename] if filename is not None else None,
        local_files_only=bool(kwargs.get("local_files_only") or HF_HUB_OFFLINE),
        token=kwargs.get("token"),
    )


def snapshot_download(repo_id: str, **kwargs: Any) -> str:
    """Download an immutable snapshot and verify selected files before use."""
    from huggingface_hub import snapshot_download as hub_snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id = _download_identity(repo_id, kwargs)
    if disabled():
        kwargs["local_files_only"] = True
    try:
        path = str(hub_snapshot_download(repo_id, **kwargs))
    except LocalEntryNotFoundError:
        if disabled():
            raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None
        raise
    _verify(repo_id, path, kwargs)
    return path


def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
    """Download and verify exactly one file from an immutable snapshot."""
    from huggingface_hub import hf_hub_download as hub_file_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id = _download_identity(repo_id, kwargs)
    if disabled():
        kwargs["local_files_only"] = True
    try:
        path = str(hub_file_download(repo_id, filename, **kwargs))
    except LocalEntryNotFoundError:
        if disabled():
            raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None
        raise
    subfolder = kwargs.get("subfolder")
    selected = f"{subfolder}/{filename}" if subfolder else filename
    _verify(repo_id, path, kwargs, filename=selected)
    return path


def bind_hub_downloads(function):
    """Bind a dependency hook's local Hub imports without mutating the Hub.

    Some optional loaders import download functions inside their post-load hook.
    A function-local builtins namespace keeps the adapter isolated from other
    threads, explicit pulls and third-party code outside that hook.
    """
    import builtins
    from functools import update_wrapper
    from types import FunctionType, SimpleNamespace

    def verified_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = builtins.__import__(name, globals, locals, fromlist, level)
        if name == "huggingface_hub" and fromlist:
            replacements = {
                "snapshot_download": snapshot_download,
                "hf_hub_download": hf_hub_download,
            }
            return SimpleNamespace(
                **{
                    key: replacements[key]
                    if key in replacements
                    else getattr(module, key)
                    for key in fromlist
                }
            )
        return module

    namespace = dict(function.__globals__)
    namespace["__builtins__"] = dict(vars(builtins), __import__=verified_import)
    adapted = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    adapted.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(adapted, function)


def tokenizer_from_pretrained(name):
    """Resolve config-selected text tokenizers before entering transformers."""
    from pathlib import Path

    from transformers import AutoTokenizer

    path = str(name)
    if not Path(path).is_dir():
        path = snapshot_download(
            path, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"]
        )
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=False, local_files_only=True
    )
