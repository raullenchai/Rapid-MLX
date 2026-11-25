# SPDX-License-Identifier: Apache-2.0
"""``--disable-model-downloads``: never fetch a model that is not on disk.

Every path that would download a model calls :func:`check` once it knows the
model is missing locally, or :func:`require_local` before it hands a model id
to a loader that downloads on a cache miss. ``rapid-mlx pull`` is the explicit
way to provision models and is not affected.
"""

from __future__ import annotations

import http.client
import os
import time
import urllib.error
from collections.abc import Callable
from typing import Any, TypeVar

from ._env import env_truthy

_T = TypeVar("_T")
_TRANSIENT_RETRY_ATTEMPTS = 3
_TRANSIENT_BACKOFF_BASE_SECONDS = 1.0
_TRANSIENT_BACKOFF_MAX_SECONDS = 4.0


def _is_transient_hub_failure(exc: BaseException) -> bool:
    """True for bounded-retry transport/rate-limit/5xx failures on an explicit cause chain."""
    import httpx
    from huggingface_hub.errors import (
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )
    from requests import exceptions as requests_exceptions

    transient = False
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (OfflineModeIsEnabled, RepositoryNotFoundError)):
            return False
        status: int | None = None
        if isinstance(current, urllib.error.HTTPError):
            status = current.code
        elif isinstance(current, HfHubHTTPError):
            status = getattr(current.response, "status_code", None)
        if status in (401, 403, 404):
            return False
        if status == 429 or (status is not None and 500 <= status < 600):
            transient = True
        if isinstance(
            current,
            (
                httpx.TransportError,
                requests_exceptions.ConnectionError,
                requests_exceptions.Timeout,
                requests_exceptions.ChunkedEncodingError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                urllib.error.URLError,
            ),
        ) and not isinstance(current, urllib.error.HTTPError):
            transient = True
        current = current.__cause__
    return transient


def _retry_transient_hub_call(
    operation: Callable[[], _T],
    *,
    max_attempts: int = _TRANSIENT_RETRY_ATTEMPTS,
    base_delay: float = _TRANSIENT_BACKOFF_BASE_SECONDS,
    max_delay: float = _TRANSIENT_BACKOFF_MAX_SECONDS,
) -> _T:
    """Retry transient Hub transfers with bounded exponential backoff, preserving cached files."""
    attempts = max(1, max_attempts)
    for attempt in range(attempts - 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient_hub_failure(exc):
                raise
            time.sleep(min(max_delay, base_delay * (2**attempt)))
    return operation()


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

    from .model_aliases import resolve_hub_model, resolve_revision

    if (revision := resolve_revision(repo_id)) is not None:
        requested = kwargs.get("revision")
        if requested is not None and requested != revision:
            raise ValueError(
                f"{repo_id} is pinned to revision {revision}; "
                f"cannot download conflicting revision {requested}"
            )
        kwargs["revision"] = revision
        repo_id = resolve_hub_model(repo_id)
    if not disabled():
        if kwargs.get("local_files_only"):
            return str(hub_snapshot_download(repo_id, **kwargs))
        return str(
            _retry_transient_hub_call(lambda: hub_snapshot_download(repo_id, **kwargs))
        )
    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        return str(hub_snapshot_download(repo_id, **local_kwargs))
    except LocalEntryNotFoundError:
        raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None


def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
    """Resolve one Hub file locally when downloads are disabled."""
    from huggingface_hub import hf_hub_download as hub_file_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from .model_aliases import resolve_hub_model, resolve_revision

    if (revision := resolve_revision(repo_id)) is not None:
        requested = kwargs.get("revision")
        if requested is not None and requested != revision:
            raise ValueError(
                f"{repo_id} is pinned to revision {revision}; "
                f"cannot download conflicting revision {requested}"
            )
        kwargs["revision"] = revision
        repo_id = resolve_hub_model(repo_id)
    if not disabled():
        if kwargs.get("local_files_only"):
            return str(hub_file_download(repo_id, filename, **kwargs))
        return str(
            _retry_transient_hub_call(
                lambda: hub_file_download(repo_id, filename, **kwargs)
            )
        )
    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        return str(hub_file_download(repo_id, filename, **local_kwargs))
    except LocalEntryNotFoundError:
        raise ModelDownloadsDisabledError(repo_id, source() or FLAG) from None
