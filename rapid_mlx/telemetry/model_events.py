# SPDX-License-Identifier: Apache-2.0
"""Privacy-safe builders for telemetry v2 model lifecycle events."""

from __future__ import annotations

import errno
import functools
import re
import threading
import urllib.error
from collections.abc import Callable
from typing import ParamSpec

_serve_failure_lock = threading.Lock()
_serve_failure_claimed = False
_P = ParamSpec("_P")


def _never_raise(func: Callable[_P, None]) -> Callable[_P, None]:
    """Keep observability from changing a host command's result or output."""

    @functools.wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> None:
        try:
            func(*args, **kwargs)
        except Exception:
            return

    return wrapped


def size_bucket(size_bytes: int | None) -> str:
    """Map checkpoint bytes to the registry's closed GiB scale."""
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        return "unknown"
    gib = 1024**3
    for upper, token in (
        (gib, "lt_1gb"),
        (2 * gib, "1_2gb"),
        (4 * gib, "2_4gb"),
        (8 * gib, "4_8gb"),
        (16 * gib, "8_16gb"),
        (32 * gib, "16_32gb"),
        (64 * gib, "32_64gb"),
    ):
        if size_bytes < upper:
            return token
    return "64gb_plus"


def pull_error_class(exc: BaseException) -> str:
    """Classify a pull exception without putting its message on the wire.

    An ``HfHubHTTPError``, including one for a 5xx response, is ``other``
    because the server answered. ``network`` is reserved for failures to obtain
    a response.
    """
    import httpx
    from huggingface_hub.utils import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )
    from requests import exceptions as requests_exceptions

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, GatedRepoError):
            return "gated"
        if isinstance(current, RepositoryNotFoundError):
            return "not_found"
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return "disk_full"
        if isinstance(
            current,
            (
                LocalEntryNotFoundError,
                requests_exceptions.ConnectionError,
                requests_exceptions.ConnectTimeout,
                requests_exceptions.ReadTimeout,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                urllib.error.URLError,
                TimeoutError,
            ),
        ):
            return "network"
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
    return "other"


def serve_error_class(exc: BaseException) -> str:
    """Reduce loader failures to the registry's closed serve categories."""
    from rapid_mlx.request import (
        ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY,
        classify_engine_abort,
    )

    if classify_engine_abort(exc) == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY:
        return "insufficient_memory"
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import RepositoryNotFoundError

    if isinstance(exc, (HfHubHTTPError, RepositoryNotFoundError)):
        return "download_failed"
    # A missing local/Hub shard is an availability failure, not evidence that
    # bytes on disk are corrupt. ModuleNotFoundError is handled separately
    # below because mlx-lm uses it for an unknown architecture module.
    if isinstance(exc, FileNotFoundError) and not isinstance(exc, ModuleNotFoundError):
        return "download_failed"
    if isinstance(exc, ModuleNotFoundError):
        missing = exc.name or ""
        if missing.startswith("mlx_lm.models."):
            return "unsupported_architecture"
        text = str(exc)
        if re.fullmatch(r"No module named ['\"]mlx_lm\.models\.[^'\"]+['\"]", text):
            return "unsupported_architecture"
    elif isinstance(exc, ValueError):
        # mlx-lm/utils.py::_get_classes translates the module import failure to
        # exactly ``ValueError: Model type <X> not supported.``.
        if re.fullmatch(r"Model type .+ not supported\.?", str(exc)):
            return "unsupported_architecture"
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "safetensor" in name or any(
        marker in text
        for marker in ("safetensor", "corrupt", "checksum", "size mismatch")
    ):
        return "corrupt_weights"
    return "other"


def model_type(alias_or_path: object) -> str:
    """Derive the registry model type from catalog modality metadata."""
    if not isinstance(alias_or_path, str):
        return "other"
    try:
        from rapid_mlx.model_aliases import resolve_profile

        profile = resolve_profile(alias_or_path)
        if profile is None:
            from rapid_mlx.audio.registry import resolve_audio_alias

            return (
                "audio" if resolve_audio_alias(alias_or_path) is not None else "other"
            )
        modality = profile.modality
        if modality == "text":
            return "vlm" if profile.supports_image_input else "llm"
        closed_modalities = {
            "embedding": "embedding",
            "image-gen": "image-gen",
            "video-gen": "video-gen",
            "text-diffusion": "text-diffusion",
        }
        return closed_modalities.get(modality, "other")
    except Exception:
        pass
    return "other"


@_never_raise
def emit_model_pulled(
    model_ref: object, source: object, size_bytes: int | None
) -> None:
    from rapid_mlx.telemetry.model_id import telemetry_model_id
    from rapid_mlx.telemetry.track import track

    if source not in ("mirror", "hf"):
        return
    props = {
        "model": telemetry_model_id(model_ref),
        "model_type": model_type(model_ref),
        "source": source,
    }
    if size_bytes is not None:
        props["size_bucket"] = size_bucket(size_bytes)
    track("model_pulled", props)


@_never_raise
def emit_model_pull_failed(
    exc: BaseException,
    *,
    model_ref: object = None,
    source: object = None,
    size_bytes: int | None = None,
) -> None:
    from rapid_mlx.telemetry.model_id import telemetry_model_id
    from rapid_mlx.telemetry.track import track

    props: dict[str, object] = {"error_class": pull_error_class(exc)}
    if model_ref is not None:
        props["model"] = telemetry_model_id(model_ref)
        props["model_type"] = model_type(model_ref)
    if source in ("mirror", "hf"):
        props["source"] = source
    if size_bytes is not None:
        props["size_bucket"] = size_bucket(size_bytes)
    track("model_pull_failed", props)


def _quant_for_ref(alias_or_path: object) -> str:
    from rapid_mlx.telemetry.quant import quant_token

    quant_ref = alias_or_path if isinstance(alias_or_path, str) else ""
    if quant_ref:
        try:
            from rapid_mlx.model_aliases import resolve_profile

            profile = resolve_profile(quant_ref)
            if profile is not None and profile.hf_path:
                resolved_quant = quant_token(profile.hf_path)
                if resolved_quant != "unknown":
                    quant_ref = profile.hf_path
        except Exception:
            pass

    return quant_token(quant_ref)


def _serve_props(
    engine: object, alias_or_path: object, auto_selected: bool
) -> dict[str, object]:
    from rapid_mlx.telemetry.model_id import engine_telemetry_id, telemetry_model_id

    return {
        "model": (
            engine_telemetry_id(engine)
            if engine is not None
            else telemetry_model_id(alias_or_path)
        ),
        "model_type": model_type(alias_or_path),
        "auto_selected": bool(auto_selected),
        "quant": _quant_for_ref(alias_or_path),
    }


@_never_raise
def emit_model_served(
    engine: object, alias_or_path: object, auto_selected: bool
) -> None:
    """Emit a successful load and make the sole served-model store note."""
    from rapid_mlx.telemetry import store, track

    if not track._upload_allowed():
        return

    props = _serve_props(engine, alias_or_path, auto_selected)
    nth = store.note_model_served(str(props["model"]))
    track.track("model_served", props, nth_model_served=nth or None)


@_never_raise
def emit_model_serve_failed(
    exc: BaseException,
    *,
    engine: object = None,
    alias_or_path: object = None,
    auto_selected: bool = False,
) -> None:
    """Claim and emit at most one logical serve failure per process."""
    global _serve_failure_claimed
    with _serve_failure_lock:
        if _serve_failure_claimed:
            return

    from rapid_mlx.telemetry.model_id import engine_telemetry_id, telemetry_model_id
    from rapid_mlx.telemetry.track import track

    props: dict[str, object] = {"error_class": serve_error_class(exc)}
    if engine is not None:
        props["model"] = engine_telemetry_id(engine)
    elif alias_or_path is not None:
        props["model"] = telemetry_model_id(alias_or_path)
    if alias_or_path is not None:
        props["model_type"] = model_type(alias_or_path)
        props["auto_selected"] = bool(auto_selected)
        props["quant"] = _quant_for_ref(alias_or_path)
    # Build every potentially-failing property before claiming the one-shot
    # latch. A telemetry-only conversion bug must not suppress a later valid
    # failure event from this process.
    with _serve_failure_lock:
        if _serve_failure_claimed:
            return
        _serve_failure_claimed = True
    track("model_serve_failed", props)


def _reset_for_tests() -> None:
    global _serve_failure_claimed
    with _serve_failure_lock:
        _serve_failure_claimed = False
