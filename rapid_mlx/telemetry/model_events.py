# SPDX-License-Identifier: Apache-2.0
"""Privacy-safe builders for telemetry v2 model lifecycle events."""

from __future__ import annotations

import errno
import threading
import urllib.error

_serve_failure_lock = threading.Lock()
_serve_failure_claimed = False


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
    """Classify a pull exception without putting its message on the wire."""
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    if isinstance(exc, GatedRepoError):
        return "gated"
    if isinstance(exc, RepositoryNotFoundError):
        return "not_found"
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "disk_full"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "network"
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
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "safetensor" in name or any(
        marker in text
        for marker in ("safetensor", "corrupt", "checksum", "size mismatch")
    ):
        return "corrupt_weights"
    if any(
        marker in text
        for marker in (
            "unsupported architecture",
            "unsupported model type",
            "not supported for this checkpoint",
            "unknown model type",
        )
    ):
        return "unsupported_architecture"
    return "other"


def model_type(alias_or_path: object) -> str:
    """Derive the registry model type from catalog modality metadata."""
    if not isinstance(alias_or_path, str):
        return "other"
    try:
        from rapid_mlx.model_aliases import resolve_profile

        profile = resolve_profile(alias_or_path)
        if profile is None:
            return "other"
        modality = profile.modality
        if modality == "text":
            return "vlm" if profile.supports_image_input else "llm"
        if modality in {
            "embedding",
            "image-gen",
            "video-gen",
            "text-diffusion",
        }:
            return modality
    except Exception:
        pass
    return "other"


def emit_model_pulled(
    model_ref: object, source: object, size_bytes: int | None
) -> None:
    from rapid_mlx.telemetry.model_id import telemetry_model_id
    from rapid_mlx.telemetry.track import track

    if source not in ("mirror", "hf"):
        return
    track(
        "model_pulled",
        {
            "model": telemetry_model_id(model_ref),
            "source": source,
            "size_bucket": size_bucket(size_bytes),
        },
    )


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
    if source in ("mirror", "hf"):
        props["source"] = source
    if size_bytes is not None:
        props["size_bucket"] = size_bucket(size_bytes)
    track("model_pull_failed", props)


def _serve_props(
    engine: object, alias_or_path: object, auto_selected: bool
) -> dict[str, object]:
    from rapid_mlx.telemetry.model_id import engine_telemetry_id
    from rapid_mlx.telemetry.quant import quant_token

    return {
        "model": engine_telemetry_id(engine),
        "model_type": model_type(alias_or_path),
        "auto_selected": bool(auto_selected),
        "quant": quant_token(alias_or_path if isinstance(alias_or_path, str) else ""),
    }


def emit_model_served(
    engine: object, alias_or_path: object, auto_selected: bool
) -> None:
    """Emit a successful load and make the sole served-model store note."""
    from rapid_mlx.telemetry import store
    from rapid_mlx.telemetry.track import track

    props = _serve_props(engine, alias_or_path, auto_selected)
    nth = store.note_model_served(str(props["model"]))
    track("model_served", props, nth_model_served=nth or None)


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
        _serve_failure_claimed = True

    from rapid_mlx.telemetry.model_id import engine_telemetry_id, telemetry_model_id
    from rapid_mlx.telemetry.quant import quant_token
    from rapid_mlx.telemetry.track import track

    props: dict[str, object] = {"error_class": serve_error_class(exc)}
    if engine is not None:
        props["model"] = engine_telemetry_id(engine)
    elif alias_or_path is not None:
        props["model"] = telemetry_model_id(alias_or_path)
    if alias_or_path is not None:
        props["model_type"] = model_type(alias_or_path)
        props["auto_selected"] = bool(auto_selected)
        props["quant"] = quant_token(
            alias_or_path if isinstance(alias_or_path, str) else ""
        )
    track("model_serve_failed", props)


def _reset_for_tests() -> None:
    global _serve_failure_claimed
    with _serve_failure_lock:
        _serve_failure_claimed = False
