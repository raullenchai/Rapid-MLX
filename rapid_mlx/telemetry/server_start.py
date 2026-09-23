# SPDX-License-Identifier: Apache-2.0
"""One-shot telemetry for the accepted serve invocation and startup outcome."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOAD_POLICIES = frozenset({"eager", "lazy", "none"})
_FAILURE_STAGES = frozenset(
    {"resolve", "download", "preflight", "prepare", "engine_start", "bind"}
)

_lock = threading.Lock()
_attempted = False
_terminal = False
_model_type: str | None = None
_load_policy: str | None = None
_current_failure_stage = "resolve"


def _track(state: str, *, failure_stage: str | None = None) -> None:
    """Build only registry-approved properties and never affect the host."""
    try:
        from rapid_mlx.telemetry.track import track

        props: dict[str, object] = {"state": state}
        if _model_type is not None:
            props["model_type"] = _model_type
        if _load_policy is not None:
            props["load_policy"] = _load_policy
        if state == "failed" and failure_stage is not None:
            props["failure_stage"] = failure_stage
        track("server_start_state", props)
    except BaseException:
        return


def attempted(model_ref: object = None, *, load_policy: object = None) -> None:
    """Emit the accepted invocation exactly once for this process."""
    global _attempted, _model_type, _load_policy
    try:
        from rapid_mlx.telemetry import posthog_sender
        from rapid_mlx.telemetry.model_events import model_type
        from rapid_mlx.telemetry.track import _upload_allowed

        if not _upload_allowed():
            return
        posthog_sender.install_atexit()
        resolved_type = model_type(model_ref)
    except BaseException:
        return
    resolved_policy = (
        load_policy
        if isinstance(load_policy, str) and load_policy in _LOAD_POLICIES
        else None
    )
    with _lock:
        if _attempted:
            return
        _model_type = resolved_type
        _load_policy = resolved_policy
        _attempted = True
    _track("attempted")


def ready() -> None:
    """Emit the sole successful terminal state after listener creation."""
    global _terminal
    with _lock:
        if not _attempted or _terminal:
            return
        _terminal = True
    _track("ready")


def failed(failure_stage: object) -> None:
    """Emit the sole failed terminal state with a closed startup stage."""
    global _terminal
    if not isinstance(failure_stage, str) or failure_stage not in _FAILURE_STAGES:
        return
    with _lock:
        if not _attempted or _terminal:
            return
        _terminal = True
    _track("failed", failure_stage=failure_stage)


def load_policy(model_ref: object, *, lazy_load: bool = False) -> str:
    """Return the actual pre-bind model-loading policy for a serve lane."""
    if lazy_load:
        return "lazy"
    try:
        from rapid_mlx.telemetry.model_events import model_type

        if model_type(model_ref) in {"audio", "image-gen", "video-gen"}:
            return "lazy"
    except BaseException:
        pass
    return "eager"


def set_failure_stage(failure_stage: str) -> None:
    """Select the stage used by the outer entrypoint exception guard."""
    global _current_failure_stage
    if failure_stage in _FAILURE_STAGES:
        _current_failure_stage = failure_stage


@contextmanager
def failure_stage(stage: str) -> Iterator[None]:
    """Emit ``failed(stage)`` if a startup boundary exits exceptionally."""
    global _current_failure_stage
    previous = _current_failure_stage
    set_failure_stage(stage)
    try:
        yield
    except BaseException:
        failed(stage)
        raise
    finally:
        _current_failure_stage = previous


def fail_current() -> None:
    """Fail at the stage selected by the active entrypoint boundary."""
    failed(_current_failure_stage)


def _reset_for_tests() -> None:
    global _attempted, _terminal, _model_type, _load_policy, _current_failure_stage
    with _lock:
        _attempted = False
        _terminal = False
        _model_type = None
        _load_policy = None
        _current_failure_stage = "resolve"
