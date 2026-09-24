# SPDX-License-Identifier: Apache-2.0
"""One-shot telemetry for the accepted serve invocation and startup outcome."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import rapid_mlx
from rapid_mlx._process_identity import (
    is_same_process,
    marker_identity,
    process_identity,
)

_LOAD_POLICIES = frozenset({"eager", "lazy", "none"})
_FAILURE_STAGES = frozenset(
    {"resolve", "download", "preflight", "prepare", "engine_start", "bind"}
)

_lock = threading.Lock()
_attempted = False
_attempted_emitted = False
_terminal = False
_model_type: str | None = None
_load_policy: str | None = None
_current_failure_stage = "resolve"
_previous_run_unterminated = False
_owns_inflight_marker = False

logger = logging.getLogger(__name__)


def _marker_path() -> Path:
    """Resolve this process's serve marker beneath the existing state root."""
    from rapid_mlx.telemetry.state import _default_telemetry_dir

    return _default_telemetry_dir() / "state" / f"serve-inflight-{os.getpid()}.json"


def _marker_pid(path: Path) -> int | None:
    prefix = "serve-inflight-"
    if not path.name.startswith(prefix) or path.suffix != ".json":
        return None
    try:
        pid = int(path.stem.removeprefix(prefix))
    except ValueError:
        return None
    return pid if pid > 0 else None


def _read_marker(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if marker_identity(value) is None:
        return None
    return cast(dict[str, object], value)


def _atomic_write_marker(path: Path) -> None:
    identity = process_identity(os.getpid())
    if identity is None:
        raise OSError("could not determine current process identity")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(
        {
            "pid": identity.pid,
            "create_time": identity.create_time,
            "boot_time": identity.boot_time,
            "app_version": rapid_mlx.__version__,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("serve marker write made no progress")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _begin_inflight_marker() -> tuple[bool, bool]:
    """Return ``(previous_unterminated, owns_marker)`` for this serve."""
    path = _marker_path()
    previous_unterminated = False
    try:
        markers = list(path.parent.glob("serve-inflight-*.json"))
    except OSError:
        markers = []
    for marker_path in markers:
        pid = _marker_pid(marker_path)
        marker = _read_marker(marker_path)
        if marker is None or pid is None or marker.get("pid") != pid:
            try:
                marker_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(
                    "could not remove invalid serve marker %s: %r", marker_path, exc
                )
            continue
        if is_same_process(marker):
            continue
        previous_unterminated = True
        try:
            marker_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("could not remove stale serve marker %s: %r", marker_path, exc)
    try:
        _atomic_write_marker(path)
    except Exception as exc:
        logger.debug("could not write serve-inflight marker: %r", exc)
        return previous_unterminated, False
    return previous_unterminated, True


def _remove_inflight_marker() -> None:
    global _owns_inflight_marker
    if not _owns_inflight_marker:
        return
    try:
        _marker_path().unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("could not remove serve-inflight marker: %r", exc)
    finally:
        _owns_inflight_marker = False


def _track(state: str, *, failure_stage: str | None = None) -> bool:
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
        if state == "attempted" and _previous_run_unterminated:
            props["previous_run_unterminated"] = True
        return track("server_start_state", props)
    except BaseException:
        return False


def attempted(model_ref: object = None, *, load_policy: object = None) -> None:
    """Emit the accepted invocation exactly once for this process."""
    global _attempted, _attempted_emitted, _model_type, _load_policy
    global _previous_run_unterminated
    global _owns_inflight_marker
    resolved_policy = (
        load_policy
        if isinstance(load_policy, str) and load_policy in _LOAD_POLICIES
        else None
    )
    with _lock:
        if _attempted:
            return
        try:
            previous, owns = _begin_inflight_marker()
        except Exception as exc:
            logger.debug("could not initialize serve-inflight marker: %r", exc)
            previous, owns = False, False
        _previous_run_unterminated = previous
        _owns_inflight_marker = owns
        _load_policy = resolved_policy
        try:
            from rapid_mlx.telemetry import posthog_sender
            from rapid_mlx.telemetry.model_events import model_type
            from rapid_mlx.telemetry.track import _upload_allowed

            if not _upload_allowed():
                _attempted = True
                return
            posthog_sender.install_atexit()
            _model_type = model_type(model_ref)
        except BaseException:
            return
        if _track("attempted") is not False:
            _attempted = True
            _attempted_emitted = True


def ready() -> None:
    """Emit the sole successful terminal state after listener creation."""
    global _terminal
    try:
        from rapid_mlx._signal_observability import ensure_crash_sink

        ensure_crash_sink()
    except BaseException:
        pass
    with _lock:
        if _terminal:
            return
        if not _attempted_emitted:
            _remove_inflight_marker()
            return
        _terminal = True
        _remove_inflight_marker()
    _track("ready")


def failed(failure_stage: object) -> None:
    """Emit the sole failed terminal state with a closed startup stage."""
    global _terminal
    if not isinstance(failure_stage, str) or failure_stage not in _FAILURE_STAGES:
        return
    with _lock:
        if _terminal:
            return
        if not _attempted_emitted:
            _remove_inflight_marker()
            return
        _terminal = True
        _remove_inflight_marker()
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
    global _attempted, _attempted_emitted, _terminal, _model_type, _load_policy
    global _current_failure_stage
    global _previous_run_unterminated, _owns_inflight_marker
    with _lock:
        _remove_inflight_marker()
        _attempted = False
        _attempted_emitted = False
        _terminal = False
        _model_type = None
        _load_policy = None
        _current_failure_stage = "resolve"
        _previous_run_unterminated = False
        _owns_inflight_marker = False
