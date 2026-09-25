# SPDX-License-Identifier: Apache-2.0
"""One-shot telemetry for the accepted serve invocation and startup outcome."""

from __future__ import annotations

import atexit
import json
import logging
import os
import stat
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
_owned_inflight_snapshot: tuple[int, int] | None = None
_marker_cleanup_registered = False

logger = logging.getLogger(__name__)

_MAX_MARKER_BYTES = 4096


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
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            file_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size > _MAX_MARKER_BYTES
            ):
                return None
            payload = bytearray()
            while len(payload) <= _MAX_MARKER_BYTES:
                chunk = os.read(fd, min(1024, _MAX_MARKER_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > _MAX_MARKER_BYTES:
                return None
        finally:
            os.close(fd)
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if marker_identity(value) is None:
        return None
    return cast(dict[str, object], value)


def _prepare_state_dir(path: Path) -> bool:
    """Create/open the private state directory without following a symlink."""
    try:
        try:
            state_stat = os.lstat(path)
        except FileNotFoundError:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            state_stat = os.lstat(path)
        if stat.S_ISLNK(state_stat.st_mode) or not stat.S_ISDIR(state_stat.st_mode):
            logger.warning("rapid-mlx state directory is unavailable: %s", path)
            return False
        if os.name == "nt":
            os.chmod(path, 0o700)
            opened_stat = os.lstat(path)
            if stat.S_ISLNK(opened_stat.st_mode) or (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ) != (state_stat.st_dev, state_stat.st_ino):
                return False
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                opened_stat = os.fstat(fd)
                if (opened_stat.st_dev, opened_stat.st_ino) != (
                    state_stat.st_dev,
                    state_stat.st_ino,
                ):
                    return False
                os.fchmod(fd, 0o700)
            finally:
                os.close(fd)
    except OSError as exc:
        logger.warning("rapid-mlx state directory is unavailable: %s", exc)
        return False
    return True


def _marker_snapshot(path: Path) -> tuple[int, int] | None:
    try:
        marker_stat = os.lstat(path)
    except OSError:
        return None
    return marker_stat.st_dev, marker_stat.st_ino


def _remove_marker_snapshot(path: Path, snapshot: tuple[int, int] | None) -> None:
    """Atomically quarantine and remove only the marker inode we inspected."""
    if snapshot is None or _marker_snapshot(path) != snapshot:
        return
    stale = path.with_name(f".{path.name}.stale-{os.getpid()}")
    try:
        os.lstat(stale)
    except FileNotFoundError:
        pass
    except OSError:
        return
    else:
        return
    try:
        os.rename(path, stale)
        stale_snapshot = _marker_snapshot(stale)
        if stale_snapshot != snapshot:
            # A replacement landed between the comparison and rename. Preserve it.
            try:
                os.link(stale, path)
            except FileExistsError:
                logger.warning(
                    "preserving quarantined serve marker %s because a newer "
                    "marker already exists at %s",
                    stale,
                    path,
                )
                return
            except OSError:
                try:
                    claim_fd = os.open(
                        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                except FileExistsError:
                    logger.warning(
                        "could not restore quarantined serve marker %s to %s; "
                        "a newer marker arrived",
                        stale,
                        path,
                    )
                    return
                os.close(claim_fd)
                try:
                    os.replace(stale, path)
                except OSError as exc:
                    path.unlink(missing_ok=True)
                    logger.warning(
                        "could not restore quarantined serve marker %s to %s: %r",
                        stale,
                        path,
                        exc,
                    )
                    return
            else:
                stale.unlink()
            return
        stale.unlink()
    except OSError as exc:
        logger.debug("could not remove serve marker %s: %r", path, exc)


def _atomic_write_marker(
    path: Path, *, startup_terminal: bool = False
) -> tuple[int, int]:
    identity = process_identity(os.getpid())
    if identity is None:
        raise OSError("could not determine current process identity")
    if not _prepare_state_dir(path.parent):
        raise OSError("state directory is unavailable")
    payload = json.dumps(
        {
            "pid": identity.pid,
            "create_time": identity.create_time,
            "boot_time": identity.boot_time,
            "app_version": rapid_mlx.__version__,
            "startup_terminal": startup_terminal,
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
            written_stat = os.fstat(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        snapshot = written_stat.st_dev, written_stat.st_ino
        try:
            if os.name == "nt":
                return snapshot
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            dir_fd = os.open(path.parent, flags)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            _remove_marker_snapshot(path, snapshot)
            raise
        return snapshot
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _begin_inflight_marker() -> tuple[bool, bool]:
    """Return ``(previous_unterminated, owns_marker)`` for this serve."""
    global _owned_inflight_snapshot
    _owned_inflight_snapshot = None
    path = _marker_path()
    previous_unterminated = False
    if not _prepare_state_dir(path.parent):
        return False, False
    try:
        markers = list(path.parent.glob("serve-inflight-*.json"))
    except OSError:
        markers = []
    for marker_path in markers:
        snapshot = _marker_snapshot(marker_path)
        pid = _marker_pid(marker_path)
        marker = _read_marker(marker_path)
        if marker is None or pid is None or marker.get("pid") != pid:
            _remove_marker_snapshot(marker_path, snapshot)
            continue
        if is_same_process(marker):
            continue
        previous_unterminated |= not bool(marker.get("startup_terminal", False))
        _remove_marker_snapshot(marker_path, snapshot)
    try:
        _owned_inflight_snapshot = _atomic_write_marker(path)
    except Exception as exc:
        logger.debug("could not write serve-inflight marker: %r", exc)
        return previous_unterminated, False
    return previous_unterminated, True


def _remove_inflight_marker() -> None:
    global _owned_inflight_snapshot, _owns_inflight_marker
    if not _owns_inflight_marker:
        return
    try:
        _remove_marker_snapshot(_marker_path(), _owned_inflight_snapshot)
    except Exception as exc:
        logger.debug("could not remove serve-inflight marker: %r", exc)
    finally:
        _owns_inflight_marker = False
        _owned_inflight_snapshot = None


def _mark_inflight_terminal() -> None:
    """Keep the liveness marker while recording successful startup."""
    global _owned_inflight_snapshot
    if not _owns_inflight_marker:
        return
    path = _marker_path()
    if _marker_snapshot(path) != _owned_inflight_snapshot:
        return
    try:
        _owned_inflight_snapshot = _atomic_write_marker(path, startup_terminal=True)
    except Exception as exc:
        logger.debug("could not mark serve startup terminal: %r", exc)


def _register_marker_cleanup() -> None:
    """Remove the process-lifetime marker on a clean interpreter exit."""

    global _marker_cleanup_registered
    if _marker_cleanup_registered:
        return
    atexit.register(_remove_inflight_marker)
    _marker_cleanup_registered = True


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
        if owns:
            _register_marker_cleanup()
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
        _terminal = True
        attempted_emitted = _attempted_emitted
        if not attempted_emitted:
            # No attempted event exists to balance, but the marker still has
            # to remain for crash-log liveness until clean process exit.
            _mark_inflight_terminal()
            return
    # Keep the marker for the lifetime of a ready server. Crash-log rotation
    # uses it to distinguish this process's open sink from stale diagnostics;
    # the atexit hook removes it on a clean shutdown. A failed telemetry write
    # likewise leaves the attempted marker behind for the next run to report.
    if _track("ready") is not False:
        with _lock:
            _mark_inflight_terminal()


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
    if _track("failed", failure_stage=failure_stage) is not False:
        _remove_inflight_marker()


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
    global _previous_run_unterminated, _owned_inflight_snapshot
    global _owns_inflight_marker
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
        _owned_inflight_snapshot = None
