# SPDX-License-Identifier: Apache-2.0
"""Best-effort telemetry v2 event emission."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import rapid_mlx
from rapid_mlx.telemetry import build_gate, common_props, envelope, state, store


@dataclass(frozen=True)
class _ProcessContext:
    platform: common_props.PlatformFacts
    surface: str
    install_id: str
    session_id: str
    app_version: str
    channel: str


class _ActiveDayStore(Protocol):
    def claim_active_day(self) -> bool: ...


_context_lock = threading.Lock()
_context: _ProcessContext | None = None
_context_resolved = False
_surface: str | None = None
_app_opened_attempted = False


def _set_surface(surface: str) -> None:
    """Set the process surface before its first v2 event."""
    global _surface
    if surface not in ("cli", "server"):
        return
    with _context_lock:
        if _context_resolved:
            return
        _surface = surface


def _process_context() -> _ProcessContext | None:
    """Resolve immutable common fields once, including a failed build gate."""
    global _context, _context_resolved
    with _context_lock:
        if _context_resolved:
            return _context

        # The build gate must be the first operation: a developer checkout or
        # untrusted build must not even read machine facts, much less create an
        # install id or initialize an emit session.
        stamp = build_gate.official_build()
        if stamp is None:
            _context_resolved = True
            return None

        # The frozen snapshot cannot change during a process and is reused by
        # every event.
        platform = common_props.read_platform_facts()
        surface = _surface or "cli"

        # Importing emit can initialize no transport; keep it out of module
        # import so a mere ``import telemetry.track`` stays cheap.
        from rapid_mlx.telemetry import emit

        install_id = state.get_or_create_client_id()
        session_id = emit.session_id()
        app_version = rapid_mlx.__version__
        _context = _ProcessContext(
            platform=platform,
            surface=surface,
            install_id=install_id,
            session_id=session_id,
            app_version=app_version,
            channel=stamp.channel,
        )
        _context_resolved = True
        return _context


def _upload_allowed() -> bool:
    """Apply the no-side-effect gates in their required order."""
    if build_gate.official_build() is None:
        return False
    from rapid_mlx.telemetry import consent_runtime

    return consent_runtime.upload_allowed()


def track(
    event: str,
    props: Mapping[str, object],
    *,
    nth_model_served: int | None = None,
) -> None:
    """Queue one registry-approved v2 event without blocking or raising."""
    try:
        if not _upload_allowed():
            return
        context = _process_context()
        if context is None:
            return

        # ``note_model_served`` uses zero as its failure sentinel. A real
        # successful note is always at least one, so zero must stay off wire.
        nth = None if nth_model_served == 0 else nth_model_served
        common = common_props.build_common_props(
            surface=context.surface,
            install_id=context.install_id,
            session_id=context.session_id,
            app_version=context.app_version,
            channel=context.channel,
            nth_model_served=nth,
            days_since_first_run_bucket=store.days_since_first_run_bucket(),
            platform=context.platform,
        )
        if common is None:
            return
        item = envelope.build_batch_item(event, props, common)
        if item is None:
            return

        # Importing the sender registers an at-fork hook, so defer it until an
        # event has passed every earlier gate.
        from rapid_mlx.telemetry import posthog_sender

        posthog_sender.get_sender().capture(item)
    except Exception:
        return


def _emit_app_opened(surface: str) -> None:
    """Attempt ``app_opened`` once per process for an eligible process."""
    global _app_opened_attempted
    try:
        if not _upload_allowed():
            return
    except Exception:
        return
    with _context_lock:
        if _app_opened_attempted:
            return
        _app_opened_attempted = True
    _set_surface(surface)
    track("app_opened", {})


def start_lifecycle(surface: str) -> None:
    """Start one eligible process lifecycle without affecting its host."""
    try:
        if surface not in ("cli", "server") or not _upload_allowed():
            return
        from rapid_mlx.telemetry import posthog_sender

        posthog_sender.install_atexit()
        _emit_app_opened(surface)
    except Exception:
        return


def emit_active_day(*, _store: _ActiveDayStore = store) -> None:
    """Emit only for the first successful-inference claim of this UTC day."""
    try:
        if not _upload_allowed():
            return
        if _store.claim_active_day() is True:
            track("active_day", {})
    except Exception:
        return


def _reset_for_tests() -> None:
    """Clear process memoization and lifecycle latches."""
    global _app_opened_attempted, _context, _context_resolved, _surface
    with _context_lock:
        _context = None
        _context_resolved = False
        _surface = None
        _app_opened_attempted = False
