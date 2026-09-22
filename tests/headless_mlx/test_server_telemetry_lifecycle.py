# SPDX-License-Identifier: Apache-2.0
"""Linux coverage pins for the server telemetry lifecycle seam."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import rapid_mlx.server as server
from rapid_mlx.telemetry import emit, model_events, posthog_sender


def test_server_shutdown_flush_is_best_effort(monkeypatch):
    calls: list[float] = []
    sender = SimpleNamespace(flush=lambda timeout: calls.append(timeout))
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)

    server._flush_v2_telemetry()
    assert calls == [2.0]

    monkeypatch.setattr(
        sender,
        "flush",
        lambda timeout: (_ for _ in ()).throw(RuntimeError("flush failed")),
    )
    server._flush_v2_telemetry()


@pytest.mark.asyncio
async def test_lifespan_load_failure_emits_model_failure(monkeypatch):
    failure = RuntimeError("load failed")
    lifecycle = SimpleNamespace(ensure_loaded=AsyncMock(side_effect=failure))
    engine = SimpleNamespace(_loaded=False)
    calls: list[tuple[BaseException, dict[str, object]]] = []

    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_primary_model_lifecycle", lifecycle)
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0.0)
    monkeypatch.setattr(server, "_model_alias", "tmax-9b")
    monkeypatch.setattr(server, "_model_path", "/unused/model-path")
    monkeypatch.setattr(server, "_telemetry_auto_selected", True)
    monkeypatch.setattr(emit, "error", lambda **_kwargs: None)
    monkeypatch.setattr(
        model_events,
        "emit_model_serve_failed",
        lambda exc, **kwargs: calls.append((exc, kwargs)),
    )

    lifespan = server.lifespan(server.app)
    with pytest.raises(RuntimeError, match="load failed") as exc_info:
        await lifespan.__anext__()

    assert exc_info.value is failure
    assert lifecycle.ensure_loaded.await_count == 1
    assert calls == [
        (
            failure,
            {
                "engine": engine,
                "alias_or_path": "tmax-9b",
                "auto_selected": True,
            },
        )
    ]
