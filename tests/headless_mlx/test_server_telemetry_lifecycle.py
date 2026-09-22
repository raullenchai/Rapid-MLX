# SPDX-License-Identifier: Apache-2.0
"""Linux coverage pins for the server telemetry lifecycle seam."""

from __future__ import annotations

from types import SimpleNamespace

import rapid_mlx.server as server
from rapid_mlx.telemetry import posthog_sender


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
