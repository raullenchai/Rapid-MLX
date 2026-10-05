# SPDX-License-Identifier: Apache-2.0
"""The process-global server app must not stay "started" between tests.

Starlette freezes an app's middleware list on its first request. The real
``serve`` path configures CORS on ``rapid_mlx.server.app``, so a test that
drives it used to fail whenever an earlier test in the same process had sent
a request through that app — an order dependence that only showed in full
runs. ``conftest`` now resets the app after every test; these two tests pin
that, in file order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from rapid_mlx import server  # noqa: E402


def test_a_request_starts_the_global_app_and_registers_middleware():
    server.app.add_middleware(GZipMiddleware)
    TestClient(server.app).get("/__not_a_route__")
    assert server.app.middleware_stack is not None
    with pytest.raises(RuntimeError, match="Cannot add middleware"):
        server.app.add_middleware(GZipMiddleware)


def test_b_the_next_test_gets_a_configurable_app_without_leftovers():
    assert server.app.middleware_stack is None
    assert not any(m.cls is GZipMiddleware for m in server.app.user_middleware)
    server.configure_cors(["https://console.example"])
