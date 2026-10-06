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

from starlette.applications import Starlette  # noqa: E402
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from rapid_mlx import server  # noqa: E402

_SCRATCH_APP = Starlette()


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


def test_c_middleware_installed_by_the_server_import_is_not_a_leftover():
    """The server module may be imported for the first time inside a test.
    What its import installs belongs to the app; what the test adds on top
    does not."""
    import_time = compile(
        "app.add_middleware(TrustedHostMiddleware)", "<server import>", "exec"
    )
    exec(  # noqa: S102 - simulates the module body of rapid_mlx.server
        import_time,
        {
            "__name__": "rapid_mlx.server",
            "app": _SCRATCH_APP,
            "TrustedHostMiddleware": TrustedHostMiddleware,
        },
    )
    _SCRATCH_APP.add_middleware(GZipMiddleware)
    assert [m.cls for m in _SCRATCH_APP.user_middleware] == [
        GZipMiddleware,
        TrustedHostMiddleware,
    ]


def test_d_only_the_import_time_middleware_survives():
    assert [m.cls for m in _SCRATCH_APP.user_middleware] == [TrustedHostMiddleware]
