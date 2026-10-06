# SPDX-License-Identifier: Apache-2.0
"""The process-global server app must not stay "started" between tests.

Starlette freezes an app's middleware list on its first request. The real
``serve`` path configures CORS on ``rapid_mlx.server.app``, so a test that
drives it used to fail whenever an earlier test in the same process had sent
a request through that app — an order dependence that only showed in full
runs. ``conftest`` wraps every test in ``server_app_state_isolated``; the
tests here either enter that block themselves or run a nested pytest session,
so each holds on its own and in any order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx
pytest_plugins = ["pytester"]

from starlette.applications import Starlette  # noqa: E402
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from rapid_mlx import server  # noqa: E402
from tests.conftest import server_app_state_isolated  # noqa: E402


def test_global_app_is_configurable_again_after_a_block_started_it():
    baseline = list(server.app.user_middleware)
    with server_app_state_isolated():
        server.app.add_middleware(GZipMiddleware)
        TestClient(server.app).get("/__not_a_route__")
        assert server.app.middleware_stack is not None
        with pytest.raises(RuntimeError, match="Cannot add middleware"):
            server.app.add_middleware(GZipMiddleware)

    assert server.app.middleware_stack is None
    assert server.app.user_middleware == baseline
    # The real serve path can configure CORS again (undone by the fixture).
    server.configure_cors(["https://console.example"])


def test_global_app_is_unstarted_even_when_nothing_was_registered():
    with server_app_state_isolated():
        TestClient(server.app).get("/__not_a_route__")
        assert server.app.middleware_stack is not None
    assert server.app.middleware_stack is None


def test_middleware_installed_by_the_server_import_is_not_a_leftover():
    """The server module may be imported for the first time inside a test.
    What its import installs belongs to the app; what the test adds on top
    does not, and the stack built with it is discarded."""
    app = Starlette()
    import_time = compile(
        "app.add_middleware(TrustedHostMiddleware)", "<server import>", "exec"
    )
    with server_app_state_isolated():
        exec(  # noqa: S102 - simulates the module body of rapid_mlx.server
            import_time,
            {
                "__name__": "rapid_mlx.server",
                "app": app,
                "TrustedHostMiddleware": TrustedHostMiddleware,
            },
        )
        app.add_middleware(GZipMiddleware)
        assert [m.cls for m in app.user_middleware] == [
            GZipMiddleware,
            TrustedHostMiddleware,
        ]
        TestClient(app).get("/__not_a_route__")
        assert app.middleware_stack is not None

    assert [m.cls for m in app.user_middleware] == [TrustedHostMiddleware]
    assert app.middleware_stack is None


def test_add_middleware_is_restored_when_the_block_raises():
    original = Starlette.add_middleware
    app = Starlette()
    with pytest.raises(ValueError, match="boom"), server_app_state_isolated():
        app.add_middleware(GZipMiddleware)
        raise ValueError("boom")
    assert Starlette.add_middleware is original
    assert app.user_middleware == []


def test_fixture_restores_the_global_app_between_two_real_tests(pytester):
    """End to end, in a nested pytest run so it does not depend on the order
    or selection of the tests in this file: the first inner test starts the
    global app and registers middleware with no explicit isolation block,
    and the second finds the app un-started and without leftovers — which
    only the autouse fixture's teardown can have done."""
    pytester.makeconftest(
        "from tests.conftest import "
        "_unstart_global_server_app_after_each_test  # noqa: F401\n"
    )
    pytester.makepyfile(
        """
        import pytest
        from starlette.middleware.gzip import GZipMiddleware
        from starlette.testclient import TestClient

        from rapid_mlx import server


        def test_first_starts_the_app():
            server.app.add_middleware(GZipMiddleware)
            TestClient(server.app).get("/__not_a_route__")
            assert server.app.middleware_stack is not None
            with pytest.raises(RuntimeError, match="Cannot add middleware"):
                server.app.add_middleware(GZipMiddleware)


        def test_second_gets_a_configurable_app():
            assert server.app.middleware_stack is None
            assert not any(
                m.cls is GZipMiddleware for m in server.app.user_middleware
            )
            server.configure_cors(["https://console.example"])
        """
    )
    # Only the fixture under test matters to the inner run; the outer
    # session's optional plugins and warning filters do not apply there.
    result = pytester.runpytest_inprocess(
        "-p", "no:cacheprovider", "-p", "no:asyncio", "-W", "default"
    )
    result.assert_outcomes(passed=2)
