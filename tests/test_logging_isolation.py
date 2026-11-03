# SPDX-License-Identifier: Apache-2.0
"""Logging changes in a test, including teardown, cannot silence its successor."""

import inspect

import pytest

from tests import conftest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "mutation",
    [
        "logging.root.setLevel(logging.CRITICAL)",
        'logging.getLogger("rapid_mlx").setLevel(logging.CRITICAL)',
        "child.setLevel(logging.CRITICAL)",
        "child.propagate = False",
        'logging.getLogger("rapid_mlx").propagate = False',
        "child.disabled = True",
        "logging.root.disabled = True",
        'logging.getLogger("rapid_mlx").disabled = True',
        'logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})',
        "logging.disable(logging.CRITICAL)",
        "child.addFilter(reject)",
        "logging.root.handlers.clear()",
        "logging.root.handlers[0].addFilter(reject)",
        "logging.root.handlers[0].setLevel(logging.CRITICAL)",
        "child.addHandler(logging.NullHandler())",
        'late = logging.getLogger("rapid_mlx.late.child"); '
        "late.disabled = True; late.propagate = False; "
        "late.setLevel(logging.CRITICAL); late.addFilter(reject); "
        "late.addHandler(logging.NullHandler())",
    ],
)
def test_logging_state_restored_between_tests(pytester, mutation):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    fixture = inspect.getsource(conftest._aaa_isolate_logging_state.__wrapped__)
    pytester.makeconftest(
        "import logging\nimport logging.config\nimport pytest\n\n"
        + fixture
        + """
child = logging.getLogger("rapid_mlx.isolation.child")
reject = lambda record: False
configured = logging.getLogger("rapid_mlx.isolation.configured")
configured.setLevel(logging.ERROR)
configured.propagate = False
configured.disabled = True
original_handler = logging.NullHandler(logging.WARNING)
original_handler.addFilter(reject)
configured.addHandler(original_handler)
configured.addFilter(reject)

@pytest.fixture
def poison_on_teardown(request):
    request.addfinalizer(poison)

def poison():
    configured.setLevel(logging.DEBUG)
    configured.propagate = True
    configured.disabled = False
    configured.handlers.clear()
    configured.filters.clear()
    original_handler.setLevel(logging.CRITICAL)
    original_handler.filters.clear()
    """
        + mutation
        + "\n"
    )
    pytester.makepyfile("""
        import logging
        import conftest

        def test_polluter(poison_on_teardown):
            pass

        def test_victim(caplog):
            # Do not fix the threshold here: restoration must recover WARNING.
            child = logging.getLogger("rapid_mlx.isolation.child")
            child.warning("existing logger reached caplog")
            late = logging.getLogger("rapid_mlx.late.child")
            late.warning("new logger reached caplog")
            assert "existing logger reached caplog" in caplog.text
            assert "new logger reached caplog" in caplog.text
            assert child.level == logging.NOTSET
            assert child.propagate and not child.disabled
            assert not child.handlers and not child.filters
            assert late.level == logging.NOTSET
            assert late.propagate and not late.disabled
            assert not late.handlers and not late.filters
            assert not logging.root.disabled
            assert not logging.getLogger("rapid_mlx").disabled
            assert logging.root.manager.disable == logging.NOTSET
            configured = conftest.configured
            assert configured.level == logging.ERROR
            assert not configured.propagate and configured.disabled
            # Pytest also attaches capture handlers to non-propagating loggers.
            assert [h for h in configured.handlers if h not in logging.root.handlers] == [
                conftest.original_handler
            ]
            assert configured.filters == [conftest.reject]
            assert conftest.original_handler.level == logging.WARNING
            assert conftest.original_handler.filters == [conftest.reject]
    """)
    # dictConfig also closes/disables third-party loggers, outside the guard's
    # deliberately limited application namespace. Contain that to a child.
    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2)
