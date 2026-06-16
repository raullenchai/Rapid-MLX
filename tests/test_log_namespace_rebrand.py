# SPDX-License-Identifier: Apache-2.0
"""
Tests for ``vllm_mlx/_log_namespace.py`` -- the runtime ``vllm_mlx.*`` ->
``rapid_mlx.*`` LogRecord-factory rebrand.

We import the module under test directly (not via ``import vllm_mlx``,
which would install the global factory and pollute every subsequent
test's caplog). For factory-install verification we save/restore the
process-wide factory and sentinel.
"""

from __future__ import annotations

import logging

import pytest

from vllm_mlx import _log_namespace
from vllm_mlx._log_namespace import (
    _rewrite_name,
    install_log_namespace_rebrand,
)


@pytest.fixture
def isolated_logging_factory():
    """Save and restore the global LogRecord factory + install sentinel.

    Each test that calls ``install_log_namespace_rebrand`` (or otherwise
    touches the global factory) must run inside this fixture so it doesn't
    leak state into the next test's caplog or into the rest of the suite.
    """
    saved_factory = logging.getLogRecordFactory()
    saved_sentinel = getattr(logging, _log_namespace._INSTALLED_SENTINEL, None)
    if hasattr(logging, _log_namespace._INSTALLED_SENTINEL):
        delattr(logging, _log_namespace._INSTALLED_SENTINEL)
    try:
        yield
    finally:
        logging.setLogRecordFactory(saved_factory)
        if hasattr(logging, _log_namespace._INSTALLED_SENTINEL):
            delattr(logging, _log_namespace._INSTALLED_SENTINEL)
        if saved_sentinel is not None:
            setattr(logging, _log_namespace._INSTALLED_SENTINEL, saved_sentinel)


# ---------------------------------------------------------------------------
# Pure ``_rewrite_name`` -- no side effects, no isolation needed.
# ---------------------------------------------------------------------------


def test_rewrite_exact_prefix():
    assert _rewrite_name("vllm_mlx") == "rapid_mlx"


def test_rewrite_dotted_child():
    assert _rewrite_name("vllm_mlx.server") == "rapid_mlx.server"


def test_rewrite_deeply_nested_child():
    assert _rewrite_name("vllm_mlx.service.helpers") == "rapid_mlx.service.helpers"


def test_does_not_rewrite_lookalike_prefix():
    # ``vllm_mlxfoo`` is not a child of ``vllm_mlx`` (no dot separator).
    # Defensive: we won't accidentally rebrand a third-party logger that
    # happens to start with the string.
    assert _rewrite_name("vllm_mlxfoo") == "vllm_mlxfoo"


def test_does_not_rewrite_unrelated_namespaces():
    for unrelated in (
        "uvicorn",
        "uvicorn.access",
        "fastapi",
        "asyncio",
        "httpx",
        "huggingface_hub",
        "",
    ):
        assert _rewrite_name(unrelated) == unrelated


# ---------------------------------------------------------------------------
# Factory installation -- side-effectful, isolated.
# ---------------------------------------------------------------------------


def test_factory_rewrites_vllm_mlx_records(isolated_logging_factory, caplog):
    install_log_namespace_rebrand()

    with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
        logging.getLogger("vllm_mlx.server").info("hello from server")

    matching = [r for r in caplog.records if r.message == "hello from server"]
    assert len(matching) == 1
    # The factory rewrites the name at record creation time, BEFORE caplog
    # captures it. So the captured record's name must already be rapid_mlx.*.
    assert matching[0].name == "rapid_mlx.server"


def test_factory_rewrites_deeply_nested_records(isolated_logging_factory, caplog):
    install_log_namespace_rebrand()

    with caplog.at_level(logging.INFO, logger="vllm_mlx.service.helpers"):
        logging.getLogger("vllm_mlx.service.helpers").info("[disconnect_guard] tick")

    matching = [r for r in caplog.records if "disconnect_guard" in r.message]
    assert len(matching) == 1
    assert matching[0].name == "rapid_mlx.service.helpers"


def test_factory_leaves_third_party_records_alone(isolated_logging_factory, caplog):
    """uvicorn / asyncio / httpx / etc. log records must NOT be rewritten.

    We have to set ``caplog.at_level`` per-logger because rapid-mlx's
    ``configure_logging`` (which runs as a side effect of importing
    ``vllm_mlx.server`` in the broader test session) pins httpx/httpcore/
    urllib3/huggingface_hub to WARNING. Setting INFO on root alone leaves
    those at WARNING and an INFO call is dropped before it ever reaches
    a record.
    """
    install_log_namespace_rebrand()

    third_party = ("uvicorn.access", "asyncio", "httpx", "huggingface_hub")
    with caplog.at_level(logging.INFO):
        for name in third_party:
            with caplog.at_level(logging.INFO, logger=name):
                logging.getLogger(name).info("third-party probe")

    names = {r.name for r in caplog.records if r.message == "third-party probe"}
    # Every third-party logger must capture its OWN name -- not a rebranded one.
    for name in third_party:
        assert name in names, f"third-party logger {name!r} record went missing"
        # And critically, no rebranded twin appears.
        assert f"rapid_mlx.{name}" not in names


def test_install_is_idempotent(isolated_logging_factory):
    """Calling install twice must NOT stack factories."""
    install_log_namespace_rebrand()
    factory_after_first = logging.getLogRecordFactory()

    install_log_namespace_rebrand()
    factory_after_second = logging.getLogRecordFactory()

    # Same factory object -- no second wrap.
    assert factory_after_first is factory_after_second


def test_install_preserves_existing_custom_factory(isolated_logging_factory):
    """If something else (structlog, a test fixture, etc.) has set a custom
    factory, install_log_namespace_rebrand must wrap it -- not replace it.
    """
    marker_attr = "_test_custom_factory_marker"

    def custom_factory(*args, **kwargs):
        record = logging.LogRecord(*args, **kwargs)
        setattr(record, marker_attr, True)
        return record

    logging.setLogRecordFactory(custom_factory)
    install_log_namespace_rebrand()

    record = logging.getLogRecordFactory()(
        "vllm_mlx.scheduler",
        logging.WARNING,
        __file__,
        0,
        "x",
        None,
        None,
    )
    # Custom factory's marker survived...
    assert getattr(record, marker_attr, False) is True
    # ...AND our rebrand ran on top.
    assert record.name == "rapid_mlx.scheduler"


def test_factory_survives_make_log_record_with_none_name(isolated_logging_factory):
    """``logging.makeLogRecord(dict)`` (socket / queue handlers reconstructing
    records from a wire-format dict) calls the active LogRecord factory with
    ``name=None`` and then patches ``record.__dict__`` afterwards. A naive
    ``record.name.startswith(...)`` in our wrapper would AttributeError. This
    test pins the guard so any future refactor that drops the ``isinstance``
    check fails fast.
    """
    install_log_namespace_rebrand()
    # Equivalent of what SocketHandler / QueueHandler do.
    record = logging.makeLogRecord(
        {
            "name": "vllm_mlx.routes.chat",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": "x",
        }
    )
    # makeLogRecord doesn't go through our factory's rename path (because
    # name=None at factory time, and the wire dict patches it AFTER our
    # wrapper has already returned). But the patched name is what handlers
    # eventually see -- and we must not crash producing the record.
    assert record.name == "vllm_mlx.routes.chat"


def test_install_rewraps_when_external_factory_is_swapped_in(
    isolated_logging_factory,
):
    """If another component (structlog, a test fixture) replaces the global
    factory AFTER ``install_log_namespace_rebrand`` has run, calling install
    again must re-wrap on top of the new factory -- not skip with "already
    installed". A bare-boolean sentinel would have silently left the new
    foreign factory un-rebranded, which is the bug codex round 1 caught.
    """
    install_log_namespace_rebrand()
    our_wrapper = logging.getLogRecordFactory()

    marker_attr = "_test_external_marker"

    def external_factory(*args, **kwargs):
        # Simulate structlog-style: ignore the wrapper above us, build a
        # fresh record from scratch with a custom attribute.
        record = logging.LogRecord(*args, **kwargs)
        setattr(record, marker_attr, True)
        return record

    logging.setLogRecordFactory(external_factory)
    # The active factory is no longer ours -- subsequent install must
    # re-wrap, not bail out.
    install_log_namespace_rebrand()
    new_wrapper = logging.getLogRecordFactory()

    assert new_wrapper is not our_wrapper, (
        "install must produce a NEW wrapper when an external factory took over"
    )
    assert new_wrapper is not external_factory, (
        "external factory must be wrapped, not active"
    )

    # Drive a record through the new chain: external_factory adds its
    # marker, our wrapper rebrands the name.
    record = new_wrapper(
        "vllm_mlx.scheduler",
        logging.INFO,
        __file__,
        0,
        "x",
        None,
        None,
    )
    assert record.name == "rapid_mlx.scheduler"
    assert getattr(record, marker_attr, False) is True


def test_factory_does_not_swallow_extra_args(isolated_logging_factory):
    """Ensure the factory passes through ``extra``/``stack_info``/etc.
    correctly -- mirroring what the standard library's default factory does.
    """
    install_log_namespace_rebrand()
    record = logging.getLogRecordFactory()(
        "vllm_mlx.engine_core",
        logging.INFO,
        "/path/to/file.py",
        42,
        "msg %s",
        ("arg",),
        None,  # exc_info
        "funcname",
        "stack info",
    )
    assert record.name == "rapid_mlx.engine_core"
    assert record.pathname == "/path/to/file.py"
    assert record.lineno == 42
    assert record.funcName == "funcname"
    assert record.stack_info == "stack info"
    assert record.getMessage() == "msg arg"
