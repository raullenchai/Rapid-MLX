# SPDX-License-Identifier: MIT
"""#3564: the remaining major server-side failures must each carry a stable,
client-safe ``error.code`` so the Desktop GUI maps them to a curated card
instead of the generic "Rapid couldn't finish that request" fallback.

This pins the ENGINE half of the contract for the failures that previously
had no code (or two divergent shapes):

* a model that fails to LOAD -> ``insufficient_memory`` (a load-time memory
  shortfall is the same user situation as a generation-time OOM) or, for any
  other cause, ``model_load_failed``;
* a request cancelled because the primary model was REPLACED under it ->
  ``model_replacement``, emitted identically by every lane that can surface it
  (the CancelledError translation and the engine-abort ``lifecycle`` kind);
* an unknown model on the chat route -> a dict envelope carrying
  ``model_not_found`` (was a bare-string 404 with no code).

Every payload's user-facing ``message`` must be a fixed, safe string -- the
raw exception text is inspected to pick the code but must NEVER leak. These
are pure-Python (no engine, no mlx): they run on the Linux lane.
"""

import asyncio
import errno

import pytest
from fastapi import HTTPException

from rapid_mlx.request import (
    ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY,
    MODEL_LOAD_FAILED_CODE,
    MODEL_REPLACEMENT_CODE,
    inference_aborted_error_payload,
    lifecycle_cancel_error_payload,
    model_load_error_payload,
)

# A raw exception message that MUST NOT leak into any client-facing payload.
_SECRET = "/Users/secret/prompt-fragment-and-internal-path.safetensors"


# ── model_replacement: one shape across every lane ──────────────────


def test_lifecycle_cancel_payload_shape():
    payload = lifecycle_cancel_error_payload()
    assert payload["code"] == MODEL_REPLACEMENT_CODE
    assert payload["type"] == "server_error"
    assert payload["param"] is None
    # Fixed, safe copy -- no ``str(exc)`` interpolation.
    assert payload["message"] == "Request cancelled by model replacement"


def test_lifecycle_cancel_is_single_source_of_truth():
    """The pre-commit HTTP 503 lane (``inference_aborted_error_payload`` with
    ``error_kind='lifecycle'``) must emit the EXACT same object the standalone
    helper does -- otherwise a model replacement reaches clients in two shapes
    (the bug this factoring fixed)."""

    class _LifecycleAbortError(Exception):
        error_kind = "lifecycle"

    assert (
        inference_aborted_error_payload(_LifecycleAbortError(_SECRET))
        == lifecycle_cancel_error_payload()
    )


# ── model load failure: classified into a stable code ───────────────


def test_model_load_payload_classifies_oom():
    exc = RuntimeError(f"Metal: unable to allocate 12GB buffer {_SECRET}")
    payload = model_load_error_payload(exc)
    assert payload["code"] == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY
    assert payload["type"] == "server_error"
    assert payload["param"] is None
    assert "memory" in payload["message"].lower()


def test_model_load_payload_classifies_other_as_load_failed():
    exc = FileNotFoundError(f"missing config.json at {_SECRET}")
    payload = model_load_error_payload(exc)
    assert payload["code"] == MODEL_LOAD_FAILED_CODE
    assert "load" in payload["message"].lower()


@pytest.mark.parametrize(
    "exc",
    [
        # Host RAM exhausted materialising weights -- a bare MemoryError with
        # no allocation-failure wording to match on.
        MemoryError(),
        # An mmap / host allocation failure while loading (POSIX ENOMEM).
        OSError(errno.ENOMEM, "Cannot allocate memory"),
    ],
)
def test_model_load_payload_classifies_canonical_oom_as_insufficient_memory(exc):
    """A load-time OOM raised as a canonical ``MemoryError`` or ``OSError(ENOMEM)``
    must map to ``insufficient_memory`` -- the memory card plus ``Retry-After``
    -- not the generic ``model_load_failed`` (wrong card, no retry hint)."""
    payload = model_load_error_payload(exc)
    assert payload["code"] == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY
    assert "memory" in payload["message"].lower()


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(f"Metal: unable to allocate {_SECRET}"),
        FileNotFoundError(f"missing config.json at {_SECRET}"),
    ],
)
def test_model_load_payload_never_leaks_raw_text(exc):
    """The raw exception (paths, prompt fragments) is inspected to pick the
    code but must never appear in the client-facing message."""
    payload = model_load_error_payload(exc)
    assert _SECRET not in payload["message"]
    assert str(exc) not in payload["message"]


# ── _raise_lifecycle_cancel_or_reraise: 503 dict envelope w/ code ───


class _EngineOwningTask:
    """A fake engine that claims ownership of the current task's abort -- the
    signal that this cancellation is a model replacement, not a client drop."""

    def consume_lifecycle_task_abort(self, task) -> bool:
        return True


class _EngineDisowningTask:
    def consume_lifecycle_task_abort(self, task) -> bool:
        return False


def test_lifecycle_cancel_raises_503_with_model_replacement_code():
    from rapid_mlx.service.helpers import _raise_lifecycle_cancel_or_reraise

    async def _run():
        with pytest.raises(HTTPException) as exc_info:
            _raise_lifecycle_cancel_or_reraise(
                _EngineOwningTask(), asyncio.CancelledError()
            )
        return exc_info.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert isinstance(exc.detail, dict)
    assert exc.detail["error"]["code"] == MODEL_REPLACEMENT_CODE
    # Same object the mid-stream SSE frame emits for the same event.
    assert exc.detail["error"] == lifecycle_cancel_error_payload()


def test_unowned_cancellation_reraises_untouched():
    """A cancellation the engine does NOT own (a real client disconnect, a
    shutdown) must propagate as CancelledError -- never masquerade as a
    model-replacement 503."""
    from rapid_mlx.service.helpers import _raise_lifecycle_cancel_or_reraise

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            _raise_lifecycle_cancel_or_reraise(
                _EngineDisowningTask(), asyncio.CancelledError()
            )

    asyncio.run(_run())


# ── ensure_engine_ready: load failure -> classified 503 ─────────────


def test_ensure_engine_ready_load_failure_maps_to_classified_503(monkeypatch):
    from rapid_mlx.service import helpers

    class _Lifecycle:
        def __init__(self, engine, exc):
            self.engine = engine
            self._exc = exc

        def acquire_request(self):
            pass

        def release_request(self):
            pass

        async def ensure_loaded(self):
            raise self._exc

    class _FakeConfig:
        def __init__(self, lifecycle):
            self.primary_model_lifecycle = lifecycle

    engine = object()
    exc = RuntimeError(f"Metal out of memory loading weights {_SECRET}")
    lifecycle = _Lifecycle(engine, exc)
    monkeypatch.setattr(helpers, "get_config", lambda: _FakeConfig(lifecycle))

    async def _run():
        with pytest.raises(HTTPException) as exc_info:
            await helpers.ensure_engine_ready(engine)
        return exc_info.value

    http = asyncio.run(_run())
    assert http.status_code == 503
    # A memory shortfall can clear if the user frees memory -> advertise retry.
    assert http.headers.get("Retry-After") == "5"
    assert http.detail["error"]["code"] == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY
    assert _SECRET not in http.detail["error"]["message"]


def test_ensure_engine_ready_generic_load_failure_keeps_retry_after_distinct_code(
    monkeypatch,
):
    """A generic (non-memory) load failure keeps the 503's standard
    ``Retry-After`` -- a load fault can be transient (a backend-init race, a
    passing I/O error) and we cannot tell that apart from a permanent one from
    the exception. What separates it from an OOM is the envelope ``code``
    (``model_load_failed`` -> "check files / choose another" card, distinct
    from the memory card), NOT the header."""
    from rapid_mlx.service import helpers

    class _Lifecycle:
        def __init__(self, engine, exc):
            self.engine = engine
            self._exc = exc

        def acquire_request(self):
            pass

        def release_request(self):
            pass

        async def ensure_loaded(self):
            raise self._exc

    class _FakeConfig:
        def __init__(self, lifecycle):
            self.primary_model_lifecycle = lifecycle

    engine = object()
    exc = FileNotFoundError(f"missing config.json at {_SECRET}")
    lifecycle = _Lifecycle(engine, exc)
    monkeypatch.setattr(helpers, "get_config", lambda: _FakeConfig(lifecycle))

    async def _run():
        with pytest.raises(HTTPException) as exc_info:
            await helpers.ensure_engine_ready(engine)
        return exc_info.value

    http = asyncio.run(_run())
    assert http.status_code == 503
    assert http.detail["error"]["code"] == MODEL_LOAD_FAILED_CODE
    # 503 keeps its standard retry hint regardless of the (non-memory) cause.
    assert http.headers.get("Retry-After") == "5"
    assert _SECRET not in http.detail["error"]["message"]


# ── chat route unknown model -> dict envelope with model_not_found ──


def test_validate_model_name_unknown_emits_model_not_found_envelope(monkeypatch):
    from rapid_mlx.service import helpers

    # Single-model mode where the served model is a LOCAL PATH: the 404 must
    # echo the client's own requested id but must NOT leak the server-side
    # path (this used to be interpolated as an ``Available:`` preview).
    server_path = "/Users/secret/local-weights/private-model"

    class _FakeConfig:
        model_registry = None
        model_name = server_path
        model_alias = None
        model_path = None

    monkeypatch.setattr(helpers, "get_config", lambda: _FakeConfig())

    with pytest.raises(HTTPException) as exc_info:
        helpers._validate_model_name("no-such-model-xyz")

    exc = exc_info.value
    assert exc.status_code == 404
    assert isinstance(exc.detail, dict), "must be a dict so code/type pass through"
    err = exc.detail["error"]
    assert err["code"] == "model_not_found"
    # The chat/count_tokens validator's established type (the GUI keys on the
    # code, so this stays consistent with the pre-existing count_tokens 404).
    assert err["type"] == "not_found_error"
    assert err["param"] == "model"
    # Echo the client's own input...
    assert "no-such-model-xyz" in err["message"]
    # ...but never the server's configured model path.
    assert server_path not in err["message"]
    assert "Available" not in err["message"]


def test_validate_model_name_empty_stays_400_bare(monkeypatch):
    """An empty model field keeps its existing 400 (a distinct client bug from
    an unknown model) -- the change is scoped to the unknown-name branch."""
    from rapid_mlx.service import helpers

    with pytest.raises(HTTPException) as exc_info:
        helpers._validate_model_name("")
    assert exc_info.value.status_code == 400
