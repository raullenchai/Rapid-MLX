# SPDX-License-Identifier: Apache-2.0
"""H6 + H22 + C4 bundle — empirical reproducers + regression tests.

Each test block starts with a 1-2 sentence rationale citing the
reproducer that originally surfaced the bug, then pins the corrected
behavior. The structure is:

- H6 — OpenAI embeddings spec supports four input formats: ``str``,
  ``list[str]``, ``list[int]`` (pre-tokenized one input), and
  ``list[list[int]]`` (batch of pre-tokenized). Production pipelines
  using a shared tokenizer send the latter two; pre-PR these 422'd at
  parse time.
- H22 — Default request timeout was 300s, which silently cuts long
  reasoning generations. Industry baseline (vLLM, OpenAI proxy) is
  600-1800s. Bump to 1800.
- C4 — No admission control. A buggy client (or simple fork bomb) can
  schedule unbounded concurrent requests, OOM the Metal allocator,
  and crash the server for every other client. Add a cap.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# H6 — Pydantic model accepts all four OpenAI input shapes
# ---------------------------------------------------------------------------


class TestEmbeddingInputFourShapes:
    """Reproducer:
        curl /v1/embeddings -d '{"input": [[1,2,3]]}'   →  422 pre-PR

    The OpenAI spec
    (https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-input)
    lists all four shapes as valid; clients using a pre-tokenized
    pipeline (LangChain, LlamaIndex with custom tokenizer) send the
    int forms by default.
    """

    def test_str_accepted(self):
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input="hello")
        assert req.input == "hello"

    def test_list_str_accepted(self):
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=["a", "b"])
        assert req.input == ["a", "b"]

    def test_list_int_accepted(self):
        """list[int] — single pre-tokenized input."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=[101, 2023, 2003, 102])
        assert req.input == [101, 2023, 2003, 102]

    def test_list_list_int_accepted(self):
        """list[list[int]] — batch of pre-tokenized inputs."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=[[1, 2, 3], [4, 5, 6]])
        assert req.input == [[1, 2, 3], [4, 5, 6]]

    def test_mixed_str_and_int_rejected(self):
        """Sanity: mixing strings and ints in the same list is NOT in
        the spec and would be ambiguous (is [1, "a"] one tokenized
        input or one int + one string?). Stay strict to avoid
        silent-wrong behavior."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import EmbeddingRequest

        with pytest.raises(ValidationError):
            EmbeddingRequest(model="x", input=[1, "a", 3])

    def test_numeric_string_not_coerced_to_int(self):
        """Pydantic by default coerces ``"123"`` → 123. Without
        ``StrictInt``, ``[["1", "2"]]`` would silently become token
        ids [1, 2] — a different embedding from the words "1" and
        "2" the caller actually sent. Codex R1 caught this."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import EmbeddingRequest

        with pytest.raises(ValidationError):
            EmbeddingRequest(model="x", input=[["1", "2"]])
        with pytest.raises(ValidationError):
            EmbeddingRequest(model="x", input=["1", 2])

    def test_bool_not_accepted_as_int(self):
        """In Python, ``bool`` is a subclass of ``int`` — JSON ``true``
        would silently become token id 1 without ``StrictInt``. A
        client passing ``[true, false]`` clearly means a boolean
        feature, not token ids."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import EmbeddingRequest

        with pytest.raises(ValidationError):
            EmbeddingRequest(model="x", input=[True, False])


class TestEmbeddingRouteEmptyTokens:
    """Empty inner token lists were silently passed through pre-fix:
    ``[[]]`` produced a zero-width tensor and ``[[1, 2], []]`` gave
    one row whose attention mask is all zeros. The pooled embedding
    is then either NaN or a meaningless zero vector — silently wrong
    output to a vector store. Reject with 400 instead."""

    def test_empty_outer_list_rejected(self, monkeypatch):
        engine = MagicMock()
        engine.count_tokens.return_value = 0
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post("/v1/embeddings", json={"model": "any", "input": []})
        finally:
            restore()
        assert r.status_code == 400

    def test_empty_inner_token_list_rejected(self, monkeypatch):
        engine = MagicMock()
        engine.embed_tokens.return_value = [[0.0]]
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [[1, 2, 3], []]},
            )
        finally:
            restore()
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_double_wrapped_empty_rejected(self, monkeypatch):
        engine = MagicMock()
        engine.embed_tokens.return_value = [[0.0]]
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [[]]},
            )
        finally:
            restore()
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# H6 — route dispatches pre-tokenized inputs without re-tokenizing
# ---------------------------------------------------------------------------


def _build_embed_app(monkeypatch, engine):
    """Mount the embeddings router with a stubbed engine."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import embeddings as emb_route

    app = FastAPI()
    app.include_router(emb_route.router)

    cfg = get_config()
    saved = {
        "embedding_engine": cfg.embedding_engine,
        "embedding_model_locked": cfg.embedding_model_locked,
        "api_key": cfg.api_key,
    }
    cfg.embedding_engine = engine
    # H-09 (route guard) requires the embedding model to be configured.
    # These tests use ``model="any"`` so accept anything by locking the
    # route to a wildcard token that's then rejected only on the
    # mismatch branch — the test's own POSTs all use ``"any"`` so they
    # pass the lock check. Setting None here would now 400 at the
    # route guard instead of exercising the path under test.
    cfg.embedding_model_locked = "any"
    cfg.api_key = None

    monkeypatch.setattr(
        "vllm_mlx.server.load_embedding_model",
        lambda *_a, **_kw: None,
        raising=False,
    )

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)

    return TestClient(app), _restore


class TestEmbeddingRouteAcceptsTokenInputs:
    def test_list_int_input_uses_token_path(self, monkeypatch):
        """The engine's ``embed`` for str must NOT be called when input
        is already tokens — there's nothing to tokenize. Calling the
        string path on int input would coerce numbers to ``str(int)``
        and produce embeddings for the WORD "123", not the token id 123."""
        engine = MagicMock()
        engine.count_tokens.return_value = 4
        engine.embed_tokens.return_value = [[0.1, 0.2]]
        # If the route mistakenly hit the str path, this would fire:
        engine.embed.side_effect = AssertionError(
            "embed(str) called on pre-tokenized input"
        )
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [101, 2023, 2003, 102]},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        # Embed must have been called with the wrapped batch.
        engine.embed_tokens.assert_called_once()
        called_with = engine.embed_tokens.call_args[0][0]
        assert called_with == [[101, 2023, 2003, 102]]

    def test_list_list_int_input_passes_batch_through(self, monkeypatch):
        engine = MagicMock()
        engine.count_tokens.return_value = 6
        engine.embed_tokens.return_value = [[0.1, 0.2], [0.3, 0.4]]
        engine.embed.side_effect = AssertionError(
            "embed(str) called on pre-tokenized input"
        )
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [[1, 2, 3], [4, 5, 6]]},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        engine.embed_tokens.assert_called_once_with([[1, 2, 3], [4, 5, 6]])

    def test_str_input_still_uses_text_path(self, monkeypatch):
        """Regression: don't break the text path while adding the
        token path."""
        engine = MagicMock()
        engine.count_tokens.return_value = 3
        engine.embed.return_value = [[0.5, 0.5]]
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": "hi"},
            )
        finally:
            restore()
        assert r.status_code == 200
        engine.embed.assert_called_once()


class TestEmbeddingEngineEmbedTokens:
    """The engine must implement ``embed_tokens`` so the route has
    a place to send pre-tokenized batches."""

    def test_embed_tokens_method_exists(self):
        from vllm_mlx.embedding import EmbeddingEngine

        assert hasattr(EmbeddingEngine, "embed_tokens"), (
            "EmbeddingEngine must expose embed_tokens(list[list[int]]) "
            "for OpenAI spec input formats 3 and 4."
        )


# ---------------------------------------------------------------------------
# H22 — default_timeout 300s → 1800s
# ---------------------------------------------------------------------------


class TestDefaultTimeout:
    """Reproducer: a DeepSeek-R1 / Qwen-thinking generation that takes
    400s is silently truncated by the 300s default. 1800s (30 min)
    matches what vLLM and most OpenAI-compat proxies ship today."""

    def test_server_config_default_is_1800(self):
        from vllm_mlx.config.server_config import ServerConfig

        cfg = ServerConfig()
        assert cfg.default_timeout == 1800.0, (
            f"default_timeout regressed to {cfg.default_timeout}s. "
            "Reasoning models and 30B+ generations need >5min headroom; "
            "1800s is the post-PR baseline."
        )

    def test_server_module_default_matches_config(self):
        """If someone bumps one default and forgets the other, the
        CLI and the route layer disagree and timeouts get applied at
        whichever lower default the request happens to hit first."""
        import vllm_mlx.server as srv
        from vllm_mlx.config.server_config import ServerConfig

        assert srv._default_timeout == ServerConfig().default_timeout

    def test_cli_and_server_argparse_default_is_1800(self):
        """Codex R1 caught this: ServerConfig had been bumped to
        1800 but BOTH CLI argparse (vllm_mlx/cli.py) AND server
        argparse (vllm_mlx/server.py) still defaulted to 300, so
        ``rapid-mlx serve`` overwrote the config default at startup
        and users still got 5min.

        Source-grep instead of parser invocation because both
        parsers are constructed inline in ``main()``/equivalent and
        re-running them would import the world. Pin the literal
        ``default=1800.0`` near the ``--timeout`` flag in each file.
        """
        from pathlib import Path

        import vllm_mlx.cli as cli_mod
        import vllm_mlx.server as srv_mod

        for mod_label, mod in (("cli", cli_mod), ("server", srv_mod)):
            src = Path(mod.__file__).read_text()
            idx = src.find('"--timeout"')
            assert idx != -1, f"{mod_label}.py no longer declares --timeout"
            window = src[idx : idx + 400]
            assert "default=1800" in window, (
                f"{mod_label}.py --timeout default regressed away from "
                "1800.0 (set both this AND ServerConfig.default_timeout)"
            )


# ---------------------------------------------------------------------------
# C4 — admission control on concurrent requests
# ---------------------------------------------------------------------------


class TestAdmissionControl:
    """Reproducer: a fork-bomb client (or naive concurrent batch
    job) spawns N concurrent requests with large max_tokens; Metal
    allocator OOMs, server crashes, every other client gets 503/
    connection reset. Cap concurrent in-flight requests at a
    configurable max.
    """

    def test_scheduler_config_has_cap(self):
        from vllm_mlx.scheduler import SchedulerConfig

        cfg = SchedulerConfig()
        assert hasattr(cfg, "max_concurrent_requests"), (
            "SchedulerConfig must expose max_concurrent_requests for "
            "admission control (default conservative)."
        )
        # Default must be set (not None) — admission control is on by default.
        assert cfg.max_concurrent_requests is not None
        assert cfg.max_concurrent_requests > 0

    def test_add_request_raises_backpressure_at_cap(self):
        """Driving ``Scheduler.add_request`` directly — not a re-
        implemented copy of the gate — proves the production cap
        check fires before tokenization. Codex R2 flagged the earlier
        version as test-by-accident because it inlined the gate
        logic; this version constructs a real ``Scheduler`` instance
        (via ``__new__`` so we skip the expensive
        tokenizer/model/engine wiring) and calls the bound method."""
        from vllm_mlx.request import Request, SamplingParams
        from vllm_mlx.scheduler import BackpressureError, Scheduler, SchedulerConfig

        # 1) The class itself must be an ordinary Exception subclass so
        #    handlers can ``except BackpressureError`` safely.
        assert issubclass(BackpressureError, Exception)

        # 2) Build a minimal Scheduler stand-in with the in-flight
        #    dict pre-populated to cap. Skip __init__ — full
        #    construction needs a tokenizer + model + ~20 args; we
        #    only need ``self.requests`` and ``self.config`` for the
        #    gate to fire.
        sched = Scheduler.__new__(Scheduler)
        sched.config = SchedulerConfig(max_concurrent_requests=2)
        sched.requests = {"req-1": object(), "req-2": object()}

        new_req = Request(
            request_id="req-3",
            prompt="hi",
            sampling_params=SamplingParams(max_tokens=8),
        )

        # The real ``add_request`` runs the cap check at the top —
        # any later attribute access (tokenizer / block_aware_cache /
        # …) would AttributeError on this bare stub, so a passing
        # ``raises(BackpressureError)`` here is proof the gate fired
        # *first*.
        with pytest.raises(BackpressureError):
            Scheduler.add_request(sched, new_req)

        # 3) Below cap → the gate passes silently. We intercept the
        #    very next attribute access (``self.tokenizer``) to stop
        #    before tokenization without needing a real model.
        sched.requests = {"req-1": object()}
        below_cap_req = Request(
            request_id="req-3",
            prompt="hi",
            sampling_params=SamplingParams(max_tokens=8),
        )
        # ``AttributeError`` proves the gate did not raise — execution
        # advanced past the cap check into the tokenize step that our
        # bare stub doesn't satisfy. If the gate had spuriously raised
        # ``BackpressureError`` below the cap, that exception would
        # surface here instead.
        with pytest.raises(AttributeError):
            Scheduler.add_request(sched, below_cap_req)

    def test_admission_returns_503_with_retry_after(self, monkeypatch):
        """End-to-end: a request that would push in-flight over the
        cap returns 503 with a Retry-After header (RFC 9110 §10.2.4).
        Backed-off clients can then retry without further
        ceremony."""
        # Build a stub chat route that hits a stub engine; the engine's
        # generate() raises BackpressureError to simulate cap-exceeded.
        from vllm_mlx.config import get_config
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import BackpressureError

        app = FastAPI()
        app.include_router(chat_route.router)

        engine = MagicMock()
        engine.is_mllm = False
        # Tool-call parser / guided gen short-circuits we don't want
        # to hit on this path.
        engine.supports_guided_generation = False

        async def _boom(*_a, **_kw):
            raise BackpressureError("max_concurrent_requests exceeded")

        # The chat route invokes ``engine.chat(...)`` on the
        # non-streaming, non-guided path (see routes/chat.py:597).
        engine.chat = _boom

        cfg = get_config()
        saved = {
            "engine": cfg.engine,
            "model_name": cfg.model_name,
            "model_alias": cfg.model_alias,
            "model_path": cfg.model_path,
            "model_registry": cfg.model_registry,
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "ready": cfg.ready,
            "api_key": cfg.api_key,
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") is not None
        # Body should hint at backpressure so SDK error messages are useful.
        detail = r.json().get("detail", "").lower()
        assert "concurrent" in detail or "backpressure" in detail or "busy" in detail

    def test_mllm_scheduler_has_cap(self):
        """Codex R1 caught this: cap was on the LLM SchedulerConfig
        only, so MLLM requests could bypass admission entirely. Mirror
        the field on MLLMSchedulerConfig and exercise the gate."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        cfg = MLLMSchedulerConfig()
        assert hasattr(cfg, "max_concurrent_requests")
        assert cfg.max_concurrent_requests is not None
        assert cfg.max_concurrent_requests > 0

    def test_mllm_add_request_raises_at_cap(self):
        """Pin the actual MLLM gate: pre-populate ``requests`` up to
        the cap, then call add_request and expect BackpressureError.
        Codex R1's prior test only checked the class existed."""
        from vllm_mlx.mllm_scheduler import (
            MLLMScheduler,
            MLLMSchedulerConfig,
        )
        from vllm_mlx.scheduler import BackpressureError

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.config = MLLMSchedulerConfig(max_concurrent_requests=1)
        sched.requests = {"req-0": MagicMock()}
        sched.waiting = []

        with pytest.raises(BackpressureError):
            sched.add_request(prompt="hi")

    def test_streaming_admission_returns_503(self, monkeypatch):
        """Codex R1's biggest miss: the streaming path didn't 503 —
        ``_disconnect_guard`` swallowed BackpressureError into an SSE
        error chunk on a 200 stream. Pre-flight ``check_admission``
        at route entry must surface 503 BEFORE StreamingResponse
        starts. Triggered by setting ``engine.check_admission`` to
        raise (simulating a saturated scheduler)."""
        from vllm_mlx.config import get_config
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import BackpressureError

        app = FastAPI()
        app.include_router(chat_route.router)

        engine = MagicMock()
        engine.is_mllm = False
        engine.supports_guided_generation = False

        def _block():
            raise BackpressureError("cap exceeded")

        engine.check_admission = _block

        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "engine",
                "model_name",
                "model_alias",
                "model_path",
                "model_registry",
                "tool_call_parser",
                "reasoning_parser",
                "ready",
                "api_key",
            )
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") is not None

    def test_check_admission_reservation_is_atomic(self):
        """Codex R2 BLOCKER closure: ``check_admission`` is *reserve-
        on-success*, not check-then-act. Two callers racing at cap-1
        cannot both succeed — exactly one wins the slot, the other
        raises ``BackpressureError``. Without the reservation counter,
        both would pass and the loser would only fail later inside
        ``add_request`` (too late for streaming to return a clean
        HTTP 503).

        Drives the real ``BatchedEngine.check_admission`` against a
        synthesised scheduler stub so we exercise the production
        lock/counter, not a copy of the gate logic."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        eng = BatchedEngine.__new__(BatchedEngine)
        eng._is_mllm = False
        eng._mllm_scheduler = None
        eng._admission_lock = threading.Lock()
        eng._admission_reservations = 0

        # Synthetic scheduler with cap=1. ``check_admission`` only
        # reads ``scheduler.config.max_concurrent_requests`` for the
        # cap; the in-flight count comes from ``_admission_reservations``
        # under the engine lock, so we don't need to populate
        # ``scheduler.requests`` here.
        #
        # ``self._engine`` is the ``AsyncEngineCore`` wrapper; the
        # real ``Scheduler`` lives on its inner ``.engine`` attribute.
        # Mirror that nesting so the production lookup path is
        # exercised, not just the dataclass — codex R4 BLOCKER root
        # cause was the lookup walking the wrong indirection.
        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=1)
        scheduler_stub.requests = {}

        inner_engine_stub = _Stub()
        inner_engine_stub.scheduler = scheduler_stub

        async_engine_stub = _Stub()
        async_engine_stub.engine = inner_engine_stub
        eng._engine = async_engine_stub

        # First reservation: succeeds (counter 0 → 1).
        eng.check_admission()
        assert eng._admission_reservations == 1

        # Second reservation at cap: must raise. Without the atomic
        # reserve-on-success, a check-then-act gate would let both
        # through because ``scheduler.requests`` is still empty.
        with pytest.raises(BackpressureError):
            eng.check_admission()

        # Counter unchanged after the failed reservation — the cap
        # check happens before the increment under the same lock,
        # so a raise must not bump the counter.
        assert eng._admission_reservations == 1

        # Release returns the slot; next reservation succeeds again.
        eng.release_admission_reservation()
        assert eng._admission_reservations == 0
        eng.check_admission()
        assert eng._admission_reservations == 1

        # Release floor: extra releases do not drive the counter
        # negative (idempotent below zero).
        eng.release_admission_reservation()
        eng.release_admission_reservation()
        eng.release_admission_reservation()
        assert eng._admission_reservations == 0

    def test_validation_error_does_not_leak_admission_slot(self, monkeypatch):
        """Codex R3 BLOCKER closure: a 400 from validation (here,
        ``messages=[]``) raised AFTER ``_check_admission_or_503``
        reserved a slot must not leak the reservation. Under the old
        flow, after ``max_concurrent_requests`` such bad requests the
        cap was permanently exhausted and every subsequent valid
        request received 503 until restart.

        The route-level ``finally`` calls
        ``_release_admission_unless_committed`` so the slot is
        returned. Uses a ``MagicMock`` engine with the real atomic
        admission counter + lock + ``release_admission_reservation``
        method attached so the helper-side accounting under the
        FastAPI test client is exercised end-to-end."""
        import threading

        from vllm_mlx.config import get_config
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.middleware.exception_handlers import install_exception_handlers
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import SchedulerConfig

        app = FastAPI()
        # D-ANTHRO-VALIDATION F11: install the shared exception
        # handlers so the canonical 400 envelope fires when Pydantic
        # rejects ``messages=[]`` at the new ``min_length=1`` constraint.
        install_exception_handlers(app)
        app.include_router(chat_route.router)

        # MagicMock engine — ``MagicMock`` doesn't enforce
        # ``BatchedEngine``'s read-only ``@property`` definitions, so
        # we can set ``is_mllm`` / ``supports_guided_generation``
        # directly while still binding the real admission methods to
        # exercise the production accounting.
        engine = MagicMock()
        engine.is_mllm = False
        engine.supports_guided_generation = False

        # Wire the real admission state onto the mock. The route
        # handler calls ``engine.check_admission()`` and
        # ``engine.release_admission_reservation()`` — both bind
        # through ``BatchedEngine`` against the mock's namespace, so
        # ``engine._admission_reservations`` is what we'll assert
        # against at the end.
        engine._admission_lock = threading.Lock()
        engine._admission_reservations = 0
        engine._is_mllm = False
        engine._mllm_scheduler = None

        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=2)
        scheduler_stub.requests = {}

        # ``self._engine.engine.scheduler`` — mirror the real
        # AsyncEngineCore-wrapping-EngineCore nesting (codex R4).
        inner_engine_stub = _Stub()
        inner_engine_stub.scheduler = scheduler_stub
        async_engine_stub = _Stub()
        async_engine_stub.engine = inner_engine_stub
        engine._engine = async_engine_stub

        # Bind the real methods so the counter is the source of truth.
        engine.check_admission = lambda: BatchedEngine.check_admission(engine)
        engine.release_admission_reservation = lambda: (
            BatchedEngine.release_admission_reservation(engine)
        )

        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "engine",
                "model_name",
                "model_alias",
                "model_path",
                "model_registry",
                "tool_call_parser",
                "reasoning_parser",
                "ready",
                "api_key",
            )
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            # Fire 5 requests that all fail validation with empty
            # ``messages``. With cap=2, the old leaky flow would
            # exhaust the cap after the 2nd request and the 3rd+
            # would return 503 instead of 400.
            statuses = []
            for _ in range(5):
                r = client.post(
                    "/v1/chat/completions",
                    json={"model": "stub", "messages": []},
                )
                statuses.append(r.status_code)
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        # Every request must surface as 400 (validation) — never 503
        # (admission exhaustion). The reservation counter must end at
        # zero, proving each request's slot was released by the
        # route-level finally.
        assert statuses == [400, 400, 400, 400, 400], statuses
        assert engine._admission_reservations == 0

    def test_check_admission_finds_llm_scheduler_through_async_wrapper(self):
        """Codex R4 BLOCKER closure: the LLM admission lookup must
        walk through ``AsyncEngineCore`` to its inner ``EngineCore``
        scheduler. Pre-fix it did ``getattr(self._engine,
        "scheduler", None)`` which silently returned ``None`` on
        every text engine — so streaming text requests at cap
        degraded to 200 SSE error chunks instead of 503.

        Reproduce the exact production nesting (``self._engine`` is
        the wrapper; the scheduler is at ``self._engine.engine.scheduler``)
        and assert the gate actually fires at the cap. Without the
        fix the second ``check_admission`` would return silently
        because ``cap`` would be derived from a missing scheduler."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        eng = BatchedEngine.__new__(BatchedEngine)
        eng._is_mllm = False
        eng._mllm_scheduler = None
        eng._admission_lock = threading.Lock()
        eng._admission_reservations = 0

        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=1)
        scheduler_stub.requests = {}

        # Match the production indirection exactly.
        inner_engine_stub = _Stub()
        inner_engine_stub.scheduler = scheduler_stub
        async_engine_stub = _Stub()
        async_engine_stub.engine = inner_engine_stub
        eng._engine = async_engine_stub

        eng.check_admission()
        assert eng._admission_reservations == 1
        with pytest.raises(BackpressureError):
            eng.check_admission()

        # Sanity: without the inner indirection (the pre-R4 lookup
        # path), AND without an explicit ``_scheduler_config``, the
        # gate must still enforce the SchedulerConfig dataclass
        # default of 256 (codex R10 closure — the prior behavior
        # of silently no-op'ing here let cold-start streaming
        # bursts slip past preflight and only the late
        # scheduler-level BackpressureError fired, degrading to a
        # 200 SSE error chunk).
        eng_no_inner = BatchedEngine.__new__(BatchedEngine)
        eng_no_inner._is_mllm = False
        eng_no_inner._mllm_scheduler = None
        eng_no_inner._admission_lock = threading.Lock()
        eng_no_inner._admission_reservations = 0
        eng_no_inner._scheduler_config = None
        # ``self._engine`` exists but has no ``engine`` attribute —
        # the pre-R4 shape. The gate now reserves under the
        # default cap (256) rather than no-op'ing.
        eng_no_inner._engine = _Stub()
        eng_no_inner.check_admission()
        assert eng_no_inner._admission_reservations == 1

    def test_check_admission_uses_scheduler_config_during_cold_start(self):
        """Codex R6 P2 closure: during cold-start (``self._engine``
        not yet wired or its inner ``engine.scheduler`` not yet
        constructed), a burst of streaming requests must still be
        gated against the *configured* cap, not slip through and have
        the loser raise ``BackpressureError`` inside the response
        generator.

        Reproduce the cold-start window by leaving ``self._engine``
        unset and assert ``check_admission`` reads the cap from
        ``self._scheduler_config`` and reserves under the atomic
        lock just like the post-init path."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        eng = BatchedEngine.__new__(BatchedEngine)
        eng._is_mllm = False
        eng._mllm_scheduler = None
        # No engine yet — pre-``start()`` cold-start window.
        eng._engine = None
        eng._admission_lock = threading.Lock()
        eng._admission_reservations = 0
        eng._scheduler_config = SchedulerConfig(max_concurrent_requests=1)

        eng.check_admission()
        assert eng._admission_reservations == 1
        # Second reservation: cap reached, must raise.
        with pytest.raises(BackpressureError):
            eng.check_admission()
        assert eng._admission_reservations == 1

    def test_mllm_scheduler_inherits_configured_concurrent_cap(self):
        """Codex R5 closure: a server started with
        ``SchedulerConfig(max_concurrent_requests=N)`` must apply the
        same cap to the MLLM scheduler. Pre-fix, ``_start_mllm`` built
        ``MLLMSchedulerConfig(...)`` without forwarding the field, so
        the MLLM admission gate always saw the default 256 and ignored
        memory-constrained deployments' lower cap.

        Drives the cap propagation directly: read the field off a
        ``SchedulerConfig`` instance and assert the resulting
        ``MLLMSchedulerConfig`` carries it."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
        from vllm_mlx.scheduler import SchedulerConfig

        configured = SchedulerConfig(max_concurrent_requests=4)
        # Mirror the propagation site in ``_start_mllm``.
        forwarded = getattr(configured, "max_concurrent_requests", 256)
        assert forwarded == 4
        mllm_cfg = MLLMSchedulerConfig(max_concurrent_requests=forwarded)
        assert mllm_cfg.max_concurrent_requests == 4

        # Sanity: omitting the source field falls back to 256 — same
        # as ``MLLMSchedulerConfig``'s dataclass default. Codex R8
        # caught the prior version which forwarded ``None`` here,
        # silently disabling both the engine-level
        # ``check_admission`` and the scheduler-level
        # ``MLLMScheduler.add_request`` cap check.
        bare = object()
        forwarded = getattr(bare, "max_concurrent_requests", 256)
        assert forwarded == 256
        assert (
            MLLMSchedulerConfig(
                max_concurrent_requests=forwarded
            ).max_concurrent_requests
            == 256
        )

    def test_cloud_routed_chat_releases_local_admission_slot(self, monkeypatch):
        """Codex R8 P2 closure: when ``cfg.cloud_router`` decides to
        route a chat completion to the cloud, the local admission
        slot reserved at route entry must be released immediately —
        the cloud round-trip uses no local scheduler/Metal resources.
        Without this, a burst of long cloud-routed requests could
        exhaust the local cap and 503 unrelated local requests while
        the local engine sits idle."""
        import threading

        from vllm_mlx.config import get_config
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import SchedulerConfig

        app = FastAPI()
        app.include_router(chat_route.router)

        # Build a MagicMock engine with the real admission methods so
        # the counter is the source of truth (mirrors the R3 test).
        engine = MagicMock()
        engine.is_mllm = False
        engine.supports_guided_generation = False
        # ``build_prompt`` + ``estimate_new_tokens`` are read by the
        # cloud-routing branch — return non-zero so the threshold
        # check fires.
        engine.build_prompt.return_value = "prompt"
        engine.estimate_new_tokens.return_value = (1000, 1000)
        # ``preserve_native_tool_format`` is read during message prep.
        engine.preserve_native_tool_format = False

        engine._admission_lock = threading.Lock()
        engine._admission_reservations = 0
        engine._is_mllm = False
        engine._mllm_scheduler = None

        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=2)
        scheduler_stub.requests = {}
        inner_engine_stub = _Stub()
        inner_engine_stub.scheduler = scheduler_stub
        async_engine_stub = _Stub()
        async_engine_stub.engine = inner_engine_stub
        engine._engine = async_engine_stub

        engine.check_admission = lambda: BatchedEngine.check_admission(engine)
        engine.release_admission_reservation = lambda: (
            BatchedEngine.release_admission_reservation(engine)
        )

        # Stub the cloud router: ``should_route_to_cloud`` returns True
        # so the branch fires; ``completion`` returns a minimal dict.
        cloud_router = MagicMock()
        cloud_router.should_route_to_cloud.return_value = True
        cloud_router.threshold = 500
        cloud_router.cloud_model = "cloud-x"

        async def _fake_completion(*_a, **_kw):
            return {
                "id": "cloud-1",
                "object": "chat.completion",
                "created": 0,
                "model": "cloud-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

        cloud_router.completion = _fake_completion

        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "engine",
                "model_name",
                "model_alias",
                "model_path",
                "model_registry",
                "tool_call_parser",
                "reasoning_parser",
                "ready",
                "api_key",
                "cloud_router",
                "pin_system_prompt",
            )
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None
        cfg.cloud_router = cloud_router
        cfg.pin_system_prompt = False

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        # Cloud completion succeeded → 200; the local admission slot
        # must have been released before the cloud call returned, so
        # the reservation counter is back at zero.
        assert r.status_code == 200, r.text
        assert engine._admission_reservations == 0

    def test_scheduler_config_accepts_explicit_admission_cap(self):
        """Codex R7 closure (via explicit CLI flag rather than
        auto-tracking ``max_num_seqs``): operators on memory-
        constrained devices can pass ``--max-concurrent-requests N``
        to lower the admission cap independently of
        ``--max-num-seqs``. The dataclass default stays at 256 so
        existing tests that intentionally send more requests than
        ``max_num_seqs`` to exercise the queue still work."""
        from vllm_mlx.scheduler import SchedulerConfig

        # Default — cap stays at 256 even when max_num_seqs is lower.
        # This preserves queue depth for tests/deployments that count
        # on it.
        cfg = SchedulerConfig(max_num_seqs=8)
        assert cfg.max_concurrent_requests == 256

        # Explicit override — operator can tighten the cap to match
        # max_num_seqs (or anything else) for memory protection.
        cfg = SchedulerConfig(max_num_seqs=8, max_concurrent_requests=8)
        assert cfg.max_concurrent_requests == 8

    def test_cold_start_admission_uses_default_cap_when_config_is_none(self):
        """Codex R10 closure: when ``BatchedEngine`` is constructed
        without an explicit ``scheduler_config`` (the ``load_model``
        default + direct test callers), ``self._scheduler_config`` is
        ``None`` until the engine finishes starting. The cold-start
        fallback must default to ``SchedulerConfig().max_concurrent_requests``
        (256) rather than silently no-op'ing — otherwise a streaming
        burst at startup slips past preflight and only the
        scheduler-level ``BackpressureError`` fires, degrading to a
        200 SSE error chunk."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        engine = MagicMock(spec=BatchedEngine)
        engine._admission_lock = threading.Lock()
        engine._is_mllm = False
        engine._mllm_scheduler = None
        # No scheduler yet (cold-start) AND no scheduler_config (the
        # exact path codex R10 flagged).
        engine._engine = None
        engine._scheduler_config = None

        default_cap = SchedulerConfig().max_concurrent_requests
        engine._admission_reservations = default_cap

        with pytest.raises(BackpressureError):
            BatchedEngine.check_admission(engine)

        # One slot below cap still admits — confirms the gate is
        # active and using the default cap, not unbounded.
        engine._admission_reservations = default_cap - 1
        BatchedEngine.check_admission(engine)
        assert engine._admission_reservations == default_cap

    def test_cloud_routable_chat_not_rejected_at_local_cap(self, monkeypatch):
        """Codex R9 closure: when ``cfg.cloud_router`` is enabled and
        the request crosses the cloud threshold, admission must not
        gate it on local-engine capacity. The local cap is pre-filled
        to the configured maximum so any pre-cloud admission check
        would 503; the cloud branch must instead return 200 with the
        cloud response, and the local reservation counter must remain
        unchanged (the cloud round-trip uses no local resources)."""
        import threading

        from vllm_mlx.config import get_config
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import SchedulerConfig

        app = FastAPI()
        app.include_router(chat_route.router)

        engine = MagicMock()
        engine.is_mllm = False
        engine.supports_guided_generation = False
        engine.build_prompt.return_value = "prompt"
        engine.estimate_new_tokens.return_value = (1000, 1000)
        engine.preserve_native_tool_format = False

        engine._admission_lock = threading.Lock()
        # Pre-fill the reservation counter to the cap. Any local-path
        # admission check at this point would raise BackpressureError
        # → 503. The cloud-routable request must bypass this.
        engine._admission_reservations = 2
        engine._is_mllm = False
        engine._mllm_scheduler = None

        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=2)
        scheduler_stub.requests = {}
        inner_engine_stub = _Stub()
        inner_engine_stub.scheduler = scheduler_stub
        async_engine_stub = _Stub()
        async_engine_stub.engine = inner_engine_stub
        engine._engine = async_engine_stub

        engine.check_admission = lambda: BatchedEngine.check_admission(engine)
        engine.release_admission_reservation = lambda: (
            BatchedEngine.release_admission_reservation(engine)
        )

        cloud_router = MagicMock()
        cloud_router.should_route_to_cloud.return_value = True
        cloud_router.threshold = 500
        cloud_router.cloud_model = "cloud-x"

        async def _fake_completion(*_a, **_kw):
            return {
                "id": "cloud-r9",
                "object": "chat.completion",
                "created": 0,
                "model": "cloud-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

        cloud_router.completion = _fake_completion

        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "engine",
                "model_name",
                "model_alias",
                "model_path",
                "model_registry",
                "tool_call_parser",
                "reasoning_parser",
                "ready",
                "api_key",
                "cloud_router",
                "pin_system_prompt",
            )
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None
        cfg.cloud_router = cloud_router
        cfg.pin_system_prompt = False

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        # Cloud routing won despite local cap being full: 200, not 503.
        assert r.status_code == 200, r.text
        # Local reservation counter unchanged — cloud path never
        # touched it (no acquire, no release).
        assert engine._admission_reservations == 2
