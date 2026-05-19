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
    cfg.embedding_model_locked = None
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
        class _Stub:
            pass

        scheduler_stub = _Stub()
        scheduler_stub.config = SchedulerConfig(max_concurrent_requests=1)
        scheduler_stub.requests = {}

        engine_stub = _Stub()
        engine_stub.scheduler = scheduler_stub
        eng._engine = engine_stub

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
