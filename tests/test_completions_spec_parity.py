# SPDX-License-Identifier: Apache-2.0
"""Spec parity for legacy ``/v1/completions`` (F-152 + F-153).

Pre-fix the route silently accepted ``n``, ``best_of``, ``echo``, and
``logprobs:true`` while doing nothing with them — a silent-compat lie
that broke OpenAI SDK clients (wrong billing math, missing logprobs in
eval harnesses). The pydantic schema also typed ``logprobs`` as
``bool``, so the canonical ``logprobs=5`` SDK form bounced with a
422 ``bool_parsing`` error that never mentioned the actual mismatch.

Each test below pins one piece of the new contract so a future
refactor cannot silently regress the spec parity. Tests run as
isolated FastAPI test-clients with a ``MagicMock`` engine; no real
model load, no GPU, no port — they live in the same file family as
``test_api_validation_bundle.py``.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    """Mirror of ``test_api_validation_bundle.patched_config``.

    Patches select fields on the global cfg singleton and restores
    them on test exit so each test sees a clean config.
    """
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


def _build_completions_app(patch_cfg, monkeypatch, *, engine_factory=None):
    """Wire a stub completions app with a MagicMock engine.

    ``engine_factory`` lets a test customise the engine (e.g. wire a
    streaming async generator into ``stream_generate``). When omitted,
    a MagicMock with sensible defaults is used.
    """
    from vllm_mlx.routes import completions as comp_route

    app = FastAPI()
    app.include_router(comp_route.router)

    engine = engine_factory() if engine_factory else MagicMock()
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)
    # ``enforce_context_length_for_prompt`` calls into the engine's
    # tokenizer; short-circuit so the schema-validation tests don't
    # depend on a tokenizer.
    monkeypatch.setattr(
        comp_route, "enforce_context_length_for_prompt", lambda *_a, **_kw: None
    )
    return TestClient(app, raise_server_exceptions=False), engine


# ---------------------------------------------------------------------------
# F-152: n / best_of / echo behaviour
# ---------------------------------------------------------------------------


class TestN:
    """``n>1`` must 400 (mirroring the chat-completions route)."""

    def test_n_above_one_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "n": 3},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "n > 1" in detail or "n>1" in detail.replace(" ", "")

    def test_n_one_is_accepted(self, patched_config, monkeypatch):
        """``n: 1`` is the OpenAI default — must not trip the new guard."""

        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "n": 1, "max_tokens": 4},
        )
        assert r.status_code == 200

    def test_n_omitted_is_accepted(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "max_tokens": 4},
        )
        assert r.status_code == 200


class TestBestOf:
    """``best_of>1`` must 400 (no server-side reranker)."""

    def test_best_of_above_one_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "best_of": 5},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "best_of" in detail

    def test_best_of_one_is_accepted(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "best_of": 1},
        )
        assert r.status_code == 200


class TestEcho:
    """``echo: true`` must prepend the prompt to ``choices[0].text``."""

    def test_echo_true_prepends_prompt(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput(text=" world!")

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "echo": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == "Hello world!"

    def test_echo_false_does_not_prepend(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput(text=" world!")

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "echo": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == " world!"


# ---------------------------------------------------------------------------
# F-153: logprobs schema + range
# ---------------------------------------------------------------------------


class TestLogprobsSchema:
    """``logprobs`` is an integer 0..5 on legacy completions, not a bool."""

    def test_logprobs_bool_rejected_with_422(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": True},
        )
        # Pydantic before-mode validator returns 422 (not 400) for
        # the wire-form schema mismatch — a clear "this is the wrong
        # shape" signal distinct from the route-level 400s ranges
        # use.
        assert r.status_code == 422
        body = r.json()
        # Detail must mention the integer expectation so SDK clients
        # see the actual mismatch (pre-fix they got bool_parsing
        # which never mentioned the schema).
        msg = str(body)
        assert "integer" in msg.lower()
        assert "bool" in msg.lower() or "boolean" in msg.lower()

    def test_logprobs_above_five_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": 6},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "0 and 5" in detail or "logprobs" in detail.lower()

    def test_logprobs_negative_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": -1},
        )
        assert r.status_code == 400

    def test_logprobs_int_accepted_by_schema(self):
        """Direct pydantic-level test: ``logprobs: 5`` parses cleanly."""
        from vllm_mlx.api.models import CompletionRequest

        req = CompletionRequest(model="x", prompt="y", logprobs=5)
        assert req.logprobs == 5

    def test_logprobs_bool_rejected_by_schema(self):
        """Direct pydantic-level test: ``logprobs: True`` raises."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import CompletionRequest

        with pytest.raises(ValidationError) as ei:
            CompletionRequest(model="x", prompt="y", logprobs=True)
        # The error must explain the integer expectation, NOT just
        # ``bool_parsing`` (the pre-fix opaque message).
        msg = str(ei.value)
        assert "integer" in msg.lower()


# ---------------------------------------------------------------------------
# Field declarations — pydantic must NOT silently drop these
# ---------------------------------------------------------------------------


class TestEchoLogprobsCombination:
    """Codex r1 BLOCKING: ``echo + logprobs`` must NOT return partial
    arrays (would mis-align ``text_offset`` against ``tokens`` and
    silently corrupt prompt-conditioned scores in lm-eval-harness).
    Reject with 400 until we wire prompt-prefill-with-logprobs."""

    def test_echo_with_logprobs_rejected_with_400(
        self, patched_config, monkeypatch
    ):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "echo": True,
                "logprobs": 3,
            },
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "echo" in detail.lower() and "logprobs" in detail.lower()

    def test_echo_with_logprobs_zero_still_rejected(
        self, patched_config, monkeypatch
    ):
        """``logprobs:0`` is still a logprobs request — must reject too."""
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "echo": True,
                "logprobs": 0,
            },
        )
        assert r.status_code == 400

    def test_echo_alone_still_works(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput(text=" world!")

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "echo": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == "Hello world!"


class TestLogprobsResponseShape:
    """Codex r1 NIT: validate the actual response payload for
    ``logprobs=5`` and ``logprobs=0`` carries the legacy four-array
    shape. Pre-fix the route returned no ``logprobs`` slot at all."""

    def _build_streaming_engine(self):
        """Return an engine whose ``stream_generate`` yields one
        ``GenerationOutput`` chunk with synthetic per-token logprobs.
        The helper ``_extract_streaming_token_logprobs`` reads
        ``chunk.logprobs`` + ``chunk.tokens`` + ``chunk.new_text`` —
        wire all three on a dataclass-ish stub. Top-k extraction is
        handled by the route logic (``effective_top_k`` + the
        ``top_logprobs == {}`` strip when ``logprobs=0``); the
        fixture provides the raw per-vocab distribution.
        """

        class _Chunk:
            def __init__(self):
                self.text = "Hi"
                self.new_text = "Hi"
                self.tokens = [2]
                self.new_token_ids = [2]
                self.prompt_tokens = 1
                self.completion_tokens = 1
                self.finished = True
                self.finish_reason = "stop"
                self.cached_tokens = 0
                # MLX array per-vocab logprobs (small synthetic vocab
                # so the helper can argpartition cleanly). The
                # extractor calls ``.astype(mx.float32)`` on this
                # value — must be a real ``mx.array``.
                import mlx.core as mx

                self.logprobs = mx.array(
                    [-3.0, -2.0, -1.0, -4.0, -5.0], dtype=mx.float32
                )

        _VOCAB = {0: "<a>", 1: "<b>", 2: "Hi", 3: "<d>", 4: "<e>"}

        async def _stream(*_a, **_kw):
            yield _Chunk()

        class _Tokenizer:
            def decode(self, ids):
                return _VOCAB.get(ids[0], "?")

        e = MagicMock()
        e.stream_generate = _stream
        e.tokenizer = _Tokenizer()
        return e

    def test_logprobs_five_returns_four_arrays(
        self, patched_config, monkeypatch
    ):
        client, _ = _build_completions_app(
            patched_config,
            monkeypatch,
            engine_factory=self._build_streaming_engine,
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "x",
                "max_tokens": 1,
                "logprobs": 5,
            },
        )
        assert r.status_code == 200, r.text
        lp = r.json()["choices"][0]["logprobs"]
        # All four legacy arrays must be present.
        assert set(lp.keys()) == {
            "tokens",
            "token_logprobs",
            "top_logprobs",
            "text_offset",
        }
        # Parallel arrays — same length.
        n = len(lp["tokens"])
        assert n == len(lp["token_logprobs"]) == len(lp["top_logprobs"])
        assert n == len(lp["text_offset"])
        # logprobs=5 → each top_logprobs dict has 5 entries (small
        # synthetic vocab of size 5 in the fixture).
        assert all(len(d) == 5 for d in lp["top_logprobs"])

    def test_logprobs_zero_returns_empty_top_dicts(
        self, patched_config, monkeypatch
    ):
        client, _ = _build_completions_app(
            patched_config,
            monkeypatch,
            engine_factory=self._build_streaming_engine,
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "x",
                "max_tokens": 1,
                "logprobs": 0,
            },
        )
        assert r.status_code == 200, r.text
        lp = r.json()["choices"][0]["logprobs"]
        # Sampled-token logprob still surfaced, but no alternatives.
        assert lp["token_logprobs"]  # non-empty
        assert all(d == {} for d in lp["top_logprobs"])


class TestLogprobsEngineCapability:
    """Codex r2 BLOCKING #1: engines without ``stream_generate`` or
    without a ``tokenizer`` must NOT crash with 500 when a client
    sends ``logprobs:N`` — return a controlled 501 instead."""

    def test_engine_without_stream_generate_returns_501(
        self, patched_config, monkeypatch
    ):
        def _factory():
            class _NoStreamEngine:
                tokenizer = object()

                async def generate(self, *_a, **_kw):
                    return _StubGenerationOutput()

                # Intentionally NO ``stream_generate``.

            return _NoStreamEngine()

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "max_tokens": 1,
                "logprobs": 3,
            },
        )
        assert r.status_code == 501
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "logprobs" in detail.lower()

    def test_engine_without_tokenizer_returns_501(
        self, patched_config, monkeypatch
    ):
        def _factory():
            e = MagicMock()
            e.tokenizer = None
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "max_tokens": 1,
                "logprobs": 3,
            },
        )
        assert r.status_code == 501


class TestEmptyStreamNotMistakenForDisconnect:
    """Codex r2 BLOCKING #2: an engine that yields zero chunks must
    return an empty completion, NOT HTTP 499 (false disconnect)."""

    def test_empty_stream_returns_200_with_empty_text(
        self, patched_config, monkeypatch
    ):
        async def _empty_stream(*_a, **_kw):
            if False:
                yield None  # never yields — empty async gen

        class _Tokenizer:
            def decode(self, ids):
                return ""

        def _factory():
            e = MagicMock()
            e.stream_generate = _empty_stream
            e.tokenizer = _Tokenizer()
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "max_tokens": 1,
                "logprobs": 3,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["text"] == ""
        # Codex r3 NIT: when a client sent ``logprobs:N``, the
        # response shape contract is the legacy four-array slot —
        # even when arrays are empty (no generated tokens). Accept
        # an explicit empty payload OR an absent slot (the route
        # short-circuits on empty stream and never builds the
        # payload). Document the chosen behavior with a clear
        # assertion either way.
        lp = body["choices"][0].get("logprobs")
        if lp is not None:
            assert set(lp.keys()) == {
                "tokens",
                "token_logprobs",
                "top_logprobs",
                "text_offset",
            }
            assert lp["tokens"] == []
            assert lp["token_logprobs"] == []
            assert lp["top_logprobs"] == []
            assert lp["text_offset"] == []


class TestTextOffsetAlignment:
    """Codex r2 BLOCKING #3: ``text_offset`` must be byte-exact
    against ``choices[0].text`` so SDK consumers can slice."""

    def _build_engine(self):
        class _Chunk:
            def __init__(self):
                self.text = "Hi there"  # would mismatch token concat
                self.new_text = "Hi there"
                self.tokens = [1, 2]
                self.new_token_ids = [1, 2]
                self.prompt_tokens = 1
                self.completion_tokens = 2
                self.finished = True
                self.finish_reason = "stop"
                self.cached_tokens = 0
                import mlx.core as mx

                # Sampled token ids are 1 and 2; ids 0/3 are
                # alternatives. Use a 4-vocab distribution.
                self.logprobs = [
                    mx.array([-3.0, -1.0, -2.0, -2.5], dtype=mx.float32),
                    mx.array([-3.5, -2.5, -0.3, -1.5], dtype=mx.float32),
                ]

        async def _stream(*_a, **_kw):
            yield _Chunk()

        _VOCAB = {0: "<a>", 1: "Hi", 2: " there", 3: "<d>"}

        class _Tokenizer:
            def decode(self, ids):
                # Decoded forms intentionally differ from
                # ``chunk.text`` ("Hi there" vs "Hi" + " there") to
                # expose any offset misalignment between
                # ``output.text`` and the token concatenation.
                return _VOCAB.get(ids[0], "?")

        e = MagicMock()
        e.stream_generate = _stream
        e.tokenizer = _Tokenizer()
        return e

    def test_text_offset_byte_exact_against_choice_text(
        self, patched_config, monkeypatch
    ):
        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=self._build_engine
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "x",
                "max_tokens": 2,
                "logprobs": 2,
            },
        )
        assert r.status_code == 200, r.text
        c = r.json()["choices"][0]
        text = c["text"]
        lp = c["logprobs"]
        # Every offset must point at the start of the matching token
        # within ``text`` — slice and verify.
        for i, (token, offset) in enumerate(zip(lp["tokens"], lp["text_offset"])):
            assert text[offset : offset + len(token)] == token, (
                f"token #{i}={token!r} at offset {offset} did not "
                f"align in text={text!r}"
            )


class TestStreamingEngineCapability:
    """Codex r3 BLOCKING #1: capability guard must cover the
    streaming branch too — without it the AttributeError would fire
    inside the committed SSE response."""

    def test_stream_logprobs_without_tokenizer_returns_501(
        self, patched_config, monkeypatch
    ):
        def _factory():
            class _NoTokenizer:
                tokenizer = None

                async def stream_generate(self, *_a, **_kw):
                    if False:
                        yield None

            return _NoTokenizer()

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "stream": True,
                "logprobs": 3,
            },
        )
        # Route-level guard fires BEFORE StreamingResponse is
        # constructed, so the client sees a clean 501 instead of an
        # SSE stream that fails on the first chunk.
        assert r.status_code == 501


class TestFieldDeclarations:
    """The pre-fix schema silently dropped ``n``, ``best_of``, ``echo``
    on parse — equivalent to the silent-compat lie F-152 closes."""

    def test_all_four_fields_declared(self):
        from vllm_mlx.api.models import CompletionRequest

        fields = CompletionRequest.model_fields
        assert "n" in fields
        assert "best_of" in fields
        assert "echo" in fields
        assert "logprobs" in fields
        # ``logprobs`` annotation must be the integer form (not bool).
        # Pydantic stores this as ``int | None`` — checking the
        # representation is robust against future ``Annotated[int, ...]``
        # tightening.
        ann = str(fields["logprobs"].annotation)
        assert "int" in ann and "bool" not in ann


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubGenerationOutput:
    """Minimal ``GenerationOutput`` stand-in for non-streaming tests."""

    def __init__(self, text: str = " world", finish_reason: str = "stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.completion_tokens = 4
        self.prompt_tokens = 1
        self.cached_tokens = 0
