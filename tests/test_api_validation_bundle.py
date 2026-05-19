# SPDX-License-Identifier: Apache-2.0
"""Route-level validation hardening (R10 sweep follow-up).

Each test pins behavior that the onboarding sweep showed was silently
broken — e.g. ``model: ""`` returning 200 with the default model,
``top_p=2.0`` accepted without error, ``logit_bias`` silently dropped,
``encoding_format=base64`` ignored on /v1/embeddings.
"""

import argparse
import base64
import struct
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    """Patch select fields on the global cfg singleton and restore on exit.

    Mirrors the pattern in test_routes.py — avoids the ``setattr``/``leak
    into next test`` hazard of touching the singleton directly.
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


# ---------------------------------------------------------------------------
# _validate_model_name — empty string must 400, not silently use default
# ---------------------------------------------------------------------------


class TestValidateModelName:
    def test_empty_string_raises_400(self):
        """``model: ""`` used to short-circuit to the default model,
        masking client bugs (typos, unset env vars)."""
        from vllm_mlx.service.helpers import _validate_model_name

        with pytest.raises(HTTPException) as ei:
            _validate_model_name("")
        assert ei.value.status_code == 400
        assert "empty" in ei.value.detail.lower()

    def test_none_still_passes_through(self):
        """``None`` continues to be a no-op so callers that pass an
        unset request.model field don't break."""
        from vllm_mlx.service.helpers import _validate_model_name

        # Should not raise.
        _validate_model_name(None)


# ---------------------------------------------------------------------------
# Chat completion validation block — top_p, max_tokens cap, logit_bias
# ---------------------------------------------------------------------------


def _build_chat_app(patch_cfg, monkeypatch):
    """Mount the chat router with a stub engine so we can hit the
    validation block without touching mlx weights."""
    from vllm_mlx.routes import chat as chat_route

    app = FastAPI()
    app.include_router(chat_route.router)

    engine = MagicMock()
    engine.is_mllm = False
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

    # get_engine() inside the route resolves to this stub; if validation
    # passes, the test's other assertions take over (or the engine call
    # fails downstream, which is fine — we only care about the 400 path).
    monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

    # raise_server_exceptions=False so downstream-pipeline failures
    # (the mocked engine returns non-coroutines) come back as 500s
    # rather than re-raising into pytest — we only care about the
    # validator response.
    return TestClient(app, raise_server_exceptions=False)


class TestChatValidation:
    def test_top_p_above_one_rejected(self, patched_config, monkeypatch):
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 2.0,
            },
        )
        assert r.status_code == 400
        assert "top_p" in r.json()["detail"]

    def test_top_p_zero_rejected(self, patched_config, monkeypatch):
        """0 is invalid per OpenAI spec — the valid range is (0, 1]."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 0,
            },
        )
        assert r.status_code == 400
        assert "top_p" in r.json()["detail"]

    def test_top_p_one_passes_validation(self, patched_config, monkeypatch):
        """1.0 is the OpenAI default and must not trigger the top_p
        validator. If downstream plumbing fails (likely, since we stub
        the engine), the failure must NOT be a top_p complaint."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 1.0,
            },
        )
        if r.status_code == 400:
            assert "top_p" not in r.json().get("detail", "")

    def test_max_tokens_over_ceiling_rejected(self, patched_config, monkeypatch):
        """Sanity ceiling at 1_000_000. Combined with admission control
        (separate PR) this prevents OOM from a buggy client passing
        max_tokens=999_999_999."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 999_999_999,
            },
        )
        assert r.status_code == 400
        assert "max_tokens" in r.json()["detail"]

    def test_logit_bias_rejected_with_clear_400(self, patched_config, monkeypatch):
        """Previously silently dropped (field not declared in schema).
        Declared + rejected with a clear message so clients can fall
        back without seeing wrong-output."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "logit_bias": {"50000": -100},
            },
        )
        assert r.status_code == 400
        assert "logit_bias" in r.json()["detail"]

    def test_empty_logit_bias_does_not_trigger_400(self, patched_config, monkeypatch):
        """Defensive clients sometimes always send ``logit_bias: {}``;
        the empty dict must NOT trigger the validator."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "logit_bias": {},
            },
        )
        if r.status_code == 400:
            assert "logit_bias" not in r.json().get("detail", "")


# ---------------------------------------------------------------------------
# Embeddings — dimensions truncation + base64 encoding
# ---------------------------------------------------------------------------


def _build_embed_app(patch_cfg, monkeypatch, embed_return):
    from vllm_mlx.routes import embeddings as emb_route

    app = FastAPI()
    app.include_router(emb_route.router)

    engine = MagicMock()
    engine.count_tokens.return_value = 3
    engine.embed.return_value = embed_return
    patch_cfg(
        embedding_engine=engine,
        embedding_model_locked=None,
        api_key=None,
    )

    monkeypatch.setattr(
        "vllm_mlx.server.load_embedding_model",
        lambda *_a, **_kw: None,
        raising=False,
    )

    return TestClient(app), engine


class TestEmbeddingsRoute:
    def test_dimensions_truncates_vector(self, patched_config, monkeypatch):
        """Slice to the requested length, then L2-normalize so the
        result is still a valid embedding (see
        test_truncated_vector_is_unit_norm for the why)."""
        embed = [[0.1 * i for i in range(384)]]
        client, _ = _build_embed_app(patched_config, monkeypatch, embed)
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hello", "dimensions": 64},
        )
        assert r.status_code == 200, r.text
        vec = r.json()["data"][0]["embedding"]
        assert len(vec) == 64
        import math as _m

        assert abs(_m.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6

    def test_dimensions_zero_rejected(self, patched_config, monkeypatch):
        embed = [[0.0] * 16]
        client, _ = _build_embed_app(patched_config, monkeypatch, embed)
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hi", "dimensions": 0},
        )
        assert r.status_code == 400

    def test_base64_encoding_round_trip(self, patched_config, monkeypatch):
        """encoding_format=base64 must produce a base64 string that
        decodes back to the original float32 vector. Catches both the
        silent-drop bug AND any byte-ordering mistake."""
        original = [0.5, -1.25, 3.0, 0.0]
        client, _ = _build_embed_app(patched_config, monkeypatch, [original])

        r = client.post(
            "/v1/embeddings",
            json={
                "model": "any",
                "input": "hello",
                "encoding_format": "base64",
            },
        )
        assert r.status_code == 200, r.text
        encoded = r.json()["data"][0]["embedding"]
        assert isinstance(encoded, str)

        decoded = struct.unpack(f"<{len(original)}f", base64.b64decode(encoded))
        assert list(decoded) == original

    def test_float_format_still_returns_list(self, patched_config, monkeypatch):
        """Default encoding_format='float' is unchanged."""
        client, _ = _build_embed_app(patched_config, monkeypatch, [[0.1, 0.2]])
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hi"},
        )
        assert r.status_code == 200
        vec = r.json()["data"][0]["embedding"]
        assert isinstance(vec, list)
        assert vec == [0.1, 0.2]

    def test_base64_plus_dimensions_combine(self, patched_config, monkeypatch):
        """Truncation happens BEFORE base64 encoding so the packed
        length matches `dimensions`. Result is also L2-normalized."""
        original = [float(i) for i in range(16)]
        client, _ = _build_embed_app(patched_config, monkeypatch, [original])

        r = client.post(
            "/v1/embeddings",
            json={
                "model": "any",
                "input": "hi",
                "dimensions": 4,
                "encoding_format": "base64",
            },
        )
        assert r.status_code == 200
        encoded = r.json()["data"][0]["embedding"]
        decoded = list(struct.unpack("<4f", base64.b64decode(encoded)))
        # Truncated [0,1,2,3] L2-normalized: norm=sqrt(14)
        import math as _m

        norm = _m.sqrt(sum(x * x for x in [0.0, 1.0, 2.0, 3.0]))
        expected = [x / norm for x in [0.0, 1.0, 2.0, 3.0]]
        for got, want in zip(decoded, expected):
            assert abs(got - want) < 1e-6

    def test_dimensions_above_model_dim_rejected(self, patched_config, monkeypatch):
        """Per OpenAI spec: requesting more dimensions than the model
        produces must 400, not silently return the full vector."""
        client, _ = _build_embed_app(
            patched_config, monkeypatch, [[0.1, 0.2, 0.3, 0.4]]
        )
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hi", "dimensions": 9999},
        )
        assert r.status_code == 400
        assert "dimensions" in r.json()["detail"].lower()

    def test_truncated_vector_is_unit_norm(self, patched_config, monkeypatch):
        """MRL-style truncation requires L2-renormalization for the
        result to be a valid cosine-similarity embedding (OpenAI
        cookbook for text-embedding-3-large). A naive slice without
        normalization is mathematically wrong for any MRL model."""
        embed = [[3.0, 4.0, 0.0, 0.0, 0.0]]
        client, _ = _build_embed_app(patched_config, monkeypatch, embed)
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hi", "dimensions": 2},
        )
        assert r.status_code == 200
        vec = r.json()["data"][0]["embedding"]
        import math as _m

        norm = _m.sqrt(sum(x * x for x in vec))
        # Expect unit norm. Sliced (3,4) has norm 5 → normalized (0.6,0.8).
        assert abs(norm - 1.0) < 1e-6
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6


class TestEmbeddingsEncodingFormatLiteral:
    """Pydantic Literal narrowing replaces the silent-fallback bug:
    unknown encoding_format values now 422 at parse time."""

    def test_unknown_encoding_format_rejected(self):
        """Catches typos like 'base65', 'BASE64', 'json' — previously
        they all silently returned a float list."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import EmbeddingRequest

        for bogus in ["base65", "BASE64", "json", "raw"]:
            with pytest.raises(ValidationError):
                EmbeddingRequest(input="hi", model="x", encoding_format=bogus)

    def test_valid_encoding_formats_accepted(self):
        from vllm_mlx.api.models import EmbeddingRequest

        for ok in ["float", "base64"]:
            req = EmbeddingRequest(input="hi", model="x", encoding_format=ok)
            assert req.encoding_format == ok


# ---------------------------------------------------------------------------
# --log-level accepts lowercase (industry convention)
# ---------------------------------------------------------------------------


class TestLogLevelLowercase:
    def _make_parser(self):
        """Mirror the same ``type`` contract used by serve_parser in
        vllm_mlx/cli.py and vllm_mlx/server.py."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--log-level",
            type=lambda s: s.upper(),
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )
        return parser

    @pytest.mark.parametrize("flag", ["debug", "info", "warning", "error"])
    def test_lowercase_accepted(self, flag):
        ns = self._make_parser().parse_args(["--log-level", flag])
        assert ns.log_level == flag.upper()

    @pytest.mark.parametrize("flag", ["DEBUG", "Info", "Warning"])
    def test_mixed_case_accepted(self, flag):
        ns = self._make_parser().parse_args(["--log-level", flag])
        assert ns.log_level == flag.upper()

    def test_unknown_level_still_rejected(self):
        with pytest.raises(SystemExit):
            self._make_parser().parse_args(["--log-level", "trace"])


# ---------------------------------------------------------------------------
# Surgical bundle (R10 follow-up #2):
# - C3: chat rejects image/video on text-only models (no silent hallucination)
# - C16: guided JSON preserves nested array-of-objects (no silent str degrade)
# - H17: `rapid-mlx ps` parses --port positioned after the model argument
# - H18: /v1/completions rejects FIM `suffix` (declared, not silently dropped)
# ---------------------------------------------------------------------------


class TestChatRejectsImageOnTextOnlyModel:
    def test_image_url_on_text_only_engine_400(self, patched_config, monkeypatch):
        """Before this fix, extract_multimodal_content silently stripped
        the image part on a text-only engine and the model would
        confidently caption an image it never saw (R9P1 sweep)."""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/x.png"},
                            },
                        ],
                    }
                ],
            },
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "image" in detail.lower() or "video" in detail.lower()

    def test_video_on_text_only_engine_400(self, patched_config, monkeypatch):
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "video", "video": "/tmp/x.mp4"},
                        ],
                    }
                ],
            },
        )
        assert r.status_code == 400
        assert (
            "image" in r.json()["detail"].lower()
            or "video" in r.json()["detail"].lower()
        )

    def test_text_only_content_still_passes_validation(
        self, patched_config, monkeypatch
    ):
        """Plain text request on a text-only engine must NOT trigger the
        vision-rejection branch. (Downstream will 500 because the engine
        is mocked — we only care that the 400 reason is not the new
        guard.)"""
        client = _build_chat_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        if r.status_code == 400:
            assert "image" not in r.json().get("detail", "").lower()
            assert "video" not in r.json().get("detail", "").lower()


class TestGuidedArrayOfObjectsSchema:
    """Before C16, ``array of objects`` (and ``array of arrays``) fell
    through to ``type_mapping.get(items_type, str)`` and silently became
    ``list[str]``. The model would then emit strings where the schema
    required objects, and the response failed validation against the
    user's own schema (R10 sweep finding)."""

    def test_array_of_objects_maps_to_list_dict(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["items"],
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        # ``list[dict]`` accepts dict elements without coercion. Before
        # the fix this would be ``list[str]`` and dicts would coerce
        # to their repr or raise.
        instance = model(items=[{"a": 1}, {"b": 2}])
        assert instance.items == [{"a": 1}, {"b": 2}]

    def test_array_of_arrays_maps_to_list_list(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "matrix": {"type": "array", "items": {"type": "array"}},
            },
            "required": ["matrix"],
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model(matrix=[[1, 2], [3, 4]])
        assert instance.matrix == [[1, 2], [3, 4]]

    def test_array_of_strings_still_works(self):
        """Sanity: the historical happy path is unchanged."""
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        assert model(tags=["a", "b"]).tags == ["a", "b"]


class TestPsCommandPortParsing:
    """``rapid-mlx ps`` used to break on the first positional argument,
    so ``serve qwen3.5-4b --port 8005`` showed port=8000 (the default).
    Verify the parser keeps scanning for flags after capturing the
    positional model."""

    def _parse_serve(self, cmd_words):
        """Re-implement ps_command's parsing loop in isolation to avoid
        importing psutil + the full CLI. Mirrors cli.py:1249-1280."""
        model = "(unknown)"
        port = "8000"
        try:
            i = cmd_words.index("serve") + 1
            model_seen = False
            while i < len(cmd_words):
                tok = cmd_words[i]
                if tok.startswith("--"):
                    if tok == "--port" and i + 1 < len(cmd_words):
                        port = cmd_words[i + 1]
                        i += 2
                    elif "=" in tok and tok.startswith("--port="):
                        port = tok.split("=", 1)[1]
                        i += 1
                    else:
                        i += 1
                else:
                    if not model_seen:
                        model = tok
                        model_seen = True
                    i += 1
        except ValueError:
            pass
        return model, port

    def test_port_after_positional_model(self):
        model, port = self._parse_serve(
            ["rapid-mlx", "serve", "qwen3.5-4b", "--port", "8005"]
        )
        assert model == "qwen3.5-4b"
        assert port == "8005"

    def test_port_before_positional_model(self):
        model, port = self._parse_serve(
            ["rapid-mlx", "serve", "--port", "8005", "qwen3.5-4b"]
        )
        assert model == "qwen3.5-4b"
        assert port == "8005"

    def test_port_equals_form(self):
        model, port = self._parse_serve(
            ["rapid-mlx", "serve", "qwen3.5-4b", "--port=9000"]
        )
        assert model == "qwen3.5-4b"
        assert port == "9000"

    def test_no_port_uses_default(self):
        model, port = self._parse_serve(["rapid-mlx", "serve", "qwen3.5-4b"])
        assert model == "qwen3.5-4b"
        assert port == "8000"


class TestCompletionsSuffixRejection:
    def _build_completions_app(self, patch_cfg, monkeypatch):
        from vllm_mlx.routes import completions as comp_route

        app = FastAPI()
        app.include_router(comp_route.router)

        engine = MagicMock()
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
        return TestClient(app, raise_server_exceptions=False)

    def test_suffix_rejected_with_400(self, patched_config, monkeypatch):
        """FIM `suffix` was silently dropped pre-PR (field not declared
        on CompletionRequest). Code-completion clients (Continue, Cody)
        would then get a non-FIM completion that ignored the suffix and
        often produced wrong code. Declare + reject so they fall back."""
        client = self._build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "def hello():",
                "suffix": "    return greeting",
            },
        )
        assert r.status_code == 400
        assert "suffix" in r.json()["detail"].lower()

    def test_empty_suffix_does_not_trigger_400(self, patched_config, monkeypatch):
        """Defensive clients sometimes always send ``suffix: ""`` — the
        empty string must NOT trip the new guard."""
        client = self._build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hello",
                "suffix": "",
            },
        )
        if r.status_code == 400:
            assert "suffix" not in r.json().get("detail", "").lower()

    def test_omitted_suffix_does_not_trigger_400(self, patched_config, monkeypatch):
        client = self._build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hello"},
        )
        if r.status_code == 400:
            assert "suffix" not in r.json().get("detail", "").lower()


class TestMLLMBatchGeneratorFailsLoud:
    """Before C2, image/video processing failures (404, decode error,
    auth) were logged at WARN and the broken input was silently dropped
    from the request — the model then captioned a non-existent image
    and hallucinated.

    The new code raises ``ValueError`` (NOT HTTPException) because
    ``_preprocess_request`` runs inside ``MLLMBatchGenerator.next()``,
    which is called from ``MLLMScheduler._step()``. That step catches
    ``(ValueError, RuntimeError)`` narrowly and converts the failure
    into a ``finish_reason="error"`` response for the request. An
    HTTPException would bubble past the narrow catch into
    ``_process_loop``'s generic ``except Exception``, which only logs
    — the client would hang until timeout instead of getting a
    structured error. (Caught by codex review on PR #416.)

    Structural test (grep-based) instead of full async invocation: the
    real ``_preprocess_request`` requires a model registry, GPU init,
    and an event loop. The bug class we're guarding against is "did
    someone reintroduce the silent-drop pattern OR re-raise as a type
    the scheduler can't catch", which structural checks pin cheaply.
    """

    def _source(self):
        from pathlib import Path

        import vllm_mlx.mllm_batch_generator as mod

        return Path(mod.__file__).read_text()

    def test_image_branch_raises_value_error(self):
        src = self._source()
        assert "Failed to process image" in src
        # Window straddles the marker so we see the raise above it.
        idx = src.find("Failed to process image")
        window = src[max(0, idx - 200) : idx + 200]
        assert "raise ValueError" in window
        # The old silent-drop pattern must not return.
        assert 'logger.warning(f"Failed to process image' not in src
        # And we must NOT use HTTPException here — the scheduler's
        # narrow except catches ValueError/RuntimeError only.
        assert "HTTPException" not in window

    def test_video_branch_raises_value_error(self):
        src = self._source()
        assert "Failed to process video" in src
        idx = src.find("Failed to process video")
        window = src[max(0, idx - 200) : idx + 200]
        assert "raise ValueError" in window
        assert 'logger.warning(f"Failed to process video' not in src
        assert "HTTPException" not in window

    def test_scheduler_catches_value_error_from_preprocess(self):
        """Pins the contract that motivates the choice of ValueError:
        ``MLLMScheduler._step`` must catch ``(ValueError, RuntimeError)``
        around ``self.batch_generator.next()``. If a future refactor
        narrows that catch or wraps the call site differently, our
        C2 fix silently regresses to "client hangs"."""
        from pathlib import Path

        import vllm_mlx.mllm_scheduler as sched_mod

        src = Path(sched_mod.__file__).read_text()
        # Find the next() call and assert the surrounding catch.
        idx = src.find("self.batch_generator.next()")
        assert idx != -1, "MLLMScheduler no longer calls batch_generator.next()"
        window = src[max(0, idx - 100) : idx + 400]
        assert "try:" in window
        assert "except (ValueError, RuntimeError)" in window
