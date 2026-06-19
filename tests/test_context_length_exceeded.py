# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the prompt-token context-length pre-check.

Even within the wire-level body cap (``middleware/body_size.py``),
a 8 MiB ASCII payload can hold ~2M tokens — well past every model
context window. ``service/helpers.py::enforce_context_length`` must
reject those prompts with HTTP 400 / ``context_length_exceeded``
BEFORE the scheduler starts prefill, so a client that fits inside
the byte cap but exceeds the context window still gets a clean
structured error instead of a wasted ~60 s prefill (F-007 /
rapid-desktop#273).

These tests exercise the helper directly with fake engines so they
run on any machine without model load — the integration with the
chat route is covered by ``test_chat_route_*`` smoke runs.
"""

from __future__ import annotations

import pytest

# Helpers under test depend on the engine contract (``_model.args.*``,
# ``tokenizer.encode``) — both surfaces exist regardless of mlx-lm
# availability, so this file runs everywhere. The chat route import
# itself transitively pulls in mlx, so we don't import it here.


class _StubArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubModel:
    def __init__(self, args=None, config=None):
        if args is not None:
            self.args = args
        if config is not None:
            self.config = config


class _StubTokenizer:
    """Minimal tokenizer surface: ``encode``, ``model_max_length``,
    ``bos_token``. Token count is intentionally deterministic
    (1 token per 4 characters) so the test can hit specific
    boundaries without depending on a real BPE."""

    def __init__(self, model_max_length=None, bos_token=None, chars_per_token=4):
        if model_max_length is not None:
            self.model_max_length = model_max_length
        self.bos_token = bos_token
        self._cpt = chars_per_token

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [0] * max(1, len(text) // self._cpt)


class _StubEngine:
    """Stand-in for ``BatchedEngine`` for the unit tests below. Only
    surfaces the attributes the helpers read — keeps the test fast
    and free of mlx-lm. The real engine wiring is exercised by the
    audio/chat integration tests."""

    is_mllm = False

    def __init__(self, model=None, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer


# ─── get_model_max_context ─────────────────────────────────────────


def test_max_context_from_args_top_level():
    """Plain text LLMs (Llama, Mistral, Qwen3 dense) expose
    ``max_position_embeddings`` directly on ``model.args``."""
    from vllm_mlx.service.helpers import get_model_max_context

    eng = _StubEngine(model=_StubModel(args=_StubArgs(max_position_embeddings=32768)))
    assert get_model_max_context(eng) == 32768


def test_max_context_from_nested_text_config():
    """Multimodal models (Qwen3.5, Gemma 4 VLM) nest the text-config
    under ``model.args.text_config.max_position_embeddings``. The
    helper must walk one level deeper."""
    from vllm_mlx.service.helpers import get_model_max_context

    text_cfg = _StubArgs(max_position_embeddings=262144)
    eng = _StubEngine(model=_StubModel(args=_StubArgs(text_config=text_cfg)))
    assert get_model_max_context(eng) == 262144


def test_max_context_from_model_config_attribute():
    """Some HF-style models expose ``model.config.max_position_embeddings``
    instead of ``model.args``. Resolution must fall through to it
    when ``args`` is absent or unhelpful."""
    from vllm_mlx.service.helpers import get_model_max_context

    cfg = _StubArgs(max_position_embeddings=8192)
    eng = _StubEngine(model=_StubModel(config=cfg))
    assert get_model_max_context(eng) == 8192


def test_max_context_from_tokenizer_when_model_silent():
    """Engines whose loader doesn't propagate the config still
    surface a useful cap via ``tokenizer.model_max_length``."""
    from vllm_mlx.service.helpers import get_model_max_context

    tok = _StubTokenizer(model_max_length=4096)
    eng = _StubEngine(tokenizer=tok)
    assert get_model_max_context(eng) == 4096


def test_max_context_ignores_hf_sentinel_value():
    """HuggingFace tokenizers report ``model_max_length=1e30`` when
    no cap is known. We must NOT treat that as a real cap (it'd
    leave the gate effectively disabled). Helper falls through to
    the fallback in that case."""
    from vllm_mlx.service.helpers import (
        _FALLBACK_MAX_CONTEXT_TOKENS,
        get_model_max_context,
    )

    tok = _StubTokenizer(model_max_length=int(1e30))
    eng = _StubEngine(tokenizer=tok)
    assert get_model_max_context(eng) == _FALLBACK_MAX_CONTEXT_TOKENS


def test_max_context_falls_back_when_no_metadata():
    """Engine with no model and no tokenizer surfaces the fallback
    constant. The fallback is intentionally enormous so legitimate
    requests pass while DoS-shape prompts (≈ millions of tokens)
    still trip."""
    from vllm_mlx.service.helpers import (
        _FALLBACK_MAX_CONTEXT_TOKENS,
        get_model_max_context,
    )

    eng = _StubEngine()
    assert get_model_max_context(eng) == _FALLBACK_MAX_CONTEXT_TOKENS


# ─── count_prompt_tokens ───────────────────────────────────────────


def test_count_prompt_tokens_uses_tokenizer_encode():
    from vllm_mlx.service.helpers import count_prompt_tokens

    eng = _StubEngine(tokenizer=_StubTokenizer(chars_per_token=4))
    assert count_prompt_tokens(eng, "x" * 400) == 100


def test_count_prompt_tokens_returns_zero_on_no_tokenizer():
    """No tokenizer → return 0 so the caller treats the check as a
    metadata edge case rather than a 500."""
    from vllm_mlx.service.helpers import count_prompt_tokens

    eng = _StubEngine()
    assert count_prompt_tokens(eng, "hello") == 0


def test_count_prompt_tokens_handles_encode_exception():
    """If the tokenizer raises (model isn't fully loaded yet, edge
    case), we still want 0 — not a 500. The downstream engine call
    will produce a clean error."""

    class _Broken:
        bos_token = None

        def encode(self, text, add_special_tokens=True):  # noqa: ARG002
            raise RuntimeError("tokenizer not ready")

    from vllm_mlx.service.helpers import count_prompt_tokens

    eng = _StubEngine(tokenizer=_Broken())
    assert count_prompt_tokens(eng, "hi") == 0


# ─── enforce_context_length ────────────────────────────────────────


def test_enforce_under_cap_is_silent():
    """Prompts inside the window pass through with no exception."""
    from vllm_mlx.service.helpers import enforce_context_length

    eng = _StubEngine(model=_StubModel(args=_StubArgs(max_position_embeddings=2048)))
    enforce_context_length(eng, prompt_tokens=512, max_tokens=512)  # 1024 ≤ 2048


def test_enforce_over_cap_raises_400_context_length_exceeded():
    """A prompt past the cap raises HTTP 400 with the OpenAI-style
    ``context_length_exceeded`` envelope. This is the load-bearing
    test — without this we go right back to the F-007 silent hang."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length

    eng = _StubEngine(model=_StubModel(args=_StubArgs(max_position_embeddings=2048)))
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(eng, prompt_tokens=3000, max_tokens=0)

    exc = excinfo.value
    assert exc.status_code == 400
    assert isinstance(exc.detail, dict)
    err = exc.detail["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"
    # The message must mention both the cap and the requested length
    # so the user can shrink the prompt without guessing.
    assert "2048" in err["message"]
    assert "3000" in err["message"]


def test_enforce_includes_max_tokens_in_budget():
    """Prompt fits alone but ``prompt + max_tokens`` exceeds the cap.
    OpenAI's own ``context_length_exceeded`` fires in this case so we
    mirror it — rejecting now avoids a mid-generation truncation."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length

    eng = _StubEngine(model=_StubModel(args=_StubArgs(max_position_embeddings=4096)))
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(eng, prompt_tokens=3500, max_tokens=1000)

    err = excinfo.value.detail["error"]
    assert err["code"] == "context_length_exceeded"
    # Requested = 3500 + 1000 = 4500
    assert "4500" in err["message"]


def test_enforce_tolerates_none_max_tokens():
    """``max_tokens`` is optional in the request; the helper must
    accept ``None`` and only validate the prompt half."""
    from vllm_mlx.service.helpers import enforce_context_length

    eng = _StubEngine(model=_StubModel(args=_StubArgs(max_position_embeddings=2048)))
    enforce_context_length(eng, prompt_tokens=2048, max_tokens=None)  # equal → ok


# ─── End-to-end shape: structured handler passes envelope through ───


def test_structured_detail_passes_through_global_handler():
    """The global HTTP exception handler in ``vllm_mlx/server.py``
    must recognise ``HTTPException(detail={"error": {...}})`` and
    emit the structured envelope unchanged, so SDKs that introspect
    ``error.code`` see ``context_length_exceeded`` rather than a
    stringified Python dict.

    This guards against a refactor of the handler that re-wraps the
    detail and silently strips ``code`` / ``param`` from
    structured-detail callers."""
    pytest.importorskip("mlx_lm")  # global handler module pulls in mlx

    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    @app.post("/v1/test")
    async def _handler():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "boom",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "param": "messages",
                }
            },
        )

    # Re-register the global handler from vllm_mlx.server on our tiny app
    # so we exercise the actual production handler, not FastAPI's default.
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from vllm_mlx.server import _http_exception_handler

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post("/v1/test")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["param"] == "messages"
    assert body["error"]["message"] == "boom"
