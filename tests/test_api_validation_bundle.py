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
        embed = [[0.1 * i for i in range(384)]]
        client, _ = _build_embed_app(patched_config, monkeypatch, embed)
        r = client.post(
            "/v1/embeddings",
            json={"model": "any", "input": "hello", "dimensions": 64},
        )
        assert r.status_code == 200, r.text
        vec = r.json()["data"][0]["embedding"]
        assert len(vec) == 64
        assert vec[:3] == [0.0, 0.1, 0.2]

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
        length matches `dimensions`."""
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
        decoded = struct.unpack("<4f", base64.b64decode(encoded))
        assert list(decoded) == [0.0, 1.0, 2.0, 3.0]


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
