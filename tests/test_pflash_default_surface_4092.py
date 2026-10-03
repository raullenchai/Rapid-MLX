# SPDX-License-Identifier: Apache-2.0
"""#4092: PFlash must never silently compress ordinary long prompts.

Two halves:

* Default policy — verified aliases default to ``--pflash auto`` (gated on
  ``--pflash-threshold``, 32 768 tokens) instead of ``always``, which used to
  compress every no-tools prompt above ~11.5K tokens (long chat-app sessions,
  RAG, document Q&A) and drop ~80% of its middle. Explicit ``--pflash always``
  keeps working.
* Surfacing — when compression does happen, the response says so: a
  ``metrics.prompt_compression`` block on the terminal response/chunk of the
  OpenAI-shaped routes and an ``X-Rapid-MLX-Prompt-Compressed: <kept>/<total>``
  header on non-streaming responses (the only signal the Anthropic schema
  allows). Responses for non-compressed requests stay byte-identical to
  origin/main (golden fixture captured from main).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.output_router import Channel
from rapid_mlx.pflash import (
    compress_tokens,
    resolve_pflash_config,
    resolve_pflash_mode_default,
)
from rapid_mlx.service.helpers import (
    PROMPT_COMPRESSED_HEADER,
    _build_response_metrics,
    _merge_response_metrics,
    prompt_compression_headers,
)

_COMPRESSION = {"original_tokens": 40_000, "kept_tokens": 8_000}
_GOLDEN = Path(__file__).parent / "fixtures" / "pflash_4092_routes_golden_main.json"


def _cli_ns(**overrides) -> SimpleNamespace:
    """argparse-shaped namespace with every PFlash flag at its CLI default."""
    base = dict(
        pflash=None,
        pflash_threshold=32_768,
        pflash_keep_ratio=None,
        pflash_min_keep_tokens=2048,
        pflash_sink_tokens=256,
        pflash_tail_tokens=2048,
        pflash_block_size=128,
        pflash_query_window=512,
        pflash_stride_blocks=8,
        pflash_include_tools=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------


class TestVerifiedDefaultIsThresholdGated:
    @pytest.mark.parametrize("n_tokens", [12_000, 20_000, 32_767])
    def test_ordinary_long_chat_prompt_is_untouched_by_default(self, n_tokens):
        # The #4092 regression: under the old ``always`` default each of these
        # no-tools prompts kept only sink + tail + 20% and dropped the rest.
        config = resolve_pflash_config(_cli_ns(), model_name="qwen3.6-27b-4bit")
        assert config.mode == "auto"
        assert config.threshold == 32_768

        result = compress_tokens(list(range(n_tokens)), config)

        assert result.compressed is False
        assert result.reason == "threshold"
        assert result.kept_tokens == n_tokens

    @pytest.mark.parametrize("n_tokens", [32_768, 40_000])
    def test_prompt_at_or_above_threshold_still_compresses_by_default(self, n_tokens):
        # The validated #649 win (>= 32K cold prefill) is kept by default;
        # "at least --pflash-threshold" includes the boundary itself.
        config = resolve_pflash_config(_cli_ns(), model_name="qwen3.6-27b-4bit")

        result = compress_tokens(list(range(n_tokens)), config)

        assert result.compressed is True
        assert result.kept_tokens < n_tokens

    def test_explicit_always_still_compresses_ordinary_long_prompt(self):
        config = resolve_pflash_config(
            _cli_ns(pflash="always"), model_name="qwen3.6-27b-4bit"
        )

        result = compress_tokens(list(range(20_000)), config)

        assert config.mode == "always"
        assert result.compressed is True

    def test_default_log_names_auto_and_the_threshold(self, caplog):
        with caplog.at_level(logging.INFO, logger="rapid_mlx.pflash"):
            mode = resolve_pflash_mode_default(
                _cli_ns(pflash_threshold=50_000), model_name="qwen3.5-4b-4bit"
            )
        assert mode == "auto"
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "--pflash auto" in message
        assert "50000 tokens" in message
        assert "--pflash always" in message


# ---------------------------------------------------------------------------
# Response metrics + header helpers
# ---------------------------------------------------------------------------


class TestPromptCompressionMetrics:
    def test_compression_only_builds_envelope_without_speculative_block(self):
        metrics = _build_response_metrics(
            GenerationOutput(text="ok", prompt_compression=_COMPRESSION)
        )
        assert metrics is not None
        assert metrics.speculative_decoding is None
        assert metrics.model_dump(exclude_none=True) == {
            "prompt_compression": _COMPRESSION
        }

    def test_both_blocks_coexist(self):
        metrics = _build_response_metrics(
            GenerationOutput(
                text="ok",
                prompt_compression=_COMPRESSION,
                spec_decode_metrics={"verify_calls": 1},
            )
        )
        assert metrics is not None
        assert metrics.speculative_decoding is not None
        assert metrics.speculative_decoding.verify_calls == 1
        assert metrics.prompt_compression is not None

    def test_non_schema_attributes_fail_closed(self):
        # MagicMock attributes and non-dict payloads must never reach the wire.
        assert _build_response_metrics(MagicMock()) is None
        assert (
            _build_response_metrics(SimpleNamespace(prompt_compression=[1, 2])) is None
        )
        assert _build_response_metrics(GenerationOutput(text="plain")) is None

    def test_header_value_is_kept_over_total(self):
        metrics = _build_response_metrics(
            GenerationOutput(text="ok", prompt_compression=_COMPRESSION)
        )
        assert prompt_compression_headers(metrics) == {
            PROMPT_COMPRESSED_HEADER: "8000/40000"
        }

    def test_no_header_without_compression(self):
        assert prompt_compression_headers(None) == {}
        speculative_only = _build_response_metrics(
            GenerationOutput(text="ok", spec_decode_metrics={"verify_calls": 1})
        )
        assert prompt_compression_headers(speculative_only) == {}

    def test_multi_prompt_merge_sums_compressed_prompts(self):
        merged = _merge_response_metrics(
            [
                GenerationOutput(text="a", prompt_compression=_COMPRESSION),
                GenerationOutput(text="plain"),
                GenerationOutput(
                    text="b",
                    prompt_compression={
                        "original_tokens": 50_000,
                        "kept_tokens": 10_000,
                    },
                ),
            ]
        )
        assert merged is not None
        assert merged.speculative_decoding is None
        assert merged.prompt_compression is not None
        assert merged.prompt_compression.model_dump() == {
            "original_tokens": 90_000,
            "kept_tokens": 18_000,
        }

    def test_merge_without_any_metrics_stays_none(self):
        assert _merge_response_metrics([GenerationOutput(text="plain")]) is None


# ---------------------------------------------------------------------------
# Engine plumbing (no MLX)
# ---------------------------------------------------------------------------


def test_output_collector_merge_keeps_terminal_compression():
    from rapid_mlx.output_collector import RequestOutputCollector
    from rapid_mlx.request import RequestOutput

    collector = RequestOutputCollector(aggregate=True)
    terminal = RequestOutput(
        request_id="r1", new_token_ids=[1], prompt_compression=_COMPRESSION
    )
    later = RequestOutput(request_id="r1", new_token_ids=[2])

    assert collector._merge_outputs(terminal, later).prompt_compression == (
        _COMPRESSION
    )
    assert collector._merge_outputs(later, terminal).prompt_compression == (
        _COMPRESSION
    )


def test_routed_outputs_carry_prompt_compression():
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    source = GenerationOutput(text="x", prompt_compression=_COMPRESSION)
    event = SimpleNamespace(text="x", token_id=1, channel=Channel.CONTENT)

    routed = engine._make_routed_output(source, event)
    sentinel = engine._routed_finish_sentinel(source)

    assert routed.prompt_compression == _COMPRESSION
    assert sentinel.prompt_compression == _COMPRESSION


@pytest.mark.requires_mlx
def test_engine_core_stream_buffer_keeps_compression():
    pytest.importorskip("mlx")
    from rapid_mlx.engine_core import EngineCore
    from rapid_mlx.request import RequestOutput

    first = RequestOutput(
        request_id="r1", new_token_ids=[1], prompt_compression=_COMPRESSION
    )
    second = RequestOutput(request_id="r1", new_token_ids=[2])

    assert EngineCore._merge_stream_buffer(None, first).prompt_compression == (
        _COMPRESSION
    )
    assert EngineCore._merge_stream_buffer(first, second).prompt_compression == (
        _COMPRESSION
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, compression: dict | None = None):
        self._compression = compression

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def _out(self, text: str, new_text: str, finished: bool, index: int):
        return GenerationOutput(
            text=text,
            raw_text=text,
            new_text=new_text,
            prompt_tokens=40_000,
            completion_tokens=index + 1,
            finished=finished,
            finish_reason="stop" if finished else None,
            # The scheduler pins compression on the terminal output only.
            prompt_compression=self._compression if finished else None,
        )

    async def chat(self, messages, **kwargs):
        return self._out("hi there", "", True, 1)

    async def generate(self, prompt, **kwargs):
        return self._out("hi there", "", True, 1)

    async def stream_chat(self, messages, **kwargs):
        yield self._out("hi", "hi", False, 0)
        yield self._out("hi there", " there", True, 1)

    async def stream_generate(self, prompt, **kwargs):
        yield self._out("hi", "hi", False, 0)
        yield self._out("hi there", " there", True, 1)


def _client(engine: _Engine) -> TestClient:
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.completions import router as completions_router
    from rapid_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    app = FastAPI()
    for router in (chat_router, completions_router, responses_router, anthropic_router):
        app.include_router(router)
    return TestClient(app)


_MESSAGES = [{"role": "user", "content": "hello"}]
_ROUTES = {
    "chat": (
        "/v1/chat/completions",
        {"model": "test-model", "messages": _MESSAGES, "max_tokens": 8},
    ),
    "completions": (
        "/v1/completions",
        {"model": "test-model", "prompt": "hello", "max_tokens": 8},
    ),
    "responses": (
        "/v1/responses",
        {"model": "test-model", "input": "hello", "max_output_tokens": 16},
    ),
    "anthropic": (
        "/v1/messages",
        {"model": "test-model", "messages": _MESSAGES, "max_tokens": 8},
    ),
}
_STREAM_BODY = {
    "chat": {"stream": True, "stream_options": {"include_usage": True}},
    "completions": {"stream": True},
    "responses": {"stream": True},
    "anthropic": {"stream": True},
}
# Every route, non-streaming and streaming: (fixture key, path, body).
_CASES = {
    **{name: (path, body) for name, (path, body) in _ROUTES.items()},
    **{
        f"{name}_stream": (path, {**body, **_STREAM_BODY[name]})
        for name, (path, body) in _ROUTES.items()
    },
}


def _normalize(body: str) -> str:
    """Blank out per-response ids, timestamps and SSE keepalive comments (the
    same rules the origin/main capture applied)."""
    body = re.sub(r"^: keepalive\n\n", "", body, flags=re.M)
    body = re.sub(
        r'"(id|created|created_at|item_id|response_id)":\s*("[^"]*"|\d+)',
        r'"\1":"X"',
        body,
    )
    body = re.sub(
        r"(chatcmpl|cmpl|resp|msg|rs|fc|call|item)_[A-Za-z0-9]+", r"\1_X", body
    )
    return re.sub(r'"(created|created_at)":\s*\d+', r'"\1":0', body)


@pytest.fixture
def restore_config():
    yield
    reset_config()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_uncompressed_response_is_byte_identical_to_main(case, restore_config):
    golden = json.loads(_GOLDEN.read_text())[case]
    path, body = _CASES[case]

    response = _client(_Engine()).post(path, json=body)

    headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in ("date", "content-length")
    }
    assert response.status_code == golden["status"]
    assert headers == golden["headers"]
    assert _normalize(response.text) == golden["body"]


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_compressed_non_stream_response_announces_compression(route, restore_config):
    path, body = _ROUTES[route]

    response = _client(_Engine(_COMPRESSION)).post(path, json=body)

    assert response.status_code == 200, response.text
    assert response.headers[PROMPT_COMPRESSED_HEADER] == "8000/40000"
    payload = response.json()
    # Client-visible usage keeps the logical prompt size.
    usage = payload["usage"]
    assert usage.get("prompt_tokens", usage.get("input_tokens")) == 40_000
    if route == "anthropic":
        # The Anthropic schema has no extension slot: header only.
        assert "metrics" not in payload
    else:
        assert payload["metrics"] == {"prompt_compression": _COMPRESSION}


def _sse_payloads(text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data:").strip())
        for line in text.splitlines()
        if line.startswith("data:") and line.strip() != "data: [DONE]"
    ]


def _find_metrics(node) -> list[dict]:
    """Every ``metrics`` object anywhere in one SSE payload."""
    if isinstance(node, dict):
        found = [node["metrics"]] if "metrics" in node else []
        for value in node.values():
            found.extend(_find_metrics(value))
        return found
    if isinstance(node, list):
        return [m for item in node for m in _find_metrics(item)]
    return []


@pytest.mark.parametrize("route", ["chat", "completions", "responses"])
def test_compressed_stream_reports_once_on_terminal_event(route, restore_config):
    path, body = _CASES[f"{route}_stream"]

    response = _client(_Engine(_COMPRESSION)).post(path, json=body)

    assert response.status_code == 200
    # Streaming headers leave before the scheduler decides; no header.
    assert PROMPT_COMPRESSED_HEADER not in response.headers
    payloads = _sse_payloads(response.text)
    carrying = [p for p in payloads if _find_metrics(p)]
    assert len(carrying) == 1, carrying
    assert _find_metrics(carrying[0]) == [{"prompt_compression": _COMPRESSION}]
    if route == "responses":
        assert carrying[0]["type"] == "response.completed"
    else:
        assert carrying[0]["choices"][0]["finish_reason"] == "stop"
