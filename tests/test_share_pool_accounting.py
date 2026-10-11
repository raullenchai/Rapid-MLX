# SPDX-License-Identifier: Apache-2.0
"""Pool status from real batched/MTP completions, with only model doubles."""

import asyncio
import json
from unittest.mock import patch

import pytest


@pytest.mark.requires_mlx
@pytest.mark.asyncio
async def test_pool_status_tracks_streaming_and_nonstreaming_mtp(capsys):
    try:
        import mlx.core  # noqa: F401 — ensure the runtime can open Metal
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Pool accounting requires a working Metal device: {exc}")
    import httpx
    from fastapi import FastAPI

    from rapid_mlx.config import reset_config
    from tests.test_share_progress import metrics_response

    # Match the repository's headless-runtime skip convention: the engine
    # imports require Metal even when only model doubles are used.
    try:
        from rapid_mlx.engine.batched import BatchedEngine
    except RuntimeError as exc:
        pytest.skip(f"Batched/MTP accounting requires a working Metal device: {exc}")
    from rapid_mlx.engine_core import AsyncEngineCore, EngineConfig
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.metrics import router as metrics_router
    from rapid_mlx.runtime.model_performance import (
        _reset_model_performance_registry_for_tests,
    )
    from rapid_mlx.scheduler import SchedulerConfig
    from rapid_mlx.share.quicksilver import _PoolStatus
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )
    from tests.test_mtp_spec_decode import _MockedQwen35Model

    class Tokenizer:
        eos_token_id = 31
        eos_token_ids = [31]
        bos_token_id = None
        chat_template = "test"

        def encode(self, text, **kwargs):
            return [1]

        def decode(self, tokens, **kwargs):
            return "ok " * len(tokens)

        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return [1] if tokenize else "prompt"

        def get_vocab(self):
            return {"ok": 7, "<eos>": 31}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)

    _reset_model_performance_registry_for_tests()
    name = "qwen3.8-27b-4bit"
    # The real scheduler installs and executes the MTP generator. Constant
    # target/draft logits make accepted draft tokens predictable across turns.
    model = _MockedQwen35Model([7] * 512, [7] * 512)
    model.mtp = object()
    tokenizer = Tokenizer()
    sc = SchedulerConfig(
        max_num_seqs=2,
        model_name=name,
        enable_prefix_cache=False,
        spec_decode="mtp",
        mtp_model_type="qwen3_5",
        mtp_disable_auto_k=True,
    )
    core = AsyncEngineCore(
        model,
        tokenizer,
        EngineConfig(model_name=name, scheduler_config=sc, force_hybrid=True),
    )
    engine = BatchedEngine(name, scheduler_config=sc)
    engine._engine = core
    engine._model = model
    engine._tokenizer = tokenizer
    engine._loaded = True
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "qwen3.8-27b"
    cfg.model_alias = name
    cfg.no_thinking = True
    cfg.max_tokens = 16
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(metrics_router)
    reset_global_counter_for_tests()
    await core.engine.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:

            async def complete(stream, max_tokens):
                response = await asyncio.wait_for(
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": cfg.model_name,
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": stream,
                            "stream_options": {"include_usage": True}
                            if stream
                            else None,
                            "max_tokens": max_tokens,
                            "temperature": 0,
                        },
                    ),
                    timeout=15,
                )
                assert response.status_code == 200, response.text
                if not stream:
                    return response.json()["usage"]["completion_tokens"]
                chunks = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                return next(
                    chunk["usage"]["completion_tokens"]
                    for chunk in chunks
                    if chunk.get("usage")
                )

            assert await complete(False, 2) == 2  # Local warm-up.
            initial = (await client.get("/metrics")).content
            # Only replace the HTTP transport. Every metric is rendered from
            # the real scheduler/ledger; none is stubbed or manually incremented.
            with patch(
                "rapid_mlx.share.quicksilver._open",
                return_value=metrics_response(initial),
            ):
                status = _PoolStatus(8765, lambda: True)
            assert status.baseline[:2] == (1, 2)
            streamed = await complete(True, 6)
            nonstreamed = await complete(False, 5)
            assert (streamed, nonstreamed) == (6, 5)
            assert core.engine.scheduler.spec_decode_runtime_method == "mtp"
            assert core.engine.model_config.is_hybrid
            assert get_global_counter().snapshot().verify_calls > 0
            # A one-token readiness-shaped request is indistinguishable in
            # request/token aggregates, but has no decode-speed observation.
            assert await complete(False, 1) == 1
            stats = engine.get_stats()
            assert stats["num_requests_processed"] == 4
            assert stats["total_completion_tokens"] == 14
            current = (await client.get("/metrics")).content
            with (
                patch(
                    "rapid_mlx.share.quicksilver._open",
                    return_value=metrics_response(current),
                ),
                patch("rapid_mlx.share.quicksilver.time.monotonic", return_value=20),
            ):
                status.poll(True)
            output = capsys.readouterr().err
            assert "3 requests served, 12 output tokens since start" in output
            assert status.last_counts == (3, 12, True)
            assert b"rapid_mlx_spec_decode" in current
            assert stats["model_performance"]["decode_observations"] == 3
    finally:
        await core.stop()
        core.engine.close()
        reset_global_counter_for_tests()
        reset_config()
        _reset_model_performance_registry_for_tests()
