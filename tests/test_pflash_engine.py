# SPDX-License-Identifier: Apache-2.0
"""Engine-layer routing tests for PFlash (#287).

These verify that ``BatchedEngine.chat`` / ``BatchedEngine.stream_chat``
correctly set ``has_tools`` and ``requires_prompt_integrity`` on
downstream calls. The route in vllm_mlx/routes/chat.py also sets
``requires_prompt_integrity`` explicitly for tool / response_format
requests; these tests cover the engine-level redundancy.

Ported from @michaelasper's reference implementation
(``tests/test_pflash_engine.py`` at commit b6089ce).
"""

import types

import pytest

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


def _engine_with_template() -> BatchedEngine:
    """Construct a BatchedEngine in a state that bypasses model load,
    so we can assert routing decisions without touching MLX."""
    engine = BatchedEngine("mlx-community/Qwen3-0.6B-4bit")
    engine._loaded = True
    engine._is_mllm = False
    engine._apply_chat_template = lambda *args, **kwargs: "rendered prompt"
    engine._compute_prefix_boundary = lambda *args, **kwargs: 0
    engine._is_hybrid_model = lambda: False
    return engine


@pytest.mark.asyncio
async def test_chat_marks_tool_prompts_as_integrity_required():
    engine = _engine_with_template()
    captured: dict = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    assert captured["has_tools"] is True
    assert captured["requires_prompt_integrity"] is True


@pytest.mark.asyncio
async def test_chat_preserves_route_supplied_integrity_flag_for_schema_prompts():
    engine = _engine_with_template()
    captured: dict = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "return json"}],
        requires_prompt_integrity=True,
    )

    assert "has_tools" not in captured or captured["has_tools"] is False
    assert captured["requires_prompt_integrity"] is True


@pytest.mark.asyncio
async def test_stream_chat_marks_tool_prompts_as_integrity_required():
    engine = _engine_with_template()
    captured: dict = {}

    async def fake_stream_generate(self, **kwargs):
        captured.update(kwargs)
        yield GenerationOutput(text="ok", finished=True)

    engine.stream_generate = types.MethodType(fake_stream_generate, engine)

    outputs = []
    async for output in engine.stream_chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    ):
        outputs.append(output)

    assert outputs
    assert captured["has_tools"] is True
    assert captured["requires_prompt_integrity"] is True


@pytest.mark.asyncio
async def test_chat_without_tools_does_not_mark_integrity():
    engine = _engine_with_template()
    captured: dict = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(messages=[{"role": "user", "content": "hi"}])

    # Regression guard: plain chat traffic must NOT carry the integrity
    # flag, otherwise PFlash would degrade into a no-op on normal flows.
    assert captured.get("has_tools", False) is False
    assert captured.get("requires_prompt_integrity", False) is False
