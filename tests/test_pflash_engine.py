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
async def test_chat_marks_tool_prompts_with_has_tools_not_integrity():
    """Tool requests get ``has_tools=True`` but NOT
    ``requires_prompt_integrity=True``. The latter is reserved for
    schema/response_format prompts that have no opt-out; tools opt out
    via ``--pflash-include-tools`` which inverts
    ``PFlashConfig.skip_when_tools``. Auto-folding tools into
    ``requires_prompt_integrity`` short-circuits the skip_when_tools
    branch and makes the documented CLI flag dead (codex r6
    BLOCKING). The route layer (``vllm_mlx/routes/chat.py``) may still
    set ``requires_prompt_integrity`` explicitly for schema requests
    that also happen to have tools — that pass-through case is
    covered by the next test.
    """
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
    # NOT set by the engine for tools — gated via ``has_tools`` instead.
    assert captured.get("requires_prompt_integrity", False) is False


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
async def test_stream_chat_marks_tool_prompts_with_has_tools_not_integrity():
    """Stream parity for the codex r6 fix: ``stream_chat`` must NOT
    auto-fold tools into ``requires_prompt_integrity`` either, otherwise
    streaming tool calls would silently take the integrity path while
    non-streaming wouldn't.
    """
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
    assert captured.get("requires_prompt_integrity", False) is False


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
