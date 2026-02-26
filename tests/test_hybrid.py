# SPDX-License-Identifier: Apache-2.0
"""Tests for HybridEngine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_mlx.engine.base import GenerationOutput


def _make_engine(**overrides):
    """Create HybridEngine with all heavy imports mocked."""
    with patch("vllm_mlx.engine.hybrid.load"), \
         patch("vllm_mlx.engine.hybrid.SimpleEngine"), \
         patch("vllm_mlx.engine.hybrid.BatchedEngine"), \
         patch("vllm_mlx.engine.hybrid.get_registry"):
        from vllm_mlx.engine.hybrid import HybridEngine
        defaults = {"model_name": "test_model"}
        defaults.update(overrides)
        return HybridEngine(**defaults)


def _make_mock_simple():
    """Create a mock SimpleEngine."""
    engine = MagicMock()
    engine.generate = AsyncMock(return_value=GenerationOutput(text="simple output"))
    engine.chat = AsyncMock(return_value=GenerationOutput(text="simple chat"))
    engine.stop = AsyncMock()
    engine._inject_shared_model = AsyncMock()
    engine.get_stats = MagicMock(return_value={"type": "simple"})
    engine.get_cache_stats = MagicMock(return_value={"cache": "simple"})

    async def _stream_gen(**kwargs):
        yield GenerationOutput(text="", new_text="chunk1", finished=False)
        yield GenerationOutput(text="chunk1chunk2", new_text="chunk2", finished=True)
    engine.stream_generate = _stream_gen

    async def _stream_chat(**kwargs):
        yield GenerationOutput(text="", new_text="c1", finished=False)
        yield GenerationOutput(text="c1c2", new_text="c2", finished=True)
    engine.stream_chat = _stream_chat

    return engine


def _make_mock_batched():
    """Create a mock BatchedEngine."""
    engine = MagicMock()
    engine.generate = AsyncMock(return_value=GenerationOutput(text="batched output"))
    engine.chat = AsyncMock(return_value=GenerationOutput(text="batched chat"))
    engine.stop = AsyncMock()
    engine.start = AsyncMock()
    engine._inject_shared_model = AsyncMock()
    engine.get_stats = MagicMock(return_value={"type": "batched"})
    engine.get_cache_stats = MagicMock(return_value={"cache": "batched"})
    engine._engine = MagicMock()
    engine._engine.engine = MagicMock()
    engine._engine.engine.engine_id = "test_id"
    engine._engine.engine.scheduler = MagicMock()
    engine._engine.engine.start = AsyncMock()

    async def _stream_gen(**kwargs):
        yield GenerationOutput(text="", new_text="b1", finished=False)
        yield GenerationOutput(text="b1b2", new_text="b2", finished=True)
    engine.stream_generate = _stream_gen

    async def _stream_chat(**kwargs):
        yield GenerationOutput(text="", new_text="bc1", finished=False)
    engine.stream_chat = _stream_chat

    return engine


# ── Init ──────────────────────────────────────────────────────────────────

class TestHybridEngineInit:
    def test_default_params(self):
        engine = _make_engine()
        assert engine._model_name == "test_model"
        assert engine._draft_model_name is None
        assert engine._num_draft_tokens == 5
        assert engine._switch_threshold == 2
        assert engine._loaded is False
        assert engine._is_mllm is False
        assert engine._current_mode is None
        assert engine._active_requests == 0

    def test_custom_params(self):
        engine = _make_engine(
            draft_model="draft", num_draft_tokens=10,
            switch_threshold=5, force_mllm=True,
        )
        assert engine._draft_model_name == "draft"
        assert engine._num_draft_tokens == 10
        assert engine._switch_threshold == 5
        assert engine._force_mllm is True


# ── Properties ────────────────────────────────────────────────────────────

class TestProperties:
    def test_model_name(self):
        engine = _make_engine()
        assert engine.model_name == "test_model"

    def test_is_mllm_default(self):
        engine = _make_engine()
        assert engine.is_mllm is False

    def test_tokenizer(self):
        engine = _make_engine()
        engine._shared_tokenizer = "tok"
        assert engine.tokenizer == "tok"


# ── Start / Stop ──────────────────────────────────────────────────────────

class TestStart:
    @pytest.mark.asyncio
    async def test_start_normal_mode(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model")
            await engine.start()

            assert engine._loaded is True
            assert engine._current_mode == "simple"
            assert engine._simple is mock_simple
            assert engine._batched is mock_batched
            mock_simple._inject_shared_model.assert_called_once()
            mock_batched._inject_shared_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_mllm_mode(self):
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine"), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", force_mllm=True)
            await engine.start()

            assert engine._is_mllm is True
            assert engine._current_mode == "batched"
            mock_batched.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        mock_model, mock_tok = MagicMock(), MagicMock()
        mock_batched = _make_mock_batched()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)) as mock_load, \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=_make_mock_simple()), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model")
            await engine.start()
            await engine.start()  # second call no-op
            mock_load.assert_called_once()


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model")
            await engine.start()
            await engine.stop()

            mock_simple.stop.assert_called_once()
            mock_batched.stop.assert_called_once()
            assert engine._shared_model is None
            assert engine._shared_tokenizer is None
            assert engine._loaded is False
            assert engine._current_mode is None


# ── _get_engine_for_request ───────────────────────────────────────────────

class TestGetEngineForRequest:
    def test_mllm_returns_batched(self):
        engine = _make_engine()
        engine._is_mllm = True
        engine._batched = MagicMock()
        assert engine._get_engine_for_request() is engine._batched

    def test_simple_mode(self):
        engine = _make_engine()
        engine._current_mode = "simple"
        engine._simple = MagicMock()
        assert engine._get_engine_for_request() is engine._simple

    def test_batched_mode(self):
        engine = _make_engine()
        engine._current_mode = "batched"
        engine._batched = MagicMock()
        assert engine._get_engine_for_request() is engine._batched


# ── Mode Switching ────────────────────────────────────────────────────────

class TestModeSwitching:
    @pytest.mark.asyncio
    async def test_switch_to_same_mode_noop(self):
        engine = _make_engine()
        engine._current_mode = "simple"
        await engine._switch_to_mode("simple")
        assert engine._current_mode == "simple"

    @pytest.mark.asyncio
    async def test_decide_mllm_always_batched(self):
        engine = _make_engine()
        engine._is_mllm = True
        engine._active_requests = 0
        mode = await engine._decide_and_switch_mode()
        assert mode == "batched"

    @pytest.mark.asyncio
    async def test_decide_above_threshold_switches_batched(self):
        engine = _make_engine(switch_threshold=2)
        engine._current_mode = "simple"
        engine._active_requests = 3

        with patch.object(engine, "_switch_to_mode", new_callable=AsyncMock) as mock_switch:
            await engine._decide_and_switch_mode(entering=True)
            mock_switch.assert_called_with("batched")

    @pytest.mark.asyncio
    async def test_decide_below_threshold_idle_switches_simple(self):
        engine = _make_engine(switch_threshold=2)
        engine._current_mode = "batched"
        engine._active_requests = 0

        with patch.object(engine, "_switch_to_mode", new_callable=AsyncMock) as mock_switch:
            await engine._decide_and_switch_mode(entering=False)
            mock_switch.assert_called_with("simple")

    @pytest.mark.asyncio
    async def test_decide_below_threshold_nonzero_stays(self):
        """When below threshold but active_requests > 0, don't switch back to simple."""
        engine = _make_engine(switch_threshold=2)
        engine._current_mode = "batched"
        engine._active_requests = 1

        with patch.object(engine, "_switch_to_mode", new_callable=AsyncMock) as mock_switch:
            await engine._decide_and_switch_mode(entering=False)
            mock_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_to_batched_starts_lazy_engine(self):
        """Switching to batched mode starts lazy engine."""
        engine = _make_engine()
        engine._current_mode = "simple"
        engine._batched_engine_started = False

        mock_batched = _make_mock_batched()
        engine._batched = mock_batched

        with patch("vllm_mlx.engine.hybrid.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_reg.return_value = mock_registry
            await engine._switch_to_mode("batched")

            assert engine._current_mode == "batched"
            assert engine._batched_engine_started is True
            mock_batched._engine.engine.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_to_simple_resets_scheduler(self):
        """Switching to simple resets batched scheduler."""
        engine = _make_engine()
        engine._current_mode = "batched"
        mock_batched = _make_mock_batched()
        engine._batched = mock_batched

        await engine._switch_to_mode("simple")

        assert engine._current_mode == "simple"
        mock_batched._engine.engine.scheduler.deep_reset.assert_called_once()


# ── generate ──────────────────────────────────────────────────────────────

class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_uses_simple(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            await engine.start()

            result = await engine.generate("test prompt")
            assert result.text == "simple output"
            assert engine._active_requests == 0

    @pytest.mark.asyncio
    async def test_generate_auto_starts(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            # Don't call start() — generate() should auto-start
            result = await engine.generate("test")
            assert engine._loaded is True
            assert result.text == "simple output"

    @pytest.mark.asyncio
    async def test_generate_decrements_on_error(self):
        mock_simple = _make_mock_simple()
        mock_simple.generate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            await engine.start()

            with pytest.raises(RuntimeError, match="boom"):
                await engine.generate("test")
            assert engine._active_requests == 0  # Properly decremented


# ── stream_generate ───────────────────────────────────────────────────────

class TestStreamGenerate:
    @pytest.mark.asyncio
    async def test_stream_generate_yields_chunks(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            await engine.start()

            chunks = []
            async for output in engine.stream_generate("test"):
                chunks.append(output.new_text)

            assert len(chunks) == 2
            assert chunks == ["chunk1", "chunk2"]
            assert engine._active_requests == 0


# ── chat ──────────────────────────────────────────────────────────────────

class TestChat:
    @pytest.mark.asyncio
    async def test_chat_delegates(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            await engine.start()

            msgs = [{"role": "user", "content": "hello"}]
            result = await engine.chat(msgs)
            assert result.text == "simple chat"
            assert engine._active_requests == 0


# ── stream_chat ───────────────────────────────────────────────────────────

class TestStreamChat:
    @pytest.mark.asyncio
    async def test_stream_chat_yields(self):
        mock_simple = _make_mock_simple()
        mock_batched = _make_mock_batched()
        mock_model, mock_tok = MagicMock(), MagicMock()

        with patch("vllm_mlx.engine.hybrid.load", return_value=(mock_model, mock_tok)), \
             patch("vllm_mlx.api.utils.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.hybrid.SimpleEngine", return_value=mock_simple), \
             patch("vllm_mlx.engine.hybrid.BatchedEngine", return_value=mock_batched), \
             patch("vllm_mlx.engine.hybrid.get_registry"):
            from vllm_mlx.engine.hybrid import HybridEngine
            engine = HybridEngine(model_name="test_model", switch_threshold=10)
            await engine.start()

            chunks = []
            msgs = [{"role": "user", "content": "hi"}]
            async for output in engine.stream_chat(msgs):
                chunks.append(output.new_text)

            assert len(chunks) == 2
            assert chunks == ["c1", "c2"]
            assert engine._active_requests == 0


# ── get_stats ─────────────────────────────────────────────────────────────

class TestGetStats:
    def test_initial_stats(self):
        engine = _make_engine()
        stats = engine.get_stats()
        assert stats["engine_type"] == "hybrid"
        assert stats["model_name"] == "test_model"
        assert stats["loaded"] is False
        assert stats["current_mode"] is None
        assert stats["active_requests"] == 0
        assert stats["switch_threshold"] == 2

    def test_stats_includes_sub_engines(self):
        engine = _make_engine()
        engine._simple = _make_mock_simple()
        engine._batched = _make_mock_batched()
        stats = engine.get_stats()
        assert "simple_engine" in stats
        assert "batched_engine" in stats
        assert stats["simple_engine"] == {"type": "simple"}
        assert stats["batched_engine"] == {"type": "batched"}


# ── get_cache_stats ───────────────────────────────────────────────────────

class TestGetCacheStats:
    def test_cache_stats_simple_mode(self):
        engine = _make_engine()
        engine._current_mode = "simple"
        engine._simple = _make_mock_simple()
        result = engine.get_cache_stats()
        assert result == {"cache": "simple"}

    def test_cache_stats_batched_mode(self):
        engine = _make_engine()
        engine._current_mode = "batched"
        engine._batched = _make_mock_batched()
        result = engine.get_cache_stats()
        assert result == {"cache": "batched"}

    def test_cache_stats_none_when_no_engine(self):
        engine = _make_engine()
        engine._current_mode = None
        result = engine.get_cache_stats()
        assert result is None
