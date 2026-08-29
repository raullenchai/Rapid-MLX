# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #1380: the embedding engine must release MLX
Metal buffers (``mx.clear_cache()``) after each completed batch, matching
every LLM path in the engine.

Without it the MLX allocator pool grows monotonically: ``padding=True`` makes
the per-batch sequence length vary, so nearly every request asks for a buffer
size the pool has never seen and cannot reuse (measured upstream: 2.3 GB → 24
GB over 320 texts). Pure-logic test: mocked model + tokenizer, no model
download and no GPU allocation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from unittest.mock import MagicMock, patch

import numpy as np

from vllm_mlx.embedding import EmbeddingEngine


def _mock_engine() -> EmbeddingEngine:
    eng = EmbeddingEngine("test-model")
    mock_output = MagicMock()
    mock_output.text_embeds.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    eng._model = MagicMock(return_value=mock_output)

    inner_tok = MagicMock(
        return_value={
            "input_ids": np.array([[1, 2], [3, 4]]),
            "attention_mask": np.array([[1, 1], [1, 1]]),
        }
    )
    tok = MagicMock()
    tok._tokenizer = inner_tok
    # A real pad token id (0). Both attrs are set because embed_tokens'
    # pad-id resolver falls through ``pad_token_id or _tokenizer.pad_token_id
    # or 0`` — a bare MagicMock on the inner tokenizer would be treated as a
    # truthy pad id and break mx.array() on the padded batch.
    tok.pad_token_id = 0
    inner_tok.pad_token_id = 0
    eng._tokenizer = tok
    return eng


@patch("vllm_mlx.embedding.mx.clear_cache")
def test_embed_releases_buffers_after_batch(mock_clear):
    eng = _mock_engine()
    result = eng.embed(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_clear.assert_called_once()


@patch("vllm_mlx.embedding.mx.clear_cache")
def test_embed_tokens_releases_buffers_after_batch(mock_clear):
    eng = _mock_engine()
    result = eng.embed_tokens([[1, 2], [3, 4]])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_clear.assert_called_once()


@patch("vllm_mlx.embedding.mx.clear_cache")
def test_clear_cache_runs_after_tolist(mock_clear):
    """The release must happen AFTER ``.tolist()`` materializes the Python
    lists — otherwise it would drop the buffers still backing the result."""
    eng = _mock_engine()
    order: list[str] = []
    eng._model.return_value.text_embeds.tolist.side_effect = lambda: (
        order.append("tolist") or [[0.1, 0.2], [0.3, 0.4]]
    )
    mock_clear.side_effect = lambda: order.append("clear")
    eng.embed(["a", "b"])
    assert order == ["tolist", "clear"]


@patch("vllm_mlx.embedding.mx.clear_cache")
def test_empty_token_batch_does_not_clear(mock_clear):
    """No forward pass runs for an empty batch, so there is nothing to
    release and clear_cache must not be called."""
    eng = _mock_engine()
    assert eng.embed_tokens([]) == []
    mock_clear.assert_not_called()
    # The early return must happen before any model call (guards against a
    # future refactor moving the guard below the forward pass).
    eng._model.assert_not_called()


@patch("vllm_mlx.embedding.mx.clear_cache")
def test_embed_tokens_ragged_batch_releases_buffers(mock_clear):
    """The leak is triggered by varied-length batches (each new padded size
    is a buffer the MLX pool can't reuse), so cover the ragged case: the
    release must still run exactly once."""
    eng = _mock_engine()
    result = eng.embed_tokens([[1, 2, 3], [4]])  # lengths differ → padded
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_clear.assert_called_once()
