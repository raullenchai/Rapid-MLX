"""DSpark scheduler rollback invariants."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from vllm_mlx.scheduler import (
    _adapt_dspark_depth,
    _install_dspark,
    _replay_dspark_committed,
)


def test_adaptive_depth_shrinks_cools_down_and_recovers() -> None:
    depth, streak, cooldown = _adapt_dspark_depth(5, 5, 1, 5, 0)
    assert (depth, streak, cooldown) == (4, 1, False)
    depth, streak, cooldown = _adapt_dspark_depth(depth, 5, 0, 4, streak)
    assert (depth, streak, cooldown) == (3, 2, False)
    depth, streak, cooldown = _adapt_dspark_depth(depth, 5, 1, 3, streak)
    assert (depth, streak, cooldown) == (1, 0, True)
    # The post-cooldown K=1 probe can grow again after a full accept.
    assert _adapt_dspark_depth(1, 5, 1, 1, 0) == (2, 0, False)


class _Cache:
    def __init__(self, values: list[int] | None = None):
        self.values = list(values or [])
        self.offset = len(self.values)

    def is_trimmable(self) -> bool:
        return False

    def extract(self, _idx: int):
        return _Cache(self.values)


def test_replay_dspark_committed_excludes_unsurfaced_drafts() -> None:
    class _Model:
        _last_dspark_hidden = mx.zeros((1, 1, 2))

        def __call__(self, tokens, *, cache):
            cache[0].values.extend(int(token) for token in tokens[0].tolist())
            cache[0].offset = len(cache[0].values)
            self._last_dspark_hidden = mx.zeros((1, tokens.shape[1], 2))
            return mx.zeros((1, tokens.shape[1], 8))

    cache = [_Cache([10])]
    restored = _replay_dspark_committed(
        _Model(), cache, mx.array([[11, 12, 13]]), token_count=1
    )

    assert restored[0].values == [10, 11]


def test_verify_failure_restores_target_cache_before_baseline_fallback() -> None:
    class _DraftCache:
        offset = 0

        def is_trimmable(self) -> bool:
            return False

    class _Model:
        args = SimpleNamespace(dspark_block_size=5)
        mtp = [object()]
        _last_dspark_hidden = mx.zeros((1, 1, 2))

        def make_dspark_cache(self):
            return [_DraftCache()]

        def dspark_forward(self, _tokens, _hidden, cache):
            cache[0].offset = 5
            return mx.array([[1, 2, 3, 4, 5, 6]]), None

        def __call__(self, _tokens, *, cache):
            cache[0].values.append(999)
            cache[0].offset += 1
            raise RuntimeError("synthetic verify failure")

    class _GenerationBatch:
        pass

    gb = _GenerationBatch()
    gb._next_tokens = mx.array([1])
    gb._next_logprobs = [mx.zeros((8,))]
    gb.uids = [7]
    gb.tokens = [[]]
    gb.prompt_cache = [_Cache([10])]
    gb.logits_processors = [[]]
    gb.next = lambda: []

    def baseline_step():
        assert gb.prompt_cache[0].values == [10]
        return [1], [mx.zeros((8,))]

    gb._step = baseline_step
    batch_gen = SimpleNamespace(_generation_batch=gb)

    assert _install_dspark(batch_gen, _Model(), {}, {}, max_draft=5)
    tokens, _logprobs = gb._step()

    assert tokens == [1]
    assert batch_gen._dspark_stats["errors"] == 1
