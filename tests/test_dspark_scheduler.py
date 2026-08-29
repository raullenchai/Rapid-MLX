"""DSpark scheduler rollback invariants."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from types import SimpleNamespace

import mlx.core as mx

from vllm_mlx.scheduler import (
    _adapt_dspark_depth,
    _install_dspark,
    _replay_dspark_committed,
)


def test_pooling_cache_rolls_back_across_compression_boundary() -> None:
    from vllm_mlx.models.deepseek_v4_cache import DeepseekV4PoolingCache
    from vllm_mlx.models.deepseek_v4_rollback import armed, trim_all

    cache = DeepseekV4PoolingCache(4)
    # Establish three pending values, then verify four inputs. Keeping the
    # confirmed input crosses one boundary; the rejected suffix must vanish.
    cache.accumulate_windows(mx.ones((1, 3, 2)), mx.ones((1, 3, 2)), 0)
    with armed():
        ready, _gate, _base = cache.accumulate_windows(
            mx.ones((1, 5, 2)), mx.ones((1, 5, 2)), 3
        )
        cache.update_and_fetch(mx.ones((1, ready.shape[1] // 4, 2)))

    assert trim_all([cache], 3)
    assert cache.offset == 1
    assert cache.remainder == 1


def test_pooling_cache_multitoken_preflight_is_atomic_without_undo() -> None:
    from mlx_lm.models.cache import CacheList, KVCache

    from vllm_mlx.cache_rollback import can_trim, trim_all
    from vllm_mlx.models.deepseek_v4_cache import PoolingCache

    kv = KVCache()
    values = mx.ones((1, 1, 5, 2))
    kv.update_and_fetch(values, values)
    pooling = PoolingCache(4)
    pooling.accumulate_windows(mx.ones((1, 1, 2)), mx.ones((1, 1, 2)), 0)
    cache = CacheList(kv, pooling)

    assert pooling.remainder == 1
    assert pooling._undo is None
    assert cache.is_trimmable()
    assert not can_trim(cache, 2)
    assert not trim_all([cache], 2)
    assert kv.offset == 5
    assert pooling.remainder == 1


def test_trim_transaction_restores_earlier_cache_when_later_trim_fails() -> None:
    from vllm_mlx.cache_rollback import trim_all

    class CursorCache:
        def __init__(self, offset, *, fail=False):
            self.offset = offset
            self.fail = fail

        def can_trim(self, n):
            return n <= self.offset

        def trim(self, n):
            self.offset -= n
            return 0 if self.fail else n

    first = CursorCache(7)
    later = CursorCache(9, fail=True)

    assert not trim_all([first, later], 2)
    assert first.offset == 7
    assert later.offset == 9


def test_trim_admission_guards_and_custom_checkpoint_restore() -> None:
    from vllm_mlx.cache_rollback import can_trim, trim_all

    class CustomCache:
        def __init__(self):
            self.offset = 3
            self.restored = False

        def can_trim(self, n):
            return n <= self.offset

        def trim_checkpoint(self):
            return self.offset

        def restore_trim_checkpoint(self, offset):
            self.offset = offset
            self.restored = True

        def trim(self, n):
            self.offset -= n
            raise RuntimeError("injected trim failure")

    class UndoCache:
        def _can_undo(self, n):
            return n == 2

    class LegacyCache:
        def __init__(self, size):
            self._size = size

        def is_trimmable(self):
            return True

        def size(self):
            return self._size

    class InvalidSizeCache(LegacyCache):
        def size(self):
            return object()

    assert can_trim(UndoCache(), 2)
    assert not can_trim(object(), 1)
    assert can_trim(LegacyCache(2), 2)
    assert not can_trim(InvalidSizeCache(2), 1)
    assert not can_trim(LegacyCache(2), -1)
    assert trim_all([], 0)
    assert not trim_all([], -1)

    custom = CustomCache()
    assert not trim_all([custom], 2)
    assert custom.offset == 3
    assert custom.restored


def test_pooling_checkpoint_restore_covers_scalar_and_batch_state() -> None:
    from vllm_mlx.models.deepseek_v4_cache import (
        BatchDeepseekV4PoolingCache,
        DeepseekV4PoolingCache,
    )

    scalar = DeepseekV4PoolingCache(4)
    scalar.accumulate_windows(mx.ones((1, 3, 4)), mx.ones((1, 3, 4)), 0)
    scalar_state = scalar.trim_checkpoint()
    scalar.remainder = 0
    scalar.restore_trim_checkpoint(scalar_state)
    assert scalar.remainder == 3
    assert scalar.can_trim(3)

    batch = BatchDeepseekV4PoolingCache(ratio=4, left_padding=[0])
    batch.accumulate_windows(mx.ones((1, 3, 4)), mx.ones((1, 3, 4)), [0])
    batch_state = batch.trim_checkpoint()
    batch.remainder = [0]
    batch.restore_trim_checkpoint(batch_state)
    assert batch.remainder == [3]
    assert batch.can_trim(3)


def test_batch_pooling_cache_rolls_back_across_compression_boundary() -> None:
    from vllm_mlx.models.deepseek_v4_cache import BatchDeepseekV4PoolingCache
    from vllm_mlx.models.deepseek_v4_rollback import armed, trim_all

    cache = BatchDeepseekV4PoolingCache(ratio=4, left_padding=[0])
    cache.accumulate_windows(mx.ones((1, 3, 4)), mx.ones((1, 3, 4)), 0)
    with armed():
        ready, _gate, _base = cache.accumulate_windows(
            mx.ones((1, 5, 4)), mx.ones((1, 5, 4)), 3
        )
        cache.update_and_fetch(mx.ones((1, ready.shape[1] // 4, 4)))

    assert trim_all([cache], 3)
    assert cache._pool_lengths == [1]
    assert cache.remainder == [1]
    assert cache.size() == 1


def test_deepseek_pooling_undo_without_completed_window_restores_overlap() -> None:
    from vllm_mlx.models.deepseek_v4_cache import (
        BatchDeepseekV4PoolingCache,
        DeepseekV4PoolingCache,
    )
    from vllm_mlx.models.deepseek_v4_rollback import armed, trim_all

    scalar = DeepseekV4PoolingCache(4)
    scalar.accumulate_windows(mx.ones((1, 3, 4)), mx.ones((1, 3, 4)), 0)
    with armed():
        scalar.accumulate_windows(mx.ones((1, 4, 4)), mx.ones((1, 4, 4)), 3)
    assert trim_all([scalar], 4)
    assert scalar.remainder == 3

    batch = BatchDeepseekV4PoolingCache(ratio=4, left_padding=[0])
    batch.accumulate_windows(mx.ones((1, 3, 4)), mx.ones((1, 3, 4)), 0)
    with armed():
        batch.accumulate_windows(mx.ones((1, 4, 4)), mx.ones((1, 4, 4)), 3)
    assert trim_all([batch], 4)
    assert batch.remainder == [3]


def test_rotating_cache_rolls_back_after_rotation() -> None:
    from mlx_lm.models.cache import RotatingKVCache

    from vllm_mlx.models.deepseek_v4_rollback import (
        armed,
        install_rotating_undo,
        trim_all,
    )

    install_rotating_undo()
    directly_trimmable = RotatingKVCache(max_size=4)
    directly_trimmable.update_and_fetch(mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2)))
    assert directly_trimmable.can_trim(1)
    reference = RotatingKVCache(max_size=4)
    reference.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    reference.update_and_fetch(mx.ones((1, 1, 1, 2)), mx.ones((1, 1, 1, 2)))
    cache = RotatingKVCache(max_size=4)
    cache.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    with armed():
        cache.update_and_fetch(mx.ones((1, 1, 4, 2)), mx.ones((1, 1, 4, 2)))

    assert not cache.can_trim(-1)
    assert cache.can_trim(1)
    checkpoint = cache.trim_checkpoint()
    cache.restore_trim_checkpoint(checkpoint)
    assert trim_all([cache], 3)
    assert (cache.offset, cache._idx) == (reference.offset, reference._idx)
    assert mx.array_equal(cache.keys, reference.keys).item()
    assert mx.array_equal(cache.values, reference.values).item()
    assert mx.array_equal(
        cache._temporal_order(cache.keys), reference._temporal_order(reference.keys)
    ).item()


def test_batch_rotating_cache_rolls_back_after_rotation() -> None:
    from mlx_lm.models.cache import BatchRotatingKVCache

    from vllm_mlx.models.deepseek_v4_rollback import (
        armed,
        install_rotating_undo,
        trim_all,
    )

    install_rotating_undo()
    reference = BatchRotatingKVCache(max_size=4, left_padding=[0])
    reference.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    reference.update_and_fetch(mx.ones((1, 1, 1, 2)), mx.ones((1, 1, 1, 2)))
    cache = BatchRotatingKVCache(max_size=4, left_padding=[0])
    cache.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    with armed():
        cache.update_and_fetch(mx.ones((1, 1, 4, 2)), mx.ones((1, 1, 4, 2)))

    assert trim_all([cache], 3)
    assert (cache._offset, cache._idx, cache.rotated) == (
        reference._offset,
        reference._idx,
        reference.rotated,
    )
    assert mx.array_equal(cache.offset, reference.offset).item()
    assert mx.array_equal(cache.left_padding, reference.left_padding).item()
    assert mx.array_equal(cache.keys, reference.keys).item()
    assert mx.array_equal(cache.values, reference.values).item()


def test_rotating_undo_supports_legacy_cache_without_amount_preflight(
    monkeypatch,
) -> None:
    from mlx_lm.models import cache as mlx_cache

    from vllm_mlx.models import deepseek_v4_rollback as rollback

    class LegacyRotating:
        def __init__(self):
            self.offset = 3

        def update_and_fetch(self, keys, values):
            return keys, values

        def is_trimmable(self):
            return True

        def trim(self, n):
            self.offset -= n
            return n

    class LegacyBatchRotating:
        def __init__(self):
            self.offset = 3

        def update_and_fetch(self, keys, values):
            return keys, values

        def is_trimmable(self):
            return True

        def trim(self, n):
            self.offset -= n
            return n

        def can_trim(self, n):
            return n <= self.offset

    monkeypatch.setattr(mlx_cache, "RotatingKVCache", LegacyRotating)
    monkeypatch.setattr(mlx_cache, "BatchRotatingKVCache", LegacyBatchRotating)
    rollback.install_rotating_undo()
    assert LegacyRotating().can_trim(2)
    assert LegacyBatchRotating().can_trim(2)


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


def test_prompt_capture_does_not_fabricate_dspark_history_after_prefix_hit() -> None:
    from vllm_mlx.models.deepseek_v4 import Model

    fake = SimpleNamespace(
        _dspark_prime_ctx=None,
        _target_cache_offset=lambda _cache: 100,
        make_dspark_cache=lambda: (_ for _ in ()).throw(
            AssertionError("must not create a logically offset empty cache")
        ),
    )

    Model._capture_dspark_prompt(
        fake,
        mx.ones((1, 4), dtype=mx.uint32),
        mx.zeros((1, 4, 8)),
        [object()],
    )

    assert fake._dspark_prime_ctx is None


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


def test_stochastic_request_uses_plain_decode_until_multiround_gate_exists() -> None:
    class _Model:
        args = SimpleNamespace(dspark_block_size=5)
        mtp = [object()]
        _last_dspark_hidden = mx.zeros((1, 1, 2))

        def make_dspark_cache(self):
            return []

        def dspark_forward(self, *_args, **_kwargs):
            raise AssertionError("stochastic request must not enter DSpark")

    class _GenerationBatch:
        pass

    gb = _GenerationBatch()
    gb._next_tokens = mx.array([1])
    gb.uids = [7]
    gb.next = lambda: []
    baseline_calls = []

    def baseline_step():
        baseline_calls.append(True)
        return [9], [mx.zeros((8,))]

    gb._step = baseline_step
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=1.0, seed=None)
    )
    batch_gen = SimpleNamespace(_generation_batch=gb)

    assert _install_dspark(
        batch_gen,
        _Model(),
        {"request-7": request},
        {7: "request-7"},
        max_draft=5,
    )
    tokens, _logprobs = gb._step()

    assert tokens == [9]
    assert baseline_calls == [True]
    assert batch_gen._dspark_stats["fallthrough_steps"] == 1
