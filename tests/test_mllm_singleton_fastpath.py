"""Singleton no-rebatch fast-path contract tests.

Pins the eligibility contract (handoff §6) and the terminal-extraction
detachment transaction (handoff §7) for the serialized MLLM lane:

* the rollback flag is validated at every config layer;
* one qualified request selects the regular-cache path and no leaf
  ``merge`` is called;
* ``_process_prompts`` can never return an unbatched batch for multiple
  requests — the legacy merge path runs instead;
* unknown / wrapped / subclassed leaves fail closed into ``merge``;
* ``MLLMBatch.extend`` refuses a singleton-regular batch;
* a queued second request stays queued while a batch is active;
* ``extract_cache`` on a singleton-regular batch returns detached,
  materialized regular leaves — never ``None``, never lazy views of the
  live state.
"""

import builtins

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
    _extract_detached_singleton_leaf,
    _singleton_regular_cache_leaves,
)
from rapid_mlx.mllm_scheduler import MLLMSchedulerConfig  # noqa: E402
from rapid_mlx.scheduler import SchedulerConfig  # noqa: E402

VOCAB = 8


def _make_request(uid: int = 0, **overrides) -> MLLMBatchRequest:
    fields = {
        "uid": uid,
        "request_id": f"r{uid}",
        "prompt": "hi",
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    fields.update(overrides)
    return MLLMBatchRequest(**fields)


def _arrays_leaves(n_layers: int = 2, n_state: int = 2):
    """Populated regular mlx-vlm ArraysCache leaves (one per layer)."""
    from mlx_vlm.models.cache import ArraysCache

    leaves = []
    for _ in range(n_layers):
        leaf = ArraysCache(n_state)
        leaf.cache = [mx.ones((1, 2, 3)), mx.ones((1, 2, 3)) * 2]
        leaf.offset = mx.array([5])
        leaves.append(leaf)
    return leaves


def _kv_leaves(n_layers: int = 2):
    """Populated regular mlx-vlm KVCache leaves (one per layer)."""
    from mlx_vlm.models.cache import KVCache

    leaves = []
    for _ in range(n_layers):
        leaf = KVCache()
        leaf.keys = mx.ones((1, 2, 5, 4))
        leaf.values = mx.ones((1, 2, 5, 4)) * 3
        leaf.offset = 5
        leaves.append(leaf)
    return leaves


def _lm_arrays_leaves(n_layers: int = 2, n_state: int = 2):
    """Populated regular mlx-lm ArraysCache leaves (one per layer)."""
    from mlx_lm.models.cache import ArraysCache

    leaves = []
    for _ in range(n_layers):
        leaf = ArraysCache(n_state)
        leaf.cache = [mx.ones((1, 2, 3)), mx.ones((1, 2, 3)) * 2]
        leaves.append(leaf)
    return leaves


def _lm_kv_leaves(n_layers: int = 2):
    """Populated regular mlx-lm KVCache leaves (one per layer)."""
    from mlx_lm.models.cache import KVCache

    leaves = []
    for _ in range(n_layers):
        leaf = KVCache()
        leaf.keys = mx.ones((1, 2, 5, 4))
        leaf.values = mx.ones((1, 2, 5, 4)) * 3
        leaf.offset = 5
        leaves.append(leaf)
    return leaves


def _stub_generator(leaves, singleton_fastpath: str = "auto"):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen._stream = mx.default_stream(mx.cpu)
    gen.allow_arrays_cache = True
    gen.prefill_batch_size = 1
    gen.completion_batch_size = 1
    gen.singleton_fastpath = singleton_fastpath
    gen.vision_prefill_token_budget = 8192
    gen.language_model = object()
    if callable(leaves):
        gen._lookup_exact_text_prefix = leaves
    else:
        gen._lookup_exact_text_prefix = lambda req: leaves
    gen._preprocess_request = lambda req: setattr(
        req, "input_ids", mx.zeros((1, 4), mx.uint32)
    )
    gen._run_vision_encoding = lambda req, cache: mx.zeros((1, 4, VOCAB))
    return gen


class TestConfigValidation:
    def test_generator_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="singleton_fastpath"):
            MLLMBatchGenerator(
                model=object(),
                processor=object(),
                singleton_fastpath="on",
            )

    def test_scheduler_config_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="mllm_singleton_fastpath"):
            SchedulerConfig(mllm_singleton_fastpath="on")

    def test_scheduler_config_accepts_off(self):
        assert (
            SchedulerConfig(mllm_singleton_fastpath="off").mllm_singleton_fastpath
            == "off"
        )

    def test_mllm_config_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="mllm_singleton_fastpath"):
            MLLMSchedulerConfig(mllm_singleton_fastpath="on")

    def test_arrays_cache_gate_still_requires_b1(self):
        with pytest.raises(ValueError, match="to all be 1"):
            MLLMSchedulerConfig(
                allow_arrays_cache=True,
                mllm_singleton_fastpath="auto",
                max_num_seqs=2,
            )


class TestEligibilityHelper:
    def test_qualified_leaves_pass(self):
        assert _singleton_regular_cache_leaves(_arrays_leaves(), True)
        assert _singleton_regular_cache_leaves(_kv_leaves(), True)

    def test_empty_is_not_eligible(self):
        assert not _singleton_regular_cache_leaves([], True)

    def test_non_serialized_lane_is_not_eligible(self):
        assert not _singleton_regular_cache_leaves(_arrays_leaves(), False)

    def test_wrapped_compound_leaf_fails_closed(self):
        from mlx_vlm.models.cache import CacheList

        leaves = [CacheList(*_kv_leaves(1))]
        assert not _singleton_regular_cache_leaves(leaves, True)

    def test_subclass_fails_closed(self):
        """Exact-type matching only: a subclass carries an unvalidated
        merge/extract lifecycle and must take the legacy merge path."""
        from mlx_vlm.models.cache import KVCache

        class CustomKVCache(KVCache):
            pass

        assert not _singleton_regular_cache_leaves([CustomKVCache()], True)

    def test_unknown_leaf_type_fails_closed(self):
        class NotACache:
            pass

        assert not _singleton_regular_cache_leaves([NotACache()], True)

    def test_missing_mlx_vlm_cache_module_fails_closed(self, monkeypatch):
        real_import = builtins.__import__

        def fail_cache_import(name, *args, **kwargs):
            if name == "mlx_vlm.models.cache":
                raise ImportError("mlx-vlm unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_cache_import)

        assert not _singleton_regular_cache_leaves([object()], True)

    def test_missing_optional_mlx_lm_cache_module_keeps_vlm_support(self, monkeypatch):
        real_import = builtins.__import__

        def fail_cache_import(name, *args, **kwargs):
            if name == "mlx_lm.models.cache":
                raise ImportError("mlx-lm cache unavailable")
            return real_import(name, *args, **kwargs)

        leaves = _arrays_leaves()
        monkeypatch.setattr(builtins, "__import__", fail_cache_import)

        assert _singleton_regular_cache_leaves(leaves, True)


class TestProcessPromptsPathSelection:
    def test_single_qualified_request_skips_merge(self, monkeypatch):
        from mlx_vlm.models.cache import ArraysCache

        merges = []
        monkeypatch.setattr(
            ArraysCache, "merge", classmethod(lambda cls, caches: merges.append(1))
        )

        leaves = _arrays_leaves()
        gen = _stub_generator(leaves)
        batch = gen._process_prompts([_make_request(0)])

        assert merges == []  # no leaf merge was called
        assert batch.cache_layout == "singleton_regular"
        assert batch.cache is not leaves  # generator copies the list…
        assert batch.cache[0] is leaves[0]  # …but keeps the same leaf objects

    def test_two_requests_must_take_merge_path(self, monkeypatch):
        """Direct ``_process_prompts([r1, r2])`` can never return an
        unbatched batch: the single-request invariant fails and the legacy
        merge path runs."""
        from mlx_vlm.models.cache import ArraysCache

        merged = []
        monkeypatch.setattr(
            ArraysCache,
            "merge",
            classmethod(lambda cls, caches: merged.append(len(caches)) or caches[0]),
        )

        gen = _stub_generator(lambda req: _arrays_leaves())
        batch = gen._process_prompts([_make_request(0), _make_request(1)])

        assert batch.cache_layout == "batched"
        # One merge call per cache layer, each with both requests' leaves.
        assert merged == [2, 2]

    def test_off_rolls_back_to_merge(self, monkeypatch):
        from mlx_vlm.models.cache import ArraysCache

        merges = []
        monkeypatch.setattr(
            ArraysCache,
            "merge",
            classmethod(lambda cls, caches: merges.append(len(caches)) or caches[0]),
        )

        gen = _stub_generator(_arrays_leaves(), singleton_fastpath="off")
        batch = gen._process_prompts([_make_request(0)])

        assert merges == [1, 1]
        assert batch.cache_layout == "batched"

    def test_merge_failure_is_propagated(self, monkeypatch):
        from mlx_vlm.models.cache import ArraysCache

        def fail_merge(cls, caches):
            raise RuntimeError("merge failed")

        monkeypatch.setattr(ArraysCache, "merge", classmethod(fail_merge))

        gen = _stub_generator(_arrays_leaves(), singleton_fastpath="off")
        with pytest.raises(RuntimeError, match="merge failed"):
            gen._process_prompts([_make_request(0)])

    def test_subclass_leaf_takes_merge_path(self, monkeypatch):
        from mlx_vlm.models.cache import KVCache

        class CustomKVCache(KVCache):
            pass

        merges = []
        monkeypatch.setattr(
            CustomKVCache,
            "merge",
            classmethod(lambda cls, caches: merges.append(len(caches)) or caches[0]),
        )

        gen = _stub_generator(lambda req: [CustomKVCache()])
        batch = gen._process_prompts([_make_request(0)])

        assert merges == [1]
        assert batch.cache_layout == "batched"

    def test_unknown_leaf_fails_closed_into_merge(self, monkeypatch):
        from mlx_vlm.models.cache import ArraysCache

        merges = []
        monkeypatch.setattr(
            ArraysCache, "merge", classmethod(lambda cls, caches: caches[0])
        )

        class NotACache:
            pass

        gen = _stub_generator(lambda req: [NotACache()])
        # The unknown leaf cannot even reach merge: the lane's cache-type
        # gate rejects it before the merge site. Fail-closed either way.
        with pytest.raises(ValueError, match="requires KVCache"):
            gen._process_prompts([_make_request(0)])


class TestQueuedAdmission:
    def test_second_request_stays_queued_while_batch_active(self):
        """Structural B=1 invariant: with an active batch, queued requests
        are never joined into it — the no-active-batch admission rule."""
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.unprocessed_requests = []
        gen.completion_batch_size = 1
        gen._stats = MLLMBatchStats()
        gen.uid_counter = 0

        active = _make_request(0)
        gen.active_batch = MLLMBatch(
            uids=[active.uid],
            request_ids=[active.request_id],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=[],
            requests=[active],
            cache_layout="singleton_regular",
        )
        gen.insert([_make_request(1), _make_request(2)])

        with (
            _no_prompt_processing(gen),
            pytest.raises(_StopGenerationError),
        ):
            gen._next()

        # The active batch was untouched; both newcomers still queued.
        assert gen.active_batch.uids == [active.uid]
        # ``insert`` reassigns UIDs from the generator counter (0-based),
        # so the queued newcomers carry 0 and 1 here.
        assert [r.uid for r in gen.unprocessed_requests] == [0, 1]

    def test_new_batch_admits_exactly_one_request(self):
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.unprocessed_requests = []
        gen.completion_batch_size = 1
        gen._stats = MLLMBatchStats()
        gen.uid_counter = 0
        gen.active_batch = None
        gen.insert([_make_request(0), _make_request(1)])

        admitted = []
        original = MLLMBatchGenerator._process_prompts

        def capture(self, requests):
            admitted.extend(r.uid for r in requests)
            raise _StopGenerationError()

        with _no_prompt_processing(gen):
            gen._process_prompts = capture.__get__(gen, MLLMBatchGenerator)
            with pytest.raises(_StopGenerationError):
                gen._next()

        assert admitted == [0]


class _StopGenerationError(Exception):
    pass


class _no_prompt_processing:
    """Stub out ``_step`` so ``_next`` exercises admission only."""

    def __init__(self, gen):
        self.gen = gen

    def __enter__(self):
        self.original = MLLMBatchGenerator._step

        def stop(self, *args, **kwargs):
            raise _StopGenerationError()

        MLLMBatchGenerator._step = stop
        return self

    def __exit__(self, *exc):
        MLLMBatchGenerator._step = self.original
        return False


class TestExtendRefusal:
    def test_extend_refuses_singleton_regular_batch(self):
        leaves = _arrays_leaves()
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=leaves,
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )
        other = MLLMBatch(
            uids=[1],
            request_ids=["r1"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=_arrays_leaves(),
            requests=[_make_request(1)],
        )
        with pytest.raises(ValueError, match="singleton-regular"):
            batch.extend(other)

    def test_extend_allows_batched_batches(self):
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=_kv_leaves(),
            requests=[_make_request(0)],
        )
        other = MLLMBatch(
            uids=[1],
            request_ids=["r1"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=_kv_leaves(),
            requests=[_make_request(1)],
        )
        batch.extend(other)
        assert batch.uids == [0, 1]


class TestSingletonExtractionDetachment:
    """Handoff §7: extraction must return detached, materialized state."""

    @pytest.mark.parametrize(
        "leaf_factory",
        [_arrays_leaves, _kv_leaves, _lm_arrays_leaves, _lm_kv_leaves],
    )
    def test_extract_returns_real_detached_leaves(self, leaf_factory):
        leaves = leaf_factory()
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=leaves,
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )

        extracted = batch.extract_cache(0)

        # Real regular leaves, one per layer — never None.
        assert len(extracted) == len(leaves)
        assert all(leaf is not None for leaf in extracted)
        assert all(type(src) is type(dst) for src, dst in zip(leaves, extracted))
        # Nothing aliases the live leaf objects.
        assert all(src is not dst for src, dst in zip(leaves, extracted))
        for src, dst in zip(leaves, extracted):
            src_offset = getattr(src, "offset", None)
            if src_offset is not None:
                dst_offset = getattr(dst, "offset", None)
                if hasattr(src_offset, "shape"):
                    assert mx.array_equal(dst_offset, src_offset)
                else:
                    assert dst_offset == src_offset

    @pytest.mark.parametrize(
        "leaf_factory",
        [_arrays_leaves, _kv_leaves, _lm_arrays_leaves, _lm_kv_leaves],
    )
    def test_extracted_state_survives_live_mutation(self, leaf_factory):
        leaves = leaf_factory()
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=leaves,
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )
        extracted = batch.extract_cache(0)

        # Materialize the extracted state before the live batch changes.
        flat = []
        for leaf in extracted:
            states = getattr(leaf, "cache", None)
            if states is not None:
                flat.extend(s for s in states if s is not None)
            else:
                flat.extend([leaf.keys, leaf.values])
        mx.eval(*flat)

        # Mutate every live allocation in place, then re-check the extracted
        # copy still carries the original values. Rebinding ``leaf.cache``
        # would not detect a lazy slice that still aliases the old allocation.
        for leaf in leaves:
            states = getattr(leaf, "cache", None)
            if states is not None:
                for state in states:
                    if state is not None:
                        state[:] = state + 100.0
                if getattr(leaf, "offset", None) is not None:
                    leaf.offset = leaf.offset + 100
            else:
                leaf.keys[:] = leaf.keys + 100.0
                leaf.values[:] = leaf.values + 100.0
                leaf.offset = leaf.offset + 100
        mx.eval(
            *[
                s
                for leaf in leaves
                for s in (getattr(leaf, "cache", None) or [leaf.keys, leaf.values])
                if s is not None
            ]
        )

        for src, dst in zip(leaves, extracted):
            src_states = getattr(src, "cache", None)
            if src_states is not None:
                for s_src, s_dst in zip(src_states, dst.cache):
                    if s_src is None:
                        assert s_dst is None
                    else:
                        # Live moved +100; the detached copy must still
                        # hold the original values (ones / ones*2).
                        assert not mx.array_equal(s_dst, s_src)
            else:
                assert not mx.array_equal(dst.keys, src.keys)
                assert not mx.array_equal(dst.values, src.values)

    def test_arrays_extract_is_contiguous_not_lazy_view(self):
        """The ArraysCache detachment must not return lazy slices of the
        live state (the failure mode of upstream ``ArraysCache.extract``)."""
        from mlx_vlm.models.cache import ArraysCache

        leaves = _arrays_leaves()
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=leaves,
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )
        extracted = batch.extract_cache(0)[0]

        assert type(extracted) is ArraysCache
        for live_state, detached_state in zip(leaves[0].cache, extracted.cache):
            assert detached_state is not live_state
            # mx.contiguous materialization: evaluating the copy cannot be
            # affected by later writes into the live arrays.
            assert detached_state.shape == live_state.shape

    def test_arrays_extract_copies_optional_metadata(self):
        leaf = _arrays_leaves(1)[0]
        leaf.left_padding = mx.array([1])
        leaf.lengths = mx.array([3])
        leaf.offset = 5

        detached = _extract_detached_singleton_leaf(leaf, 0)

        assert mx.array_equal(detached.left_padding, mx.array([1]))
        assert mx.array_equal(detached.lengths, mx.array([3]))
        assert detached.offset == 5

    def test_extract_without_optional_mlx_lm_cache_module(self, monkeypatch):
        real_import = builtins.__import__

        def fail_cache_import(name, *args, **kwargs):
            if name == "mlx_lm.models.cache":
                raise ImportError("mlx-lm cache unavailable")
            return real_import(name, *args, **kwargs)

        leaf = _kv_leaves(1)[0]
        monkeypatch.setattr(builtins, "__import__", fail_cache_import)

        detached = _extract_detached_singleton_leaf(leaf, 0)

        assert detached.offset == leaf.offset
        assert mx.array_equal(detached.keys, leaf.keys)

    def test_extract_rejects_an_unqualified_leaf(self):
        with pytest.raises(TypeError, match="unsupported singleton-regular"):
            _extract_detached_singleton_leaf(object(), 0)


class TestLegacySemanticsAlignment:
    """Round-1 review: the fast path must remain numerically faithful to the
    legacy merge path, and its engagement must be observable."""

    def test_adopted_leaves_drop_prefill_state_bookkeeping(self, monkeypatch):
        """Legacy ``ArraysCache.merge`` rebuilds each leaf, so its
        left_padding/lengths bookkeeping is gone (decode masks come out
        None). The fast path must reset the adopted leaves the same way —
        including on warm APC clones that still carry that state."""

        leaves = _arrays_leaves()
        for leaf in leaves:
            leaf.left_padding = mx.array([0])
            leaf.lengths = mx.array([5])

        gen = _stub_generator(leaves)
        batch = gen._process_prompts([_make_request(0)])

        assert batch.cache_layout == "singleton_regular"
        for leaf in batch.cache:
            assert getattr(leaf, "left_padding", None) is None
            assert getattr(leaf, "lengths", None) is None
        assert gen._stats.singleton_batches == 1

    def test_off_phase_never_counts_singleton_batches(self, monkeypatch):
        gen = _stub_generator(_arrays_leaves(), singleton_fastpath="off")
        gen._process_prompts([_make_request(0)])
        assert gen._stats.singleton_batches == 0

    def test_filter_rejects_non_identity_keep_idx(self):
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[mx.zeros(1)],
            max_tokens=[8],
            num_tokens=[0],
            cache=_kv_leaves(),
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )
        batch.filter([0])  # the B=1 identity filter stays legal
        with pytest.raises(ValueError, match="singleton-regular"):
            batch.filter([])

    def test_extract_trims_kv_slab_padding(self):
        """Upstream KVCache slab-allocates; the detached copy must be
        trimmed to ``offset`` exactly like ``KVCache.extract``, not carry a
        whole slab of zero padding."""
        from mlx_vlm.models.cache import KVCache

        leaf = KVCache()
        leaf.keys = mx.ones((1, 2, 8, 4))
        leaf.values = mx.ones((1, 2, 8, 4)) * 3
        leaf.offset = 3
        batch = MLLMBatch(
            uids=[0],
            request_ids=["r0"],
            y=mx.zeros((1,), dtype=mx.uint32),
            logprobs=[],
            max_tokens=[8],
            num_tokens=[0],
            cache=[leaf],
            requests=[_make_request(0)],
            cache_layout="singleton_regular",
        )

        extracted = batch.extract_cache(0)[0]

        assert type(extracted) is KVCache
        assert extracted.keys.shape == (1, 2, 3, 4)
        assert extracted.values.shape == (1, 2, 3, 4)
        assert extracted.offset == 3
