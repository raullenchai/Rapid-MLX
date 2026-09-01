"""CPU/mock contracts for the Rapid continuous self-MTP data plane."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vllm_mlx.spec_decode.mtp import mlx_backend as backend_module
from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPCapabilities,
    ContinuousSelfMTPConfig,
    ContinuousSelfMTPRuntime,
    RapidForwardSeams,
    SelfMTPCachePair,
    SelfMTPLaneSpec,
    SelfMTPSampling,
    abort_batched_self_mtp,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPUnsupportedError as ContinuousSelfMTPUnsupported,
)
from vllm_mlx.spec_decode.mtp.mlx_backend import (
    RapidMLXSelfMTPBackend,
    RapidRaggedCacheAdapter,
)


class _NumpyOps:
    @staticmethod
    def uint32(value):
        return np.asarray(value, dtype=np.uint32)

    @staticmethod
    def concatenate(values, *, axis):
        return np.concatenate(list(values), axis=axis)

    @staticmethod
    def pad(value, widths):
        return np.pad(value, widths)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis)

    @staticmethod
    def logprobs(logits):
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    @staticmethod
    def argmax_int(logprobs):
        return int(np.argmax(logprobs, axis=-1))


class _LayerCache:
    def __init__(self, label, rows=None):
        self.label = label
        self.rows = list([label] if rows is None else rows)
        self.events = []

    @classmethod
    def merge(cls, caches):
        merged = cls("merged", [])
        for cache in caches:
            merged.rows.extend(cache.rows)
        return merged

    def extend(self, other):
        self.events.append(("extend", tuple(other.rows)))
        self.rows.extend(other.rows)

    def prepare(self, *, lengths, right_padding):
        self.events.append(("prepare", tuple(lengths), tuple(right_padding)))

    def finalize(self):
        self.events.append(("finalize",))

    def start_speculation(self):
        self.events.append(("start_speculation",))

    def stop_speculation(self):
        self.events.append(("stop_speculation",))

    def extract(self, index):
        self.events.append(("extract", index))
        return type(self)(f"{self.label}:{index}", [self.rows[index]])

    def filter(self, indices):
        self.events.append(("filter", tuple(indices)))
        self.rows = [self.rows[index] for index in indices]


class _FakeForwards:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _logits(tokens, chosen):
        batch, width = tokens.shape
        logits = np.full((batch, width, 32), -20.0)
        for row in range(batch):
            for position in range(width):
                logits[row, position, chosen(row, position)] = 20.0
        return logits

    def target(self, inputs, *, cache, return_hidden, n_confirmed):
        tokens = np.asarray(inputs)
        self.calls.append(("target", tokens.copy(), cache, return_hidden, n_confirmed))
        hidden = np.repeat(tokens[..., None].astype(float), 4, axis=-1)
        if tokens.shape[1] == 3:
            # Row 0 accepts d1 then rejects d2; row 1 accepts both.
            selected = (
                [int(tokens[0, 1]), 19, 20]
                if tokens.shape[0] == 1
                else [
                    [int(tokens[0, 1]), 19, 20],
                    [int(tokens[1, 1]), int(tokens[1, 2]), 10],
                ]
            )
            if tokens.shape[0] == 1:
                logits = self._logits(tokens, lambda _r, p: selected[p])
            else:
                logits = self._logits(tokens, lambda r, p: selected[r][p])
        else:
            logits = self._logits(tokens, lambda r, p: (int(tokens[r, p]) + 1) % 32)
        return logits, hidden

    def draft(self, hidden, token_ids, cache, *, return_hidden):
        tokens = np.asarray(token_ids)
        self.calls.append(
            ("draft", np.asarray(hidden).copy(), tokens.copy(), cache, return_hidden)
        )
        logits = self._logits(tokens, lambda r, p: (int(tokens[r, p]) + 1) % 32)
        post = np.repeat(tokens[..., None].astype(float) + 0.5, 4, axis=-1)
        return logits, post


def _runtime():
    forward = _FakeForwards()
    backend = RapidMLXSelfMTPBackend(
        target_cache_factory=lambda: [_LayerCache("target")],
        draft_cache_factory=lambda: [_LayerCache("draft")],
        array_ops=_NumpyOps(),
        prefill_step_size=8,
    )
    cache_events = []

    def preflight(group, drops, **kwargs):
        cache_events.append(("preflight", tuple(drops), kwargs))

    def trim(group, drops, **kwargs):
        cache_events.append(("trim", tuple(drops), kwargs))

    cache_adapter = RapidRaggedCacheAdapter(preflight=preflight, trim=trim)
    runtime = ContinuousSelfMTPRuntime(
        config=ContinuousSelfMTPConfig(enabled=True),
        capabilities=ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
        ),
        forwards=RapidForwardSeams(forward.target, forward.draft),
        compute=backend,
        caches=cache_adapter,
    )
    return runtime, forward, cache_events


def _prepare(runtime, uid, prompt):
    return prepare_self_mtp_lane(
        SelfMTPLaneSpec(
            uid=uid,
            prompt=prompt,
            max_tokens=12,
            num_draft=2,
        ),
        runtime,
    )


def test_k2_recursive_draft_target_verify_and_delivery_commit():
    runtime, forward, cache_events = _runtime()
    detached0, first0 = _prepare(runtime, 0, [1, 2])
    detached1, first1 = _prepare(runtime, 1, [5, 6])
    assert (first0.token, first1.token) == (3, 7)
    batch = attach_self_mtp_lanes(None, [detached0, detached1])

    proposal = propose_batched_self_mtp(batch)
    assert proposal.draft_depths == (2, 2)
    assert proposal.accepted_lengths == (1, 2)
    assert [[token.token for token in row] for row in proposal.outputs] == [
        [4, 19],
        [8, 9, 10],
    ]
    verify = [call for call in forward.calls if call[0] == "target"][-1]
    assert verify[1].tolist() == [[3, 4, 5], [7, 8, 9]]
    assert verify[-1] == 2  # Rapid n_confirmed ABI for a K=2 verify.
    assert not [event for event in cache_events if event[0] == "trim"]

    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[2, 3],
        terminal=[False, False],
    )
    assert [lane.cur for lane in batch.lanes] == [19, 10]
    assert [lane.pending_tokens for lane in batch.lanes] == [[3, 4], [7, 8, 9]]
    assert [lane.ntoks for lane in batch.lanes] == [3, 4]
    assert ("trim", (1, 0), {"verify_size": 3, "validate": False}) in cache_events
    assert ("trim", (2, 2), {"verify_size": 3, "validate": False}) in cache_events


def test_abort_restores_lane_and_cache_proposal_boundary():
    runtime, _forward, _cache_events = _runtime()
    detached0, _ = _prepare(runtime, 0, [1, 2])
    detached1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [detached0, detached1])
    target_cache = batch.caches.target[0]
    draft_cache = batch.caches.draft[0]
    target_cache.offset = 7
    draft_cache.offset = 9
    lane_boundaries = [
        (
            lane.cur,
            lane.seed_hidden,
            lane.token_prefix.copy(),
            lane.ntoks,
            lane.pending_hidden,
            list(lane.pending_tokens),
        )
        for lane in batch.lanes
    ]

    proposal = propose_batched_self_mtp(batch)
    # Simulate mutations made after verification but before delivery commits.
    batch.lanes[0].cur = 31
    batch.lanes[0].pending_tokens = [99]
    target_cache.offset = 70
    draft_cache.offset = 90

    abort_batched_self_mtp(batch, proposal)

    assert batch.proposal_open is False
    assert target_cache.offset == 7
    assert draft_cache.offset == 9
    for lane, boundary in zip(batch.lanes, lane_boundaries):
        cur, seed_hidden, token_prefix, ntoks, pending_hidden, pending_tokens = boundary
        assert lane.cur == cur
        assert lane.seed_hidden is seed_hidden
        assert np.array_equal(lane.token_prefix, token_prefix)
        assert lane.ntoks == ntoks
        assert lane.pending_hidden is pending_hidden
        assert lane.pending_tokens == pending_tokens


def test_next_cycle_flushes_persistent_pending_pairs_before_new_drafts():
    runtime, forward, _events = _runtime()
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [lane0, lane1])
    first = propose_batched_self_mtp(batch)
    commit_batched_self_mtp(
        batch, first, emitted_counts=[2, 3], terminal=[False, False]
    )

    draft_calls_before = len([call for call in forward.calls if call[0] == "draft"])
    propose_batched_self_mtp(batch)
    draft_calls = [call for call in forward.calls if call[0] == "draft"]
    first_cycle_call = draft_calls[draft_calls_before]
    # Pending [old cur + accepted drafts] plus current bonus are flushed in one
    # recursive first-head job; rows are padded to the longest valid length.
    assert first_cycle_call[2].shape == (2, 4)
    assert first_cycle_call[2][0].tolist() == [3, 4, 19, 0]
    assert first_cycle_call[2][1].tolist() == [7, 8, 9, 10]


def test_terminal_delivery_prefix_updates_cur_seed_and_detach_flushes_debt():
    runtime, forward, _events = _runtime()
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [lane0, lane1])
    proposal = propose_batched_self_mtp(batch)
    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[1, 2],
        terminal=[True, True],
    )
    assert [lane.cur for lane in batch.lanes] == [4, 9]
    assert [lane.pending_tokens for lane in batch.lanes] == [[3], [7, 8]]

    draft_count = len([call for call in forward.calls if call[0] == "draft"])
    batch, detached = detach_self_mtp_lanes(batch, [0, 1])
    assert batch.lanes == []
    assert [item.lane.uid for item in detached] == [0, 1]
    flushes = [call for call in forward.calls if call[0] == "draft"][draft_count:]
    assert [call[2].tolist() for call in flushes] == [[[3]], [[7, 8]]]
    assert all(item.lane.pending_tokens == [] for item in detached)


def test_one_token_remaining_uses_target_only_cycle():
    runtime, forward, cache_events = _runtime()
    detached, first = prepare_self_mtp_lane(
        SelfMTPLaneSpec(uid=0, prompt=[1, 2], max_tokens=2, num_draft=2),
        runtime,
    )
    batch = attach_self_mtp_lanes(None, [detached])

    proposal = propose_batched_self_mtp(batch)
    assert proposal.draft_depths == (0,)
    assert proposal.accepted_lengths == (0,)
    assert len(proposal.outputs[0]) == 1
    assert proposal.outputs[0][0].from_draft is False
    assert not [event for event in cache_events if event[0] == "trim"]

    commit_batched_self_mtp(batch, proposal, emitted_counts=[1], terminal=[True])
    assert batch.lanes[0].ntoks == 2
    assert batch.lanes[0].cur == proposal.outputs[0][0].token
    assert first.from_draft is False
    assert [call[-1] for call in forward.calls if call[0] == "target"][-1] == 0


def test_cohort_uses_uniform_depth_for_scalar_confirmed_boundary():
    runtime, forward, _cache_events = _runtime()
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    lane1.lane.max_tokens = lane1.lane.ntoks + 2  # Room for only K=1.
    batch = attach_self_mtp_lanes(None, [lane0, lane1])

    proposal = propose_batched_self_mtp(batch)

    assert proposal.draft_depths == (1, 1)
    verify = [call for call in forward.calls if call[0] == "target"][-1]
    assert verify[1].shape == (2, 2)
    assert verify[-1] == 1
    abort_batched_self_mtp(batch, proposal)


def test_transformed_sampling_fails_closed_without_residual_hooks():
    runtime, _forward, _events = _runtime()
    runtime = ContinuousSelfMTPRuntime(
        config=runtime.config,
        capabilities=ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
            transformed_sampling=True,
        ),
        forwards=runtime.forwards,
        compute=runtime.compute,
        caches=runtime.caches,
    )
    with pytest.raises(ContinuousSelfMTPUnsupported, match="per-lane RNG"):
        prepare_self_mtp_lane(
            SelfMTPLaneSpec(
                uid=0,
                prompt=[1, 2],
                max_tokens=8,
                num_draft=2,
                sampling=SelfMTPSampling(temperature=0.8),
            ),
            runtime,
        )


def test_cache_adapter_merge_extend_extract_filter_and_atomic_preflight():
    events = []
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda group, drops, **kwargs: events.append(
            ("preflight", tuple(drops), kwargs)
        ),
        trim=lambda group, drops, **kwargs: events.append(
            ("trim", tuple(drops), kwargs)
        ),
    )
    one = SelfMTPCachePair([_LayerCache("t1")], [_LayerCache("d1")])
    two = SelfMTPCachePair([_LayerCache("t2")], [_LayerCache("d2")])
    merged = adapter.attach(None, [one, two])
    assert merged.target[0].rows == ["t1", "t2"]
    assert merged.draft[0].rows == ["d1", "d2"]

    adapter.rollback(
        merged,
        target_drops=[1, 0],
        draft_drops=[2, 2],
        verify_width=3,
    )
    assert [event[0] for event in events] == [
        "preflight",
        "preflight",
        "trim",
        "trim",
    ]
    remaining, detached = adapter.detach(merged, [1], [0])
    assert remaining.target[0].rows == ["t1"]
    assert detached[0].target[0].rows == ["t2"]

    third = SelfMTPCachePair([_LayerCache("t3")], [_LayerCache("d3")])
    adapter.attach(remaining, [third])
    assert remaining.target[0].rows == ["t1", "t3"]


@pytest.mark.parametrize("name", ["QuantizedKVCache", "SinkWindowKVCache"])
def test_cache_adapter_rejects_quantized_and_windowed_classes(name):
    unsupported = type(name, (_LayerCache,), {})
    pair = SelfMTPCachePair([unsupported("target")], [_LayerCache("draft")])
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *a, **k: None, trim=lambda *a, **k: None
    )
    with pytest.raises(ContinuousSelfMTPUnsupported, match="unsupported"):
        adapter.attach(None, [pair])


def test_backend_has_no_eager_mlx_import_and_uses_only_rapid_forward_seams():
    source = (
        Path(__file__).parents[1]
        / "vllm_mlx"
        / "spec_decode"
        / "mtp"
        / "mlx_backend.py"
    ).read_text()
    tree = ast.parse(source)
    eager_mlx = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name.startswith("mlx") for alias in getattr(node, "names", []))
            or getattr(node, "module", "") == "mlx.core"
        )
    ]
    assert eager_mlx == []
    assert "forwards.target(" in source
    assert "forwards.draft(" in source


def test_lazy_mlx_array_adapter_delegates_to_the_installed_runtime(monkeypatch):
    fake = ModuleType("mlx.core")
    fake.uint32 = np.uint32
    fake.array = lambda value, dtype: np.asarray(value, dtype=dtype)
    fake.concatenate = lambda values, axis: np.concatenate(values, axis=axis)
    fake.pad = lambda value, widths: np.pad(value, widths)
    fake.expand_dims = lambda value, axis: np.expand_dims(value, axis)
    fake.logsumexp = lambda value, axis, keepdims: np.log(
        np.exp(value).sum(axis=axis, keepdims=keepdims)
    )
    fake.argmax = lambda value, axis: np.asarray(np.argmax(value, axis=axis))
    parent = ModuleType("mlx")
    parent.__path__ = []
    parent.core = fake
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)

    ops = backend_module._MLXArrayOps()
    values = ops.uint32([1, 2])
    assert values.dtype == np.uint32
    assert ops.concatenate([values, values], axis=0).tolist() == [1, 2, 1, 2]
    assert ops.pad(values, [(1, 0)]).tolist() == [0, 1, 2]
    assert ops.expand_dims(values, 0).shape == (1, 2)
    lps = ops.logprobs(np.asarray([0.0, 1.0]))
    assert np.exp(lps).sum() == pytest.approx(1.0)
    assert ops.argmax_int(lps) == 1


def test_cache_boundary_restores_nested_state_and_rejects_row_motion():
    child = SimpleNamespace(cache=["a"], offset=1)
    parent = SimpleNamespace(caches=[child], left_padding=np.asarray([0]), offset=2)
    boundary = backend_module._cache_boundary(parent)
    child.cache.append("b")
    child.offset = 9
    child.keys = "transient"
    parent.offset = 7
    backend_module._restore_cache_boundary(boundary)
    assert child.cache == ["a"]
    assert child.offset == 1
    assert not hasattr(child, "keys")
    assert parent.offset == 2

    moved = backend_module._cache_boundary(parent)
    parent.left_padding = np.asarray([1])
    with pytest.raises(ContinuousSelfMTPUnsupported, match="rows moved"):
        backend_module._restore_cache_boundary(moved)

    assert backend_module._plain_vector(None) is None
    assert backend_module._plain_vector(np.asarray([1, 2])) == [1, 2]
    assert backend_module._plain_vector(3) == 3


def test_cache_step_helpers_finalize_every_prepared_child_on_failure():
    class _FailPrepare(_LayerCache):
        def prepare(self, *, lengths, right_padding):
            raise RuntimeError("prepare boom")

    good = _LayerCache("good")
    bad = _FailPrepare("bad")
    composite = SimpleNamespace(caches=[good, bad])
    with pytest.raises(RuntimeError, match="prepare boom"):
        backend_module._prepare_cache(composite, [1], [0])
    assert ("finalize",) in good.events

    with pytest.raises(ContinuousSelfMTPUnsupported, match="no prepare"):
        backend_module._prepare_cache(object(), [1], [0])
    with pytest.raises(ContinuousSelfMTPUnsupported, match="no finalize"):
        backend_module._finalize_cache(object())

    native = SimpleNamespace(events=[])
    native.prepare_self_mtp_step = lambda **kwargs: native.events.append(
        ("prepare", kwargs)
    )
    native.finalize_self_mtp_step = lambda: native.events.append(("finalize",))
    backend_module._prepare_cache(native, [1], [0])
    backend_module._finalize_cache(native)
    assert native.events[-1] == ("finalize",)

    nested_good = _LayerCache("nested-good")
    nested = SimpleNamespace(caches=[nested_good])
    backend_module._prepare_cache(nested, [1], [0])
    backend_module._finalize_cache(nested)
    assert nested_good.events[-1] == ("finalize",)

    prepared = _LayerCache("prepared")
    with pytest.raises(RuntimeError, match="prepare boom"):
        backend_module._prepare_group([prepared, bad], [1])
    assert prepared.events[-1] == ("finalize",)

    class _FailFinalize(_LayerCache):
        def finalize(self):
            self.events.append(("finalize",))
            raise RuntimeError("finalize boom")

    failing = _FailFinalize("failing")
    trailing = _LayerCache("trailing")
    with pytest.raises(RuntimeError, match="finalize boom"):
        backend_module._finalize_group([failing, trailing])
    assert ("finalize",) in trailing.events

    nested_failure = SimpleNamespace(caches=[failing, trailing])
    with pytest.raises(RuntimeError, match="finalize boom"):
        backend_module._finalize_cache(nested_failure)


def test_backend_validation_surfaces_fail_closed_before_model_work():
    with pytest.raises(ValueError, match="prefill_step_size"):
        RapidMLXSelfMTPBackend(array_ops=_NumpyOps(), prefill_step_size=0)
    with pytest.raises(ValueError, match="K=2"):
        RapidMLXSelfMTPBackend(array_ops=_NumpyOps(), draft_depth=1)

    backend = RapidMLXSelfMTPBackend(array_ops=_NumpyOps())
    with pytest.raises(ContinuousSelfMTPUnsupported, match="explicit factory"):
        backend._cache(None, None, "target")
    with pytest.raises(ValueError, match="non-empty"):
        backend_module._as_group([], "target")
    with pytest.raises(ContinuousSelfMTPUnsupported, match="must return"):
        backend._forward_pair("not-a-pair", "target forward")

    runtime, _forward, _events = _runtime()
    lane, _ = _prepare(runtime, 1, [1, 2])
    assert backend._prefix(lane.lane, []).tolist() == [1, 2]
    lane.lane.sampling = SelfMTPSampling(temperature=0.5)
    with pytest.raises(ContinuousSelfMTPUnsupported, match="greedy sampling only"):
        backend._distribution(lane.lane, lane.lane.token_prefix, np.zeros(4))

    lane.lane.sampling = SelfMTPSampling(has_logits_processors=True)
    with pytest.raises(ContinuousSelfMTPUnsupported, match="injected hook"):
        backend._apply_processor(lane.lane, lane.lane.token_prefix, np.zeros(4))
    processed = RapidMLXSelfMTPBackend(
        array_ops=_NumpyOps(),
        logits_processor=lambda _lane, _prefix, logits: logits + 1,
    )
    assert np.array_equal(
        processed._apply_processor(lane.lane, lane.lane.token_prefix, np.zeros(4)),
        np.ones(4),
    )


def test_long_prefill_threads_previous_hidden_into_recursive_draft():
    runtime, forward, _events = _runtime()
    runtime.compute.prefill_step_size = 2
    prepared, first = prepare_self_mtp_lane(
        SelfMTPLaneSpec(uid=9, prompt=list(range(1, 8)), max_tokens=16, num_draft=2),
        runtime,
    )
    assert first.from_draft is False
    assert prepared.lane.token_prefix.tolist() == list(range(1, 8))
    draft_calls = [call for call in forward.calls if call[0] == "draft"]
    assert len(draft_calls) >= 3


@pytest.mark.parametrize(
    ("sampling", "num_draft", "message"),
    [
        (SelfMTPSampling(), 1, "requires K=2"),
        (SelfMTPSampling(uses_xtc=True), 2, "XTC"),
        (SelfMTPSampling(temperature=0.5), 2, "greedy sampling only"),
        (SelfMTPSampling(has_logits_processors=True), 2, "injected hook"),
    ],
)
def test_prepare_refuses_unattested_sampling_and_depth(sampling, num_draft, message):
    runtime, _forward, _events = _runtime()
    spec = SelfMTPLaneSpec(
        uid=1,
        prompt=[1, 2],
        max_tokens=8,
        num_draft=num_draft,
        sampling=sampling,
    )
    with pytest.raises(ContinuousSelfMTPUnsupported, match=message):
        runtime.compute.prepare(spec, runtime.forwards)


def test_prepare_rejects_empty_prompt_and_bad_forward_shape():
    runtime, _forward, _events = _runtime()
    empty = SelfMTPLaneSpec(uid=1, prompt=[], max_tokens=8, num_draft=2)
    with pytest.raises(ValueError, match="non-empty rank-1"):
        runtime.compute.prepare(empty, runtime.forwards)

    bad_forwards = RapidForwardSeams(
        lambda *_args, **_kwargs: "bad",
        lambda *_args, **_kwargs: "bad",
    )
    valid = SelfMTPLaneSpec(uid=1, prompt=[1, 2], max_tokens=8, num_draft=2)
    with pytest.raises(ContinuousSelfMTPUnsupported, match="must return"):
        runtime.compute.prepare(valid, bad_forwards)


def test_proposal_boundary_and_pending_pair_corruption_fail_closed():
    runtime, _forward, _events = _runtime()
    detached, _ = _prepare(runtime, 0, [1, 2])
    batch = attach_self_mtp_lanes(None, [detached])
    backend = runtime.compute

    with pytest.raises(ValueError, match="empty batch"):
        backend.propose([], batch.caches, runtime.forwards)

    proposal = propose_batched_self_mtp(batch)
    with pytest.raises(RuntimeError, match="already open"):
        backend.propose(batch.lanes, batch.caches, runtime.forwards)
    abort_batched_self_mtp(batch, proposal)

    batch.lanes[0].pending_hidden = np.zeros((1, 1, 4))
    batch.lanes[0].pending_tokens = []
    with pytest.raises(RuntimeError, match="hidden has no pending tokens"):
        backend.propose(batch.lanes, batch.caches, runtime.forwards)
    backend.abort(batch.lanes, batch.caches, None, None)

    batch.lanes[0].pending_hidden = None
    batch.lanes[0].pending_tokens = [3]
    with pytest.raises(RuntimeError, match="tokens have no pending hidden"):
        backend.propose(batch.lanes, batch.caches, runtime.forwards)
    backend.abort(batch.lanes, batch.caches, None, None)


def test_commit_abort_and_detach_reject_foreign_or_missing_state():
    runtime, _forward, _events = _runtime()
    detached, _ = _prepare(runtime, 0, [1, 2])
    batch = attach_self_mtp_lanes(None, [detached])
    backend = runtime.compute

    foreign = SimpleNamespace(payload=object())
    with pytest.raises(TypeError, match="foreign cycle payload"):
        backend.commit(batch.lanes, foreign, emitted_counts=(1,), terminal=(False,))
    with pytest.raises(RuntimeError, match="no matching proposal boundary"):
        backend.abort(batch.lanes, batch.caches, None, None)

    proposal = propose_batched_self_mtp(batch)
    with pytest.raises(RuntimeError, match="membership changed"):
        backend.abort([], batch.caches, proposal._computation, None)

    batch.lanes[0].pending_tokens = []
    backend.detach_lane(batch.lanes[0], batch.caches)
    batch.lanes[0].pending_tokens = [1]
    batch.lanes[0].pending_hidden = None
    with pytest.raises(RuntimeError, match="no pending hidden"):
        backend.detach_lane(batch.lanes[0], batch.caches)
    batch.lanes[0].pending_hidden = np.zeros((1, 1, 4))
    batch.lanes[0].backend_state = None
    with pytest.raises(ContinuousSelfMTPUnsupported, match="no Rapid forward seam"):
        backend.detach_lane(batch.lanes[0], batch.caches)


def test_commit_merges_existing_pending_pairs_and_rejects_double_commit():
    runtime, _forward, _events = _runtime()
    detached, _ = _prepare(runtime, 0, [1, 2])
    batch = attach_self_mtp_lanes(None, [detached])
    computation = runtime.compute.propose(batch.lanes, batch.caches, runtime.forwards)
    lane = batch.lanes[0]
    lane.pending_tokens = [99]
    lane.pending_hidden = np.zeros((1, 1, 4))
    runtime.compute.commit(
        batch.lanes,
        computation,
        emitted_counts=(len(computation.outputs[0]),),
        terminal=(False,),
    )
    assert lane.pending_tokens[0] == 99
    with pytest.raises(RuntimeError, match="no matching proposal boundary"):
        runtime.compute.commit(
            batch.lanes,
            computation,
            emitted_counts=(len(computation.outputs[0]),),
            terminal=(False,),
        )


def test_commit_rejects_pending_tokens_without_their_hidden_pair():
    runtime, _forward, _events = _runtime()
    detached, _ = _prepare(runtime, 0, [1, 2])
    batch = attach_self_mtp_lanes(None, [detached])
    computation = runtime.compute.propose(batch.lanes, batch.caches, runtime.forwards)
    lane = batch.lanes[0]

    # Corruption can occur after proposal but before commit if an outer
    # scheduler mutates lane state. Commit must reject the split pair before
    # it publishes any accepted token or advances the proposal boundary.
    lane.pending_tokens = [99]
    lane.pending_hidden = None
    with pytest.raises(RuntimeError, match="pending tokens have no pending hidden"):
        runtime.compute.commit(
            batch.lanes,
            computation,
            emitted_counts=(len(computation.outputs[0]),),
            terminal=(False,),
        )

    lane.pending_tokens = []
    runtime.compute.abort(batch.lanes, batch.caches, computation, None)


def test_ragged_adapter_rejects_incomplete_cache_transaction_surfaces():
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    pair = SelfMTPCachePair([_LayerCache("t")], [_LayerCache("d")])
    assert adapter.attach(pair, []) is pair
    with pytest.raises(ValueError, match="attach no cache rows"):
        adapter.attach(None, [])
    with pytest.raises(ValueError, match="empty target"):
        adapter._merge([], "target")
    with pytest.raises(ValueError, match="equal non-zero width"):
        adapter._merge([[], []], "target")

    no_merge = type("NoMerge", (), {})
    with pytest.raises(ContinuousSelfMTPUnsupported, match="no merge surface"):
        adapter._merge([[no_merge()], [no_merge()]], "target")
    mixed = type("Mixed", (_LayerCache,), {})
    with pytest.raises(ContinuousSelfMTPUnsupported, match="mixed target"):
        adapter._merge([[_LayerCache("a")], [mixed("b")]], "target")

    current = SelfMTPCachePair([_LayerCache("t")], [_LayerCache("d")])
    wider = SelfMTPCachePair(
        [_LayerCache("t1"), _LayerCache("t2")],
        [_LayerCache("d1")],
    )
    with pytest.raises(ValueError, match="layer widths differ"):
        adapter.attach(current, [wider])

    class _NoExtend(_LayerCache):
        extend = None

    broken = SelfMTPCachePair([_NoExtend("t")], [_NoExtend("d")])
    with pytest.raises(ContinuousSelfMTPUnsupported, match="no extend surface"):
        adapter.attach(broken, [SelfMTPCachePair([_NoExtend("x")], [_NoExtend("y")])])

    class _NoExtract(_LayerCache):
        extract = None

    with pytest.raises(ContinuousSelfMTPUnsupported, match="no extract surface"):
        adapter.detach(SelfMTPCachePair([_NoExtract("t")], [_NoExtract("d")]), [0], [])

    class _NoFilter(_LayerCache):
        filter = None

    with pytest.raises(ContinuousSelfMTPUnsupported, match="no filter surface"):
        adapter.detach(SelfMTPCachePair([_NoFilter("t")], [_NoFilter("d")]), [0], [0])

    remaining, rows = adapter.detach(pair, [0], [])
    assert remaining.target == [] and remaining.draft == []
    assert len(rows) == 1


def test_ragged_adapter_default_hooks_and_nested_speculation_lifecycle():
    adapter = RapidRaggedCacheAdapter()
    assert callable(adapter._preflight)
    assert callable(adapter._trim)

    leaf = _LayerCache("leaf")
    nested = SimpleNamespace(caches=[leaf])
    backend_module._set_cache_speculation([nested], on=True)
    backend_module._set_cache_speculation([nested], on=False)
    assert ("start_speculation",) in leaf.events
    assert ("stop_speculation",) in leaf.events
