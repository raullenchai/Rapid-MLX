from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vllm_mlx.speculative.native_mtp import transaction


class _AppendCache:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self.state = np.array([offset])

    def trim(self, count: int) -> None:
        self.offset -= count

    def prepare(self, *, right_padding) -> None:
        self.padding = right_padding

    def finalize(self) -> None:
        self.finalized = True


class _BatchAppendCache(_AppendCache):
    def __init__(self, idx: int = 0):
        self._idx = idx

    def trim(self, count: int) -> None:
        self._idx -= count


class _CacheList:
    def __init__(self, caches):
        self.caches = caches


class _TemporalCache:
    def __init__(self):
        self.state = "before"
        self._before = None
        self._generation = 0

    def start_speculation(self, length: int) -> int:
        self._before = self.state
        self._length = length
        self._generation += 1
        return self._generation

    def validate_speculation(self, lengths, generation) -> None:
        assert generation == self._generation
        assert all(0 <= length <= self._length for length in lengths)

    def commit_speculation(self, lengths, generation) -> None:
        self.validate_speculation(lengths, generation)
        self.state = (self._before, lengths[0])
        self._before = None

    def abort_speculation(self, generation) -> None:
        assert generation == self._generation
        if self._before is not None:
            self.state = self._before
            self._before = None


@pytest.fixture
def fake_cache_types(monkeypatch):
    monkeypatch.setattr(
        transaction,
        "_cache_types",
        lambda: {
            "temporal": (_TemporalCache,),
            "append": (_AppendCache, _BatchAppendCache),
            "batch": (_BatchAppendCache,),
            "list": _CacheList,
        },
    )


@pytest.fixture
def fake_mx(monkeypatch):
    mx = SimpleNamespace(
        array=np.array,
        zeros=np.zeros,
        zeros_like=np.zeros_like,
        arange=np.arange,
        concatenate=np.concatenate,
        argmax=np.argmax,
        take_along_axis=np.take_along_axis,
        int32=np.int32,
        async_eval=lambda *_args: None,
    )
    monkeypatch.setattr(transaction, "_mx", lambda: mx)
    return mx


def test_cache_transaction_commits_only_the_accepted_prefix(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    temporal = _TemporalCache()
    with transaction.CacheTransaction([append, temporal], 3) as active:
        append.offset += 3
        temporal.state = "after-three"
        active.commit([2])

    assert append.offset == 9
    assert temporal.state == ("before", 2)

    unchanged = _AppendCache(offset=4)
    with transaction.CacheTransaction([unchanged], 2) as no_forward:
        no_forward.commit([1])
    assert unchanged.offset == 4


def test_cache_transaction_aborts_both_cache_kinds(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    temporal = _TemporalCache()
    with transaction.CacheTransaction([append, temporal], 3):
        append.offset += 3
        temporal.state = "after-three"

    assert append.offset == 7
    assert temporal.state == "before"


def test_cache_transaction_rejects_partial_forward(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    with (
        pytest.raises(RuntimeError, match="full verification block"),
        transaction.CacheTransaction([append], 3) as active,
    ):
        append.offset += 2
        active.commit([1])
    assert append.offset == 7


def test_cache_transaction_validates_protocol_and_ragged_batch(
    fake_cache_types,
) -> None:
    leaf = _AppendCache(offset=3)
    nested = _CacheList([None, _CacheList([leaf, leaf])])
    assert list(transaction.iter_leaf_caches([nested])) == [leaf, leaf]

    with pytest.raises(ValueError, match="positive length"):
        transaction.CacheTransaction([leaf], 0)
    with pytest.raises(ValueError, match="does not support"):
        transaction.CacheTransaction([object()], 1)

    active = transaction.CacheTransaction([leaf], 2)
    with pytest.raises(ValueError, match="between 0 and 2"):
        active.validate([3])
    with pytest.raises(ValueError, match="between 0 and 2"):
        active.validate([])
    leaf.offset += 2
    with pytest.raises(ValueError, match="Ragged acceptance"):
        active.validate([1, 2])
    active.abort()
    active.abort()
    with pytest.raises(RuntimeError, match="already finished"):
        active.validate([1])

    batch = _BatchAppendCache(idx=4)
    with transaction.CacheTransaction([batch], 3) as batch_tx:
        batch._idx += 3
        batch_tx.commit([1, 2])
    assert batch._idx == 6
    assert batch.padding == [1, 0]
    assert batch.finalized is True


def test_cache_transaction_detects_replacement_and_start_failure(
    fake_cache_types, monkeypatch
) -> None:
    leaf = _AppendCache()
    caches = [leaf]
    active = transaction.CacheTransaction(caches, 1)
    caches[0] = _AppendCache()
    with pytest.raises(RuntimeError, match="replaced cache objects"):
        active.validate([1])
    active.abort()

    temporal = _TemporalCache()
    append = _AppendCache(offset=5)
    monkeypatch.setattr(
        temporal,
        "start_speculation",
        lambda _length: (_ for _ in ()).throw(RuntimeError("start failed")),
    )
    with pytest.raises(RuntimeError, match="start failed"):
        transaction.CacheTransaction([append, temporal], 2)
    assert append.offset == 5


def test_speculative_stats_snapshot_is_reporting_compatible() -> None:
    stats = transaction.SpeculativeStats()
    stats.record([1, 2, 3], [1, 2, 9])
    assert stats.snapshot() == (1, 2, 3)
    stats.record([], [1])
    stats.record([1], [])
    assert stats.snapshot() == (1, 2, 3)


class _Draft:
    def __init__(self, block_size: int = 3):
        self.config = SimpleNamespace(block_size=block_size)
        self.cache = [_AppendCache()]

    def make_cache(self, _model):
        return self.cache

    def __call__(
        self, tokens, hidden, cache, position, *, target_model=None, lengths=None
    ):
        return _draft_forward(tokens, hidden, cache, position, lengths=lengths)


def _draft_forward(tokens, hidden, cache, position, *, lengths=None):
    width = tokens.shape[1]
    for leaf in cache:
        leaf.offset += width
    logits = np.zeros((tokens.shape[0], width, 8), dtype=np.float32)
    logits[..., 4] = 1
    draft_hidden = np.arange(tokens.shape[0] * width * 2).reshape(
        tokens.shape[0], width, 2
    )
    return logits, draft_hidden


def _seeded_state(fake_mx):
    target = [_AppendCache()]
    draft = [_AppendCache()]
    state = transaction.SpeculativeCache(
        target, draft, [0], np.array([[7]], dtype=np.int32)
    )
    state.seed = transaction.DraftState(
        np.array([[2]], dtype=np.int32), np.zeros((1, 1, 2), dtype=np.float32)
    )
    return state


def test_speculative_cache_create_positions_and_prefill(
    fake_cache_types, fake_mx
) -> None:
    drafter = _Draft()
    with pytest.raises(ValueError, match="batch size one"):
        transaction.SpeculativeCache.create([], drafter, 2)
    state = transaction.SpeculativeCache.create([], drafter, 1)
    np.testing.assert_array_equal(state.positions(3), [[0, 1, 2]])

    tokens = np.array([[1, 2]], dtype=np.int32)
    hidden = np.zeros((1, 2, 2), dtype=np.float32)
    state.bonus = np.array([[7]], dtype=np.int32)
    state.prefill(tokens, hidden, _draft_forward)
    assert state.position.tolist() == [2]
    assert state.seed.token.tolist() == [[4, 4]]

    with pytest.raises(ValueError, match="hidden states"):
        state.prefill(tokens, hidden[:, :1], _draft_forward)
    with pytest.raises(ValueError, match="hidden states"):
        state.prefill(tokens[:, :0], hidden[:, :0], _draft_forward)


def test_speculative_cache_propose_abort_and_failures(
    fake_cache_types, fake_mx
) -> None:
    state = _seeded_state(fake_mx)
    assert state.propose(0, _draft_forward).shape == (1, 0)
    with pytest.raises(RuntimeError, match="previous speculative round"):
        state.propose(1, _draft_forward)
    state.abort()

    proposals = state.propose(3, _draft_forward)
    assert proposals.tolist() == [[2, 4, 4]]
    state.abort()
    assert state.target[0].offset == 0
    assert state.draft[0].offset == 0

    state.seed = None
    with pytest.raises(ValueError, match="Prefill MTP"):
        state.propose(1, _draft_forward)

    state.seed = transaction.DraftState(
        np.array([[2]]), np.zeros((1, 1, 2), dtype=np.float32)
    )

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("draft failed")

    with pytest.raises(RuntimeError, match="draft failed"):
        state.propose(2, fail_forward)
    assert state._target_round is None


def test_speculative_cache_verify_commit_and_stats(fake_cache_types, fake_mx) -> None:
    state = _seeded_state(fake_mx)
    proposals = state.propose(2, _draft_forward)
    assert state.verify_inputs(proposals).tolist() == [[7, 2, 4]]
    hidden = np.arange(6).reshape(1, 3, 2)
    state.record_verification(hidden)
    state.target[0].offset += 3
    state.tokens = [[10]]
    state.commit([[2, 6]], _draft_forward)
    assert state.position.tolist() == [2]
    assert state.bonus.tolist() == [[6]]
    assert state.seed.token.tolist() == [[4, 4]]
    assert state.tokens == [[10, 2, 6]]
    assert state.stats[0].snapshot() == (1, 1, 2)
    assert state._target_round is None

    state.propose(1, _draft_forward)
    state.commit([[]], _draft_forward)
    assert state._target_round is None


def test_speculative_cache_rolls_back_target_if_replay_commit_fails(
    fake_cache_types, fake_mx, monkeypatch
) -> None:
    state = _seeded_state(fake_mx)
    state.propose(1, _draft_forward)
    state.record_verification(np.zeros((1, 2, 2)))
    state.target[0].offset += 2
    original_commit = transaction.CacheTransaction.commit

    def fail_draft_commit(self, lengths):
        if self.caches is state.draft:
            raise RuntimeError("replay commit failed")
        return original_commit(self, lengths)

    monkeypatch.setattr(transaction.CacheTransaction, "commit", fail_draft_commit)
    with pytest.raises(RuntimeError, match="replay commit failed"):
        state.commit([[2]], _draft_forward)
    assert state.target[0].offset == 0


def test_speculative_prefill_lifecycle(fake_cache_types, fake_mx) -> None:
    passthrough = transaction.SpeculativePrefill("mtp", None)
    output = SimpleNamespace(hidden_states=[np.zeros((1, 1, 2))])
    passthrough.append(output)
    assert passthrough.finish(output) is output

    tokens = np.array([[1, 2, 3]], dtype=np.int32)
    drafter = _Draft()
    prefill = transaction.SpeculativePrefill("mtp", drafter, tokens=tokens)
    target = [_AppendCache()]
    model = SimpleNamespace(language_model="target")
    prefill.start(model, target, drafter)
    prefill.append(SimpleNamespace(hidden_states=[np.zeros((1, 2, 2))]))
    assert prefill.consumed == 2
    final = SimpleNamespace(hidden_states=[np.zeros((1, 1, 2))])
    assert prefill.finish(final, np.array([5])) is final
    assert prefill.state.bonus.tolist() == [[5]]

    existing = _seeded_state(fake_mx)
    prefill.start(model, target, drafter, state=existing)
    assert prefill.state is existing


def test_accepted_greedy_applies_processors_and_stops_on_mismatch(fake_mx) -> None:
    proposals = np.array([[2, 3]], dtype=np.int32)
    logits = np.zeros((1, 3, 6), dtype=np.float32)
    logits[0, 0, 2] = logits[0, 1, 5] = logits[0, 2, 4] = 1
    calls = []

    def processor(context, scores):
        calls.append(context.tolist())
        return scores

    assert transaction._accepted_greedy(proposals, logits, 3, [processor], [1]) == [
        [2, 5]
    ]
    assert calls == [[1], [1, 2]]


class _RoundState:
    def __init__(self, proposals, logits):
        self.proposals = np.array([proposals], dtype=np.int32)
        self.bonus = np.array([[9]], dtype=np.int32)
        self.target = []
        self.position = np.array([0])
        self.position_offset = np.array([0])
        self.logits = logits
        self.tokens = None
        self.commits = []
        self.aborts = 0

    def propose(self, count, _forward):
        return self.proposals[:, :count]

    def verify_inputs(self, proposals):
        return np.concatenate([self.bonus, proposals], axis=1)

    def positions(self, length):
        return np.arange(length)[None]

    def record_verification(self, hidden):
        self.hidden = hidden

    def commit(self, emitted, _forward):
        self.commits.append([list(row) for row in emitted])

    def abort(self):
        self.aborts += 1


class _Target:
    config = SimpleNamespace(eos_token_id=[6])

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(logits=self.logits, hidden_states=[np.zeros((1, 3, 2))])


def test_mtp_rounds_yields_commits_and_stops(fake_mx) -> None:
    logits = np.zeros((1, 3, 8), dtype=np.float32)
    logits[0, 0, 2] = logits[0, 1, 3] = logits[0, 2, 6] = 1
    state = _RoundState([2, 3], logits)
    target = _Target(logits)
    drafter = _Draft(block_size=3)
    got = list(
        transaction.mtp_rounds(
            target,
            drafter,
            [],
            np.zeros((1, 1, 2)),
            prompt_tokens=np.array([[1]]),
            first_bonus=np.array([9]),
            max_tokens=5,
            eos_token_ids={6},
            state=state,
        )
    )
    assert [tokens for tokens, _meta in got] == [[2], [3], [6]]
    assert state.commits == [[[2, 3, 6]]]
    assert state.aborts == 1


def test_mtp_rounds_rejects_bad_limits_and_commits_on_early_close(fake_mx) -> None:
    drafter = _Draft(block_size=1)
    with pytest.raises(ValueError, match="integer token limit"):
        next(
            transaction.mtp_rounds(
                object(),
                drafter,
                [],
                None,
                prompt_tokens=np.array([[1]]),
                first_bonus=np.array([2]),
                max_tokens=[3],
            )
        )
    with pytest.raises(ValueError, match="block size"):
        next(
            transaction.mtp_rounds(
                object(),
                drafter,
                [],
                None,
                prompt_tokens=np.array([[1]]),
                first_bonus=np.array([2]),
                max_tokens=3,
            )
        )

    logits = np.zeros((1, 3, 8), dtype=np.float32)
    logits[0, 0, 2] = logits[0, 1, 3] = logits[0, 2, 4] = 1
    state = _RoundState([2, 3], logits)
    rounds = transaction.mtp_rounds(
        _Target(logits),
        _Draft(block_size=3),
        [],
        np.zeros((1, 1, 2)),
        prompt_tokens=np.array([[1]]),
        first_bonus=np.array([9]),
        max_tokens=4,
        stop_check=lambda _row, token: token == 3,
        state=state,
    )
    assert next(rounds)[0] == [2]
    rounds.close()
    assert state.commits == [[[2]]]
    assert state.aborts == 1


def test_run_speculative_rounds_contract_and_close(fake_mx, monkeypatch) -> None:
    assert (
        list(
            transaction.run_speculative_rounds(
                object(),
                object(),
                [],
                None,
                None,
                None,
                None,
                draft_kind="mtp",
                max_tokens=0,
            )
        )
        == []
    )
    with pytest.raises(ValueError, match="greedy MTP"):
        next(
            transaction.run_speculative_rounds(
                object(),
                object(),
                [],
                None,
                np.array(1),
                None,
                None,
                draft_kind="other",
                max_tokens=2,
                sampler_is_greedy=True,
            )
        )

    closed = []

    def fake_rounds(*_args, **_kwargs):
        try:
            yield [4], {}
            yield [5], {}
        finally:
            closed.append(True)

    monkeypatch.setattr(transaction, "mtp_rounds", fake_rounds)
    target = SimpleNamespace(config=SimpleNamespace(eos_token_id=6))
    model = SimpleNamespace(language_model=target)
    output = SimpleNamespace(hidden_states=[np.zeros((1, 1, 2))])
    rounds = transaction.run_speculative_rounds(
        model,
        object(),
        [],
        np.array([[1]]),
        np.array(3),
        None,
        output,
        draft_kind="mtp",
        max_tokens=3,
        sampler_is_greedy=True,
    )
    assert list(rounds) == [(3, None), (4, None), (5, None)]
    assert closed == [True]


def test_speculative_prefill_kwargs_contract() -> None:
    assert transaction.speculative_prefill_kwargs("mtp", object()) == {
        "return_hidden": True
    }
    with pytest.raises(ValueError, match="only MTP"):
        transaction.speculative_prefill_kwargs("other", object())


def test_lazy_mlx_and_cache_import_helpers(monkeypatch) -> None:
    fake_core = ModuleType("mlx.core")
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    assert transaction._mx() is fake_core

    root = ModuleType("mlx_vlm")
    root.__path__ = []
    models = ModuleType("mlx_vlm.models")
    models.__path__ = []
    cache = ModuleType("mlx_vlm.models.cache")
    names = (
        "ArraysCache",
        "BatchKVCache",
        "BatchPoolingCache",
        "BatchQuantizedKVCache",
        "CacheList",
        "KVCache",
        "PoolingCache",
        "QuantizedKVCache",
    )
    for name in names:
        setattr(cache, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", cache)
    resolved = transaction._cache_types()
    assert resolved["list"] is cache.CacheList
    assert resolved["batch"] == (cache.BatchKVCache, cache.BatchQuantizedKVCache)


def test_mtp_rounds_constructs_state_from_prompt(fake_mx, monkeypatch) -> None:
    logits = np.zeros((1, 1, 8), dtype=np.float32)
    state = _RoundState([], logits)
    state.prefills = []

    def prefill(tokens, hidden, forward):
        state.prefills.append((tokens.tolist(), hidden.shape, callable(forward)))

    state.prefill = prefill
    monkeypatch.setattr(
        transaction.SpeculativeCache,
        "create",
        classmethod(lambda cls, cache, drafter, batch: state),
    )
    got = list(
        transaction.mtp_rounds(
            _Target(logits),
            _Draft(block_size=2),
            [],
            np.zeros((1, 1, 2)),
            prompt_tokens=np.array([[1]]),
            first_bonus=np.array([6]),
            max_tokens=2,
            eos_token_ids={6},
        )
    )
    assert got == []
    assert state.prefills == [([[1]], (1, 1, 2), True)]
    assert state.aborts == 1


def test_generation_hook_replaces_only_speculative_seams(monkeypatch) -> None:
    root = ModuleType("mlx_vlm")
    root.__path__ = []
    generate = ModuleType("mlx_vlm.generate")
    generate.__path__ = []
    ar = ModuleType("mlx_vlm.generate.ar")
    original_generate_step = object()
    ar.generate_step = original_generate_step
    ar.SpeculativePrefill = object()
    ar.run_speculative_rounds = object()
    ar.speculative_prefill_kwargs = object()
    generate.ar = ar
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.ar", ar)

    transaction.install_generation_hooks()

    assert ar.generate_step is original_generate_step
    assert ar.SpeculativePrefill is transaction.SpeculativePrefill
    assert ar.run_speculative_rounds is transaction.run_speculative_rounds
    assert ar.speculative_prefill_kwargs is transaction.speculative_prefill_kwargs


def test_generation_hook_fails_closed_without_complete_seam(monkeypatch) -> None:
    root = ModuleType("mlx_vlm")
    root.__path__ = []
    generate = ModuleType("mlx_vlm.generate")
    generate.__path__ = []
    generate.ar = SimpleNamespace(generate_step=object())
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)

    with pytest.raises(RuntimeError, match="qualified generation seam"):
        transaction.install_generation_hooks()
