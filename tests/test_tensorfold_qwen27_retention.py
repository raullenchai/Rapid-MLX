"""Model-free ownership, admission and lifecycle contracts for optional cuts."""

from types import SimpleNamespace

import pytest

from rapid_mlx.speculative.tensorfold_qwen27 import _checkpoint_engine


class Store:
    slots = 8

    def __init__(self, budget_bytes=10_000):
        self.budget_bytes = budget_bytes
        self.entries = {}
        self.transfers = []

    def __len__(self):
        return len(self.entries)

    @property
    def nbytes(self):
        return sum(e.cache[0] for e in self.entries.values())

    def peek(self, prompt, usable=None):
        matches = [
            e
            for e in self.entries.values()
            if len(e.tokens) < len(prompt)
            and prompt[: len(e.tokens)] == e.tokens
            and (usable is None or usable(len(e.tokens)))
        ]
        return max(matches, key=lambda e: len(e.tokens), default=None)

    def match(self, prompt, usable=None, *, take=False):
        assert take, "the adapter may transfer only its explicitly owned cut"
        e = self.peek(prompt, usable)
        if e is None:
            return None
        self.entries.pop(tuple(e.tokens))
        self.transfers.append(e.tokens)
        return len(e.tokens), e.cache, e.last_prompt

    def insert(self, tokens, cache, *, last_prompt, pinned=False):
        previous = self.entries.get(tuple(tokens))
        self.entries[tuple(tokens)] = SimpleNamespace(
            tokens=list(tokens),
            cache=cache,
            last_prompt=list(last_prompt),
            pinned=pinned or bool(previous and previous.pinned),
        )
        # Tests never ask this stub to emulate the runtime's victim policy.
        assert len(self) <= self.slots
        assert self.nbytes <= self.budget_bytes


def prompt(conversation=0, *, edit=False, short_final=False):
    ids = [100 + conversation] * 7960
    for at in (0, 33, 7950):
        ids[at] = 10
    ids[7951] = 11
    if edit:
        ids[7900] = 999
    if short_final:
        ids[7900] = 10
    return ids


class Engine:
    prefill_step = 2048
    prefill_plan = SimpleNamespace(openers=(10,), assistant=(10, 11))

    def __init__(self):
        self.calls = []
        self.snapshot_bytes = {}

    def prompt_chunks(self, ids):
        starts = [0, 2048, 4096, 6144, 7950]
        return SimpleNamespace(
            starts=starts, floor=lambda at: max(n for n in starts if n <= at)
        )

    def cache_nbytes(self, cache):
        return cache[0]

    def begin_stream(self, stream, **kwargs):
        self.calls.append(kwargs)

        def steps():
            stream.history_checkpoints = []
            for at in kwargs["checkpoints_at"]:
                if kwargs["cached_tokens"] < at < len(stream.prompt_ids):
                    stream.history_checkpoints.append(
                        (stream.prompt_ids[:at], [self.snapshot_bytes.get(at, 100)])
                    )
                    yield at
            return "work"

        return steps()


def run(engine, ids, *, cached=0, selected=(7950,), cache=None):
    stream = SimpleNamespace(prompt_ids=ids, history_checkpoints=[])
    steps = engine.begin_stream(
        stream, cache=cache, cached_tokens=cached, checkpoints_at=selected
    )
    while True:
        try:
            next(steps)
        except StopIteration as done:
            assert done.value == "work"
            break
    return stream


def keep(store, stream):
    for tokens, cache in stream.history_checkpoints:
        store.insert(tokens, cache, last_prompt=stream.prompt_ids)


def setup_cut():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    first = run(engine, prompt())
    keep(store, first)
    return store, engine


def test_one_tail_cut_and_next_conversation_releases_only_owned_prefix():
    store, engine = setup_cut()
    old_boundary = tuple(prompt()[:7950])
    old_cut = tuple(prompt()[:6144])
    assert len(store) == 2
    assert tuple(engine.interior_tokens) == old_cut
    other = run(engine, prompt(1))
    assert old_cut not in store.entries
    assert old_boundary in store.entries
    assert store.transfers == [list(old_cut)]
    assert engine.interior_releases == 1
    assert [len(tokens) for tokens, _ in other.history_checkpoints] == [6144, 7950]
    keep(store, other)
    assert old_boundary in store.entries


@pytest.mark.parametrize("count", [7, 8])
def test_no_extra_copy_when_original_selections_fill_slots(count):
    store = Store()
    for c in range(count):
        store.insert([200 + c], [1], last_prompt=[200 + c, 0])
    engine = _checkpoint_engine(Engine, lambda: store)()
    result = run(engine, prompt())
    assert engine.calls[-1]["checkpoints_at"] == (7950,)
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]
    assert engine.interior_tokens is None


@pytest.mark.parametrize("current_bytes", [700, 900])
def test_byte_admission_drops_extra_but_keeps_every_original(current_bytes):
    store = Store(1000)
    store.insert([200], [current_bytes], last_prompt=[200, 0])
    engine = _checkpoint_engine(Engine, lambda: store)()
    engine.snapshot_bytes = {6144: 300, 7950: 200}
    result = run(engine, prompt())
    assert engine.calls[-1]["checkpoints_at"] == [6144, 7950]
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]
    assert engine.interior_tokens is None
    assert store.nbytes == current_bytes


@pytest.mark.parametrize("byte_pressure", [False, True])
def test_kept_cut_released_before_original_insert_under_pressure(byte_pressure):
    store, engine = setup_cut()
    if byte_pressure:
        store.budget_bytes = 250
    else:
        for c in range(6):
            store.insert([200 + c], [1], last_prompt=[200 + c, 0])
    boundaries = set(store.entries) - {tuple(engine.interior_tokens)}
    edited = run(engine, prompt(edit=True), cached=6144)
    assert engine.interior_tokens is None
    assert engine.interior_releases == 1
    assert set(store.entries) == boundaries
    keep(store, edited)
    assert boundaries <= set(store.entries)


def test_compatible_latest_edit_and_regenerate_retain_single_existing_cut():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    edited = run(engine, prompt(edit=True), cached=6144)
    keep(store, edited)
    assert tuple(engine.interior_tokens) == key
    again = run(engine, prompt(edit=True), cached=7950, selected=())
    assert again.history_checkpoints == []
    assert tuple(engine.interior_tokens) == key
    assert engine.interior_releases == 0


def test_short_final_message_releases_prior_long_message_cut():
    store, engine = setup_cut()
    old_boundary = tuple(prompt()[:7950])
    result = run(engine, prompt(short_final=True), cached=6144)
    assert engine.interior_tokens is None
    assert old_boundary in store.entries
    assert engine.interior_releases == 1
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]


def test_restored_cut_that_becomes_history_is_not_released_later():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    shortened = prompt()[:6144] + [100] * 56 + [10, 11] + [0] * 6
    run(engine, shortened, cached=6144, selected=())
    assert engine.interior_tokens is None
    assert key in store.entries
    assert store.entries[key].last_prompt == shortened
    run(engine, prompt(1))
    assert key in store.entries
    assert list(key) not in store.transfers


def test_original_or_pinned_checkpoint_is_never_owned_or_removed():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    original = run(engine, prompt(), selected=(6144, 7950))
    assert engine.interior_tokens is None
    assert [len(tokens) for tokens, _ in original.history_checkpoints] == [6144, 7950]
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    entry = store.entries[key]
    store.insert(entry.tokens, entry.cache, last_prompt=entry.last_prompt, pinned=True)
    run(engine, prompt(1))
    assert key in store.entries
    assert store.entries[key].pinned
    assert list(key) not in store.transfers


@pytest.mark.parametrize("stop_after,expected", [(1, [6144]), (2, [7950])])
@pytest.mark.parametrize("abort", ["close", "throw"])
def test_close_preserves_runtime_progress_and_originals(stop_after, expected, abort):
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    stream = SimpleNamespace(prompt_ids=prompt(), history_checkpoints=[])
    steps = engine.begin_stream(stream, checkpoints_at=[7950])
    for _ in range(stop_after):
        next(steps)
    if abort == "close":
        steps.close()
    else:
        with pytest.raises(RuntimeError, match="stopped"):
            steps.throw(RuntimeError("stopped"))
    assert [len(tokens) for tokens, _ in stream.history_checkpoints] == expected
    assert engine.interior_tokens is None


def test_cache_off_and_already_cached_prefix_forward_unchanged():
    engine = _checkpoint_engine(Engine, lambda: None)()
    cache = object()
    run(engine, prompt(), cached=6144, selected=(), cache=cache)
    assert engine.calls[-1] == dict(cache=cache, cached_tokens=6144, checkpoints_at=())
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    result = run(engine, prompt(), cached=6144)
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]
    assert engine.interior_tokens is None


def test_raw_completion_does_not_promote_an_interior_to_history():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    raw = prompt()[:6200]
    run(engine, raw, cached=6144, selected=())
    assert engine.interior_tokens is None
    assert key not in store.entries


def test_release_without_owned_cut_does_not_touch_store():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    engine._release_interior(store)
    assert store.transfers == []
    assert engine.interior_releases == 0


def test_missing_assistant_marker_does_not_promote_restored_cut():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    engine.prefill_plan = SimpleNamespace(openers=(10,), assistant=())
    shortened = prompt()[:6144] + [100] * 56 + [10, 11] + [0] * 6
    shortened[6150] = 10
    run(engine, shortened, cached=6144, selected=())
    assert engine.interior_tokens is None
    assert key not in store.entries
    assert store.transfers == [list(key)]


def test_interrupted_existing_cut_becomes_runtime_progress():
    # A cancelled prefill can return the restored cut as its retry progress.
    # It must then outlive the adapter's next incompatible prompt.
    store, engine = setup_cut()
    tokens = engine.interior_tokens
    stream = SimpleNamespace(prompt_ids=prompt(edit=True), history_checkpoints=[])
    steps = engine.begin_stream(stream, cached_tokens=6144, checkpoints_at=[7950])
    next(steps)
    stream.history_checkpoints.append((tokens, [100]))
    steps.close()
    assert engine.interior_tokens is None
    keep(store, stream)
    run(engine, prompt(1))
    assert tuple(tokens) in store.entries
    assert tokens not in store.transfers
