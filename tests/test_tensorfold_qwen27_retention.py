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


def setup_cut(engine_type=Engine):
    store = Store()
    engine = _checkpoint_engine(engine_type, lambda: store)()
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
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store, engine = setup_cut(PlannedEngine)
    if byte_pressure:
        store.budget_bytes = 250
    else:
        for c in range(6):
            store.insert([200 + c], [1], last_prompt=[200 + c, 0])
    boundaries = set(store.entries) - {tuple(engine.interior_tokens)}
    # A longer suffix resumes the native history, leaving the optional cut
    # owned until pending original snapshots force its release.
    ids = prompt() + [100] * 4000
    starts = engine.prompt_chunks(ids)
    chosen = checkpoints.choose_checkpoints(11000, 7950, prompt(), ids)
    selected = sorted(
        at for at in {starts.floor(n) for n in chosen} if 7950 < at < len(ids)
    )
    assert selected == [9998]
    edited = run(engine, ids, cached=7950, selected=selected)
    assert engine.interior_tokens is None
    assert engine.interior_releases == 1
    assert set(store.entries) == boundaries
    keep(store, edited)
    assert boundaries <= set(store.entries)


def test_resumed_latest_edit_keeps_expendable_ownership():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    edited = run(engine, prompt(edit=True), cached=6144)
    keep(store, edited)
    assert engine.interior_tokens is None
    assert key in store.entries
    assert tuple(engine.resumed_tokens) == key
    again = run(engine, prompt(edit=True), cached=7950, selected=())
    assert again.history_checkpoints == []
    assert engine.interior_tokens is None
    assert engine.interior_releases == 0


def test_resumed_short_final_message_keeps_expendable_ownership():
    store, engine = setup_cut()
    old_boundary = tuple(prompt()[:7950])
    result = run(engine, prompt(short_final=True), cached=6144)
    assert engine.interior_tokens is None
    assert old_boundary in store.entries
    assert engine.resumed_tokens == prompt()[:6144]
    assert engine.interior_releases == 0
    assert tuple(prompt()[:6144]) in store.entries
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]


def test_original_or_pinned_checkpoint_is_never_owned_or_removed():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    original = run(engine, prompt(), selected=(6144, 7950))
    assert engine.interior_tokens == prompt()[:4096]
    assert [len(tokens) for tokens, _ in original.history_checkpoints] == [
        4096,
        6144,
        7950,
    ]
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    entry = store.entries[key]
    store.insert(entry.tokens, entry.cache, last_prompt=entry.last_prompt, pinned=True)
    run(engine, prompt(1))
    assert key in store.entries
    assert engine.resumed_tokens is None
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


def test_raw_completion_resuming_cut_keeps_expendable_ownership():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    raw = prompt()[:6200]
    run(engine, raw, cached=6144, selected=())
    assert engine.interior_tokens is None
    assert key in store.entries
    assert tuple(engine.resumed_tokens) == key


def test_release_without_owned_cut_does_not_touch_store():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store)()
    engine._release_interior(store)
    assert store.transfers == []
    assert engine.interior_releases == 0


def test_missing_assistant_marker_does_not_change_resumed_cut_policy():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    engine.prefill_plan = SimpleNamespace(openers=(10,), assistant=())
    shortened = prompt()[:6144] + [100] * 56 + [10, 11] + [0] * 6
    shortened[6150] = 10
    run(engine, shortened, cached=6144, selected=())
    assert engine.interior_tokens is None
    assert key in store.entries
    assert tuple(engine.resumed_tokens) == key
    assert store.transfers == []


def test_interrupted_resumed_cut_keeps_ownership():
    # Cancellation keeps ownership of the restored cut.
    store, engine = setup_cut()
    tokens = engine.interior_tokens
    stream = SimpleNamespace(prompt_ids=prompt(edit=True), history_checkpoints=[])
    steps = engine.begin_stream(stream, cached_tokens=6144, checkpoints_at=[7950])
    assert engine.interior_tokens is None
    next(steps)
    steps.close()
    assert engine.interior_tokens is None
    keep(store, stream)
    run(engine, prompt(1))
    assert tuple(tokens) in store.entries
    assert tokens not in store.transfers


class PlannedEngine(Engine):
    def __init__(self):
        super().__init__()
        plan = pytest.importorskip("tensorfold.engine.prefill_plan")
        self.prefill_plan = plan.PrefillPlan(
            step=2048, openers=(10,), min_chunk=256, assistant=(10, 11)
        )

    def prompt_chunks(self, ids):
        return self.prefill_plan.chunks(ids)


def test_aligned_history_boundary_keeps_latest_eligible_interior():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    engine = _checkpoint_engine(PlannedEngine, lambda: store)()
    ids = [100] * 8202
    for at in (0, 33, 8192):
        ids[at] = 10
    ids[8193] = 11
    assert engine.prompt_chunks(ids).starts == [0, 2048, 4096, 6144, 8192]
    first = run(engine, ids, selected=(8192,))
    assert [len(t) for t, _ in first.history_checkpoints] == [6144, 8192]
    assert engine.interior_tokens == ids[:6144]
    keep(store, first)
    edited = ids.copy()
    edited[7900] = 999
    assert (
        store.longest(edited, usable=lambda n: n in engine.prompt_chunks(edited))
        == 6144
    )


@pytest.mark.parametrize("history", [0, 6200])
def test_scheduler_restored_cut_survives_other_chat_and_regeneration(history):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    engine = _checkpoint_engine(PlannedEngine, lambda: store)()

    def request(engine, store, ids, history):
        starts = engine.prompt_chunks(ids)
        hit = store.match(ids, usable=lambda n: n in starts)
        cached, cache, last = hit if hit else (0, None, None)
        chosen = checkpoints.choose_checkpoints(history, cached, last, ids)
        selected = sorted(
            at for at in {starts.floor(n) for n in chosen} if cached < at < len(ids)
        )
        stream = run(engine, ids, cached=cached, cache=cache, selected=selected)
        keep(store, stream)
        return cached, selected

    # Compare main and the hook using actual scheduler selections, including
    # raw completions containing the same assistant marker as shortened chats.
    main_store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    main = PlannedEngine()
    for current, cache_store in ((main, main_store), (engine, store)):
        request(current, cache_store, prompt(), 7950)
    owned = engine.interior_tokens.copy()
    shortened = owned + [100] * 56 + [10, 11] + [100] * 8
    assert engine.prompt_chunks(shortened).floor(6200) == 6144
    entry = store.peek(shortened, usable=lambda n: n == 6144)
    assert request(engine, store, shortened, history) == (6144, [])
    assert request(main, main_store, shortened, history) == (
        0,
        [6144] if history else [],
    )
    assert engine.interior_tokens is None
    assert engine.interior_releases == 0
    assert engine.resumed_tokens == owned
    # The native match updates last_prompt; the hook neither reinserts nor
    # replaces that entry or its cache.
    assert store.peek(shortened, usable=lambda n: n == 6144) is entry
    assert entry.last_prompt == shortened
    for current, cache_store in ((main, main_store), (engine, store)):
        request(current, cache_store, prompt(1), 7950)
    assert store.peek(shortened, usable=lambda n: n == 6144) is entry
    assert engine.interior_releases == 0
    assert request(engine, store, shortened, history)[0] == 6144
    assert request(main, main_store, shortened, history)[0] == (6144 if history else 0)


def test_cold_raw_completion_delegates_empty_scheduler_selection():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    engine = _checkpoint_engine(PlannedEngine, lambda: store)()
    ids = prompt()
    selected = checkpoints.choose_checkpoints(0, 0, None, ids)
    assert selected == []
    result = run(engine, ids, selected=selected)
    assert result.history_checkpoints == []
    assert engine.calls[-1]["checkpoints_at"] == []
    assert engine.interior_tokens is None
    assert len(store) == 0


def runtime_request(engine, store, ids, history):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    chunks = engine.prompt_chunks(ids)
    hit = store.match(ids, usable=lambda n: n in chunks)
    cached, cache, last = hit if hit else (0, None, None)
    chosen = checkpoints.choose_checkpoints(history, cached, last, ids)
    selected = sorted(
        at for at in {chunks.floor(n) for n in chosen} if cached < at < len(ids)
    )
    stream = run(engine, ids, cached=cached, cache=cache, selected=selected)
    keep(store, stream)
    return cached, selected


def short_prompt(conversation):
    ids = [200 + conversation] * 1000
    for at in (0, 33, 990):
        ids[at] = 10
    ids[991] = 11
    return ids


@pytest.mark.parametrize("history", [0, 7950])
@pytest.mark.parametrize("byte_pressure", [False, True])
def test_resumed_cut_released_before_native_pressure_matches_main(
    history, byte_pressure
):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    results = []
    for wrapped in (False, True):
        store = checkpoints.CheckpointStore(
            8,
            copier=list,
            sizer=lambda c: c[0],
            budget_bytes=800 if byte_pressure else None,
        )
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store)
            if wrapped
            else PlannedEngine
        )()
        runtime_request(engine, store, prompt(), 7950)
        if history:
            edited = prompt(edit=True)
        else:
            # Raw assistant markers must not promote a restored cut to unowned.
            edited = prompt()[:6144] + [100] * 56 + [10, 11] + [100] * 8
        runtime_request(engine, store, edited, history)
        for c in range(6 if history else 7):
            runtime_request(engine, store, short_prompt(c), 990)
        results.append(
            (store.evictions, runtime_request(engine, store, prompt(), 7950)[0])
        )
    assert results == [(0, 7950), (0, 7950)]
    assert engine.resumed_tokens is None
    assert engine.interior_releases >= 1


def test_resumed_and_fresh_cut_coexist_and_resumed_releases_first():
    store, engine = setup_cut()
    owned = engine.interior_tokens.copy()
    keep(store, run(engine, prompt(edit=True), cached=6144))
    keep(store, run(engine, prompt(1)))
    assert engine.resumed_tokens == owned
    assert engine.interior_tokens == prompt(1)[:6144]
    # Fill the remaining slots without invoking native eviction. The next
    # original needs one slot; begin_stream must release resumed ownership first.
    for c in range(3):
        store.insert([200 + c], [1], last_prompt=[200 + c, 0])
    run(engine, prompt(1), cached=7950, selected=(7990,))
    assert engine.resumed_tokens is None
    assert engine.interior_tokens == prompt(1)[:6144]
    assert store.transfers == [owned]


def test_resumed_pressure_in_finally_with_new_optional_admission():
    store, engine = setup_cut()
    owned = engine.interior_tokens.copy()
    keep(store, run(engine, prompt(edit=True), cached=6144))
    store.budget_bytes = 450
    result = run(engine, prompt(1))
    assert engine.resumed_tokens is None
    assert engine.interior_tokens == prompt(1)[:6144]
    assert store.transfers == [owned]
    keep(store, result)
    assert store.nbytes == 400


def test_explicit_original_selection_promotes_only_matching_owned_tokens():
    store, engine = setup_cut()
    owned = engine.interior_tokens.copy()
    keep(store, run(engine, prompt(edit=True), cached=6144))
    # Exercise the public hook selection contract directly. The native scheduler
    # filters restored positions; this is not evidence of genuine chat history.
    keep(store, run(engine, prompt(), selected=(6144, 7950)))
    assert engine.resumed_tokens is None
    run(engine, prompt(1))
    assert tuple(owned) in store.entries


def test_two_resumes_bound_ownership_to_one_resumed_cut():
    store, engine = setup_cut()
    first = engine.interior_tokens.copy()
    keep(store, run(engine, prompt(edit=True), cached=6144))
    keep(store, run(engine, prompt(1)))
    second = engine.interior_tokens.copy()
    keep(store, run(engine, prompt(1, edit=True), cached=6144))
    assert engine.resumed_tokens == second
    assert engine.interior_tokens is None
    assert store.transfers == [first]


def test_pressure_can_lose_ambiguous_real_shortened_history():
    # The public hook cannot tell this cut from raw-completion surplus. Record
    # the remaining limit explicitly rather than claiming universal main parity.
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    results = []
    for wrapped in (False, True):
        store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store)
            if wrapped
            else PlannedEngine
        )()
        earlier = short_prompt(20)
        runtime_request(engine, store, earlier, 990)
        extended = earlier + [220] * 990 + [10, 11] + [220] * 8
        runtime_request(engine, store, extended, 1990)
        runtime_request(engine, store, prompt(), 7950)
        shortened = prompt()[:6144] + [100] * 56 + [10, 11] + [100] * 8
        runtime_request(engine, store, shortened, 6200)
        for c in range(5):
            runtime_request(engine, store, short_prompt(c), 990)
        results.append(runtime_request(engine, store, shortened, 6200)[0])
    assert results == [6144, 0]
