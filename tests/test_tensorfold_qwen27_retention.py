"""Model-free ownership, admission and lifecycle contracts for optional cuts."""

import os
import random
import threading
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
    stream = SimpleNamespace(stream_id="test", prompt_ids=ids, history_checkpoints=[])
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
    engine = _checkpoint_engine(engine_type, lambda: store, {})()
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
    engine = _checkpoint_engine(Engine, lambda: store, {})()
    result = run(engine, prompt())
    assert engine.calls[-1]["checkpoints_at"] == (7950,)
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]
    assert engine.interior_tokens is None


@pytest.mark.parametrize("current_bytes", [700, 900])
def test_byte_admission_drops_extra_but_keeps_every_original(current_bytes):
    store = Store(1000)
    store.insert([200], [current_bytes], last_prompt=[200, 0])
    engine = _checkpoint_engine(Engine, lambda: store, {})()
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
    # A later native history supersedes the optional cut before any pending
    # original insertion can exert pressure.
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
    assert engine.interior_tokens == list(key)
    assert key in store.entries
    assert tuple(engine.interior_tokens) == key
    again = run(engine, prompt(edit=True), cached=7950, selected=())
    assert again.history_checkpoints == []
    assert engine.interior_tokens is None
    assert engine.interior_releases == 1


def test_original_or_pinned_checkpoint_is_never_owned_or_removed():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store, {})()
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
    assert store.entries[key].pinned
    assert list(key) not in store.transfers


@pytest.mark.parametrize("stop_after,expected", [(1, [6144]), (2, [7950])])
@pytest.mark.parametrize("abort", ["close", "throw"])
def test_close_preserves_runtime_progress_and_originals(stop_after, expected, abort):
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store, {})()
    stream = SimpleNamespace(
        stream_id="test", prompt_ids=prompt(), history_checkpoints=[]
    )
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
    engine = _checkpoint_engine(Engine, lambda: None, {})()
    cache = object()
    run(engine, prompt(), cached=6144, selected=(), cache=cache)
    assert engine.calls[-1] == dict(cache=cache, cached_tokens=6144, checkpoints_at=())
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store, {})()
    result = run(engine, prompt(), cached=6144)
    assert [len(tokens) for tokens, _ in result.history_checkpoints] == [7950]
    assert engine.interior_tokens is None


def test_raw_completion_resuming_cut_keeps_expendable_ownership():
    store, engine = setup_cut()
    key = tuple(engine.interior_tokens)
    raw = prompt()[:6200]
    run(engine, raw, cached=6144, selected=())
    assert engine.interior_tokens == list(key)
    assert key in store.entries
    assert tuple(engine.interior_tokens) == key


def test_release_without_owned_cut_does_not_touch_store():
    store = Store()
    engine = _checkpoint_engine(Engine, lambda: store, {})()
    engine._release_interior(store)
    assert store.transfers == []
    assert engine.interior_releases == 0


def test_interrupted_resumed_cut_keeps_ownership():
    # Cancellation keeps ownership of the restored cut.
    store, engine = setup_cut()
    tokens = engine.interior_tokens
    stream = SimpleNamespace(
        stream_id="test", prompt_ids=prompt(edit=True), history_checkpoints=[]
    )
    steps = engine.begin_stream(stream, cached_tokens=6144, checkpoints_at=[7950])
    assert engine.interior_tokens == tokens
    next(steps)
    steps.close()
    assert engine.interior_tokens == tokens
    keep(store, stream)
    run(engine, prompt(1))
    assert tuple(tokens) not in store.entries
    assert tokens in store.transfers


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
    histories = {}
    engine = _checkpoint_engine(PlannedEngine, lambda: store, histories)()
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
    histories = {}
    engine = _checkpoint_engine(PlannedEngine, lambda: store, histories)()

    request = runtime_request

    # Compare main and the hook using actual scheduler selections, including
    # raw completions containing the same assistant marker as shortened chats.
    main_store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    main = PlannedEngine()
    engine.test_histories = histories
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
    assert engine.interior_tokens == (None if history else owned)
    assert engine.interior_releases == 0
    promoted = store.peek(shortened, usable=lambda n: n == 6144)
    assert promoted.cache is entry.cache
    assert promoted.last_prompt == shortened
    assert promoted.born == (len(shortened) if history else len(prompt()))
    entry = promoted
    for current, cache_store in ((main, main_store), (engine, store)):
        request(current, cache_store, prompt(1), 7950)
    assert (store.peek(shortened, usable=lambda n: n == 6144) is entry) == bool(history)
    assert engine.interior_releases == (0 if history else 1)
    assert request(engine, store, shortened, history)[0] == (6144 if history else 0)
    assert request(main, main_store, shortened, history)[0] == (6144 if history else 0)


def test_cold_raw_completion_delegates_empty_scheduler_selection():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    histories = {}
    engine = _checkpoint_engine(PlannedEngine, lambda: store, histories)()
    ids = prompt()
    selected = checkpoints.choose_checkpoints(0, 0, None, ids)
    assert selected == []
    result = run(engine, ids, selected=selected)
    assert result.history_checkpoints == []
    assert engine.calls[-1]["checkpoints_at"] == []
    assert engine.interior_tokens is None
    assert len(store) == 0


def runtime_request(engine, store, ids, history, shared=()):
    histories = getattr(engine, "test_histories", {})
    histories["test"] = (history, tuple(shared))
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    chunks = engine.prompt_chunks(ids)
    hit = store.match(ids, usable=lambda n: n in chunks)
    cached, cache, last = hit if hit else (0, None, None)
    chosen = checkpoints.choose_checkpoints(history, cached, last, ids)
    selected = sorted(
        at
        for at in {
            *(chunks.floor(n) for n in chosen),
            *(chunks.floor(n) for n in shared),
        }
        if cached < at < len(ids)
    )
    stream = run(engine, ids, cached=cached, cache=cache, selected=selected)
    shared_at = {chunks.floor(n) for n in shared} - {0}
    for tokens, snapshot in stream.history_checkpoints:
        store.insert(tokens, snapshot, last_prompt=ids, pinned=len(tokens) in shared_at)
    histories.pop("test", None)
    return engine.calls[-1]["cached_tokens"], selected


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
        histories = {}
        store = checkpoints.CheckpointStore(
            8,
            copier=list,
            sizer=lambda c: c[0],
            budget_bytes=800 if byte_pressure else None,
        )
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store, histories)
            if wrapped
            else PlannedEngine
        )()
        engine.test_histories = histories
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
    assert engine.interior_releases >= 1


def test_shortened_history_with_superseded_boundary_matches_main_under_pressure():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    results = []
    for wrapped in (False, True):
        histories = {}
        store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store, histories)
            if wrapped
            else PlannedEngine
        )()
        engine.test_histories = histories
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
    assert results == [6144, 6144]


@pytest.mark.parametrize("history", [0, 6143, 6144, 6200, 6210, 7000])
def test_promotion_mirrors_cold_scheduler_history_floor(history):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    histories = {}
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    engine = _checkpoint_engine(PlannedEngine, lambda: store, histories)()
    engine.test_histories = histories
    runtime_request(engine, store, prompt(), 7950)
    owned = engine.interior_tokens.copy()
    shortened = owned + [100] * 56 + [10, 11] + [100] * 8
    chunks = engine.prompt_chunks(shortened)
    cold = {
        chunks.floor(n)
        for n in checkpoints.choose_checkpoints(history, 0, None, shortened)
    } - {0}
    runtime_request(engine, store, shortened, history)
    assert (engine.interior_tokens is None) == (6144 in cold)


def test_missing_stream_id_registry_entry_defaults_to_zero():
    store, engine = setup_cut(PlannedEngine)
    owned = engine.interior_tokens.copy()
    shortened = owned + [100] * 56 + [10, 11] + [100] * 8
    run(engine, shortened, cached=6144, selected=())
    assert engine.interior_tokens == owned
    run(engine, prompt(1))
    assert tuple(owned) not in store.entries


def test_promoted_creation_length_preserves_native_eviction_order():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    results = []
    for wrapped in (False, True):
        histories = {}
        store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store, histories)
            if wrapped
            else PlannedEngine
        )()
        engine.test_histories = histories
        runtime_request(engine, store, prompt(), 7950)
        shortened = prompt()[:6144] + [100] * 56 + [10, 11] + [100] * 8
        runtime_request(engine, store, shortened, 6200)
        continuation = shortened + [100] * 980 + [10, 11] + [100] * 8
        runtime_request(engine, store, continuation, 7190)
        for c in range(6):
            runtime_request(engine, store, short_prompt(c), 990)
        results.append(
            (store.evictions, runtime_request(engine, store, prompt(), 7950)[0])
        )
    assert results == [(1, 7950), (1, 7950)]


def test_pending_bytes_release_resumed_optional_before_original_transfer():
    store, engine = setup_cut()
    owned = engine.interior_tokens.copy()
    store.budget_bytes = 350
    engine.snapshot_bytes = {7950: 200}
    edited = run(engine, prompt(edit=True), cached=6144)
    assert engine.interior_tokens is None
    assert store.transfers == [owned]
    assert [len(tokens) for tokens, _ in edited.history_checkpoints] == [7950]
    keep(store, edited)
    assert store.nbytes == 300


@pytest.mark.parametrize("shared", [False, True])
def test_counterfactual_native_stable_or_shared_cut_survives_unrelated_chat(shared):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    results = []
    for wrapped in (False, True):
        histories = {}
        store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
        engine = (
            _checkpoint_engine(PlannedEngine, lambda store=store: store, histories)
            if wrapped
            else PlannedEngine
        )()
        engine.test_histories = histories
        shorter = prompt()[:4096] + [10, 11] + [100] * 8
        if not shared:
            runtime_request(engine, store, shorter, 4096)
        runtime_request(engine, store, prompt(), 7950)
        edited = prompt(edit=True)
        runtime_request(engine, store, edited, 7950, shared=(7000,) if shared else ())
        assert getattr(engine, "interior_tokens", None) is None
        promoted = store.peek(edited, usable=lambda n: n == 6144)
        assert promoted.last_prompt == edited
        assert promoted.born == len(edited)
        assert promoted.pinned == shared
        runtime_request(engine, store, short_prompt(0), 990)
        later = edited.copy()
        later[7000] = 777
        results.append(runtime_request(engine, store, later, 7950)[0])
    assert results == [6144, 6144]


def test_counterfactual_ignores_unusable_entry_and_does_not_touch_other_metadata():
    store, engine = setup_cut(PlannedEngine)
    # A strict prefix that is not a chunk start must not become main's hit.
    store.insert(prompt()[:6000], [1], last_prompt=prompt())
    unusable = store.entries[tuple(prompt()[:6000])]
    owned = engine.interior_tokens.copy()
    stream = SimpleNamespace(stream_id="absent", prompt_ids=prompt(edit=True))
    assert engine._native_selection_without_owned(store, stream, owned) == set()
    assert store.entries[tuple(prompt()[:6000])] is unusable
    assert unusable.last_prompt == prompt()


class SmallPlannedEngine(Engine):
    prefill_step = 16

    def __init__(self):
        super().__init__()
        plan = pytest.importorskip("tensorfold.engine.prefill_plan")
        self.prefill_plan = plan.PrefillPlan(
            step=16, openers=(10,), min_chunk=2, assistant=(10, 11)
        )

    def prompt_chunks(self, ids):
        return self.prefill_plan.chunks(ids)


def synthetic_requests(seed):
    """Adversarial lens may change independently of the unchanged prefix."""
    rng = random.Random(seed)
    conversations = {}
    for _ in range(24):
        conversation = rng.randrange(4)
        action = rng.choice(("new", "edit", "extend", "repeat", "shorten"))
        if conversation not in conversations or action == "new":
            length = rng.choice((30, 64, 66, 80, 96))
            ids = [100 + conversation] * length
            ids[0] = ids[3] = ids[-2] = 10
            ids[-1] = 11
        else:
            ids = conversations[conversation].copy()
            if action == "edit":
                ids[rng.randrange(4, len(ids) - 2)] = rng.randrange(200, 210)
            elif action == "extend":
                ids += [100 + conversation] * rng.choice((16, 32)) + [10, 11]
            elif action == "shorten":
                ids = ids[: rng.randrange(5, max(6, len(ids) - 2))] + [10, 11]
        conversations[conversation] = ids
        history = len(ids) - 2 if rng.randrange(5) else 0
        shared = (3,) if rng.randrange(4) == 0 else ()
        yield ids, history, shared


class RenderedTokenizer:
    """Model-free token renderer: role openers, message bodies, generation suffix."""

    @staticmethod
    def apply_chat_template(messages, *, add_generation_prompt=False, **kwargs):
        ids = []
        for message in messages:
            ids += [10, {"system": 12, "user": 13, "assistant": 11}[message["role"]]]
            ids += message["content"] + [14]
        return ids + ([10, 11] if add_generation_prompt else [])

    @staticmethod
    def encode(rendered):
        return rendered


def realistic_requests(seed):
    """Render chat boundaries and use the product's actual shared-lens derivation."""
    from rapid_mlx.speculative.tensorfold_qwen27_server import _prompt_boundaries

    tokenizer = RenderedTokenizer()
    app = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_lock=threading.Lock(),
        system_prefix_len=lambda messages, *args, **kwargs: len(
            tokenizer.apply_chat_template(messages[:1])
        ),
    )
    rng = random.Random(seed)
    conversations = {}
    systems = {}
    for _ in range(24):
        conversation = rng.randrange(4)
        action = rng.choice(("new", "edit", "extend", "repeat", "shorten"))
        if conversation not in conversations or action == "new":
            system = rng.randrange(3)
            messages = [
                {
                    "role": "system",
                    "content": [300 + system] * (520, 1040, 2600)[system],
                },
                {
                    "role": "user",
                    "content": [100 + conversation] * rng.choice((30, 64, 96)),
                },
            ]
        else:
            messages = [
                dict(m, content=m["content"].copy())
                for m in conversations[conversation]
            ]
            if action == "edit":
                body = messages[-1]["content"]
                body[rng.randrange(len(body))] = rng.randrange(200, 210)
            elif action == "extend":
                messages += [
                    {
                        "role": "assistant",
                        "content": [400 + conversation] * rng.choice((8, 16)),
                    },
                    {
                        "role": "user",
                        "content": [100 + conversation] * rng.choice((30, 64, 96)),
                    },
                ]
            elif action == "shorten":
                if len(messages) > 2:
                    messages = messages[:-2]
                else:
                    messages[-1]["content"] = messages[-1]["content"][
                        : rng.randrange(4, 20)
                    ]
        conversations[conversation] = messages
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        history, shared = _prompt_boundaries(app, ids, messages, False)
        system_key = tuple(tokenizer.apply_chat_template(messages[:1]))
        assert systems.setdefault(system_key, shared) == shared
        assert ids[:history] == tokenizer.apply_chat_template(messages)
        yield ids, history, shared


def classify_requests(requests, engine_factory=_checkpoint_engine):
    """Exact skipped-key witnesses; no blanket exemption for later pressure effects.

    Every allowed deficit must name the same token prefix selected by main on
    an earlier request with main_hit < position < resumed_owned. Witnesses expire
    when main removes that state, or HEAD materializes it. Unrelated eviction
    victims never inherit a witness, even when caused by a skipped pin.
    """
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    stores = [
        checkpoints.CheckpointStore(
            8, copier=list, sizer=lambda c: c[0], budget_bytes=800
        )
        for _ in range(2)
    ]
    contexts = {}
    engines = [
        SmallPlannedEngine(),
        engine_factory(SmallPlannedEngine, lambda: stores[1], contexts)(),
    ]
    engines[1].test_histories = contexts
    witnesses = {}
    events = []
    sequence = []
    for number, (ids, history, shared) in enumerate(requests, 1):
        sequence.append((ids.copy(), history, shared))
        previous_head = {tuple(e.tokens) for e in stores[1]._entries}
        prior_witnesses = witnesses.copy()
        owned = getattr(engines[1], "interior_tokens", None)
        main_reuse, selected = runtime_request(
            engines[0], stores[0], ids, history, shared
        )
        head_reuse, _ = runtime_request(engines[1], stores[1], ids, history, shared)
        main_keys = {tuple(e.tokens) for e in stores[0]._entries}
        head_keys = {tuple(e.tokens) for e in stores[1]._entries}
        witnesses = {
            key: value
            for key, value in witnesses.items()
            if key in main_keys and key not in head_keys
        }
        if owned is not None and head_reuse == len(owned) and ids[:head_reuse] == owned:
            for at in selected:
                key = tuple(ids[:at])
                if main_reuse < at < head_reuse and key in main_keys - head_keys:
                    witness = dict(
                        request=number,
                        main_hit=main_reuse,
                        position=at,
                        owned=head_reuse,
                    )
                    witnesses[key] = witness
                    events.append(
                        dict(request=number, kind="skip", key=key, witness=witness)
                    )
        deficits = [("deleted", key) for key in (previous_head & main_keys) - head_keys]
        if head_reuse < main_reuse:
            deficits.append(("reuse", tuple(ids[:main_reuse])))
        for kind, key in deficits:
            events.append(
                dict(
                    request=number,
                    kind=kind,
                    key=key,
                    witness=(prior_witnesses if kind == "reuse" else witnesses).get(
                        key
                    ),
                )
            )
    return events, sequence


def assert_classified(seed, requests):
    events, sequence = classify_requests(requests)
    failures = [event for event in events if event["witness"] is None]
    assert not failures, f"seed={seed}, failures={failures!r}, sequence={sequence!r}"


@pytest.mark.parametrize(
    "seed", range(int(os.environ.get("TENSORFOLD_PROPERTY_SEEDS", "200")))
)
def test_randomized_native_retention_and_reuse(seed):
    assert_classified(seed, synthetic_requests(seed))


@pytest.mark.parametrize(
    "seed", range(int(os.environ.get("TENSORFOLD_REALISTIC_SEEDS", "20")))
)
def test_rendered_chat_native_retention_and_reuse(seed):
    assert_classified(seed, realistic_requests(seed))


@pytest.mark.parametrize("ordinary_hit", [0, 19])
def test_skipped_native_pin_recomputes_from_ordinary_hit(ordinary_hit):
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    contexts = {}
    engine = _checkpoint_engine(SmallPlannedEngine, lambda: store, contexts)()
    engine.test_histories = contexts
    ids = [100] * 96
    ids[0] = ids[3] = ids[-2] = 10
    ids[-1] = 11
    runtime_request(engine, store, ids, 94)
    assert len(engine.interior_tokens) == 83
    if ordinary_hit:
        store.insert(ids[:ordinary_hit], [100], last_prompt=ids)
    shorter = ids[:90] + [10, 11]
    reused, _ = runtime_request(engine, store, shorter, 90, (51,))
    assert reused == ordinary_hit
    assert store.peek(shorter, usable=lambda n: n == 51).pinned
    final = ids[:62] + [10, 11]
    assert runtime_request(engine, store, final, 62)[0] == 51


@pytest.mark.parametrize("seed,skips,reuse,deleted", [(12, 1, 1, 0), (92, 1, 3, 2)])
def test_classifier_requires_exact_skipped_key(seed, skips, reuse, deleted):
    def unprotected(engine_type, getter, contexts):
        guarded = _checkpoint_engine(engine_type, getter, contexts)

        class Unprotected(guarded):
            def _native_selection_without_owned(self, store, stream, owned):
                # Disable only the new guard, reproducing the prior policy.
                return {
                    at
                    for at in super()._native_selection_without_owned(
                        store, stream, owned
                    )
                    if at >= len(owned)
                }

        return Unprotected

    events, _ = classify_requests(synthetic_requests(seed), unprotected)
    assert sum(e["kind"] == "skip" for e in events) == skips
    assert sum(e["kind"] == "reuse" for e in events) == reuse
    assert sum(e["kind"] == "deleted" for e in events) == deleted
    for event in events:
        if event["kind"] == "deleted":
            assert len(event["key"]) in (62, 81)
            assert event["witness"] is None
        else:
            witness = event["witness"]
            assert len(event["key"]) == witness["position"] == 3
            assert witness["main_hit"] == 0 < 3 < witness["owned"]


def test_skipped_selection_after_native_transfer_still_recomputes():
    checkpoints = pytest.importorskip("tensorfold.server.checkpoints")
    store = checkpoints.CheckpointStore(8, copier=list, sizer=lambda c: c[0])
    contexts = {}
    engine = _checkpoint_engine(SmallPlannedEngine, lambda: store, contexts)()
    engine.test_histories = contexts
    ids = [100] * 96
    ids[0] = ids[3] = ids[-2] = 10
    ids[-1] = 11
    runtime_request(engine, store, ids, 94)
    shorter = ids[:90] + [10, 11]
    cached, cache, _ = store.match(shorter, usable=lambda n: n == 83, take=True)
    contexts["test"] = (90, (3,))
    stream = run(engine, shorter, cached=cached, cache=cache, selected=(90,))
    assert engine.calls[-1]["cached_tokens"] == 0
    assert {len(tokens) for tokens, _ in stream.history_checkpoints} >= {3, 90}


@pytest.mark.parametrize(
    "seed", [23, 60, 186, 489, 496, 588, 592, 724, 739, 1109, 1504, 1689]
)
def test_optional_resume_preserves_ordinary_hit_eviction_order(seed):
    # These sequences touch a system hit on main, an optional cut on HEAD,
    # then rotate systems past the three-pin limit under native byte pressure.
    assert_classified(seed, realistic_requests(seed))
