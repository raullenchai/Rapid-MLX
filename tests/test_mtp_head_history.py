# SPDX-License-Identifier: Apache-2.0
"""Execute the generator's head helpers on CPU, without importing Metal."""

import ast
import copy
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "rapid_mlx/spec_decode/mtp/generator.py"
GENERATOR = next(
    n
    for n in ast.parse(SOURCE.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "mtp_generate_step"
)


class _Cache:
    def __init__(self):
        self.rows = []

    @property
    def offset(self):
        return len(self.rows)

    def is_trimmable(self):
        return True

    def trim(self, n):
        assert 0 <= n <= self.offset
        if n:
            del self.rows[-n:]


def _harness(legacy=False, fused=False):
    cache = _Cache()
    quantized = []
    target_drops = []

    def forward(hidden, ids, caches, return_hidden=False):
        cache.rows.extend(zip(hidden[0, :, 0].tolist(), ids[0].tolist()))
        logits = np.eye(64)[ids + 1]
        cascade = np.full(hidden.shape, -99.0)
        return (logits, cascade) if return_hidden else logits

    def greedy(hidden, ids, caches):
        logits, cascade = forward(hidden, ids, caches, return_hidden=True)
        return logits.argmax(axis=-1), cascade

    mx = SimpleNamespace(
        concatenate=np.concatenate,
        array=np.array,
        uint32=np.uint32,
        stream=lambda _: nullcontext(),
        eval=lambda *args: None,
    )
    env = dict(
        mx=mx,
        rebuild_head_history=not legacy,
        head_pending=None,
        head_seed=None,
        mtp_cache=[cache],
        ntoks=1,
        generation_stream=None,
        _is_greedy=True,
        _mtp_supports_fused_greedy=fused,
        _mtp_supports_hidden=True,
        model=SimpleNamespace(mtp_forward=forward, mtp_greedy=greedy),
        quantize_cache_fn=lambda caches: quantized.append(caches),
        _process_and_sample=lambda prev, logits, *a, **kw: (
            logits.argmax(axis=-1),
            None,
            None,
        ),
        logits_processors=None,
        _xtc_cell=None,
        _rollback_draft=lambda n, **kw: target_drops.append((n, kw)),
        _clear_rollback=lambda: None,
        time=time,
        pending_draft_ms=0,
        _snapshot_processor_state=lambda: (),
        _restore_processor_state=lambda state: None,
        _timing_add=lambda *args: None,
        _launch_drafts=lambda result: result,
    )
    # These nested functions use enclosing generator state. Executing them in
    # one namespace preserves that state without importing the MLX runtime.
    names = {
        "_trim_head",
        "_flush_head",
        "_rollback_verify_round",
        "_step_mtp",
        "_step_mtp_chain",
        "_draft_chain_timed",
        "_prefill",
        "_lookup_drafts",
        "_sync_prompt_lookup_history",
    }
    funcs = [
        n for n in GENERATOR.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]

    class _Standalone(ast.NodeTransformer):
        def visit_Nonlocal(self, node):
            return ast.copy_location(ast.Global(names=node.names), node)

    module = _Standalone().visit(ast.Module(body=funcs, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), env)
    return env, cache, quantized, target_drops


def _settle_verify(env, accepted, emitted=None, *, lookup=False):
    """Execute the actual post-verify rebuild block, including terminal caps."""
    block = next(
        n
        for n in ast.walk(GENERATOR)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "rebuild_head_history"
        and any(
            isinstance(c, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "kept" for t in c.targets)
            for c in n.body
        )
    )
    env.update(
        pending_is_prompt_lookup=lookup,
        k_len=3,
        accepted_count=accepted,
        emissions=[None] * (accepted if emitted is None else emitted),
        hidden=np.array([[[110.0], [120.0], [130.0], [140.0]]]),
        y=np.array([11]),
        drafts_arr=np.array([12, 13, 14]),
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), env)


def _generation_harness(
    max_tokens, depths, *, bootstrap=False, prefetch=False, legacy=False
):
    """Run the real park/verify loop with the existing CPU head and target +1."""
    env, cache, _, _ = _harness(legacy=legacy)
    picks = iter(depths)
    env["mx"].__dict__.update(
        int32=np.int32,
        stack=np.stack,
        async_eval=lambda *args: None,
        clear_cache=lambda: None,
    )

    def backbone(ids, prev, n_predict=1, **kwargs):
        toks = ids[-n_predict:] + 1
        lps = np.zeros((n_predict, 64))
        return toks, lps, lps, ids[None, :, None] * 10.0, prev, [()] * n_predict

    env.update(
        ntoks=0,
        y=np.array([11 if bootstrap else 10]),
        server_bootstrap=(np.array([[[100.0]]]), np.zeros(64)) if bootstrap else None,
        max_tokens=max_tokens,
        prev_tokens=None,
        pending_drafts=None,
        pending_is_prompt_lookup=False,
        _request_depth=None,
        _prompt_lookup_index=None,
        _next_depth=lambda: next(picks),
        _lookup_proposal=lambda: None,
        _admit_mtp_depth=lambda k: k,
        _step_backbone=backbone,
        _remember_generated=lambda tok: None,
        _record_round=lambda *args, **kwargs: None,
        _gate_cost=lambda ms, rows: ms,
        _copy_draft_gate=SimpleNamespace(
            observe=lambda **kwargs: None, allow=lambda: True
        ),
        _uniform=lambda **kwargs: np.zeros(kwargs["shape"]),
        accept_counter=SimpleNamespace(
            record_attempt=lambda: None,
            record_verify=lambda *args: None,
            record_accept=lambda **kwargs: None,
            record_reject=lambda: None,
        ),
        stop_tokens=None,
        may_run_ahead=lambda: prefetch,
        _CACHE_CLEAR_INTERVAL=256,
        last_cache_block=0,
    )
    planner = copy.deepcopy(
        next(n for n in GENERATOR.body if getattr(n, "name", None) == "_plan_next")
    )
    exec(compile(ast.Module(body=[planner], type_ignores=[]), str(SOURCE), "exec"), env)
    start = next(
        i
        for i, n in enumerate(GENERATOR.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_pipeline_parks" for t in n.targets
        )
    )
    body = copy.deepcopy(GENERATOR.body[start:])

    class SharedState(ast.NodeTransformer):
        # The existing head helpers and loop must share enclosing state.
        def visit_Nonlocal(self, node):
            return ast.copy_location(ast.Global(names=node.names), node)

        def visit_AnnAssign(self, node):
            return ast.copy_location(
                ast.Assign(targets=[node.target], value=node.value), node
            )

    class LoopStores(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                self.names.add(node.id)

        def visit_FunctionDef(self, node):
            pass  # Nested helpers keep their own local variables.

    body = [SharedState().visit(n) for n in body]
    stores = LoopStores()
    for node in body:
        stores.visit(node)
    runner = ast.parse("def run(): pass").body[0]
    runner.body = [ast.Global(names=sorted(stores.names)), *body]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        env,
    )
    return env, cache


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
@pytest.mark.parametrize("depth", [0, 1])
def test_server_bootstrap_counts_the_already_delivered_token(max_tokens, depth):
    direct, _ = _generation_harness(max_tokens, [depth] * max_tokens)
    server, _ = _generation_harness(max_tokens, [depth] * max_tokens, bootstrap=True)
    direct_tokens = [tok for tok, _, _ in direct["run"]()]
    server_tokens = [tok for tok, _, _ in server["run"]()]
    assert direct_tokens == list(range(11, 11 + max_tokens))
    assert len(server_tokens) == max_tokens - 1
    assert 11 not in server_tokens
    assert server_tokens == direct_tokens[1:][: max_tokens - 1]
    # The scheduler delivers 11 before entering this generator.
    assert [11, *server_tokens] == direct_tokens[:max_tokens]
    # The old scheduler handoff starts from that sample without suppression;
    # its extra token is truncated by the batch's outer request limit.
    legacy, _ = _generation_harness(max_tokens, [depth] * max_tokens)
    legacy["y"] = np.array([11])
    legacy_tokens = [tok for tok, _, _ in legacy["run"]()]
    assert len(legacy_tokens) == max_tokens
    assert server_tokens == legacy_tokens[: max_tokens - 1]


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("parking_route", ["zero_depth", "admission"])
def test_park_transitions_append_each_committed_head_pair_once(prefetch, parking_route):
    # park->park, park->mtp, mtp->park, then repeat to observe resumed chains.
    env, cache = _generation_harness(10, [0, 1, 0, 0, 1, 0, 0], prefetch=prefetch)
    if parking_route == "admission":
        # Positive desired depth which the real planner cannot admit exercises
        # plan == 'park', whereas zero depth exercises parks_next directly.
        env["_next_depth"] = lambda: 1
        env["_admit_mtp_depth"] = lambda k: k if env["ntoks"] in (2, 6) else 0
    forwards = []
    trimmed = []
    chain_boundaries = []
    target_boundaries = []
    delivered = []
    forward = env["model"].mtp_forward
    trim = cache.trim
    chain = env["_draft_chain_timed"]
    backbone = env["_step_backbone"]

    def observed_backbone(*args, **kwargs):
        target_boundaries.append(len(delivered))
        return backbone(*args, **kwargs)

    def counted_forward(hidden, ids, caches, **kwargs):
        forwards.extend(zip(hidden[0, :, 0].tolist(), ids[0].tolist()))
        return forward(hidden, ids, caches, **kwargs)

    def counted_trim(n):
        if n:
            trimmed.extend(cache.rows[-n:])
        trim(n)

    def checked_chain(*args, **kwargs):
        assert env["head_pending"] is None
        assert cache.offset == env["ntoks"] - 1
        assert cache.rows == [(10.0 * (t - 1), t) for t in range(11, 10 + env["ntoks"])]
        chain_boundaries.append(env["ntoks"])
        return chain(*args, **kwargs)

    env["model"].mtp_forward = counted_forward
    cache.trim = counted_trim
    env["_draft_chain_timed"] = checked_chain
    env["_step_backbone"] = observed_backbone
    for tok, _, _ in env["run"]():
        delivered.append(tok)
    assert delivered == list(range(11, 21))
    # Prove that the requested prefetch mode actually changed launch timing.
    assert target_boundaries[:2] == ([0, 0] if prefetch else [0, 1])
    assert chain_boundaries == [2, 6]
    expected = [(10.0 * (t - 1), t) for t in range(11, 21)]
    assert cache.rows == expected
    # Verify deliberately trims provisional draft rows and rebuilds them.
    # Every other mtp_forward row must commit exactly once, including parks.
    assert Counter(forwards) == Counter(expected) + Counter(trimmed)
    assert trimmed == [(110.0, 12), (150.0, 16)]


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("prior_mtp", [False, True])
@pytest.mark.parametrize("next_park", [False, True])
def test_lookup_transitions_rebuild_true_committed_pairs(
    accepted, prior_mtp, next_park
):
    env, cache = _generation_harness(14, [])
    lookup_at = 3 if prior_mtp else 1
    lookup_end = lookup_at + accepted + 1
    env["_next_depth"] = lambda: 0 if next_park and env["ntoks"] == lookup_end else 1

    def proposal():
        if env["ntoks"] != lookup_at:
            return None
        drafts = list(range(11 + lookup_at, 14 + lookup_at))
        if accepted < 3:
            drafts[accepted] = 50
        return drafts, 2

    env["_lookup_proposal"] = proposal
    forwards, trimmed, boundaries, parks = [], [], [], []
    forward = env["model"].mtp_forward
    trim = cache.trim
    chain = env["_draft_chain_timed"]
    backbone = env["_step_backbone"]

    def counted_forward(hidden, ids, caches, **kwargs):
        forwards.extend(zip(hidden[0, :, 0].tolist(), ids[0].tolist()))
        return forward(hidden, ids, caches, **kwargs)

    def counted_trim(n):
        if n:
            trimmed.extend(cache.rows[-n:])
        trim(n)

    def checked_chain(*args, **kwargs):
        pending = env["head_pending"]
        deferred = (
            []
            if pending is None
            else list(zip(pending[0][0, :, 0].tolist(), pending[1].tolist()))
        )
        assert cache.offset + len(deferred) == env["ntoks"] - 1
        assert cache.rows + deferred == [
            (10.0 * (t - 1), t) for t in range(11, 10 + env["ntoks"])
        ]
        assert kwargs.get("cache_commit") is None
        boundaries.append(env["ntoks"])
        return chain(*args, **kwargs)

    def observed_backbone(ids, *args, **kwargs):
        if ids.size == 1:
            parks.append(env["ntoks"])
        return backbone(ids, *args, **kwargs)

    env["model"].mtp_forward = counted_forward
    cache.trim = counted_trim
    env["_draft_chain_timed"] = checked_chain
    env["_step_backbone"] = observed_backbone
    assert [tok for tok, _, _ in env["run"]()] == list(range(11, 25))
    # Park depth is selected before the parked token advances ntoks, so the
    # zero-depth choice also schedules a second park before MTP resumes.
    assert lookup_end + 2 * int(next_park) in boundaries
    assert (lookup_end in parks) == next_park
    assert (1 in boundaries) == prior_mtp
    expected = [(10.0 * (t - 1), t) for t in range(11, 25)]
    assert cache.rows == expected
    # Removing provisional rows leaves each committed pair exactly once.
    assert Counter(forwards) == Counter(expected) + Counter(trimmed)


@pytest.mark.parametrize(
    "accepted,expected",
    [
        (0, [(110, 12), (120, 13), (130, 14), (140, 15), (150, 16)]),
        (1, [(110, 11), (120, 13), (130, 14), (140, 15), (150, 16), (160, 17)]),
        (
            3,
            [
                (110, 11),
                (120, 12),
                (130, 13),
                (130, 14),
                (140, 15),
                (150, 16),
                (160, 17),
            ],
        ),
    ],
)
def test_legacy_lookup_to_mtp_retains_original_history(accepted, expected):
    env, cache = _generation_harness(8, [], legacy=True)
    env["_next_depth"] = lambda: 1

    def proposal():
        if env["ntoks"] != 1:
            return None
        drafts = [12, 13, 14]
        if accepted < 3:
            drafts[accepted] = 50
        return drafts, 2

    env["_lookup_proposal"] = proposal
    assert [tok for tok, _, _ in env["run"]()] == list(range(11, 19))
    assert cache.rows == expected
    assert env["head_seed"] is None
    assert env["head_pending"] is None


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("cap", [2, 3, 4, 5])
def test_lookup_terminal_flush_in_generation_loop(accepted, cap):
    env, cache = _generation_harness(cap, [])
    env["_next_depth"] = lambda: 1
    drafts = [12, 13, 14]
    if accepted < 3:
        drafts[accepted] = 50
    env["_lookup_proposal"] = lambda: (drafts, 2) if env["ntoks"] == 1 else None
    tokens = [tok for tok, _, _ in env["run"]()]
    assert tokens == list(range(11, 11 + cap))
    assert cache.rows == [(10.0 * (t - 1), t) for t in tokens]
    assert env["head_pending"] is None


@pytest.mark.parametrize("eos", [12, 13, 14])
def test_lookup_eos_flushes_only_committed_prefix(eos):
    env, cache = _generation_harness(10, [])
    env["_next_depth"] = lambda: 1
    env["_lookup_proposal"] = lambda: ([12, 13, 14], 2)
    env["stop_tokens"] = {eos}
    tokens = [tok for tok, _, _ in env["run"]()]
    assert tokens == list(range(11, eos + 1))
    assert cache.rows == [(10.0 * (t - 1), t) for t in tokens]
    assert env["head_pending"] is None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("fused", [False, True])
def test_committed_head_rows(legacy, accepted, fused):
    env, cache, quantized, drops = _harness(legacy, fused)
    chain = env["_step_mtp_chain"]
    chain(np.array([[[100.0]]]), np.array([11]), None, 3)
    _settle_verify(env, accepted)
    if accepted < 3:
        env["_rollback_verify_round"](3 - accepted, verify_size=4)
        assert drops == [(3 - accepted, {"verify_size": 4})]
    else:
        assert drops == []
    commit = None
    if legacy and accepted == 3:
        commit = (np.array([[[130.0]]]), np.array([14]))
    calls = []
    original = env["_step_mtp"]

    def observe(hidden, tok, prev, **kwargs):
        # Before each chain's first *new* row, physical KV plus deferred
        # committed rows must equal emitted tokens minus one.
        pending = kwargs.get("cache_commit")
        calls.append(cache.offset + (0 if pending is None else pending[1].size))
        return original(hidden, tok, prev, **kwargs)

    env["_step_mtp"] = observe
    result = chain(
        np.array([[[110.0 + accepted * 10]]]),
        np.array([12 + accepted]),
        None,
        3,
        cache_commit=commit,
    )
    assert result[0][0] == 13 + accepted
    assert calls[0] == accepted + (0 if legacy and accepted < 3 else 1)
    assert quantized
    if not legacy:
        assert cache.rows[: accepted + 2] == [
            (float(10 * (t - 1)), t) for t in range(11, 13 + accepted)
        ]
    elif accepted >= 2:
        # Old accepted rows retain predicted cascade hiddens.
        assert cache.rows[1][0] == -99.0


@pytest.mark.parametrize("accepted,emitted", [(0, 0), (1, 1), (2, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("lookup", [False, True])
def test_rebuild_flushes_terminal_or_park_handoff(accepted, emitted, lookup):
    env, cache, quantized, _ = _harness()
    if lookup:
        env["head_seed"] = np.array([[[100.0]]])
    else:
        env["_step_mtp_chain"](np.array([[[100.0]]]), np.array([11]), None, 3)
    _settle_verify(env, accepted, emitted, lookup=lookup)
    assert cache.offset == 0
    env["_flush_head"]()
    assert cache.rows == [(float(10 * (t - 1)), t) for t in range(11, 12 + emitted)]
    assert env["head_pending"] is None
    assert len(quantized) == (1 if lookup else 4)


def test_rebuild_park_and_resume():
    env, cache, _, _ = _harness()
    chain = env["_step_mtp_chain"]
    chain(np.array([[[100.0]]]), np.array([11]), None, 3)
    _settle_verify(env, 3)
    env.update(last_committed_hidden=np.array([[[140.0]]]), y=np.array([15]))
    block = next(
        n
        for n in ast.walk(GENERATOR)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "rebuild_head_history and head_pending is not None"
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), env)
    env["_flush_head"]()
    assert cache.rows == [(float(10 * (t - 1)), t) for t in range(11, 16)]
    # One intervening parked step predicts 16 from true hidden(15).
    env["head_pending"] = (np.array([[[150.0]]]), np.array([16]))
    env["_flush_head"]()
    env["ntoks"] = 7
    assert cache.offset == 6  # seven emitted tokens, latest row not appended
    chain(np.array([[[160.0]]]), np.array([17]), None, 3)
    assert cache.rows[:7] == [(float(10 * (t - 1)), t) for t in range(11, 18)]


@pytest.mark.parametrize("lookup", [False, True])
def test_failed_chain_restores_deferred_rows(lookup):
    env, cache, _, _ = _harness()
    assert "_draft_chain_timed" in env
    count = 4 if lookup else 1
    expected = [(float(10 * (t - 1)), t) for t in range(11, 11 + count)]
    pending = (
        np.array([[[h] for h, _ in expected]]),
        np.array([t for _, t in expected]),
    )
    env["head_pending"] = pending
    step = env["_step_mtp"]
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected draft fault")
        return step(*args, **kwargs)

    env["_step_mtp"] = fail_second
    with pytest.raises(RuntimeError, match="injected draft fault"):
        env["_draft_chain_timed"](np.array([[[110.0]]]), np.array([12]), None, 3)
    assert cache.rows == expected
    assert env["head_pending"] is None


def test_prefill_shift_and_first_chain():
    env, cache, _, _ = _harness()
    env["mx"].clear_cache = lambda: None
    env.update(
        model_cache=[],
        prefill_step_size=3,
        _target_forward=lambda ids, **kwargs: (None, ids[..., None] * 10),
    )
    remaining = env["_prefill"](np.arange(10, 18), None)
    assert remaining.tolist() == [17]
    assert cache.rows == [(float(10 * (t - 1)), t) for t in range(11, 18)]
    assert cache.offset == 7
    env["_step_mtp_chain"](np.array([[[170.0]]]), np.array([18]), None, 3)
    assert cache.rows[:8] == [(float(10 * (t - 1)), t) for t in range(11, 19)]


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("lookup", [False, True])
def test_terminal_bonus_or_residual_is_committed(accepted, lookup):
    env, cache, _, _ = _harness()
    if lookup:
        env["head_seed"] = np.array([[[100.0]]])
    else:
        env["_step_mtp_chain"](np.array([[[100.0]]]), np.array([11]), None, 3)
    _settle_verify(env, accepted, lookup=lookup)
    env.update(
        round_done=True,
        eos_cut=False,
        emissions=[(t,) for t in range(12, 13 + accepted)],
    )
    block = next(
        n
        for n in ast.walk(GENERATOR)
        if isinstance(n, ast.If)
        and "len(emissions) == accepted_count + 1" in ast.unparse(n.test)
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), env)
    env["_flush_head"]()
    assert cache.rows == [(float(10 * (t - 1)), t) for t in range(11, 13 + accepted)]


@pytest.mark.parametrize("sync_failure", [False, True])
@pytest.mark.parametrize("lookup", [False, True])
def test_verify_exception_keeps_true_y(sync_failure, lookup):
    env, cache, _, drops = _harness()
    cache.rows = [(90.0, 10)]
    if lookup:
        env["head_seed"] = np.array([[[100.0]]])
    else:
        env["_step_mtp_chain"](np.array([[[100.0]]]), np.array([11]), None, 3)
    handlers = [
        n
        for n in ast.walk(GENERATOR)
        if isinstance(n, ast.ExceptHandler)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "head_pending"
                for t in child.targets
            )
            and ast.unparse(child.value) == "(head_seed, y)"
            for child in ast.walk(n)
        )
    ]
    assert len(handlers) == 2
    handler = next(
        h
        for h in handlers
        if ("_rollback_verify_round" in ast.unparse(h)) == sync_failure
    )
    env.update(
        y=np.array([11]),
        k_len=3,
        accepted_count=0,
        pending_is_prompt_lookup=lookup,
        processor_boundary=(),
    )
    fault = ast.parse('raise RuntimeError("injected verify fault")').body
    block = ast.Try(body=fault, handlers=[handler], orelse=[], finalbody=[])
    with pytest.raises(RuntimeError, match="injected verify fault"):
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                str(SOURCE),
                "exec",
            ),
            env,
        )
    assert cache.rows == [(90.0, 10), (100.0, 11)]
    assert bool(drops) == sync_failure


@pytest.mark.parametrize("value,expected", [(None, True), ("0", True), ("1", False)])
@pytest.mark.parametrize("supported", [False, True])
def test_history_default_and_emergency_rollback(
    monkeypatch, value, expected, supported
):
    if value is None:
        monkeypatch.delenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY", value)
    assignment = next(
        n
        for n in GENERATOR.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "rebuild_head_history"
            for t in n.targets
        )
    )
    env = {"os": os, "model": SimpleNamespace(mtp_head_prefill_capture=supported)}
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"),
        env,
    )
    assert env["rebuild_head_history"] is (expected and supported)


def test_repeated_rejections_do_not_accumulate_missing_head_rows():
    env, cache, _, drops = _harness()
    for round_index in range(5):
        # Each reject emits one target residual. Prior rows are deferred into
        # the first call; settling before observation gives the logical offset.
        env["_flush_head"]()
        assert cache.offset == round_index
        env["_step_mtp_chain"](np.array([[[100.0]]]), np.array([11]), None, 3)
        _settle_verify(env, 0)
        env["_rollback_verify_round"](3, verify_size=4)
    env["_flush_head"]()
    assert cache.offset == 5
    assert drops == [(3, {"verify_size": 4})] * 5
