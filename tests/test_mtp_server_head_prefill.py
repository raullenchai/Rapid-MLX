# SPDX-License-Identifier: Apache-2.0
"""CPU chunk/lifecycle checks and native scheduler checks for head capture."""

import ast
import importlib.util
import logging
import math
import os
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "rapid_mlx/spec_decode/mtp/server_head_prefill.py"
spec = importlib.util.spec_from_file_location("head_capture_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Model:
    def __init__(self):
        self.rows = []

    def make_mtp_cache(self):
        return [SimpleNamespace(state=None)]

    def __call__(self, inputs, cache=None, return_hidden=False):
        hidden = inputs[..., None] * 10
        return (inputs, hidden) if return_hidden else inputs

    def mtp_forward(self, hidden, ids, cache):
        self.rows.extend(zip(hidden[0, :, 0].tolist(), ids[0].tolist()))


def test_capture_pairs_across_chunks_and_leaves_prompt_end_hidden():
    model = Model()
    quantized = []
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    capture = module.HeadPrefillCapture(model, mx, quantized.append)
    for chunk in ([10, 11, 12], [13], [14, 15], [16]):
        capture(np.array([chunk]), cache=[])
    assert model.rows == [(10 * t, t + 1) for t in range(10, 16)]
    assert capture.hidden.tolist() == [[[160]]]
    assert len(quantized) == 4


def test_prefix_hit_only_fills_computed_suffix():
    model = Model()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    capture = module.HeadPrefillCapture(model, mx, lambda c: None)
    capture(np.array([[15]]), cache=[])
    assert model.rows == []
    capture(np.array([[16, 17]]), cache=[])
    assert model.rows == [(150, 16), (160, 17)]


def test_instance_hook_restore_and_ineligible_batch():
    model = Model()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)

    class Batch:
        def __init__(self):
            self.model = model
            self.uids = [7]
            self.prompt_cache = []

        def prompt(self, tokens):
            self.model(np.array(tokens), cache=[])

    pb = Batch()
    captures, close = module.install_prompt_capture(
        SimpleNamespace(_prompt_batch=pb), model, mx, lambda u: True, lambda c: None
    )
    pb.prompt([[10, 11]])
    assert pb.model is model
    assert 7 in captures
    pb.uids = [7, 8]
    pb.prompt([[12], [13]])
    assert captures == {}
    assert model.rows == [(100, 11)]
    close()
    assert type(pb) is Batch


def test_generate_uses_prompt_end_hidden_and_restores_model():
    model = Model()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)

    class Batch:
        def __init__(self):
            self.model = model
            self.uids = [7]
            self.tokens = [[]]
            self.prompt_cache = []

        def prompt(self, tokens):
            self.tokens[0].extend(tokens[0])
            self.model(np.array(tokens), cache=[])

        def generate(self, tokens):
            self.prompt([tokens[0][:-1]])
            self.model(np.array([[tokens[0][-1]]]), cache=[])
            self.tokens[0].append(tokens[0][-1])
            result = SimpleNamespace(
                model=self.model,
                tokens=self.tokens,
                _next_tokens=np.array([18]),
            )
            self.uids = []
            return result

    pb = Batch()
    captures, close = module.install_prompt_capture(
        SimpleNamespace(_prompt_batch=pb), model, mx, lambda u: True, lambda c: None
    )
    pb.prompt([[10, 11]])
    generation = pb.generate([[12, 13]])
    assert generation.model is model
    assert pb.model is model
    assert model.rows == [(100, 11), (110, 12), (120, 13)]
    assert captures[7].hidden.tolist() == [[[130]]]
    assert captures[7].first_token == 18
    assert captures[7].prompt_length == 4
    close()
    assert captures == {}


def test_bootstrap_reuses_sample_without_target_or_sampler_call():
    import ast
    import textwrap

    source = SOURCE.with_name("generator.py")
    tree = ast.parse(source.read_text())
    generator = next(
        n for n in tree.body if getattr(n, "name", None) == "mtp_generate_step"
    )
    park = next(n for n in generator.body if getattr(n, "name", None) == "_park_step")
    # Preserve real closure behavior; the second call goes through the target.
    factory = ast.parse(
        textwrap.dedent(
            """\
            def factory(bootstrap, prev_tokens, _step_backbone, mx):
                pass
            """
        )
    ).body[0]
    factory.body = [park, ast.Return(value=ast.Name(id="_park_step", ctx=ast.Load()))]
    env = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
            str(source),
            "exec",
        ),
        env,
    )
    hidden = np.array([[[170]]])
    lp = np.array([0.0])
    calls = []

    def target(ids, prev, **kwargs):
        calls.append(ids.tolist())
        return [19], [lp], None, np.array([[[180]]]), prev, None

    mx = SimpleNamespace(uint32=np.uint32, async_eval=lambda *a: None)
    park_step = env["factory"]((hidden, lp), "prompt history", target, mx)
    token, actual_lp, actual_hidden, prev = park_step(np.array([18]))
    assert token == 18
    assert actual_lp is lp
    assert actual_hidden is hidden
    assert prev == "prompt history"
    assert calls == []
    assert park_step(np.array([18]))[0] == 19
    assert calls == [[18]]


class EmptyKV:
    def empty(self):
        return True

    @property
    def state(self):
        raise AssertionError("empty KVCache.state must not be read")


def test_single_token_capture_never_reads_unallocated_head_state():
    model = Model()
    model.make_mtp_cache = lambda: [EmptyKV()]
    evaluated = []
    mx = SimpleNamespace(
        concatenate=np.concatenate, eval=lambda *a: evaluated.extend(a)
    )
    capture = module.HeadPrefillCapture(model, mx, lambda c: None)
    capture(np.array([[15]]), cache=[])
    assert model.rows == []
    assert capture.hidden.tolist() == [[[150]]]
    assert len(evaluated) == 1


def test_restored_target_cache_captures_suffix_generate_handoff():
    """A prefix hit captures its computed suffix without reading empty KV."""
    model = Model()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)

    class Batch:
        def __init__(self):
            self.model = model
            self.uids = [7]
            # Full-attention absolute offset includes the restored prefix;
            # row history contains only the subsequently processed suffix.
            self.prompt_cache = [SimpleNamespace(offset=23, empty=lambda: False)]
            self.tokens = [[15]]

        def generate(self, tokens):
            self.model(np.array(tokens), cache=self.prompt_cache)
            return SimpleNamespace(
                model=self.model, tokens=self.tokens, _next_tokens=np.array([18])
            )

    pb = Batch()
    captures, close = module.install_prompt_capture(
        SimpleNamespace(_prompt_batch=pb), model, mx, lambda u: True, lambda c: None
    )
    generation = pb.generate([[16]])
    assert captures[7].hidden.tolist() == [[[160]]]
    assert captures[7].first_token == 18
    assert captures[7].prompt_length == 1
    assert generation.model is model
    assert pb.model is model
    assert model.rows == []
    close()


def test_scheduler_handoff_attaches_private_cache_and_bootstrap():
    """Execute the real handoff expressions with a captured full prompt."""
    import ast

    scheduler = SOURCE.parents[2] / "scheduler.py"
    tree = ast.parse(scheduler.read_text())
    handoff = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "mtp_generate_step"
        and any(k.arg is None for k in n.keywords)
    )
    model = Model()
    model.make_mtp_cache = lambda: [SimpleNamespace(offset=0, state=None)]
    forward = model.mtp_forward

    def append(hidden, ids, cache):
        forward(hidden, ids, cache)
        cache[0].offset += ids.shape[1]

    model.mtp_forward = append
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    capture = module.HeadPrefillCapture(model, mx, lambda c: None)
    capture(np.array([[10, 11, 12]]))
    capture(np.array([[13]]))
    assert capture.cache[0].offset == 4 - 1
    assert capture.hidden.tolist() == [[[130]]]
    backbone = [object()]
    lp = object()
    env = dict(
        head_capture=capture, gb=SimpleNamespace(prompt_cache=backbone), first_lp=lp
    )
    # Use the scheduler's actual assignments and generator arguments.
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "head_kwargs" for t in n.targets)
    )
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(scheduler), "exec"),
        env,
    )
    keyword = next(k for k in handoff.keywords if k.arg == "prompt_cache")
    cache = eval(compile(ast.Expression(keyword.value), str(scheduler), "eval"), env)
    assert cache == backbone + capture.cache
    assert backbone == [cache[0]]  # shared prefix list never gains head KV
    hidden, actual_lp = env["head_kwargs"]["server_bootstrap"]
    assert hidden.tolist() == [[[130]]]  # same seed as direct final prompt call
    assert actual_lp is lp
    env["head_capture"] = None
    assert (
        eval(compile(ast.Expression(keyword.value), str(scheduler), "eval"), env)
        is backbone
    )


def test_scheduler_capture_gate_accepts_text_wrapper_and_rollback(monkeypatch):
    import ast
    import os

    scheduler = SOURCE.parents[2] / "scheduler.py"
    tree = ast.parse(scheduler.read_text())
    gate = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "RAPID_MLX_MTP_LEGACY_HEAD_HISTORY" in ast.unparse(n.test)
    )
    inner = SimpleNamespace(mtp_head_prefill_capture=True)
    env = dict(
        os=os, mtp_model=inner, batch_gen=SimpleNamespace(_prompt_batch=object())
    )
    for model in (inner, SimpleNamespace(language_model=inner)):
        env["model"] = model
        monkeypatch.delenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY", raising=False)
        assert eval(compile(ast.Expression(gate.test), str(scheduler), "eval"), env)
        monkeypatch.setenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY", "1")
        assert not eval(compile(ast.Expression(gate.test), str(scheduler), "eval"), env)
    monkeypatch.delenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY")
    env["model"] = SimpleNamespace(language_model=object())
    assert not eval(compile(ast.Expression(gate.test), str(scheduler), "eval"), env)


def test_prompt_failure_restores_instance_and_discards_capture():
    import pytest

    model = Model()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)

    class Batch:
        def __init__(self):
            self.model = model
            self.uids = [7]

        def prompt(self, tokens):
            self.model(np.array(tokens))
            raise RuntimeError("prompt failed")

    pb = Batch()
    captures, close = module.install_prompt_capture(
        SimpleNamespace(_prompt_batch=pb), model, mx, lambda u: True, lambda c: None
    )
    with pytest.raises(RuntimeError, match="prompt failed"):
        pb.prompt([[10, 11]])
    assert pb.model is model
    assert captures == {}
    close()


def test_bootstrap_delivery_is_suppressed_once():
    import ast

    tree = ast.parse(SOURCE.with_name("generator.py").read_text())
    gate = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "suppress_bootstrap"
    )
    fn = ast.FunctionDef(
        name="deliver",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[ast.Global(names=["suppress_bootstrap"]), gate],
        decorator_list=[],
    )
    env = dict(suppress_bootstrap=True, main_tok_id=18, main_lp=None)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        env,
    )
    assert list(env["deliver"]()) == []
    assert list(env["deliver"]()) == [(18, None, False)]


# Run the complete installer on Linux without importing the scheduler's Metal
# boot hooks. Keep its closure and source locations intact; the native variant
# below runs the same cases through the normally imported scheduler as well.
@pytest.fixture(params=["cpu", pytest.param("native", marks=pytest.mark.requires_mlx)])
def installed_capture(request, monkeypatch):
    from rapid_mlx.tuning import mtp as tuning_mtp

    # Lifecycle fixtures provide NumPy model/cache facts, with no device profile.
    monkeypatch.setattr(tuning_mtp, "prefill_block_size", lambda model: 0)
    mx = SimpleNamespace(
        array=np.array,
        uint32=np.uint32,
        concatenate=np.concatenate,
        eval=lambda *a: None,
    )
    if request.param == "native":
        pytest.importorskip("mlx.core")
        from rapid_mlx import scheduler

        monkeypatch.setattr(scheduler, "mx", mx)
        install = scheduler._install_mtp_vendored
    else:
        source = SOURCE.parents[2] / "scheduler.py"
        tree = ast.parse(source.read_text())
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_install_mtp_vendored", "_ensure_request_mtp_counter")
        ]
        env = dict(
            __package__="rapid_mlx",
            Any=Any,
            mx=mx,
            math=math,
            os=os,
            deque=deque,
            logger=logging.getLogger(__name__),
        )
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
            env,
        )
        install = env["_install_mtp_vendored"]

    from rapid_mlx.spec_decode.mtp import server_head_prefill
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    seen = []
    generator = ModuleType("rapid_mlx.spec_decode.mtp.generator")
    generator._effective_prompt_lookup_policy = lambda model: PromptLookupPolicy(
        enabled_by_default=True
    )
    generator._prompt_lookup_is_enabled = lambda *a, **kw: True

    def generate(**kwargs):
        seen.append(kwargs)
        return (row for row in [(19, np.array([0.0]), False)])

    generator.mtp_generate_step = generate
    monkeypatch.setitem(sys.modules, generator.__name__, generator)
    quantized = []
    mlx_generate = ModuleType("mlx_lm.generate")
    mlx_generate.maybe_quantize_kv_cache = lambda cache, **kw: quantized.append(
        (cache, kw)
    )
    monkeypatch.setitem(sys.modules, mlx_generate.__name__, mlx_generate)
    monkeypatch.delenv("RAPID_MLX_MTP_LEGACY_HEAD_HISTORY", raising=False)

    # Same bookkeeping contract as test_mtp_cli_wiring._StubBatchGen, with
    # NumPy arrays so the whole installer can execute on the no-Metal shard.
    class GenerationBatch:
        def __init__(self):
            self.uids = []
            self.tokens = []
            self.logits_processors = []
            self.prompt_cache = [object()]
            self.max_tokens = [16]
            self._next_tokens = None
            self._next_logprobs = []
            self.orig_step_calls = 0

        def _step(self):
            self.orig_step_calls += 1
            if self._next_tokens is None:
                return [], []
            tokens = self._next_tokens.tolist()
            self.tokens[0].extend(tokens)
            self._next_tokens = np.array([999], dtype=np.uint32)
            self._next_logprobs = [np.array([0.0])]
            return tokens, self._next_logprobs

        def next(self):
            return []

    model = Model()
    model.mtp = object()
    model.mtp_head_prefill_capture = True
    gb = GenerationBatch()

    class PromptBatch:
        def __init__(self):
            self.model = model
            self.uids = [7]
            self.tokens = [[]]

        def prompt(self, tokens):
            self.model(np.array(tokens))
            self.tokens[0].extend(tokens[0])

        def generate(self, tokens):
            self.model(np.array(tokens))
            self.tokens[0].extend(tokens[0])
            self.uids = []
            if self.fail_generate:
                raise RuntimeError("generation failed after uid transfer")
            return SimpleNamespace(
                model=self.model,
                tokens=self.tokens,
                _next_tokens=np.array([18]),
            )

        fail_generate = False

    pb = PromptBatch()
    batch = SimpleNamespace(_generation_batch=gb, _prompt_batch=pb)
    captures = {}
    original_install = server_head_prefill.install_prompt_capture

    def observe_install(*args):
        actual, close = original_install(*args)
        captures["actual"] = actual
        return actual, close

    monkeypatch.setattr(server_head_prefill, "install_prompt_capture", observe_install)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0.0),
        prompt_token_ids=[1, 2, 10, 11, 12],
    )
    assert install(
        batch,
        model=model,
        requests={"req-7": req},
        uid_to_request_id={7: "req-7"},
        disable_auto_k=True,
        # Fix the harness depth without probing MLX hardware for tuning.
        explicit_depth=True,
    )
    try:
        yield SimpleNamespace(
            batch=batch,
            gb=gb,
            pb=pb,
            model=model,
            captures=captures["actual"],
            seen=seen,
            quantized=quantized,
            original_class=PromptBatch,
        )
    finally:
        batch._mtp_vendored_teardown()
        assert type(pb) is PromptBatch
        assert captures["actual"] == {}


def _finish_prompt(harness):
    harness.pb.prompt([[10, 11]])
    generation = harness.pb.generate([[12]])
    harness.gb.uids = [7]
    harness.gb.tokens = generation.tokens
    harness.gb._next_tokens = generation._next_tokens
    harness.gb._next_logprobs = [np.array([0.0])]
    return harness.captures[7]


def test_installed_capture_handoff_trims_lookup_history(installed_capture):
    h = installed_capture
    capture = _finish_prompt(h)
    assert capture.first_token == 18
    assert capture.prompt_length == 3
    assert h.pb.model is h.model
    assert h.model.rows == [(100, 11), (110, 12)]
    assert h.quantized == [
        (capture.cache, dict(quantized_kv_start=0, kv_group_size=64, kv_bits=None)),
        (capture.cache, dict(quantized_kv_start=0, kv_group_size=64, kv_bits=None)),
    ]
    assert h.gb._step()[0] == [18]
    assert h.gb.orig_step_calls == 0
    assert h.gb.tokens == [[10, 11, 12, 18]]
    assert h.captures == {}
    kwargs = h.seen[0]
    assert kwargs["prompt_cache"] == h.gb.prompt_cache + capture.cache
    hidden, lp = kwargs["server_bootstrap"]
    assert hidden is capture.hidden
    assert lp is h.gb._next_logprobs[0]
    assert kwargs["prompt_lookup_enabled"] is True
    # Include the cached prefix but exclude the already sampled generated token.
    assert kwargs["prompt_lookup_history"] == [1, 2, 10, 11, 12]


@pytest.mark.parametrize("stale_field", ["first_token", "prompt_length"])
def test_installed_capture_drops_stale_handoff(installed_capture, stale_field):
    h = installed_capture
    capture = _finish_prompt(h)
    setattr(capture, stale_field, getattr(capture, stale_field) + 1)
    assert h.gb._step()[0] == [18]
    assert h.captures == {}
    assert h.gb.orig_step_calls == 0
    assert "server_bootstrap" not in h.seen[0]
    assert h.seen[0]["prompt_cache"] is h.gb.prompt_cache
    assert h.seen[0]["prompt_lookup_history"] == [1, 2, 10, 11, 12, 18]


@pytest.mark.parametrize("missing", ["token", "logprobs"])
def test_installed_capture_discards_missing_staged_output(installed_capture, missing):
    h = installed_capture
    _finish_prompt(h)
    if missing == "token":
        h.gb._next_tokens = None
    else:
        h.gb._next_logprobs = []
    h.gb._step()
    assert h.captures == {}
    assert h.seen == []
    assert h.gb.orig_step_calls == 1
    assert h.batch._mtp_vendored_stats["fallthrough_steps"] == 1


def test_installed_capture_generate_failure_clears_transferred_uid(installed_capture):
    h = installed_capture
    h.pb.prompt([[10, 11]])
    assert 7 in h.captures
    h.pb.fail_generate = True
    with pytest.raises(RuntimeError, match="generation failed after uid transfer"):
        h.pb.generate([[12]])
    assert h.pb.uids == []
    assert h.pb.model is h.model
    assert h.captures == {}
    assert h.seen == []


def test_installed_capture_teardown_discards_pending_prompt(installed_capture):
    h = installed_capture
    h.pb.prompt([[10, 11]])
    assert 7 in h.captures
    assert type(h.pb) is not h.original_class
    # The fixture's teardown asserts both release and restoration while a
    # provisional prompt capture is still pending (no generation state exists).


@pytest.mark.parametrize(
    "route", ["batch_growth", "disabled", "waiting", "unsupported_processor"]
)
def test_installed_capture_discards_plain_route(installed_capture, route):
    h = installed_capture
    _finish_prompt(h)
    if route == "batch_growth":
        h.gb.uids = [7, 8]
    elif route == "disabled":
        h.gb._mtp_vendored_disabled_uids[7] = "req-7"
    elif route == "waiting":
        h.batch._unprocessed_sequences = [object()]
    else:
        h.gb.logits_processors = [[lambda tokens, logits: logits]]
    assert h.gb._step()[0] == [18]
    assert h.captures == {}
    assert h.seen == []
    assert h.gb.orig_step_calls == 1


def test_installed_capture_terminal_failure_discards_provisional_state(
    installed_capture,
):
    h = installed_capture
    capture = _finish_prompt(h)
    assert h.gb._step()[0] == [18]
    # A provisional capture must not survive a fail-closed boundary either.
    h.captures[7] = capture
    h.gb.uids = [7, 8]
    with pytest.raises(RuntimeError, match="batch grew"):
        h.gb._step()
    assert h.captures == {}
    assert 7 in h.gb._mtp_vendored_terminal_uids
    assert h.gb.orig_step_calls == 0


@pytest.mark.parametrize("rows", [1, 19, 32, 33, 63, 64, 65, 563])
def test_measured_prefill_alignment_preserves_logits_pairs_and_caches(
    monkeypatch, rows
):
    from rapid_mlx.tuning import mtp as policy

    monkeypatch.setattr(policy, "prefill_block_size", lambda model: 32)
    seen = []
    expected_cache = []

    class RecordingModel(Model):
        def __call__(self, inputs, cache=None, return_hidden=False):
            seen.append(inputs.shape[1])
            assert cache is expected_cache
            return super().__call__(inputs, cache, return_hidden)

    model = RecordingModel()
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    capture = module.HeadPrefillCapture(model, mx, lambda c: None)
    inputs = np.arange(rows)[None]
    logits = capture(inputs, cache=expected_cache)
    assert np.array_equal(logits, inputs)
    assert model.rows == [(10 * t, t + 1) for t in range(rows - 1)]
    assert capture.hidden.tolist() == [[[(rows - 1) * 10]]]
    boundary = rows // 32 * 32
    assert seen == ([boundary, rows - boundary] if 0 < boundary < rows else [rows])


def test_profile_alignment_matches_cold_prefix_snapshot_calls(monkeypatch):
    from rapid_mlx.tuning import mtp as policy

    monkeypatch.setattr(policy, "prefill_block_size", lambda model: 32)
    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    cached, uncached = Model(), Model()
    left = module.HeadPrefillCapture(cached, mx, lambda c: None)
    right = module.HeadPrefillCapture(uncached, mx, lambda c: None)
    tokens = np.arange(564)[None]
    left(tokens[:, :544], cache=[])
    left(tokens[:, 544:-1], cache=[])
    left(tokens[:, -1:], cache=[])
    right(tokens[:, :-1], cache=[])
    right(tokens[:, -1:], cache=[])
    assert cached.rows == uncached.rows
    assert np.array_equal(left.hidden, right.hidden)


def test_prefill_alignment_requires_a_shared_cache(monkeypatch):
    from rapid_mlx.tuning import mtp as policy

    monkeypatch.setattr(policy, "prefill_block_size", lambda model: 32)
    seen = []

    class RecordingModel(Model):
        def __call__(self, inputs, cache=None, return_hidden=False):
            seen.append(inputs.shape[1])
            return super().__call__(inputs, cache, return_hidden)

    mx = SimpleNamespace(concatenate=np.concatenate, eval=lambda *a: None)
    capture = module.HeadPrefillCapture(RecordingModel(), mx, lambda c: None)
    # Without a cache, a trailing chunk would lose the leading chunk's state.
    logits = capture(np.arange(40)[None])
    assert seen == [40]
    assert np.array_equal(logits, np.arange(40)[None])
