# SPDX-License-Identifier: Apache-2.0
"""CPU-only chunk and lifecycle checks for the server head capture."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
