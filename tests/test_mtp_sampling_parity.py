# SPDX-License-Identifier: Apache-2.0
"""Execute the sampling arithmetic on CPU without importing Metal.

The same harness can use real MLX arrays in the Apple Silicon tests. AST
extraction retains production filenames and line numbers for coverage.
"""

import __future__

import ast
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SAMPLER = ROOT / "rapid_mlx/_sampler_fast_path.py"
SEEDED = ROOT / "rapid_mlx/_seeded_sampler.py"
GENERATOR = ROOT / "rapid_mlx/spec_decode/mtp/generator.py"


def _load_functions(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        env,
    )


def _logsumexp(x, axis=-1, keepdims=True):
    maximum = np.max(x, axis=axis, keepdims=True)
    return maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True))


def _put_along_axis(x, indices, values, axis):
    result = x.copy()
    np.put_along_axis(result, indices, values, axis)
    return result


def sampling_harness(
    mx, *, temp=0.7, top_p=0.95, top_k=0, min_p=0, seeded=False, fused=True, chain=()
):
    """Capture categorical inputs from both actual production callables."""
    captured = []

    def categorical(logits):
        captured.append(logits)
        return mx.argmax(logits, axis=-1)

    env = dict(mx=mx, Callable=object)
    _load_functions(
        SAMPLER,
        {
            "_fused_top_p_logits",
            "make_fused_top_p_temp_sampler",
            "is_fused_top_p_eligible",
        },
        env,
    )
    _load_functions(SEEDED, {"_make_seeded_logits_filter", "_apply_argmax_rescue"}, env)
    env.update(
        logits_processors=None,
        _is_greedy=temp == 0,
        lane_rng=object() if seeded else None,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        min_tokens_to_keep=1,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=[],
        env_truthy=lambda name: not fused,
        _make_sampler_chain=lambda *args: (list(chain), None),
        _categorical=categorical,
    )
    tree = ast.parse(GENERATOR.read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"_seeded_filter", "_use_fused_sampler"}
            or isinstance(target, ast.Tuple)
            and any(
                isinstance(item, ast.Name) and item.id == "_filter_chain"
                for item in target.elts
            )
            for target in node.targets
        )
    ]
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(GENERATOR), "exec"),
        env,
    )
    _load_functions(GENERATOR, {"_process_and_sample"}, env)
    return env, captured, categorical


@pytest.fixture
def cpu_mx():
    return SimpleNamespace(
        float32=np.float32,
        bool_=np.bool_,
        zeros=np.zeros,
        ones=np.ones,
        any=np.any,
        max=np.max,
        inf=np.inf,
        exp=np.exp,
        argsort=lambda x, axis: np.argsort(x, axis=axis, kind="stable"),
        take_along_axis=np.take_along_axis,
        cumsum=np.cumsum,
        arange=np.arange,
        where=np.where,
        full_like=np.full_like,
        put_along_axis=_put_along_axis,
        logsumexp=_logsumexp,
        argmax=np.argmax,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("temp", [0.3, 1.0, 1.7])
@pytest.mark.parametrize("top_p", [1e-9, 0.5, 0.95, 0.999999])
@pytest.mark.parametrize("top_k", [0, 1, 3, 8])
def test_cpu_mtp_matches_serving_distribution(cpu_mx, temp, top_p, top_k, dtype):
    env, captured, categorical = sampling_harness(
        cpu_mx, temp=temp, top_p=top_p, top_k=top_k
    )
    logits = np.array([[-9, -2, 0, 0, 0, 1, 2, 3]], dtype=dtype)
    _, logprobs, acceptance = env["_process_and_sample"](None, logits)
    mtp_scaled = captured.pop()
    cpu_mx.random = SimpleNamespace(categorical=categorical)
    env["make_fused_top_p_temp_sampler"](temp, top_p, top_k)(logprobs)
    sorted_scaled = captured.pop()
    indices = np.argsort(np.exp(logprobs.astype(np.float32)), axis=-1, kind="stable")
    serving_scaled = _put_along_axis(
        np.full_like(sorted_scaled, -np.inf), indices, sorted_scaled, -1
    )
    np.testing.assert_array_equal(mtp_scaled, serving_scaled)
    expected = serving_scaled - _logsumexp(serving_scaled)
    np.testing.assert_allclose(acceptance, expected, rtol=0, atol=1e-6)
    assert acceptance.dtype == np.float32


@pytest.mark.parametrize("temp", [0, 0.3, 1.7])
def test_cpu_fallback_and_greedy(cpu_mx, temp):
    def filter_row(row):
        return np.where(np.arange(row.shape[-1]) == 0, -np.inf, row)

    env, captured, _ = sampling_harness(
        cpu_mx, temp=temp, fused=False, chain=[] if temp == 0 else [filter_row]
    )
    logits = np.array([1, 2, 3], dtype=np.float32)
    token, logprobs, acceptance = env["_process_and_sample"](None, logits)
    assert token == 2
    if temp == 0:
        assert not captured
        np.testing.assert_array_equal(acceptance, logprobs)
    else:
        scaled = filter_row(logprobs) * (1 / temp)
        np.testing.assert_array_equal(captured.pop(), scaled)
        np.testing.assert_array_equal(acceptance, scaled - _logsumexp(scaled))


def test_cpu_unfiltered_temperature(cpu_mx):
    env, captured, _ = sampling_harness(cpu_mx, temp=0.3, fused=False)
    _, logprobs, acceptance = env["_process_and_sample"](
        None, np.array([1, 2, 3], dtype=np.float32)
    )
    scaled = logprobs * (1 / 0.3)
    np.testing.assert_array_equal(captured.pop(), scaled)
    np.testing.assert_array_equal(acceptance, scaled - _logsumexp(scaled))


@pytest.mark.parametrize("temp,top_p", [(0, 0.95), (1, 0), (1, 1)])
def test_cpu_fused_rejects_ineligible(cpu_mx, temp, top_p):
    env, _, _ = sampling_harness(cpu_mx)
    with pytest.raises(ValueError):
        env["make_fused_top_p_temp_sampler"](temp, top_p)


@pytest.mark.parametrize(
    "temp,top_p,min_p,top_k,xtc,disabled,expected",
    [
        (1, 0.95, 0, 20, 0, False, True),
        (1, 0.95, 0, 0, 0, False, True),
        (0, 0.95, 0, 20, 0, False, False),
        (1, 1, 0, 20, 0, False, False),
        (1, 0, 0, 0, 0, False, False),
        (1, 0.95, 0.1, 20, 0, False, False),
        (1, 0.95, 0, 20, 0.1, False, False),
        (1, 0.95, 0, 20, 0, True, False),
    ],
)
@pytest.mark.parametrize("seeded", [False, True])
def test_cpu_selects_same_serving_path(
    temp, top_p, min_p, top_k, xtc, disabled, expected, seeded
):
    env = dict(
        lane_rng=object() if seeded else None,
        temp=temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        xtc_probability=xtc,
        env_truthy=lambda name: disabled,
    )
    _load_functions(SAMPLER, {"is_fused_top_p_eligible"}, env)
    tree = ast.parse(GENERATOR.read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_use_fused_sampler"
            for t in node.targets
        )
    )
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(GENERATOR), "exec"),
        env,
    )
    assert env["_use_fused_sampler"] is (expected and not seeded)


def test_cpu_cutoff_tie_retains_same_token_ids(cpu_mx):
    env, captured, _ = sampling_harness(cpu_mx, top_k=3)
    env["_process_and_sample"](None, np.array([-3, -2, 0, 0, 0, 2], dtype=np.float32))
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(captured.pop())), [3, 4, 5]
    )


def test_cpu_shared_xtc_draw_reaches_filters(cpu_mx):
    cell = [None]
    draw = object()

    def xtc_filter(row):
        assert cell[0] is draw
        return row

    env, _, _ = sampling_harness(cpu_mx, fused=False, chain=[xtc_filter])
    env["_xtc_cell"] = cell
    env["_process_and_sample"](None, np.array([1, 2, 3], dtype=np.float32), draw)


def test_cpu_rejection_recovers_serving_distribution(cpu_mx):
    """The acceptance mass plus residual must equal serving's target mass."""
    env, _, _ = sampling_harness(cpu_mx, temp=0.7, top_k=3)
    _, _, p_log = env["_process_and_sample"](
        None, np.log(np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float32))
    )
    _, _, q_log = env["_process_and_sample"](
        None, np.log(np.array([0.02, 0.03, 0.15, 0.80], dtype=np.float32))
    )
    p, q = np.exp(p_log), np.exp(q_log)
    accepted_mass = np.minimum(p, q)
    residual = np.maximum(p - q, 0)
    rejection_probability = 1 - accepted_mass.sum()
    recovered = accepted_mass + rejection_probability * residual / residual.sum()
    np.testing.assert_allclose(recovered, p, rtol=0, atol=1e-7)
    assert residual.sum() > 0.5


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("temp", [0.3, 0.7, 1.7])
@pytest.mark.parametrize("top_p", [0, 1e-9, 0.5, 0.95, 1])
@pytest.mark.parametrize("top_k", [0, 1, 3, 100])
@pytest.mark.parametrize("min_p", [0, 0.1, 1, 1.1])
def test_cpu_seeded_mtp_matches_serving(cpu_mx, dtype, temp, top_p, top_k, min_p):
    env, captured, categorical = sampling_harness(
        cpu_mx, temp=temp, top_p=top_p, top_k=top_k, min_p=min_p, seeded=True
    )
    env.update(
        threading=threading,
        RequestSeededRNG=lambda seed: SimpleNamespace(next_key=lambda: seed),
    )
    _load_functions(SEEDED, {"make_seeded_sampler"}, env)
    cpu_mx.random = SimpleNamespace(categorical=lambda logits, key: categorical(logits))
    serving = env["make_seeded_sampler"](
        seed=42, temperature=temp, top_p=top_p, top_k=top_k, min_p=min_p
    )
    # Distinct rows exercise tied cutoffs, all-equal inputs and batch rescue.
    logits = np.array([[-3, -2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0]], dtype=dtype)
    _, logprobs, acceptance = env["_process_and_sample"](None, logits)
    mtp_scaled = captured.pop()
    serving(logprobs)
    serving_scaled = captured.pop()
    np.testing.assert_array_equal(mtp_scaled, serving_scaled)
    np.testing.assert_array_equal(
        acceptance, serving_scaled - _logsumexp(serving_scaled)
    )
    assert serving_scaled.dtype == np.float32
    assert np.isfinite(serving_scaled).any(axis=-1).all()
    if (temp, top_p, top_k, min_p) == (0.7, 0.95, 3, 0):
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(mtp_scaled[0])), [2, 3, 5]
        )


@pytest.mark.parametrize("seeded", [False, True])
def test_cpu_seeded_rejection_and_bonus(cpu_mx, seeded):
    env, _, _ = sampling_harness(cpu_mx, seeded=seeded, top_k=3)
    _, _, p_log = env["_process_and_sample"](
        None, np.array([-3, -2, 0, 0, 0, 2], dtype=np.float32)
    )
    _, _, q_log = env["_process_and_sample"](
        None, np.array([2, 0, 0, 0, -2, -3], dtype=np.float32)
    )
    p, q = np.exp(p_log), np.exp(q_log)
    accepted = np.minimum(p, q)
    residual = np.maximum(p - q, 0)
    recovered = accepted + (1 - accepted.sum()) * residual / residual.sum()
    np.testing.assert_allclose(recovered, p, rtol=0, atol=1e-7)
    assert residual.sum() > 0
    # The bonus draw is another target call through the same filter.
    _, _, bonus = env["_process_and_sample"](
        None, np.array([-3, -2, 0, 0, 0, 2], dtype=np.float32)
    )
    np.testing.assert_array_equal(bonus, p_log)


def test_cpu_generator_imports_shared_filters(monkeypatch, cpu_mx):
    """Resolve the generator's real relative imports with a CPU array backend."""
    mlx = ModuleType("mlx")
    mlx.core = cpu_mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", cpu_mx)
    modules = {}
    for name, path in [
        ("rapid_mlx._sampler_fast_path", SAMPLER),
        ("rapid_mlx._seeded_sampler", SEEDED),
    ]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    imports = [
        node
        for node in ast.parse(GENERATOR.read_text()).body
        if isinstance(node, ast.ImportFrom)
        and node.module in {"_env", "_sampler_fast_path", "_seeded_sampler"}
    ]
    env = {"__package__": "rapid_mlx.spec_decode.mtp"}
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), str(GENERATOR), "exec"), env
    )
    assert (
        env["_make_seeded_logits_filter"]
        is modules["rapid_mlx._seeded_sampler"]._make_seeded_logits_filter
    )
    assert (
        env["_fused_top_p_logits"]
        is modules["rapid_mlx._sampler_fast_path"]._fused_top_p_logits
    )
