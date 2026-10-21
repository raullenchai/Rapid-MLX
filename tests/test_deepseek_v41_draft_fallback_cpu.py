"""Execute draft fallback host contracts without importing the Metal runtime."""

import ast
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def kernels():
    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/draft_kernels.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and node.names[0].name == "mlx.core")
        and not (isinstance(node, ast.ImportFrom) and node.level)
    ]
    namespace = {"mx": np, "__name__": "draft_fallback_cpu"}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_deferred_flags_isolate_failure_and_warn_once(kernels, bad, caplog):
    draft = SimpleNamespace(adapter=SimpleNamespace(), w=SimpleNamespace())
    kernels.begin_round(draft)
    x = np.ones((5, 32))
    stock = lambda value: value
    first = kernels._checked(
        draft.adapter, "mtp.0.ffn", x, (2, 3, 4), lambda value: value * bad, stock
    )
    kernels._checked(draft.w, "head", first, "bfloat16", stock, stock)
    kernels._checked(draft.adapter, "mtp.1.ffn", x, (4, 4, 4), stock, stock)
    draft._draft_kernel_status = kernels.round_status(draft)
    assert draft._draft_kernel_status.tolist() == [False, True, True]
    assert kernels.disable_nonfinite(draft)
    assert not kernels.disable_nonfinite(draft)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "mtp.0.ffn" in caplog.text and "bits=(2, 3, 4)" in caplog.text
    assert not hasattr(draft.w, "_rapid_draft_disabled")
    kernels.begin_round(draft)
    # A new shape and bit combination must not resurrect a disabled layer.
    actual = kernels._checked(
        draft.adapter,
        "mtp.0.ffn",
        x[:1],
        (8, 8, 8),
        lambda _: pytest.fail("disabled kernel ran"),
        stock,
    )
    np.testing.assert_array_equal(actual, x[:1])
    np.testing.assert_array_equal(kernels.round_status(draft), [True])
    assert not kernels.disable_nonfinite(draft)
    assert not kernels.disable_nonfinite(SimpleNamespace())


def test_finite_difference_dispatches_once_and_unsupported_uses_stock(kernels):
    owner = SimpleNamespace(_rapid_draft_pending=[])
    calls = []
    x = np.ones((5, 32))

    def candidate(value):
        calls.append(value)
        return value + 10

    stock = lambda value: value
    actual = kernels._checked(owner, "head", x, "bfloat16", candidate, stock)
    np.testing.assert_array_equal(actual, x + 10)
    assert len(calls) == 1 and owner._rapid_draft_pending[0][3]
    np.testing.assert_array_equal(
        kernels._checked(owner, "unsupported", x, (), None, stock), x
    )
    # Component diagnostics outside a proposal still dispatch directly.
    np.testing.assert_array_equal(
        kernels._checked(SimpleNamespace(), "head", x, (), candidate, stock), x + 10
    )


def test_draft_head_reuses_attached_resident_weight_without_qualification(kernels):
    """Run the production attachment/reader and dispatch across draft rounds."""
    root = Path(__file__).resolve().parents[1]
    path = root / "rapid_mlx/models/deepseek_v41_native/dspark.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DSparkWeights"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("attach_target", "read")
    ]
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    weights = namespace["DSparkWeights"]()
    weights._arrays, weights.entries, weights.resident = {}, {}, {}
    weight = np.ones((64, 128), dtype=np.float32)
    model = SimpleNamespace(
        embed=SimpleNamespace(weight=weight), head=SimpleNamespace(weight=weight)
    )
    weights.attach_target(model)
    launches = []

    def launch(resident, x):
        assert resident is weight
        launches.append(resident)
        return x @ resident.T

    # No _matches_stock entry point: any draft qualification would fail here.
    kernels.head.__globals__["head_gemv"] = SimpleNamespace(
        _supports_weight=lambda w, x: w is weight, _launch=launch
    )
    draft = SimpleNamespace(w=weights, adapter=SimpleNamespace())
    for rows in (5, 1, 5):
        kernels.begin_round(draft)
        assert weights.read("head.weight") is model.head.weight
        assert weights.resident["head.weight"] is model.head.weight
        np.testing.assert_array_equal(
            kernels.head(weights, np.ones((rows, 128))), np.full((rows, 64), 128)
        )
        draft._draft_kernel_status = kernels.round_status(draft)
        assert not kernels.disable_nonfinite(draft)
    assert len(launches) == 3


@pytest.mark.parametrize("rows", range(1, 6))
def test_draft_head_selects_existing_shared_weight_kernel(kernels, rows):
    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/head_gemv.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_launch", "_shares_weight_read")
        or isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
    ]
    calls = []
    fail = False
    weight = np.ones((16384, 128), dtype=np.float32)
    x = np.ones((rows, 128), dtype=np.float32)

    def kernel(shared):
        def launch(**kwargs):
            calls.append((shared, kwargs))
            value, resident = kwargs["inputs"]
            assert resident is weight
            if fail:
                return (np.full((rows, 16384), np.nan),)
            return (value @ resident.T,)

        return launch

    namespace = dict(
        mx=np,
        _kernel=kernel,
        MAX_ROWS=8,
        SHARED_READ_MIN_OUTPUTS=16384,
        _OUTPUTS_PER_THREADGROUP=4,
        _LANES=32,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    kernels.head.__globals__["head_gemv"] = SimpleNamespace(
        _supports_weight=lambda *_: True, _launch=namespace["_launch"]
    )
    stock_calls = []

    def stock(base, value):
        stock_calls.append((base, value.shape))
        return value @ weight.T

    weights = SimpleNamespace(entries={}, read=lambda _: weight, linear=stock)
    draft = SimpleNamespace(w=weights, adapter=SimpleNamespace())
    kernels.begin_round(draft)
    np.testing.assert_array_equal(kernels.head(weights, x), np.full((rows, 16384), 128))
    assert len(calls) == 1
    shared, kwargs = calls[0]
    assert shared is (rows > 1)
    assert kwargs["grid"][1] == 1
    assert kwargs["output_shapes"] == [(rows, 16384)]
    assert not stock_calls
    # A finite input producing non-finite head logits disables this component
    # and the next attempt uses the original stock projection at the same depth.
    fail = True
    kernels.begin_round(draft)
    assert np.all(np.isnan(kernels.head(weights, x)))
    draft._draft_kernel_status = kernels.round_status(draft)
    assert kernels.disable_nonfinite(draft)
    np.testing.assert_array_equal(kernels.head(weights, x), np.full((rows, 16384), 128))
    assert len(calls) == 2
    assert stock_calls == [("head", (rows, 128))]


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, None])
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_propose_projects_only_consumed_rows(kernels, depth, device, packed):
    """Execute the full proposal orchestration with NumPy component doubles."""
    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/dspark.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSpark"
    )
    propose = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose"
    )
    propose.body = [n for n in propose.body if not isinstance(n, ast.ImportFrom)]
    events, head_inputs, markov_inputs = [], [], []
    namespace = dict(
        mx=SimpleNamespace(
            **{
                name: getattr(np, name)
                for name in (
                    "array",
                    "int32",
                    "float32",
                    "concatenate",
                    "full",
                    "repeat",
                    "broadcast_to",
                    "all",
                    "isfinite",
                )
            },
            async_eval=lambda *_: None,
            eval=lambda *_: None,
        ),
        draft_kernels=kernels,
        _hc_pre=lambda hidden, pre: hidden[:, 0],
        _hc_post=lambda value, hidden, post, combine: hidden + value[:, None],
    )
    exec(
        compile(ast.Module(body=[propose], type_ignores=[]), str(path), "exec"),
        namespace,
    )

    def mixes(base, kind, hidden):
        events.append((base, kind, hidden.shape[0]))
        return None, None, None

    def norm(base, value):
        events.append((base, "norm", value.shape[0]))
        return value + 1

    def head(value):
        head_inputs.append(value.copy())
        return value

    def markov(base, logits, previous):
        markov_inputs.append(logits.copy())
        return previous + 1, logits, np.ones((1, 3))

    draft = SimpleNamespace(
        block_size=5,
        position=2,
        layers=[0, 1, 2],
        c={"dspark_noise_token_id": 0, "hc_mult": 4},
        w=SimpleNamespace(
            embedding=lambda base, ids: np.repeat(np.arange(5)[:, None], 3, axis=1),
            linear=lambda base, value: value,
        ),
        adapter=SimpleNamespace(mixes=mixes, norm=norm, moe=lambda base, value: value),
        attention=lambda base, value, stage: value,
        head=head,
        markov=markov,
    )
    if packed:
        draft.adapter._packed_mtp_moe = True
    proposals, status = namespace["propose"](draft, 4, depth=depth, device=device)
    consumed = 5 if depth is None else depth
    assert len(proposals) == consumed + 1
    np.testing.assert_array_equal(proposals, np.arange(4, 5 + consumed))
    assert len(head_inputs) == 1 and head_inputs[0].shape == (consumed, 3)
    assert head_inputs[0].dtype == np.float32
    # Each full-block stage maps h to 4*h + 3; final norm adds one.
    expected = np.repeat((64 * (np.arange(consumed) + 1))[:, None], 3, axis=1)
    np.testing.assert_array_equal(head_inputs[0], expected)
    assert len(markov_inputs) == consumed
    np.testing.assert_array_equal(np.concatenate(markov_inputs), head_inputs[0])
    assert events and all(rows == 5 for _, _, rows in events)
    if device:
        assert status
    else:
        assert status.shape == (consumed, 6)


def test_target_qualification_failure_is_cached_per_owner_weight(caplog):
    """Four tiny target owners produce eight warnings, not repeated checks."""
    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/head_gemv.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    check = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_matches_stock"
    )
    launches = []

    def broken_launch(weight, probe, shared):
        launches.append(weight)
        raise RuntimeError("FP32 weight forced into BF16 kernel")

    namespace = {
        "mx": SimpleNamespace(
            random=SimpleNamespace(
                normal=lambda shape, **kw: np.ones(shape), key=lambda seed: seed
            )
        ),
        "MAX_ROWS": 8,
        "_CHECK_ATTR": "_rapid_exact_head_check",
        "_launch": broken_launch,
        "_shares_weight_read": lambda _: False,
        "weakref": weakref,
        "logger": logging.getLogger("target_qualification_cpu"),
    }
    exec(
        compile(ast.Module(body=[check], type_ignores=[]), str(path), "exec"), namespace
    )
    owners = [SimpleNamespace(weight=np.ones((4, 32))) for _ in range(4)]
    for _ in range(12):
        for owner in owners:
            assert not namespace["_matches_stock"](owner)
    assert len(launches) == 4
    assert len(caplog.records) == 8
    owners[0].weight = np.zeros((4, 32))
    assert not namespace["_matches_stock"](owners[0])
    assert len(launches) == 5
    assert len(caplog.records) == 10


@pytest.mark.parametrize("failure", ["mtp.0.ffn", "head"])
def test_serving_retries_at_existing_boundary_and_restores_cache(
    kernels, failure, caplog
):
    from dataclasses import dataclass

    from tests.test_deepseek_v41_draft_depth import DraftDepthPolicy

    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/serving.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            or isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ("stream_generate", "_match", "GenerationChunk")
        )
    ]
    layer = SimpleNamespace(window=8, win_kv=np.arange(8)[None, :])
    initial_window = layer.win_kv.copy()
    rollbacks, evaluations, observations = [], [], []
    cache = SimpleNamespace(offset=2, layers=[layer])

    def rollback(offset):
        rollbacks.append(offset)
        cache.offset = offset

    cache.rollback = rollback

    class Drafter:
        def __init__(self):
            self.adapter, self.w = SimpleNamespace(), SimpleNamespace()
            self.calls = 0

        def observe(self, *_args):
            pass

        def propose(self, seed, *, depth, device):
            self.calls += 1
            if self.calls == 2:
                assert rollbacks == [2]
                assert cache.offset == 2
                np.testing.assert_array_equal(layer.win_kv, initial_window)
                assert not observations  # failed verify hidden was never observed
            kernels.begin_round(self)
            owner = self.w if failure == "head" else self.adapter
            kernels._checked(
                owner,
                failure,
                np.ones((5, 4)),
                (2, 3, 4),
                lambda x: x * np.nan,
                lambda x: x,
            )
            self._draft_kernel_status = kernels.round_status(self)
            return np.ones(depth + 1, dtype=int), np.all(self._draft_kernel_status)

    def model(ids, _cache, **kwargs):
        if kwargs.get("enable_rollback"):
            slots = (cache.offset + np.arange(ids.shape[1])) % layer.window
            layer.win_kv[:, slots] = 99
            cache.offset += ids.shape[1]
        logits = np.zeros((*ids.shape, 2))
        logits[..., 1] = 1
        return logits, np.zeros((*ids.shape, 1))

    model.make_cache = lambda **kw: cache
    namespace = {
        "dataclass": dataclass,
        "time": SimpleNamespace(perf_counter=lambda: 0),
        "DraftDepthPolicy": DraftDepthPolicy,
        "draft_depth": lambda: 4,
        "draft_kernels": kernels,
        "DSpark": Drafter,
        "_observe_verified": lambda *args: observations.append(args),
        "NaiveStreamingDetokenizer": lambda _: SimpleNamespace(
            text="", add_token=lambda _: None, finalize=lambda: None
        ),
        "mx": SimpleNamespace(
            array=np.array,
            argmax=np.argmax,
            int32=np.int32,
            arange=np.arange,
            eval=lambda *args: evaluations.append(args),
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    draft = Drafter()
    runtime = SimpleNamespace(drafter=draft, reset=lambda: None)
    chunks = list(
        namespace["stream_generate"](
            model,
            SimpleNamespace(eos_token_id=9),
            [1, 2],
            runtime=runtime,
            max_tokens=6,
            optimize_prefill=False,
        )
    )
    assert [chunk.token for chunk in chunks] == [1] * 6
    assert draft.calls == 2  # failed attempt, then healthy five-token round
    assert len(evaluations) == 4  # prefill, failed verify, retry, final seed
    assert len(caplog.records) == 1 and failure in caplog.text
