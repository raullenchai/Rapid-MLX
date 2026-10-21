"""Execute draft fallback host contracts without importing the Metal runtime."""

import ast
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
