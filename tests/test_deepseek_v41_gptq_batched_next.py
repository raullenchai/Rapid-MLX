"""CPU-only qualification of the staged routed-factor pipeline."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq_batched as current
from scripts import deepseek_v41_gptq_batched as next_batch
from scripts.deepseek_v41_gptq_quant import QuantConfig, inverse_factor


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("count", [0, 3, 2048])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_inverse_factor_bitwise(count, dtype):
    assert next_batch.inverse_factor is current.inverse_factor is inverse_factor
    rng = np.random.default_rng(1017)
    x = rng.normal(size=(96, 32)).astype(dtype)
    x[:, 4] = 0
    h = x.T @ x if count else np.zeros((32, 32), dtype=dtype)
    before = h.tobytes()
    config = QuantConfig(group_size=32)
    expected = inverse_factor(mx.array(h.astype(np.float32)), config, count)
    # Execute the variant on a real worker, with NumPy input/output only.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: next_batch.inverse_factor_numpy(h, config, count), range(8)
            )
        )
    for upper, dead, stats in results:
        assert type(upper) is np.ndarray and type(dead) is np.ndarray
        assert upper.dtype == np.float32 and dead.dtype == np.bool_
        assert upper.tobytes() == np.asarray(expected[0]).tobytes()
        assert dead.tobytes() == np.asarray(expected[1]).tobytes()
        assert stats == expected[2]
    assert h.tobytes() == before


@pytest.mark.parametrize(
    "h,error",
    [
        (np.zeros((2, 3)), "square"),
        (np.full((2, 2), np.nan), "nonfinite"),
        (-np.eye(2), "negative"),
    ],
)
def test_numpy_inverse_factor_rejects_bad_hessian(h, error):
    with pytest.raises(ValueError, match=error):
        next_batch.inverse_factor_numpy(h, QuantConfig(), 3)


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_worker_count(monkeypatch, value):
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", value)
    with pytest.raises(ValueError):
        next_batch.factor_workers()


def test_factor_memory_budget(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1e12)
    )
    monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", "64")
    monkeypatch.setenv("RAPID_MLX_GPTQ_WORKING_GB", "48")
    args = SimpleNamespace(dim=5120, moe_inter_dim=2304, n_routed_experts=256)
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", "1")
    one = next_batch.expert_batch_size(args, dict(memory_limit_gb=400))
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", "4")
    four = next_batch.expert_batch_size(args, dict(memory_limit_gb=400))
    assert 1 <= four < one <= 64
    monkeypatch.setenv("RAPID_MLX_GPTQ_WORKING_GB", "0.000001")
    with pytest.raises(MemoryError, match="cannot hold one expert"):
        next_batch.expert_batch_size(args, dict(memory_limit_gb=400))


def build_layer(root, source, c, reference, module, monkeypatch):
    from scripts import deepseek_v41_gptq as b
    from scripts.deepseek_v41_gptq_quality import bind_quality

    # The driver's lazy import resolves this attribute. No source file is edited.
    quantize = module.quantize_routed
    with monkeypatch.context() as patch:
        patch.setattr(current, "quantize_routed", quantize)
        parts, scratch = root / "parts", root / "state"
        calibration = [np.array([9, 8, 7, 6, 5, 4, 3, 2], np.int32)]
        with b.MemoryGuard(c) as guard:
            b.initialize(
                source,
                c,
                calibration,
                scratch,
                parts,
                "fixture",
                guard,
                token_map=list(range(32)),
                heldout_reference=reference,
            )
            if reference is not None:
                bind_quality(
                    source,
                    parts,
                    root.parent / "reference",
                    root / "quality",
                    windows=calibration,
                )
            report = b.run_layer(source, c, parts, scratch, 0, guard)
    return parts, scratch, report


@pytest.mark.parametrize(
    "batch,workers,heldout,pilot",
    [
        (1, 1, False, False),
        (
            3,
            4,
            True,
            True,
        ),  # Ragged final batch, unobserved/sparse experts, pilot 3-bit.
        (2, 2, True, False),
    ],
)
def test_complete_layer_byte_identity(
    tmp_path, monkeypatch, batch, workers, heldout, pilot
):
    from tests.test_deepseek_v41_gptq_quality import fixture

    source, c, _, _, _ = fixture(tmp_path)
    c = {
        **c,
        "pilot": pilot,
        "dense_8bit_layers": list(range(source.args.n_layers)) if not pilot else [],
    }
    reference = (
        json.loads((tmp_path / "reference" / "reference.json").read_text())
        if heldout
        else None
    )
    monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", str(batch))
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", str(workers))
    spills = []
    original_spill = next_batch.RoutedSpill

    class CountedSpill(original_spill):
        def __init__(self, *args):
            super().__init__(*args)
            self.writes = 0
            spills.append(self)

        def write_output(self, guard):
            self.writes += 1
            super().write_output(guard)

    a, sa, ar = build_layer(
        tmp_path / "current", source, c, reference, current, monkeypatch
    )
    monkeypatch.setattr(next_batch, "RoutedSpill", CountedSpill)
    b, sb, br = build_layer(
        tmp_path / "next", source, c, reference, next_batch, monkeypatch
    )
    assert len(spills) == (2 if heldout else 1)
    assert all(spill.writes == 1 for spill in spills)
    assert all(
        type(spill.x) is np.ndarray and spill.x.dtype == np.uint16 for spill in spills
    )
    assert all(spill.output.dtype == np.float32 for spill in spills)
    assert ar == br  # Includes every damping/dead-column/calibration/heldout record.
    names = [
        "model-layer-0.safetensors",
        "layer-0-stats.json",
        "layer-0-complete.json",
    ]
    if heldout:
        names.append("heldout-drift.jsonl")
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # Exact accumulation order is observable in the propagated output boundary.
    for path in (sa / "boundary-1").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (sb / path.relative_to(sa)).read_bytes()


def empty_spill(monkeypatch):
    class EmptySpill:
        def __init__(self, *args):
            pass

        def chunks(self, expert):
            return iter(())

        def write_output(self, guard):
            pass

    monkeypatch.setattr(next_batch, "RoutedSpill", EmptySpill)
    monkeypatch.setattr(next_batch, "check_spill_budget", lambda *args: None)


@pytest.mark.parametrize("failure", ["worker", "gram"])
def test_pool_shutdown_and_failure_propagation(monkeypatch, failure):
    empty_spill(monkeypatch)
    from scripts import deepseek_v41_gptq as b

    pools = []
    entered = threading.Event()

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)
            self.joined = False

        def shutdown(self, wait=True, *, cancel_futures=False):
            assert wait and cancel_futures
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
            self.joined = True

    def worker(h, config, count):
        assert threading.current_thread() is not threading.main_thread()
        assert isinstance(h, np.ndarray)
        entered.set()
        raise RuntimeError("worker factor failed")

    calls = 0

    def gram(rows, dim, **kwargs):
        nonlocal calls
        assert threading.current_thread() is threading.main_thread()
        calls += 1
        if failure == "gram" and calls == 2:
            assert entered.wait(5)
            raise RuntimeError("main gram failed")
        return mx.eye(dim), 3

    monkeypatch.setattr(next_batch, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(next_batch, "inverse_factor_numpy", worker)
    monkeypatch.setattr(next_batch, "gram", gram)
    monkeypatch.setattr(next_batch, "expert_batch_size", lambda *args: 2)
    # Inputs are deliberately never consumed by the synthetic Gram.
    monkeypatch.setattr(b, "ffn_chunks", lambda *args: iter(()))
    source = SimpleNamespace(args=SimpleNamespace(dim=32, n_routed_experts=4))
    with pytest.raises(
        RuntimeError,
        match="worker factor failed" if failure == "worker" else "main gram failed",
    ):
        next_batch.quantize_routed(
            source,
            0,
            None,
            {},
            {},
            0,
            b.recipe(),
            SimpleNamespace(check=lambda: None),
            None,
        )
    assert len(pools) == 1 and pools[0].joined
    assert not any(t.is_alive() for t in pools[0]._threads)


def test_lookahead_down_submission_and_thread_boundaries(monkeypatch):
    empty_spill(monkeypatch)
    from scripts import deepseek_v41_gptq as b

    # Three batches prove b+2 waits for b's down/output; all experts observed.
    submitted, fits_at = [], []
    lookahead_started = threading.Event()
    gpu_started = threading.Event()
    down_started = threading.Event()
    gram_calls = 0
    original_factor = next_batch.inverse_factor_numpy
    original_gptq = next_batch.gptq_batched
    original_mx = next_batch.mx

    class MainThreadMLX:
        def __getattr__(self, name):
            assert threading.current_thread() is threading.main_thread()
            return getattr(original_mx, name)

    class Pool(ThreadPoolExecutor):
        def submit(self, fn, *args):
            assert threading.current_thread() is threading.main_thread()
            index = len(submitted)
            submitted.append(index)

            def job():
                assert threading.current_thread() is not threading.main_thread()
                assert all(not isinstance(a, mx.array) for a in args)
                if index == 2:
                    lookahead_started.set()
                    assert gpu_started.wait(5), "lookahead was awaited before GPU work"
                if index == 4:
                    down_started.set()
                return fn(*args)

            return super().submit(job)

    def gram(rows, dim, **kwargs):
        nonlocal gram_calls
        assert threading.current_thread() is threading.main_thread()
        # Each preceding Gram must have been submitted before the next starts.
        assert len(submitted) == gram_calls
        if gram_calls == 5:
            assert down_started.wait(5), "first down factor not submitted immediately"
        gram_calls += 1
        return mx.eye(dim), 3

    def gptq(*args, **kwargs):
        fits_at.append(len(submitted))
        if len(fits_at) == 1:
            assert lookahead_started.wait(5)
            gpu_started.set()
        return original_gptq(*args, **kwargs)

    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", "4")
    monkeypatch.setattr(next_batch, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(next_batch, "mx", MainThreadMLX())
    monkeypatch.setattr(next_batch, "gram", gram)
    monkeypatch.setattr(next_batch, "gptq_batched", gptq)
    monkeypatch.setattr(next_batch, "expert_batch_size", lambda *args: 2)
    monkeypatch.setattr(b, "ffn_chunks", lambda *args: iter(()))
    monkeypatch.setattr(b, "write_packed", lambda *args: None)
    # Empty projection-error inputs avoid disk I/O; Gram counts are synthetic.
    source = SimpleNamespace(
        args=SimpleNamespace(
            dim=32, moe_inter_dim=32, n_routed_experts=6, swiglu_limit=0
        ),
        tensor=lambda name: mx.ones((32, 32), mx.bfloat16),
    )
    spec = SimpleNamespace(bits=2, group_size=32)
    specs = {
        f"layers.0.ffn.experts.{proj}": spec
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    stats = next_batch.quantize_routed(
        source,
        0,
        None,
        specs,
        {},
        0,
        b.recipe(dict(group_size=32, block_size=32, pilot=False)),
        SimpleNamespace(check=lambda: None, report=lambda: {}),
        None,
    )
    assert len(stats) == 18
    assert fits_at == [4, 4, 6, 8, 8, 10, 10, 10, 12]
    assert gram_calls == 12
    # The direct NumPy path is still the original function, with no MLX facade use.
    assert next_batch.inverse_factor_numpy is original_factor


@pytest.mark.parametrize("chunk", [1, 4, 7, 100])
def test_ram_chunks_match_ffn_chunks(tmp_path, chunk):
    from scripts.deepseek_v41_gptq import ffn_chunks

    rng = np.random.default_rng(1017)
    total = 23
    args = SimpleNamespace(dim=32, n_activated_experts=3, n_routed_experts=7)
    paths = {key: tmp_path / f"{key}.bin" for key in ("x", "ids", "gates", "y")}
    raw = rng.integers(0, 65536, (total, args.dim), dtype=np.uint16)
    # Duplicate routes deliberately exercise stable row/slot order as well.
    ids = rng.integers(0, 5, (total, 3), dtype=np.int32)
    gates = rng.random(ids.shape, dtype=np.float32)
    output = rng.normal(size=(total, args.dim)).astype(np.float32)
    for key, data in zip(("x", "ids", "gates", "y"), (raw, ids, gates, output)):
        data.tofile(paths[key])
    checks = []
    guard = SimpleNamespace(check=lambda: checks.append(True))
    spill = next_batch.RoutedSpill(paths, total, args, {"ffn_chunk": chunk}, guard)
    assert all(
        type(a) is np.ndarray
        for a in (spill.x, spill.rows, spill.weights, spill.output)
    )
    assert spill.x.tobytes() == raw.tobytes()
    assert spill.output.tobytes() == output.tobytes()
    for expert in [None, *range(args.n_routed_experts)]:
        expected = list(ffn_chunks(paths, total, args, {"ffn_chunk": chunk}, expert))
        actual = list(spill.chunks(expert))
        assert len(actual) == len(expected)
        for (ar, ax, ag), (er, ex, eg) in zip(actual, expected):
            assert ar.dtype == er.dtype and ar.tobytes() == er.tobytes()
            assert ax.dtype == ex.dtype == mx.bfloat16
            assert (
                np.asarray(ax.view(mx.uint16)).tobytes()
                == np.asarray(ex.view(mx.uint16)).tobytes()
            )
            if eg is None:
                assert ag is None
            else:
                assert ag.dtype == eg.dtype == np.float32
                assert ag.tobytes() == eg.tobytes()
    assert len(list(spill.chunks(6))) == 0
    if chunk == 4:
        assert any((np.count_nonzero(ids == e) % chunk) for e in range(5))
    # Every duplicate route must contribute, independently of chunk size.
    for expert in range(args.n_routed_experts):
        for rows, _, _ in spill.chunks(expert):
            update = rng.normal(size=(len(rows), args.dim)).astype(np.float32)
            spill.add_output(rows, update)
            for row, contribution in zip(rows, update):
                output[row] += contribution
    spill.write_output(guard)
    assert paths["y"].read_bytes() == output.tobytes()
    assert checks


def test_ram_spill_budget_rejects_before_allocation():
    args = SimpleNamespace(dim=5120, n_activated_experts=8)
    # Actual workload reserves ~45.9 GB; 60 GB baseline leaves ~14 GB headroom.
    assert 45e9 < next_batch.RoutedSpill.allowance(1474560, args) < 47e9
    next_batch.check_spill_budget(1474560, args, {"memory_limit_gb": 180}, None)
    with pytest.raises(MemoryError, match="RAM routed spill estimate"):
        next_batch.check_spill_budget(3_000_000, args, {"memory_limit_gb": 180}, None)


def test_spill_read_checks_sizes_and_guard(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="spill size mismatch"):
        next_batch.RoutedSpill.read(path, np.uint16, (2, 2), lambda: None)
    np.zeros((2, 2), np.uint16).tofile(path)

    def fail():
        raise MemoryError("guard rejected RAM copy")

    with pytest.raises(MemoryError, match="guard rejected RAM copy"):
        next_batch.RoutedSpill.read(path, np.uint16, (2, 2), fail)


@pytest.mark.parametrize("limiter", ["working", "configured", "available"])
@pytest.mark.parametrize(
    "budget, expected", [(511, None), (1023, None), (1024, 1), (1536, 2)]
)
def test_expert_batch_rejects_insufficient_capacity(
    monkeypatch, limiter, budget, expected
):
    import psutil

    # 512 bytes of factor scratch plus 512 bytes per expert for this shape.
    args = SimpleNamespace(dim=4, moe_inter_dim=2, n_routed_experts=4)
    monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", "2")
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", "1")
    monkeypatch.setenv(
        "RAPID_MLX_GPTQ_WORKING_GB", str(budget / 1e9 if limiter == "working" else 1)
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=budget * 4 if limiter == "available" else 1e9
        ),
    )
    config = dict(memory_limit_gb=budget * 4 / 1e9 if limiter == "configured" else 1)
    if expected is None:
        with pytest.raises(MemoryError, match="cannot hold one expert"):
            next_batch.expert_batch_size(args, config)
    else:
        assert next_batch.expert_batch_size(args, config) == expected
