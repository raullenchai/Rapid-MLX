"""Host-only guard/lifecycle tests, plus optional no-download Metal equivalence."""

import copy
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from rapid_mlx import batchgen_fastpath as fp


class KV:
    def __init__(self, length=4):
        self.length = length

    def size(self):
        return self.length

    def merge(self, layers):
        result = BatchKV()
        result._idx = max(layer.size() for layer in layers)
        result.left_padding = [result._idx - layer.size() for layer in layers]
        result.offset = SimpleNamespace(shape=(len(layers),))
        return result


class BatchKV:
    def __init__(self):
        self.left_padding = [0]
        self._right_padding = None
        self._idx = 4
        self.offset = SimpleNamespace(shape=(1,))
        self.calls = []

    def make_mask(self, N, return_array=False, **kwargs):  # noqa: N803 - upstream keyword API
        self.calls.append((N, return_array, kwargs))
        return "stock"

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["left_padding"] is not None:
            self.left_padding = kwargs["left_padding"]
        if any(kwargs["right_padding"] or []):
            self._right_padding = kwargs["right_padding"]

    def finalize(self):
        self.calls.append("finalize")
        if self._right_padding:
            self.left_padding = self._right_padding
        self._right_padding = None

    def trim(self, n):
        self._idx -= n
        return n

    def filter(self, indices):
        self.calls.append(list(indices))
        self.offset = SimpleNamespace(shape=(len(indices),))

    def extend(self, other):
        self.left_padding = self.left_padding + other.left_padding
        self.offset = SimpleNamespace(
            shape=(self.offset.shape[0] + other.offset.shape[0],)
        )
        self._idx = max(self._idx, other._idx)


def plain_mask(n, offset, return_array, window):
    return n, offset, return_array, window


def _merge_caches(caches):
    raise AssertionError("global merge must never be replaced or called")


class PromptProcessingBatch:
    def __init__(self, caches):
        self.prompt_cache = _merge_caches(caches)
        self.uids = list(range(len(caches)))
        self.tokens = [[] for _ in caches]

    def split(self, indices):
        indices = sorted(indices)
        new = copy.deepcopy(self)
        self.filter(sorted(set(range(len(self.uids))) - set(indices)))
        new.filter(indices)
        return new

    def filter(self, indices):
        self.uids = [self.uids[i] for i in indices]
        self.tokens = [self.tokens[i] for i in indices]
        if not indices:
            self.prompt_cache.clear()
        else:
            for layer in self.prompt_cache:
                layer.filter(indices)


class Generator:
    def _make_batch(self, caches):
        return PromptProcessingBatch(caches)


@pytest.fixture
def runtime(monkeypatch):
    gen = SimpleNamespace(
        BatchGenerator=Generator, PromptProcessingBatch=PromptProcessingBatch
    )
    cache = SimpleNamespace(
        BatchKVCache=BatchKV, KVCache=KV, create_attention_mask=plain_mask
    )
    monkeypatch.setattr(
        fp.importlib,
        "import_module",
        lambda name: gen if name.endswith("generate") else cache,
    )
    monkeypatch.setattr(fp, "version", lambda name: "0.31.3")
    monkeypatch.setattr(
        fp,
        "_SOURCE_HASHES",
        {
            "BatchKVCache.make_mask": hashlib.sha256(
                textwrap.dedent(inspect.getsource(BatchKV.make_mask)).encode()
            ).hexdigest()
        },
    )
    monkeypatch.delenv("RAPID_MLX_BATCHGEN_FASTPATH", raising=False)
    return gen, cache


def adapted(runtime, lengths=(4,)):
    generator = Generator()
    assert fp.install_batchgen_fastpath(generator)
    return generator._make_batch([[KV(length)] for length in lengths])


@pytest.mark.parametrize(
    "lengths,eligible", [((4,), True), ((4, 4), True), ((0, 0), True), ((2, 4), False)]
)
def test_eligibility_and_masks(runtime, lengths, eligible):
    batch = adapted(runtime, lengths)
    layer = batch.prompt_cache[0]
    assert layer._unpadded is eligible
    for n in (1, 3):
        for array in (False, True):
            for window in (None, 2):
                result = layer.make_mask(n, return_array=array, window_size=window)
                assert result == (
                    (n, max(lengths), array, window) if eligible else "stock"
                )
    assert layer.make_mask(2, unknown=True) == "stock"
    assert "_make_batch" not in Generator().__dict__


@pytest.mark.parametrize(
    "kwargs,eligible",
    [
        ({}, True),
        ({"left_padding": [0]}, True),
        ({"left_padding": [1]}, False),
        ({"right_padding": [0]}, True),
        ({"right_padding": [1]}, False),
    ],
)
def test_prepare_and_finalize(runtime, kwargs, eligible):
    layer = adapted(runtime).prompt_cache[0]
    layer.prepare(**kwargs)
    assert layer._unpadded is eligible
    layer.finalize()
    assert layer._unpadded is eligible


def test_invalidation(runtime):
    layer = adapted(runtime).prompt_cache[0]
    layer._right_padding = [1]
    layer.finalize()
    assert not layer._unpadded
    layer = adapted(runtime).prompt_cache[0]
    layer.left_padding = [0]
    assert not layer._unpadded
    layer = adapted(runtime).prompt_cache[0]
    assert layer.trim(1) == 1
    assert not layer._unpadded
    layer.prepare(left_padding=[0])
    assert not layer._unpadded


@pytest.mark.parametrize(
    "indices,fast",
    [
        ([0, 1], True),
        ([1, 0], False),
        ([0], False),
        ([0, 0], False),
        ((0, 1), False),
        ([0.0, 1.0], False),
        ([False, True], False),
    ],
)
def test_filter(runtime, indices, fast):
    layer = adapted(runtime, (4, 4)).prompt_cache[0]
    layer.filter(indices)
    assert layer.calls == ([] if fast else [list(indices)])
    assert layer._unpadded is fast


@pytest.mark.parametrize("lengths", [(4,), (4, 4), (2, 4)])
def test_split_lifecycle(runtime, lengths):
    batch = adapted(runtime, lengths)
    indices = list(range(len(lengths)))
    old_cache_list = batch.prompt_cache
    old_tokens = batch.tokens
    new = batch.split(indices)
    assert batch.uids == batch.tokens == batch.prompt_cache == []
    assert old_cache_list == []
    assert new.uids == indices
    assert new.prompt_cache is not old_cache_list
    assert new.tokens is not batch.tokens and new.tokens is not old_tokens
    assert new.prompt_cache[0].calls == ([] if len(set(lengths)) == 1 else [indices])


def test_partial_split(runtime):
    batch = adapted(runtime, (4, 4))
    new = batch.split([1])
    assert batch.uids == [0] and new.uids == [1]
    assert batch.prompt_cache[0].calls == [[0]]
    assert new.prompt_cache[0].calls == [[1]]


@pytest.mark.parametrize(
    "kind,eligible",
    [("equal", True), ("unequal", False), ("padded", False), ("foreign", False)],
)
def test_extend(runtime, kind, eligible):
    # Same installed subclass, as in a single generator's admissions.
    generator = Generator()
    assert fp.install_batchgen_fastpath(generator)
    left = generator._make_batch([[KV(4)]]).prompt_cache[0]
    right = generator._make_batch([[KV(5 if kind == "unequal" else 4)]]).prompt_cache[0]
    if kind == "padded":
        right.left_padding = [1]
    if kind == "foreign":
        right = BatchKV()
    left.extend(right)
    assert left._unpadded is eligible


def test_nonplain_merge(runtime):
    class OtherKV(KV):
        pass

    generator = Generator()
    assert fp.install_batchgen_fastpath(generator)
    result = generator._make_batch([[OtherKV()]]).prompt_cache[0]
    assert type(result) is BatchKV


@pytest.mark.parametrize("release", ["0.31.2", "0.31.4", "0.32.0", "0.31.3rc1"])
def test_version_gate(runtime, monkeypatch, release):
    monkeypatch.setattr(fp, "version", lambda _: release)
    assert not fp.install_batchgen_fastpath(Generator())


def test_kill_switch(runtime, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_BATCHGEN_FASTPATH", "0")
    assert not fp.install_batchgen_fastpath(Generator())


def test_drift_and_subclass(runtime, monkeypatch):
    monkeypatch.setattr(fp, "_SOURCE_HASHES", {"BatchKVCache.make_mask": "bad"})
    assert not fp.install_batchgen_fastpath(Generator())

    class Other(Generator):
        pass

    assert not fp.install_batchgen_fastpath(Other())


@pytest.mark.parametrize(
    "error",
    [
        fp.PackageNotFoundError("mlx-lm"),
        ImportError(),
        AttributeError(),
        OSError(),
        TypeError(),
        KeyError(),
        ValueError(),
    ],
)
def test_unavailable(runtime, monkeypatch, error):
    def fail(_):
        raise error

    monkeypatch.setattr(fp, "version", fail)
    assert not fp.install_batchgen_fastpath(Generator())


@pytest.fixture(scope="module")
def metal_runtime():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from rapid_mlx import _mlx_compat

    _mlx_compat.install()
    return mx


@pytest.mark.parametrize("lengths", [(4,), (4, 4), (2, 4)])
def test_real_masks_and_attention(lengths, metal_runtime):
    mx = metal_runtime
    import importlib

    from mlx_lm.models import cache

    gen = importlib.import_module("mlx_lm.generate")
    caches = []
    for length in lengths:
        c = cache.KVCache()
        c.update_and_fetch(mx.ones((1, 1, length, 8)), mx.ones((1, 1, length, 8)))
        caches.append([c])
    stock = gen.PromptProcessingBatch(
        None, list(range(len(lengths))), copy.deepcopy(caches)
    )
    stub = Generator()
    fp._attach(stub, gen, cache)
    # Invoke the private admission class through the cloned function globals.
    cls = stub._make_batch.__func__.__globals__["PromptProcessingBatch"]
    fast = cls(None, list(range(len(lengths))), copy.deepcopy(caches))
    a, b = stock.prompt_cache[0], fast.prompt_cache[0]
    for n in (1, 3):
        q = mx.random.normal((len(lengths), 1, n, 8))
        k = mx.random.normal((len(lengths), 1, max(lengths) + n, 8))
        v = mx.random.normal(k.shape)
        for array in (False, True):
            for window in (None, 2):
                am = a.make_mask(n, return_array=array, window_size=window)
                bm = b.make_mask(n, return_array=array, window_size=window)
                if bm is not None and not isinstance(bm, str):
                    assert mx.all(am == bm).item()
                ao = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=am)
                bo = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=bm)
                assert mx.allclose(ao, bo, atol=1e-5).item()
    moved = fast.split(list(range(len(lengths))))
    baseline = stock.split(list(range(len(lengths))))
    assert not fast.prompt_cache and not stock.prompt_cache
    for x, y in zip(moved.prompt_cache[0].state, baseline.prompt_cache[0].state):
        assert mx.array_equal(x, y).item()


def test_real_generator_tokens(metal_runtime):
    mx = metal_runtime
    import importlib

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.llama import Model, ModelArgs

    gen = importlib.import_module("mlx_lm.generate")
    mx.random.seed(7)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )
    mx.eval(model.parameters())
    prefix = make_prompt_cache(model)
    model(mx.array([[1, 2, 3, 4]]), cache=prefix)
    mx.eval([c.state for c in prefix])

    def run(fast):
        bg = gen.BatchGenerator(
            model, max_tokens=7, stop_tokens=[], prefill_batch_size=2
        )
        if fast:
            assert fp.install_batchgen_fastpath(bg)
        bg.insert([[5, 6, 7]], caches=[copy.deepcopy(prefix)])
        result = []
        for step in range(10):
            if step == 2:
                bg.insert([[8, 9, 10]])
            result.extend((r.uid, r.token) for r in bg.next_generated())
        bg.close()
        return result

    assert run(True) == run(False)


@pytest.mark.parametrize(
    "padding,expected",
    [
        (None, True),
        ([], True),
        ((0, 0), True),
        ([1], False),
        ([0.0], False),
        (object(), False),
    ],
)
def test_only_host_integer_padding(padding, expected):
    assert fp._host_zero(padding) is expected


def test_generator_has_no_self_cycle(runtime):
    import weakref

    generator = Generator()
    fp.install_batchgen_fastpath(generator)
    ref = weakref.ref(generator)
    del generator
    assert ref() is None


def test_installed_source_contract():
    """Audit the actual installed sources even on hosts without Metal access."""
    import ast
    from importlib.metadata import distribution

    try:
        dist = distribution("mlx-lm")
    except fp.PackageNotFoundError:
        pytest.skip("mlx-lm not installed")
    if dist.version != "0.31.3":
        pytest.skip("requires audited release")
    for path, expected in fp._SOURCE_HASHES.items():
        file = (
            "generate.py"
            if path.startswith(("PromptProcessingBatch.", "BatchGenerator."))
            else "models/cache.py"
        )
        if path == "create_causal_mask":
            file = "models/base.py"
        source = dist.locate_file("mlx_lm/" + file).read_text()
        node = ast.parse(source)
        parts = path.split(".")
        for part in parts[:2]:
            candidates = [x for x in node.body if getattr(x, "name", None) == part]
            node = candidates[-1] if parts[-1] == "fset" else candidates[0]
        start = min([node.lineno] + [x.lineno for x in node.decorator_list])
        snippet = textwrap.dedent(
            "".join(source.splitlines(True)[start - 1 : node.end_lineno])
        )
        assert hashlib.sha256(snippet.encode()).hexdigest() == expected, path


@pytest.mark.parametrize(
    "mode,suffix,quant,installed",
    [
        ("none", False, False, True),
        ("dflash", False, False, False),
        ("mtp", False, False, False),
        ("dspark", False, False, False),
        ("none", True, False, False),
        ("none", False, True, False),
    ],
)
def test_scheduler_wiring_without_mlx(
    runtime, monkeypatch, mode, suffix, quant, installed
):
    """Execute the production factory with fake optional runtimes on Linux."""
    import ast
    import logging
    import sys
    from pathlib import Path
    from types import ModuleType

    filename = Path(fp.__file__).with_name("scheduler.py")
    tree = ast.parse(filename.read_text())
    scheduler_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    factory = next(
        n
        for n in scheduler_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "_create_batch_generator"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            factory,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    bg = Generator()
    namespace = {
        "__name__": "rapid_mlx.scheduler",
        "BatchGenerator": lambda **_: bg,
        "make_sampler": lambda **_: None,
        "_assemble_stop_tokens": lambda *_: [],
        "_install_dense_sampler_fastpath": lambda _: None,
        "_config_vetted_mtp_supports_spec_decode": lambda _: False,
        "_install_dspark": lambda *_, **__: False,
        "_install_suffix_decoding": lambda *_, **__: False,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(module, str(filename), "exec"), namespace)
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.default_device = lambda: 0
    core.default_stream = lambda _: 0
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    for name, values in {
        "execution_reference": {"observe_plain_steps": lambda *_: None},
        "singleton_cache_fastpath": {"install_singleton_cache_fastpath": lambda: None},
        "compiled_decode": {"install_compiled_decode": lambda *_, **__: None},
    }.items():
        fake = ModuleType("rapid_mlx." + name)
        fake.__dict__.update(values)
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config = SimpleNamespace(
        spec_decode=mode,
        enable_suffix_decoding=suffix,
        kv_cache_quantization=quant,
        prefill_batch_size=2,
        completion_batch_size=2,
        prefill_step_size=8,
        suffix_max_draft=2,
        suffix_max_suffix_len=8,
        suffix_min_confidence=0.5,
        suffix_min_draft_len=1,
    )
    scheduler = SimpleNamespace(
        config=config,
        model=object(),
        model_config=SimpleNamespace(supports_spec_decode=False),
        requests={},
        uid_to_request_id={},
        _get_stop_tokens=lambda: [],
    )
    params = SimpleNamespace(temperature=0, top_p=1, min_p=0, top_k=0, max_tokens=8)
    assert namespace["_create_batch_generator"](scheduler, params) is bg
    assert getattr(bg, "_rapid_unpadded_fastpath", False) is installed


def test_preexisting_instance_hook(runtime):
    bg = Generator()
    bg._make_batch = lambda _: None
    assert not fp.install_batchgen_fastpath(bg)


def test_wrapped_method_is_not_hidden_by_inspect(runtime, monkeypatch):
    from functools import wraps

    original = BatchKV.make_mask

    @wraps(original)
    def changed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(BatchKV, "make_mask", changed)
    assert not fp.install_batchgen_fastpath(Generator())


@pytest.mark.parametrize("name", ["Rotating", "Quantized", "Composite"])
def test_special_cache_types_stay_stock(runtime, name):
    special_batch = type(name + "Batch", (BatchKV,), {})

    class SpecialKV(KV):
        def merge(self, layers):
            return special_batch()

    generator = Generator()
    assert fp.install_batchgen_fastpath(generator)
    result = generator._make_batch([[SpecialKV()]]).prompt_cache[0]
    assert type(result) is special_batch
    assert result.make_mask(2) == "stock"
    result.filter([0])
    assert result.calls[-1] == [0]
