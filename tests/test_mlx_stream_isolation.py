# SPDX-License-Identifier: Apache-2.0
"""Executor suites must not leave worker-owned generation streams behind."""

import ast
import importlib.machinery
import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.conftest import (
    _capture_native_mlx_stream_runtime,
    _get_native_mlx_stream_runtime,
    _is_native_mlx_stream_runtime,
    _MlxStreamImportObserver,
    _repair_mlx_generation_streams,
    _repair_mlx_generation_streams_per_test,
    preserve_mlx_generation_streams,
)

_GENERATION_MODULES = (
    "mlx_lm.generate",
    "mlx_vlm.generate",
    "rapid_mlx.models.mlx_vlm_vendored.generate.common",
)


class ThreadStream:
    def __init__(self):
        self.owner = threading.get_ident()

    def eval(self):
        if self.owner != threading.get_ident():
            raise RuntimeError("There is no Stream(gpu, 2) in current thread.")
        return 7


def _trust_stream_runtime(monkeypatch, mx):
    """Exercise repair mechanics with an explicitly trusted CPU stream model."""
    mx.Stream = mx.ThreadLocalStream = ThreadStream
    monkeypatch.setattr(
        sys.modules[_repair_mlx_generation_streams.__module__],
        "_get_native_mlx_stream_runtime",
        lambda: mx,
    )


@pytest.mark.parametrize("late_import", [False, True])
@pytest.mark.parametrize("failed_test", [False, True])
def test_worker_stream_restored_after_executor_shutdown(
    monkeypatch, late_import, failed_test
):
    """Execute the real initializer with thread-bound streams, without Metal."""
    caller_stream = ThreadStream()
    state = threading.local()
    state.stream = caller_stream

    def default_stream(_device):
        if not hasattr(state, "stream"):
            state.stream = ThreadStream()
        return state.stream

    mx = ModuleType("mlx.core")
    mx.default_device = lambda: "gpu"
    mx.default_stream = default_stream
    mx.set_default_stream = lambda stream: setattr(state, "stream", stream)
    mx.synchronize = lambda stream: stream.eval()
    mx.new_stream = lambda _device: ThreadStream()
    mx.new_thread_local_stream = lambda _device: ThreadStream()
    _trust_stream_runtime(monkeypatch, mx)
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    names = ("mlx_lm.generate", "mlx_vlm.generate")
    originals = {name: ThreadStream() for name in names}
    for name in names:
        if late_import:
            monkeypatch.delitem(sys.modules, name, raising=False)
        else:
            monkeypatch.setitem(
                sys.modules, name, SimpleNamespace(generation_stream=originals[name])
            )

    # Loading just this function avoids importing the inference runtime on
    # hosts without Metal; its stream writes are the production statements.
    source = Path(__file__).resolve().parents[1] / "rapid_mlx/engine_core.py"
    tree = ast.parse(source.read_text())
    initializer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_mlx_step_thread"
    )
    namespace = {"mx": mx, "sys": sys, "logger": logging.getLogger(__name__)}
    exec(
        compile(ast.Module(body=[initializer], type_ignores=[]), source, "exec"),
        namespace,
    )
    isolate = contextmanager(preserve_mlx_generation_streams.__wrapped__)

    def run_suite():
        with isolate():
            if late_import:
                for name in names:
                    monkeypatch.setitem(
                        sys.modules,
                        name,
                        SimpleNamespace(generation_stream=ThreadStream()),
                    )
            with ThreadPoolExecutor(
                max_workers=1, initializer=namespace["_init_mlx_step_thread"]
            ) as executor:
                for name in names:
                    assert (
                        executor.submit(
                            lambda name=name: sys.modules[name].generation_stream.eval()
                        ).result()
                        == 7
                    )
            # Reproduce the later consumer's failure before suite teardown.
            for name in names:
                with pytest.raises(RuntimeError, match="no Stream"):
                    sys.modules[name].generation_stream.eval()
            mx.set_default_stream(ThreadStream())
            if failed_test:
                raise ValueError("failed test")

    if failed_test:
        with pytest.raises(ValueError, match="failed test"):
            run_suite()
    else:
        run_suite()

    assert mx.default_stream(mx.default_device()).eval() == 7
    for name in names:
        restored = sys.modules[name].generation_stream
        assert restored is not originals[name]
        assert restored.eval() == 7


@pytest.mark.parametrize(
    "filename",
    [
        "test_batching.py",
        "test_batching_deterministic.py",
        "test_mtp_cli_wiring.py",
        "test_qwen36_native_text_cache.py",
        "test_no_mllm_flag.py",
    ],
)
def test_leaking_suites_request_stream_restoration(filename):
    """Module fixtures must tear down before generation streams are restored."""
    root = Path(__file__).resolve().parent
    suite = ast.parse((root / filename).read_text())
    fixtures = {}
    module_fixtures = []
    for node in ast.walk(suite):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            func = call.func if call else decorator
            if not isinstance(func, ast.Attribute) or func.attr != "fixture":
                continue
            options = (
                {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                if call
                else {}
            )
            name = options.get("name", node.name)
            fixtures[name] = {
                arg.arg
                for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
            if options.get("scope") == "module":
                module_fixtures.append(name)

    # Check all module fixtures so a worker hidden behind a helper or an engine
    # fixture cannot evade the contract through its name. Function fixtures
    # already tear down before module-scoped restoration.
    for name in module_fixtures:
        pending = [name]
        visited = set()
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            pending.extend(fixtures.get(dependency, ()))
        assert "preserve_mlx_generation_streams" in visited, (
            f"{filename}: module fixture {name!r} must depend on "
            "preserve_mlx_generation_streams so workers shut down before restoration"
        )

    # The deterministic suite requests restoration through its executor;
    # suites with workers created in test bodies still need the module marker.
    if filename == "test_batching_deterministic.py":
        assert "mlx_executor" in module_fixtures
        return
    marker = next(
        node
        for node in suite.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "usefixtures"
        and any(
            isinstance(arg, ast.Constant)
            and arg.value == "preserve_mlx_generation_streams"
            for arg in node.args
        )
        for node in ast.walk(marker)
    )


@pytest.mark.parametrize("has_mlx", [False, True])
def test_audio_suite_preserves_streams_only_when_mlx_is_available(has_mlx):
    """Exercise optional fixture wiring without skipping the no-MLX audio lane."""
    source = Path(__file__).with_name("test_audio_model_worker.py")
    tree = ast.parse(source.read_text())
    fixture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_preserve_audio_generation_streams"
    )
    decorator = fixture.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert isinstance(decorator.func, ast.Attribute)
    assert decorator.func.attr == "fixture"
    assert {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords
    } == {
        "scope": "module",
        "autouse": True,
    }
    fixture.decorator_list = []
    namespace = {
        "importlib": SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda _name: object() if has_mlx else None)
        )
    }
    exec(
        compile(ast.Module(body=[fixture], type_ignores=[]), source, "exec"), namespace
    )
    requested = []
    namespace[fixture.name](SimpleNamespace(getfixturevalue=requested.append))
    assert requested == (["preserve_mlx_generation_streams"] if has_mlx else [])


def test_module_executor_joins_before_stream_restoration():
    """The module executor must depend on restoration so it tears down first."""
    root = Path(__file__).resolve().parent
    deterministic = ast.parse((root / "test_batching_deterministic.py").read_text())
    executor = next(
        node
        for node in deterministic.body
        if isinstance(node, ast.FunctionDef) and node.name == "mlx_executor"
    )
    assert "preserve_mlx_generation_streams" in [a.arg for a in executor.args.args]


@pytest.fixture
def stream_runtime(monkeypatch):
    """Model a process default that a joined worker can poison without Metal."""
    state = SimpleNamespace(default=ThreadStream())
    mx = SimpleNamespace(
        default_device=lambda: "gpu",
        default_stream=lambda _device: state.default,
        set_default_stream=lambda stream: setattr(state, "default", stream),
        new_stream=lambda _device: ThreadStream(),
        new_thread_local_stream=lambda _device: ThreadStream(),
        synchronize=lambda stream: stream.eval(),
    )
    _trust_stream_runtime(monkeypatch, mx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    for name in _GENERATION_MODULES:
        monkeypatch.setitem(
            sys.modules, name, SimpleNamespace(generation_stream=ThreadStream())
        )
    return mx


def _bind_worker_streams(mx):
    def bind():
        stream = mx.new_stream(mx.default_device())
        for name in _GENERATION_MODULES:
            sys.modules[name].generation_stream = stream
        mx.set_default_stream(stream)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(bind).result()


@pytest.mark.parametrize(
    "fixture",
    [_repair_mlx_generation_streams_per_test, preserve_mlx_generation_streams],
)
@pytest.mark.parametrize("failed_test", [False, True])
def test_repairs_already_stale_setup_and_teardown(stream_runtime, fixture, failed_test):
    mx = stream_runtime

    def assert_usable():
        assert mx.default_stream("gpu").eval() == 7
        for name in _GENERATION_MODULES:
            assert sys.modules[name].generation_stream.eval() == 7

    # Contaminate BEFORE fixture setup: restoring this snapshot would fail.
    _bind_worker_streams(mx)
    with pytest.raises(RuntimeError, match="no Stream"):
        mx.synchronize(mx.default_stream("gpu"))

    def run_test():
        with contextmanager(fixture.__wrapped__)():
            assert_usable()
            _bind_worker_streams(mx)
            # A repair must not intervene inside the test body.
            with pytest.raises(RuntimeError, match="no Stream"):
                assert_usable()
            if failed_test:
                raise ValueError("failed test")

    if failed_test:
        with pytest.raises(ValueError, match="failed test"):
            run_test()
    else:
        run_test()
    assert_usable()


def test_repair_preserves_usable_streams(stream_runtime):
    mx = stream_runtime
    default = mx.default_stream("gpu")
    streams = [sys.modules[name].generation_stream for name in _GENERATION_MODULES]
    _repair_mlx_generation_streams()
    assert mx.default_stream("gpu") is default
    assert all(
        sys.modules[name].generation_stream is stream
        for name, stream in zip(_GENERATION_MODULES, streams)
    )


def test_repair_without_imported_mlx(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    before = set(sys.modules)
    _repair_mlx_generation_streams()
    assert set(sys.modules) == before


def test_repair_ignores_partial_mlx_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", ModuleType("mlx.core"))
    _repair_mlx_generation_streams()


def test_repair_handles_distinct_and_aliased_generation_modules(
    stream_runtime, monkeypatch
):
    mx = stream_runtime
    _bind_worker_streams(mx)
    # Public and vendored imports may resolve to the same module object.
    monkeypatch.setitem(
        sys.modules, _GENERATION_MODULES[1], sys.modules[_GENERATION_MODULES[2]]
    )
    valid = ThreadStream()
    sys.modules[_GENERATION_MODULES[0]].generation_stream = valid
    _repair_mlx_generation_streams()
    assert sys.modules[_GENERATION_MODULES[0]].generation_stream is valid
    assert sys.modules[_GENERATION_MODULES[1]].generation_stream.eval() == 7


@pytest.mark.parametrize(
    "error", [RuntimeError("retired stream"), StopIteration(), ValueError("bad stub")]
)
@pytest.mark.parametrize("api", ["synchronize", "default_stream", "default_device"])
def test_repair_ignores_probe_errors(stream_runtime, monkeypatch, error, api):
    def fail(_stream):
        raise error

    default_stream = stream_runtime.default_stream
    original_default = default_stream("gpu")
    before = [sys.modules[name].generation_stream for name in _GENERATION_MODULES]
    monkeypatch.setattr(stream_runtime, api, lambda *args: fail(None))
    _repair_mlx_generation_streams()
    assert default_stream("gpu") is original_default
    assert [
        sys.modules[name].generation_stream for name in _GENERATION_MODULES
    ] == before


def test_repair_leaves_non_stream_generation_doubles_alone(stream_runtime, monkeypatch):
    fake = object()
    module = sys.modules[_GENERATION_MODULES[0]]
    module.generation_stream = fake

    def synchronize(stream):
        assert isinstance(stream, ThreadStream)
        stream.eval()

    monkeypatch.setattr(stream_runtime, "synchronize", synchronize)
    _repair_mlx_generation_streams()
    assert module.generation_stream is fake


def test_repair_rejects_complete_fake_core(monkeypatch):
    def forbidden(*args):
        pytest.fail("repair called a fake MLX API")

    mx = ModuleType("mlx.core")
    for name in (
        "default_device",
        "default_stream",
        "set_default_stream",
        "new_stream",
        "new_thread_local_stream",
        "synchronize",
    ):
        setattr(mx, name, forbidden)
    mx.Stream = mx.ThreadLocalStream = ThreadStream
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    assert not _is_native_mlx_stream_runtime(mx)
    _repair_mlx_generation_streams()


@pytest.mark.parametrize(
    "api",
    [
        "default_device",
        "default_stream",
        "set_default_stream",
        "new_stream",
        "new_thread_local_stream",
        "synchronize",
    ],
)
def test_native_runtime_gate_rejects_patched_functions(monkeypatch, api):
    """Even a Python wrapper copying the native function's module is a double."""

    mx = ModuleType("mlx.core")
    mx.__spec__ = importlib.machinery.ModuleSpec(
        "mlx.core", importlib.machinery.ExtensionFileLoader("mlx.core", "core.so")
    )
    mx.Stream = mx.ThreadLocalStream = type("Stream", (), {"__module__": "mlx.core"})
    for name in (
        "default_device",
        "default_stream",
        "set_default_stream",
        "new_stream",
        "new_thread_local_stream",
        "synchronize",
    ):
        setattr(mx, name, len)
    assert _is_native_mlx_stream_runtime(mx)
    wrapper = lambda *args: next(iter(()))
    wrapper.__module__ = "mlx.core"
    monkeypatch.setattr(mx, api, wrapper)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    assert not _is_native_mlx_stream_runtime(mx)
    _repair_mlx_generation_streams()


@pytest.mark.parametrize("replace_module", [False, True])
def test_captured_runtime_repairs_streams_despite_doubles(
    stream_runtime, monkeypatch, replace_module
):
    cf = sys.modules[_repair_mlx_generation_streams.__module__]
    monkeypatch.setattr(cf, "_NATIVE_MLX_STREAM_RUNTIME", None)
    monkeypatch.setattr(
        cf, "_is_native_mlx_stream_runtime", lambda mx: mx is stream_runtime
    )
    _capture_native_mlx_stream_runtime(stream_runtime)
    captured = cf._NATIVE_MLX_STREAM_RUNTIME
    monkeypatch.setattr(cf, "_get_native_mlx_stream_runtime", lambda: captured)
    _bind_worker_streams(stream_runtime)
    default_stream = stream_runtime.default_stream
    fake = object()
    sys.modules[_GENERATION_MODULES[1]].generation_stream = fake

    def forbidden(*args):
        pytest.fail("repair invoked a patched stream API")

    for name in cf._MLX_STREAM_APIS:
        monkeypatch.setattr(stream_runtime, name, forbidden)
    if replace_module:
        monkeypatch.setitem(sys.modules, "mlx.core", ModuleType("mlx.core"))
        monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    _capture_native_mlx_stream_runtime(sys.modules.get("mlx.core"))
    assert cf._NATIVE_MLX_STREAM_RUNTIME is captured
    _repair_mlx_generation_streams()
    assert default_stream("gpu").eval() == 7
    assert sys.modules[_GENERATION_MODULES[0]].generation_stream.eval() == 7
    assert sys.modules[_GENERATION_MODULES[2]].generation_stream.eval() == 7
    assert sys.modules[_GENERATION_MODULES[1]].generation_stream is fake


def test_import_observer_captures_before_importer_can_patch(
    stream_runtime, monkeypatch
):
    cf = sys.modules[_repair_mlx_generation_streams.__module__]
    monkeypatch.setattr(cf, "_NATIVE_MLX_STREAM_RUNTIME", None)
    monkeypatch.setattr(
        cf, "_is_native_mlx_stream_runtime", lambda mx: mx is stream_runtime
    )
    original_sync = stream_runtime.synchronize

    class Loader(importlib.machinery.ExtensionFileLoader):
        def exec_module(self, module):
            module.synchronize = original_sync

    loader = Loader("mlx.core", "core.so")
    spec = importlib.machinery.ModuleSpec("mlx.core", loader)

    class Finder:
        def find_spec(self, fullname, path, target=None):
            assert fullname == "mlx.core"
            return spec

    observer = _MlxStreamImportObserver()
    monkeypatch.setattr(sys, "meta_path", [observer, Finder()])
    assert observer.find_spec("unrelated", None) is None
    assert observer.find_spec("mlx.core", None) is spec
    spec.loader.exec_module(stream_runtime)
    stream_runtime.synchronize = lambda stream: None
    assert cf._NATIVE_MLX_STREAM_RUNTIME.synchronize is original_sync
    assert cf._NATIVE_MLX_STREAM_RUNTIME.module is stream_runtime
    assert observer.find_spec("mlx.core", None) is None


def test_real_generator_after_joined_worker_stream_contamination(monkeypatch):
    """No weights: repair dead-worker globals before evaluating a tiny model."""
    # Importing MLX itself can abort when installed without a usable Metal
    # device. Probe in a child before loading it into the pytest worker.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; raise SystemExit(not mx.metal.is_available())",
        ],
        capture_output=True,
        timeout=30,
    )
    if probe.returncode:
        pytest.skip("requires importable MLX and Metal")
    pytest.importorskip("mlx.core", exc_type=ImportError)
    runtime = _get_native_mlx_stream_runtime()
    assert runtime is not None, "native MLX import was not captured"
    mx = runtime.module
    import importlib

    from mlx_lm.models.llama import Model, ModelArgs

    gen = importlib.import_module("mlx_lm.generate")

    def bind():
        stream = runtime.new_stream(runtime.default_device())
        for name in _GENERATION_MODULES:
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "generation_stream"):
                module.generation_stream = stream
        runtime.set_default_stream(stream)
        runtime.synchronize(stream)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(bind).result()
        with pytest.raises(RuntimeError, match="There is no Stream"):
            runtime.synchronize(gen.generation_stream)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "mlx.core", ModuleType("mlx.core"))
            patch.setitem(sys.modules, "mlx", ModuleType("mlx"))

            def forbidden(*args):
                pytest.fail("repair called a patched native API")

            patch.setattr(mx, "synchronize", forbidden)
            patch.setattr(mx, "default_stream", forbidden)
            _repair_mlx_generation_streams()
        runtime.synchronize(gen.generation_stream)
        value = mx.array([1, 2]) + 1
        mx.eval(value)
        assert value.tolist() == [2, 3]
        model = Model(
            ModelArgs(
                model_type="llama",
                hidden_size=32,
                num_hidden_layers=1,
                intermediate_size=64,
                num_attention_heads=2,
                rms_norm_eps=1e-5,
                vocab_size=32,
            )
        )
        mx.eval(model.parameters())
        bg = gen.BatchGenerator(model, max_tokens=2, stop_tokens=[])
        try:
            bg.insert([[1, 2, 3]])
            responses = []
            for _ in range(4):
                responses.extend(bg.next_generated())
            assert responses
        finally:
            bg.close()
    finally:
        _repair_mlx_generation_streams()
