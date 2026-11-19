# SPDX-License-Identifier: Apache-2.0
"""Executor suites must not leave worker-owned generation streams behind."""

import ast
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
            raise RuntimeError("There is no Stream in current thread.")
        return 7


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
        "test_mtp_cli_wiring.py",
        "test_qwen36_native_text_cache.py",
        "test_no_mllm_flag.py",
    ],
)
def test_leaking_suites_request_stream_restoration(filename):
    """Guard the module-level restoration of suites that initialize MLX workers."""
    root = Path(__file__).resolve().parent
    suite = ast.parse((root / filename).read_text())
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


def test_repair_does_not_hide_unrelated_runtime_errors(stream_runtime, monkeypatch):
    def fail(_stream):
        raise RuntimeError("unrelated device failure")

    with monkeypatch.context() as patch:
        patch.setattr(stream_runtime, "synchronize", fail)
        with pytest.raises(RuntimeError, match="unrelated device failure"):
            _repair_mlx_generation_streams()


def test_real_generator_after_joined_worker_stream_contamination():
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
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    import importlib

    from mlx_lm.models.llama import Model, ModelArgs

    gen = importlib.import_module("mlx_lm.generate")

    def bind():
        stream = mx.new_stream(mx.default_device())
        for name in _GENERATION_MODULES:
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "generation_stream"):
                module.generation_stream = stream
        mx.set_default_stream(stream)
        mx.synchronize(stream)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(bind).result()
        with pytest.raises(RuntimeError, match="There is no Stream"):
            mx.synchronize(gen.generation_stream)
        _repair_mlx_generation_streams()
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
