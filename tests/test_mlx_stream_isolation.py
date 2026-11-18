# SPDX-License-Identifier: Apache-2.0
"""Executor suites must not leave worker-owned generation streams behind."""

import ast
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.conftest import preserve_mlx_generation_streams


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

    assert mx.default_stream(mx.default_device()) is caller_stream
    for name in names:
        restored = sys.modules[name].generation_stream
        assert restored is (caller_stream if late_import else originals[name])
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
