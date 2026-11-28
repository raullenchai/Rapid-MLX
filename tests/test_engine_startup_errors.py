# SPDX-License-Identifier: Apache-2.0
"""Startup phase attribution never turns an unknown runtime error into OOM."""

import asyncio
import errno
from concurrent.futures import ThreadPoolExecutor

import pytest

from rapid_mlx.engine_startup_errors import (
    startup_boundary_for,
    startup_failure_boundary,
)
from rapid_mlx.local_serve_diagnostics import startup_error_lines
from rapid_mlx.telemetry.model_events import serve_error_class, serve_failure_subtype


@pytest.mark.parametrize("boundary", ["weights_load", "draft_attach"])
def test_worker_failure_retains_identity_and_closed_boundary(boundary):
    error = RuntimeError("private model /path and user content")

    def fail():
        raise error

    with (
        ThreadPoolExecutor(max_workers=1) as worker,
        pytest.raises(RuntimeError) as caught,
        startup_failure_boundary(boundary),
    ):
        worker.submit(fail).result()
    assert caught.value is error
    assert serve_failure_subtype(error) == boundary
    assert serve_error_class(error) == "other"
    assert startup_error_lines(error, None) is None


@pytest.mark.parametrize("error", [MemoryError(), OSError(errno.ENOMEM, "")])
def test_boundary_does_not_override_proven_memory_failure(error):
    with pytest.raises(type(error)), startup_failure_boundary("weights_load"):
        raise error
    assert serve_error_class(error) == "insufficient_memory"
    assert startup_error_lines(error, None) is not None


def test_explicit_cause_retains_boundary():
    try:
        with startup_failure_boundary("draft_attach"):
            raise RuntimeError("private")
    except RuntimeError as cause:
        wrapper = RuntimeError("relay")
        wrapper.__cause__ = cause
    assert serve_failure_subtype(wrapper) == "draft_attach"


def test_nested_boundary_preserves_inner_operation():
    with (
        pytest.raises(RuntimeError) as caught,
        startup_failure_boundary("draft_attach"),
        startup_failure_boundary("weights_load"),
    ):
        raise RuntimeError("private")
    assert serve_failure_subtype(caught.value) == "weights_load"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_cancellation_and_interrupt_are_unchanged(error):
    with (
        pytest.raises(type(error)) as caught,
        startup_failure_boundary("weights_load"),
    ):
        raise error
    assert caught.value is error
    assert startup_boundary_for(error) is None


@pytest.mark.parametrize("frozen", [False, True])
def test_annotation_failure_never_masks_original(frozen):
    class ReadOnlyError(RuntimeError):
        def __setattr__(self, name, value):
            if frozen or name == "_rapid_mlx_startup_boundary":
                raise TypeError("read only")
            super().__setattr__(name, value)

    error = ReadOnlyError("private")

    def fail():
        raise error

    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(fail)
        # Baseline: this frozen exception crosses the worker boundary intact.
        with pytest.raises(ReadOnlyError) as baseline:
            future.result()
        assert baseline.value is error
        with (
            pytest.raises(ReadOnlyError) as caught,
            startup_failure_boundary("weights_load"),
        ):
            future.result()
    assert caught.value is error
    assert serve_failure_subtype(error) == "runtime_error"


@pytest.mark.parametrize("value", ["private/path", [], None, 1])
def test_unknown_annotation_is_not_emitted(value):
    error = RuntimeError("private")
    error._rapid_mlx_startup_boundary = value
    assert serve_failure_subtype(error) == "runtime_error"


@pytest.mark.requires_mlx
@pytest.mark.parametrize("lane", ["text", "disk_stream", "multimodal"])
def test_engine_weight_loader_records_actual_boundary(monkeypatch, lane):
    """Exercise real startup methods; injected loader fails before GPU work."""
    from types import SimpleNamespace

    from rapid_mlx import engine_core
    from rapid_mlx.engine import batched
    from rapid_mlx.models import mllm
    from rapid_mlx.utils import tokenizer

    error = RuntimeError("private model load diagnostic")

    def fail(*args, **kwargs):
        raise error

    # Worker initialization is the only potential MLX operation before the
    # fake loader fails. Replace it; retain real Future.result propagation.
    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(tokenizer, "load_model_with_fallback", fail)
    monkeypatch.setattr(batched, "_load_lazy_and_install_disk_stream", fail)
    monkeypatch.setattr(mllm, "MLXMultimodalLM", fail)
    engine = object.__new__(batched.BatchedEngine)
    engine._model_name = "test-model"
    engine._trust_remote_code = False
    engine._scheduler_config = SimpleNamespace(spec_decode="none")
    engine._model_load_executor = None
    engine._enable_disk_stream = lane == "disk_stream"
    engine._force_mllm = False
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(
                engine._start_mllm() if lane == "multimodal" else engine._start_llm()
            )
        assert caught.value is error
        assert serve_failure_subtype(error) == "weights_load"
        assert serve_error_class(error) == "other"
    finally:
        if engine._model_load_executor is not None:
            engine._model_load_executor.shutdown(wait=True)


def test_hostile_annotation_getter_keeps_original_failure():
    class HostileError(RuntimeError):
        def __getattribute__(self, name):
            if name == "_rapid_mlx_startup_boundary":
                raise ValueError("private")
            return super().__getattribute__(name)

    error = HostileError("private")
    with (
        pytest.raises(HostileError) as caught,
        startup_failure_boundary("weights_load"),
    ):
        raise error
    assert caught.value is error
    assert startup_boundary_for(error) is None
    assert serve_failure_subtype(error) == "runtime_error"


def test_invalid_boundary_is_rejected_before_operation():
    with (
        pytest.raises(ValueError, match="Unknown engine startup boundary"),
        startup_failure_boundary("private/path"),
    ):
        pytest.fail("an invalid boundary must not run the operation")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["extra", "shape", "early", "non_strict", "quantize", "ok"]
)
async def test_stock_vlm_weight_boundary(monkeypatch, tmp_path, case):
    import sys
    from types import SimpleNamespace as Namespace

    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.model_load_errors import IncompatibleWeights
    from rapid_mlx.models import mllm

    message = "Received 1 parameters not in model: \ntransformer.wte.weight."
    if case == "shape":
        message = "Expected shape (8, 4) but received shape (8, 1) for parameter w"
    failure = ValueError(message)

    def fail(*args, **kwargs):
        raise failure

    class Module:
        config = Namespace()
        load_weights = (lambda *args, **kwargs: None) if case == "ok" else fail

    def load(path, *, trust_remote_code):
        if case == "early":
            raise failure
        if case == "quantize":
            nn.quantize()
        model = Module()
        model.load_weights([], strict=case != "non_strict")
        return model, Namespace()

    load.__module__ = "mlx_vlm.utils"
    nn = Namespace(Module=Module, quantize=fail)
    scheduler_config = Namespace(
        __dataclass_fields__={"prefill_step_size": Namespace(default=512)}
    )
    for name, stub in {
        "mlx": Namespace(nn=nn),
        "mlx.nn": nn,
        "mlx_vlm": Namespace(load=load),
        "mlx_vlm.utils": Namespace(load_config=lambda *a, **kw: {}),
        "rapid_mlx.engine_core": Namespace(_init_mlx_step_thread=lambda: None),
        "rapid_mlx.scheduler": Namespace(SchedulerConfig=None),
        "rapid_mlx.mllm_scheduler": Namespace(
            MLLMScheduler=None, MLLMSchedulerConfig=scheduler_config
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, stub)
    for suffix in ("runtime_fix", "processor_patch", "forget_gate_quant_fix"):
        monkeypatch.setitem(
            sys.modules,
            "rapid_mlx.patches.glm5_next_" + suffix.rsplit("_", 1)[0],
            Namespace(**{f"install_glm5_next_{suffix}": lambda: None}),
        )
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(mllm, "_resolve_checkpoint", lambda _: str(tmp_path))
    if case == "ok":
        instance = mllm.MLXMultimodalLM(str(tmp_path))
        instance.load()
        assert instance._loaded and isinstance(instance.model, Module)
        return

    engine = object.__new__(BatchedEngine)
    engine._force_mllm = engine._trust_remote_code = False
    engine._model_name = "owner/checkpoint" if case == "shape" else str(tmp_path)
    engine._scheduler_config = Namespace()
    engine._model_load_executor = None
    typed = case in ("extra", "shape")
    with pytest.raises(IncompatibleWeights if typed else ValueError) as caught:
        await engine._start_mllm()
    assert engine._model_load_executor is None
    assert startup_boundary_for(caught.value) == "weights_load"
    if typed:
        assert caught.value.__cause__ is failure
        assert serve_error_class(caught.value) == "incompatible_weights"
    else:
        assert caught.value is failure
        assert serve_error_class(failure) == "other"
