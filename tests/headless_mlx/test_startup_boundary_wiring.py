"""Exercise startup ownership seams using inert MLX imports, never Metal."""

from types import SimpleNamespace

import pytest

from rapid_mlx.telemetry.model_events import serve_failure_subtype


def test_text_scheduler_setup_boundary(monkeypatch):
    from rapid_mlx import engine_core

    error = Exception("private cache")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(engine_core, "Scheduler", fail)
    monkeypatch.setattr(
        engine_core,
        "get_registry",
        lambda: SimpleNamespace(acquire=lambda **kwargs: None),
    )
    with pytest.raises(Exception) as caught:
        engine_core.EngineCore(SimpleNamespace(), SimpleNamespace())
    assert caught.value is error
    assert serve_failure_subtype(error) == "cache_setup"


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["probe", "scheduler"])
async def test_multimodal_cache_setup_boundary(monkeypatch, seam):
    from rapid_mlx import engine_core, mllm_scheduler
    from rapid_mlx.engine import batched
    from rapid_mlx.models import mllm
    from rapid_mlx.scheduler import SchedulerConfig

    error = Exception("private cache")

    def fail(*args, **kwargs):
        raise error

    class FakeMLLM:
        model = SimpleNamespace()
        processor = SimpleNamespace(chat_template="test")
        config = {}

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            pass

    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(mllm, "MLXMultimodalLM", FakeMLLM)
    monkeypatch.setattr(
        batched,
        "_probe_mllm_cache_type",
        fail if seam == "probe" else lambda model: None,
    )
    monkeypatch.setattr(
        batched, "_clone_mllm_worker_processor", lambda processor: processor
    )
    monkeypatch.setattr(mllm_scheduler, "MLLMScheduler", fail)
    engine = batched.BatchedEngine(
        "private", force_mllm=True, scheduler_config=SchedulerConfig()
    )
    try:
        with pytest.raises(Exception) as caught:
            await engine._start_mllm()
        assert caught.value is error
        assert serve_failure_subtype(error) == "cache_setup"
    finally:
        if engine._model_load_executor is not None:
            engine._model_load_executor.shutdown(wait=True)


@pytest.mark.parametrize("native", [True, False])
def test_multimodal_config_load_boundary(monkeypatch, native):
    import sys
    from types import ModuleType

    from rapid_mlx.models import media_backend, mllm

    error = Exception("private config")

    def fail(*args, **kwargs):
        raise error

    utils = ModuleType("mlx_lm.utils" if native else "mlx_vlm.utils")
    utils.load_config = fail
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    vlm = ModuleType("mlx_vlm")
    vlm.load = fail
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setattr(
        media_backend,
        "native_media_model_type",
        lambda name: "lfm3_vl" if native else None,
    )
    monkeypatch.setattr(
        media_backend, "resolve_native_checkpoint", lambda name: "private"
    )
    monkeypatch.setattr(mllm, "_resolve_checkpoint", lambda name: "private")
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    for name, installer in (
        ("glm5_next_runtime", "install_glm5_next_runtime_fix"),
        ("glm5_next_processor", "install_glm5_next_processor_patch"),
        ("glm5_next_forget_gate_quant", "install_glm5_next_forget_gate_quant_fix"),
    ):
        dependency = ModuleType("rapid_mlx.patches." + name)
        setattr(dependency, installer, lambda: None)
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    instance = mllm.MLXMultimodalLM("private", enable_cache=False)
    with pytest.raises(Exception) as caught:
        instance.load()
    assert caught.value is error
    assert serve_failure_subtype(error) == "config_load"
