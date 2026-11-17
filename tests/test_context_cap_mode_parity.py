# SPDX-License-Identifier: Apache-2.0
"""Plain MLLM capacity and dispatch agree; text/MTP keeps main accounting."""

from types import SimpleNamespace

import pytest

from rapid_mlx.kv_estimation import footprint_from_config


def _config():
    return dict(
        model_type="qwen3_5_text",
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        hidden_size=5120,
        vocab_size=248320,
        full_attention_interval=4,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def _expected_mllm_peak(context, chunk):
    # Two step-rounded full-attention buffers, one recurrent state, workspace.
    return (
        156893184
        + 2 * 65536 * (((context + 255) // 256) * 256)
        + 112 * context * min(context, chunk)
    )


def _expected_mllm_cap(available, minimum=64):
    return max(
        t for t in range(1, 262145) if _expected_mllm_peak(t, minimum) < available
    )


def _expected_mllm_chunk(context, available, configured=2048):
    return max(
        c
        for c in range(1, configured + 1)
        if _expected_mllm_peak(context, c) < available
    )


def _parameters(dtype="bfloat16"):
    return lambda: {"embed_tokens": {"weight": SimpleNamespace(dtype=dtype)}}


@pytest.mark.parametrize(
    ("dtype", "width"), [("float32", 4), ("bfloat16", 2), ("uint32", 4), (None, 4)]
)
def test_live_dtype_controls_unquantized_kv_width(dtype, width):
    from rapid_mlx.kv_estimation import footprint_from_model

    model = SimpleNamespace(
        config=dict(_config(), dtype="bfloat16"), parameters=_parameters(dtype)
    )
    assert footprint_from_model(model) == footprint_from_config(
        dict(
            _config(), layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16
        ),
        width,
    )


def test_quantized_packed_weights_do_not_determine_activation_width():
    from rapid_mlx.kv_estimation import footprint_from_model

    model = SimpleNamespace(
        config=_config(),
        parameters=lambda: {
            "weight": SimpleNamespace(dtype="uint32"),
            "scales": SimpleNamespace(dtype="bfloat16"),
        },
    )
    assert footprint_from_model(model).per_token_growth_bytes == 65536


@pytest.mark.parametrize("broken_parameters", [False, True])
def test_unverifiable_live_dtype_keeps_conservative_fallback(broken_parameters):
    from rapid_mlx.kv_estimation import footprint_from_model

    model = SimpleNamespace(config=dict(_config(), dtype="bfloat16"))
    if broken_parameters:

        def parameters():
            raise RuntimeError("parameters unavailable")

        model.parameters = parameters
    assert footprint_from_model(model) == footprint_from_config(
        dict(
            _config(), layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16
        ),
        4,
    )


def test_mixed_live_parameters_use_widest_floating_dtype():
    from rapid_mlx.kv_estimation import footprint_from_model

    model = SimpleNamespace(
        config=_config(),
        parameters=lambda: {
            "layers": [{"weight": SimpleNamespace(dtype="mlx.core.bfloat16")}],
            "projection": SimpleNamespace(dtype="mlx.core.float32"),
        },
    )
    assert footprint_from_model(model).per_token_growth_bytes == 131072


@pytest.fixture
def prefill_scheduler(monkeypatch):
    """Run the production budget and dispatch methods with no MLX import."""
    import ast
    import logging
    import time
    from pathlib import Path

    from rapid_mlx import (
        errors,
        kv_estimation,
        memory_budget,
    )
    from rapid_mlx import (
        request as request_module,
    )

    source = Path(__file__).resolve().parents[1] / "rapid_mlx" / "scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    methods = {
        "_footprint_from_model_dims",
        "_resolve_kv_bytes_per_token",
        "_resolve_kv_fixed_baseline_bytes",
        "projected_memory_max_context",
        "memory_context_ceiling",
        "_prefill_workspace_units",
        "_prefill_workspace_bytes",
        "_prefill_workspace_at",
        "_prefill_workspace_prior",
        "_resolve_prefill_attention_beta",
        "_bounded_prefill_enabled",
        "_prefill_chunk_for_context",
        "_largest_fitting_prefill_chunk",
        "_prefill_memory_context",
        "_mllm_kv_growth_reserve",
        "_active_prefill_token_count",
        "_select_adaptive_prefill_size",
        "_apply_adaptive_prefill_size",
        "_prefill_cohort_fits",
        "_mllm_prefill_chunk",
        "_enforce_metal_cap_at_admission",
        "fail_step",
        "preflight_metal_admission",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
    ns = dict(vars(kv_estimation))
    ns.update(vars(errors))
    ns.update(vars(request_module))
    ns.update(
        __package__="rapid_mlx",
        Any=object,
        Request=object,
        SchedulerOutput=SimpleNamespace,
        logger=logging.getLogger(__name__),
        time=time,
        _prompt_len=lambda r: len(r.prompt_token_ids or ()),
        _COLD_PREFILL_TRANSIENT_KV_MULTIPLE=3.5,
        _PREFILL_ATTENTION_SCORE_COPIES=8,
        _PREFILL_WORKSPACE_MEASURED_HEADROOM=1.25,
        mx=SimpleNamespace(
            metal=SimpleNamespace(is_available=lambda: True),
            device_info=lambda: {"max_recommended_working_set_size": 27 * 1024**3},
            clear_cache=lambda: None,
            get_cache_memory=lambda: 0,
        ),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), ns)
    monkeypatch.setattr(memory_budget, "process_utilization_floor", lambda: (0, 0))
    obj = ns["Scheduler"]()
    obj.model = SimpleNamespace(
        config=dict(_config(), num_attention_heads=24), parameters=_parameters()
    )
    obj.config = SimpleNamespace(
        prefill_step_size=2048,
        adaptive_prefill=False,
        metal_cap_kv_bytes_per_token=0,
        gpu_memory_utilization=0.9,
        mtp_max_k=3,
    )
    obj._mllm_context_accounting = True
    obj._kv_bytes_per_token_resolved = False
    obj._model_weights_bytes = int(15.2 * 1024**3)
    obj._resolve_metal_cap_bytes = lambda: int(27 * 1024**3 * 0.9)
    obj._current_metal_active_bytes = lambda: obj._model_weights_bytes
    obj._current_process_resident_bytes = obj._current_metal_active_bytes
    obj._bound_prefill_chunk_for_decode = lambda size: size
    obj.running = {}
    obj.waiting = []
    obj._prefill_memory_rows = lambda: []
    obj._reclaim_prefix_cache_for_prefill = lambda *a, **kw: 0
    obj._prefill_memory_sample_minimum = lambda: 256
    obj._prefill_unmaterialized_bytes = lambda r, n: (
        obj._resolve_kv_fixed_baseline_bytes()
        + obj._resolve_kv_bytes_per_token()
        * (r.model_prompt_tokens + r.sampling_params.max_tokens)
    )
    obj._estimate_request_kv_bytes = lambda r: obj._prefill_unmaterialized_bytes(r, 0)
    obj.requests = {}
    obj.memory_aware_cache = None
    obj.paged_cache_manager = None
    obj.num_metal_cap_violations = 0
    obj._metal_cap_warning_logged = False
    obj._metal_cap_effective_utilization = 0.9
    obj._reclaim_unreachable_metal = lambda: False
    obj._has_reclaimable_kv = lambda: False
    obj.observe_idle_resident_floor = lambda active: None
    obj.batch_generator = SimpleNamespace(
        _prompt_batch=SimpleNamespace(tokens=[[]]),
        _currently_processing=[],
        _unprocessed_sequences=[],
    )
    return obj


def _production_methods(filename, class_name, names, namespace):
    import __future__

    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "rapid_mlx" / filename
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[class_name]


@pytest.mark.parametrize("cached", [0, 48000])
@pytest.mark.parametrize("unsafe", [None, "mask", "extra", "media", "over_budget"])
def test_mllm_long_text_dispatch_uses_fitting_chunk(prefill_scheduler, cached, unsafe):
    from rapid_mlx.mllm_prefill import effective_mllm_prefill_chunk

    s = prefill_scheduler
    cap = s.memory_context_ceiling(262144)
    calls = []
    if unsafe == "over_budget":
        cap = 180000

    class Tokens:
        ndim = 2

        def __init__(self, length):
            self.shape = (1, length)

        def __getitem__(self, key):
            index = key[1]
            return Tokens(len(range(self.shape[1])[index]))

    def model(ids, **kwargs):
        calls.append(ids.shape[1])
        return SimpleNamespace(logits="last logits")

    cls = _production_methods(
        "mllm_batch_generator.py",
        "MLLMBatchGenerator",
        {"_vision_encoding_steps"},
        dict(
            Any=object,
            MLLMBatchRequest=object,
            effective_mllm_prefill_chunk=effective_mllm_prefill_chunk,
            _is_text_only_request=lambda r: r.pixel_values is None,
            _attention_mask_is_droppable=lambda mask: mask is None,
            mx=SimpleNamespace(eval=lambda _: None, clear_cache=lambda: None),
        ),
    )
    gen = cls()
    gen.model = model
    # MLLM's default is bumped to 8192, then clamped by the existing helper.
    gen.prefill_step_size = 8192
    gen._prefill_memory_guard = s._mllm_prefill_chunk
    gen.clear_prefix_cache = lambda **kw: True
    gen._record_text_prefill_checkpoint = lambda *a: None
    req = SimpleNamespace(
        input_ids=Tokens(cap - 1 - cached),
        max_tokens=1,
        cached_tokens=cached,
        prefix_boundary=0,
        shared_prefix_store_at=0,
        pixel_values=object() if unsafe == "media" else None,
        attention_mask=object() if unsafe == "mask" else None,
        image_grid_thw=None,
        extra_kwargs={"token_type_ids": object()} if unsafe == "extra" else {},
    )
    steps = gen._vision_encoding_steps(req, [SimpleNamespace(state=None)])
    if unsafe:
        from rapid_mlx.errors import MemoryContextExceededError
        from rapid_mlx.service.helpers import context_overflow_from_memory_error

        with pytest.raises(MemoryContextExceededError) as caught:
            next(steps)
        assert calls == []
        assert context_overflow_from_memory_error(None, caught.value).status_code == 400
        return
    while True:
        try:
            next(steps)
        except StopIteration as done:
            assert done.value == "last logits"
            break
    assert max(calls) == _expected_mllm_chunk(
        cap, s._resolve_metal_cap_bytes() - s._model_weights_bytes
    )
    assert sum(calls) == cap - 1 - cached
    assert calls[-1] == 1


@pytest.mark.parametrize("chunkable", [False, True])
def test_mllm_singleton_budget_guard_rejects_before_forward(
    prefill_scheduler, chunkable
):
    from rapid_mlx.errors import MemoryContextExceededError
    from rapid_mlx.service.helpers import context_overflow_from_memory_error

    s = prefill_scheduler
    prompt = 180000 if chunkable else 32790
    with pytest.raises(MemoryContextExceededError) as caught:
        s._mllm_prefill_chunk(prompt, 32, 2048, chunkable=chunkable)
    error = context_overflow_from_memory_error(None, caught.value)
    assert error.status_code == 400
    assert error.error_payload["code"] == "context_length_exceeded"


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("existing_generator", [False, True])
def test_native_start_wires_mllm_guard(
    monkeypatch, prefill_scheduler, existing_generator, plain
):
    import asyncio
    import logging
    import sys
    from unittest.mock import AsyncMock, Mock

    engine_core = SimpleNamespace(AsyncEngineCore=None, EngineConfig=None)
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine_core", engine_core)

    s = prefill_scheduler
    s.preflight_metal_admission = lambda: None
    candidate = SimpleNamespace(engine=SimpleNamespace(scheduler=s, start=AsyncMock()))
    monkeypatch.setattr(engine_core, "AsyncEngineCore", lambda **kw: candidate)
    monkeypatch.setattr(engine_core, "EngineConfig", lambda **kw: kw)
    # Only the constructor import is needed; never load the GPU scheduler.
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.scheduler",
        SimpleNamespace(SchedulerConfig=lambda: None),
    )
    cls = _production_methods(
        "engine/batched.py",
        "BatchedEngine",
        {"_start_qwen36_native_text_engine"},
        dict(
            __package__="rapid_mlx.engine",
            BaseEngine=object,
            Qwen36NativeCacheTextWrapper=lambda model: model,
            logger=logging.getLogger(__name__),
        ),
    )
    obj = cls()
    obj._model_load_executor = Mock()
    obj.tokenizer = object()
    obj._scheduler_config = object()
    obj._model_name = "local-test-model"
    obj._profile_name = None
    obj._stream_interval = 1
    obj._gpu_memory_utilization = 0.9
    obj._tool_logits_processor_factory = None
    obj._force_hybrid = obj._no_hybrid = False
    obj._force_spec_decode = False
    obj._no_spec_decode = plain
    obj._bind_prompt_host_cache = Mock()
    bg = SimpleNamespace() if existing_generator else None
    obj._mllm_scheduler = SimpleNamespace(batch_generator=bg)
    asyncio.run(obj._start_qwen36_native_text_engine(object()))
    assert obj._engine is candidate
    assert s._mllm_context_accounting is plain
    if not plain:
        assert not hasattr(obj._mllm_scheduler, "_prefill_memory_guard")
        return
    assert obj._mllm_scheduler._prefill_memory_guard == s._mllm_prefill_chunk
    if bg is not None:
        assert bg._prefill_memory_guard == s._mllm_prefill_chunk


@pytest.mark.parametrize("stream", [False, True])
def test_mllm_late_guard_error_survives_delivery(prefill_scheduler, stream):
    import asyncio
    import logging
    from unittest.mock import Mock

    from rapid_mlx import errors
    from rapid_mlx.request import RequestOutput
    from rapid_mlx.service.helpers import context_overflow_from_memory_error

    cls = _production_methods(
        "mllm_scheduler.py",
        "MLLMScheduler",
        {"_step_no_queue", "stream_outputs", "generate"},
        dict(
            vars(errors),
            __package__="rapid_mlx",
            logger=logging.getLogger(__name__),
            MLLMSchedulerOutput=lambda: SimpleNamespace(outputs=[]),
            ClientRequestError=ValueError,
            classify_engine_abort=lambda _: "engine_aborted",
            RequestOutput=RequestOutput,
        ),
    )
    obj = cls()
    request = SimpleNamespace(request_id="plain")
    obj._process_pending_aborts = lambda: None
    obj._schedule_waiting = lambda: []
    obj.running = {"plain": request}
    obj.waiting = []
    obj.request_id_to_uid = {"plain": 7}
    obj._record_terminal_performance = Mock()
    obj._cleanup_finished = Mock()
    obj.batch_generator = Mock()
    obj.batch_generator.active_batch = SimpleNamespace(uids=[7])
    obj.batch_generator.failed_step_uids.return_value = {7}

    def reject():
        prefill_scheduler._mllm_prefill_chunk(180000, 32, 2048, chunkable=True)

    obj.batch_generator.next.side_effect = reject
    output = obj._step_no_queue()
    assert isinstance(output.outputs[0].error, errors.MemoryContextExceededError)
    obj.abort_request = Mock()
    obj.output_queues = {}

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait(output.outputs[0])
        obj.output_queues["plain"] = queue
        obj.add_request_async = AsyncMock(return_value="plain")
        if stream:
            return [o async for o in obj.stream_outputs("plain")]
        return await obj.generate("test")

    from unittest.mock import AsyncMock

    with pytest.raises(errors.MemoryContextExceededError) as caught:
        asyncio.run(run())
    assert context_overflow_from_memory_error(None, caught.value).status_code == 400


def test_lazy_mllm_generator_inherits_guard(monkeypatch, prefill_scheduler):
    import sys
    from unittest.mock import Mock

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        SimpleNamespace(make_sampler=lambda **kw: None),
    )
    cls = _production_methods(
        "mllm_scheduler.py",
        "MLLMScheduler",
        {"_ensure_batch_generator"},
        dict(MLLMBatchGenerator=lambda **kw: SimpleNamespace()),
    )
    obj = cls()
    obj.batch_generator = None
    obj.config = Mock()
    obj.model = obj.processor = obj.mm_processor = object()
    obj.stop_tokens = set()
    obj.cooperative_prefill = True
    obj._prefill_memory_guard = prefill_scheduler._mllm_prefill_chunk
    obj._ensure_batch_generator()
    assert obj.batch_generator._prefill_memory_guard == obj._prefill_memory_guard


def test_mllm_live_allocations_cannot_consume_workspace(prefill_scheduler):
    from rapid_mlx.errors import MetalMemoryBackpressureError

    s = prefill_scheduler
    # It fits against idle weights, but a peer owns the remaining headroom.
    s._current_metal_active_bytes = s._resolve_metal_cap_bytes
    with pytest.raises(MetalMemoryBackpressureError):
        s._mllm_prefill_chunk(32790, 32, 2048, chunkable=True)


@pytest.mark.parametrize("context", [128, 1024])
def test_mllm_keeps_dispatch_default_when_workspace_fits(prefill_scheduler, context):
    s = prefill_scheduler
    # Preserve main's helper clamp even when the dispatch setting is 8192.
    assert s._mllm_prefill_chunk(context - 1, 1, 8192, chunkable=True) == 2048


@pytest.mark.parametrize("context", [40000, 65000, 76000])
def test_mllm_chooses_largest_fitting_chunk(prefill_scheduler, context):
    s = prefill_scheduler
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    chunk = s._mllm_prefill_chunk(context - 1, 1, 8192, chunkable=True)
    available = (
        s._resolve_metal_cap_bytes()
        - s._model_weights_bytes
        - s._resolve_kv_fixed_baseline_bytes()
        - 2 * 65536 * (((context + 255) // 256) * 256)
    )
    assert s._prefill_workspace_at(context, chunk) < available
    assert s._prefill_workspace_at(context, chunk + 1) >= available


def test_mllm_chunk_reduction_logs_once_per_dispatch(prefill_scheduler, caplog):
    import logging

    s = prefill_scheduler
    with caplog.at_level(logging.INFO):
        assert s._mllm_prefill_chunk(1023, 1, 8192, chunkable=True) == 2048
        chunk = s._mllm_prefill_chunk(64999, 1, 8192, chunkable=True)
    records = [r for r in caplog.records if "MLLM prefill chunk bound:" in r.message]
    assert len(records) == 1
    assert f"2048 -> {chunk}, context=65000" in records[0].message


@pytest.mark.parametrize("configured", [32, 64, 2048, 8192])
def test_mllm_cap_uses_dispatch_chunk_policy(prefill_scheduler, configured):
    from rapid_mlx.errors import MemoryContextExceededError

    s = prefill_scheduler
    s._mllm_prefill_step_size = configured
    cap = s.memory_context_ceiling(262144)
    chunk = s._mllm_prefill_chunk(cap - 1, 1, configured, chunkable=True)
    assert chunk == _expected_mllm_chunk(
        cap,
        s._resolve_metal_cap_bytes() - s._model_weights_bytes,
        min(configured, 2048),
    )
    with pytest.raises(MemoryContextExceededError) as caught:
        s._mllm_prefill_chunk(cap, 1, configured, chunkable=True)
    assert caught.value.memory_ceiling == cap


@pytest.mark.parametrize("weights", [15_200_000_000, 15_400_000_000])
def test_reported_m5_budget_chunks_and_cap(prefill_scheduler, weights):
    s = prefill_scheduler
    s._model_weights_bytes = weights
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    s.config.gpu_memory_utilization = 27_100_000_000 / (27 * 1024**3)
    fixed = s._resolve_kv_fixed_baseline_bytes()
    assert fixed == 156893184
    assert s._prefill_workspace_at(77820, 2048) == 3.5 * 65536 * 77820
    from rapid_mlx.errors import MemoryContextExceededError

    available = 27_100_000_000 - weights
    for context in (32550, 77852):
        assert s._mllm_prefill_chunk(
            context - 1, 1, 8192, chunkable=True
        ) == _expected_mllm_chunk(context, available)
    cap = _expected_mllm_cap(available)
    assert cap == (84736 if weights == 15_200_000_000 else 83456)
    assert s.memory_context_ceiling(262144) == cap
    assert s._mllm_prefill_chunk(
        cap - 1, 1, 8192, chunkable=True
    ) == _expected_mllm_chunk(cap, available)
    with pytest.raises(MemoryContextExceededError):
        s._mllm_prefill_chunk(130481, 32, 64, chunkable=True)


def test_previous_measured_plain_budget_uses_calibrated_cap(prefill_scheduler):
    s = prefill_scheduler
    # fix5 reported 95,246 with growth 65,536 and 768 * 64 workspace.
    # Preserve its inferred request budget to isolate this calibration change.
    available = 156893184 + 114688 * 95246 + 1
    cap_bytes = s._model_weights_bytes + available
    s._resolve_metal_cap_bytes = lambda: cap_bytes
    s.config.gpu_memory_utilization = cap_bytes / (27 * 1024**3)
    assert s.memory_context_ceiling(262144) == _expected_mllm_cap(available)
    assert [
        s._mllm_prefill_chunk(c - 1, 1, 8192, chunkable=True)
        for c in (22893, 32768, 78000)
    ] == [_expected_mllm_chunk(c, available) for c in (22893, 32768, 78000)]


@pytest.mark.parametrize("context", [2, 1024, 32550, 65536, 76800])
@pytest.mark.parametrize("cached", [False, True])
def test_every_idle_context_through_cap_fits(prefill_scheduler, context, cached):
    s = prefill_scheduler
    s.config.gpu_memory_utilization = 27_100_000_000 / (27 * 1024**3)
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    assert s.memory_context_ceiling(262144) == 76800
    prefix = context // 2 if cached else 0
    fixed = s._resolve_kv_fixed_baseline_bytes()
    growth = s._resolve_kv_bytes_per_token()
    s._current_metal_active_bytes = lambda: (
        s._model_weights_bytes + (fixed + growth * prefix if prefix else 0)
    )
    chunk = s._mllm_prefill_chunk(
        context - 1, 1, 2048, chunkable=True, cached_tokens=prefix
    )
    assert (
        fixed
        + 2 * growth * (((context + 255) // 256) * 256)
        + s._prefill_workspace_at(context, min(chunk, context - 1))
        < 27_100_000_000 - s._model_weights_bytes
    )


def test_mllm_cap_exact_boundary(prefill_scheduler):
    from rapid_mlx.errors import MemoryContextExceededError

    s = prefill_scheduler
    cap = s.memory_context_ceiling(262144)
    # Exhaust the whole finite advertised range, including chunk transitions.
    for context in range(2, cap + 1):
        s._mllm_prefill_chunk(context - 1, 1, 2048, chunkable=True)
    with pytest.raises(MemoryContextExceededError):
        s._mllm_prefill_chunk(cap, 1, 2048, chunkable=True)
    with pytest.raises(MemoryContextExceededError):
        s._mllm_prefill_chunk(180000, 32, 2048, chunkable=True)


def test_mllm_reclaims_retained_prefix_before_backpressure(prefill_scheduler):
    s = prefill_scheduler
    cap = s.memory_context_ceiling(262144)
    active = [s._model_weights_bytes + 3 * 1024**3]
    s._current_metal_active_bytes = lambda: active[0]
    calls = []

    def reclaim():
        calls.append(True)
        active[0] = s._model_weights_bytes

    assert s._mllm_prefill_chunk(
        cap - 1, 1, 2048, chunkable=True, reclaim=reclaim
    ) == _expected_mllm_chunk(
        cap, s._resolve_metal_cap_bytes() - s._model_weights_bytes
    )
    assert calls == [True]


@pytest.mark.parametrize(
    "layout,growth,fixed",
    [
        ("explicit", 65536, 156893184),
        ("interval", 524288, 0),
        ("dense", 262144, 0),
        ("unknown", 524288, 0),
    ],
)
@pytest.mark.parametrize("holder", ["config", "args"])
def test_mtp_estimate_and_cap_match_main(
    prefill_scheduler, layout, growth, fixed, holder
):
    # Golden terms from main's unmodified estimator at the task base.
    # Explicit bf16 checkpoint map is the actual MTP lane; a live dataclass
    # without dtype/map intentionally keeps main's conservative fallback.
    s = prefill_scheduler
    s._mllm_context_accounting = False
    config = _config()
    if layout == "explicit":
        config.update(
            dtype="bfloat16",
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
        )
    elif layout == "dense":
        config.update(dtype="bfloat16", layer_types=["full_attention"] * 64)
    elif layout == "unknown":
        config["model_type"] = "unknown"
    s.model = SimpleNamespace(
        **{holder: {"text_config": config}},
        parameters=_parameters(),
        mtp=SimpleNamespace(layers=[object()]),
    )
    assert s._resolve_kv_bytes_per_token() == growth
    assert s._resolve_kv_fixed_baseline_bytes() == fixed
    expected = (
        s._resolve_metal_cap_bytes() - s._model_weights_bytes - fixed - 1
    ) // growth
    assert s.memory_context_ceiling(262144) == expected
    # The MLLM guard cannot alter text/MTP admission or chunks.
    assert s._mllm_prefill_chunk(180000, 32, 2048, chunkable=True) == 2048
    assert not s._bounded_prefill_enabled()


def test_measured_mtp_cap_is_main_cap(prefill_scheduler):
    s = prefill_scheduler
    s._mllm_context_accounting = False
    config = dict(
        _config(),
        dtype="bfloat16",
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
    )
    s.model = SimpleNamespace(
        args={"text_config": config}, mtp=SimpleNamespace(layers=[object()])
    )
    s.config.gpu_memory_utilization = 27_100_000_000 / (27 * 1024**3)
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    assert s.memory_context_ceiling(262144) == 162082


def test_mllm_layout_recovery_does_not_change_shared_estimator():
    from rapid_mlx.kv_estimation import footprint_from_model

    config = _config()
    model = SimpleNamespace(config=config, parameters=_parameters())
    assert footprint_from_config(config).per_token_growth_bytes == 524288
    recovered = footprint_from_model(model)
    assert recovered.per_token_growth_bytes == 65536
    assert recovered.fixed_baseline_bytes == 156893184
    assert "layer_types" not in config


def test_plain_preflight_counts_shared_owner_residency(prefill_scheduler):
    s = prefill_scheduler
    loaded = s._model_weights_bytes + 512 * 1024**2
    s._current_metal_active_bytes = lambda: loaded
    s.preflight_metal_admission()
    assert s._steady_resident_bytes == loaded
    available = s._resolve_metal_cap_bytes() - loaded
    assert s.memory_context_ceiling(262144) == _expected_mllm_cap(available)


@pytest.mark.parametrize(
    "changes",
    [
        {"full_attention_interval": 0},
        {"full_attention_interval": True},
        {"full_attention_interval": "4"},
        {"model_type": "unknown"},
        {"layer_types": ["linear_attention"]},
        {"linear_conv_kernel_dim": None},
    ],
)
def test_mllm_recovery_rejects_ambiguous_layout(changes):
    from rapid_mlx.kv_estimation import footprint_from_model

    model = SimpleNamespace(config=dict(_config(), **changes), parameters=_parameters())
    assert footprint_from_model(model).fixed_baseline_bytes == 0


def test_mllm_explicit_map_matches_live_interval():
    from rapid_mlx.kv_estimation import footprint_from_model

    live = SimpleNamespace(
        config=SimpleNamespace(**_config()), parameters=_parameters()
    )
    raw = dict(
        _config(),
        dtype="bfloat16",
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
    )
    checkpoint = SimpleNamespace(args={"text_config": raw}, parameters=_parameters())
    assert footprint_from_model(live) == footprint_from_model(checkpoint)


def test_text_singleton_keeps_main_progress_rule(prefill_scheduler):
    s = prefill_scheduler
    s._mllm_context_accounting = False
    request = SimpleNamespace(
        request_id="text-singleton",
        model_prompt_tokens=180000,
        num_prompt_tokens=180000,
        prompt_token_ids=[1] * 180000,
        cached_tokens=0,
        sampling_params=SimpleNamespace(max_tokens=32),
    )
    assert s._prefill_cohort_fits(request)


@pytest.mark.parametrize(
    "extra_resident,at_cap", [(0, False), (0, True), (256 * 1024**2, False)]
)
def test_sequential_plain_requests_reclaim_and_chunk(
    prefill_scheduler, extra_resident, at_cap
):
    """Finished KV and allocator slabs must yield before the next forward."""
    from rapid_mlx.mllm_prefill import effective_mllm_prefill_chunk

    s = prefill_scheduler
    s._model_weights_bytes = 15_400_000_000
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    s.config.gpu_memory_utilization = 27_100_000_000 / (27 * 1024**3)
    retained = [0]
    pending_free = [0]
    events = []
    s._current_metal_active_bytes = lambda: (
        s._model_weights_bytes + extra_resident + retained[0] + pending_free[0]
    )

    class Tokens:
        ndim = 2

        def __init__(self, length):
            self.shape = (1, length)

        def __getitem__(self, key):
            return Tokens(len(range(self.shape[1])[key[1]]))

    calls = []

    def model(ids, **kwargs):
        calls.append(ids.shape[1])
        return SimpleNamespace(logits="last logits")

    def clear_allocator():
        events.append("allocator")
        pending_free[0] = 0

    def clear_prefix(**kwargs):
        assert kwargs == {"reset_stats": False}
        events.append("prefix")
        pending_free[0] += retained[0]
        retained[0] = 0

    # Include the real reclamation at the guard, not a callback which magically
    # resets active to weights (the old test hid the missing allocator flush).
    guard_globals = s._mllm_prefill_chunk.__func__.__globals__
    guard_globals["mx"].clear_cache = clear_allocator
    cls = _production_methods(
        "mllm_batch_generator.py",
        "MLLMBatchGenerator",
        {"_vision_encoding_steps"},
        dict(
            Any=object,
            effective_mllm_prefill_chunk=effective_mllm_prefill_chunk,
            _is_text_only_request=lambda r: True,
            _attention_mask_is_droppable=lambda mask: True,
            mx=SimpleNamespace(eval=lambda _: None, clear_cache=lambda: None),
        ),
    )
    gen = cls()
    gen.model = model
    gen.prefill_step_size = 8192
    gen.completion_batch_size = 1
    gen.active_batch = None
    gen._prefill_memory_guard = s._mllm_prefill_chunk
    gen.clear_prefix_cache = clear_prefix
    gen._record_text_prefill_checkpoint = lambda *a: None
    final_prompt = s.memory_context_ceiling(262144) - 32 if at_cap else 80000
    for prompt in (32518, 77820, final_prompt):
        assert prompt + 32 <= s.memory_context_ceiling(262144)
        req = SimpleNamespace(
            input_ids=Tokens(prompt),
            max_tokens=32,
            cached_tokens=0,
            prefix_boundary=0,
            shared_prefix_store_at=0,
            pixel_values=None,
            attention_mask=None,
            image_grid_thw=None,
            extra_kwargs={},
        )
        calls.clear()
        events.clear()
        list(gen._vision_encoding_steps(req, [SimpleNamespace(state=None)]))
        assert sum(calls) == prompt
        assert calls[-1] == 1
        assert max(calls) == _expected_mllm_chunk(
            prompt + 32,
            s._resolve_metal_cap_bytes() - s._model_weights_bytes - extra_resident,
        )
        if prompt >= 40000:
            assert 64 <= max(calls) < 2048
            assert events == ["prefix", "allocator"]
        # Simulate request completion retaining detached KV from that request.
        retained[0] = 1024**3


@pytest.mark.parametrize("idle", [False, True])
def test_mllm_post_reclamation_failure_reports_budget(prefill_scheduler, caplog, idle):
    import logging

    from rapid_mlx.errors import (
        MemoryContextExceededError,
        MetalMemoryBackpressureError,
    )
    from rapid_mlx.service.helpers import context_overflow_from_memory_error

    s = prefill_scheduler
    old_cap = s.memory_context_ceiling(262144)
    active = s._model_weights_bytes + 1024**3
    s._current_metal_active_bytes = lambda: active
    reclaimed = []
    s._reclaim_unreachable_metal = lambda: reclaimed.append("gc")
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(
            MemoryContextExceededError if idle else MetalMemoryBackpressureError
        ) as caught,
    ):
        s._mllm_prefill_chunk(old_cap - 32, 32, 2048, chunkable=True, idle=idle)
    message = next(
        r.message for r in caplog.records if "prefill rejected:" in r.message
    )
    for term in (
        "limit",
        "active",
        "cache",
        "resident",
        "kv_needed",
        "growth_reserve",
        "workspace",
        "chunk",
    ):
        assert f"{term}=" in message
    assert f"active={active}" in message
    assert "chunk=0" in message
    if idle:
        assert reclaimed == ["gc"]
        assert context_overflow_from_memory_error(None, caught.value).status_code == 400
        assert s.memory_context_ceiling(262144) == caught.value.memory_ceiling < old_cap
        # Subsequent requests fail the cap check without entering a forward.
        with pytest.raises(MemoryContextExceededError):
            s._mllm_prefill_chunk(old_cap - 32, 32, 2048, chunkable=True, idle=True)
    else:
        assert reclaimed == []
        assert s.memory_context_ceiling(262144) == old_cap


@pytest.mark.parametrize("context", [32518, 32768, 77820, 130481])
@pytest.mark.parametrize(
    "ratio,expected", [(0.69, 2048), (0.70, 1024), (0.80, 512), (0.88, 256)]
)
@pytest.mark.parametrize("use_rss", [False, True])
def test_mllm_and_text_share_pressure_thresholds(
    prefill_scheduler, context, ratio, expected, use_rss
):
    s = prefill_scheduler
    s.config.adaptive_prefill = True
    s._resolve_metal_cap_bytes = lambda: 100_000
    s._current_metal_active_bytes = lambda: 0 if use_rss else int(ratio * 100_000)
    s._current_process_resident_bytes = lambda: int(ratio * 100_000) if use_rss else 0
    s._active_prefill_token_count = lambda: context
    text = s._select_adaptive_prefill_size()
    mllm = s._mllm_prefill_chunk(context, 32, 2048, chunkable=True, adaptive_only=True)
    assert mllm == text == (expected if context >= 32768 else 2048)


def test_mllm_prefill_samples_pressure_and_never_regrows(prefill_scheduler):
    import logging

    from rapid_mlx.mllm_prefill import effective_mllm_prefill_chunk

    s = prefill_scheduler
    s.config.adaptive_prefill = True
    s._resolve_metal_cap_bytes = lambda: 100_000_000_000
    active = [69_000_000_000]
    s._current_metal_active_bytes = lambda: active[0]
    calls = []

    class Tokens:
        ndim = 2

        def __init__(self, length):
            self.shape = (1, length)

        def __getitem__(self, key):
            return Tokens(len(range(self.shape[1])[key[1]]))

    def model(ids, **kwargs):
        calls.append(ids.shape[1])
        active[0] = [70, 80, 88, 60][min(len(calls) - 1, 3)] * 1_000_000_000
        return SimpleNamespace(logits=None)

    cls = _production_methods(
        "mllm_batch_generator.py",
        "MLLMBatchGenerator",
        {"_vision_encoding_steps"},
        dict(
            effective_mllm_prefill_chunk=effective_mllm_prefill_chunk,
            _is_text_only_request=lambda r: True,
            _attention_mask_is_droppable=lambda mask: True,
            logger=logging.getLogger(__name__),
            mx=SimpleNamespace(eval=lambda _: None, clear_cache=lambda: None),
        ),
    )
    gen = cls()
    gen.model = model
    gen.prefill_step_size = 8192
    gen._prefill_memory_guard = s._mllm_prefill_chunk
    gen.clear_prefix_cache = lambda **kw: None
    gen._record_text_prefill_checkpoint = lambda *a: None
    req = SimpleNamespace(
        input_ids=Tokens(10000),
        max_tokens=32,
        cached_tokens=90000,
        prefix_boundary=0,
        shared_prefix_store_at=0,
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        extra_kwargs={},
    )
    list(gen._vision_encoding_steps(req, [SimpleNamespace(state=None)]))
    assert calls[:5] == [2048, 1024, 512, 256, 256]
    assert max(calls[3:]) == 256
    assert sum(calls) == 10000
    assert calls[-1] == 1


def test_shared_workspace_prior_preserves_fractional_text_estimate(prefill_scheduler):
    s = prefill_scheduler
    # Preserve text-lane arithmetic for geometries not divisible by 2048.
    s._resolve_kv_bytes_per_token = lambda: 12345
    request = SimpleNamespace(prompt_token_ids=list(range(1000)), cached_tokens=0)
    expected = int(3.5 * 12345 / 2048 * 1000 * 1000)
    assert s._prefill_workspace_at(1000, 1000) == expected
    assert s._prefill_workspace_bytes(request) == expected


@pytest.mark.parametrize("cached_tokens", [0, 86000])
def test_mllm_growth_copy_rejects_130k_even_with_tiny_chunks(
    prefill_scheduler, cached_tokens
):
    from rapid_mlx.errors import MemoryContextExceededError

    s = prefill_scheduler
    s._model_weights_bytes = 15_400_000_000
    s._resolve_metal_cap_bytes = lambda: 27_100_000_000
    s.config.gpu_memory_utilization = 27_100_000_000 / (27 * 1024**3)
    fixed = s._resolve_kv_fixed_baseline_bytes()
    s._current_metal_active_bytes = lambda: (
        s._model_weights_bytes + (fixed + 65536 * cached_tokens if cached_tokens else 0)
    )
    assert 83000 < s.memory_context_ceiling(262144) < 90000
    with pytest.raises(MemoryContextExceededError):
        s._mllm_prefill_chunk(
            130481, 32, 64, chunkable=True, cached_tokens=cached_tokens
        )


@pytest.mark.parametrize("context", [1, 255, 256, 257, 77852, 130513])
def test_growth_reserve_rounds_both_buffers_and_leaves_mtp_unchanged(
    prefill_scheduler, context
):
    s = prefill_scheduler
    reserve = s._mllm_kv_growth_reserve(context, 65536)
    assert 65536 * context + reserve == 2 * 65536 * (((context + 255) // 256) * 256)
    s._mllm_context_accounting = False
    assert s._mllm_kv_growth_reserve(context, 65536) == 0


def test_reused_prefix_still_needs_growth_buffer(prefill_scheduler):
    from rapid_mlx.errors import MetalMemoryBackpressureError

    s = prefill_scheduler
    context = 32000
    fixed = s._resolve_kv_fixed_baseline_bytes()
    cached = fixed + 65536 * (context - 1)
    # Almost all persistent KV is already resident, but the next growth copy
    # cannot fit. The prefix credit must not cancel that copy reservation.
    s._current_metal_active_bytes = lambda: (
        s._resolve_metal_cap_bytes() - 65536 * context
    )
    with pytest.raises(MetalMemoryBackpressureError):
        s._mllm_prefill_chunk(
            context - 1, 1, 64, chunkable=True, cached_tokens=context - 1
        )
    assert cached > 0
