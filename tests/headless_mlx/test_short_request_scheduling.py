# SPDX-License-Identifier: Apache-2.0
"""Run tensor-free scheduler/admission contracts with the headless import seam."""

import numpy as np
import pytest

from rapid_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from tests.test_admission_wave import (  # noqa: F401
    sleep_spy,
    test_admission_legacy_env_wins_over_lattice,
    test_garbage_env_falls_back_to_default,
    test_lattice_admission_window,
    test_lone_request_pays_one_tick,
    test_nonfinite_env_falls_back_to_default,
)
from tests.test_agent_session_prefix_cache import (  # noqa: F401
    test_prefix_boundary_min_tokens_scheduling,
    test_prefix_threshold_cached_insert_failure_rechecks_cold_length,
    test_prefix_threshold_leaves_trimmable_policy_unchanged,
    test_prefix_threshold_preserves_fetch_for_a_short_warm_tail,
)


@pytest.mark.parametrize(
    "threshold,length,cached,recurrent,shared,shapes,stored",
    [
        (None, 73, 0, True, 0, [58, 14, 1], [58]),
        (0, 73, 0, True, 0, [58, 14, 1], [58]),
        (128, 73, 0, True, 0, [73], []),
        (128, 128, 0, True, 0, [58, 69, 1], [58]),
        (128, 200, 150, True, 0, [50], []),
        (128, 73, 0, False, 0, [58, 14, 1], [58]),
        (128, 73, 0, True, 60, [60, 12, 1], [60]),
    ],
)
def test_mllm_text_boundary_shapes_without_metal(
    monkeypatch, threshold, length, cached, recurrent, shared, shapes, stored
):
    from types import SimpleNamespace

    import rapid_mlx.mllm_batch_generator as module

    if threshold is None:
        monkeypatch.delenv("RAPID_MLX_PREFIX_BOUNDARY_MIN_TOKENS", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_PREFIX_BOUNDARY_MIN_TOKENS", str(threshold))
    # Patch the namespace directly: setattr/delattr on a MagicMock can leave
    # deleted-child markers that break later parameter cases.
    monkeypatch.setitem(vars(module.mx), "eval", lambda *_: None)
    monkeypatch.setitem(vars(module.mx), "clear_cache", lambda: None)
    calls, stores = [], []

    class Model:
        def __call__(self, ids, **kwargs):
            calls.append(ids.shape[1])
            return SimpleNamespace(logits=np.zeros((1, 1, 8)))

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = gen.language_model = Model()
    gen.prefill_step_size = 512
    gen._record_text_prefill_checkpoint = lambda *_: None
    gen._store_exact_text_prefix = lambda req, cache, prefix_len: stores.append(
        prefix_len
    )

    # The normal text single-forward path delegates through this media-aware
    # helper. No media inputs or tensor computation are involved in this seam.
    def single(request, ids, cache, kwargs, **extra):
        if False:
            yield
        return gen.model(ids, cache=cache, **kwargs)

    gen._media_forward_steps = single
    request = MLLMBatchRequest(
        uid=0,
        request_id="test",
        prompt="test",
        input_ids=np.arange(length)[cached:],
        prefix_boundary=58 if not cached else 175,
        cached_tokens=cached,
        shared_prefix_store_at=shared,
    )
    cache = [SimpleNamespace(state=[], is_trimmable=lambda: not recurrent)]
    gen._run_vision_encoding(request, cache)
    assert calls == shapes
    assert stores == stored


@pytest.mark.parametrize("mode", ["plain", "mtp"])
@pytest.mark.parametrize("batch", [1, 2])
def test_scheduling_resolution_carries_loaded_model_facts(monkeypatch, mode, batch):
    from types import SimpleNamespace

    import mlx.core as mx

    from rapid_mlx import lattice
    from rapid_mlx.lattice import scheduling

    monkeypatch.setitem(
        vars(mx),
        "device_info",
        lambda: {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120)
        )
    )
    contexts = []

    def resolve(name, ctx):
        contexts.append(ctx)
        return lattice.Decision(128, "test")

    monkeypatch.setattr(scheduling, "resolve", resolve)
    assert scheduling.skip_prefix_boundary(model, 73, mode=mode, batch=batch)
    (ctx,) = contexts
    assert ctx.model.hidden_size == 5120
    assert ctx.hardware.gpu_generation == 17
    assert ctx.hardware.tier == "Max"
    assert ctx.mode == mode
    assert ctx.batch == batch


@pytest.mark.parametrize("waiting,expected", [(1, []), (2, [0.008]), (8, [0.008])])
def test_measured_admission_cell_preserves_burst_hold(
    monkeypatch,
    sleep_spy,  # noqa: F811 -- imported pytest fixture
    waiting,
    expected,
):
    from types import SimpleNamespace

    import mlx.core as mx

    from rapid_mlx import lattice
    from rapid_mlx.lattice import defaults
    from tests.test_admission_wave import _make_engine, _run

    monkeypatch.setitem(
        vars(mx),
        "device_info",
        lambda: {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
    )
    monkeypatch.setattr("rapid_mlx.lattice.profiles.mlx_version", lambda: "0.32.4")
    monkeypatch.setattr("rapid_mlx.lattice.profiles.macos_version", lambda: "27.0.1")
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    monkeypatch.delenv("RAPID_ADMISSION_WINDOW_MS", raising=False)
    eng = _make_engine()
    weight = SimpleNamespace(weight=object(), bits=4, group_size=64, mode="affine")
    eng.model = SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120),
        named_modules=lambda: [("model.layers.0.mlp.up_proj", weight)],
    )
    eng.scheduler.config = SimpleNamespace(spec_decode="mtp")
    eng.scheduler.get_num_waiting = lambda: waiting
    calls, _, holder = sleep_spy
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == pytest.approx(expected)


@pytest.fixture
def qualified_short_request_model(monkeypatch):
    from types import SimpleNamespace

    import mlx.core as mx

    from rapid_mlx import lattice
    from rapid_mlx.lattice import defaults

    monkeypatch.setitem(
        vars(mx),
        "device_info",
        lambda: {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
    )
    monkeypatch.setattr("rapid_mlx.lattice.profiles.mlx_version", lambda: "0.32.4")
    monkeypatch.setattr("rapid_mlx.lattice.profiles.macos_version", lambda: "27.0.1")
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_PREFIX_BOUNDARY_MIN_TOKENS", raising=False)
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    weight = SimpleNamespace(weight=object(), bits=4, group_size=64, mode="affine")
    return SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120),
        named_modules=lambda: [("model.layers.0.mlp.up_proj", weight)],
    )


@pytest.mark.parametrize(
    "running,admitted,capacity", [(0, 1, 4), (0, 2, 4), (1, 1, 4), (0, 2, 1)]
)
@pytest.mark.parametrize("mode", ["plain", "mtp"])
def test_qualified_boundary_admission_batch(
    monkeypatch, qualified_short_request_model, running, admitted, capacity, mode
):
    from unittest.mock import MagicMock

    from tests.test_agent_session_prefix_cache import _register, _scheduler

    sched = _scheduler(
        monkeypatch,
        non_trimmable_exact_prefix_reuse=True,
        spec_decode="mtp" if mode == "mtp" else "none",
        max_num_seqs=capacity,
    )
    sched.model = qualified_short_request_model
    # A runtime that has declined MTP can admit the ordinary decode width;
    # scheduling still resolves the request's configured MTP policy.
    sched.spec_decode_runtime_attempted = True
    sched.spec_decode_runtime_method = None
    for i in range(running):
        req = _register(sched, 300 + i, 73)
        req.output_token_ids = [999]
        sched.running[req.request_id] = req
    requests = [_register(sched, 400 + i, 73) for i in range(admitted)]
    for i, req in enumerate(requests):
        req.prompt_token_ids = [token + i * 1000 for token in req.prompt_token_ids]
        req.remaining_tokens = req.prompt_token_ids
        req.prefix_boundary = 58
        req.exact_prefix_boundary = True
        sched.waiting.append(req)
    generator = MagicMock()
    generator.insert.side_effect = [[400 + i] for i in range(admitted)]
    generator.insert_segments.side_effect = [[400 + i] for i in range(admitted)]
    sched.batch_generator = generator
    monkeypatch.setattr(sched, "_ensure_batch_generator", lambda *_: True)
    monkeypatch.setattr(sched, "_get_request_sampler", lambda _: MagicMock())
    monkeypatch.setattr(sched, "_register_uid_processors", MagicMock())
    scheduled = sched._schedule_waiting()
    assert scheduled == requests[: capacity - running]
    skip = mode == "mtp" and min(capacity, running + admitted) == 1
    assert generator.insert.call_count == (len(scheduled) if skip else 0)
    assert generator.insert_segments.call_count == (0 if skip else len(scheduled))
    for req in scheduled:
        assert getattr(req, "_cache_snapshot_boundary", req.prefix_boundary) == (
            0 if skip else 58
        )


@pytest.mark.parametrize("running,admitted", [(0, 1), (0, 2), (1, 1)])
@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("qualified", [False, True])
def test_mllm_boundary_uses_prompt_cohort(
    monkeypatch,
    qualified_short_request_model,
    running,
    admitted,
    cooperative,
    qualified,
):
    from contextlib import nullcontext
    from copy import deepcopy
    from importlib import import_module
    from types import SimpleNamespace

    from rapid_mlx import lattice
    from rapid_mlx import mllm_batch_generator as module
    from rapid_mlx.lattice import defaults
    from rapid_mlx.lattice.scheduling import scheduling_value

    cache_module = import_module("mlx_lm.models.cache")

    # The measured builtin is MTP-only. A plain-mode test overlay with the
    # same qualification exercises the MLLM consumer without changing policy.
    if qualified:
        cell = deepcopy(
            next(
                cell
                for cell in defaults.builtin_profile().cells
                if "prefix_boundary_min_tokens" in cell["knobs"]
            )
        )
        cell["workload"]["mode"] = "plain"
        monkeypatch.setattr(
            lattice, "overlay_profile", lambda: defaults.Profile("test", (cell,))
        )
    calls, stores = [], []

    class Model:
        args = qualified_short_request_model.args
        named_modules = staticmethod(qualified_short_request_model.named_modules)

        def __call__(self, ids, **kwargs):
            calls.append(ids.shape[1])
            return SimpleNamespace(logits=np.zeros((1, 1, 8)))

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = gen.language_model = Model()

    assert scheduling_value("prefix_boundary_min_tokens", gen.model) == (
        256 if qualified else 0
    )
    gen.processor = SimpleNamespace(tokenizer=object())
    gen.prefill_step_size = 512
    gen.active_batch = [object()] * running if running else None
    gen._stats = SimpleNamespace(prompt_tokens=0)
    gen._stream = object()
    gen.vision_prefill_token_budget = 8192
    gen._preprocess_request = lambda _: None
    gen._lookup_exact_text_prefix = lambda _: None
    gen._record_text_prefill_checkpoint = lambda *_: None
    gen._store_exact_text_prefix = lambda req, cache, prefix_len: stores.append(
        prefix_len
    )
    monkeypatch.setitem(vars(module.mx), "eval", lambda *_: None)
    monkeypatch.setitem(vars(module.mx), "clear_cache", lambda: None)
    monkeypatch.setitem(vars(module.mx), "stream", lambda _: nullcontext())
    monkeypatch.setattr(module, "_prefill_cap_violation", lambda *_: None)
    monkeypatch.setitem(
        vars(cache_module),
        "make_prompt_cache",
        lambda _: [SimpleNamespace(state=[], is_trimmable=lambda: False)],
    )

    def single(request, ids, cache, kwargs, **extra):
        if False:
            yield
        return gen.model(ids, cache=cache, **kwargs)

    gen._media_forward_steps = single

    class PrefillCompleteError(Exception):
        pass

    # Stop after the first real prefill, before tensor sampling/cache merge.
    # Its boundary must already account for every later row in the cohort.
    encode = gen._vision_encoding_steps

    def stop_after_prefill(*args, **kwargs):
        yield from encode(*args, **kwargs)
        raise PrefillCompleteError

    gen._vision_encoding_steps = stop_after_prefill
    requests = [
        MLLMBatchRequest(
            uid=i,
            request_id=str(i),
            prompt="test",
            input_ids=np.arange(73),
            prefix_boundary=58,
        )
        for i in range(admitted)
    ]
    with pytest.raises(PrefillCompleteError):
        list(gen._process_prompt_steps(requests, cooperative=cooperative))
    skip = qualified and running + admitted == 1
    assert calls == ([73] if skip else [58, 14, 1])
    assert stores == ([] if skip else [58])
