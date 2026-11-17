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
    monkeypatch.setattr(module.mx, "eval", lambda *_: None)
    monkeypatch.setattr(module.mx, "clear_cache", lambda: None)
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
def test_scheduling_resolution_carries_loaded_model_facts(monkeypatch, mode):
    from types import SimpleNamespace

    import mlx.core as mx

    from rapid_mlx import lattice
    from rapid_mlx.lattice import scheduling

    monkeypatch.setattr(
        mx,
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
    assert scheduling.skip_prefix_boundary(model, 73, mode=mode)
    (ctx,) = contexts
    assert ctx.model.hidden_size == 5120
    assert ctx.hardware.gpu_generation == 17
    assert ctx.hardware.tier == "Max"
    assert ctx.mode == mode


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

    monkeypatch.setattr(
        mx,
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
