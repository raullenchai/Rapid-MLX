# SPDX-License-Identifier: Apache-2.0
"""Live speculative configuration vs request-level fallback contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("method", [None, "", "none", " NONE ", 7])
def test_disabled_speculative_methods_have_no_live_policy(method):
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    assert resolve_speculative_request_policy(method) is None


def test_mtp_policy_reports_tools_as_safe_ordinary_decode_fallback():
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    policy = resolve_speculative_request_policy(" MTP ")
    assert policy is not None
    assert policy.method == "mtp"
    assert policy.request_fallback_features == ("tools",)


def test_other_speculative_methods_do_not_inherit_mtp_tool_policy():
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    policy = resolve_speculative_request_policy("suffix")
    assert policy is not None
    assert policy.method == "suffix"
    assert policy.request_fallback_features == ()


def test_model_profile_reads_policy_from_matching_live_scheduler(monkeypatch):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(
        config=SimpleNamespace(spec_decode="mtp"),
        spec_decode_runtime_method="mtp",
        spec_decode_runtime_attempted=True,
    )
    engine = object()
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda candidate: scheduler)

    info = models_route._resolve_speculative_decoding("served-model")

    assert info is not None
    assert info.configured is True
    assert info.method == "mtp"
    assert info.runtime_state == "active"
    assert info.request_fallback_features == ["tools"]
    assert info.model_dump() == {
        "configured": True,
        "method": "mtp",
        "runtime_state": "active",
        "request_fallback_features": ["tools"],
    }


def test_model_profile_reports_pending_before_lazy_runtime_install(
    monkeypatch,
):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(
        config=SimpleNamespace(spec_decode="mtp"),
        spec_decode_runtime_method=None,
        spec_decode_runtime_attempted=False,
    )
    engine = object()
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda candidate: scheduler)

    info = models_route._resolve_speculative_decoding("served-model")

    assert info is not None
    assert info.runtime_state == "pending"


def test_model_profile_reports_unavailable_after_runtime_install_gate_miss(
    monkeypatch,
):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(
        config=SimpleNamespace(spec_decode="mtp"),
        spec_decode_runtime_method=None,
        spec_decode_runtime_attempted=True,
    )
    engine = object()
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda candidate: scheduler)

    info = models_route._resolve_speculative_decoding("served-model")

    assert info is not None
    assert info.runtime_state == "unavailable"


def test_model_profile_reads_legacy_suffix_configuration(monkeypatch):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(
        config=SimpleNamespace(
            spec_decode="none",
            enable_suffix_decoding=True,
        ),
        spec_decode_runtime_method="suffix",
        spec_decode_runtime_attempted=True,
    )
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: object())
    monkeypatch.setattr(models_route, "_scheduler_of", lambda _engine: scheduler)

    info = models_route._resolve_speculative_decoding("served-model")

    assert info is not None
    assert info.method == "suffix"
    assert info.runtime_state == "active"


def test_model_profile_omits_disabled_speculative_configuration(monkeypatch):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(
        config=SimpleNamespace(
            spec_decode="none",
            enable_suffix_decoding=False,
        )
    )
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: object())
    monkeypatch.setattr(models_route, "_scheduler_of", lambda _engine: scheduler)

    assert models_route._resolve_speculative_decoding("served-model") is None


def test_model_profile_fails_closed_when_policy_probe_raises(monkeypatch):
    from vllm_mlx.routes import models as models_route
    from vllm_mlx.speculative import request_policy

    scheduler = SimpleNamespace(config=SimpleNamespace(spec_decode="mtp"))
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: object())
    monkeypatch.setattr(models_route, "_scheduler_of", lambda _engine: scheduler)

    def fail_probe(_method):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        request_policy,
        "resolve_speculative_request_policy",
        fail_probe,
    )

    assert models_route._resolve_speculative_decoding("served-model") is None


def _stub_runtime_scheduler(monkeypatch, scheduler_module, **config_overrides):
    """Build the narrow real-Scheduler seam shared by installer-state tests."""

    batch_generator = SimpleNamespace()
    monkeypatch.setattr(
        scheduler_module,
        "BatchGenerator",
        lambda *args, **kwargs: batch_generator,
    )
    monkeypatch.setattr(
        scheduler_module,
        "_install_dense_sampler_fastpath",
        lambda _batch_generator: None,
    )
    monkeypatch.setattr(
        "vllm_mlx.singleton_cache_fastpath.install_singleton_cache_fastpath",
        lambda: None,
    )

    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.model = object()
    scheduler.tokenizer = object()
    scheduler.model_config = SimpleNamespace(
        supports_spec_decode=True,
        name="served-model",
    )
    scheduler.requests = {}
    scheduler.uid_to_request_id = {}
    scheduler.uid_to_request_processors = {}
    scheduler.spec_decode_runtime_method = "stale"
    scheduler.spec_decode_runtime_attempted = True
    scheduler._get_stop_tokens = lambda: set()
    config = {
        "prefill_batch_size": 1,
        "completion_batch_size": 1,
        "prefill_step_size": 1,
        "kv_cache_quantization": False,
        "kv_cache_turboquant": None,
        "spec_decode": "none",
        "mtp_model_type": "qwen4_exp",
        "mtp_max_k": 3,
        "mtp_disable_auto_k": False,
        "model_name": "served-model",
        "mtp_sidecar": None,
        "dspark_num_speculative_tokens": 5,
        "enable_suffix_decoding": False,
        "suffix_max_draft": 8,
        "suffix_max_suffix_len": 64,
        "suffix_min_confidence": 0.5,
        "suffix_min_draft_len": 2,
    }
    config.update(config_overrides)
    scheduler.config = SimpleNamespace(**config)
    return scheduler, batch_generator


@pytest.mark.parametrize(
    ("installer_succeeds", "expected_method"),
    [(True, "mtp"), (False, None)],
)
def test_scheduler_publishes_only_a_successfully_installed_mtp_runtime(
    monkeypatch,
    installer_succeeds,
    expected_method,
):
    pytest.importorskip("mlx")
    import vllm_mlx.scheduler as scheduler_module
    from vllm_mlx.request import SamplingParams

    monkeypatch.setattr(
        scheduler_module,
        "_install_mtp_vendored",
        lambda *args, **kwargs: installer_succeeds,
    )
    scheduler, batch_generator = _stub_runtime_scheduler(
        monkeypatch,
        scheduler_module,
        spec_decode="mtp",
    )

    created = scheduler._create_batch_generator(SamplingParams(max_tokens=8))

    assert created is batch_generator
    assert scheduler.spec_decode_runtime_method == expected_method
    assert scheduler.spec_decode_runtime_attempted is True


@pytest.mark.parametrize("method", ["dspark", "suffix"])
def test_scheduler_publishes_other_successfully_installed_runtimes(
    monkeypatch,
    method,
):
    pytest.importorskip("mlx")
    import vllm_mlx.scheduler as scheduler_module
    from vllm_mlx.request import SamplingParams

    monkeypatch.setattr(
        scheduler_module,
        "_install_dspark",
        lambda *args, **kwargs: method == "dspark",
    )
    monkeypatch.setattr(
        scheduler_module,
        "_install_suffix_decoding",
        lambda *args, **kwargs: method == "suffix",
    )
    scheduler, batch_generator = _stub_runtime_scheduler(
        monkeypatch,
        scheduler_module,
        spec_decode="dspark" if method == "dspark" else "none",
        enable_suffix_decoding=method == "suffix",
    )

    created = scheduler._create_batch_generator(SamplingParams(max_tokens=8))

    assert created is batch_generator
    assert scheduler.spec_decode_runtime_method == method
    assert scheduler.spec_decode_runtime_attempted is True


def test_closing_batch_generator_retires_published_speculative_runtime():
    pytest.importorskip("mlx")
    import vllm_mlx.scheduler as scheduler_module

    closed = []
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.batch_generator = SimpleNamespace(close=lambda: closed.append(True))
    scheduler.spec_decode_runtime_method = "mtp"
    scheduler.spec_decode_runtime_attempted = True

    scheduler._close_batch_generator()

    assert closed == [True]
    assert scheduler.batch_generator is None
    assert scheduler.spec_decode_runtime_method is None
    assert scheduler.spec_decode_runtime_attempted is False


def test_model_card_carries_live_speculative_policy(monkeypatch):
    from vllm_mlx.api.models import SpeculativeDecodingInfo
    from vllm_mlx.routes import models as models_route

    expected = SpeculativeDecodingInfo(
        configured=True,
        method="mtp",
        runtime_state="active",
        request_fallback_features=["tools"],
    )
    monkeypatch.setattr(
        models_route, "_resolve_speculative_decoding", lambda _model_id: expected
    )
    monkeypatch.setattr(models_route, "_resolve_context_window", lambda _model_id: None)
    monkeypatch.setattr(
        models_route,
        "_resolve_max_model_len",
        lambda _model_id, _native_context: None,
    )
    monkeypatch.setattr(models_route, "_audio_lane_snapshot", lambda: None)
    monkeypatch.setattr(
        models_route, "_served_lane_fields", lambda _model_id: (None, None)
    )
    monkeypatch.setattr(models_route, "_resolve_audio_entry", lambda _model_id: None)
    monkeypatch.setattr(models_route, "_locked_embedding_id", lambda: None)
    monkeypatch.setattr(
        models_route, "_reported_hybrid", lambda _model_id, static: static
    )
    monkeypatch.setattr(
        models_route,
        "effective_parsers_for",
        lambda _model_id, tool, reasoning: (tool, reasoning),
    )
    monkeypatch.setattr(
        models_route, "_detect_capabilities", lambda *_args, **_kwargs: ["text"]
    )
    monkeypatch.setattr(
        models_route,
        "_reported_modality",
        lambda _model_id, modality, _is_text_only=False: modality,
    )

    info = models_route._build_model_info("qwen3.8-27b-4bit")

    assert info.speculative_decoding == expected
    assert info.model_dump()["speculative_decoding"] == expected.model_dump()


@pytest.mark.parametrize("engine,scheduler", [(None, object()), (object(), None)])
def test_model_profile_never_advertises_unattached_runtime(
    monkeypatch, engine, scheduler
):
    from vllm_mlx.routes import models as models_route

    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda _engine: scheduler)

    assert models_route._resolve_speculative_decoding("unloaded-model") is None
