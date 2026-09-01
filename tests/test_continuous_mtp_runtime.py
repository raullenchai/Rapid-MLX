"""Pure-Python contracts for continuous self-MTP runtime assembly."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp import continuous_runtime as runtime_module
from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPUnsupportedError,
)
from vllm_mlx.spec_decode.mtp.ragged_cache import (
    preflight_ragged_cache,
    trim_ragged_cache,
)


def _descriptor(family: str = "qwen3_5", **changes):
    values = {
        "protocol_version": 1,
        "model_family": family,
        "batch_forward": "mtp_batch_forward",
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "dynamic_join": True,
        "flash_dynamic_membership_attested": False,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    values.update(changes)
    return values


class _InjectedTextModel:
    model_type = "qwen3_5_text"

    def __init__(self, descriptor=None):
        self.args = SimpleNamespace(hidden_size=8)
        self.model = SimpleNamespace(layers=[object()])
        self.batched_mtp_capability = descriptor or _descriptor()
        self.calls = []

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden=False,
        n_confirmed=0,
    ):
        self.calls.append(
            (
                "target",
                inputs,
                cache,
                input_embeddings,
                return_hidden,
                n_confirmed,
            )
        )
        return "target-logits", "target-hidden"

    def mtp_batch_forward(self, hidden, token_ids, mtp_cache):
        self.calls.append(("draft", hidden, token_ids, mtp_cache))
        return "draft-logits", "post-hidden"

    def make_mtp_cache(self):
        self.calls.append(("make-mtp-cache",))
        return ["draft-cache"]


class _OuterModel:
    def __init__(self, inner):
        self.language_model = inner
        self.batched_mtp_capability = inner.batched_mtp_capability


class _NoConfirmedTarget(_InjectedTextModel):
    def __call__(self, inputs, cache=None, return_hidden=False):
        return inputs, cache, return_hidden


class _ArrayOpsStub:
    pass


@pytest.fixture
def ragged_install_stub(monkeypatch):
    """Isolate runtime assembly from the separately tested mlx-lm adapter.

    Assembly owns descriptor validation and seam wiring.  The adapter's
    version and class-shape contracts live in ``test_mtp_ragged_cache.py``;
    stubbing only that installation boundary keeps these contracts runnable
    in the hosted no-MLX matrix without pretending a fake tensor runtime is
    production MLX.
    """
    calls = []

    def install(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "vllm_mlx.spec_decode.mtp.ragged_cache.install_ragged_cache_rollback",
        install,
    )
    return calls


def test_assembler_resolves_inner_model_and_wires_forward_and_cache_seams(
    monkeypatch,
    ragged_install_stub,
):
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    target_cache_calls = []
    monkeypatch.setattr(
        runtime_module,
        "_make_prompt_cache",
        lambda model: target_cache_calls.append(model) or ["target-cache"],
    )

    runtime = runtime_module.assemble_continuous_self_mtp_runtime(
        outer,
        array_ops=_ArrayOpsStub(),
        prefill_step_size=17,
    )

    assert runtime.config.enabled is True
    assert runtime.config.architecture == "qwen3_5"
    assert runtime.config.allow_dynamic_membership is False
    assert runtime.capabilities.missing_fixed_core() == ()
    assert runtime.capabilities.dynamic_membership is False
    assert runtime.compute.prefill_step_size == 17

    assert runtime.forwards.target("ids", "target-kv", n_confirmed=2) == (
        "target-logits",
        "target-hidden",
    )
    assert inner.calls[-1] == (
        "target",
        "ids",
        "target-kv",
        None,
        True,
        2,
    )
    assert runtime.forwards.draft("hidden", "tokens", "draft-kv") == (
        "draft-logits",
        "post-hidden",
    )
    assert inner.calls[-1] == ("draft", "hidden", "tokens", "draft-kv")

    assert runtime.compute.target_cache_factory() == ["target-cache"]
    assert target_cache_calls == [inner]
    assert runtime.compute.draft_cache_factory() == ["draft-cache"]
    assert runtime.caches._preflight is preflight_ragged_cache
    assert runtime.caches._trim is trim_ragged_cache
    assert ragged_install_stub == [{"qwen4_state_cls": None, "qsa_cls": None}]


def test_dynamic_membership_requires_policy_and_dense_attestation(
    ragged_install_stub,
):
    inner = _InjectedTextModel()
    enabled = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    policy_off = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        array_ops=_ArrayOpsStub(),
    )
    assert enabled.capabilities.dynamic_membership is True
    assert policy_off.capabilities.dynamic_membership is False
    assert ragged_install_stub == [
        {"qwen4_state_cls": None, "qsa_cls": None},
        {"qwen4_state_cls": None, "qsa_cls": None},
    ]


def test_qwen4_is_not_attested_by_the_dense_adapter():
    descriptor = _descriptor("qwen4_exp")
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"
    outer = _OuterModel(inner)

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="unsupported model"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            outer,
            array_ops=_ArrayOpsStub(),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"protocol_version": 2}, "protocol_version"),
        ({"recursive_draft_depth": 1}, "recursive_draft_depth"),
        ({"fixed_membership": False}, "fixed_membership"),
        ({"target_return_hidden": False}, "target_return_hidden"),
        ({"mtp_return_hidden": False}, "mtp_return_hidden"),
        ({"confirmed_target_forward": False}, "confirmed_target_forward"),
        ({"ragged_rollback": False}, "ragged_rollback"),
        ({"atomic_cache_commit": False}, "atomic_cache_commit"),
        ({"quantized_cache": True}, "quantized_cache"),
        ({"windowed_cache": True}, "windowed_cache"),
        ({"xtc": True}, "xtc"),
        ({"batch_forward": None}, "batch_forward"),
        ({"model_family": "unknown"}, "unsupported model family"),
    ],
)
def test_descriptor_mismatches_fail_closed(changes, message):
    inner = _InjectedTextModel(_descriptor(**changes))

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match=message):
        runtime_module.assemble_continuous_self_mtp_runtime(
            inner,
            array_ops=_ArrayOpsStub(),
        )


def test_missing_injected_surfaces_and_target_abi_fail_closed():
    no_confirmed = _NoConfirmedTarget()
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="n_confirmed"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_confirmed,
            array_ops=_ArrayOpsStub(),
        )

    no_draft = _InjectedTextModel()
    no_draft.mtp_batch_forward = None
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="not callable"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_draft,
            array_ops=_ArrayOpsStub(),
        )

    no_cache = _InjectedTextModel()
    no_cache.make_mtp_cache = None
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="make_mtp_cache"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_cache,
            array_ops=_ArrayOpsStub(),
        )


def test_outer_and_resolved_inner_descriptors_must_match():
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    outer.batched_mtp_capability = _descriptor(protocol_version=2)

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="descriptors disagree"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            outer,
            array_ops=_ArrayOpsStub(),
        )


def test_prompt_cache_factory_uses_the_lazy_mlx_lm_boundary(monkeypatch):
    cache_module = ModuleType("mlx_lm.models.cache")
    calls = []
    cache_module.make_prompt_cache = lambda model: calls.append(model) or "cache"
    models_module = ModuleType("mlx_lm.models")
    models_module.__path__ = []
    mlx_lm_module = ModuleType("mlx_lm")
    mlx_lm_module.__path__ = []
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)

    model = object()
    assert runtime_module._make_prompt_cache(model) == "cache"
    assert calls == [model]


def test_descriptor_resolution_and_target_introspection_fail_closed(monkeypatch):
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="no batched"):
        runtime_module._descriptor_for(object())
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="cannot resolve"):
        runtime_module._resolve_inner(object(), "unsupported")
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="not callable"):
        runtime_module._require_target_abi(object())

    inner = _InjectedTextModel()
    monkeypatch.setattr(
        runtime_module.inspect,
        "signature",
        lambda _value: (_ for _ in ()).throw(ValueError("opaque")),
    )
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="cannot inspect"):
        runtime_module._require_target_abi(inner)


def test_resolved_inner_descriptor_and_hidden_return_are_enforced(
    monkeypatch,
    ragged_install_stub,
):
    outer = SimpleNamespace(batched_mtp_capability=_descriptor())
    mismatched_inner = _InjectedTextModel(_descriptor(protocol_version=2))
    monkeypatch.setattr(
        runtime_module,
        "_resolve_inner",
        lambda _model, _family: mismatched_inner,
    )
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="same descriptor"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            outer,
            array_ops=_ArrayOpsStub(),
        )

    inner = _InjectedTextModel()
    monkeypatch.setattr(
        runtime_module,
        "_resolve_inner",
        lambda _model, _family: inner,
    )
    runtime = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        array_ops=_ArrayOpsStub(),
    )
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="return hidden"):
        runtime.forwards.mtp_forward("hidden", "tokens", "cache", return_hidden=False)


def test_future_fixed_capability_addition_remains_fail_closed(
    monkeypatch,
    ragged_install_stub,
):
    monkeypatch.setattr(
        runtime_module.ContinuousSelfMTPCapabilities,
        "missing_fixed_core",
        lambda _self: ("future_contract",),
    )
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="future_contract"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            _InjectedTextModel(),
            array_ops=_ArrayOpsStub(),
        )
