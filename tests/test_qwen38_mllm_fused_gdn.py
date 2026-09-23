"""Contracts for the exact-artifact Qwen3.8 MLLM fused-GDN canary."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import qwen38_mllm_fused_gdn as canary
from rapid_mlx.engine.batched import (
    BatchedEngine,
    _install_qwen38_mllm_fused_gdn_canary,
)
from rapid_mlx.qwen_runtime_plan import QwenTargetIdentity, _mint_verified_qwen_target
from rapid_mlx.runtime import qwen_artifact


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __mul__(self, _value):
        return self

    def astype(self, dtype):
        return FakeArray(self.shape, dtype)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeRandom:
    @staticmethod
    def key(value):
        return value

    @staticmethod
    def normal(shape, key=None):
        assert key is not None
        return FakeArray(shape, "bf16")


class FakeMx:
    bfloat16 = "bf16"
    float32 = "fp32"
    array = FakeArray
    random = FakeRandom()

    @staticmethod
    def concatenate(values, axis=0):
        assert axis == 1
        left, right = values
        return FakeArray(
            (left.shape[0], left.shape[1] + right.shape[1], left.shape[2]),
            left.dtype,
        )

    @staticmethod
    def array_equal(left, right):
        return FakeScalar(left.shape == right.shape and left.dtype == right.dtype)

    @staticmethod
    def zeros(shape, dtype):
        return FakeArray(shape, dtype)

    @staticmethod
    def eval(*_values):
        return None


class FakeCache:
    def __init__(self, size=2):
        self.cache = [None] * size
        self._left_padding = None
        self._left_padding_advance = 0
        self._lengths = None
        self._lengths_advance = 0
        self._speculation = None
        self._speculation_generation = 0
        self.metadata_revision = 0
        self.window_updates = 0
        self.recurrent_updates = 0
        self.advances = 0
        self.fail_recurrent = False

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    @property
    def left_padding(self):
        return self._left_padding

    @property
    def lengths(self):
        return self._lengths

    @property
    def is_speculating(self):
        return self._speculation is not None

    @property
    def history_capacity(self):
        return 0

    def update_window(self, index, source, width, *, lengths=None):
        assert lengths is None
        self.window_updates += 1
        self.cache[index] = FakeArray((1, width, source.shape[2]), source.dtype)
        return self.cache[index]

    def update_recurrent(self, index, length, update):
        assert length == 1
        self.recurrent_updates += 1
        if self.fail_recurrent:
            raise RuntimeError("post-commit failure")
        output, state = update(self.cache[index], None)
        self.cache[index] = state
        return output, state

    def advance(self, amount):
        self.advances += amount


class FakeGdn:
    stock_calls = 0

    def __init__(self):
        self.training = False
        self.sharding_group = None
        self.conv1d = SimpleNamespace(
            weight=FakeArray((canary._CONV_DIM, canary._CONV_KERNEL, 1), "bf16")
        )
        self.A_log = FakeArray((canary._NUM_VALUE_HEADS,), "fp32")
        self.dt_bias = FakeArray((canary._NUM_VALUE_HEADS,), "bf16")
        self.norm = SimpleNamespace(
            weight=FakeArray((canary._VALUE_HEAD_DIM,), "bf16"), eps=1e-6
        )
        self.out_proj_raises = False

    def __call__(self, inputs, mask=None, cache=None):
        type(self).stock_calls += 1
        return "stock"

    def in_proj_qkv(self, _inputs):
        return FakeArray((1, 1, canary._CONV_DIM), "bf16")

    def in_proj_z(self, _inputs):
        return FakeArray((1, 1, canary._VALUE_DIM), "bf16")

    def _project_gates(self, _inputs):
        gates = FakeArray((1, 1, canary._NUM_VALUE_HEADS), "bf16")
        return gates, gates

    def out_proj(self, output):
        if self.out_proj_raises:
            raise RuntimeError("out projection failed")
        assert output.shape == (1, 1, canary._VALUE_DIM)
        return FakeArray((1, 1, canary._HIDDEN_SIZE), "bf16")


def _kernel(*_args, **kwargs):
    assert kwargs["qwen35_semantics"] is True
    assert kwargs["num_value_heads"] == 48
    return (
        FakeArray((1, 1, canary._VALUE_DIM), "bf16"),
        FakeArray((1, canary._CONV_KERNEL - 1, canary._CONV_DIM), "bf16"),
        FakeArray((1, 48, 128, 128), "fp32"),
    )


def _patch():
    layers = [FakeGdn() for _ in range(canary._GDN_LAYERS)]
    patch = canary.Qwen38MllmFusedGdnCanary(
        FakeGdn,
        FakeCache,
        layers,
        FakeMx,
        _kernel,
        lambda _cache, _steps: None,
        lambda _cache, _steps: None,
    )
    patch.threadgroup_y = 32
    patch.qualified = True
    return patch, layers


def _cache():
    cache = FakeCache()
    cache[0] = FakeArray((1, 3, canary._CONV_DIM), "bf16")
    cache[1] = FakeArray((1, 48, 128, 128), "fp32")
    return cache


def _input():
    return FakeArray((1, 1, canary._HIDDEN_SIZE), "bf16")


def _truth():
    return SimpleNamespace(
        source_repo=canary._REPO,
        revision=canary._REVISION,
        target_subfolder=None,
        identity_status=SimpleNamespace(value="verified_hub_snapshot"),
        verification_id=canary._VERIFICATION_ID,
        config_sha256=canary._CONFIG_SHA256,
        outer_model_type="qwen3_5",
        text_model_type="qwen3_5_text",
        geometry=SimpleNamespace(
            hidden_size=5120,
            num_hidden_layers=64,
            num_attention_heads=24,
            num_key_value_heads=4,
            full_attention_interval=4,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            layer_types=canary._LAYER_TYPES,
            layer_types_sha256=canary._LAYER_TYPES_SHA256,
            layer_type_counts=(("full_attention", 16), ("linear_attention", 48)),
        ),
        quantization=SimpleNamespace(bits=4, group_size=64, mode="affine"),
        target_weights=SimpleNamespace(
            layout=SimpleNamespace(value="indexed_safetensors"),
            shard_count=3,
            index_sha256=canary._INDEX_SHA256,
            file_identities=canary._SHARDS,
            missing_shard_count=0,
        ),
    )


def _verified_target():
    identity = QwenTargetIdentity(
        target_repo=canary._REPO,
        target_revision=canary._REVISION,
        target_subfolder=None,
        outer_model_type="qwen3_5",
        language_model_type="qwen3_5_text",
        quantization=json.dumps(
            {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "override_bits": [],
                "override_count": 0,
                "override_group_sizes": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        weight_layout=json.dumps(
            {
                "layout": "indexed_safetensors",
                "shard_count": 3,
                "missing_shard_count": 0,
                "shards": [name for name, _ in canary._SHARDS],
                "index_sha256": canary._INDEX_SHA256,
                "file_identities": dict(canary._SHARDS),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        layer_layout=canary._LAYER_TYPES,
        cache_geometry=(("hidden_size", "5120"),),
    )
    return _mint_verified_qwen_target(
        identity=identity,
        verification_id=canary._VERIFICATION_ID,
        verification_authority=canary._VERIFICATION_AUTHORITY,
    )


def _real_truth(tmp_path: Path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures/qwen_artifacts/qwen38_27b_4bit"
    metadata = json.loads((fixture / "snapshot.json").read_text(encoding="utf-8"))
    hub = tmp_path / "hub"
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    snapshot = repo_cache / "snapshots" / metadata["revision"]
    blobs = repo_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for filename in ("config.json", "model.safetensors.index.json"):
        (snapshot / filename).write_bytes((fixture / filename).read_bytes())
    for shard in metadata["target_shards"]:
        blob = blobs / metadata["target_blob_ids"][shard]
        blob.touch()
        (snapshot / shard).symlink_to(os.path.relpath(blob, snapshot))
    monkeypatch.setattr(
        qwen_artifact, "_configured_hub_cache_root", lambda: hub.resolve()
    )
    truth = qwen_artifact.probe_resolved_qwen_artifact(
        snapshot, repo_id=metadata["source_repo"]
    )
    assert truth is not None
    return truth, qwen_artifact.to_verified_runtime_target(truth)


def test_operator_gate_is_default_off_and_explicit(monkeypatch):
    monkeypatch.delenv(canary.ENV_VAR, raising=False)
    assert canary.operator_enabled() is False
    monkeypatch.setenv(canary.ENV_VAR, "1")
    assert canary.operator_enabled() is True
    monkeypatch.setenv(canary.ENV_VAR, "0")
    assert canary.operator_enabled() is False


def test_exact_b0_truth_requires_private_mint_and_full_receipt(tmp_path, monkeypatch):
    truth, verified = _real_truth(tmp_path, monkeypatch)
    assert canary._artifact_exact(truth, verified) is True
    forged = SimpleNamespace(
        identity=verified.identity,
        verification_id=verified.verification_id,
        verification_authority=verified.verification_authority,
    )
    assert canary._artifact_exact(truth, forged) is False
    assert canary._artifact_exact(_truth(), verified) is False


def test_disabled_installer_does_not_touch_runtime(monkeypatch):
    monkeypatch.delenv(canary.ENV_VAR, raising=False)
    monkeypatch.setattr(
        canary,
        "_installed_version",
        lambda _package: (_ for _ in ()).throw(AssertionError("version probed")),
    )
    assert canary.contract_status()["runtime_versions_required"]["mlx"] == "0.32.2"
    assert canary.install_qwen38_mllm_fused_gdn_canary(
        object(), _truth(), _verified_target()
    ) == (None, "operator_disabled")


def test_precommit_failure_falls_back_without_cache_mutation():
    patch, layers = _patch()
    cache = _cache()
    layers[0].out_proj_raises = True
    FakeGdn.stock_calls = 0
    assert patch._call(layers[0], _input(), None, cache) == "stock"
    assert FakeGdn.stock_calls == 1
    assert cache.window_updates == cache.recurrent_updates == cache.advances == 0


def test_postcommit_failure_propagates_without_stock_replay():
    patch, layers = _patch()
    cache = _cache()
    cache.fail_recurrent = True
    FakeGdn.stock_calls = 0
    with pytest.raises(RuntimeError, match="post-commit"):
        patch._call(layers[0], _input(), None, cache)
    assert cache.window_updates == cache.recurrent_updates == 1
    assert cache.advances == 0
    assert FakeGdn.stock_calls == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cache, value: setattr(cache, "_speculation", {}),
        lambda cache, value: setattr(cache, "_lengths", object()),
        lambda cache, value: setattr(value, "shape", (2, 1, canary._HIDDEN_SIZE)),
        lambda cache, value: cache.cache.append(None),
    ],
)
def test_unknown_or_speculative_runtime_state_stays_stock(mutate):
    patch, layers = _patch()
    cache = _cache()
    value = _input()
    mutate(cache, value)
    FakeGdn.stock_calls = 0
    assert patch._call(layers[0], value, None, cache) == "stock"
    assert FakeGdn.stock_calls == 1
    assert cache.window_updates == 0


def test_class_patch_is_exact_instance_allowlisted_and_reversible():
    patch, layers = _patch()
    outsider = FakeGdn()
    original = FakeGdn.__call__
    FakeGdn.stock_calls = 0
    try:
        patch.install()
        cache = _cache()
        assert layers[0](_input(), None, cache).shape == (
            1,
            1,
            canary._HIDDEN_SIZE,
        )
        assert patch.hits == 1
        assert outsider(_input(), None, _cache()) == "stock"
        assert FakeGdn.stock_calls == 1
    finally:
        patch.close()
    assert FakeGdn.__call__ is original


def test_install_compare_and_swap_rejects_foreign_class_patch():
    patch, _ = _patch()
    original = FakeGdn.__call__

    def foreign(*_args, **_kwargs):
        return "foreign"

    try:
        FakeGdn.__call__ = foreign
        with pytest.raises(RuntimeError, match="changed before canary install"):
            patch.install()
        assert FakeGdn.__call__ is foreign
        assert patch.installed is False
    finally:
        FakeGdn.__call__ = original


def test_probe_commits_exactly_32_selected_instance_steps_and_resets_hits():
    calls = []

    def kernel(*args, **kwargs):
        calls.append(kwargs["threadgroup_y"])
        if kwargs["threadgroup_y"] == 32:
            raise RuntimeError("reject first threadgroup")
        return _kernel(*args, **kwargs)

    patch, _ = _patch()
    patch.fused_kernel = kernel

    def stock(_layer, _inputs, _mask, cache):
        cache[0] = FakeArray((1, 3, canary._CONV_DIM), "bf16")
        cache[1] = FakeArray((1, 48, 128, 128), "fp32")
        cache.advance(1)
        return FakeArray((1, 1, canary._HIDDEN_SIZE), "bf16")

    patch.original = stock
    assert patch.qualify() is True
    assert calls[0] == 32
    assert patch.threadgroup_y == 16
    assert patch.probe_steps_committed == 32
    assert patch.hits == 0
    assert patch.layer_hits == [0] * 48


def test_non_bool_speculation_marker_fails_closed():
    patch, layers = _patch()
    cache = _cache()
    cache.__class__.is_speculating = property(lambda _self: 0)
    try:
        FakeGdn.stock_calls = 0
        assert patch._call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
    finally:
        cache.__class__.is_speculating = property(
            lambda self: self._speculation is not None
        )


def test_contract_status_preserves_raw_false_adjudication():
    status = canary.contract_status()
    assert status["experiment_commit"] == canary.EXPERIMENT_COMMIT
    assert status["methodology_sha256"] == canary.EXPERIMENT_METHODOLOGY_SHA256
    assert status["receipt_sha256"] == canary.EXPERIMENT_RECEIPT_SHA256
    assert status["raw_aggregate_pass"] is False
    assert status["adjudication"] == "vlm_differential_pass_semantic_aggregate_invalid"
    assert status["runtime_versions_required"] == {
        "mlx": "0.32.2",
        "mlx-lm": "0.31.3",
        "mlx-vlm": "0.7.1",
    }
    assert status["kernel_sha256"] == canary._KERNEL_SHA256


def test_kernel_contract_binds_versions_source_and_signature(monkeypatch):
    from rapid_mlx.kernels.qwen4_fused_gdn_decode import fused_gdn_decode

    assert canary._kernel_contract_failure(fused_gdn_decode) is None
    real_version = canary._installed_version
    monkeypatch.setattr(
        canary,
        "_installed_version",
        lambda package: "wrong" if package == "mlx" else real_version(package),
    )
    assert canary._kernel_contract_failure(fused_gdn_decode) == "runtime_version_drift"
    monkeypatch.setattr(canary, "_installed_version", real_version)
    monkeypatch.setattr(canary, "_source_matches", lambda *_args: False)
    assert canary._kernel_contract_failure(fused_gdn_decode) == "kernel_source_drift"


def test_pinned_mlx_vlm_language_and_cache_sources_are_exact():
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    assert canary._source_matches(
        Qwen3_5GatedDeltaNet.__call__,
        canary._LANGUAGE_MODULE,
        canary._LANGUAGE_SHA256,
    )
    assert canary._source_matches(
        ArraysCache,
        canary._CACHE_MODULE,
        canary._CACHE_SHA256,
    )


def test_runtime_contract_rejects_loaded_text_model_type_drift():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer

    layers = []
    for layer_type in canary._LAYER_TYPES:
        layer = Qwen3_5DecoderLayer.__new__(Qwen3_5DecoderLayer)
        layer.is_linear = layer_type == "linear_attention"
        layers.append(layer)
    language_model = SimpleNamespace(
        layers=layers,
        args=SimpleNamespace(
            model_type="forged_qwen3_5_text",
            hidden_size=5120,
            num_hidden_layers=64,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            full_attention_interval=4,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        ),
    )

    with pytest.raises(RuntimeError, match="language config geometry drifted"):
        canary._runtime_contract(language_model)


def test_qwen36_existing_32_head_path_is_unchanged():
    from rapid_mlx import qwen35_fused_gdn_decode as qwen36_path

    assert qwen36_path._NUM_VALUE_HEADS == 32
    assert canary._NUM_VALUE_HEADS == 48
    assert qwen36_path.install_qwen35_fused_gdn_decode is not (
        canary.install_qwen38_mllm_fused_gdn_canary
    )


class _ImmediateExecutor:
    def __init__(self):
        self.submits = 0

    def submit(self, function, *args):
        self.submits += 1
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def _boot_engine():
    return SimpleNamespace(
        _qwen38_mllm_fused_gdn_status={
            "requested": False,
            "qualified": False,
            "active": False,
            "fallback_reason": "operator_disabled",
        },
        _qwen38_mllm_fused_gdn_canary=None,
        _qwen_artifact_snapshot_source="snapshot",
        _qwen_artifact_repo_id=canary._REPO,
        _model_load_executor=_ImmediateExecutor(),
    )


def test_boot_seam_default_off_does_no_probe_or_worker_qualification(monkeypatch):
    engine = _boot_engine()
    monkeypatch.setattr(canary, "operator_enabled", lambda: False)
    monkeypatch.setattr(
        canary,
        "actual_runtime_versions",
        lambda: (_ for _ in ()).throw(AssertionError("version probe")),
    )
    monkeypatch.setattr(
        qwen_artifact,
        "probe_resolved_qwen_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("artifact")),
    )
    _install_qwen38_mllm_fused_gdn_canary(engine, object())
    assert engine._model_load_executor.submits == 0
    assert engine._qwen38_mllm_fused_gdn_status["requested"] is False
    assert engine._qwen38_mllm_fused_gdn_status["active"] is False
    assert engine._qwen38_mllm_fused_gdn_status["fallback_reason"] == (
        "operator_disabled"
    )


def test_boot_seam_artifact_probe_exception_keeps_stock_and_status(monkeypatch):
    engine = _boot_engine()
    monkeypatch.setattr(canary, "operator_enabled", lambda: True)
    monkeypatch.setattr(canary, "actual_runtime_versions", lambda: {"mlx": "0.32.2"})
    monkeypatch.setattr(
        qwen_artifact,
        "probe_resolved_qwen_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("probe failed")),
    )
    _install_qwen38_mllm_fused_gdn_canary(engine, object())
    status = engine._qwen38_mllm_fused_gdn_status
    assert engine._model_load_executor.submits == 0
    assert status["requested"] is True
    assert status["qualified"] is status["active"] is False
    assert status["fallback_reason"] == "qualification_exception"
    assert status["raw_aggregate_pass"] is False


@pytest.mark.parametrize(
    ("installer_result", "expected_reason", "raises"),
    [
        ((None, "real_weight_probe_failed"), "real_weight_probe_failed", False),
        (None, "qualification_exception", True),
    ],
)
def test_boot_seam_installer_failure_keeps_stock(
    monkeypatch, installer_result, expected_reason, raises
):
    engine = _boot_engine()
    monkeypatch.setattr(canary, "operator_enabled", lambda: True)
    monkeypatch.setattr(canary, "actual_runtime_versions", lambda: {})
    truth = object()
    verified = object()
    monkeypatch.setattr(
        qwen_artifact, "probe_resolved_qwen_artifact", lambda *_a, **_k: truth
    )
    monkeypatch.setattr(
        qwen_artifact, "to_verified_runtime_target", lambda _truth: verified
    )
    if raises:
        monkeypatch.setattr(
            canary,
            "install_qwen38_mllm_fused_gdn_canary",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("installer")),
        )
    else:
        monkeypatch.setattr(
            canary,
            "install_qwen38_mllm_fused_gdn_canary",
            lambda *_args: installer_result,
        )
    _install_qwen38_mllm_fused_gdn_canary(engine, object())
    status = engine._qwen38_mllm_fused_gdn_status
    assert engine._qwen38_mllm_fused_gdn_canary is None
    assert status["active"] is status["qualified"] is False
    assert status["fallback_reason"] == expected_reason


def test_boot_seam_success_preserves_contract_and_probe_evidence(monkeypatch):
    engine = _boot_engine()
    monkeypatch.setattr(canary, "operator_enabled", lambda: True)
    monkeypatch.setattr(canary, "actual_runtime_versions", lambda: {"mlx": "0.32.2"})
    monkeypatch.setattr(
        qwen_artifact, "probe_resolved_qwen_artifact", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        qwen_artifact, "to_verified_runtime_target", lambda _truth: object()
    )
    patch = SimpleNamespace(
        qualified=True,
        installed=True,
        probe_steps_committed=32,
    )
    monkeypatch.setattr(
        canary,
        "install_qwen38_mllm_fused_gdn_canary",
        lambda *_args: (patch, None),
    )
    _install_qwen38_mllm_fused_gdn_canary(engine, object())
    status = engine._qwen38_mllm_fused_gdn_status
    assert status["requested"] is status["qualified"] is status["active"] is True
    assert status["fallback_reason"] is None
    assert status["probe_steps_committed"] == 32
    assert status["receipt_sha256"] == canary.EXPERIMENT_RECEIPT_SHA256
    assert status["raw_aggregate_pass"] is False


@pytest.mark.asyncio
async def test_engine_stop_restores_patch_before_executor_shutdown():
    events = []

    class Patch:
        def close(self):
            events.append("close")

    class Scheduler:
        async def stop(self):
            events.append("scheduler-stop")

    class Executor:
        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:  # pragma: no cover - Future parity
                future.set_exception(exc)
            return future

        def shutdown(self, wait=False):
            assert wait is False
            events.append("executor-shutdown")

    engine = object.__new__(BatchedEngine)
    engine._clear_qwen_runtime_observability = lambda: None
    engine._abort_all_guided_requests = lambda: None
    engine._engine = None
    engine._mllm_scheduler = Scheduler()
    engine._qwen38_mllm_fused_gdn_canary = Patch()
    engine._qwen38_mllm_fused_gdn_status = {"active": True}
    engine._model_load_executor = Executor()
    engine._is_mllm = True
    engine._start_time = 1
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = object()
    engine._mllm_instance = object()
    engine._prompt_host_cache = None
    engine._loaded = True
    engine._engine_started = True
    engine._mllm_native_text_engine = True

    await engine.stop()

    assert events == ["scheduler-stop", "close", "executor-shutdown"]
    assert engine._qwen38_mllm_fused_gdn_canary is None
