# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from vllm_mlx.patches.mla_absorbed_verify import (
    _use_absorbed,
    latent_length,
    max_absorbed_queries,
)

_HAS_MLX = find_spec("mlx") is not None and find_spec("mlx.core") is not None
if _HAS_MLX:
    import mlx.core as mx
else:  # pragma: no cover - exercised by the no-MLX CI lane
    mx = None

requires_mlx = pytest.mark.skipif(not _HAS_MLX, reason="requires MLX")


def test_asymptotic_thresholds_match_model_geometry() -> None:
    assert max_absorbed_queries(512, 128, 128) == 170
    assert max_absorbed_queries(512, 192, 256) == 398


def test_threshold_preserves_strict_crossover_inequality() -> None:
    # r=4, d=4 has an exact asymptotic crossover at L=4. Absorbed MLA must
    # stop at 3 because equal cost is not a win.
    assert max_absorbed_queries(4, 2, 2) == 3


@pytest.mark.parametrize("query_len", [2, 3, 32, 169, 398])
def test_cold_cache_rejects_multi_token_absorption(query_len: int) -> None:
    for dims in ((512, 128, 128), (512, 192, 256)):
        assert max_absorbed_queries(*dims, cache_len=query_len) < query_len


def test_warm_cache_threshold_is_monotonic() -> None:
    lengths = (256, 1024, 8192, 32768)
    thresholds = [max_absorbed_queries(512, 128, 128, value) for value in lengths]
    assert thresholds == sorted(thresholds)
    assert thresholds[-1] == 169


@pytest.mark.parametrize(
    "args",
    [
        (0, 128, 128, None),
        (512, 0, 128, None),
        (512, 128, -1, None),
        (512, 128, 128, 0),
        (64, 128, 128, None),
    ],
)
def test_invalid_or_nonbeneficial_geometry_fails_closed(args) -> None:
    assert max_absorbed_queries(*args) == 1


def test_latent_length_plain_and_quantized() -> None:
    class Array:
        shape = (1, 1, 37, 8)

    array = Array()
    assert latent_length(array) == 37
    assert latent_length((array, object(), object())) == 37


def test_latent_length_rejects_missing_sequence_axis() -> None:
    with pytest.raises(ValueError, match="sequence dimension"):
        latent_length(object())
    with pytest.raises(ValueError, match="sequence dimension"):
        latent_length(type("Scalar", (), {"shape": (1,)})())


def test_disabled_stats_do_not_acquire_hot_path_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    class ExplodingLock:
        def __enter__(self):
            raise AssertionError("disabled counters must not acquire the stats lock")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(patch, "_STATS_ENABLED", False)
    monkeypatch.setattr(patch, "_STATS_LOCK", ExplodingLock())
    before = dict(patch._STATS)
    patch._increment_stat("forwards")
    assert before == patch._STATS


def test_rapid_gate_requires_qualified_warm_cache() -> None:
    attention = type(
        "Attention",
        (),
        {"kv_lora_rank": 512, "qk_nope_head_dim": 128, "v_head_dim": 128},
    )()
    latent = type("Latent", (), {"shape": (1, 1, 1023, 512)})()
    assert _use_absorbed(attention, 3, latent) is False
    latent.shape = (1, 1, 1024, 512)
    assert _use_absorbed(attention, 3, latent) is True


@pytest.mark.parametrize("enabled", [False, True])
@requires_mlx
@pytest.mark.requires_mlx
def test_real_serve_import_installs_exact_supported_targets(enabled: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import json
import vllm_mlx.utils.tokenizer
from vllm_mlx.patches.mla_absorbed_verify import mla_absorbed_verify_stats
print(json.dumps(mla_absorbed_verify_stats()))
"""
    env = os.environ.copy()
    if enabled:
        env["RAPID_MLX_MLA_ABSORBED_VERIFY"] = "1"
    else:
        env.pop("RAPID_MLX_MLA_ABSORBED_VERIFY", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    stats = json.loads(result.stdout.strip().splitlines()[-1])
    assert stats["installed"] is True
    assert stats["enabled"] is enabled
    if not enabled:
        assert stats["provider"] in {"disabled", "upstream"}
        if stats["provider"] == "disabled":
            assert stats["targets"] == []
        return
    assert stats["provider"] in {"rapid", "upstream"}
    if stats["provider"] == "rapid":
        assert stats["targets"] == [
            "deepseek_v3.DeepseekV3Attention",
            "glm4_moe_lite.Glm4MoeLiteAttention",
            "kimi_linear.KimiMLAAttention",
            "longcat_flash.LongcatFlashMLA",
        ]


def _run_install_probe(code: str, *, enabled: bool = True) -> dict:
    import json

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if enabled:
        env["RAPID_MLX_MLA_ABSORBED_VERIFY"] = "1"
    else:
        env.pop("RAPID_MLX_MLA_ABSORBED_VERIFY", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@requires_mlx
@pytest.mark.requires_mlx
def test_default_off_does_not_wrap_attention() -> None:
    stats = _run_install_probe(
        """
import json
from mlx_lm.models import glm4_moe_lite
original = glm4_moe_lite.Glm4MoeLiteAttention.__call__
from vllm_mlx.patches.mla_absorbed_verify import install_mla_absorbed_verify, mla_absorbed_verify_stats
install_mla_absorbed_verify()
stats = mla_absorbed_verify_stats()
stats["unchanged"] = glm4_moe_lite.Glm4MoeLiteAttention.__call__ is original
print(json.dumps(stats))
""",
        enabled=False,
    )
    assert stats["provider"] in {"disabled", "upstream"}
    assert stats["unchanged"] is True


@requires_mlx
@pytest.mark.requires_mlx
def test_unknown_upstream_method_fails_closed_for_that_family() -> None:
    stats = _run_install_probe(
        """
import json
from mlx_lm.models import deepseek_v3
def changed(self, x, mask=None, cache=None):
    return x
deepseek_v3.DeepseekV3Attention.__call__ = changed
from vllm_mlx.patches.mla_absorbed_verify import install_mla_absorbed_verify, mla_absorbed_verify_stats
install_mla_absorbed_verify()
print(json.dumps(mla_absorbed_verify_stats()))
"""
    )
    assert stats["provider"] == "rapid"
    assert "deepseek_v3.DeepseekV3Attention" not in stats["targets"]
    assert "glm4_moe_lite.Glm4MoeLiteAttention" in stats["targets"]


@requires_mlx
@pytest.mark.requires_mlx
def test_upstream_provider_prevents_double_patch() -> None:
    stats = _run_install_probe(
        """
import json
from mlx_lm.models import deepseek_v3, mla
original = deepseek_v3.DeepseekV3Attention.__call__
mla.max_absorbed_queries = lambda *args, **kwargs: 1
from vllm_mlx.patches.mla_absorbed_verify import install_mla_absorbed_verify, mla_absorbed_verify_stats
install_mla_absorbed_verify()
stats = mla_absorbed_verify_stats()
stats["unchanged"] = deepseek_v3.DeepseekV3Attention.__call__ is original
print(json.dumps(stats))
"""
    )
    assert stats["provider"] == "upstream"
    assert stats["unchanged"] is True


@requires_mlx
@pytest.mark.requires_mlx
def test_unqualified_mlx_lm_version_fails_closed() -> None:
    stats = _run_install_probe(
        """
import importlib.metadata
import json
from mlx_lm.models import deepseek_v3
original = deepseek_v3.DeepseekV3Attention.__call__
real_version = importlib.metadata.version
importlib.metadata.version = lambda name: "0.31.4" if name == "mlx-lm" else real_version(name)
from vllm_mlx.patches.mla_absorbed_verify import install_mla_absorbed_verify, mla_absorbed_verify_stats
install_mla_absorbed_verify()
stats = mla_absorbed_verify_stats()
stats["unchanged"] = deepseek_v3.DeepseekV3Attention.__call__ is original
print(json.dumps(stats))
"""
    )
    assert stats["provider"] == "unsupported"
    assert stats["targets"] == []
    assert stats["unchanged"] is True


@requires_mlx
@pytest.mark.requires_mlx
def test_deepseek_v32_indexer_patch_is_not_replaced() -> None:
    stats = _run_install_probe(
        """
import json
from mlx_lm.models import deepseek_v32
from vllm_mlx.patches.deepseek_v32_indexer_gate import install_deepseek_v32_indexer_gate
install_deepseek_v32_indexer_gate()
indexed = deepseek_v32.DeepseekV32Attention.__call__
from vllm_mlx.patches.mla_absorbed_verify import install_mla_absorbed_verify, mla_absorbed_verify_stats
install_mla_absorbed_verify()
stats = mla_absorbed_verify_stats()
stats["unchanged"] = deepseek_v32.DeepseekV32Attention.__call__ is indexed
print(json.dumps(stats))
"""
    )
    assert stats["unchanged"] is True
    assert all(not target.startswith("deepseek_v32.") for target in stats["targets"])


def _isolate_installer_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo collection-time installs from other test modules."""
    from mlx_lm.models import mla

    from vllm_mlx.patches import mla_absorbed_verify as patch

    originals = getattr(mla, "_RAPID_MLX_MLA_ABSORBED_ORIGINALS", {})
    for module_name, class_name in patch._SUPPORTED_SOURCE_HASHES:
        module = __import__(f"mlx_lm.models.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)
        monkeypatch.setattr(
            cls, "__call__", originals.get((module_name, class_name), cls.__call__)
        )
        marker = f"_RAPID_MLX_MLA_ABSORBED_{class_name}"
        if hasattr(module, marker):
            monkeypatch.delattr(module, marker)
    if hasattr(mla, "_RAPID_MLX_MLA_ABSORBED_ORIGINALS"):
        monkeypatch.delattr(mla, "_RAPID_MLX_MLA_ABSORBED_ORIGINALS")
    monkeypatch.setattr(patch, "_INSTALLED", False)
    monkeypatch.setattr(patch, "_PROVIDER", "none")
    monkeypatch.setattr(patch, "_ENABLED", False)
    monkeypatch.setattr(patch, "_STATS_ENABLED", True)
    monkeypatch.setattr(patch, "_PATCHED_TARGETS", set())
    monkeypatch.setattr(patch, "_STATS", {key: 0 for key in patch._STATS})


def _tiny_attention(module_name: str, *, q_lora_rank: int | None = 16):
    if module_name in {"deepseek_v3", "glm4_moe_lite"}:
        module = __import__(f"mlx_lm.models.{module_name}", fromlist=["ModelArgs"])
        class_name = (
            "DeepseekV3Attention"
            if module_name == "deepseek_v3"
            else "Glm4MoeLiteAttention"
        )
        config = module.ModelArgs(
            hidden_size=32,
            num_attention_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            attention_bias=False,
        )
        return module, class_name, getattr(module, class_name)(config)

    if module_name == "kimi_linear":
        from mlx_lm.models import kimi_linear as module

        config = module.ModelArgs(
            model_type="kimi_linear",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=64,
            head_dim=8,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            linear_attn_config={},
            model_max_length=256,
            num_experts=1,
            moe_intermediate_size=16,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
        )
        return module, "KimiMLAAttention", module.KimiMLAAttention(config)

    from mlx_lm.models import longcat_flash as module

    config = module.ModelArgs(
        model_type="longcat_flash",
        attention_method="mla",
        zero_expert_type="none",
        hidden_size=32,
        ffn_hidden_size=64,
        moe_topk=1,
        expert_ffn_hidden_size=16,
        n_routed_experts=1,
        zero_expert_num=0,
        num_layers=1,
        vocab_size=64,
        max_position_embeddings=256,
        num_attention_heads=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        attention_bias=False,
    )
    return module, "LongcatFlashMLA", module.LongcatFlashMLA(config)


@pytest.mark.parametrize(
    "module_name",
    ["deepseek_v3", "glm4_moe_lite", "kimi_linear", "longcat_flash"],
)
@requires_mlx
@pytest.mark.requires_mlx
def test_patched_attention_matches_stock_contract(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    from mlx_lm.models import mla
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    mx.random.seed(7)
    module, class_name, attention = _tiny_attention(module_name)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    patch.install_mla_absorbed_verify()
    original = mla._RAPID_MLX_MLA_ABSORBED_ORIGINALS[(module_name, class_name)]

    # A cold L=3 call must retain the stock materialized branch exactly.
    cold_x = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
    cold_stock = original(attention, cold_x, None, None)
    before = patch.mla_absorbed_verify_stats()
    monkeypatch.setattr(patch, "_ENABLED", True)
    cold_candidate = attention(cold_x, None, None)
    mx.eval(cold_stock, cold_candidate)
    assert mx.array_equal(cold_stock, cold_candidate).item()
    after_cold = patch.mla_absorbed_verify_stats()
    assert after_cold["materialized"] == before["materialized"] + 1

    # At T=1027, L=3 is below the r=8/d=8 crossover and inside Rapid's
    # dogfood-qualified long-context window. Compare the absorbed
    # factorization to stock on cloned warm caches.
    prefill = mx.random.normal((1, 1024, 32)).astype(mx.bfloat16)
    stock_cache = KVCache()
    candidate_cache = KVCache()
    original(attention, prefill, None, stock_cache)
    original(attention, prefill, None, candidate_cache)
    mx.eval(
        stock_cache.keys,
        stock_cache.values,
        candidate_cache.keys,
        candidate_cache.values,
    )
    verify_x = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
    stock_mask = create_attention_mask(verify_x, stock_cache, return_array=True)
    candidate_mask = create_attention_mask(verify_x, candidate_cache, return_array=True)
    stock = original(attention, verify_x, stock_mask, stock_cache)
    candidate = attention(verify_x, candidate_mask, candidate_cache)
    mx.eval(stock, candidate)

    delta = mx.abs(stock.astype(mx.float32) - candidate.astype(mx.float32))
    assert float(mx.max(delta)) < 1e-5
    assert stock_cache.offset == candidate_cache.offset
    assert mx.array_equal(stock_cache.keys, candidate_cache.keys).item()
    assert mx.array_equal(stock_cache.values, candidate_cache.values).item()

    # The verification call must leave an equivalent cache for subsequent
    # single-token decoding, not merely produce matching immediate logits.
    follow_x = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)
    stock_follow = original(attention, follow_x, None, stock_cache)
    candidate_follow = attention(follow_x, None, candidate_cache)
    mx.eval(stock_follow, candidate_follow)
    follow_delta = mx.abs(
        stock_follow.astype(mx.float32) - candidate_follow.astype(mx.float32)
    )
    assert float(mx.max(follow_delta)) < 1e-5
    after_warm = patch.mla_absorbed_verify_stats()
    assert after_warm["absorbed"] == after_cold["absorbed"] + 1


@requires_mlx
@pytest.mark.requires_mlx
def test_standard_attention_without_q_lora_matches_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_lm.models import mla

    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    _, class_name, attention = _tiny_attention("deepseek_v3", q_lora_rank=None)
    patch.install_mla_absorbed_verify()
    original = mla._RAPID_MLX_MLA_ABSORBED_ORIGINALS[("deepseek_v3", class_name)]
    x = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)

    stock = original(attention, x, None, None)
    candidate = attention(x, None, None)
    mx.eval(stock, candidate)

    assert mx.array_equal(stock, candidate).item()


@requires_mlx
@pytest.mark.requires_mlx
def test_installer_idempotence_and_default_off_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.delenv("RAPID_MLX_MLA_ABSORBED_VERIFY", raising=False)
    patch.install_mla_absorbed_verify()
    assert patch.is_installed() is True
    assert patch.mla_absorbed_verify_stats()["provider"] == "disabled"

    # A second install is a literal no-op.
    patch.install_mla_absorbed_verify()
    assert patch.mla_absorbed_verify_stats()["provider"] == "disabled"


@requires_mlx
@pytest.mark.requires_mlx
def test_in_process_upstream_provider_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_lm.models import mla

    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.setattr(mla, "max_absorbed_queries", lambda *args: 1, raising=False)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    patch.install_mla_absorbed_verify()

    assert patch.mla_absorbed_verify_stats()["provider"] == "upstream"


@requires_mlx
@pytest.mark.requires_mlx
def test_in_process_unqualified_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    monkeypatch.setattr(patch, "_mlx_lm_version", lambda: None)
    patch.install_mla_absorbed_verify()

    assert patch.mla_absorbed_verify_stats()["provider"] == "unsupported"


@requires_mlx
@pytest.mark.requires_mlx
@pytest.mark.parametrize("case", ["missing", "marked", "unknown"])
def test_in_process_target_compatibility_gates(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from mlx_lm.models import deepseek_v3

    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    monkeypatch.setattr(patch, "_mlx_lm_version", lambda: "0.31.3")

    if case == "missing":
        targets = {("not_a_real_mlx_lm_model", "MissingAttention"): "hash"}
    else:
        key = ("deepseek_v3", "DeepseekV3Attention")
        targets = {key: "not-the-real-source-hash"}
        if case == "marked":
            marker = "_RAPID_MLX_MLA_ABSORBED_DeepseekV3Attention"
            monkeypatch.setattr(deepseek_v3, marker, True, raising=False)
    monkeypatch.setattr(patch, "_SUPPORTED_SOURCE_HASHES", targets)
    patch.install_mla_absorbed_verify()

    stats = patch.mla_absorbed_verify_stats()
    assert stats["provider"] == "none"
    if case == "marked":
        assert stats["targets"] == ("deepseek_v3.DeepseekV3Attention",)
    else:
        assert stats["targets"] == ()


def test_source_hash_fails_closed_for_uninspectable_callable() -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    assert patch._source_hash(len) is None


@requires_mlx
@pytest.mark.requires_mlx
def test_disabled_and_single_token_paths_delegate_to_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    _isolate_installer_state(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    _, _, attention = _tiny_attention("deepseek_v3")
    patch.install_mla_absorbed_verify()
    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)

    monkeypatch.setattr(patch, "_ENABLED", False)
    before = patch.mla_absorbed_verify_stats()
    mx.eval(attention(x))
    disabled = patch.mla_absorbed_verify_stats()
    assert disabled["disabled"] == before["disabled"] + 1

    monkeypatch.setattr(patch, "_ENABLED", True)
    mx.eval(attention(x))
    enabled = patch.mla_absorbed_verify_stats()
    assert enabled["single_token"] == disabled["single_token"] + 1


@requires_mlx
@pytest.mark.requires_mlx
def test_quantized_cache_shape_delegates_to_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.patches import mla_absorbed_verify as patch

    class QuantizedLikeCache:
        bits = 4
        offset = 0

        def update_and_fetch(self, keys, values):
            raise RuntimeError("stock quantized MLA path")

    _isolate_installer_state(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MLA_ABSORBED_VERIFY", "1")
    _, _, attention = _tiny_attention("deepseek_v3")
    patch.install_mla_absorbed_verify()
    x = mx.zeros((1, 3, 32), dtype=mx.bfloat16)
    before = patch.mla_absorbed_verify_stats()

    with pytest.raises(RuntimeError, match="stock quantized MLA path"):
        attention(x, cache=QuantizedLikeCache())

    after = patch.mla_absorbed_verify_stats()
    assert after["unsupported_cache"] == before["unsupported_cache"] + 1
    assert after["absorbed"] == before["absorbed"]
    assert after["materialized"] == before["materialized"]
