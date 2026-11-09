# SPDX-License-Identifier: Apache-2.0
"""Exercise projection enrollment through the real multimodal load method."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx

pytest.importorskip("mlx.core")
mlx_vlm = pytest.importorskip("mlx_vlm")


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "lfm2", "qwen3_vl"])
@pytest.mark.parametrize("mode", ["default", "none", "mtp", "auto", "future", None])
def test_load_enrolls_only_plain_qwen_backbones(
    monkeypatch, tmp_path, model_type, mode
):
    import mlx_vlm.utils

    from rapid_mlx import (
        dense_decode_fusion,
        gdn_in_proj_fusion,
        moe_fusion,
        qwen35_fused_gdn_decode,
        qwen35_moe_router,
    )
    from rapid_mlx.models import media_backend, mllm
    from rapid_mlx.patches import (
        glm5_next_forget_gate_quant,
        glm5_next_processor,
        glm5_next_runtime,
    )
    from rapid_mlx.utils import model_file_guard, tokenizer

    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    enrolled = []
    recurrent = []
    dense = []
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(mllm, "_resolve_checkpoint", lambda _: str(tmp_path))
    monkeypatch.setattr(media_backend, "native_media_model_type", lambda _: None)
    monkeypatch.setattr(mlx_vlm, "load", lambda *a, **k: (model, processor))
    monkeypatch.setattr(
        mlx_vlm.utils, "load_config", lambda *a, **k: {"model_type": model_type}
    )
    monkeypatch.setattr(model_file_guard, "validate_model_code", lambda *a, **k: None)
    monkeypatch.setattr(
        model_file_guard, "validate_vlm_processor_policy", lambda *a, **k: None
    )
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *a: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *a: None)
    monkeypatch.setattr(
        glm5_next_runtime, "install_glm5_next_runtime_fix", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_processor, "install_glm5_next_processor_patch", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_forget_gate_quant,
        "install_glm5_next_forget_gate_quant_fix",
        lambda: None,
    )
    monkeypatch.setattr(moe_fusion, "fuse_gate_up", lambda *a, **k: None)
    monkeypatch.setattr(qwen35_moe_router, "install_qwen35_moe_router", lambda *a: None)
    monkeypatch.setattr(
        qwen35_fused_gdn_decode,
        "install_qwen35_fused_gdn_decode",
        lambda loaded, **kw: recurrent.append((loaded, kw)),
    )
    monkeypatch.setattr(
        gdn_in_proj_fusion,
        "fuse_gdn_in_proj",
        lambda loaded: enrolled.append(loaded),
    )

    monkeypatch.setattr(
        dense_decode_fusion,
        "install_dense_decode_fusion",
        lambda loaded, **kw: dense.append((loaded, kw)),
    )

    kwargs = {} if mode == "default" else {"spec_decode": mode}
    wrapper = mllm.MLXMultimodalLM("unit/dense-qwen", enable_cache=False, **kwargs)
    wrapper.load()
    wrapper.load()
    expected = model_type in {"qwen3_5", "qwen3_5_moe"} and mode in {"default", "none"}
    dense_kw = {"spec_decode": "none" if mode == "default" else mode}
    assert dense == ([(model, dense_kw)] if model_type in {"qwen3_5", "lfm2"} else [])
    assert wrapper._loaded
    assert enrolled == ([model] if expected else [])

    if model_type == "qwen3_5" and expected:
        assert recurrent == [(model, {"allow_dense": True})]
    elif model_type == "qwen3_5_moe":
        assert recurrent == [(model, {})]
    else:
        assert recurrent == []
