# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Defect 1 indexer gate.

Pins the surgical fix shipped in
``vllm_mlx.patches.deepseek_v32_indexer_gate`` for REAP-pruned DeepseekV32
configs (e.g. ``mlx-community/pipenetwork-GLM-5.2-REAP50-MLX-4bit``):

1. Without the gate installed, ``mlx_lm.utils.load_model`` on a config
   carrying ``indexer_types`` aborts with ``Missing N parameters:
   ...indexer...`` for each ``"shared"`` layer. This is the bug we are
   patching around — pin it so we notice if mlx_lm upstream changes
   behavior.

2. With the gate installed, the same config loads cleanly; ``"shared"``
   layers have ``self_attn.indexer is None`` while ``"full"`` layers
   still carry an ``Indexer`` instance.

3. Backward-compat: a config WITHOUT ``indexer_types`` keeps upstream
   behavior (every layer has an Indexer; gate is a no-op).

4. Edge case: an all-``"shared"`` ``indexer_types`` raises ``ValueError``
   on model construction — there is no valid REAP config without at
   least one ``"full"`` anchor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import pytest

# ----------------------------------------------------------------------
# Synthetic glm_moe_dsa config + weight forge (no GLM-5.2 download)
# ----------------------------------------------------------------------


def _config(num_layers: int, indexer_types: Iterable[str] | None) -> dict:
    cfg = {
        "model_type": "glm_moe_dsa",
        "vocab_size": 256,
        "hidden_size": 64,
        "index_head_dim": 16,
        "index_n_heads": 4,
        "index_topk": 4,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": None,
        "n_routed_experts": None,
        "routed_scaling_factor": 1.0,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "qk_nope_head_dim": 32,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 0,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0},
        "attention_bias": False,
    }
    if indexer_types is not None:
        cfg["indexer_types"] = list(indexer_types)
    return cfg


def _layer_keys(layer_idx: int, mode: str, cfg: dict) -> dict:
    H = cfg["hidden_size"]
    Q = cfg["q_lora_rank"]
    KV = cfg["kv_lora_rank"]
    QKR = cfg["qk_rope_head_dim"]
    QK_NOPE = cfg["qk_nope_head_dim"]
    VH = cfg["v_head_dim"]
    NH = cfg["num_attention_heads"]
    INT = cfg["intermediate_size"]
    IH = cfg["index_head_dim"]
    IN = cfg["index_n_heads"]
    prefix = f"model.layers.{layer_idx}"
    keys = {
        f"{prefix}.input_layernorm.weight": (H,),
        f"{prefix}.post_attention_layernorm.weight": (H,),
        f"{prefix}.self_attn.q_a_proj.weight": (Q, H),
        f"{prefix}.self_attn.q_a_layernorm.weight": (Q,),
        f"{prefix}.self_attn.q_b_proj.weight": (NH * (QK_NOPE + QKR), Q),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": (KV + QKR, H),
        f"{prefix}.self_attn.kv_a_layernorm.weight": (KV,),
        f"{prefix}.self_attn.embed_q.weight": (NH, KV, QK_NOPE),
        f"{prefix}.self_attn.unembed_out.weight": (NH, VH, KV),
        f"{prefix}.self_attn.o_proj.weight": (H, NH * VH),
        f"{prefix}.mlp.gate_proj.weight": (INT, H),
        f"{prefix}.mlp.up_proj.weight": (INT, H),
        f"{prefix}.mlp.down_proj.weight": (H, INT),
    }
    if mode == "full":
        keys.update(
            {
                f"{prefix}.self_attn.indexer.wq_b.weight": (IN * IH, Q),
                f"{prefix}.self_attn.indexer.wk.weight": (IH, H),
                f"{prefix}.self_attn.indexer.k_norm.weight": (IH,),
                f"{prefix}.self_attn.indexer.k_norm.bias": (IH,),
                f"{prefix}.self_attn.indexer.weights_proj.weight": (IN, H),
            }
        )
    return keys


def _forge_repro(
    tmp_path: Path,
    indexer_types: list[str] | None,
    layer_modes_for_safetensors: list[str] | None = None,
) -> Path:
    """Write a tiny config + safetensors stub for the test.

    ``layer_modes_for_safetensors`` defaults to ``indexer_types`` (one
    indexer per "full" layer, none per "shared" layer). Tests that want
    to lie about what's on disk (e.g. emit indexer weights for every
    layer when ``indexer_types`` is None) can override.
    """
    cfg = _config(num_layers=4, indexer_types=indexer_types)
    repro = tmp_path / "model"
    repro.mkdir(parents=True, exist_ok=True)
    with open(repro / "config.json", "w") as f:
        json.dump(cfg, f)

    layer_modes = layer_modes_for_safetensors or (
        indexer_types if indexer_types is not None else ["full"] * 4
    )
    weights = {
        "model.embed_tokens.weight": mx.zeros(
            (cfg["vocab_size"], cfg["hidden_size"]), dtype=mx.float32
        ),
        "model.norm.weight": mx.zeros((cfg["hidden_size"],), dtype=mx.float32),
        "lm_head.weight": mx.zeros(
            (cfg["vocab_size"], cfg["hidden_size"]), dtype=mx.float32
        ),
    }
    for i, mode in enumerate(layer_modes):
        for k, shape in _layer_keys(i, mode, cfg).items():
            weights[k] = mx.zeros(shape, dtype=mx.float32)
    mx.save_safetensors(str(repro / "model.safetensors"), weights)
    return repro


@pytest.fixture
def repro_dir(tmp_path: Path) -> Path:
    return tmp_path


# ----------------------------------------------------------------------
# 1. WITHOUT-patch baseline — pins the upstream-bug surface.
# ----------------------------------------------------------------------


def test_upstream_without_gate_fails_with_missing_indexer_keys(monkeypatch, repro_dir):
    """Pin the bug we are patching around.

    Uninstall the gate, then call ``mlx_lm.utils.load_model`` on a 4-layer
    glm_moe_dsa config with ``indexer_types=["full","shared","full","shared"]``.
    Upstream should abort with ``Missing 10 parameters: ...indexer...``.
    """
    from vllm_mlx.patches.deepseek_v32_indexer_gate import (
        install_deepseek_v32_indexer_gate,
        uninstall_deepseek_v32_indexer_gate,
    )

    # Ensure baseline: gate uninstalled for the duration of this test.
    install_deepseek_v32_indexer_gate()  # capture originals
    uninstall_deepseek_v32_indexer_gate()
    monkeypatch.setattr(
        "vllm_mlx.patches.deepseek_v32_indexer_gate._INSTALLED",
        False,
        raising=False,
    )

    repro = _forge_repro(repro_dir, ["full", "shared", "full", "shared"])

    from mlx_lm.utils import load_model

    with pytest.raises(ValueError) as exc_info:
        load_model(repro)

    msg = str(exc_info.value)
    assert "Missing" in msg
    assert "indexer" in msg
    # Each "shared" layer is missing exactly the 5 Indexer keys; 2 shared
    # layers * 5 keys = 10 total. Pin the count so a future upstream
    # refactor of Indexer.__init__ is caught here.
    assert "Missing 10 parameters" in msg

    # Re-arm the gate for downstream tests in this session.
    install_deepseek_v32_indexer_gate()


# ----------------------------------------------------------------------
# 2. WITH-patch — the actual fix.
# ----------------------------------------------------------------------


def test_gate_loads_mixed_full_shared_config(repro_dir):
    """4-layer ``["full","shared","full","shared"]`` config loads cleanly,
    and shared layers REUSE the prior full layer's indexer topk (the REAP
    contract — not a dense fallback).

    Pins four properties beyond "no crash on load":

    1. Only "full" layers retain an Indexer; "shared" layers have
       ``self_attn.indexer is None`` (load-time gate fired correctly).
    2. The shared-layer attention codepath is actually invoked during
       forward (the counter ``_SHARED_LAYER_FORWARD_COUNT`` increments
       by exactly the number of shared layers traversed). Catches
       regressions where a refactor accidentally routes shared layers
       back through the upstream Indexer-bearing ``__call__``.
    3. The REAP reuse path fires: every shared layer consumes a
       non-None ``topk_indices`` threaded from the prior full layer
       (``_SHARED_LAYER_REUSE_COUNT == 2``,
       ``_SHARED_LAYER_DENSE_FALLBACK_COUNT == 0``). The synthetic
       config has ``index_topk=4`` and the prompt has 5 tokens, so
       the indexer's internal early-exit (``k.shape[2] <= index_topk
       → return None``) does NOT fire and the threaded topk is real.
       Codex finding #1 on PR #967 round 2: the shared layer must NOT
       silently take the dense fallback when the model author's intent
       is sparse top-K reuse.
    4. Forward output is finite (no NaN/Inf) on both the reuse and
       dense paths.
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    gate.install_deepseek_v32_indexer_gate()
    repro = _forge_repro(repro_dir, ["full", "shared", "full", "shared"])

    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)
    layers = model.model.layers
    assert len(layers) == 4
    # (1) load-time gate
    assert layers[0].self_attn.indexer is not None
    assert layers[1].self_attn.indexer is None
    assert layers[2].self_attn.indexer is not None
    assert layers[3].self_attn.indexer is None

    # (2) shared-layer code path is exercised exactly once per shared layer
    # per forward call. Reset counters, run forward, assert deltas.
    gate._SHARED_LAYER_FORWARD_COUNT = 0
    gate._SHARED_LAYER_REUSE_COUNT = 0
    gate._SHARED_LAYER_DENSE_FALLBACK_COUNT = 0
    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    out = model(prompt)
    mx.eval(out)
    assert out.shape == (1, 5, 256)
    # 2 "shared" layers in this config; forward visits each once.
    assert gate._SHARED_LAYER_FORWARD_COUNT == 2, (
        "shared-layer attention path was not invoked exactly twice — "
        f"got {gate._SHARED_LAYER_FORWARD_COUNT}. A regression may have "
        "routed shared layers back through the upstream Indexer path."
    )

    # (3) REAP reuse path is the one that fired (prompt=5 tok > index_topk=4
    # so the prior full layer's indexer DID produce a topk tensor).
    assert gate._SHARED_LAYER_REUSE_COUNT == 2, (
        "shared-layer REAP reuse path did not fire on both shared layers — "
        f"got reuse={gate._SHARED_LAYER_REUSE_COUNT}, "
        f"dense_fallback={gate._SHARED_LAYER_DENSE_FALLBACK_COUNT}. The "
        "prior full layer's topk_indices were not threaded through the "
        "model loop, so shared layers silently ran dense attention instead "
        "of reusing the top-K KV selection (codex PR #967 round-2 finding)."
    )
    assert gate._SHARED_LAYER_DENSE_FALLBACK_COUNT == 0

    # (4) logits are finite (synthetic zero weights make absolute values
    # tiny but the shared-layer attention path should not produce NaN/Inf).
    import math

    out_max = float(mx.max(mx.abs(out)).item())
    assert math.isfinite(out_max), (
        f"forward output contains NaN/Inf (max |.|={out_max}); the "
        "shared-layer attention path produced numerically invalid logits"
    )


# ----------------------------------------------------------------------
# 3. Backward-compat — config without ``indexer_types`` is unchanged.
# ----------------------------------------------------------------------


def test_gate_is_noop_when_indexer_types_absent(repro_dir):
    """Non-REAP config loads with every layer having an Indexer.

    This pins backward-compat: the gate must NOT change behavior on
    configs lacking ``indexer_types`` (legitimate non-REAP DSv32 /
    GLM-4.6 models). All 4 layers get full Indexer construction and
    every safetensors key matches.
    """
    from vllm_mlx.patches.deepseek_v32_indexer_gate import (
        install_deepseek_v32_indexer_gate,
    )

    install_deepseek_v32_indexer_gate()
    # All layers emit indexer weights — what upstream non-REAP models do.
    repro = _forge_repro(repro_dir, indexer_types=None)

    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)
    layers = model.model.layers
    assert len(layers) == 4
    for i, layer in enumerate(layers):
        assert layer.self_attn.indexer is not None, (
            f"layer {i} should retain Indexer when indexer_types is absent"
        )


# ----------------------------------------------------------------------
# 4. Edge case — all-"shared" is rejected with a clear error.
# ----------------------------------------------------------------------


def test_gate_rejects_all_shared_indexer_types(repro_dir):
    """``indexer_types=["shared"]*4`` is invalid; raise clearly.

    A real REAP-pruned model always has at least one ``"full"`` anchor —
    the shared layers reuse a prior full layer's indexer output. An all-
    shared config has no Indexer at all and could never be inferenced.
    Fail fast at model build instead of crashing later in forward.

    The (a) index-0 check fires first (``shared`` at the first layer has
    no anchor to reuse), so the ``ValueError`` message specifically calls
    out the first-layer violation; the all-shared check is the (b)
    defensive backstop covered by ``test_gate_rejects_shared_at_index_zero``.
    """
    from vllm_mlx.patches.deepseek_v32_indexer_gate import (
        install_deepseek_v32_indexer_gate,
    )

    install_deepseek_v32_indexer_gate()
    # Use "full" safetensors-mode so the failure is the validator's
    # error, not "missing indexer keys" downstream.
    repro = _forge_repro(
        repro_dir,
        ["shared"] * 4,
        layer_modes_for_safetensors=["full"] * 4,
    )

    from mlx_lm.utils import load_model

    with pytest.raises(
        ValueError, match="(?:no 'full' anchor|first layer must be 'full')"
    ):
        load_model(repro)


def test_gate_rejects_shared_at_index_zero(repro_dir):
    """``["shared", "full", "full", "full"]`` is invalid — index-0 has no
    anchor to reuse.

    Codex finding (PR #967, round 1): the original ``_validate_anchor``
    only rejected all-``"shared"`` configs and accepted ``["shared",
    "full", ...]`` as valid. The first-layer guard added in response
    rejects any ``indexer_types[0] != "full"``; pin it here.
    """
    from vllm_mlx.patches.deepseek_v32_indexer_gate import (
        install_deepseek_v32_indexer_gate,
    )

    install_deepseek_v32_indexer_gate()
    repro = _forge_repro(
        repro_dir,
        ["shared", "full", "full", "full"],
        # Lie about safetensors so the validator fires before
        # missing-keys does (layer 0's safetensors emits "full" keys).
        layer_modes_for_safetensors=["full"] * 4,
    )

    from mlx_lm.utils import load_model

    with pytest.raises(ValueError, match="first layer must be 'full'"):
        load_model(repro)
