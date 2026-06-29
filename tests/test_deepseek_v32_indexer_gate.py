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


def test_decode_step_after_prefill_keeps_in_call_reuse(repro_dir):
    """Across forward calls (prefill → decode), the per-call topk reuse
    semantics is preserved: each full layer's forward refreshes
    ``last_topk_indices`` for the immediately following shared layers
    within the SAME call.

    Codex iteration history:

    * PR #967 round 3 BLOCKING flagged a (hypothetical) regression
      where a decode call's leading shared layer could see ``None``.
      The round-3 fix added a "seed from prior forward" loop.
    * PR #967 round 5 BLOCKING reversed: applying a prior call's
      persisted ``_last_topk_indices`` at a fresh decode step would
      be stale (computed at a different query/cache length). The
      seed loop was reverted; the architecturally-correct remedy if
      ever needed is inter-rank communication of the current full
      layer's topk, which is out of scope for the surgical D1 fix.

    Pin the now-stable behavior: with a valid REAP layout
    ``["full","shared","full","shared"]`` (every shared layer
    immediately preceded by a full layer in the iteration), both
    prefill and the subsequent decode step keep every shared layer
    on the REAP reuse path — no dense fallback.
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    gate.install_deepseek_v32_indexer_gate()
    repro = _forge_repro(repro_dir, ["full", "shared", "full", "shared"])

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)

    # Prefill
    cache = make_prompt_cache(model)
    gate._SHARED_LAYER_REUSE_COUNT = 0
    gate._SHARED_LAYER_DENSE_FALLBACK_COUNT = 0
    prefill = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(prefill, cache=cache)
    mx.eval(logits)
    assert gate._SHARED_LAYER_REUSE_COUNT == 2
    assert gate._SHARED_LAYER_DENSE_FALLBACK_COUNT == 0

    # Decode step (single new token). Each full layer in the iteration
    # refreshes ``last_topk_indices`` BEFORE the following shared
    # layer reads it, so in-call reuse semantics holds whether or not
    # any prior forward's state is reachable.
    gate._SHARED_LAYER_REUSE_COUNT = 0
    gate._SHARED_LAYER_DENSE_FALLBACK_COUNT = 0
    next_tok = mx.array([[6]], dtype=mx.int32)
    logits = model(next_tok, cache=cache)
    mx.eval(logits)
    assert gate._SHARED_LAYER_REUSE_COUNT == 2, (
        "decode step lost in-call topk reuse — "
        f"reuse={gate._SHARED_LAYER_REUSE_COUNT}, "
        f"dense_fallback={gate._SHARED_LAYER_DENSE_FALLBACK_COUNT}"
    )
    assert gate._SHARED_LAYER_DENSE_FALLBACK_COUNT == 0


def test_empty_local_layer_slice_delegates_to_upstream(repro_dir):
    """Pipeline-parallel layout with an out-of-range ``start_idx``
    safely delegates to ``_orig_model_call`` instead of crashing with
    ``IndexError``.

    Codex finding #3 on PR #967 round 5: the patched ``Model.__call__``
    used to dereference ``self.layers[self.start_idx]`` unconditionally.

    Codex finding #1 on PR #967 round 6: a prior version of this test
    only reimplemented the bounds-check inline and never invoked the
    patched call, so removing the guard would not have caused it to
    fail. This rewrite monkey-patches ``_orig_model_call`` to a
    sentinel-recording stub and invokes the patched call on a model
    whose ``start_idx`` is out of range, then asserts the sentinel was
    reached. If the bounds-check guard were removed, the patched call
    would raise ``IndexError`` BEFORE the sentinel is recorded.
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    gate.install_deepseek_v32_indexer_gate()
    # Non-REAP config — the orig path is the upstream Indexer-bearing one.
    repro = _forge_repro(repro_dir, indexer_types=None)

    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)
    inner = model.model

    # Out-of-range start_idx — no local layer to inspect.
    inner.start_idx = len(inner.layers)

    # Stub ``_orig_model_call`` so we can detect that delegation
    # actually happened (rather than letting the upstream forward
    # execute, which would still try to iterate a now-bogus slice).
    sentinel = {"reached": False, "args": None}

    def _stub_orig(self, x, cache=None):
        sentinel["reached"] = True
        sentinel["args"] = (x.shape, cache is None)
        return mx.zeros((x.shape[0], x.shape[1], 64), dtype=mx.float32)

    saved_orig = gate._orig_model_call
    try:
        gate._orig_model_call = _stub_orig
        # Invoke the actual patched call. Bug behavior (no bounds check)
        # would raise IndexError on ``self.layers[self.start_idx]`` before
        # reaching the delegation. Fixed behavior delegates cleanly.
        from mlx_lm.models import deepseek_v32 as ds

        # The patched method is bound at install time and stored on the
        # class. Call it directly so we're definitely exercising the
        # patched ``__call__``.
        x = mx.array([[1, 2, 3]], dtype=mx.int32)
        _ = ds.DeepseekV32Model.__call__(inner, x, cache=None)
    finally:
        gate._orig_model_call = saved_orig

    assert sentinel["reached"], (
        "_patched_model_call should have delegated to _orig_model_call "
        "when start_idx is out of range; sentinel was never recorded. "
        "Either the bounds-check guard fired incorrectly or the patched "
        "call short-circuited before reaching delegation."
    )


def test_reuse_path_runs_for_run_of_consecutive_shared_layers(repro_dir):
    """Layout with multiple consecutive shared layers after one full
    anchor — the production-shaped pattern in GLM-5.2 REAP configs.

    Codex finding #3 on PR #967 round 6: prior coverage only used the
    alternating ``["full","shared","full","shared"]`` pattern, so a
    regression that broke reuse for the second or third consecutive
    shared layer would not be caught. The actual GLM-5.2 base config
    has runs like ``["full","full","full","shared","shared","shared",
    "full","shared",...]``.

    Pin the in-call reuse semantic on a ``["full","shared","shared",
    "shared"]`` layout: a SINGLE prefill forward must drive every one
    of the three consecutive shared layers through the REAP reuse
    path (not dense fallback), because each shared layer's
    ``last_topk_indices`` carries forward across the layer loop
    until the next full layer would refresh it.
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    gate.install_deepseek_v32_indexer_gate()
    repro = _forge_repro(repro_dir, ["full", "shared", "shared", "shared"])

    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)
    layers = model.model.layers
    # Only layer 0 retains an Indexer; layers 1,2,3 are shared.
    assert layers[0].self_attn.indexer is not None
    for i in (1, 2, 3):
        assert layers[i].self_attn.indexer is None, f"layer {i} should be shared"

    # Forward — every shared layer must take the REAP reuse path.
    gate._SHARED_LAYER_REUSE_COUNT = 0
    gate._SHARED_LAYER_DENSE_FALLBACK_COUNT = 0
    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    out = model(prompt)
    mx.eval(out)
    assert gate._SHARED_LAYER_REUSE_COUNT == 3, (
        "all 3 consecutive shared layers must consume the layer-0 full "
        f"anchor's topk — reuse={gate._SHARED_LAYER_REUSE_COUNT}, "
        f"dense_fallback={gate._SHARED_LAYER_DENSE_FALLBACK_COUNT}. The "
        "layer loop must carry last_topk_indices across consecutive "
        "shared layers without resetting it (codex PR #967 round-6 "
        "finding)."
    )
    assert gate._SHARED_LAYER_DENSE_FALLBACK_COUNT == 0


def test_pp_shard_starting_on_shared_layer_raises_clear_error(repro_dir):
    """Pipeline-parallel shard whose first locally-iterated layer is
    ``"shared"`` raises a clear ``ValueError`` instead of silently
    falling back to dense attention.

    Codex finding #2 on PR #967 round 6: previously such a shard
    would silently take the dense fallback on its leading shared
    layers, changing inference semantics relative to the published
    REAP contract. Fail fast with a clear error message instead;
    cross-rank communication of the current full layer's topk is
    the architecturally-correct remedy if ever needed, but that's
    out of scope for the surgical D1 fix.
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    gate.install_deepseek_v32_indexer_gate()
    repro = _forge_repro(repro_dir, ["full", "shared", "shared", "shared"])

    from mlx_lm.utils import load_model

    model, _cfg = load_model(repro)
    inner = model.model

    # Simulate a PP rank that starts mid-model on a shared layer.
    inner.start_idx = 1
    inner.num_layers = 3

    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    with pytest.raises(ValueError, match="pipeline-parallel shard"):
        model(prompt)


def test_uninstall_restores_originals_across_module_reload(repro_dir):
    """``uninstall`` restores upstream callables even after a module-
    reload-style re-install (codex finding #2, PR #967 round 3).

    Sequence:
    1. install + uninstall on a fresh module — captures originals,
       restores them. Sanity.
    2. install (no uninstall yet) — sets the upstream marker.
    3. Simulate a module reload: clear THIS module's ``_orig_*``
       globals + ``_INSTALLED`` flag, then call ``install`` again.
       Bug behavior would leave ``_orig_*`` = None (because the
       early-return path skips capture); fixed behavior copies the
       originals from the upstream-stashed slot.
    4. uninstall — must restore the un-patched callables.
    5. Without the gate, loading a REAP config fails (proves the
       un-patched callables really are back in place).
    """
    from vllm_mlx.patches import deepseek_v32_indexer_gate as gate

    # (1) install + uninstall round-trip — fresh state.
    gate.uninstall_deepseek_v32_indexer_gate()  # be defensive
    gate._INSTALLED = False
    gate._orig_attn_call = None
    gate._orig_decoder_init = None
    gate._orig_indexer_call = None
    gate._orig_model_call = None
    gate._orig_from_dict = None
    gate.install_deepseek_v32_indexer_gate()
    assert gate._INSTALLED is True
    assert gate._orig_attn_call is not None
    gate.uninstall_deepseek_v32_indexer_gate()
    assert gate._INSTALLED is False
    assert gate._orig_attn_call is None

    # (2) install (now the upstream module has the marker).
    gate.install_deepseek_v32_indexer_gate()
    from mlx_lm.models import deepseek_v32 as ds

    assert getattr(ds, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False)

    # (3) simulate module-reload: clear the module-side globals and
    # call install again. The reload path must populate ``_orig_*``
    # from the upstream stash, not capture the currently-patched
    # callables.
    gate._INSTALLED = False
    gate._orig_attn_call = None
    gate._orig_decoder_init = None
    gate._orig_indexer_call = None
    gate._orig_model_call = None
    gate._orig_from_dict = None
    gate.install_deepseek_v32_indexer_gate()
    assert gate._INSTALLED is True
    assert gate._orig_attn_call is not None, (
        "install after module reload did not populate _orig_attn_call; "
        "uninstall would silently leave the patches in place. Codex "
        "PR #967 round-3 finding #2."
    )

    # (4) uninstall — should restore the true upstream callables.
    gate.uninstall_deepseek_v32_indexer_gate()
    assert gate._INSTALLED is False
    assert not getattr(ds, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False)

    # (5) Without the gate, loading a REAP config crashes with missing
    # indexer keys — proves the upstream un-patched callables really
    # are back in place.
    repro = _forge_repro(repro_dir, ["full", "shared", "full", "shared"])

    from mlx_lm.utils import load_model

    with pytest.raises(ValueError) as exc_info:
        load_model(repro)
    msg = str(exc_info.value)
    assert "Missing 10 parameters" in msg, msg
    assert "indexer" in msg, msg

    # Re-arm the gate for downstream tests.
    gate.install_deepseek_v32_indexer_gate()


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
