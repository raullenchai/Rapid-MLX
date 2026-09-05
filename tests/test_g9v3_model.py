# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored G9v3 (``ai9stars/G9v3-39A5B``) MoE backbone.

Pins the contract that:

1. The vendored module is importable and registers into mlx-lm's
   importlib lookup (``_register_vendored_archs``).
2. Config parsing reproduces the released 39B-A5B shape from an empty
   config, derives ``head_dim`` like the remote ``G9v3Config``, and rejects
   configs the port cannot honour (non-SiLU activation, bad GQA/MoE
   arities).
3. Layer 0 is dense and layers ``>= first_k_dense_replace`` are MoE.
4. A tiny synthetic config constructs + runs the model: logits shape and
   incremental (cache) decode matching the full-prefill forward.
5. The gated-attention ``q_proj`` split takes the *first* ``head_dim`` of
   every ``2 * head_dim`` head slice as the query and the second as the
   gate (transformers ``torch.chunk`` semantics).
6. The router reproduces the remote ``G9v3TopkRouter`` maths (sigmoid,
   bias for selection only, normalised raw scores × scaling factor).
7. ``sanitize`` stacks per-expert checkpoint tensors into the ``SwitchGLU``
   layout (bf16 and quantized triples), round-trips through strict
   ``load_weights``, and fails loudly on a missing expert.
8. The curated alias pins the minicpm/qwen3 parsers and MoE flags.

Numeric fidelity against the remote transformers implementation was
verified out-of-band (numbers in the PR body of #3046): identical random
weights in fp32 (3 layers, gated attention, 8 experts / 2 active, random
``e_score_correction_bias``, full-prefill and token-by-token) agree to
~1e-5; the released weights in fp32 (config-truncated to 3 and 8 layers)
agree to ~1e-6 per layer / 99.6-100% top-1 over all prompt positions; the
full 38-layer bf16 comparison sits inside transformers' own eager-vs-sdpa
bf16 spread. The reference needs ``trust_remote_code`` + torch, which CI
does not install, so that check is not repeated here.
"""

import importlib
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from vllm_mlx.models import g9v3  # noqa: E402

TINY = dict(
    model_type="g9v3",
    hidden_size=64,
    num_hidden_layers=3,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=256,
)


def tiny_model(**overrides):
    args = g9v3.ModelArgs.from_dict(dict(TINY, **overrides))
    model = g9v3.Model(args)
    # Router params are zero-initialised (ties everywhere); give them a
    # deterministic random routing so the tests exercise real top-k paths.
    mx.random.seed(0)
    for layer in model.layers:
        if isinstance(layer.mlp, g9v3.MoE):
            gate = layer.mlp.gate
            gate.weight = 0.5 * mx.random.normal(gate.weight.shape)
            gate.e_score_correction_bias = 0.5 * mx.random.normal(
                gate.e_score_correction_bias.shape
            )
    mx.eval(model.parameters())
    return model


@pytest.fixture(autouse=True)
def _clear_vendored_register():
    """Registration is sys.modules-level state — reset around each test."""
    sys.modules.pop("mlx_lm.models.g9v3", None)
    yield
    sys.modules.pop("mlx_lm.models.g9v3", None)


def test_module_contract():
    assert hasattr(g9v3, "Model")
    assert hasattr(g9v3, "ModelArgs")
    assert g9v3.ModelArgs.__dataclass_fields__["model_type"].default == "g9v3"


def test_register_vendored_archs_makes_mlx_lm_loader_find_it():
    from vllm_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    assert "mlx_lm.models.g9v3" not in sys.modules
    _register_vendored_archs()
    assert "mlx_lm.models.g9v3" in sys.modules
    assert "g9v3" in _VENDORED_MODEL_TYPES

    # mlx-lm's _get_classes() does exactly this lookup.
    mod = importlib.import_module("mlx_lm.models.g9v3")
    assert mod is sys.modules["mlx_lm.models.g9v3"]
    assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")

    # Idempotent.
    _register_vendored_archs()
    assert importlib.import_module("mlx_lm.models.g9v3") is mod


def _reset_g9v3_registration(monkeypatch):
    from vllm_mlx.utils import tokenizer as tok

    monkeypatch.delitem(sys.modules, "mlx_lm.models.g9v3", raising=False)
    monkeypatch.setattr(tok, "_VENDORED_MODEL_TYPES", set(tok._VENDORED_MODEL_TYPES))
    tok._VENDORED_MODEL_TYPES.discard("g9v3")
    return tok


def test_registration_defers_to_native_mlx_lm(monkeypatch):
    """When mlx-lm ships its own ``mlx_lm.models.g9v3`` the vendored copy
    stays out of ``sys.modules`` but the model type is still marked vendored
    (keeps the tokenizer fallback off transformers' AutoConfig)."""
    import importlib.util

    tok = _reset_g9v3_registration(monkeypatch)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mlx_lm.models.g9v3":
            return importlib.util.spec_from_loader(name, loader=None)
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    tok._register_vendored_archs()
    assert "mlx_lm.models.g9v3" not in sys.modules
    assert "g9v3" in tok._VENDORED_MODEL_TYPES


def test_registration_marks_preimported_native_module(monkeypatch):
    """A native module imported before the hook still gets tokenizer fallback."""
    tok = _reset_g9v3_registration(monkeypatch)
    native = types.ModuleType("mlx_lm.models.g9v3")
    monkeypatch.setitem(sys.modules, "mlx_lm.models.g9v3", native)

    tok._register_vendored_archs()

    assert sys.modules["mlx_lm.models.g9v3"] is native
    assert "g9v3" in tok._VENDORED_MODEL_TYPES


def test_registration_survives_find_spec_errors(monkeypatch):
    """A broken/partial mlx-lm install that makes ``find_spec`` raise still
    ends with the vendored module registered."""
    import importlib.util

    tok = _reset_g9v3_registration(monkeypatch)
    real_find_spec = importlib.util.find_spec

    def raising_find_spec(name, *args, **kwargs):
        if name == "mlx_lm.models.g9v3":
            raise ValueError("mlx_lm.models.g9v3.__spec__ is None")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", raising_find_spec)
    tok._register_vendored_archs()
    assert sys.modules["mlx_lm.models.g9v3"] is g9v3
    assert "g9v3" in tok._VENDORED_MODEL_TYPES


def test_registration_warns_when_vendored_import_fails(monkeypatch, caplog):
    """An import failure of the vendored module is logged, not raised: the
    rest of the loader (and every other vendored family) keeps working."""
    import vllm_mlx.models

    tok = _reset_g9v3_registration(monkeypatch)
    # ``from ..models import g9v3`` takes the package attribute when it is
    # already bound; drop it so the import goes through sys.modules, where
    # ``None`` makes it raise ImportError.
    monkeypatch.delattr(vllm_mlx.models, "g9v3", raising=False)
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.g9v3", None)
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.utils.tokenizer"):
        tok._register_vendored_archs()
    assert "mlx_lm.models.g9v3" not in sys.modules
    assert "g9v3" not in tok._VENDORED_MODEL_TYPES
    assert any(
        "g9v3 vendored module failed to register" in r.message for r in caplog.records
    )


def test_released_defaults():
    """An empty config builds the released 39B-A5B shape."""
    args = g9v3.ModelArgs.from_dict({"model_type": "g9v3"})
    assert (args.hidden_size, args.num_hidden_layers, args.vocab_size) == (
        2048,
        38,
        130560,
    )
    assert (args.num_attention_heads, args.num_key_value_heads, args.head_dim) == (
        32,
        2,
        128,
    )
    assert args.use_gated_attention is True
    assert args.attention_bias is False
    assert args.rope_theta == 5000000.0 and args.rope_scaling is None
    assert (args.n_routed_experts, args.num_experts_per_tok, args.n_shared_experts) == (
        320,
        32,
        1,
    )
    assert (args.intermediate_size, args.moe_intermediate_size) == (8192, 512)
    assert args.first_k_dense_replace == 1
    assert (args.n_group, args.topk_group, args.norm_topk_prob) == (1, 1, True)
    assert args.routed_scaling_factor == 3.66
    assert args.tie_word_embeddings is False


def test_head_dim_derived_like_remote_config():
    args = g9v3.ModelArgs.from_dict(dict(TINY, head_dim=None))
    assert args.head_dim == TINY["hidden_size"] // TINY["num_attention_heads"]
    model = g9v3.Model(args)
    assert model(mx.array([[1, 2, 3]])).shape == (1, 3, TINY["vocab_size"])


@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(hidden_act="gelu"), "hidden_act"),
        (dict(num_key_value_heads=3), "num_attention_heads"),
        (dict(num_key_value_heads=0), "num_key_value_heads .* must be positive"),
        (dict(num_attention_heads=0, head_dim=None), "num_attention_heads .* positive"),
        (dict(hidden_size=0), "hidden_size .* positive"),
        (dict(num_hidden_layers=0), "num_hidden_layers .* positive"),
        (dict(intermediate_size=-1), "intermediate_size .* positive"),
        (dict(vocab_size=0), "vocab_size .* positive"),
        (dict(head_dim=0), "head_dim .* positive"),
        (dict(hidden_size=2, head_dim=None), "head_dim .* positive"),
        (dict(moe_intermediate_size=0), "moe_intermediate_size .* positive"),
        (dict(n_shared_experts=-1), "n_shared_experts .* >= 0"),
        # 8 experts in 2 groups, 1 group kept -> only 4 experts selectable.
        (dict(n_group=2, topk_group=1, num_experts_per_tok=8), "selectable"),
        (dict(num_experts_per_tok=9), "num_experts_per_tok"),
        (dict(num_experts_per_tok=0), "num_experts_per_tok"),
        (dict(n_group=3), "n_group"),
        (dict(n_group=2, topk_group=3), "topk_group"),
        (dict(first_k_dense_replace=4), "first_k_dense_replace"),
    ],
)
def test_config_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        g9v3.ModelArgs.from_dict(dict(TINY, **bad))


def test_grouped_routing_config_builds_and_runs():
    """A grouped-routing config within the selectable bound goes through
    ``group_expert_select``'s n_group > 1 path."""
    args = g9v3.ModelArgs.from_dict(
        dict(TINY, n_group=2, topk_group=1, num_experts_per_tok=4)
    )
    model = g9v3.Model(args)
    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, TINY["vocab_size"])
    assert bool(mx.all(mx.isfinite(logits)))


def test_cache_length_must_match_layers():
    from mlx_lm.models.cache import make_prompt_cache

    model = tiny_model()
    cache = make_prompt_cache(model)
    x = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="cache has 2 entries for 3 layers"):
        model(x, cache=cache[:-1])
    with pytest.raises(ValueError, match="cache has 4 entries for 3 layers"):
        model(x, cache=cache + [None])


def test_layer_kinds_follow_first_k_dense_replace():
    model = tiny_model()
    kinds = [type(layer.mlp) for layer in model.layers]
    assert kinds == [g9v3.MLP, g9v3.MoE, g9v3.MoE]

    all_moe = g9v3.Model(g9v3.ModelArgs.from_dict(dict(TINY, first_k_dense_replace=0)))
    assert all(isinstance(layer.mlp, g9v3.MoE) for layer in all_moe.layers)

    # An all-dense config must not need MoE arities to be valid at all.
    all_dense = g9v3.Model(
        g9v3.ModelArgs.from_dict(
            dict(
                TINY, first_k_dense_replace=3, n_routed_experts=0, num_experts_per_tok=0
            )
        )
    )
    assert all(isinstance(layer.mlp, g9v3.MLP) for layer in all_dense.layers)
    assert all_dense(mx.array([[1, 2, 3]])).shape == (1, 3, TINY["vocab_size"])


def test_forward_shape_and_cache_parity():
    model = tiny_model()
    ids = mx.random.randint(0, TINY["vocab_size"], (2, 24))

    logits = model(ids)
    assert logits.shape == (2, 24, TINY["vocab_size"])
    assert bool(mx.isfinite(logits).all())

    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in model.layers]
    steps = [model(ids[:, i : i + 1], cache=cache) for i in range(ids.shape[1])]
    inc = mx.concatenate(steps, axis=1)
    diff = float(mx.abs(inc - logits).max())
    assert diff < 2e-3, diff


def test_input_embeddings_used_raw():
    model = tiny_model()
    ids = mx.array([[1, 2, 3]])
    via_ids = model(ids)
    via_embeds = model(ids, input_embeddings=model.model.embed_tokens(ids))
    assert float(mx.abs(via_ids - via_embeds).max()) < 1e-6


def test_gated_q_proj_split_layout():
    """Per head the ``2 * head_dim`` slice is ``[query | gate]``. With the gate
    rows zeroed, ``sigmoid(0) = 0.5`` halves the ungated attention output;
    a swapped layout would zero the queries instead."""
    args_gated = g9v3.ModelArgs.from_dict(TINY)
    args_plain = g9v3.ModelArgs.from_dict(dict(TINY, use_gated_attention=False))
    gated, plain = g9v3.Attention(args_gated), g9v3.Attention(args_plain)
    assert gated.q_proj.weight.shape[0] == 2 * plain.q_proj.weight.shape[0]

    n_heads, hd, hidden = (
        TINY["num_attention_heads"],
        TINY["head_dim"],
        TINY["hidden_size"],
    )
    for name in ("k_proj", "v_proj", "o_proj"):
        getattr(gated, name).weight = getattr(plain, name).weight
    wq = np.array(plain.q_proj.weight).reshape(n_heads, hd, hidden)
    w_gated = np.concatenate([wq, np.zeros_like(wq)], axis=1).reshape(
        n_heads * 2 * hd, hidden
    )
    gated.q_proj.weight = mx.array(w_gated)

    x = mx.random.normal((1, 7, hidden))
    out_gated, out_plain = gated(x, mask="causal"), plain(x, mask="causal")
    assert float(mx.abs(out_gated - 0.5 * out_plain).max()) < 1e-5
    # And the gate actually gates: swapping the halves changes the output.
    swapped = np.concatenate([np.zeros_like(wq), wq], axis=1).reshape(
        n_heads * 2 * hd, hidden
    )
    gated.q_proj.weight = mx.array(swapped)
    assert float(mx.abs(gated(x, mask="causal") - 0.5 * out_plain).max()) > 1e-3


def _reference_router(x, weight, bias, top_k, scaling, norm_topk_prob):
    """Numpy port of the remote ``G9v3TopkRouter.forward`` (n_group = 1)."""
    logits = x.astype(np.float32) @ weight.astype(np.float32).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    idx = np.argsort(-(scores + bias), axis=-1)[..., :top_k]
    weights = np.take_along_axis(scores, idx, axis=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return idx, weights * scaling


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_router_matches_remote_reference(norm_topk_prob):
    args = g9v3.ModelArgs.from_dict(
        dict(TINY, num_experts_per_tok=3, norm_topk_prob=norm_topk_prob)
    )
    gate = g9v3.MoEGate(args)
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(TINY["n_routed_experts"], TINY["hidden_size"])).astype(
        np.float32
    )
    bias = rng.normal(size=(TINY["n_routed_experts"],)).astype(np.float32)
    gate.weight, gate.e_score_correction_bias = mx.array(weight), mx.array(bias)
    x = rng.normal(size=(5, TINY["hidden_size"])).astype(np.float32)

    inds, scores = gate(mx.array(x))
    inds, scores = np.array(inds), np.array(scores)
    ref_inds, ref_scores = _reference_router(
        x, weight, bias, 3, args.routed_scaling_factor, norm_topk_prob
    )
    # Selection order is unspecified on both sides: compare as index → weight.
    for row in range(x.shape[0]):
        got = dict(zip(inds[row].tolist(), scores[row].tolist()))
        want = dict(zip(ref_inds[row].tolist(), ref_scores[row].tolist()))
        assert got.keys() == want.keys(), (row, got, want)
        for e in want:
            assert abs(got[e] - want[e]) < 1e-5, (row, e, got[e], want[e])
    if norm_topk_prob:
        np.testing.assert_allclose(
            scores.sum(axis=-1), args.routed_scaling_factor, rtol=1e-5
        )


def test_sanitize_stacks_experts_and_round_trips_strict_load():
    model = tiny_model()
    stacked = dict(tree_flatten(model.parameters()))
    n_experts = TINY["n_routed_experts"]

    # Explode the MLX layout back into transformers' per-expert keys.
    checkpoint = {}
    for k, v in stacked.items():
        if ".switch_mlp." in k:
            prefix, rest = k.split(".switch_mlp.")
            for e in range(n_experts):
                checkpoint[f"{prefix}.experts.{e}.{rest}"] = v[e]
        else:
            checkpoint[k] = v
    assert "model.layers.1.mlp.experts.7.down_proj.weight" in checkpoint
    assert "model.layers.0.mlp.gate_proj.weight" in checkpoint  # dense layer untouched
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in checkpoint

    out = model.sanitize(dict(checkpoint))
    assert not any(".experts." in k for k in out)
    assert sorted(out) == sorted(stacked)
    for k in stacked:
        assert out[k].shape == stacked[k].shape, k
        assert bool(mx.array_equal(out[k], stacked[k])), k

    fresh = g9v3.Model(g9v3.ModelArgs.from_dict(TINY))
    fresh.load_weights(list(out.items()), strict=True)
    ids = mx.array([[3, 1, 4, 1, 5]])
    assert float(mx.abs(fresh(ids) - model(ids)).max()) < 1e-6

    # Already-stacked (an MLX export) passes through unchanged.
    again = model.sanitize(dict(stacked))
    assert sorted(again) == sorted(stacked)


def test_sanitize_stacks_quantized_triples_and_rejects_missing_expert():
    model = tiny_model()
    prefix = "model.layers.1.mlp"
    weights = {}
    for e in range(TINY["n_routed_experts"]):
        for kind in ("weight", "scales", "biases"):
            weights[f"{prefix}.experts.{e}.up_proj.{kind}"] = mx.full((2, 2), e)
    out = model.sanitize(dict(weights))
    assert sorted(out) == [
        f"{prefix}.switch_mlp.up_proj.{k}" for k in ("biases", "scales", "weight")
    ]
    assert out[f"{prefix}.switch_mlp.up_proj.weight"].shape == (
        TINY["n_routed_experts"],
        2,
        2,
    )
    assert out[f"{prefix}.switch_mlp.up_proj.scales"][5, 0, 0].item() == 5

    broken = dict(weights)
    del broken[f"{prefix}.experts.3.up_proj.weight"]
    with pytest.raises(ValueError, match=r"up_proj\.weight, first missing: 3"):
        model.sanitize(broken)

    # Missing expert 0 must not bypass the check (it used to gate on it).
    broken = dict(weights)
    del broken[f"{prefix}.experts.0.up_proj.weight"]
    with pytest.raises(ValueError, match=r"7 of 8 expert tensors .*first missing: 0"):
        model.sanitize(broken)

    # An expert index beyond n_routed_experts is a config/checkpoint mismatch.
    extra = dict(weights)
    extra[f"{prefix}.experts.8.up_proj.weight"] = mx.zeros((2, 2))
    with pytest.raises(ValueError, match="unexpected expert index: 8"):
        model.sanitize(extra)


def test_quant_predicate_keeps_non_experts_at_8_bits():
    """``mlx_lm.convert`` picks the model's predicate up: routed experts get
    the requested bits, every other quantizable module is pinned to 8-bit
    (fine-grained entries in the saved ``quantization`` config)."""
    from mlx_lm.utils import quantize_model

    model = tiny_model()
    model, qcfg = quantize_model(model, dict(TINY), group_size=64, bits=4)
    moe = model.layers[1].mlp
    assert moe.switch_mlp.gate_proj.bits == 4 and moe.switch_mlp.up_proj.bits == 4
    assert moe.shared_experts.gate_proj.bits == 8
    assert model.layers[0].mlp.gate_proj.bits == 8
    attn = model.layers[1].self_attn
    assert (attn.q_proj.bits, attn.k_proj.bits, attn.v_proj.bits, attn.o_proj.bits) == (
        8,
        8,
        8,
        8,
    )
    assert model.model.embed_tokens.bits == 8 and model.lm_head.bits == 8
    # Router stays a raw array: never quantized.
    assert isinstance(moe.gate.weight, mx.array)

    q = qcfg["quantization"]
    assert (q["group_size"], q["bits"]) == (64, 4)
    assert q["model.layers.1.self_attn.q_proj"] == {"group_size": 64, "bits": 8}
    assert q["lm_head"] == {"group_size": 64, "bits": 8}
    assert "model.layers.1.mlp.switch_mlp.gate_proj" not in q
    # The quantized tiny model still runs.
    assert model(mx.array([[1, 2, 3]])).shape == (1, 3, TINY["vocab_size"])


def test_cast_predicate_keeps_router_bias_in_float32():
    """``mlx_lm.convert``'s dtype pass casts every floating parameter the
    model's ``cast_predicate`` accepts; the F32 ``e_score_correction_bias``
    must survive it (rounding it to bf16 changes which experts fire)."""
    from mlx.utils import tree_map_with_path

    model = tiny_model()
    cast_predicate = model.cast_predicate

    def set_dtype(k, v):  # verbatim shape of mlx_lm.convert's pass
        if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
            return v.astype(mx.bfloat16)
        return v

    model.update(tree_map_with_path(set_dtype, model.parameters()))
    moe = model.layers[1].mlp
    assert moe.gate.e_score_correction_bias.dtype == mx.float32
    assert moe.gate.weight.dtype == mx.bfloat16
    assert model.layers[1].self_attn.q_proj.weight.dtype == mx.bfloat16
    assert model.model.embed_tokens.weight.dtype == mx.bfloat16


def test_tied_embeddings_variant():
    model = g9v3.Model(g9v3.ModelArgs.from_dict(dict(TINY, tie_word_embeddings=True)))
    assert "lm_head" not in dict(model.children())
    out = model.sanitize({"lm_head.weight": 1, "model.embed_tokens.weight": 2})
    assert sorted(out) == ["model.embed_tokens.weight"]
    assert model(mx.array([[1, 2, 3]])).shape == (1, 3, TINY["vocab_size"])


def test_alias_pins_parsers_and_moe_flags():
    aliases = json.loads(
        (Path(__file__).resolve().parents[1] / "vllm_mlx" / "aliases.json").read_text()
    )
    entry = aliases["g9v3-39a5b-4bit"]
    assert entry["hf_path"] == "rapid-mlx/G9v3-39A5B-MLX-4bit"
    # Chat template: ChatML + Qwen3-style <think> + MiniCPM-style
    # <function name=…><param name=…> XML tool calls.
    assert entry["tool_call_parser"] == "minicpm"
    assert entry["reasoning_parser"] == "qwen3"
    assert entry["is_moe"] is True
    assert entry["is_hybrid"] is False
    assert entry["supports_spec_decode"] is False
    assert entry["min_memory_gb"] == 32
