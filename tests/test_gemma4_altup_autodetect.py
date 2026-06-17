# SPDX-License-Identifier: Apache-2.0
"""Pin the gemma-4 altup-projection autodetect fix.

v0.7.26 release dogfood found that ``rapid-mlx serve gemma-4-e2b-4bit``
crashed at first decode step with::

    ValueError: [quantized_matmul] The weight matrix should be uint32
    but received bfloat16

Root cause: ``mlx-community/gemma-4-e2b-it-4bit`` (and ``-e4b-``) ship
with the ``per_layer_model_projection`` Linear stored as bare
bfloat16 on disk — mlx-community's QAT pipeline intentionally keeps a
handful of "altup" projection layers in full precision. Our prior
loader applied ``nn.quantize(model, ...)`` blanket, converted that
Linear to QuantizedLinear, then loaded the bf16 weight into it. At
inference time ``mx.quantized_matmul`` raised on the dtype mismatch
and the server emitted 0 tokens.

The fix is in ``load_gemma4_text``: before ``nn.quantize`` runs, scan
sanitized weights, mark any path whose ``.weight`` is bf16/fp16/fp32
with NO ``.scales`` companion (= "kept fp16 on purpose"), and skip
quantization on those paths.

These tests pin the two pure-logic pieces of that fix
(``_bare_fp_weight_paths`` and ``_path_matches_any_suffix``) so a
future refactor can't silently re-introduce the regression. The
end-to-end serve verification is covered by re-running the dogfood
on ``gemma-4-e2b-4bit`` post-merge.
"""

from __future__ import annotations

import mlx.core as mx

from vllm_mlx.models.gemma4_text import (
    _bare_fp_weight_paths,
    _path_matches_any_suffix,
)


def _fp_tensor(dtype=mx.bfloat16):
    return mx.zeros((2, 2), dtype=dtype)


def _q_tensor_uint32():
    return mx.zeros((2, 2), dtype=mx.uint32)


# ---------------------------------------------------------------------- #
# _bare_fp_weight_paths                                                  #
# ---------------------------------------------------------------------- #


def test_bare_fp_weight_paths_finds_altup_projection():
    """The exact mlx-community/gemma-4-e2b layout: per_layer_model_projection
    has a bare bfloat16 ``.weight`` and no ``.scales`` / ``.biases``.
    """
    sanitized = {
        # Properly quantized layer (uint32 weight + scales + biases)
        "language_model.model.layers.0.self_attn.q_proj.weight": _q_tensor_uint32(),
        "language_model.model.layers.0.self_attn.q_proj.scales": _fp_tensor(),
        "language_model.model.layers.0.self_attn.q_proj.biases": _fp_tensor(),
        # The "altup" projection that mlx-community kept as bf16
        "language_model.model.per_layer_model_projection.weight": _fp_tensor(),
    }
    skip = _bare_fp_weight_paths(sanitized)
    assert "language_model.model.per_layer_model_projection" in skip
    assert "language_model.model.layers.0.self_attn.q_proj" not in skip


def test_bare_fp_weight_paths_empty_when_all_quantized():
    """Variant where every Linear was quantized (gemma-4-12b/26b/31b).

    The 12b/26b/31b checkpoints don't even have ``per_layer_model_projection``,
    but the broader invariant is: when every weight has a uint32 ``.weight``
    + ``.scales``, the skip-set is empty and ``nn.quantize`` behaves exactly
    like it did before the fix. This is the regression-proof for larger
    variants.
    """
    sanitized = {
        "language_model.model.layers.0.self_attn.q_proj.weight": _q_tensor_uint32(),
        "language_model.model.layers.0.self_attn.q_proj.scales": _fp_tensor(),
        "language_model.model.layers.0.self_attn.q_proj.biases": _fp_tensor(),
        "language_model.model.layers.0.mlp.gate_proj.weight": _q_tensor_uint32(),
        "language_model.model.layers.0.mlp.gate_proj.scales": _fp_tensor(),
        "language_model.model.layers.0.mlp.gate_proj.biases": _fp_tensor(),
    }
    skip = _bare_fp_weight_paths(sanitized)
    assert skip == set(), (
        f"Expected empty skip-set when every weight is uint32 + scales — "
        f"got {skip}. Regression: this would over-skip quantization on "
        f"checkpoints where every layer was properly quantized."
    )


def test_bare_fp_weight_paths_treats_fp16_and_fp32_the_same():
    """Different storage dtypes for un-quantized layers all skip the same way."""
    sanitized = {
        "model.norm.weight": _fp_tensor(mx.float16),
        "model.embed_tokens.weight": _fp_tensor(mx.float32),
        "model.per_layer_model_projection.weight": _fp_tensor(mx.bfloat16),
    }
    skip = _bare_fp_weight_paths(sanitized)
    assert skip == {
        "model.norm",
        "model.embed_tokens",
        "model.per_layer_model_projection",
    }


def test_bare_fp_weight_paths_ignores_non_weight_tails():
    """Keys without a ``.weight`` suffix don't pollute the skip-set.

    Some modules ship ``.bias``, ``.running_mean``, etc. without ``.weight``
    — we only care about Linear-like layers whose quantization status is
    determined by the ``.weight``+``.scales`` shape.
    """
    sanitized = {
        "model.layer_norm.bias": _fp_tensor(),
        "model.attention.attn_mask": _fp_tensor(),
    }
    assert _bare_fp_weight_paths(sanitized) == set()


# ---------------------------------------------------------------------- #
# _path_matches_any_suffix                                               #
# ---------------------------------------------------------------------- #


def test_path_matches_when_full_suffix_aligns():
    """nn.quantize visits with ``language_model.model.X``; sanitized keys
    use the same prefix, so direct equality wins.
    """
    suffixes = {"language_model.model.per_layer_model_projection"}
    assert _path_matches_any_suffix(
        "language_model.model.per_layer_model_projection",
        suffixes,
    )


def test_path_does_not_match_sibling_module():
    """Don't over-match: per-layer ``self_attn.k_proj`` must not collide
    with ``per_layer_model_projection`` just because both end in
    ``_projection``-ish tokens.
    """
    suffixes = {"language_model.model.per_layer_model_projection"}
    assert not _path_matches_any_suffix(
        "language_model.model.layers.0.self_attn.k_proj",
        suffixes,
    )


def test_path_does_not_match_different_layer_index():
    """layers.0.q_proj and layers.5.q_proj are different modules even
    though both end in ``q_proj`` — the predicate must distinguish."""
    suffixes = {"language_model.model.layers.0.self_attn.q_proj"}
    assert not _path_matches_any_suffix(
        "language_model.model.layers.5.self_attn.q_proj",
        suffixes,
    )


def test_path_matches_empty_suffixes_is_false():
    """Fast-path: an empty skip-set should always return False, so
    variants with no bare-fp layers (e.g. all of 12b/26b/31b) pay
    zero per-module cost in the predicate."""
    assert not _path_matches_any_suffix("anything.anywhere", set())
