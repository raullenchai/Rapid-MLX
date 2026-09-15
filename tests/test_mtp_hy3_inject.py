# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the HY3 (Tencent Hunyuan 3) native MTP inject module.

Mirrors ``tests/test_mtp_inject_and_install.py`` +
``tests/test_mtp_gemma4_assistant_inject.py`` (the qwen3_5 / gemma4 paths)
for ``model_type == "hy_v3"``. Everything runs against a tiny synthetic
HY3 model — end-to-end forward through the real 155 GB target is out of
scope for CI (covered by the operator smoke + the weekly Golden Path job).

Coverage
--------

1. **Detection** — ``detect_mtp_eligibility`` reads HY3's
   ``num_nextn_predict_layers`` (root + ``text_config``) and does NOT
   cross-contaminate the Qwen3.5 ``mtp_num_hidden_layers`` key.
2. **Head build** — :func:`build_hy3_mtp_module` produces the
   DeepSeek-V3-shaped param tree (``enorm`` / ``hnorm`` / ``eh_proj`` /
   a single MoE ``DecoderLayer`` / ``norm``).
3. **Wiring probe** — ``inject_hy3_mtp_support(..., allow_random_init=True)``
   attaches the four MTP contract surfaces, and ``validate_hy3_mtp_support``
   returns True; a forward through them produces the right shapes.
4. **Sidecar refusal (fail-closed default)** — no sidecar + no
   ``allow_random_init`` would resolve the default HF repo; with a bogus
   sidecar path it returns False and leaves the model unmodified.
5. **Missing-tensor refusal** — a sidecar that omits a required tensor
   is rejected (no partial-random head).
6. **Architecture guard** — a non-``hy_v3`` model (lacking ``num_nextn``)
   builds no head.
7. **Dispatcher routing** — ``model_type == "hy_v3"`` routes to
   ``inject_hy3_mtp_support`` / ``validate_hy3_mtp_support`` in the
   dispatch tables.
8. **Default sidecar** — the module exposes the published-repo default so
   a bare ``--speculative-config '{"method":"mtp"}'`` boot resolves it.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _tiny_hy3_args(**overrides):
    """Minimal ``hy_v3.ModelArgs`` for a fake HY3 target."""
    from rapid_mlx.models.hy_v3 import ModelArgs

    defaults = dict(
        model_type="hy_v3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        expert_hidden_dim=32,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0},
        num_nextn_predict_layers=1,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def _build_tiny_hy3_model(**overrides):
    from rapid_mlx.models.hy_v3 import Model

    return Model(_tiny_hy3_args(**overrides))


def _resolve_inner(model):
    from rapid_mlx.spec_decode.mtp.hy3_inject import _resolve_inner_model

    return _resolve_inner_model(model)


# ---------------------------------------------------------------------------
# 1. Detection — num_nextn_predict_layers vs mtp_num_hidden_layers
# ---------------------------------------------------------------------------


def test_detect_hy3_num_nextn_chain():
    """HY3 with ``num_nextn_predict_layers == 1`` → CHAIN."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_hy3_num_nextn_under_text_config():
    """HY3 nextn count nested under ``text_config`` still resolves CHAIN."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "text_config": {"num_nextn_predict_layers": 1}}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_hy3_zero_nextn_rejected():
    """HY3 with ``num_nextn_predict_layers == 0`` (stripped) → NONE."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "num_nextn_predict_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_hy3_mtp_key_fallback():
    """HY3 also accepts the Qwen-style ``mtp_num_hidden_layers`` fallback."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_qwen35_does_not_read_num_nextn():
    """Cross-contamination guard: Qwen3.5 must NOT pick up HY3's
    ``num_nextn_predict_layers`` key — its head lives under
    ``mtp_num_hidden_layers`` only."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "qwen3_5", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_deepseek_v3_num_nextn_still_off_allowlist():
    """``deepseek_v3`` uses the same nextn convention but is NOT on the
    MTP allowlist — must stay NONE (no accidental promotion)."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "deepseek_v3", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


# ---------------------------------------------------------------------------
# 2. Head build — DeepSeek-V3-shaped param tree
# ---------------------------------------------------------------------------


def test_build_hy3_mtp_module_param_tree():
    """The head exposes ``enorm`` / ``hnorm`` / ``eh_proj`` / ``norm`` plus
    a single MoE ``DecoderLayer`` whose param names match a quantized
    backbone MoE layer (so the sidecar tree lines up 1:1)."""
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module

    args = _tiny_hy3_args()
    mtp = build_hy3_mtp_module(args, 1)
    keys = {k for k, _ in tree_flatten(mtp.parameters())}
    for want in (
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "norm.weight",
        "layers.0.mlp.router.gate.weight",
        "layers.0.mlp.router.expert_bias",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.shared_mlp.gate_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.q_proj.weight",
    ):
        assert want in keys, f"MTP head missing expected tensor {want!r}"


def test_build_hy3_mtp_module_rejects_multi_layer():
    """HY3 ships exactly one next-n layer; >1 must refuse loudly."""
    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module

    args = _tiny_hy3_args()
    with pytest.raises(ValueError):
        build_hy3_mtp_module(args, 2)
    with pytest.raises(ValueError):
        build_hy3_mtp_module(args, 0)


# ---------------------------------------------------------------------------
# 3. Wiring probe — four surfaces attach under random init
# ---------------------------------------------------------------------------


def test_inject_attaches_four_surfaces_random_init():
    """``allow_random_init=True`` attaches ``mtp`` / ``mtp_forward`` /
    ``make_mtp_cache`` / ``__call__(return_hidden, n_confirmed)`` and a
    forward through them produces the right shapes."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, allow_random_init=True) is True
    assert validate_hy3_mtp_support(model) is True

    ids = mx.array([[1, 2, 3, 4]])
    out, hidden = model(ids, return_hidden=True)
    assert out.shape == (1, 4, 128)
    assert hidden.shape == (1, 4, 64)

    mtp_cache = model.make_mtp_cache()
    mtp_logits = model.mtp_forward(hidden, ids, mtp_cache)
    mx.eval(mtp_logits)
    assert mtp_logits.shape == (1, 4, 128)
    assert bool(mx.all(mx.isfinite(mtp_logits)).item())

    # Prove the head consumes BOTH next_token_ids AND the prev hidden state via
    # eh_proj(concat([enorm(embed(ids)), hnorm(hidden)])) — codex R5 BLOCKING
    # #2. Vary ONLY ids (hidden fixed) then ONLY hidden (ids fixed); each
    # perturbation must move the logits. A prior single-forward "both differ"
    # check would pass even if one input were ignored, so perturb them
    # independently with a fresh cache each time.
    ids_b = mx.array([[5, 6, 7, 8]])
    logits_ids_perturbed = model.mtp_forward(hidden, ids_b, model.make_mtp_cache())
    mx.eval(logits_ids_perturbed)
    assert not bool(mx.allclose(logits_ids_perturbed, mtp_logits).item()), (
        "changing next_token_ids did not change MTP logits — ids input ignored"
    )

    hidden_b = hidden + 1.0
    logits_hidden_perturbed = model.mtp_forward(hidden_b, ids, model.make_mtp_cache())
    mx.eval(logits_hidden_perturbed)
    assert not bool(mx.allclose(logits_hidden_perturbed, mtp_logits).item()), (
        "changing the hidden state did not change MTP logits — hidden input ignored"
    )


def test_inject_forward_warm_cache_offset():
    """The injected __call__ must pass a single KVCache (cache[0]) — not the
    list — to create_attention_mask so the offset-aware make_mask path runs.

    Regression for a codex R6 false positive that proposed passing the whole
    cache list: mlx-lm's create_attention_mask gates on
    ``hasattr(cache, "make_mask")``, which a list lacks, so the list form would
    silently drop the warm-cache offset. Here we prefill a real KVCache with a
    prompt, then continue with more tokens, and require the warm continuation to
    match the equivalent slice of a single full-length forward (offset correct).
    """
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.spec_decode.mtp.hy3_inject import inject_hy3_mtp_support

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, allow_random_init=True) is True

    full_ids = mx.array([[1, 2, 3, 4, 5, 6]])
    # (a) one full-length forward, no cache.
    full_out, _ = model(full_ids, return_hidden=True)
    mx.eval(full_out)

    # (b) prefill first 4 tokens into a warm cache, then continue with 2 more.
    cache = [KVCache() for _ in range(len(model.model.layers))]
    _ = model(full_ids[:, :4], cache=cache, return_hidden=True)
    cont_out, _ = model(full_ids[:, 4:], cache=cache, return_hidden=True)
    mx.eval(cont_out)

    # The warm continuation's logits for positions 4..5 must match the full
    # forward's positions 4..5 — only true if the attention offset is applied
    # (i.e. cache[0].make_mask ran). A dropped offset would misalign attention.
    assert cont_out.shape == (1, 2, 128)
    assert bool(mx.allclose(cont_out, full_out[:, 4:, :], atol=1e-4).item()), (
        "warm-cache continuation diverged from full forward — attention offset "
        "was not applied (mask must receive cache[0], not the list)"
    )


def test_validate_false_before_inject():
    """A vanilla HY3 model has none of the MTP surfaces."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import validate_hy3_mtp_support

    model = _build_tiny_hy3_model()
    assert validate_hy3_mtp_support(model) is False


# ---------------------------------------------------------------------------
# 4. Sidecar refusal / missing-tensor refusal
# ---------------------------------------------------------------------------


def test_inject_refuses_unresolvable_sidecar():
    """A sidecar path that resolves to nothing → False, model unmodified."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model()
    ok = inject_hy3_mtp_support(model, mtp_sidecar="/nonexistent/path/nowhere")
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_refuses_sidecar_missing_tensor(tmp_path):
    """A sidecar missing a required tensor is rejected (no partial-random
    head), mirroring the qwen3_5 coverage check."""
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    # Build the (unquantized) param tree, then drop one tensor.
    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = dict(tree_flatten(template.parameters()))
    weights.pop("eh_proj.weight")  # required — must trip the coverage gate
    for k in list(weights):
        mx.eval(weights[k])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    # Base is unquantized here, so the injected head is unquantized too;
    # the coverage gate compares against that same tree.
    ok = inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir))
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_loads_synthetic_full_sidecar(tmp_path):
    """A complete sidecar (all required tensors) round-trips through
    inject → validate → True."""
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = dict(tree_flatten(template.parameters()))
    for k in list(weights):
        mx.eval(weights[k])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is True
    assert validate_hy3_mtp_support(model) is True


def test_inject_accepts_bf16_sidecar_against_fp32_template(tmp_path):
    """A real sidecar carries bf16 unquantized tensors (RMSNorm weights,
    biases) while the freshly-initialised template is fp32. The shape/dtype
    guard must compare the dtype *category* (float vs int/packed), NOT the
    exact float width, or it rejects a perfectly valid bf16 sidecar.

    Regression for the codex-BLOCKING finding: the earlier synthetic test
    (``test_inject_loads_synthetic_full_sidecar``) missed this because both
    the sidecar and the template originated from the same fp32 initialiser.
    Here we force the sidecar to bf16 so the two sides differ in float width.
    """
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = {}
    for k, v in tree_flatten(template.parameters()):
        vb = v.astype(mx.bfloat16)  # sidecar ships bf16, template is fp32
        mx.eval(vb)
        weights[k] = vb
    assert any(v.dtype == mx.bfloat16 for v in weights.values())
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is True
    assert validate_hy3_mtp_support(model) is True


def test_inject_refuses_integer_packed_wrong_kind(tmp_path):
    """The guard must still reject a genuine quant mismatch: a tensor that
    arrives as a packed integer (uint32) where the unquantized template
    expects a floating tensor is a float-vs-int category flip and is refused
    (this is what an 8-bit-packed sidecar landing on an unquantized/4-bit
    head looks like)."""
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = dict(tree_flatten(template.parameters()))
    for k in list(weights):
        mx.eval(weights[k])
    # Flip one float tensor to a packed-integer (uint32) while KEEPING its exact
    # expected shape, so ONLY the dtype-category guard can reject it — if shape
    # also differed, the shape check would pass the test even with the dtype
    # guard removed (codex R5 BLOCKING #3). This isolates the dtype-category
    # branch: a packed sidecar landing on an unquantized/differently-quantized
    # head shows up as int-where-float-expected.
    expected_shape = weights["eh_proj.weight"].shape
    weights["eh_proj.weight"] = mx.zeros(expected_shape, dtype=mx.uint32)
    mx.eval(weights["eh_proj.weight"])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_refuses_same_shape_wrong_integer_dtype(tmp_path):
    """Tighter dtype guard (codex R8 BLOCKING #1): the check requires EXACT
    dtype equality unless BOTH sides are floating.

    This targets the branch the tightening actually ADDED. The coarse
    float-vs-int-category predecessor already rejected a non-float tensor in a
    float slot, so a bool-where-float substitution proves nothing (codex R10
    BLOCKING: that test passed even against the loose guard). The distinguishing
    case is *two different non-float dtypes*: a quantized head expects a packed
    ``uint32`` parameter, and a same-shape signed ``int32`` sidecar tensor —
    which the coarse guard waved through (both "int") — must now be refused."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        _detect_base_quantization,
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    bits, gs = 4, 32

    def _q_pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("mlp.router.gate"):
            return {"group_size": gs, "bits": 8}
        return True

    # Quantize the base so inject detects a quant spec and EXPECTS a packed
    # uint32 head — the only way a non-float parameter is the expected dtype.
    model = _build_tiny_hy3_model()
    nn.quantize(model.model, group_size=gs, bits=bits, class_predicate=_q_pred)
    assert _detect_base_quantization(_resolve_inner(model)) == {
        "bits": bits,
        "group_size": gs,
    }

    # Build + quantize a head the same way inject does, then corrupt ONE packed
    # uint32 tensor into a same-shape int32 (a *different* non-float dtype).
    args = _tiny_hy3_args()
    head = build_hy3_mtp_module(args, 1)
    nn.quantize(head, group_size=gs, bits=bits, class_predicate=_q_pred)
    packed = dict(tree_flatten(head.parameters()))
    for k in list(packed):
        mx.eval(packed[k])
    uint32_key = next(k for k, v in packed.items() if v.dtype == mx.uint32)
    packed[uint32_key] = mx.zeros(packed[uint32_key].shape, dtype=mx.int32)
    mx.eval(packed[uint32_key])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), packed)

    # int32-where-uint32-expected: same shape, both non-float, dtype differs.
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_quantized_base_packed_sidecar_round_trip(tmp_path):
    """Exercise the PRODUCTION packed path (codex R5 NIT): quantize the base
    model so inject detects a quant spec and quantizes the head, build a sidecar
    whose tensors are the head's quantized (packed uint32) params, then inject +
    run mtp_forward. Every other sidecar-loading test uses an unquantized tiny
    model, so this is the only coverage of the nn.quantize layout + packed load.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        _detect_base_quantization,
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    bits, gs = 4, 32  # smallest tiny-model quantizable dim is 32 (experts)

    # Quantize the BASE model so inject's _detect_base_quantization returns a
    # spec and the head is quantized to match.
    model = _build_tiny_hy3_model()

    def _q_pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("mlp.router.gate"):
            return {"group_size": gs, "bits": 8}
        return True

    nn.quantize(model.model, group_size=gs, bits=bits, class_predicate=_q_pred)
    detected = _detect_base_quantization(_resolve_inner(model))
    assert detected == {"bits": bits, "group_size": gs}

    # Build + quantize a head the SAME way inject does, then dump its (packed)
    # params as the sidecar so load_weights sees real uint32 packed tensors.
    args = _tiny_hy3_args()
    head = build_hy3_mtp_module(args, 1)
    nn.quantize(head, group_size=gs, bits=bits, class_predicate=_q_pred)
    packed = dict(tree_flatten(head.parameters()))
    for k in list(packed):
        mx.eval(packed[k])
    assert any(v.dtype == mx.uint32 for v in packed.values()), (
        "expected packed uint32 tensors in a quantized head"
    )
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), packed)

    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is True
    assert validate_hy3_mtp_support(model) is True

    ids = mx.array([[1, 2, 3, 4]])
    _, hidden = model(ids, return_hidden=True)
    logits = model.mtp_forward(hidden, ids, model.make_mtp_cache())
    mx.eval(logits)
    assert logits.shape == (1, 4, 128)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_inject_refuses_multi_nextn_layer_config():
    """A HY3 config advertising num_nextn_predict_layers=2 must fail-closed to
    False (HY3's head builder only supports exactly 1 layer), NOT crash boot
    with a builder ValueError (codex R4 BLOCKING #1)."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model(num_nextn_predict_layers=2)
    # Must return False cleanly (no exception) even with allow_random_init.
    assert inject_hy3_mtp_support(model, allow_random_init=True) is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_returns_false_on_corrupt_sidecar(tmp_path):
    """A truncated / malformed sidecar file must be caught and turned into a
    False return (this function's documented contract), NOT propagate an
    exception that aborts server boot (codex R4 BLOCKING #2)."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    # Write bytes that are NOT a valid safetensors file.
    (side_dir / "model-mtp.safetensors").write_bytes(b"not a real safetensors file")

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is False
    assert validate_hy3_mtp_support(model) is False


# ---------------------------------------------------------------------------
# 5. Architecture guard
# ---------------------------------------------------------------------------


def test_inject_refuses_model_without_num_nextn():
    """A HY3-shaped model whose config advertises no next-n layer builds
    no head (num_nextn_predict_layers=0)."""
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model(num_nextn_predict_layers=0)
    ok = inject_hy3_mtp_support(model, allow_random_init=True)
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


# ---------------------------------------------------------------------------
# 6. Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatch_tables_route_hy_v3():
    """``hy_v3`` is registered in both dispatch tables → the family
    inject / validate entry points."""
    from rapid_mlx.spec_decode.mtp.dispatch import (
        _MTP_INJECT_DISPATCH,
        _MTP_VALIDATE_DISPATCH,
    )

    assert _MTP_INJECT_DISPATCH["hy_v3"] == (
        "rapid_mlx.spec_decode.mtp.hy3_inject",
        "inject_hy3_mtp_support",
    )
    assert _MTP_VALIDATE_DISPATCH["hy_v3"] == (
        "rapid_mlx.spec_decode.mtp.hy3_inject",
        "validate_hy3_mtp_support",
    )


def test_dispatch_inject_routes_and_attaches():
    """End-to-end through the dispatcher: ``dispatch_mtp_inject(model,
    'hy_v3', allow_random_init=True)`` attaches the surfaces, and
    ``dispatch_mtp_validate`` confirms them."""
    from rapid_mlx.spec_decode.mtp import (
        dispatch_mtp_inject,
        dispatch_mtp_validate,
    )

    model = _build_tiny_hy3_model()
    assert dispatch_mtp_inject(model, "hy_v3", allow_random_init=True) is True
    assert dispatch_mtp_validate(model, "hy_v3") is True


# ---------------------------------------------------------------------------
# 7. Default sidecar repo
# ---------------------------------------------------------------------------


def test_default_sidecar_repo_constant():
    """The module exposes the published-sidecar default so a bare
    ``--speculative-config '{"method":"mtp"}'`` boot auto-resolves it."""
    from rapid_mlx.spec_decode.mtp import hy3_inject

    assert hy3_inject.DEFAULT_HY3_MTP_SIDECAR == "mlx-community/Hy3-preview-MTP-4bit"


# ---------------------------------------------------------------------------
# 8. Codex-hardening regressions (PR #1094 review)
# ---------------------------------------------------------------------------


def test_inject_resolver_reads_mtp_num_hidden_layers_fallback():
    """detect + inject must agree on the layer-count key set. A hy_v3 config
    that only carries ``mtp_num_hidden_layers`` (the hand-converted alias)
    must be resolvable by the inject resolver too — otherwise detection deems
    it eligible but inject refuses (codex BLOCKING #2)."""
    from rapid_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility
    from rapid_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    # Detection accepts the fallback key.
    cfg = {"model_type": "hy_v3", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(cfg) is MTPEligibility.CHAIN

    # Inject must accept a model whose args carry ONLY the fallback key.
    model = _build_tiny_hy3_model(num_nextn_predict_layers=0)
    model.args.mtp_num_hidden_layers = 1  # hand-converted alias on the args
    assert inject_hy3_mtp_support(model, allow_random_init=True) is True
    assert validate_hy3_mtp_support(model) is True


def test_default_sidecar_pinned_revision_constant():
    """The default sidecar carries an immutable pinned revision (codex
    BLOCKING #1) — never resolves a mutable HEAD in production."""
    from rapid_mlx.spec_decode.mtp import hy3_inject

    rev = hy3_inject.DEFAULT_HY3_MTP_SIDECAR_REVISION
    assert isinstance(rev, str) and len(rev) == 40  # full commit SHA
    assert all(c in "0123456789abcdef" for c in rev)


def test_default_sidecar_refused_on_quant_mismatch(monkeypatch):
    """The default (4-bit gs64) sidecar is refused when the base is a
    different quant (codex BLOCKING #3) — it would load shape-incompatible
    tensors. An explicit sidecar bypasses the gate."""
    from rapid_mlx.spec_decode.mtp import hy3_inject

    model = _build_tiny_hy3_model()

    # Pretend the base is 8-bit (not the default sidecar's 4-bit).
    monkeypatch.setattr(
        hy3_inject,
        "_detect_base_quantization",
        lambda inner: {"bits": 8, "group_size": 64},
    )
    # No explicit sidecar → default path → quant gate must refuse.
    ok = hy3_inject.inject_hy3_mtp_support(model)
    assert ok is False
    assert hy3_inject.validate_hy3_mtp_support(model) is False
