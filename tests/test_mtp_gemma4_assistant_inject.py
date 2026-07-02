# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Gemma 4 assistant MTP inject module.

Covers the wiring contract and the sidecar loading contract against a
tiny fake Google-shaped checkpoint. End-to-end forward through a real
target model is out of scope for CI (needs the 12B target + 806 MB
assistant weights); the operator smoke script covers that layer.

Coverage
--------

1. **Config parse + module build** — a Google-shaped ``config.json``
   parses through :func:`_build_assistant_model_args` and produces a
   4-layer AssistantModel whose weight tree names match Google's
   safetensors layout.
2. **Wiring probe** — with ``allow_random_init=True``, the four MTP
   contract surfaces attach onto a synthetic Gemma 4 target text
   model.
3. **Weight-loading smoke** — a hand-built tiny safetensors that
   matches the assistant's parameter tree round-trips through
   save → :func:`inject_mtp_support` → :func:`validate_mtp_support`
   → True.
4. **Sidecar refusal (fail-closed default)** — no sidecar + no
   ``allow_random_init`` → False, model unmodified.
5. **Architecture-guard** — a config whose ``model_type`` is NOT one
   of the known assistant strings is REFUSED, not silently loaded.
6. **Dispatcher routing** — ``gemma4`` / ``gemma4_unified`` /
   ``gemma4_text`` route to this module; ``qwen3_5`` + fake
   ``gemma4-assistant`` sidecar → still routes to qwen3_5 (dispatcher
   is model_type-based, no fingerprint sniffing).
"""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mtp_state():
    """Reset shared MTP module state between tests.

    Same recipe as ``tests/test_mtp_spec_decode.py::_reset_mtp_module_state``.
    """
    import sys

    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

    _unpatch_for_tests()
    reset_global_counter_for_tests()

    # Rebind the mlx-lm generation_stream to this thread's default so
    # tests running after an mlx-step executor thread don't cross-thread
    # crash (matches the ``mlx-mx-new-stream-crashes-across-threads``
    # gotcha).
    import mlx_lm.generate  # noqa: F401

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )


def _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4):
    """Return a Google-shaped assistant config.json dict, sized small.

    Mirrors the top-level + ``text_config`` layout Google ships for
    ``google/gemma-4-12B-it-assistant``. Dims are tiny so tests stay
    under a second.
    """
    return {
        "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
        "model_type": "gemma4_unified_assistant",
        "backbone_hidden_size": backbone,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "intermediate_size": hidden * 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": n_layers,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 64,
            "layer_types": ["sliding_attention"] * (n_layers - 1) + ["full_attention"],
            "vocab_size": 128,
            "vocab_size_per_layer_input": 0,
            "rms_norm_eps": 1e-6,
            "attention_k_eq_v": True,
            "tie_word_embeddings": True,
            "final_logit_softcapping": None,
            "use_double_wide_mlp": False,
            "enable_moe_block": False,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            },
        },
    }


def _tiny_gemma4_target_args(hidden=128):
    """Minimal ``gemma4_text.ModelArgs`` for a fake target."""
    from mlx_lm.models.gemma4_text import ModelArgs

    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=64,
        sliding_window_pattern=6,
        max_position_embeddings=128,
        final_logit_softcapping=None,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
        # Target's layer_types — last 4 layers match assistant order
        # (3 sliding + 1 full) so cross-KV indices align.
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    return args


def _build_tiny_gemma4_target_model():
    """Instantiate a tiny Gemma 4 target model."""
    from mlx_lm.models.gemma4_text import Model

    return Model(_tiny_gemma4_target_args())


# ---------------------------------------------------------------------------
# 1. Config parse + module build
# ---------------------------------------------------------------------------


def test_build_assistant_model_args_parses_google_shape():
    """Google-shaped config parses into a usable ``gemma4_text.ModelArgs``.

    Locks the field-mapping contract in :func:`_build_assistant_model_args`
    against a config that looks structurally like Google's actual
    ``config.json`` (verified against
    ``google/gemma-4-12B-it-assistant`` at PR-3 authoring time).
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None, "Google-shaped config should build args"
    assert args.hidden_size == 64
    assert args.num_hidden_layers == 4
    assert args.intermediate_size == 128
    assert args.num_kv_shared_layers == 4
    assert args.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert args.tie_word_embeddings is True
    assert args.head_dim == 16
    assert args.global_head_dim == 32
    assert getattr(args, "backbone_hidden_size", None) == 128


def test_build_assistant_model_args_rejects_mismatched_backbone_hidden():
    """When assistant.backbone_hidden_size ≠ target.hidden_size, refuse.

    Cross-projection weights (pre_projection ``[hidden, 2 × backbone]``)
    would not fit; refusing loudly beats a silent shape error at
    forward time.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128)
    # Target's hidden_size mismatches assistant's backbone_hidden_size.
    args = _build_assistant_model_args(cfg, target_backbone_hidden=256)
    assert args is None, "Mismatched backbone/target dims must refuse the build"


def test_build_assistant_model_matches_google_weight_tree():
    """The AssistantModel's parameter tree matches Google's safetensors keys.

    The Google ``gemma-4-12B-it-assistant`` safetensors ships:

        model.embed_tokens.weight
        model.layers.{i}.self_attn.q_proj.weight
        model.layers.{i}.self_attn.q_norm.weight
        model.layers.{i}.self_attn.o_proj.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
        model.layers.{i}.layer_scalar
        model.norm.weight
        pre_projection.weight
        post_projection.weight

    Locks that shape here so a future refactor that renames a param
    (or forgets to remove per-layer-input gates) fails this test.
    """
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    model = _build_assistant_model(args, backbone_hidden_size=128)
    keys = {k for k, _ in tree_flatten(model.parameters())}

    # Top-level cross-projection layers.
    assert "pre_projection.weight" in keys
    assert "post_projection.weight" in keys

    # Backbone shell.
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys

    # Every layer has Q-only self_attn (no k_proj / v_proj — shared K/V
    # from target), Gemma sandwich norms, and layer_scalar.
    per_layer_required = (
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "layer_scalar",
    )
    for i in range(4):
        for suffix in per_layer_required:
            assert f"model.layers.{i}.{suffix}" in keys, (
                f"missing key model.layers.{i}.{suffix}"
            )

    # NO k_proj / v_proj / k_norm / v_norm on any layer (shared K/V).
    for i in range(4):
        for absent in (
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.v_norm.weight",
        ):
            assert f"model.layers.{i}.{absent}" not in keys, (
                f"unexpected key model.layers.{i}.{absent} — assistant "
                "layers must be K/V-shared (no own K/V weights)."
            )


# ---------------------------------------------------------------------------
# 2. Wiring probe — four surfaces attach under allow_random_init=True
# ---------------------------------------------------------------------------


def test_inject_attaches_four_surfaces_under_random_init():
    """allow_random_init=True must attach the four MTP contract surfaces.

    Extended ``__call__`` gets ``return_hidden`` + ``n_confirmed``,
    ``.mtp`` + ``.mtp_forward`` + ``.make_mtp_cache`` land on the
    inner text model.
    """
    import inspect

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    result = inject_mtp_support(model, allow_random_init=True)
    assert result is True, "random-init inject should attach surfaces"

    assert getattr(model, "mtp", None) is not None
    assert callable(getattr(model, "mtp_forward", None))
    assert callable(getattr(model, "make_mtp_cache", None))
    sig = inspect.signature(type(model).__call__)
    assert "return_hidden" in sig.parameters
    assert "n_confirmed" in sig.parameters

    # The validator returns True — random-init still satisfies the
    # SIGNATURE contract, and unlike the PR #989 scaffold path, the
    # AssistantModel does NOT set an ``_mtp_is_scaffold`` marker.
    # Production callers get a truthful "yes this is wired" signal.
    assert validate_mtp_support(model) is True


# ---------------------------------------------------------------------------
# 3. Weight-loading smoke — real safetensors round-trip
# ---------------------------------------------------------------------------


def test_inject_loads_synthetic_google_shaped_sidecar(tmp_path):
    """A synthetic tiny sidecar matching the assistant's parameter tree loads.

    Steps:
      1. Build the AssistantModel from a Google-shaped config with tiny
         dims so weight materialization stays under a second.
      2. Save its randomly-initialized parameters to a synthetic
         safetensors + config.json in a temp dir.
      3. Point ``inject_mtp_support(target, mtp_sidecar=tmp_path)`` at
         the dir and verify the four surfaces attach.

    This exercises the sidecar resolver + config parser + weight loader
    + coverage check ALL together. Doesn't validate correctness of the
    forward pass — that's the operator smoke script's job.
    """
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
        validate_mtp_support,
    )

    # Match the target's hidden_size — tiny target has hidden=128.
    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    model_template = _build_assistant_model(args, backbone_hidden_size=128)
    mx.eval(model_template.parameters())
    flat = dict(tree_flatten(model_template.parameters()))
    assert flat, "template parameters should be non-empty"

    # Write the sidecar dir.
    sidecar_dir = tmp_path / "gemma-4-12B-it-assistant"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(sidecar_dir / "model.safetensors"), flat)

    # Now inject onto a fresh target.
    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    ok = inject_mtp_support(target, mtp_sidecar=str(sidecar_dir))
    assert ok is True, "sidecar-driven inject should succeed"
    assert validate_mtp_support(target) is True

    # Loaded weights match template byte-for-byte.
    loaded = dict(tree_flatten(target.mtp.parameters()))
    assert set(loaded.keys()) == set(flat.keys())
    for k in flat:
        diff = mx.sum(loaded[k] != flat[k]).item()
        assert diff == 0, f"{k}: differs by {diff} entries"


def test_inject_refuses_sidecar_missing_tensor(tmp_path):
    """A sidecar missing a required tensor is REFUSED.

    Matches the qwen3_5 side coverage check — prevents silent
    partial-random-init shipping.
    """
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    tpl = _build_assistant_model(args, backbone_hidden_size=128)
    mx.eval(tpl.parameters())
    flat = dict(tree_flatten(tpl.parameters()))

    # Drop a required tensor.
    flat.pop("pre_projection.weight")

    sidecar_dir = tmp_path / "broken-assistant"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(sidecar_dir / "model.safetensors"), flat)

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")
    original_class = type(target)
    ok = inject_mtp_support(target, mtp_sidecar=str(sidecar_dir))
    assert ok is False, "missing-tensor sidecar must be refused"
    # Model must not have been patched.
    assert type(target) is original_class
    assert getattr(target, "mtp", None) is None


# ---------------------------------------------------------------------------
# 4. Sidecar refusal — fail-closed default
# ---------------------------------------------------------------------------


def test_inject_refuses_no_sidecar_by_default():
    """Default ``allow_random_init=False`` + no sidecar → False + unmodified."""
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    original_class = type(target)
    assert inject_mtp_support(target) is False
    assert getattr(target, "mtp", None) is None
    assert not callable(getattr(target, "mtp_forward", None))
    assert not callable(getattr(target, "make_mtp_cache", None))
    assert type(target) is original_class


# ---------------------------------------------------------------------------
# 5. Architecture guard — non-assistant model_type is refused
# ---------------------------------------------------------------------------


def test_inject_refuses_non_assistant_model_type(tmp_path):
    """A sidecar dir whose config.json declares a non-assistant model_type
    is REFUSED — prevents accidentally loading a base Gemma 4 checkpoint
    or an unrelated MTP head.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    # Corrupt the model_type — pretend this is a base Gemma 4 dump.
    cfg["model_type"] = "gemma4_unified"

    sidecar_dir = tmp_path / "wrong-arch-sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "config.json").write_text(json.dumps(cfg))
    # Write minimal safetensors so we know the refusal fires BEFORE
    # weight loading (not because there's nothing to load).
    mx.save_safetensors(str(sidecar_dir / "model.safetensors"), {})

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    result = inject_mtp_support(target, mtp_sidecar=str(sidecar_dir))
    assert result is False, "non-assistant model_type must be refused"
    assert getattr(target, "mtp", None) is None


def test_build_assistant_model_args_rejects_layer_types_length_mismatch():
    """When ``layer_types`` length doesn't match ``num_hidden_layers``,
    fail closed at build time — a bad list would crash later inside
    ``DecoderLayer.Attention`` with an opaque IndexError.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    # Corrupt: 3 layer types for 4 layers.
    cfg["text_config"]["layer_types"] = ["sliding_attention"] * 3

    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is None, "schema mismatch on layer_types must refuse the build"


# ---------------------------------------------------------------------------
# 5b. mtp_cache safety
# ---------------------------------------------------------------------------


def test_make_mtp_cache_slots_are_generator_safe():
    """Lock the ``make_mtp_cache`` safety analysis (docstring on the
    method): empty ``KVCache`` slots must survive every generator
    walk without raising.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    assert inject_mtp_support(target, allow_random_init=True) is True

    caches = target.make_mtp_cache()
    assert isinstance(caches, list) and len(caches) == len(target.mtp.model.layers)

    for c in caches:
        # Walk 1: ``quantize_cache_fn`` — kv_bits=None short-circuits,
        # but with kv_bits set it calls ``to_quantized`` on each slot.
        # Confirm the empty-slot path doesn't raise.
        q = c.to_quantized(group_size=64, bits=4)
        assert q is not None
        assert q.offset == 0

        # Walk 2: ``_prefill`` scans ``[c.state for ... if hasattr(c, 'state')]``.
        # ``hasattr`` on an empty KVCache swallows the AttributeError
        # raised by the getter and returns False — so the empty slot
        # is skipped entirely (no ``mx.eval((None, None))`` corner case).
        assert not hasattr(c, "state"), (
            "empty KVCache should not advertise state via hasattr — the "
            "generator's ``if hasattr(c, 'state')`` skip depends on this."
        )

        # Walk 3: trim(1) on empty is a no-op returning 0 (source
        # inspection of KVCache.trim).
        assert c.trim(1) == 0
        assert c.offset == 0


# ---------------------------------------------------------------------------
# 6. Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_routes_gemma4_families_to_this_module():
    """``gemma4`` / ``gemma4_unified`` / ``gemma4_text`` / ``gemma4_unified_text``
    all route to ``gemma4_inject``.
    """
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch

    for mt in ("gemma4", "gemma4_unified", "gemma4_text", "gemma4_unified_text"):
        assert mt in _dispatch._MTP_INJECT_DISPATCH, (
            f"{mt!r} missing from inject dispatch"
        )
        module_path, _ = _dispatch._MTP_INJECT_DISPATCH[mt]
        assert module_path == "vllm_mlx.spec_decode.mtp.gemma4_inject"

        assert mt in _dispatch._MTP_VALIDATE_DISPATCH
        v_path, _ = _dispatch._MTP_VALIDATE_DISPATCH[mt]
        assert v_path == "vllm_mlx.spec_decode.mtp.gemma4_inject"


def test_dispatcher_still_routes_qwen3_5():
    """Qwen3.5 routing is unaffected — locks the shared table entries."""
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch

    for mt in ("qwen3_5", "qwen3_5_moe"):
        assert mt in _dispatch._MTP_INJECT_DISPATCH
        module_path, _ = _dispatch._MTP_INJECT_DISPATCH[mt]
        assert module_path == "vllm_mlx.spec_decode.mtp.qwen3_5_inject"


def test_dispatcher_returns_false_for_unknown_model_type():
    """Fail-closed on unknown model_type — no KeyError, no fallback."""
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject

    result = dispatch_mtp_inject(
        object(),
        model_type="not_a_real_model_type_xyzzy",
        allow_random_init=True,
    )
    assert result is False


def test_dispatcher_swallows_family_exceptions(monkeypatch):
    """The dispatcher's "never raises" contract must hold even if the
    family injector raises (loader bug, weight shape mismatch, etc.).
    Codex round-3 blocker: unwrapped ``fn(...)`` calls could propagate.
    """
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch
    from vllm_mlx.spec_decode.mtp import gemma4_inject

    def _raising_inject(model, mtp_sidecar=None, *, allow_random_init=False):
        raise RuntimeError("simulated loader crash inside gemma4_inject")

    monkeypatch.setattr(gemma4_inject, "inject_mtp_support", _raising_inject)

    result = _dispatch.dispatch_mtp_inject(
        object(),
        model_type="gemma4_unified",
        allow_random_init=True,
    )
    assert result is False, "dispatcher must convert family exceptions to False"


def test_dispatcher_validate_swallows_family_exceptions(monkeypatch):
    """Same as above but for ``dispatch_mtp_validate``."""
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch
    from vllm_mlx.spec_decode.mtp import gemma4_inject

    def _raising_validate(model):
        raise RuntimeError("simulated validator crash")

    monkeypatch.setattr(gemma4_inject, "validate_mtp_support", _raising_validate)

    result = _dispatch.dispatch_mtp_validate(object(), model_type="gemma4_unified")
    assert result is False


def test_gemma4_text_modelargs_carries_fields_this_module_reads():
    """Hard-fail (NOT skip) if mlx-lm's ``gemma4_text.ModelArgs`` no
    longer accepts a field this inject reads. Skip-happy tests hide
    upstream schema drift; this test is the canary.

    Codex round-3 nit: without a hard-fail here, a future mlx-lm
    version that renames (say) ``num_kv_shared_layers`` to
    ``kv_shared_layer_count`` would silently skip every wiring probe
    test, and reviewers would see all-green with zero coverage.
    """
    from mlx_lm.models.gemma4_text import ModelArgs

    # These are the exact fields the inject reads via
    # ``_build_assistant_model_args``. If any drops, hard-fail so a
    # follow-up can update the field mapping.
    required_fields = {
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "head_dim",
        "global_head_dim",
        "rms_norm_eps",
        "vocab_size",
        "num_key_value_heads",
        "num_global_key_value_heads",
        "num_kv_shared_layers",
        "hidden_size_per_layer_input",
        "rope_parameters",
        "sliding_window",
        "sliding_window_pattern",
        "max_position_embeddings",
        "attention_k_eq_v",
        "final_logit_softcapping",
        "use_double_wide_mlp",
        "enable_moe_block",
        "tie_word_embeddings",
        "layer_types",
    }
    dataclass_fields = set(ModelArgs.__dataclass_fields__.keys())
    missing = required_fields - dataclass_fields
    assert not missing, (
        f"mlx-lm gemma4_text.ModelArgs dropped fields the Gemma 4 inject "
        f"depends on: {sorted(missing)}. Update _build_assistant_model_args."
    )


def test_dispatcher_routes_gemma4_unified_to_gemma4_inject(monkeypatch):
    """The dispatcher forwards ``model`` + kwargs verbatim to gemma4_inject.

    Guards against silent argument drops (e.g. dropping ``mtp_sidecar``
    would let a production caller silently random-init a drafter).
    """
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch
    from vllm_mlx.spec_decode.mtp import gemma4_inject

    calls: list[dict] = []

    def _fake_inject(model, mtp_sidecar=None, *, allow_random_init=False):
        calls.append(
            {
                "model": model,
                "mtp_sidecar": mtp_sidecar,
                "allow_random_init": allow_random_init,
            }
        )
        return True

    monkeypatch.setattr(gemma4_inject, "inject_mtp_support", _fake_inject)

    sentinel_model = object()
    sentinel_sidecar = "google/gemma-4-12B-it-assistant"

    result = _dispatch.dispatch_mtp_inject(
        sentinel_model,
        model_type="gemma4_unified",
        mtp_sidecar=sentinel_sidecar,
        allow_random_init=False,
    )
    assert result is True
    assert len(calls) == 1
    assert calls[0]["model"] is sentinel_model
    assert calls[0]["mtp_sidecar"] == sentinel_sidecar
    assert calls[0]["allow_random_init"] is False


# ---------------------------------------------------------------------------
# 7. Outer-wrapper delegation
# ---------------------------------------------------------------------------


def test_inject_delegates_surfaces_to_outer_wrapper():
    """When called with an outer VLM wrapper, the THREE attribute surfaces
    (.mtp / .mtp_forward / .make_mtp_cache) delegate to the inner text model.

    The extended ``__call__(return_hidden, n_confirmed)`` signature is
    deliberately NOT delegated on the outer — matches the Qwen3.5
    contract (all callers unwrap outer → inner before invoking the
    extended signature).
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        inner = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    class _FakeOuterVLM:
        def __init__(self, lm):
            self.language_model = lm

    outer = _FakeOuterVLM(inner)

    result = inject_mtp_support(outer, allow_random_init=True)
    assert result is True

    assert getattr(outer, "mtp", None) is not None
    assert callable(getattr(outer, "mtp_forward", None))
    assert callable(getattr(outer, "make_mtp_cache", None))

    # make_mtp_cache returns a list from the inner scaffold — assert
    # it's a real list of cache instances.
    cache = outer.make_mtp_cache()
    assert isinstance(cache, list)
    assert len(cache) == inner.args.num_hidden_layers or len(cache) == len(
        inner.mtp.model.layers
    )


# ---------------------------------------------------------------------------
# 8. Codex round-6 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_shape_mismatched_tensor(tmp_path):
    """Codex round-6 blocking-fix locked in.

    A sidecar carrying all the EXPECTED keys but with an INCOMPATIBLE
    shape on one tensor (e.g. a checkpoint from a different assistant
    size sharing the layout names) must not propagate the load_weights
    exception up — inject_mtp_support has to fail closed and return
    ``False``, matching its documented contract.
    """
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    model_template = _build_assistant_model(args, backbone_hidden_size=128)
    mx.eval(model_template.parameters())
    flat = dict(tree_flatten(model_template.parameters()))

    # Corrupt ONE tensor to a totally wrong shape.
    bad_key = "model.embed_tokens.weight"
    assert bad_key in flat, "template should have embed_tokens.weight"
    original_shape = flat[bad_key].shape
    flat[bad_key] = mx.zeros((original_shape[0] + 7, original_shape[1] + 3))

    sidecar_dir = tmp_path / "shape-mismatch-assistant"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(sidecar_dir / "model.safetensors"), flat)

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    # Should return False (not raise) even though the loaded tensor's
    # shape is incompatible with the template.
    ok = inject_mtp_support(target, mtp_sidecar=str(sidecar_dir))
    assert ok is False, "shape-mismatched sidecar must fail closed"


def test_validate_refuses_when_outer_wrapper_missing_delegated_surface():
    """Codex round-6 blocking-fix locked in.

    If a caller manually replicates the inner-side wiring but skips the
    outer-wrapper delegation (which the generator would need for
    ``outer.mtp_forward(...)`` / ``outer.make_mtp_cache()``),
    ``validate_mtp_support`` must return False — a green validate that
    later AttributeErrors inside the generator is worse than a red
    validate up front.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        inner = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    class _FakeOuterVLM:
        def __init__(self, lm):
            self.language_model = lm

    outer = _FakeOuterVLM(inner)

    # Normal inject wires everything.
    assert inject_mtp_support(outer, allow_random_init=True) is True
    assert validate_mtp_support(outer) is True

    # Simulate a partially-rolled-back state: strip the outer-only
    # delegations while leaving the inner correctly patched.
    for attr in ("mtp", "mtp_forward", "make_mtp_cache"):
        if hasattr(outer, attr):
            delattr(outer, attr)

    # Now validate on the outer must refuse — even though the inner
    # itself still has all four surfaces.
    assert validate_mtp_support(outer) is False
    # And validate on the inner directly still succeeds — we only
    # tightened the outer-check, not the inner-check.
    assert validate_mtp_support(inner) is True


# ---------------------------------------------------------------------------
# 9. Codex round-7 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_vocab_size_mismatch(tmp_path):
    """Codex round-7 blocking-fix locked in.

    A sidecar with the correct ``backbone_hidden_size`` but a
    different ``vocab_size`` (i.e. paired to a different tokenizer)
    must fail closed. Otherwise the drafter's tied ``embed_tokens``
    would return logits over the wrong vocabulary and the caller
    would silently draft garbage tokens.
    """
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    # Build a sidecar with an intentionally WRONG vocab_size (assistant
    # says 999, but the tiny target we build below has vocab_size=128;
    # inject must refuse). Codex round-8 nit: comment previously said
    # "128 vs 128" which contradicted the code — now correctly states
    # the mismatch numbers.
    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    # Override vocab_size on the assistant sidecar to 999 (vs target's 128).
    cfg["text_config"]["vocab_size"] = 999  # tiny target uses vocab_size=128
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    model_template = _build_assistant_model(args, backbone_hidden_size=128)
    mx.eval(model_template.parameters())
    flat = dict(tree_flatten(model_template.parameters()))

    sidecar_dir = tmp_path / "vocab-mismatch-assistant"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(sidecar_dir / "model.safetensors"), flat)

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    # Should refuse before any weights load.
    ok = inject_mtp_support(target, mtp_sidecar=str(sidecar_dir))
    assert ok is False, "vocab-mismatched sidecar must fail closed"


def test_mtp_forward_rejects_batch_greater_than_one():
    """Codex round-7 blocking-fix locked in.

    ``mtp_forward`` fans out one shared ``_mtp_target_cache`` reference
    across the query — for B>1 that cache list can't be safely split
    per-request. Rejecting B>1 up front prevents cross-request K/V
    leakage until the follow-up multi-request path lands.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        target = _build_tiny_gemma4_target_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 ModelArgs schema mismatch: {exc}")

    assert inject_mtp_support(target, allow_random_init=True) is True

    # Prime _mtp_target_cache with a set of empty caches to bypass the
    # early "backbone was never called" guard.
    from mlx_lm.models.cache import KVCache

    n_layers = len(target.model.layers)
    target._mtp_target_cache = [KVCache() for _ in range(n_layers)]

    hidden_size = target.args.hidden_size
    # (B=2, T=1, H) — should raise
    hidden_states = mx.zeros((2, 1, hidden_size))
    next_token_ids = mx.zeros((2, 1), dtype=mx.int32)
    mtp_cache = target.make_mtp_cache()

    with pytest.raises(ValueError, match="batch=1"):
        target.mtp_forward(hidden_states, next_token_ids, mtp_cache)
