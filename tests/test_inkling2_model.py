# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Tiny configuration and checkpoint conversion adapted from the source in
# rapid_mlx/models/inkling2/NOTICE, under its accompanying MIT license.
"""Text-only catalog, loop/cache, quantization and loader regressions."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rapid_mlx.model_aliases import resolve_model, resolve_profile, resolve_revision
from rapid_mlx.model_auto_config import detect_model_config

REPO = "mlx-community/Inkling-2-Mini-4bit"
REVISION = "9856dc53edead71405f8348930937c3638ff21a2"

CONFIG = {
    "model_type": "inkling2",
    "image_token_id": 127,
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4,
        "n_layers_per_attn_res_block": 3,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 64,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
    },
}


CONFIG["model_type"] = "inkling2_loop"
CONFIG["text_config"]["loop_config"] = {
    "core_start": 3,
    "core_end": 6,
    "num_passes": 3,
    "gate_rank": 8,
    "is_varloop": False,
}


# Complete reviewed source of the five numerical-core files omitted by
# .coveragerc. See NOTICE for provenance. Review edits before updating pins;
# this contract reads bytes only and requires neither MLX nor Metal.
_NUMERICAL_CORE_SHA256 = {
    "layers.py": "a05bb4dfa45f9533e3c0c244bcf57689b773035147887f986f82599e8c5cc772",
    "kernels.py": "9f5fbf99563ada812a3f264eb18268b6fa8ad02fdc8ef6109e53edfb967d82c9",
    "model.py": "d4244b2ab9864a3d99c2122ba27897c945a45004c6802ebf37f4a0bd7b99a091",
    "switch_layers.py": "31fb0b6e31112276694f4fa8ba25a92cfc994f01bb90780792e54fd8fd4c54c9",
    "checkpoint.py": "7ba19519e069ce47319641f473a696dd3cfa82c246f2eec40949268a6bd2935f",
}


def test_numerical_core_source_contract():
    import configparser

    root = Path(__file__).parents[1]
    coverage = configparser.ConfigParser()
    coverage.read(root / ".coveragerc")
    prefix = "rapid_mlx/models/inkling2/"
    omitted = {
        path.strip().removeprefix(prefix)
        for path in coverage["run"]["omit"].splitlines()
        if path.strip().startswith(prefix)
    }
    assert omitted == _NUMERICAL_CORE_SHA256.keys()
    for filename, expected in _NUMERICAL_CORE_SHA256.items():
        assert hashlib.sha256((root / prefix / filename).read_bytes()).hexdigest() == (
            expected
        ), filename


@pytest.fixture
def glue_loader(monkeypatch):
    """Execute glue in isolation with CPU array/import doubles, even with MLX installed.

    No production package import (which initializes Metal kernels), distributed
    process group, or GPU is needed. All sys.modules changes are scoped to a test.
    """
    import rapid_mlx

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Array(np.ndarray):
        def __new__(cls, value, dtype=None):
            return np.asarray(value, dtype=dtype).view(cls)

    mx = module(
        "mlx.core",
        array=Array,
        Dtype=np.dtype,
        float32=np.float32,
        float16=np.float16,
        contiguous=Mock(side_effect=np.ascontiguousarray),
        eval=Mock(),
        concatenate=Mock(side_effect=np.concatenate),
        distributed=SimpleNamespace(Group=object, all_gather=Mock()),
    )
    module("mlx", core=mx, nn=SimpleNamespace(Module=object))
    shim = module("rapid_mlx._mlx_compat", install=Mock())
    monkeypatch.setattr(rapid_mlx, "_mlx_compat", shim, raising=False)
    module("mlx_lm", __path__=[])
    module("mlx_lm.models", __path__=[])
    module("mlx_lm.kernels", __path__=[])

    def load(filename):
        name = f"rapid_mlx.models.inkling2.{filename}"
        path = (
            Path(__file__).parents[1] / "rapid_mlx/models/inkling2" / f"{filename}.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        return result

    return SimpleNamespace(load=load, module=module, mx=mx, shim=shim)


@pytest.fixture
def compat_loader(glue_loader):
    def load(*, global_scale=True, routing=True):
        class Native:
            if global_scale:

                def to_quantized(self, *, global_scale=False):
                    pass
            else:

                def to_quantized(self):
                    pass

        native = glue_loader.module(
            "mlx_lm.models.switch_layers",
            SwitchLinear=Native,
            _gather_sort=object(),
            _scatter_unsort=object(),
        )
        fallback = glue_loader.module(
            "rapid_mlx.models.inkling2.switch_layers",
            SwitchLinear=object(),
            _gather_sort=object(),
            _scatter_unsort=object(),
        )
        if routing:
            router = glue_loader.module(
                "mlx_lm.kernels.expert_routing", expert_routing=object()
            ).expert_routing
        else:
            # Missing export forces ImportError even if the real helper was loaded.
            glue_loader.module("mlx_lm.kernels.expert_routing")
            router = None
        result = glue_loader.load("compat")
        selected = native if global_scale else fallback
        for name in ("SwitchLinear", "_gather_sort", "_scatter_unsort"):
            assert getattr(result, name) is getattr(selected, name)
        assert result.expert_routing is router
        glue_loader.shim.install.assert_called_once_with()
        return result

    return load


@pytest.mark.parametrize("global_scale", [False, True])
@pytest.mark.parametrize("routing", [False, True])
def test_compat_version_fallbacks(compat_loader, global_scale, routing):
    compat_loader(global_scale=global_scale, routing=routing)


def test_compat_scalar_and_row_threads(compat_loader):
    compat = compat_loader()
    value = compat.scalar(1.5)
    assert value.shape == () and value.dtype == np.float32 and value.item() == 1.5
    assert compat.scalar(1.5) is value
    assert compat.scalar(1.5, np.float16).dtype == np.float16
    for dim, expected in [(1, 32), (511, 32), (512, 64), (8192, 1024), (16384, 1024)]:
        assert compat.row_threads(dim) == expected


def test_compat_gather_rank_order(compat_loader, glue_loader):
    compat = compat_loader()
    x = glue_loader.mx.array(np.arange(12).reshape(2, 3, 2))
    group = SimpleNamespace(size=lambda: 2)
    glue_loader.mx.distributed.all_gather.return_value = np.concatenate(
        [x.reshape(-1, 2), x.reshape(-1, 2) + 100], axis=0
    )
    actual = compat.gather_last_axis(x, group)
    np.testing.assert_array_equal(actual, np.concatenate([x, x + 100], axis=-1))
    args, kwargs = glue_loader.mx.distributed.all_gather.call_args
    np.testing.assert_array_equal(args[0], x.reshape(-1, 2))
    assert kwargs == {"group": group}


@pytest.mark.parametrize("sharding", ["all-to-sharded", "sharded-to-all"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("chunked", [False, True])
def test_compat_shard_experts(compat_loader, glue_loader, sharding, rank, chunked):
    compat = compat_loader()
    mx = glue_loader.mx

    class LargeArray(mx.array):
        # Report a large logical allocation to exercise the chunk budget with
        # tiny storage. Slicing/concatenation still operate on real CPU values.
        @property
        def size(self):
            return self.shape[0] * (1 << 30)

    weights = mx.array(np.arange(3 * 4 * 6).reshape(3, 4, 6))
    if chunked:
        weights = weights.view(LargeArray)
    scales = mx.array([1.0, 2.0, 3.0])
    nested = {"weight": weights}
    module = SimpleNamespace(
        parameters=lambda: {"weight": weights, "scales": scales, "nested": nested},
        update=Mock(),
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    compat.shard_experts_inplace(module, sharding, group)
    updates = module.update.call_args.args[0]
    assert set(updates) == {"weight"}
    expected = (
        weights[:, rank * 2 : (rank + 1) * 2]
        if sharding == "all-to-sharded"
        else weights[..., rank * 3 : (rank + 1) * 3]
    )
    np.testing.assert_array_equal(updates["weight"], expected)
    assert mx.eval.call_count == mx.contiguous.call_count == (3 if chunked else 1)
    assert mx.concatenate.call_count == int(chunked)


@pytest.mark.parametrize("loop", [None, {}, {"core_start": 3, "core_end": 6}])
@pytest.mark.parametrize("vision", [None, {"hidden_size": 1024}])
def test_config_optional_fields(glue_loader, loop, vision):
    class BaseModelArgs:
        @classmethod
        def from_dict(cls, config):
            return cls(**config)

    glue_loader.module("mlx_lm.models.base", BaseModelArgs=BaseModelArgs)
    config = glue_loader.load("config")
    text = {**CONFIG["text_config"], "loop_config": loop}
    args = config.ModelArgs(
        model_type="inkling2_loop", text_config=text, vision_config=vision
    )
    assert isinstance(args.text, config.TextArgs)
    assert args.text.hidden_size == 64
    assert list(args.text.loop_core) == ([3, 4, 5] if loop else [])
    assert args.text.loop_config == loop
    assert args.vision is None
    assert args.vision_config == vision
    assert args.image_token_id == 200054
    # Exercise omission of all optional fields, including loop_config.
    text.pop("loop_config")
    defaults = config.ModelArgs("inkling2_loop", text, image_token_id=127)
    assert defaults.text.loop_config is None and list(defaults.text.loop_core) == []
    assert defaults.vision_config is None and defaults.image_token_id == 127


@pytest.fixture(scope="module")
def runtime():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from rapid_mlx.models import inkling2

    return mx, inkling2


def create_model(runtime, *, dtype=None, loop=True):
    mx, inkling2 = runtime
    from mlx.utils import tree_flatten, tree_unflatten

    config = json.loads(json.dumps(CONFIG))
    if not loop:
        config["text_config"].pop("loop_config")
    mx.random.seed(0)
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(("norm.weight", "l2_clip.scale")):
            value = 1 + 0.1 * mx.random.normal(value.shape)
        else:
            fan_in = value.shape[-1] if value.ndim > 1 else 16
            value = mx.random.normal(value.shape) * fan_in**-0.5
        # Exercise learned queries, router and message gates rather than their
        # zero initialization. Keep precision-sensitive parameters in float32.
        if dtype is not None and model.cast_predicate(key):
            value = value.astype(dtype)
        weights.append((key, value))
    model.update(tree_unflatten(weights))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_catalog():
    assert resolve_model("inkling-2-mini-4bit") == REPO
    assert resolve_revision("inkling-2-mini-4bit") == REVISION
    for name in ("inkling-2-mini-4bit", REPO):
        p = resolve_profile(name)
        assert p.model_type == "inkling2_loop"
        assert p.is_moe and p.is_text_only and p.modality == "text"
        assert p.is_hybrid and p.is_hybrid_explicit
        assert not p.supports_spec_decode and not p.mtp_default_enabled
        assert p.tool_call_parser == "inkling" and p.reasoning_parser == "inkling"
        assert detect_model_config(name) == p


@pytest.mark.parametrize("directory", ["checkpoint", "Qwen3-8B"])
def test_local_metadata(tmp_path, directory):
    tmp_path = tmp_path / directory
    tmp_path.mkdir()
    config = {**CONFIG, "vision_config": {"hidden_size": 1024}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    p = detect_model_config(str(tmp_path))
    assert p.is_moe and p.is_text_only and p.is_hybrid and p.is_hybrid_explicit
    assert not p.supports_spec_decode
    assert p.tool_call_parser == "inkling" and p.reasoning_parser == "inkling"


def test_preflight():
    from rapid_mlx.byom.preflight import RAPID_MODEL_TYPES

    assert "inkling2_loop" in RAPID_MODEL_TYPES


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.requires_mlx
def test_forward(runtime, dtype):
    mx, _ = runtime
    model = create_model(runtime, dtype=getattr(mx, dtype))
    logits = model(mx.array([[1, 2, 3], [4, 5, 6]]))
    mx.eval(logits)
    assert logits.shape == (2, 3, 120)
    assert logits.dtype == mx.float32
    assert mx.all(mx.isfinite(logits)).item()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.requires_mlx
def test_cached_decode_matches_full(runtime, batch, loop):
    mx, _ = runtime
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    # Full-model cache parity uses float32, as in the source regression.
    # In bf16, prefill/decode use different attention kernels; rounding can
    # change top-k expert selection and amplify differences through loop passes.
    model = create_model(runtime, dtype=mx.float32, loop=loop)
    tokens = mx.random.randint(0, 120, (batch, 20))
    full = model(tokens)
    cache = model.make_cache()
    visits = model.model.language_model.layer_visits
    assert len(cache) == (14 if loop else 8)
    assert any(isinstance(c[0], RotatingKVCache) for c in cache)
    assert any(type(c[0]) is KVCache for c in cache)
    assert len({id(c) for c in cache}) == len(visits)
    # Chunked prefill crosses the short sliding window, followed by single
    # tokens to cover ring-buffer wraparound and the singleton SDPA path.
    chunks = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
    chunks += [model(tokens[:, i : i + 1], cache=cache) for i in range(12, 20)]
    actual = mx.concatenate(chunks, axis=1)
    mx.eval(full, actual)
    assert mx.allclose(actual, full, atol=1e-4, rtol=1e-4).item()
    assert all(c[0].offset == 20 for c in cache)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.requires_mlx
def test_bfloat16_attention_cache_matches_full(runtime, batch, sliding):
    mx, inkling2 = runtime
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from rapid_mlx.models.inkling2.layers import Attention

    args = inkling2.ModelArgs.from_dict(CONFIG).text
    attention = Attention(args, sliding=sliding)
    mx.random.seed(3)
    length = 20
    hq, hkv, dim = (
        attention.num_heads,
        attention.num_key_value_heads,
        attention.head_dim,
    )
    query = (0.5 * mx.random.normal((batch, hq, length, dim))).astype(mx.bfloat16)
    key = (0.5 * mx.random.normal((batch, hkv, length, dim))).astype(mx.bfloat16)
    value = mx.random.normal((batch, hkv, length, dim)).astype(mx.bfloat16)
    bias = mx.random.normal((batch, length, hq, args.rel_extent)).astype(mx.bfloat16)
    sinks = mx.random.normal((batch, length, hq)).astype(mx.bfloat16)
    full = attention._attention(query, key, value, bias, sinks, None)
    cache = RotatingKVCache(max_size=args.sliding_window_size) if sliding else KVCache()
    chunks = []
    # Identical bf16 projections isolate cache positions and kernel numerics
    # from projection rounding and discontinuous expert routing downstream.
    for start, end in [(0, 6), (6, 12)] + [(i, i + 1) for i in range(12, length)]:
        keys, values = cache.update_and_fetch(
            key[:, :, start:end], value[:, :, start:end]
        )
        chunks.append(
            attention._attention(
                query[:, :, start:end],
                keys,
                values,
                bias[:, start:end],
                sinks[:, start:end],
                cache,
            )
        )
    actual = mx.concatenate(chunks, axis=1)
    mx.eval(full, actual)
    # The source kernel tests use TOL[mx.bfloat16] == 1e-2, with a float32
    # comparison. Whole-model bf16 logits have no upstream parity tolerance.
    assert mx.allclose(
        actual.astype(mx.float32), full.astype(mx.float32), atol=1e-2, rtol=1e-2
    ).item()
    assert cache.offset == length


@pytest.mark.parametrize("prefix", ["model.visual.", "model.vision_tower."])
@pytest.mark.requires_mlx
def test_sanitize_drops_vision(runtime, prefix):
    mx, inkling2 = runtime
    config = {**CONFIG, "vision_config": {"hidden_size": 1024}}
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    assert model.args.vision is None
    assert "vision_tower" not in model.model
    text_weight = mx.ones((128, 64))
    weights = {
        "model.llm.embed.weight": text_weight,
        prefix + "backbone.layers.0.weight": mx.zeros((1,)),
    }
    sanitized = model.sanitize(weights)
    assert set(sanitized) == {"model.language_model.embed_tokens.weight"}
    assert mx.array_equal(next(iter(sanitized.values())), text_weight).item()


_CHECKPOINT_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
    (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
    (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
    (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
]


def checkpoint(model) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(model.parameters()))
    ckpt = {}
    denominator = mx.array([7.5])
    for k, v in weights.items():
        if k.endswith(".message_input.query"):
            q, kv = mx.concatenate([v[:, :1], v[:, 3:]], axis=1), v[:, 1:3]
            ckpt["model.llm.middle_cycle_cross_pass_queries"] = q.flatten(0, 1)
            ckpt["model.llm.middle_cycle_cross_pass_kv_queries"] = kv.flatten(0, 1)
            continue
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$", k):
            prefix, name, param = m.groups()
            prefix = prefix.replace("model.language_model.", "model.llm.")
            w = f"{prefix}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if param == "weight":
                ckpt[w] = v.view(mx.uint8)
            elif param == "scales":
                ckpt[f"{w}_scale"] = v
            else:
                ckpt[f"{w}_amax"] = v * denominator / 2688.0
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _CHECKPOINT_NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.requires_mlx
def test_sanitize_mixed_experts(runtime, converted, legacy, monkeypatch):
    mx, inkling2 = runtime
    from mlx import nn
    from mlx.utils import tree_flatten

    if legacy:
        from rapid_mlx.models.inkling2 import checkpoint as checkpoint_module
        from rapid_mlx.models.inkling2 import layers, switch_layers

        monkeypatch.setattr(layers, "SwitchLinear", switch_layers.SwitchLinear)
        monkeypatch.setattr(
            checkpoint_module, "SwitchLinear", switch_layers.SwitchLinear
        )

    model = create_model(runtime)

    def predicate(path, module):
        # The source quantization helper applies this capability guard before
        # invoking its expert predicate. nn.quantize calls ours directly.
        if not hasattr(module, "to_quantized") or ".mlp.experts." not in path:
            return False
        if ".layers.7." in path:
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}

    nn.quantize(model, class_predicate=predicate)
    expected = dict(tree_flatten(model.parameters()))
    weights = expected if converted else checkpoint(model)
    target = inkling2.Model(inkling2.ModelArgs.from_dict(CONFIG))
    sanitized = target.sanitize(weights)
    target.load_weights(list(sanitized.items()), strict=True)
    assert set(sanitized) == set(expected)
    for name, value in expected.items():
        assert mx.allclose(sanitized[name], value, atol=1e-6).item(), name
    tokens = mx.array([[1, 2, 3]])
    assert mx.allclose(model(tokens), target(tokens), atol=1e-4, rtol=1e-4).item()


@pytest.mark.requires_mlx
def test_public_loader_text_only(runtime, tmp_path):
    mx, _ = runtime
    from mlx import nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    source = create_model(runtime)
    nn.quantize(
        source,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: (
            hasattr(m, "to_quantized") and source.quant_predicate(p, m)
        ),
    )
    config = {
        **CONFIG,
        "vision_config": {"hidden_size": 1024},
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "model_file": "untrusted.py",
        "auto_map": {"AutoModel": "untrusted.Model"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "untrusted.py").write_text(
        "raise AssertionError('checkpoint code ran')"
    )
    weights = dict(tree_flatten(source.parameters()))
    weights["model.vision_tower.weight"] = mx.zeros((2, 2))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    Tokenizer(WordLevel({"<unk>": 0, "hi": 1, "<eos>": 2})).save(
        str(tmp_path / "tokenizer.json")
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>", "unk_token": "<unk>"})
    )
    # A fresh process enters through the production loader before importing the
    # architecture; no direct model import can hide missing registration.
    script = """
import sys
import mlx.core as mx
from rapid_mlx.utils.tokenizer import load_model_with_fallback, _VENDORED_MODEL_TYPES
model, tokenizer = load_model_with_fallback(sys.argv[1])
from rapid_mlx.models.inkling2 import Model
from mlx_lm.utils import _get_classes
assert isinstance(model, Model)
assert _get_classes({"model_type": "inkling2_loop"})[0] is Model
assert "inkling2_loop" in _VENDORED_MODEL_TYPES
mx.save(sys.argv[2], model(mx.array([[1, 2, 3]])))
"""
    output = tmp_path / "logits.npy"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(output)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert mx.allclose(
        source(mx.array([[1, 2, 3]])), mx.load(str(output)), atol=1e-4
    ).item()
