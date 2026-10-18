# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the GatedDeltaNet input-projection fusion.

The fusion must be BYTE-exact and strictly opt-out-able. The core risk
this file pins: the fused concat is only byte-identical to the four
stock projections while the quantized matmul stays on the narrow-batch
kernel (M = B*S small); wider inputs must take the sliced per-projection
path. These tests cover byte equality across decode/prefill widths and
a cached prefill+decode sequence, the structural gate, install-probe
gating, rewrite semantics, idempotency, and the env kill-switch.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

from rapid_mlx import gdn_in_proj_fusion


def _bits_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and gdn_in_proj_fusion._bytes_of(a) == gdn_in_proj_fusion._bytes_of(b)
    )


def _tiny_args(**overrides):
    base = dict(
        model_type="qwen3_5_text",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        vocab_size=128,
    )
    base.update(overrides)
    return TextModelArgs(**base)


class TinyGDNModel(nn.Module):
    """Minimal container so fuse_gdn_in_proj's module scan finds the layers."""

    def __init__(self, n_layers=2, quantize=True):
        super().__init__()
        args = _tiny_args()
        self.layers = [GatedDeltaNet(args) for _ in range(n_layers)]
        if quantize:
            nn.quantize(self, group_size=32, bits=4)
        mx.eval(self.parameters())
        self.hidden_size = args.hidden_size


def _inputs(rows, hidden, seed=3):
    mx.random.seed(seed)
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _run_all(model, x, caches=None):
    outs = []
    for i, layer in enumerate(model.layers):
        cache = caches[i] if caches is not None else None
        outs.append(layer(x, None, cache))
    mx.eval(*outs)
    return outs


class TestByteExactness:
    @pytest.mark.parametrize("rows", [1, 2, 8, 16, 64])
    def test_fused_output_is_byte_identical(self, rows):
        model = TinyGDNModel()
        x = _inputs(rows, model.hidden_size)
        before = _run_all(model, x)
        n = gdn_in_proj_fusion.fuse_gdn_in_proj(model)
        assert n == 2
        after = _run_all(model, x)
        for b, a in zip(before, after):
            assert _bits_equal(b, a), f"fusion changed bytes at rows={rows}"

    def test_cached_prefill_then_decode_byte_identical(self):
        model = TinyGDNModel()
        xp = _inputs(16, model.hidden_size, seed=7)
        xd = _inputs(1, model.hidden_size, seed=8)

        caches = [ArraysCache(size=2) for _ in model.layers]
        p_before = _run_all(model, xp, caches)
        d_before = _run_all(model, xd, caches)
        states_before = [(c[0], c[1]) for c in caches]
        mx.eval(*[s for pair in states_before for s in pair])

        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2

        caches = [ArraysCache(size=2) for _ in model.layers]
        p_after = _run_all(model, xp, caches)
        d_after = _run_all(model, xd, caches)
        states_after = [(c[0], c[1]) for c in caches]
        mx.eval(*[s for pair in states_after for s in pair])

        for b, a in zip(p_before + d_before, p_after + d_after):
            assert _bits_equal(b, a)
        for (c0b, c1b), (c0a, c1a) in zip(states_before, states_after):
            assert _bits_equal(c0b, c0a)
            assert _bits_equal(c1b, c1a)


class TestRewriteSemantics:
    def test_originals_dropped_and_container_reused(self):
        model = TinyGDNModel()
        gdn = model.layers[0]
        qkv_before = gdn.in_proj_qkv
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        # Fresh container (failure atomicity): the live in_proj_qkv is
        # never mutated, but its quantization params carry over.
        assert gdn.in_proj_fused is not qkv_before
        assert gdn.in_proj_fused.group_size == qkv_before.group_size
        assert gdn.in_proj_fused.bits == qkv_before.bits
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            assert not hasattr(gdn, name)
        assert gdn._rapid_gdn_bounds[-1] == gdn.in_proj_fused.weight.shape[0] * 1
        # Fused rows = qkv + z + b + a for the tiny geometry.
        key_dim, value_dim = 2 * 32, 4 * 32
        assert gdn._rapid_gdn_bounds == [
            2 * key_dim + value_dim,
            2 * key_dim + 2 * value_dim,
            2 * key_dim + 2 * value_dim + 4,
            2 * key_dim + 2 * value_dim + 8,
        ]

    def test_idempotent(self):
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0

    def test_unfused_instance_still_works_after_class_patch(self):
        fused_model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(fused_model) == 2
        # A fresh, unfused instance dispatches through the patched class
        # __call__ and must take the stock path.
        stock_model = TinyGDNModel()
        x = _inputs(4, stock_model.hidden_size)
        outs = _run_all(stock_model, x)
        assert all(o.shape == (1, 4, stock_model.hidden_size) for o in outs)


class TestStructuralGate:
    def test_unquantized_rejected(self):
        model = TinyGDNModel(quantize=False)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0

    def test_subclass_rejected(self):
        class CustomGDN(GatedDeltaNet):
            pass

        model = TinyGDNModel()
        model.layers[0].__class__ = CustomGDN
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_sharded_rejected(self):
        model = TinyGDNModel()
        model.layers[0].sharding_group = object()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_mixed_quantization_rejected(self):
        model = TinyGDNModel()
        gdn = model.layers[0]
        w = gdn.in_proj_z
        deq = mx.dequantize(
            w.weight, w.scales, w.biases, group_size=w.group_size, bits=w.bits
        )
        requant = nn.QuantizedLinear(
            deq.shape[1], deq.shape[0], bias=False, group_size=64, bits=8
        )
        wq, sq, bq = mx.quantize(deq, group_size=64, bits=8)
        requant.weight, requant.scales, requant.biases = wq, sq, bq
        gdn.in_proj_z = requant
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_non_affine_mode_rejected(self):
        # The sliced wide path calls mx.quantized_matmul with the affine
        # interpretation; a quartet in any other mode must stay stock.
        model = TinyGDNModel()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            getattr(model.layers[0], name).mode = "mxfp4"
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_non_gdn_model_noop(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0


class TestGating:
    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_GDN_IN_PROJ_FUSION", "0")
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    def test_projection_probe_failure_disables(self, monkeypatch):
        model = TinyGDNModel()
        monkeypatch.setattr(
            gdn_in_proj_fusion, "_probe_projection_parity", lambda gdn: frozenset()
        )
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    def test_whole_layer_probe_failure_disables(self, monkeypatch):
        model = TinyGDNModel()
        monkeypatch.setattr(
            gdn_in_proj_fusion,
            "_probe_whole_layer_parity",
            lambda *a, **k: False,
        )
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")


class TestFailureContainment:
    def test_probe_exception_contained(self, monkeypatch):
        def boom(gdn):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(gdn_in_proj_fusion, "_probe_projection_parity", boom)
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")
        # Model still runs stock.
        x = _inputs(2, model.hidden_size)
        _run_all(model, x)

    def test_fuse_one_exception_contained(self, monkeypatch):
        model = TinyGDNModel()
        entered = {"commit": False}
        real = gdn_in_proj_fusion._fuse_one

        def boom(gdn, dtypes=None):
            # Leave the synthetic whole-layer probe's _fuse_one intact so
            # the install reaches the real commit phase, then blow up on
            # the first real-model layer (before any mutation).
            if any(gdn is layer for layer in model.layers):
                entered["commit"] = True
                raise RuntimeError("commit blew up")
            real(gdn, dtypes)

        monkeypatch.setattr(gdn_in_proj_fusion, "_fuse_one", boom)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert entered["commit"]
        for layer in model.layers:
            assert hasattr(layer, "in_proj_qkv")
        x = _inputs(2, model.hidden_size)
        _run_all(model, x)


class TestPartialCommit:
    def test_failure_between_layers_leaves_model_functional(self, monkeypatch):
        # A crash after some layers were committed must leave a model
        # whose fused layers are correct and whose stock layers are
        # untouched — outputs byte-identical to the unfused model.
        model = TinyGDNModel()
        x = _inputs(2, model.hidden_size)
        before = _run_all(model, x)

        real = gdn_in_proj_fusion._fuse_one
        calls = {"n": 0}

        def fuse_then_boom(gdn, dtypes=None):
            # The synthetic whole-layer probe also calls _fuse_one; only
            # count commits against the real model's layers.
            if any(gdn is layer for layer in model.layers):
                if calls["n"] >= 1:
                    raise RuntimeError("boom on second layer")
                calls["n"] += 1
            real(gdn, dtypes)

        monkeypatch.setattr(gdn_in_proj_fusion, "_fuse_one", fuse_then_boom)
        # The true committed count is reported, not 0.
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1
        # Module-scan order is not guaranteed; exactly one layer was
        # committed before the crash, the other is untouched stock.
        fused_flags = [hasattr(layer, "in_proj_fused") for layer in model.layers]
        stock_flags = [hasattr(layer, "in_proj_qkv") for layer in model.layers]
        assert sum(fused_flags) == 1
        assert sum(stock_flags) == 1
        assert all(f != st for f, st in zip(fused_flags, stock_flags))
        after = _run_all(model, x)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)


class TestDtypeGate:
    def test_unprobed_dtype_takes_sliced_path(self, monkeypatch):
        model = TinyGDNModel()
        x32 = _inputs(2, model.hidden_size).astype(mx.float32)
        before = _run_all(model, x32)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        # float32 was never probed: narrow float32 inputs must take the
        # sliced (byte-exact-by-construction) path, so outputs match.
        gdn = model.layers[0]
        assert mx.float32 not in gdn._rapid_gdn_dtypes
        after = _run_all(model, x32)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)

    def test_float16_narrow_byte_identical(self):
        model = TinyGDNModel()
        x16 = _inputs(2, model.hidden_size).astype(mx.float16)
        before = _run_all(model, x16)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        after = _run_all(model, x16)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)


class TestWidthDispatch:
    def test_narrow_and_wide_paths_agree(self):
        # Same input, same fused layer: once through the fused single
        # matmul, once with the dtype gate emptied to force the sliced
        # path — the two dispatch arms must agree byte-for-byte.
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        rows = gdn_in_proj_fusion._FUSED_MAX_ROWS
        x = _inputs(rows, model.hidden_size, seed=11)
        gdn = model.layers[0]
        assert x.dtype in gdn._rapid_gdn_dtypes
        y_fused = gdn(x, None, None)
        mx.eval(y_fused)
        saved = gdn._rapid_gdn_dtypes
        try:
            gdn._rapid_gdn_dtypes = frozenset()
            y_sliced = gdn(x, None, None)
            mx.eval(y_sliced)
        finally:
            gdn._rapid_gdn_dtypes = saved
        assert _bits_equal(y_fused, y_sliced)

    def test_wide_path_uses_slices(self, monkeypatch):
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        gdn = model.layers[0]
        captured = []
        orig = mx.quantized_matmul

        def counting(*args, **kwargs):
            captured.append(args[1])
            return orig(*args, **kwargs)

        monkeypatch.setattr(mx, "quantized_matmul", counting)
        x = _inputs(64, model.hidden_size)
        gdn(x, None, None)
        # Four sliced projection matmuls, in quartet order, each carrying
        # EXACTLY the corresponding row block of the fused weight array
        # (content-exact, so reusing one slice twice cannot pass).
        assert len(captured) >= 4
        fused_w = gdn.in_proj_fused.weight
        bounds = [0] + list(gdn._rapid_gdn_bounds)
        for i in range(4):
            expected = fused_w[bounds[i] : bounds[i + 1]]
            assert captured[i].shape == expected.shape
            assert bool(mx.array_equal(captured[i], expected))


class TestVLMFamily:
    @pytest.fixture(autouse=True)
    def vlm(self):
        pytest.importorskip("mlx_vlm")

    @staticmethod
    def model(*, quantize=True, qualified=False):
        import dataclasses

        from mlx_vlm.models.qwen3_5.config import TextConfig
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

        fields = {field.name for field in dataclasses.fields(TextConfig)}
        values = {k: v for k, v in vars(_tiny_args()).items() if k in fields}
        values.update(rms_norm_eps=1e-6, max_position_embeddings=1024)
        if qualified:
            values.update(
                hidden_size=2048,
                linear_num_value_heads=32,
                linear_num_key_heads=16,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
            )
        model = nn.Sequential(Qwen3_5GatedDeltaNet(TextConfig(**values)))
        model.set_dtype(mx.bfloat16)
        if quantize:
            nn.quantize(model, group_size=32, bits=4)
        model.eval()
        mx.eval(model.parameters())
        return model

    @staticmethod
    def cache_type(native):
        from mlx_vlm.models.cache import ArraysCache as VLMCache

        from rapid_mlx.engine.batched import _qwen36_text_arrays_cache_type

        return _qwen36_text_arrays_cache_type() if native else VLMCache

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize(
        "batch,rows,masked",
        [(1, 1, False), (4, 1, False), (1, 16, False), (4, 16, True)],
    )
    def test_outputs_and_cache(self, native, batch, rows, masked):
        model = self.model()
        layer = model.layers[0]
        x = mx.repeat(_inputs(rows, layer.hidden_size), batch, axis=0)
        xd = mx.repeat(_inputs(1, layer.hidden_size, seed=9), batch, axis=0)
        mask = (
            mx.arange(rows)[None, :]
            < mx.array([rows - i for i in range(batch)])[:, None]
            if masked
            else None
        )

        def run():
            cache = self.cache_type(native)(size=2)
            cache.lengths = (
                mx.array([rows - i for i in range(batch)]) if masked else None
            )
            y = layer(x, mask, cache)
            prefill_state = [cache[0], cache[1]]
            cache.lengths = None
            z = layer(xd, None, cache)
            result = [y, *prefill_state, z, cache[0], cache[1]]
            mx.eval(result)
            return result

        before = run()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        after = run()
        assert all(_bits_equal(a, b) for a, b in zip(before, after))

    def test_no_cache_and_mismatched_batch(self):
        model = self.model()
        layer = model.layers[0]
        x = _inputs(1, layer.hidden_size)
        cache = self.cache_type(False)(size=2)
        cache[0] = mx.zeros((2, 3, layer.conv_dim), dtype=x.dtype)
        mask = mx.ones((2, 1), dtype=mx.bool_)
        before = layer(x, mask, cache)
        plain = layer(x)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1
        cache[0] = mx.zeros((2, 3, layer.conv_dim), dtype=x.dtype)
        cache[1] = None
        assert _bits_equal(before, layer(x, mask, cache))
        assert _bits_equal(plain, layer(x))

    @pytest.mark.parametrize(
        "failure",
        [
            "import",
            "symbol",
            "projection",
            "whole",
            "structure",
            "subclass",
            "disabled",
        ],
    )
    def test_fail_closed(self, monkeypatch, failure):
        model = self.model(quantize=failure != "structure")
        if failure == "import":
            import sys

            monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.language", None)
        elif failure == "symbol":
            import mlx_vlm.models.qwen3_5.language as language

            monkeypatch.delattr(language, "_qwen3_5_advance_lengths_info")
        elif failure == "projection":
            monkeypatch.setattr(
                gdn_in_proj_fusion, "_probe_projection_parity", lambda _: frozenset()
            )
        elif failure == "whole":
            monkeypatch.setattr(
                gdn_in_proj_fusion, "_probe_whole_layer_parity", lambda *args: False
            )
        elif failure == "subclass":

            class Derived(type(model.layers[0])):
                pass

            model.layers[0].__class__ = Derived
        elif failure == "disabled":
            monkeypatch.setenv("RAPID_MLX_GDN_IN_PROJ_FUSION", "0")
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    @pytest.mark.parametrize("decode_first", [False, True])
    @pytest.mark.parametrize("projection_qualified", [False, True])
    def test_decode_composition(self, monkeypatch, decode_first, projection_qualified):
        import copy

        from rapid_mlx import qwen35_fused_gdn_decode as decode

        model = self.model(qualified=True)
        model.layers.append(copy.deepcopy(model.layers[0]))
        stock_projections = {
            id(item): tuple(gdn_in_proj_fusion._proj_modules(item))
            for item in model.layers
        }
        layer = model.layers[0]
        cls = type(layer)
        stock_call = getattr(cls, "_rapid_gdn_in_proj_original_call", cls.__call__)
        monkeypatch.setattr(cls, "__call__", stock_call)
        for marker in (
            "_rapid_gdn_in_proj_fused_call",
            "_rapid_qwen35_fused_gdn_patched",
        ):
            monkeypatch.setattr(cls, marker, False, raising=False)
        for marker in (
            "_rapid_gdn_in_proj_original_call",
            "_rapid_qwen35_fused_gdn_original_call",
        ):
            monkeypatch.setattr(cls, marker, stock_call, raising=False)
        x = _inputs(1, layer.hidden_size)

        def run():
            cache = self.cache_type(True)(size=2)
            layer(x, None, cache)
            out = layer(x, None, cache)
            mx.eval(out, cache[0], cache[1])
            return [out, cache[0], cache[1]]

        before = run()
        calls = []
        real_kernel = decode.fused_gdn_decode

        def spy(*args, **kwargs):
            calls.append(True)
            return real_kernel(*args, **kwargs)

        monkeypatch.setattr(decode, "fused_gdn_decode", spy)
        if projection_qualified:
            # Exercise the composition contract on hardware whose real fused
            # matmul fails byte qualification. Projection arithmetic has its
            # own parity tests; here the stock quartet supplies exact values
            # through the fused wrapper so both optimizations stay installed.
            monkeypatch.setattr(
                gdn_in_proj_fusion,
                "_probe_projection_parity",
                lambda _item: frozenset({mx.bfloat16}),
            )
            monkeypatch.setattr(
                gdn_in_proj_fusion, "_probe_whole_layer_parity", lambda *args: True
            )
            monkeypatch.setattr(
                gdn_in_proj_fusion,
                "_fused_projections",
                lambda item, values: tuple(
                    part(values) for part in stock_projections[id(item)]
                ),
            )
        else:
            monkeypatch.setattr(
                gdn_in_proj_fusion,
                "_probe_projection_parity",
                lambda _item: frozenset(),
            )
        if decode_first:
            assert decode.install_qwen35_fused_gdn_decode(model) == 2
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == (
            2 if projection_qualified else 0
        )
        assert all(
            hasattr(item, "in_proj_fused") == projection_qualified
            for item in model.layers
        )
        assert decode.install_qwen35_fused_gdn_decode(model) == 2
        assert all(getattr(item, decode._TAG) for item in model.layers)
        after = run()
        assert all(_bits_equal(a, b) for a, b in zip(before, after))

        assert calls  # Complete singleton decode reached the composed kernel.

    @pytest.mark.parametrize("native", [False, True])
    def test_install_probe_rejects_cache_drift(self, monkeypatch, native):
        model = self.model()
        layer_type = type(model.layers[0])
        imports = gdn_in_proj_fusion._vlm_gdn_imports()
        nn_module, update, _, *helpers = imports
        gdn_in_proj_fusion._ensure_call_patch(layer_type, nn_module, update, *helpers)
        original = layer_type.__call__
        target_cache_type = self.cache_type(native)

        def drift(self, inputs, mask=None, cache=None):
            out = original(self, inputs, mask, cache)
            if (
                hasattr(self, "in_proj_fused")
                and inputs.shape[1] == 1
                and type(cache) is target_cache_type
            ):
                cache[1] = cache[1] + 1
            return out

        monkeypatch.setattr(layer_type, "__call__", drift)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    def test_each_layer_probed_independently_without_lm(self, monkeypatch):
        model = self.model()
        model.layers.append(self.model().layers[0])
        monkeypatch.setattr(gdn_in_proj_fusion, "_gdn_imports", lambda: None)
        real_probe = gdn_in_proj_fusion._probe_projection_parity
        probed = []

        def probe(layer):
            probed.append(layer)
            return real_probe(layer)

        monkeypatch.setattr(gdn_in_proj_fusion, "_probe_projection_parity", probe)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        assert {id(layer) for layer in probed} == {id(layer) for layer in model.layers}
