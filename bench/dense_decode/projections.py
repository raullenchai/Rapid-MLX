# SPDX-License-Identifier: Apache-2.0
"""Process-local dense decode ablations, for development benchmarks only."""

import importlib

import mlx.core as mx
import mlx.nn as nn


class Group(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.bits = parts[0].bits
        self.gs = parts[0].group_size
        if any(
            getattr(p, "mode", "affine") != "affine" or p.get("bias") is not None
            for p in parts
        ):
            raise ValueError("Only bias-free affine projection groups are supported")
        geometry = {
            (
                m.bits,
                m.group_size,
                m.weight.shape[1:],
                m.weight.dtype,
                m.scales.shape[1:],
                m.scales.dtype,
                m.biases.dtype,
            )
            for m in parts
        }
        if len(geometry) != 1:
            raise ValueError("Projection groups require matching quantization metadata")
        self.weight = mx.concatenate([m.weight for m in parts])
        self.scales = mx.concatenate([m.scales for m in parts])
        self.biases = mx.concatenate([m.biases for m in parts])
        self.widths = [m.weight.shape[0] for m in parts]
        mx.eval(self.parameters())
        start = 0
        for m, n in zip(parts, self.widths):
            m.weight = self.weight[start : start + n]
            m.scales = self.scales[start : start + n]
            m.biases = self.biases[start : start + n]
            start += n

    def __call__(self, x):
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.gs,
            bits=self.bits,
        )


def install(model, arm):
    if arm in {"stock", "wired0", "wiredfit"}:
        return 0
    if type(model).__module__ not in {
        "mlx_vlm.models.qwen3_5.language",
        "mlx_vlm.models.lfm2.language",
    }:
        raise ValueError(
            "These graph ablations are qualified only for the Qwen3.5 and LFM2 language classes"
        )
    count = 0
    if arm in ("gu", "both", "all", "allfull"):
        for _, m in list(model.named_modules()):
            names = (
                ("gate_proj", "up_proj", "down_proj")
                if all(hasattr(m, n) for n in ("gate_proj", "up_proj", "down_proj"))
                else ("w1", "w3", "w2")
            )
            if not all(hasattr(m, n) for n in names):
                continue
            parts = [getattr(m, n) for n in names[:2]]
            if not all(
                isinstance(v, nn.QuantizedLinear) and v.get("bias") is None
                for v in parts
            ):
                continue
            if parts[0].weight.shape != parts[1].weight.shape:
                continue
            cls = type(m)
            original = cls.__call__
            swiglu = importlib.import_module(cls.__module__).swiglu
            if not getattr(cls, "_r4", False):

                def call(self, x, _orig=original, _sw=swiglu, _names=names):
                    group = getattr(self, "_r4_group", None)
                    if group is None or x.size // x.shape[-1] != 1:
                        return _orig(self, x)
                    g, u = mx.split(group(x), 2, axis=-1)
                    return getattr(self, _names[2])(_sw(g, u))

                cls.__call__ = call
                cls._r4 = True
            m._r4_group = Group(parts)
            count += 1

    class Part(nn.Module):
        def __init__(self, original, group, index):
            super().__init__()
            self.original = original
            object.__setattr__(self, "_group", group)
            object.__setattr__(self, "_index", index)

        def __call__(self, x):
            g = self._group
            i = self._index
            if x.size // x.shape[-1] != 1:
                return self.original(x)
            if i == 0:
                g.pending = g(x)
                g.pending_x = x
                g.pending_seen = set()
            if getattr(g, "pending_x", None) is not x:
                return self.original(x)
            start = sum(g.widths[:i])
            y = g.pending[..., start : start + g.widths[i]]
            g.pending_seen.add(i)
            if len(g.pending_seen) == len(g.widths):
                g.pending = None
                g.pending_x = None
                g.pending_seen.clear()
            return y

    if arm in ("both", "all", "allfull"):
        for _, m in list(model.named_modules()):
            names = next(
                (
                    ns
                    for ns in [
                        ("q_proj", "k_proj", "v_proj"),
                        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
                    ]
                    if all(hasattr(m, n) for n in ns)
                ),
                None,
            )
            if not names:
                continue
            parts = [getattr(m, n) for n in names]
            if not all(
                type(v) is nn.QuantizedLinear and v.get("bias") is None for v in parts
            ):
                continue
            if (
                len(
                    {
                        (v.bits, v.group_size, v.weight.shape[1], v.scales.dtype)
                        for v in parts
                    }
                )
                != 1
            ):
                continue
            group = Group(parts)
            for i, (name, v) in enumerate(zip(names, parts)):
                setattr(m, name, Part(v, group, i))
            count += 1
    if arm == "compile":
        for _, m in list(model.named_modules()):
            names = (
                ("gate_proj", "up_proj", "down_proj")
                if all(hasattr(m, n) for n in ("gate_proj", "up_proj", "down_proj"))
                else ("w1", "w3", "w2")
            )
            if not all(hasattr(m, n) for n in names):
                continue
            if not all(type(getattr(m, n)) is nn.QuantizedLinear for n in names):
                continue
            cls = type(m)
            original = cls.__call__
            if not getattr(cls, "_r4_compiled", False):

                def call(self, x, _orig=original):
                    fn = getattr(self, "_r4_compiled_fn", None)
                    return (
                        fn(x)
                        if fn is not None and x.size // x.shape[-1] == 1
                        else _orig(self, x)
                    )

                cls.__call__ = call
                cls._r4_compiled = True
            else:
                original = cls._r4_compile_original
            cls._r4_compile_original = original
            fn = mx.compile(
                lambda x, _m=m, _orig=original: _orig(_m, x), shapeless=True
            )
            object.__setattr__(m, "_r4_compiled_fn", fn)
            count += 1
    if arm.startswith("ladder"):
        stride = int(arm[6:])
        layers = model.layers if hasattr(model, "layers") else model.model.layers
        for i, m in enumerate(layers):
            cls = type(m)
            if not getattr(cls, "_r4_ladder", False):
                original = cls.__call__

                def call(self, *args, _orig=original, **kw):
                    out = _orig(self, *args, **kw)
                    if getattr(self, "_r4_fire", False) and out.shape[1] == 1:
                        mx.async_eval(out)
                    return out

                cls.__call__ = call
                cls._r4_ladder = True
            m._r4_fire = (i + 1) % stride == 0
    if arm in ("fullnorm", "allfull"):
        from bench.dense_decode.full_norm import install as install_full

        count += install_full(model)
    if arm in ("resnorm", "all"):
        from bench.dense_decode.residual_norm import install as install_norm

        count += install_norm(model)
    if arm in ("gdn", "gdnfull"):
        from rapid_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj

        count = fuse_gdn_in_proj(model)
        if arm == "gdnfull":
            from rapid_mlx.qwen35_fused_gdn_decode import (
                install_qwen35_fused_gdn_decode,
            )

            count += install_qwen35_fused_gdn_decode(model, allow_dense=True)
    return count
