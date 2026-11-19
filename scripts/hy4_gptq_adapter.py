"""Hy4 Flash naming, native-affine recipes and streamed calibration forward."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rapid_mlx.models.hy_v4_flash_config import (
    ModelArgs,
    checkpoint_key,
    dequantize_fp8,
)
from scripts.streamed_gptq_io import (
    TensorReader,
    load_boundary,
    mx_module,
    save_boundary,
)


@dataclass(frozen=True)
class Recipe:
    name: str = "hy4-gptq-v1"
    keep_embeddings_bf16: bool = False
    gate_up_3bit_layers: tuple[int, ...] = ()

    def __post_init__(self):
        if self.name not in ("hy4-gptq-v1", "hy4-gptq-lowbit"):
            raise ValueError("unknown Hy4 recipe")
        if type(self.keep_embeddings_bf16) is not bool or any(
            type(i) is not int or i < 0 for i in self.gate_up_3bit_layers
        ):
            raise ValueError("invalid recipe override")
        if len(set(self.gate_up_3bit_layers)) != len(self.gate_up_3bit_layers):
            raise ValueError("duplicate promoted layers")
        if self.name == "hy4-gptq-v1" and self.gate_up_3bit_layers:
            raise ValueError("layer promotions apply only to lowbit")

    def spec(self, name):
        """Return (bits, group, method); None means preserve source precision."""
        name = checkpoint_key(name)
        if name is None or name.endswith("_scale_inv"):
            return None
        if ".switch_mlp." in name and name.endswith(".weight"):
            if self.name == "hy4-gptq-v1":
                bits = 4
            else:
                layer = int(name.split(".")[2])
                bits = (
                    3
                    if name.endswith(".down_proj.weight")
                    or layer in self.gate_up_3bit_layers
                    else 2
                )
            return bits, 64, "gptq"
        if name.endswith(".multi_head_embedding.embedding.weight"):
            return 4, 32, "rtn"
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return None if self.keep_embeddings_bf16 else (8, 64, "rtn")
        if not name.endswith(".weight") or any(
            part in name
            for part in (
                "hc_",
                "norm",
                ".mlp.gate.",
                ".indexer.key_gate_projection.",
            )
        ):
            return None
        if ".engram_layer." in name:
            return None
        if ".self_attn." in name or ".mlp." in name:
            return 8, 64, "rtn"
        raise ValueError(f"unclassified text weight: {name}")


class Hy4Adapter:
    """Source reads are bounded by a matrix, one expert, or an Engram row chunk."""

    def __init__(self, source, recipe=Recipe()):
        self.source = Path(source)
        self.config = json.loads((self.source / "config.json").read_text())
        if self.config.get("model_type") != "hy_v4_flash" or self.config.get(
            "quantization"
        ):
            raise ValueError("expected an unquantized Hy4 source")
        self.args = ModelArgs.from_dict(self.config)
        self.recipe = recipe
        if any(i >= self.args.num_hidden_layers for i in recipe.gate_up_3bit_layers):
            raise ValueError("promoted layer outside model")
        self.reader = TensorReader(source)
        self.names = {}
        for original in self.reader.index.tensors:
            target = checkpoint_key(original)
            if target is None or target.endswith("_scale_inv"):
                continue
            targets = [(target, None)]
            if ".switch_mlp.gate_up_proj.weight" in target:
                targets = [
                    (target.replace("gate_up_proj", p), half)
                    for half, p in enumerate(("gate_proj", "up_proj"))
                ]
            for name, half in targets:
                if name in self.names:
                    raise ValueError("duplicate normalized source tensor")
                self.names[name] = original, half
                spec = recipe.spec(name)
                if spec and self.shape(name)[-1] % spec[1]:
                    raise ValueError(f"nonintegral affine group: {name}")
        self.block = tuple(
            self.config.get(
                "quantization_config",
                self.config.get("text_config", {}).get("quantization_config", {}),
            ).get("weight_block_size", [128, 128])
        )
        if len(self.block) != 2 or any(type(v) is not int or v < 1 for v in self.block):
            raise ValueError("invalid FP8 block size")

    def shape(self, name):
        original, half = self.names[name]
        shape = list(self.reader.index.tensors[original].shape)
        if half is not None:
            if len(shape) != 3 or shape[-2] % 2:
                raise ValueError("invalid packed gate/up bank")
            shape[-2] //= 2
        return tuple(shape)

    def names_for(self, layer=None):
        return [
            n
            for n in self.names
            if (int(m[1]) if (m := re.match(r"model\.layers\.(\d+)\.", n)) else None)
            == layer
        ]

    def tensor(self, name, *, expert=None, rows=None):
        import mlx.core as mx

        original, half = self.names[name]
        t = self.reader.index.tensors[original]
        if len(t.shape) == 3 and expert is None:
            raise ValueError("expert banks must be read one expert at a time")
        if rows is not None:
            start, stop = rows
            if not 0 <= start < stop <= self.shape(name)[-2]:
                raise ValueError("invalid canonical row slice")
        selected_rows = rows
        if half is not None:
            width = t.shape[-2] // 2
            start, stop = rows or (0, width)
            selected_rows = (start + half * width, stop + half * width)
        if t.dtype not in ("F8_E4M3", "F8_E4M3FN"):
            return self.reader.array(original, expert=expert, rows=selected_rows)
        scale_name = original + "_scale_inv"
        if scale_name not in self.reader.index.tensors:
            raise ValueError("missing FP8 scales")
        scale_t = self.reader.index.tensors[scale_name]
        # Align a partial row read with the publisher's FP8 blocks.
        row_scaled = scale_t.shape == t.shape[:-1]
        expected_scale_shape = (
            *t.shape[:-2],
            (t.shape[-2] + self.block[0] - 1) // self.block[0],
            (t.shape[-1] + self.block[1] - 1) // self.block[1],
        )
        if not row_scaled and scale_t.shape != expected_scale_shape:
            raise ValueError("invalid FP8 block scale shape")
        start, stop = selected_rows or (0, t.shape[-2])
        aligned = start if row_scaled else start // self.block[0] * self.block[0]
        raw = self.reader.raw(original, expert=expert, rows=(aligned, stop))
        scale_rows = (
            (aligned, stop)
            if row_scaled
            else (aligned // self.block[0], (stop + self.block[0] - 1) // self.block[0])
        )
        scale = np.array(
            self.reader.array(scale_name, expert=expert, rows=scale_rows).astype(
                mx.float32
            )
        )
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid FP8 scales")
        decoded = dequantize_fp8(raw, scale, self.block)[start - aligned :]
        return mx.array(decoded).astype(mx.bfloat16)

    def cache(self):
        from mlx_lm.models.cache import CacheList, KVCache

        return CacheList(KVCache(), KVCache(), KVCache())

    def forward(self, block, hidden, tokens):
        import mlx.core as mx

        return block(hidden, mx.array(tokens[None]), self.cache())

    def layer(self, index):
        from rapid_mlx.models.hy_v4_flash import DecoderLayer

        block = DecoderLayer(self.args, index)
        if self.args.mlp_layer_types[index] == "sparse":
            block.mlp.switch_mlp = None
        return block

    def routed_inputs(self, block, x):
        import mlx.core as mx

        from rapid_mlx.models.hy_v4_flash_ops import route

        indices, weights = route(
            block.mlp.gate(x.astype(mx.float32)),
            block.mlp.expert_bias,
            self.args.num_experts_per_tok,
            self.args.norm_topk_prob,
            self.args.routed_scaling_factor,
            mx,
        )
        latent = block.mlp.routed_expert_down_proj(x)
        mx.eval(latent, indices, weights)
        return (
            latent.reshape(-1, latent.shape[-1]),
            indices.reshape(-1, indices.shape[-1]),
            weights.reshape(-1, weights.shape[-1]),
        )

    def intermediate(self, gate, up, x):
        from rapid_mlx.models.hy_v4_flash import ClampedSwiGLU

        return ClampedSwiGLU(self.args.swiglu_limit)(x @ up.T, x @ gate.T)


def parent_at(module, path):
    parts = path.split(".")
    for part in parts[:-1]:
        module = (
            module[int(part)] if isinstance(module, list) else getattr(module, part)
        )
    return module, parts[-1]


def assign(module, path, value):
    parent, key = parent_at(module, path)
    if isinstance(parent, list):
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def packed(reader, name):
    base = name.removesuffix(".weight")
    return tuple(
        reader.array(base + "." + suffix) for suffix in ("weight", "scales", "biases")
    )


def linear_value(reader, name, spec, *, expert=None):
    mx = mx_module()
    if spec is None:
        return reader.array(name, expert=expert)
    base = name.removesuffix(".weight")
    values = tuple(
        reader.array(base + "." + s, expert=expert)
        for s in ("weight", "scales", "biases")
    )
    return mx.dequantize(*values, bits=spec[0], group_size=spec[1])


def load_layer(adapter, layer, reader=None):
    """Instantiate exactly one layer; routed banks are replaced before evaluation."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    mx = mx_module()
    block = adapter.layer(layer)
    prefix = f"model.layers.{layer}."
    for name in adapter.names_for(layer):
        if ".switch_mlp." in name:
            continue
        local = name.removeprefix(prefix)
        spec = adapter.recipe.spec(name) if reader else None
        if spec:
            parent, key = parent_at(block, local.removesuffix(".weight"))
            old = parent[int(key)] if isinstance(parent, list) else getattr(parent, key)
            shape = adapter.shape(name)
            if isinstance(old, nn.Embedding):
                module = nn.QuantizedEmbedding(
                    shape[0], shape[1], bits=spec[0], group_size=spec[1]
                )
            else:
                module = nn.QuantizedLinear(
                    shape[1],
                    shape[0],
                    bias=getattr(old, "bias", None) is not None,
                    bits=spec[0],
                    group_size=spec[1],
                )
                if getattr(old, "bias", None) is not None:
                    module.bias = old.bias
            module.weight, module.scales, module.biases = packed(reader, name)
            assign(block, local.removesuffix(".weight"), module)
        elif reader is None and name.endswith(".multi_head_embedding.embedding.weight"):
            assign(
                block, local.removesuffix(".weight"), source_embedding(adapter, name)
            )
        else:
            value = adapter.tensor(name) if reader is None else reader.array(name)
            if any(
                p in name
                for p in (
                    "hc_",
                    "expert_bias",
                    "mlp.gate.weight",
                    ".indexer.key_",
                    ".indexer.intra_group_ape",
                )
            ) and mx.issubdtype(value.dtype, mx.floating):
                value = value.astype(mx.float32)
            assign(block, local, value)
    # Reject incomplete sources rather than silently evaluating random weights.
    expected = {
        prefix + key
        for key, _ in tree_flatten(block.parameters())
        if "switch_mlp" not in key
    }
    available = set(adapter.names)
    if reader:
        available = set(reader.index.tensors)
    if expected - available:
        raise ValueError(f"missing layer tensors: {sorted(expected - available)}")
    if adapter.args.mlp_layer_types[layer] == "sparse":
        block.mlp.switch_mlp = disk_switch(adapter, layer, reader)
    return block


def source_embedding(adapter, name):
    import mlx.nn as nn

    mx = mx_module()

    class SelectedRows(nn.Module):
        def __call__(self, ids):
            unique, inverse = np.unique(np.array(ids), return_inverse=True)
            rows = mx.concatenate(
                [adapter.tensor(name, rows=(int(i), int(i) + 1)) for i in unique]
            )
            return rows[mx.array(inverse)].reshape(*ids.shape, rows.shape[-1])

    return SelectedRows()


def disk_switch(adapter, layer, reader):
    import mlx.nn as nn

    mx = mx_module()

    class DiskSwitch(nn.Module):
        def __call__(self, x, indices):
            raw_indices = np.array(indices).reshape(-1, indices.shape[-1])
            flat = x.reshape(-1, x.shape[-1])
            output = mx.zeros((flat.shape[0], indices.shape[-1], x.shape[-1]), x.dtype)
            base = f"model.layers.{layer}.mlp.switch_mlp."
            for expert in np.unique(raw_indices):
                positions, slots = np.where(raw_indices == expert)
                selected = flat[mx.array(positions)]
                values = []
                for p in ("gate_proj", "up_proj", "down_proj"):
                    name = base + p + ".weight"
                    value = (
                        adapter.tensor(name, expert=int(expert))
                        if reader is None
                        else linear_value(
                            reader, name, adapter.recipe.spec(name), expert=int(expert)
                        )
                    )
                    values.append(value)
                gate, up, down = values
                result = adapter.intermediate(gate, up, selected) @ down.T
                output[mx.array(positions), mx.array(slots)] = result.astype(x.dtype)
                mx.eval(output)
            return output.reshape(*indices.shape, x.shape[-1])

    return DiskSwitch()


def initial_hidden(adapter, tokens, reader=None):
    mx = mx_module()
    name = "model.embed_tokens.weight"
    if reader:
        spec = adapter.recipe.spec(name)
        if spec:
            base = name.removesuffix(".weight")
            values = [
                reader.array(base + "." + s) for s in ("weight", "scales", "biases")
            ]
            x = mx.dequantize(
                *(v[mx.array(tokens)] for v in values), bits=spec[0], group_size=spec[1]
            )
        else:
            x = reader.array(name)[mx.array(tokens)]
    else:
        x = source_embedding(adapter, name)(mx.array(tokens))
    return mx.broadcast_to(
        x[None, :, None],
        (1, len(tokens), adapter.args.hc_mult, adapter.args.hidden_size),
    ).astype(mx.bfloat16)


def logits(adapter, windows, work, layers, reader=None, check=lambda: None):
    """Layer-major teacher forcing; only one decoder layer is instantiated."""
    import mlx.nn as nn

    from rapid_mlx.models.hy_v4_flash import HCHead

    mx = mx_module()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    paths = [work / f"window-{i}.safetensors" for i in range(len(windows))]
    for path, tokens in zip(paths, windows, strict=True):
        save_boundary(path, initial_hidden(adapter, tokens, reader))
    for layer in layers:
        block = load_layer(adapter, layer, reader)
        for path, tokens in zip(paths, windows, strict=True):
            hidden = adapter.forward(block, load_boundary(path), tokens)
            save_boundary(path, hidden)
            del hidden
            check()
        del block
        gc.collect()
        mx.clear_cache()
    head = HCHead(adapter.args)
    norm = nn.RMSNorm(adapter.args.hidden_size, eps=adapter.args.rms_norm_eps)
    for name in adapter.names_for(None):
        if name.startswith("model.hc_head."):
            value = reader.array(name) if reader else adapter.tensor(name)
            assign(head, name.removeprefix("model.hc_head."), value.astype(mx.float32))
    norm.weight = (
        reader.array("model.norm.weight")
        if reader
        else adapter.tensor("model.norm.weight")
    )
    name = (
        "model.embed_tokens.weight"
        if adapter.args.tie_word_embeddings
        else "lm_head.weight"
    )
    for path in paths:
        hidden = norm(head(load_boundary(path)))[0].astype(mx.float32)
        # Compute vocabulary rows in chunks; full logits are needed for KL.
        output = []
        for start in range(0, adapter.args.vocab_size, 4096):
            stop = min(adapter.args.vocab_size, start + 4096)
            if reader:
                spec = adapter.recipe.spec(name)
                if spec:
                    base = name.removesuffix(".weight")
                    triple = [
                        reader.array(base + "." + s, rows=(start, stop))
                        for s in ("weight", "scales", "biases")
                    ]
                    weight = mx.dequantize(*triple, bits=spec[0], group_size=spec[1])
                else:
                    weight = reader.array(name, rows=(start, stop))
            else:
                weight = adapter.tensor(name, rows=(start, stop))
            result = hidden @ weight.astype(mx.float32).T
            output.append(np.array(result))
            check()
        yield np.concatenate(output, axis=-1)
