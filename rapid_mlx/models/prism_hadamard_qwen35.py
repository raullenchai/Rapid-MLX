# SPDX-License-Identifier: MIT
# Adapted from Prism ML's Ternary-Bonsai-2-27B-mlx-2bit runtime at
# 3f926b415992eaa2ae9dd7b573706494d6bbf787 (runtime.py, artifact.py,
# vision_artifact.py). Rapid changes: schema/shape validation, native module
# replacement, and no imports of executable code from the model snapshot.
#
# MIT License
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Data-only loader for Bonsai 2's rotated, affine 2-bit Qwen3.5 pack."""

import math

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten


def _fwht(x, block, signs, *, inverse=False):
    shape, dtype = x.shape, x.dtype
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block))
    x = x.reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """Keep weights packed; rotate activations (or inverse-rotate embeddings)."""

    def __init__(self, arrays, block, signs, embedding):
        super().__init__()
        self.weight, self.scales, self.biases = arrays
        self.block, self.signs, self.embedding = block, signs, embedding

    def __call__(self, x):
        if self.embedding:
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=128,
                    bits=2,
                )
                .reshape(*x.shape, -1)
                .astype(mx.float16)
            )
            return (
                _fwht(out, self.block, self.signs, inverse=True) if self.block else out
            )
        if self.block:
            x = _fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )


def _install_packed(language_model, config, weights):
    modules = dict(language_model.named_modules())
    replacements = []
    seen = set()
    records = config.get("modules")
    if not isinstance(records, list) or not records:
        raise ValueError("Bonsai 2 requires a non-empty packed module manifest")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid Bonsai 2 module manifest record: {record!r}")
        name = record.get("path")
        if not isinstance(name, str):
            raise ValueError(f"Invalid Bonsai 2 module manifest path: {name!r}")
        original = modules.get(name)
        if name in seen or not isinstance(original, (nn.Linear, nn.Embedding)):
            raise ValueError(f"Invalid or duplicate Bonsai 2 module: {name}")
        seen.add(name)
        if record.get("dtype") != "float16" or record.get("embedding") != isinstance(
            original, nn.Embedding
        ):
            raise ValueError(f"Invalid Bonsai 2 module kind or dtype: {name}")
        prefix = "language_model." + name
        try:
            arrays = [
                weights[prefix + "." + suffix]
                for suffix in ("weight", "scales", "biases")
            ]
        except KeyError as exc:
            raise ValueError(
                f"Missing Bonsai 2 packed tensor for {name}: {exc.args[0]}"
            ) from exc
        rows, width = original.weight.shape
        expected = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
        if (
            width % 128
            or [a.shape for a in arrays] != expected
            or arrays[0].dtype != mx.uint32
        ):
            raise ValueError(f"Invalid Bonsai 2 packed tensor shape or dtype: {name}")
        for array in arrays[1:]:
            if (
                array.dtype not in (mx.float16, mx.float32, mx.bfloat16)
                or not mx.all(mx.isfinite(array)).item()
            ):
                raise ValueError(f"Invalid Bonsai 2 affine parameters: {name}")
        block = record.get("block")
        signs = weights.get(prefix + ".signs")
        if block not in (0, 512, 1024, 2048, 4096):
            raise ValueError(f"Unsupported Bonsai 2 Hadamard block: {block}")
        if block:
            if width % block or signs is None or signs.shape != (width,):
                raise ValueError(f"Invalid Bonsai 2 transform dimensions: {name}")
            if not mx.all((signs == 1) | (signs == -1)).item():
                raise ValueError(f"Invalid Bonsai 2 sign vector: {name}")
        elif signs is not None:
            raise ValueError(f"Unexpected Bonsai 2 sign vector: {name}")
        replacements.append(
            (name, Packed(arrays, block, signs, record.get("embedding")))
        )
    language_model.update_modules(tree_unflatten(replacements))


def _build_processor(directory):
    from mlx_vlm.models.qwen3_5 import Qwen3VLProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(str(directory)),
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=(directory / "chat_template.jinja").read_text(),
    )
    processor.detokenizer = load_tokenizer(directory, return_tokenizer=False)(tokenizer)
    eos = getattr(tokenizer, "eos_token_ids", None) or tokenizer.eos_token_id
    processor.stopping_criteria = StoppingCriteria(eos, tokenizer)
    tokenizer.stopping_criteria = processor.stopping_criteria
    return processor


def load(model_path, config):
    """Load schema-v2 weights without running the snapshot's runtime/ Python."""
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig
    from mlx_vlm.utils import get_model_path

    if (
        config.get("schema_version") != 2
        or config.get("model_type") != "prism_hadamard_qwen35"
        or config.get("base_model_type") != "qwen3_5"
        or config.get("tensor_namespace") != "mlx-vlm-qwen3_5"
        or config.get("gdn_activation_layout") != "grouped"
        or config.get("components") != {"text": True, "vision": True, "mtp": False}
        or config.get("quantization")
        != {"bits": 2, "group_size": 128, "mode": "affine"}
    ):
        raise ValueError(
            "Unsupported Bonsai 2 pack: expected schema-v2 Qwen3.5 affine 2-bit/g128"
        )
    directory = get_model_path(model_path)
    model = Model(ModelConfig.from_dict({**config, "model_type": "qwen3_5"}))
    weights = mx.load(str(directory / "model.safetensors"))
    _install_packed(model.language_model, config, weights)
    # The pack already uses MLX tensor layout; do not sanitize/reorder it again.
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, _build_processor(directory)
