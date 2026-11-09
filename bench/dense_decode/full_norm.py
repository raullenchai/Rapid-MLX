# SPDX-License-Identifier: Apache-2.0
"""Experimental residual seams across adjacent decoder layers."""

from bench.dense_decode.residual_norm import seam


def install(model):
    count = 0
    for _, m in list(model.named_modules()):
        cls = type(m)
        if cls.__name__ not in ("Qwen3_5DecoderLayer", "Lfm2DecoderLayer"):
            continue
        if not getattr(cls, "_r4_full_seam", False):
            original = cls.__call__
            if cls.__name__ == "Qwen3_5DecoderLayer":

                def call(
                    self,
                    x,
                    mask=None,
                    cache=None,
                    position_ids=None,
                    position_embeddings=None,
                    _orig=original,
                ):
                    if x.size // x.shape[-1] != 1 or x.shape[-1] > 4096:
                        return _orig(
                            self, x, mask, cache, position_ids, position_embeddings
                        )
                    if self.is_linear:
                        r = self.linear_attn(self.input_layernorm(x), mask, cache)
                    else:
                        r = self.self_attn(
                            self.input_layernorm(x),
                            mask=mask,
                            cache=cache,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                    h, n = seam(x, r, self.post_attention_layernorm)
                    return finish(self, h, self.mlp(n))
            else:

                def call(self, x, mask=None, cache=None, gdn_sink=None, _orig=original):
                    if x.size // x.shape[-1] != 1 or x.shape[-1] > 4096:
                        return _orig(self, x, mask, cache, gdn_sink)
                    if self.is_attention_layer:
                        r = self.self_attn(
                            self.operator_norm(x), mask=mask, cache=cache
                        )
                    else:
                        r = self.conv(
                            self.operator_norm(x),
                            mask=mask,
                            cache=cache,
                            gdn_sink=gdn_sink,
                        )
                    h, n = seam(x, r, self.ffn_norm)
                    return finish(self, h, self.feed_forward(n))

            cls.__call__ = call
            cls._r4_full_seam = True
        count += 1
    layers = model.layers
    final = getattr(model.model, "norm", None)
    if final is None:
        final = model.model.embedding_norm
    next_norms = [
        getattr(layer, "input_layernorm", None) or layer.operator_norm
        for layer in layers[1:]
    ] + [final]
    for layer, norm in zip(layers, next_norms):
        object.__setattr__(layer, "_r4_next_norm", norm)
    import mlx.nn as nn

    cls = nn.RMSNorm
    if not getattr(cls, "_r4_cached_norm", False):
        original = cls.__call__

        def cached_norm(self, x):
            record = self.__dict__.pop("_r4_normalized", None)
            if record is not None and record[0] is x:
                return record[1]
            return original(self, x)

        cls.__call__ = cached_norm
        cls._r4_cached_norm = True
    return count


def finish(layer, a, b):
    norm = getattr(layer, "_r4_next_norm", None)
    if norm is None:
        return a + b
    out, z = seam(a, b, norm)
    object.__setattr__(norm, "_r4_normalized", (out, z))
    return out
