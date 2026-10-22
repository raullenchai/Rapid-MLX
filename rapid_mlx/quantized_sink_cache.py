# SPDX-License-Identifier: Apache-2.0
"""Lossless packed prefix handoff for models with quantized sink attention."""

import mlx.core as mx

from . import _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import QuantizedKVCache  # noqa: E402

from .quantized_batch_cache import (  # noqa: E402
    QuantizedBatchKVCache,
    _QuantizableKVCache,
)


class SinkQuantizedKVCache(QuantizedKVCache):
    """Keep packed state through prefix storage, persistence and batch merging."""

    preserve_quantized = True

    def size(self):
        return self.offset

    @classmethod
    def merge(cls, caches):
        first = caches[0]
        return SinkQuantizedBatchKVCache.merge(
            caches, group_size=first.group_size, bits=first.bits
        )


class SinkQuantizableKVCache(_QuantizableKVCache):
    """Fresh full-attention cache whose batch keeps packed prefix state."""

    @classmethod
    def merge(cls, caches):
        first = caches[0]
        return SinkQuantizedBatchKVCache.merge(
            caches, group_size=first.q_group_size, bits=first.q_bits
        )


class SinkQuantizedBatchKVCache(QuantizedBatchKVCache):
    """Use ordinary batch bookkeeping; extract and merge without requantizing."""

    def extract(self, idx):
        cache = SinkQuantizedKVCache(self.group_size, self.bits)
        if self.keys is not None:
            padding = int(self.left_padding[idx].item())
            cache.keys, cache.values = [
                tuple(
                    mx.contiguous(m[idx : idx + 1, :, padding : self._idx])
                    for m in triple
                )
                for triple in (self.keys, self.values)
            ]
            cache.offset = self._idx - padding
        return cache

    @classmethod
    def merge(cls, caches, group_size=64, bits=8):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        padding = [max_length - n for n in lengths]
        obj = cls(padding, group_size, bits)
        if max_length == 0:
            return obj
        packed = [
            c
            if isinstance(c, QuantizedKVCache)
            else c.to_quantized(group_size=group_size, bits=bits)
            for c in caches
        ]
        if any((c.group_size, c.bits) != (group_size, bits) for c in packed):
            raise ValueError(
                "Cannot merge caches with different quantization parameters"
            )
        for name in ("keys", "values"):
            ref = next(getattr(c, name) for c, length in zip(packed, lengths) if length)
            triples = []
            for c, length, pad in zip(packed, lengths, padding):
                triple = getattr(c, name)
                triples.append(
                    [
                        mx.pad(m[..., :length, :], ((0, 0), (0, 0), (pad, 0), (0, 0)))
                        if length
                        else mx.zeros(
                            (1, *r.shape[1:2], max_length, r.shape[-1]), dtype=r.dtype
                        )
                        for m, r in zip(triple or ref, ref)
                    ]
                )
            setattr(
                obj,
                name,
                [mx.concatenate([t[i] for t in triples], axis=0) for i in range(3)],
            )
        obj.offset += max_length
        obj._idx = max_length
        return obj
