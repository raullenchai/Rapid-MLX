# SPDX-License-Identifier: Apache-2.0
"""Rapid's parameter-free PLE adapter, installed after sanitize before quantize."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .qwen4_ple_sidecar import (ADAPTER_VERSION, PLESidecarReader, _bound_load_source, own_load_reader, require_load_source, validate_artifact)


class FileBackedPLEEmbedding(nn.Module):
    is_file_backed = True
    adapter_version = ADAPTER_VERSION

    def __init__(self, reader: PLESidecarReader):
        super().__init__()
        self._reader = reader
        self.rows_per_shard = reader.manifest['rows_per_shard']
        self.dims = reader.manifest['dims']

    @property
    def stats(self):
        return self._reader.stats

    def __call__(self, indices):
        # Rapid retains its existing exact MLX n-gram hashing. Only the table
        # lookup changes; all selected ids are validated before file access.
        mx.eval(indices)
        bits = self._reader.lookup_bits(np.asarray(indices))
        return mx.array(bits).view(mx.bfloat16)

    def close(self):
        self._reader.close()


def install_file_backed_ple(model, weights: dict, sidecar_path, model_path, *, cache_bytes=0):
    """Remove exactly one validated set of sanitized shards.N triples.

    This runs from Model.sanitize before mlx-lm quantizes/evaluates parameters.
    No original shard module or tensor is retained by the replacement reader.
    """
    from .qwen4_exp import ShardedEmbedding

    require_load_source(model_path)
    matches = [(index, layer.ple.ple_embedding)
               for index, layer in enumerate(model.language_model.model.layers)
               if getattr(layer, 'ple', None) is not None]
    if len(matches) != 1:
        raise ValueError('PLE offload requires exactly one Rapid PLE layer')
    index, embedding = matches[0]
    resident = embedding.ngram_embedding
    if not isinstance(resident, ShardedEmbedding):
        raise ValueError('PLE load hook requires an unmodified Rapid ShardedEmbedding')
    reader = PLESidecarReader(model_path, sidecar_path, cache_bytes=cache_bytes)
    try:
        own_load_reader(reader)
        manifest = reader.manifest
        prefix = f'language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding'
        geometry = (resident.rows_per_shard, len(resident.shards), int(resident.shards[0].weight.shape[-1]))
        if prefix != manifest['tensor_prefix'] or geometry != (manifest['rows_per_shard'], manifest['num_shards'], manifest['dims']):
            raise ValueError('PLE sidecar geometry differs from the instantiated Rapid table')
        expected = {}
        for shard in range(manifest['num_shards']):
            if tuple(resident.shards[shard].weight.shape) != (manifest['rows_per_shard'], manifest['dims']):
                raise ValueError('Rapid PLE shard dimensions are inconsistent')
            for part, width, dtype in [('weight', manifest['dims'] // 8, mx.uint32),
                                        ('scales', manifest['dims'] // 32, mx.bfloat16),
                                        ('biases', manifest['dims'] // 32, mx.bfloat16)]:
                key = f'{prefix}.shards.{shard}.{part}'
                value = weights.get(key)
                if value is None or tuple(value.shape) != (manifest['rows_per_shard'], width) or value.dtype != dtype:
                    raise ValueError(f'missing or invalid sanitized PLE tensor {key}')
                expected[key] = True
        present = {key for key in weights if key.startswith(prefix + '.')}
        if present != set(expected):
            raise ValueError('unexpected PLE tensor aliases; refuse partial removal')
        pruned = {key: value for key, value in weights.items() if key not in expected}
        replacement = FileBackedPLEEmbedding(reader)
        if replacement.parameters():
            raise AssertionError('file-backed PLE replacement owns MLX parameters')
        embedding.ngram_embedding = replacement
        model._ple_offload_receipt = dict(reader.receipt, removed_tensors=len(expected), resident_ple_tensors=0,
                                         cache_max_bytes=cache_bytes)
        return pruned
    except BaseException:
        try:
            reader.close()
        except BaseException:
            pass
        raise


def load_file_backed_qwen4(model_path, sidecar_path, *, cache_bytes=0, lazy=False):
    """Production-accessible strict loader; bind PLE to the actual load source.

    Callers supply one model path. They cannot independently override the PLE
    source identity. The context is local to this load/thread and never changes
    mlx-lm classes or global loader functions.
    """
    from mlx_lm.utils import load_model
    from .qwen4_exp import Model, ModelArgs

    if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or not 0 <= cache_bytes <= 512 * 1024**2:
        raise ValueError('PLE row cache must be between0 and512 MiB')
    model_path = Path(model_path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    if os.environ.get('MLX_QWEN4_PLE_NVME'):
        raise ValueError('clear MLX_QWEN4_PLE_NVME for Rapid; use its per-load PLE helper or RAPID_MLX_QWEN4_PLE_NVME')
    validate_artifact(model_path, sidecar_path)
    # Both older and newer mlx-lm loaders honor this override before resolving
    # custom architecture files; force the exact vendored getter on either API.
    options = dict(ple_nvme_sidecar=str(sidecar_path), ple_nvme_model_path=str(model_path),
                   ple_nvme_cache_bytes=cache_bytes, model_file=None)
    with _bound_load_source(model_path):
        model, config = load_model(model_path, strict=True, lazy=lazy, model_config=options,
                                   get_model_classes=lambda config: (Model, ModelArgs))
        if type(model) is not Model or not getattr(model, '_ple_offload_receipt', None):
            raise RuntimeError('PLE load did not return the exact vendored Rapid model with an offload receipt')
    return model, config


def close_file_backed_ple(model):
    for _, module in model.named_modules():
        if isinstance(module, FileBackedPLEEmbedding):
            module.close()
