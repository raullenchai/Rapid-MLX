# SPDX-License-Identifier: Apache-2.0
"""Read-only Qwen4 q4/g32 PLE sidecars; metadata/NumPy only, no MLX import.

Layout and single-round BF16 dequantization follow the lab's unified
qwen4_ple_nvme mechanism. This adapter deliberately retains Rapid's hash path.
Artifact binding uses the index digest, complete tensor geometry, every shard's
edge rows and a random row sample; it is not an exhaustive content digest.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import threading

import numpy as np

ADAPTER_VERSION = 1
_LOAD_SOURCE = ContextVar('rapid_qwen4_ple_load_source', default=None)


@contextmanager
def _bound_load_source(model_path):
    token = _LOAD_SOURCE.set(Path(model_path).resolve())
    try:
        yield
    finally:
        _LOAD_SOURCE.reset(token)


def require_load_source(model_path):
    if _LOAD_SOURCE.get() != Path(model_path).resolve():
        raise ValueError('PLE source must be bound by load_file_backed_qwen4; arbitrary source overrides are refused')


def _positive_int(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if zero else "positive"} integer')
    return value


def load_manifest(sidecar_path) -> dict:
    path = Path(sidecar_path)
    if fnmatch.fnmatch(path.name, 'model*.safetensors'):
        raise ValueError('PLE sidecar must not match the model weight-file glob')
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if manifest.get('format') != 'qwen4-ple-rows' or manifest.get('version') != 1:
        raise ValueError('unsupported PLE sidecar manifest format/version')
    if {k: manifest.get(k) for k in ('bits', 'group_size', 'mode')} != dict(bits=4, group_size=32, mode='affine'):
        raise ValueError('PLE sidecar requires affine q4/group32')
    dims = _positive_int(manifest.get('dims'), 'dims')
    if dims % 32:
        raise ValueError('PLE dimensions must be divisible by32')
    geometry = dict(weight_bytes=dims // 2, scales_bytes=dims // 16, biases_bytes=dims // 16,
                    row_bytes=dims // 2 + dims // 8)
    for name, expected in geometry.items():
        if manifest.get(name) != expected:
            raise ValueError(f'invalid PLE {name}: expected {expected}')
    shards = _positive_int(manifest.get('num_shards'), 'num_shards')
    rows = _positive_int(manifest.get('rows_per_shard'), 'rows_per_shard')
    if manifest.get('total_rows') != rows * shards:
        raise ValueError('invalid PLE total row count')
    offset = _positive_int(manifest.get('data_offset'), 'data_offset', zero=True)
    digests = manifest.get('shard_sha256', [])
    if len(digests) != shards or any(not isinstance(d, str) or not re.fullmatch('[0-9a-f]{64}', d) for d in digests):
        raise ValueError('PLE manifest requires one SHA256 per shard')
    if path.stat().st_size != offset + rows * shards * geometry['row_bytes']:
        raise ValueError('PLE sidecar file size differs from manifest')
    return manifest


def _source_refs(model_path: Path, manifest: dict, weight_map: dict) -> dict:
    prefix = manifest['tensor_prefix']
    parts = {'weight': ('U32', manifest['dims'] // 8, 4),
             'scales': ('BF16', manifest['dims'] // 32, 2),
             'biases': ('BF16', manifest['dims'] // 32, 2)}
    expected = {f'{prefix}.shard_{i}.{part}' for i in range(manifest['num_shards']) for part in parts}
    present = {key for key in weight_map if key.startswith(prefix + '.')}
    if present != expected:
        raise ValueError('PLE source index does not contain exactly the expected shard triples')
    headers, refs = {}, {}
    for name in sorted(expected):
        source = (model_path / weight_map[name]).resolve()
        if not source.is_relative_to(model_path.resolve()):
            raise ValueError('PLE source tensor path escapes model directory')
        if source not in headers:
            with source.open('rb') as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError('truncated safetensors header length')
                size = struct.unpack('<Q', raw)[0]
                if size > min(source.stat().st_size - 8, 64 * 1024**2):
                    raise ValueError('invalid safetensors header length')
                headers[source] = (json.loads(stream.read(size)), 8 + size)
        header, data_start = headers[source]
        info = header.get(name, {})
        shard, part = name.split('.shard_')[1].split('.')
        dtype, columns, itemsize = parts[part]
        shape = [manifest['rows_per_shard'], columns]
        if info.get('dtype') != dtype or info.get('shape') != shape:
            raise ValueError(f'PLE source dtype/shape mismatch at {name}')
        offsets = info.get('data_offsets', [])
        if len(offsets) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets):
            raise ValueError(f'invalid source offsets at {name}')
        start, end = offsets
        row_bytes = columns * itemsize
        if start < 0 or end - start != shape[0] * row_bytes or data_start + end > source.stat().st_size:
            raise ValueError(f'PLE source byte range mismatch at {name}')
        refs[int(shard), part] = (source, data_start + start, row_bytes)
    return refs


def validate_artifact(model_path, sidecar_path, *, random_rows=256) -> dict:
    """Validate before constructing a model or importing MLX."""
    model_path, sidecar_path = Path(model_path).resolve(), Path(sidecar_path).resolve()
    manifest = load_manifest(sidecar_path)
    index_bytes = (model_path / 'model.safetensors.index.json').read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != manifest.get('source_index_sha256'):
        raise ValueError('PLE sidecar source index digest mismatch')
    config = json.loads((model_path / 'config.json').read_text())
    text = config.get('text_config', config)
    layer_ids = text.get('ple_layer_ids', [])
    if config.get('model_type') != 'qwen4_exp' or len(layer_ids) != 1:
        raise ValueError('this PLE loader requires exactly one Qwen4 PLE layer')
    prefix = f'language_model.model.layers.{layer_ids[0] - 1}.ple.ple_embedding.ngram_embedding'
    heads = (text.get('ngram_size', 3) - 1) * text.get('heads_per_ngram', 8)
    embed_dim = text.get('ple_embed_dim')
    if not isinstance(embed_dim, int) or heads <= 0 or embed_dim % heads:
        raise ValueError('invalid PLE head/embedding configuration')
    if manifest['tensor_prefix'] != prefix or manifest['dims'] != embed_dim // heads or manifest['num_shards'] != text.get('split_ngram_parts', 128):
        raise ValueError('PLE manifest geometry/prefix differs from model config')
    refs = _source_refs(model_path, manifest, json.loads(index_bytes)['weight_map'])
    _positive_int(random_rows, 'random_rows', zero=True)
    if random_rows > 4096:
        raise ValueError('PLE random validation sample exceeds4096 rows')
    rows_per_shard = manifest['rows_per_shard']
    picks = {i * rows_per_shard + edge for i in range(manifest['num_shards']) for edge in (0, rows_per_shard - 1)}
    picks.update(map(int, np.random.default_rng().integers(0, manifest['total_rows'], size=random_rows)))
    handles = {}
    try:
        with sidecar_path.open('rb') as sidecar:
            for row in sorted(picks):
                shard, local = divmod(row, rows_per_shard)
                pieces = []
                for part in ('weight', 'scales', 'biases'):
                    path, start, width = refs[shard, part]
                    if path not in handles:
                        handles[path] = path.open('rb')
                    stream = handles[path]
                    stream.seek(start + local * width)
                    pieces.append(stream.read(width))
                sidecar.seek(manifest['data_offset'] + row * manifest['row_bytes'])
                if sidecar.read(manifest['row_bytes']) != b''.join(pieces):
                    raise ValueError(f'PLE sidecar/source content mismatch at row {row}')
                # Nonfinite quantization parameters are outside the supported
                # single-round parity domain, including on validated edge rows.
                dequant_rows_numpy(np.frombuffer(b''.join(pieces), dtype=np.uint8)[None, :], manifest['dims'])
    finally:
        for stream in handles.values():
            stream.close()
    return {'adapter_version': ADAPTER_VERSION, 'manifest': manifest, 'checked_rows': len(picks),
            'content_validation': 'every shard edge plus random rows; not exhaustive',
            'source_index_sha256': manifest['source_index_sha256'], 'sidecar': str(sidecar_path)}


def dequant_rows_numpy(rows: np.ndarray, dims: int) -> np.ndarray:
    """Affine q4/g32 -> BF16 bits with one FP32->BF16 round-to-nearest-even."""
    if dims <= 0 or dims % 32:
        raise ValueError('PLE dequant dimensions must be a positive multiple of32')
    rows = np.asarray(rows)
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != dims // 2 + dims // 8:
        raise ValueError('invalid packed PLE row shape/dtype')
    weight_bytes, group_bytes = dims // 2, dims // 16
    words = np.ascontiguousarray(rows[:, :weight_bytes]).view('<u4')
    scale_bits = np.ascontiguousarray(rows[:, weight_bytes:weight_bytes + group_bytes]).view('<u2')
    bias_bits = np.ascontiguousarray(rows[:, weight_bytes + group_bytes:]).view('<u2')
    scales = (scale_bits.astype(np.uint32) << 16).view(np.float32)
    biases = (bias_bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(scales).all() or not np.isfinite(biases).all():
        raise ValueError('nonfinite PLE quantization parameters')
    shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
    quant = ((words[..., None] >> shifts) & np.uint32(15)).reshape(-1, dims).astype(np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        values = quant * np.repeat(scales, 32, axis=1) + np.repeat(biases, 32, axis=1)
    if not np.isfinite(values).all():
        raise ValueError('PLE affine dequantization overflow')
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    if ((rounded & 0x7F80) == 0x7F80).any():
        raise ValueError('PLE BF16 dequantization overflow')
    return rounded


class PLESidecarReader:
    """Bounded selected-row pread reader; no resident table or MLX arrays."""
    def __init__(self, model_path, sidecar_path, *, random_rows=256, cache_bytes=0):
        _positive_int(cache_bytes, 'cache_bytes', zero=True)
        if cache_bytes > 512 * 1024**2:
            raise ValueError('PLE row cache must not exceed512 MiB')
        before = self._stat_identity(os.stat(sidecar_path))
        self.receipt = validate_artifact(model_path, sidecar_path, random_rows=random_rows)
        self.manifest = self.receipt['manifest']
        self._fd = os.open(sidecar_path, os.O_RDONLY)
        self._identity = self._stat_identity(os.fstat(self._fd))
        if self._identity != before:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError('PLE sidecar changed during validation')
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self.lookups = self.rows = self.unique_rows = self.bytes_read = 0
        self.cache_hits = self.cache_misses = self.cache_evictions = 0
        self.cache_bytes = cache_bytes
        # Charge512 bytes per entry for Python key/bytes/OrderedDict overhead,
        # in addition to payload; conservative on the supported CPython runtime.
        self._cache_charge = self.manifest['row_bytes'] + 512
        self._cache_capacity = cache_bytes // self._cache_charge
        self._cache = OrderedDict()

    @property
    def stats(self):
        return dict(lookups=int(self.lookups), rows=int(self.rows), unique_rows=int(self.unique_rows),
                    bytes_read=int(self.bytes_read), cache_hits=int(self.cache_hits),
                    cache_misses=int(self.cache_misses), cache_evictions=int(self.cache_evictions),
                    cache_rows=len(self._cache), cache_max_bytes=self.cache_bytes,
                    cache_charged_bytes=len(self._cache) * self._cache_charge,
                    cache_payload_bytes=len(self._cache) * self.manifest['row_bytes'])

    @staticmethod
    def _stat_identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def close(self):
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._cache.clear()

    def __del__(self):
        if getattr(self, '_fd', None) is not None:
            os.close(self._fd)
            self._fd = None

    def lookup_bits(self, indices) -> np.ndarray:
        values = np.asarray(indices)
        if values.dtype.kind not in 'iu':
            raise ValueError('PLE indices must be integers')
        if values.size and (values.min() < 0 or values.max() >= self.manifest['total_rows']):
            raise IndexError('PLE index outside vocabulary')
        unique, inverse = np.unique(values.astype(np.int64).reshape(-1), return_inverse=True)
        dims, row_bytes = self.manifest['dims'], self.manifest['row_bytes']
        output = np.empty((unique.size, dims), dtype=np.uint16)
        with self._lock:
            if self._fd is None or os.getpid() != self._pid:
                raise RuntimeError('PLE reader is closed or inherited across fork')
            if self._stat_identity(os.fstat(self._fd)) != self._identity:
                raise RuntimeError('PLE sidecar changed after validation')
            # Only4096 packed rows + dequant scratch are staged at once.
            for begin in range(0, unique.size, 4096):
                batch = unique[begin:begin + 4096]
                packed = np.empty((len(batch), row_bytes), dtype=np.uint8)
                for i, row in enumerate(batch):
                    key = int(row)
                    raw = self._cache.pop(key, None)
                    if raw is None:
                        self.cache_misses += 1
                        raw = os.pread(self._fd, row_bytes, self.manifest['data_offset'] + key * row_bytes)
                        if len(raw) != row_bytes:
                            raise RuntimeError('short PLE sidecar read')
                        self.bytes_read += row_bytes
                    else:
                        self.cache_hits += 1
                    if self._cache_capacity:
                        self._cache[key] = raw
                        while len(self._cache) > self._cache_capacity:
                            self._cache.popitem(last=False)
                            self.cache_evictions += 1
                    packed[i] = np.frombuffer(raw, dtype=np.uint8)
                output[begin:begin + len(batch)] = dequant_rows_numpy(packed, dims)
            if self._stat_identity(os.fstat(self._fd)) != self._identity:
                self._cache.clear()
                raise RuntimeError('PLE sidecar changed during lookup')
            self.lookups += 1
            self.rows += values.size
            self.unique_rows += unique.size
        return output[inverse].reshape(*values.shape, dims)
