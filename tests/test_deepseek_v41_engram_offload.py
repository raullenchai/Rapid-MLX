from __future__ import annotations

import struct
from concurrent.futures import Future

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from vllm_mlx.models.deepseek_v41_native.engram import (
    DiskQuantizedEngramEmbedding,
    QuantizedEngramEmbedding,
)


def _modules(tmp_path, *, cache_rows=4):
    rows, dim, group_size, bits = 8, 64, 32, 2
    logical = mx.arange(rows * dim, dtype=mx.float32).reshape(rows, dim) / 97
    weight, scales, biases = mx.quantize(logical, group_size=group_size, bits=bits)
    scales = scales.astype(mx.bfloat16)
    biases = biases.astype(mx.bfloat16)
    keys = {
        "weight": "layers.1.engram.embed.weight",
        "scales": "layers.1.engram.embed.scales",
        "biases": "layers.1.engram.embed.biases",
    }
    path = tmp_path / "table.safetensors"
    mx.save_safetensors(
        str(path),
        {
            keys["weight"]: weight,
            keys["scales"]: scales,
            keys["biases"]: biases,
        },
    )
    resident = QuantizedEngramEmbedding(rows, dim, group_size, bits)
    resident.weight = weight
    resident.scales = scales
    resident.biases = biases
    disk = DiskQuantizedEngramEmbedding(
        path,
        weight_key=keys["weight"],
        scales_key=keys["scales"],
        biases_key=keys["biases"],
        num_embeddings=rows,
        dim=dim,
        group_size=group_size,
        bits=bits,
        cache_rows=cache_rows,
    )
    return resident, disk, keys, path


def test_disk_engram_matches_resident_affine_rows_and_caches_repeats(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path)
    indices = mx.array([[[1, 3, 1], [7, 3, 2]]], mx.int32)

    expected = resident(indices)
    actual = disk(indices)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert disk.cache_misses == 4
    assert disk.cache_hits == 2

    repeated = disk(mx.array([1, 3, 2], mx.int32))
    mx.eval(repeated)
    assert disk.cache_misses == 4
    assert disk.cache_hits == 5


def test_disk_engram_prefetch_matches_requested_rows(tmp_path, monkeypatch):
    resident, disk, _keys, _path = _modules(tmp_path)
    indices = mx.array([[[1, 3, 1], [7, 3, 2]]], mx.int32)
    gather_calls = 0
    original = disk._gather_rows

    def counted_gather(flat):
        nonlocal gather_calls
        gather_calls += 1
        return original(flat)

    monkeypatch.setattr(disk, "_gather_rows", counted_gather)

    disk.prefetch(indices)
    assert disk._pending is not None
    disk._pending[1].result()
    expected = resident(indices)
    actual = disk(indices)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert gather_calls == 1
    assert disk.cache_misses == 4
    assert disk.cache_hits == 2


def test_disk_engram_prefetch_mismatch_falls_back_to_requested_rows(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path)
    disk.prefetch(mx.array([1, 3], mx.int32))
    requested = mx.array([2, 7], mx.int32)

    expected = resident(requested)
    actual = disk(requested)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_disk_engram_discards_failed_stale_prefetch(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path)
    disk.prefetch(np.array([8], dtype=np.int64))
    requested = mx.array([2, 7], mx.int32)

    expected = resident(requested)
    actual = disk(requested)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_disk_engram_replaces_failed_prefetch_with_current_request(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path)
    disk.prefetch(np.array([8], dtype=np.int64))
    assert disk._pending is not None
    with pytest.raises(IndexError, match="outside"):
        disk._pending[1].result()
    requested = mx.array([2, 7], mx.int32)

    disk.prefetch(requested)
    expected = resident(requested)
    actual = disk(requested)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_disk_engram_gather_can_exceed_lru_capacity(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path, cache_rows=1)
    indices = mx.array([1, 3, 2], mx.int32)

    expected = resident(indices)
    actual = disk(indices)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert len(disk._cache) == 1


def test_disk_engram_bounds_close_and_cache_validation(tmp_path):
    _resident, disk, _keys, path = _modules(tmp_path, cache_rows=0)
    with pytest.raises(IndexError, match="outside"):
        disk(mx.array([8], mx.int32))
    disk.close()
    disk.close()
    with pytest.raises(RuntimeError, match="closed"):
        disk(mx.array([0], mx.int32))
    with pytest.raises(ValueError, match="cache_rows"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key="layers.1.engram.embed.weight",
            scales_key="layers.1.engram.embed.scales",
            biases_key="layers.1.engram.embed.biases",
            num_embeddings=8,
            dim=64,
            group_size=32,
            bits=2,
            cache_rows=-1,
        )
    with pytest.raises(ValueError, match="unsupported affine"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key="layers.1.engram.embed.weight",
            scales_key="layers.1.engram.embed.scales",
            biases_key="layers.1.engram.embed.biases",
            num_embeddings=8,
            dim=64,
            group_size=32,
            bits=5,
        )


def test_disk_engram_bypass_paths_handle_empty_and_uncached_rows(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path, cache_rows=0)

    empty = disk(mx.array([], mx.int32))
    indices = mx.array([1, 3, 1], mx.int32)
    actual = disk(indices)
    expected = resident(indices)
    mx.eval(empty, actual, expected)

    assert empty.shape == (0, 64)
    assert mx.array_equal(actual, expected).item()
    assert disk.cache_misses == 2
    assert disk.cache_hits == 0


def test_disk_engram_closed_views_and_prefetch_fail_cleanly(tmp_path):
    _resident, disk, _keys, _path = _modules(tmp_path)
    disk.close()

    with pytest.raises(RuntimeError, match="closed"):
        disk._views()
    with pytest.raises(RuntimeError, match="closed"):
        disk.prefetch(np.array([1], dtype=np.int64))


def test_disk_engram_close_drains_pending_prefetch(tmp_path):
    _resident, disk, _keys, _path = _modules(tmp_path)
    disk.prefetch(np.array([1, 3], dtype=np.int64))
    assert disk._pending is not None
    pending = disk._pending[1]

    disk.close()

    assert pending.done()
    assert disk._executor is None
    with pytest.raises(RuntimeError, match="closed"):
        disk(mx.array([1], mx.int32))


def test_disk_engram_rejects_header_contract_mismatch(tmp_path):
    _resident, disk, keys, path = _modules(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key=keys["weight"],
            scales_key=keys["scales"],
            biases_key=keys["biases"],
            num_embeddings=7,
            dim=64,
            group_size=32,
            bits=2,
        )

    with pytest.raises(ValueError, match="byte length"):
        disk._tensor_view(
            {
                "outside": {
                    "dtype": "U32",
                    "shape": [8, 4],
                    "data_offsets": [len(disk._mapping), len(disk._mapping) + 128],
                }
            },
            "outside",
            dtype="U32",
            numpy_dtype=np.dtype("<u4"),
            shape=(8, 4),
        )


def test_disk_engram_rejects_oversized_header_before_read(tmp_path):
    path = tmp_path / "oversized.safetensors"
    path.write_bytes(struct.pack("<Q", 100_000_001))

    with pytest.raises(ValueError, match="header length"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key="weight",
            scales_key="scales",
            biases_key="biases",
            num_embeddings=8,
            dim=64,
            group_size=32,
            bits=2,
        )


def test_disk_engram_rejects_truncated_header_prefix(tmp_path):
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"short")

    with pytest.raises(ValueError, match="truncated"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key="weight",
            scales_key="scales",
            biases_key="biases",
            num_embeddings=8,
            dim=64,
            group_size=32,
            bits=2,
        )


def test_disk_engram_constructor_preserves_error_after_partial_view(
    tmp_path, monkeypatch
):
    _resident, disk, keys, path = _modules(tmp_path)
    disk.close()
    calls = 0
    original = DiskQuantizedEngramEmbedding._tensor_view

    def fail_after_first_view(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("malformed scales")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        DiskQuantizedEngramEmbedding, "_tensor_view", fail_after_first_view
    )
    with pytest.raises(ValueError, match="malformed scales"):
        DiskQuantizedEngramEmbedding(
            path,
            weight_key=keys["weight"],
            scales_key=keys["scales"],
            biases_key=keys["biases"],
            num_embeddings=8,
            dim=64,
            group_size=32,
            bits=2,
        )


def test_disk_engram_rejects_overlapping_tensor_ranges():
    with pytest.raises(ValueError, match="non-contiguous"):
        DiskQuantizedEngramEmbedding._validate_header_ranges(
            {
                "weight": {"data_offsets": [0, 64]},
                "scales": {"data_offsets": [32, 96]},
                "__metadata__": {"format": "mlx"},
            },
            data_size=96,
        )


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ([], "invalid safetensors header"),
        ({"weight": []}, "tensor entry"),
        ({"weight": {"data_offsets": "0,1"}}, "tensor offsets"),
        ({"weight": {"data_offsets": [-1, 0]}}, "tensor offsets"),
    ],
)
def test_disk_engram_rejects_malformed_header_entries(header, message):
    with pytest.raises(ValueError, match=message):
        DiskQuantizedEngramEmbedding._validate_header_ranges(header, data_size=1)


def test_disk_engram_rejects_missing_tensor(tmp_path):
    _resident, disk, _keys, _path = _modules(tmp_path)

    with pytest.raises(ValueError, match="missing Engram tensor"):
        disk._tensor_view(
            {},
            "missing",
            dtype="U32",
            numpy_dtype=np.dtype("<u4"),
            shape=(8, 4),
        )


def test_disk_engram_discard_future_consumes_cancellation():
    future = Future()
    DiskQuantizedEngramEmbedding._discard_future(future)

    assert future.cancelled()


@pytest.mark.parametrize(
    ("header", "data_size", "message"),
    [
        ({"weight": {"data_offsets": [1, 2]}}, 2, "non-contiguous"),
        ({"weight": {"data_offsets": [0, 1]}}, 2, "do not cover"),
    ],
)
def test_disk_engram_rejects_incomplete_tensor_coverage(header, data_size, message):
    with pytest.raises(ValueError, match=message):
        DiskQuantizedEngramEmbedding._validate_header_ranges(header, data_size)


def test_disk_engram_allows_empty_safetensors_tensor_ranges():
    DiskQuantizedEngramEmbedding._validate_header_ranges(
        {
            "empty": {"data_offsets": [0, 0]},
            "weight": {"data_offsets": [0, 1]},
        },
        data_size=1,
    )
