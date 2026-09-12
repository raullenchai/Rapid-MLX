from __future__ import annotations

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


def test_disk_engram_prefetch_matches_requested_rows(tmp_path):
    resident, disk, _keys, _path = _modules(tmp_path)
    indices = mx.array([[[1, 3, 1], [7, 3, 2]]], mx.int32)

    disk.prefetch(indices)
    expected = resident(indices)
    actual = disk(indices)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
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
