"""Tiny CPU-only score artifacts; no checkpoint or calibration corpus required."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_deepseek_v41_reap import _overlap
from scripts.deepseek_v41_merge_calibration import (
    COMPATIBILITY_FIELDS,
    merge_calibration,
    save_scores,
)
from scripts.deepseek_v41_rd_common import file_hash


def metadata(tokens=8):
    return dict(
        schema_version=1,
        model_revision="tiny",
        config_sha256="config",
        checkpoint_config_sha256="config",
        tokenizer_sha256="tokenizer",
        slice_names=["hot", "cold", "general_reference"],
        routing_topk=1,
        keep_experts=2,
        tokens_per_slice=[tokens] * 3,
        score_definitions=dict(
            reap="sum normalized_gate * output_L2",
            man="routed-token mean output_L2",
            msan="routed-token mean output_L2_squared",
        ),
        normalization="raw score max over domains; equal tokens per slice; halves checked separately",
        teacher="source_fp4_fp8_dequantized_bf16_native",
        engram="source_fp8_rowwise",
        execution="layer_major_full_prefill",
        split_overlap_columns=["reap", "man", "msan"],
    )


def raw(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 4, size=(3, 2, 2, 4), dtype=np.int64)
    norms = rng.integers(1, 10, size=counts.shape) * counts
    squared = rng.integers(1, 30, size=counts.shape) * counts
    return norms * 0.25, norms, squared, counts


def write_pass(path, arrays, meta=None):
    meta = copy.deepcopy(meta or metadata())
    save_scores(path, *arrays, meta, meta["keep_experts"], meta["tokens_per_slice"][0])
    return path


def rewrite(path, *, meta_update=None, arrays_update=None):
    with np.load(path, allow_pickle=False) as data:
        content = {key: data[key] for key in data.files}
    if meta_update:
        meta = json.loads(str(content["metadata"]))
        meta.update(meta_update)
        content["metadata"] = np.array(json.dumps(meta))
    content.update(arrays_update or {})
    np.savez(path, **content)


def test_sums_halves_means_overlap_and_provenance(tmp_path):
    a, b = raw(10), raw(20)
    pa = write_pass(tmp_path / "a.npz", a)
    pb = write_pass(tmp_path / "b.npz", b)
    # Stale derived values must not survive the merge.
    rewrite(
        pa,
        arrays_update={
            "robust_reap": np.full((2, 4), -1),
            "saliency": np.full((2, 4), -1),
            "split_keep_overlap": np.full((3, 3), -1),
        },
    )
    output = merge_calibration([pa, pb], tmp_path / "merged.npz")
    with np.load(output) as merged, np.load(pa) as left, np.load(pb) as right:
        assert set(merged.files) == set(left.files)
        for key in ("reap_by_slice", "counts_by_slice", "counts", "saliency"):
            # saliency in A was deliberately corrupted above; use raw data.
            expected = (
                (a[0] + b[0]).sum((0, 1))
                if key == "saliency"
                else left[key] + right[key]
            )
            np.testing.assert_allclose(merged[key], expected)
        for h, arrays in enumerate((a, b)):
            c = arrays[3].sum(1)
            np.testing.assert_array_equal(merged["counts_halves_by_slice"][:, h], c)
            np.testing.assert_allclose(merged["reap_halves"][:, h], arrays[0].sum(1))
            np.testing.assert_allclose(
                merged["man_halves"][:, h], arrays[1].sum(1) / np.maximum(c, 1)
            )
        total_counts = (a[3] + b[3]).sum(1)
        for i, name in enumerate(("reap", "man", "msan")):
            total = (a[i] + b[i]).sum(1)
            expected = total if i == 0 else total / np.maximum(total_counts, 1)
            np.testing.assert_allclose(merged[name + "_by_slice"], expected)
            np.testing.assert_allclose(merged["robust_" + name], expected.max(0))
            expected_overlap = [
                _overlap(x[0], x[1], 2) for x in merged[name + "_halves"]
            ]
            np.testing.assert_allclose(
                merged["split_keep_overlap"][:, i], expected_overlap
            )
        np.testing.assert_allclose(
            merged["saliency_halves"], merged["reap_halves"].sum(0)
        )
        np.testing.assert_array_equal(
            merged["counts_halves"], merged["counts_halves_by_slice"].sum(0)
        )
        np.testing.assert_allclose(merged["robust_saliency"], merged["robust_reap"])
        assert merged["tokens"].item() == 48
        meta = json.loads(str(merged["metadata"]))
        assert meta["tokens_per_slice"] == [16] * 3
        assert meta["merge"]["tokens_per_slice_halves"] == [[8] * 3, [8] * 3]
        for h, p in enumerate((pa, pb)):
            source = meta["merge"]["sources"][h]
            assert source["file"] == str(p.resolve())
            assert source["sha256"] == file_hash(p)
            assert source["half"] == h


@pytest.mark.parametrize("n", [3, 4])
def test_multiple_pass_grouping(tmp_path, n):
    arrays = [raw(i) for i in range(n)]
    paths = [write_pass(tmp_path / f"{i}.npz", a) for i, a in enumerate(arrays)]
    output = merge_calibration(paths, tmp_path / "merged.npz")
    with np.load(output) as merged:
        for h in (0, 1):
            np.testing.assert_allclose(
                merged["reap_halves"][:, h], sum(a[0].sum(1) for a in arrays[h::2])
            )
        meta = json.loads(str(merged["metadata"]))
        assert [s["half"] for s in meta["merge"]["sources"]] == [
            i % 2 for i in range(n)
        ]
        assert meta["merge"]["tokens_per_slice_halves"] == [
            [8 * len(arrays[h::2])] * 3 for h in (0, 1)
        ]


def test_splitting_one_run_halves_reproduces_every_derived_field(tmp_path):
    arrays = raw(42)
    original = write_pass(tmp_path / "original.npz", arrays, metadata(16))
    paths = []
    for half in (0, 1):
        # Repartition each original half into a synthetic pass's two subhalves.
        # Keep all routes and scores in the second subhalf of each pass.
        parts = tuple(
            np.stack((np.zeros_like(a[:, half]), a[:, half]), axis=1) for a in arrays
        )
        paths.append(write_pass(tmp_path / f"half-{half}.npz", parts))
    output = merge_calibration(paths, tmp_path / "merged.npz")
    with np.load(original) as expected, np.load(output) as actual:
        assert set(actual.files) == set(expected.files)
        for key in expected.files:
            if key == "metadata":
                continue
            if expected[key].dtype.kind in "US":
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14)


@pytest.mark.parametrize("key", COMPATIBILITY_FIELDS)
def test_metadata_mismatch_rejected(tmp_path, key):
    pa = write_pass(tmp_path / "a.npz", raw(1))
    pb = write_pass(tmp_path / "b.npz", raw(2))
    rewrite(pb, meta_update={key: "different"})
    output = tmp_path / "merged.npz"
    with pytest.raises((ValueError, TypeError), match=key + "|invalid|inconsistent"):
        merge_calibration([pa, pb], output)
    assert not output.exists()


def test_invalid_arrays_and_missing_metadata_rejected(tmp_path):
    pa = write_pass(tmp_path / "a.npz", raw(1))
    pb = write_pass(tmp_path / "b.npz", raw(2))
    rewrite(pb, arrays_update={"man_halves": np.full((3, 2, 2, 4), np.nan)})
    with pytest.raises(ValueError, match="man_halves"):
        merge_calibration([pa, pb], tmp_path / "bad.npz")
    meta = metadata()
    del meta["config_sha256"]
    write_pass(pb, raw(2), meta)
    with pytest.raises(ValueError, match="missing metadata config_sha256"):
        merge_calibration([pa, pb], tmp_path / "bad.npz")


def test_input_count_duplicates_and_overwrite_rejected(tmp_path):
    pa = write_pass(tmp_path / "a.npz", raw(1))
    pb = write_pass(tmp_path / "b.npz", raw(2))
    with pytest.raises(ValueError, match="at least two"):
        merge_calibration([pa], tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="duplicate"):
        merge_calibration([pa, pa], tmp_path / "bad.npz")
    before = file_hash(pa)
    with pytest.raises(ValueError, match="output must differ"):
        merge_calibration([pa, pb], pa)
    with pytest.raises(FileExistsError):
        merge_calibration([pa, pb], write_pass(tmp_path / "exists.npz", raw(3)))
    assert file_hash(pa) == before


def test_cli_writes_allocator_schema(tmp_path):
    pa = write_pass(tmp_path / "a.npz", raw(1))
    pb = write_pass(tmp_path / "b.npz", raw(2))
    output = tmp_path / "merged.npz"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/deepseek_v41_merge_calibration.py"
    )
    subprocess.run(
        [sys.executable, "-B", str(script), str(pa), str(pb), "--output", str(output)],
        check=True,
        cwd=tmp_path,
    )
    with np.load(output, allow_pickle=False) as merged:
        assert merged["reap_halves"].shape == (3, 2, 2, 4)
        assert merged["tokens"].item() == 48


def test_overlap_compares_passes_instead_of_original_halves(tmp_path):
    counts = np.ones((3, 2, 2, 4), dtype=np.int64)
    paths = []
    for i, order in enumerate(([10, 9, 1, 0], [0, 1, 9, 10])):
        values = np.broadcast_to(order, counts.shape).copy().astype(float)
        paths.append(
            write_pass(tmp_path / f"{i}.npz", (values, values, values, counts))
        )
        with np.load(paths[-1]) as original:
            np.testing.assert_array_equal(original["split_keep_overlap"], 1)
    output = merge_calibration(paths, tmp_path / "merged.npz")
    with np.load(output) as merged:
        np.testing.assert_array_equal(merged["split_keep_overlap"], 0)


def test_layer_expert_shape_mismatch_rejected(tmp_path):
    pa = write_pass(tmp_path / "a.npz", raw(1))
    pb = write_pass(tmp_path / "b.npz", tuple(a[:, :, :1] for a in raw(2)))
    with pytest.raises(ValueError, match="shape mismatch"):
        merge_calibration([pa, pb], tmp_path / "merged.npz")
