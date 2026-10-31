# SPDX-License-Identifier: Apache-2.0
"""Model identities included in GPU validation cache keys."""

import os
import types
from pathlib import Path

import huggingface_hub.constants
import pytest

from scripts.pr_validate import gpu_cache
from scripts.pr_validate.steps import stress_e2e_bench


@pytest.fixture
def cache_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    for name in ("hf_hub_download", "snapshot_download", "model_info"):
        monkeypatch.setattr(
            huggingface_hub, name, lambda *a, **kw: pytest.fail("network")
        )
    monkeypatch.setattr(gpu_cache, "_repo_state", lambda _: ("a" * 40, True))
    monkeypatch.setattr(gpu_cache, "_cache_dir", lambda _: tmp_path / "results")
    monkeypatch.setattr(gpu_cache, "_installed_fingerprint", lambda: "packages")
    monkeypatch.delenv("PR_VALIDATE_NO_GPU_CACHE", raising=False)
    choices = [types.SimpleNamespace(family="text", model_id="org/model")]
    monkeypatch.setattr(stress_e2e_bench, "_selected_choices", lambda: ({}, choices))
    messages = []
    ctx = types.SimpleNamespace(
        head_sha="a" * 40,
        base_sha="b" * 40,
        repo_root=tmp_path,
        run_log=messages.append,
    )

    def snapshot(model, revision, blob):
        repo = tmp_path / ("models--" + model.replace("/", "--"))
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "blobs").mkdir(exist_ok=True)
        (repo / "blobs" / blob).write_text(blob)
        directory = repo / "snapshots" / revision
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights.safetensors").symlink_to(Path("../../blobs") / blob)
        (repo / "refs" / "main").write_text(revision)
        return directory

    return ctx, choices, messages, snapshot


def test_key_tracks_selected_snapshots_and_blobs(cache_setup):
    ctx, choices, messages, snapshot = cache_setup
    snapshot("org/model", "a" * 40, "blob1")
    first = gpu_cache.cache_key(ctx)
    assert first and gpu_cache.cache_key(ctx) == first
    directory = snapshot("org/model", "b" * 40, "blob2")
    second = gpu_cache.cache_key(ctx)
    assert second and second != first
    link = directory / "weights.safetensors"
    link.unlink()
    link.symlink_to(Path("../../blobs/blob1"))
    third = gpu_cache.cache_key(ctx)
    assert third and third != second
    snapshot("org/other", "c" * 40, "blob3")
    choices[0].model_id = "org/other"
    fourth = gpu_cache.cache_key(ctx)
    assert fourth and fourth != third
    choices[0].model_id = "org/missing"
    assert gpu_cache.cache_key(ctx) is None
    assert (
        messages[-1] == "GPU cache disabled: model weights not resolvable (org/missing)"
    )


def test_key_changes_when_blob_is_rewritten_in_place(cache_setup):
    ctx, _, _, snapshot = cache_setup
    directory = snapshot("org/model", "a" * 40, "blob1")
    first = gpu_cache.cache_key(ctx)
    (directory / "weights.safetensors").write_text("different contents")
    second = gpu_cache.cache_key(ctx)
    assert first and second and second != first


def test_local_directory_and_dangling_links(cache_setup, tmp_path):
    ctx, choices, _, snapshot = cache_setup
    local = tmp_path / "local-model"
    local.mkdir()
    (local / "weights").write_text("one")
    choices[0].model_id = str(local)
    first = gpu_cache.cache_key(ctx)
    (local / "weights").write_text("different")
    assert first and first != gpu_cache.cache_key(ctx)
    choices[0].model_id = "org/model"
    directory = snapshot("org/model", "a" * 40, "blob1")
    (directory / "missing").symlink_to("../../blobs/missing")
    assert gpu_cache.cache_key(ctx) is None


def test_local_file_rewrite_with_restored_mtime_changes_key(cache_setup, tmp_path):
    ctx, choices, _, _ = cache_setup
    local = tmp_path / "local-model"
    local.mkdir()
    weights = local / "weights"
    weights.write_text("one")
    choices[0].model_id = str(local)
    first = gpu_cache.cache_key(ctx)
    original = weights.stat()

    weights.write_text("two")
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert weights.stat().st_size == original.st_size
    assert weights.stat().st_mtime_ns == original.st_mtime_ns
    second = gpu_cache.cache_key(ctx)
    assert first and second and second != first


def test_selected_choices_reports_registry_load_failure(monkeypatch):
    def fail_load():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(stress_e2e_bench, "_load_registry", fail_load)

    with pytest.raises(
        stress_e2e_bench._SelectionError, match="could not load model registry"
    ):
        stress_e2e_bench._selected_choices()


def test_selected_choices_reports_ram_probe_failure(monkeypatch):
    def fail_probe():
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(stress_e2e_bench, "_load_registry", lambda: {})
    monkeypatch.setattr(stress_e2e_bench, "_available_ram_gb", fail_probe)

    with pytest.raises(
        stress_e2e_bench._SelectionError, match="RAM probe failed"
    ) as exc:
        stress_e2e_bench._selected_choices()
    assert "PR_VALIDATE_RAM_GB" in str(exc.value)


def test_selected_choices_reports_when_no_model_fits(monkeypatch):
    monkeypatch.setattr(stress_e2e_bench, "_load_registry", lambda: {})
    monkeypatch.setattr(stress_e2e_bench, "_available_ram_gb", lambda: 16.0)
    monkeypatch.setattr(
        stress_e2e_bench, "_select_models", lambda registry, usable_gb: []
    )

    with pytest.raises(
        stress_e2e_bench._SelectionError, match="no model in registry fits"
    ):
        stress_e2e_bench._selected_choices()


def test_run_matrix_reports_selection_error(monkeypatch):
    def fail_selection():
        raise stress_e2e_bench._SelectionError("boom")

    monkeypatch.setattr(stress_e2e_bench, "_selected_choices", fail_selection)
    step = stress_e2e_bench.StressE2EBenchStep()
    ctx = types.SimpleNamespace(run_log=lambda message: None)

    result = step._run_matrix(ctx)

    assert result.status == "error"
    assert result.summary == "boom"
