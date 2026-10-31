# SPDX-License-Identifier: Apache-2.0
"""Pinned sidecars retain their loader-resolved snapshot in benchmarks."""

import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO = "z-lab/Qwen3.8-27B-DFlash2"
REVISION = "a" * 40


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def snapshot(repo, revision):
        path = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"dummy")
        return path

    return snapshot


def assert_identity(path, repo, revision):
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.run_builder import unresolved_model_identity

    found_repo, snapshot, subfolder, _ = local_runner._sidecar_identity(path)
    assert (found_repo, snapshot, subfolder) == (repo, path, None)
    primary = unresolved_model_identity("test/primary", "text_generation")
    combined = local_runner._with_sidecar_identity(primary, found_repo, snapshot)
    sidecar = next(c for c in combined["components"] if c["component_id"] == "sidecar")
    assert sidecar["source"]["resolved_revision"] == revision


@pytest.mark.parametrize(
    ("alias", "drafter_repo"),
    [
        ("qwen3.5-27b-8bit", None),
        ("qwen3.8-27b-4bit", REPO),
    ],
)
def test_dflash_pinned_sidecar_path(fake_cache, monkeypatch, alias, drafter_repo):
    from rapid_mlx.cli import _dflash_load_kwargs
    from rapid_mlx.speculative import serial_benchmark
    from rapid_mlx.speculative.dflash import runtime as dflash

    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.__path__ = []
    speculative = ModuleType("mlx_vlm.speculative")
    speculative.__path__ = []
    utils = ModuleType("mlx_vlm.utils")
    common = ModuleType("mlx_vlm.speculative.common")
    upstream_drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters = ModuleType("rapid_mlx.models.mlx_vlm_vendored.speculative.drafters")
    server = ModuleType("rapid_mlx.speculative.dflash.server")
    mlx_vlm.speculative = speculative
    mlx_vlm.utils = utils
    speculative.drafters = upstream_drafters
    for module in (
        mlx_vlm,
        speculative,
        utils,
        common,
        upstream_drafters,
        drafters,
        server,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    args = SimpleNamespace(
        model=alias,
        _original_alias=alias,
        _dflash_drafter_repo=drafter_repo,
        dflash_backend=None,
        _dflash_experimental=True,
    )
    kwargs = _dflash_load_kwargs(args)
    repo = kwargs["drafter_repo"]
    revision = kwargs["drafter_revision"]
    assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    if drafter_repo is not None:
        assert repo == drafter_repo
    snapshot = fake_cache(repo, revision)
    calls = []

    def get_model_path(repo, *, revision):
        calls.append((repo, revision))
        assert (repo, revision) == (kwargs["drafter_repo"], kwargs["drafter_revision"])
        return snapshot

    monkeypatch.setattr(dflash, "have_runtime", lambda: True)
    monkeypatch.setattr(utils, "get_model_path", get_model_path, raising=False)
    monkeypatch.setattr(
        drafters,
        "load_drafter",
        lambda source, kind: (
            SimpleNamespace(config={"model_type": "dflash2"}),
            "dflash",
        ),
        raising=False,
    )
    runtime = dflash.load_runtime(
        repo,
        target_revision=kwargs["main_model_revision"],
        drafter_revision=revision,
        expected_algorithm=None,
    )
    assert calls == [(repo, revision)]
    assert runtime.drafter_path == str(snapshot)

    monkeypatch.setattr(common, "_dflash_block_total", lambda *args: 16, raising=False)
    monkeypatch.setattr(
        server, "dflash_generation_kwargs", lambda *args: {}, raising=False
    )
    monkeypatch.setattr(serial_benchmark, "_runtime_handle", lambda *args: args[5])
    facts = serial_benchmark.dflash_runtime_handle(
        None,
        None,
        runtime,
        drafter_repo=repo,
        drafter_revision=revision,
        target_path=None,
    )
    assert facts["draft_path"] == str(snapshot)
    assert_identity(facts["draft_path"], repo, revision)


def test_bare_repo_without_ref_fails_with_method_neutral_error(fake_cache):
    from rapid_mlx.community_bench import local_runner

    fake_cache(REPO, REVISION)
    with pytest.raises(local_runner.AccelerationConfigError) as exc:
        local_runner._sidecar_identity(REPO)
    assert "speculative sidecar" in str(exc.value)
    assert "MTP" not in str(exc.value)


def test_native_mtp_pinned_sidecar_path(fake_cache, monkeypatch):
    from rapid_mlx.speculative import serial_benchmark
    from rapid_mlx.speculative.native_mtp.runtime import load_runtime

    repo, revision = "example/mtp", "b" * 40
    snapshot = fake_cache(repo, revision)
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.__path__ = []
    speculative = ModuleType("mlx_vlm.speculative")
    speculative.__path__ = []
    upstream_drafters = ModuleType("mlx_vlm.speculative.drafters")
    utils = ModuleType("mlx_vlm.utils")
    drafters = ModuleType("rapid_mlx.models.mlx_vlm_vendored.speculative.drafters")
    mlx_vlm.speculative = speculative
    mlx_vlm.utils = utils
    speculative.drafters = upstream_drafters
    for module in (mlx_vlm, speculative, upstream_drafters, utils, drafters):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def get_model_path(requested_repo, *, revision):
        assert (requested_repo, revision) == (repo, "b" * 40)
        return snapshot

    monkeypatch.setattr(utils, "get_model_path", get_model_path, raising=False)
    monkeypatch.setattr(
        drafters,
        "load_drafter",
        lambda source, kind: (
            SimpleNamespace(
                config=SimpleNamespace(model_type="qwen3_5_mtp", block_size=4)
            ),
            "mtp",
        ),
        raising=False,
    )
    runtime = load_runtime(
        repo, target_revision="t" * 40, drafter_revision=revision, block_size=4
    )
    assert runtime.drafter_path == str(snapshot)
    monkeypatch.setattr(serial_benchmark, "_runtime_handle", lambda *args: args[5])
    facts = serial_benchmark.native_mtp_runtime_handle(
        None, None, runtime, None, pair=None, target_path=None
    )
    assert facts["draft_path"] == str(snapshot)
    assert_identity(facts["draft_path"], repo, revision)
