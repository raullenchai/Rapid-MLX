"""Offline alias preflight must resolve copied caches without refs/main (#4796)."""

import json
from dataclasses import replace

import huggingface_hub
import pytest
from huggingface_hub import constants

from rapid_mlx import cli, model_aliases
from tests.test_mtp_default_off_3115 import _args

REPO = "mlx-community/Qwen3.5-9B-4bit"
ALIAS = "qwen3.5-9b-4bit"
SHA_A = "a" * 40
SHA_B = "b" * 40
CONFIG = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path / "models--mlx-community--Qwen3.5-9B-4bit"


def _snapshot(cache, sha=SHA_A, config=CONFIG):
    path = cache / "snapshots" / sha / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if config is not None:
        path.write_text(json.dumps(config))
    return path


def _main(cache, sha=SHA_A):
    ref = cache / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(sha)


def test_single_snapshot_without_main_enables_alias_mtp(cache):
    _snapshot(cache)
    assert cli._gather_kv_cache_dtype_inputs(ALIAS)[0] == CONFIG
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config.method == "mtp"


def test_main_ref_remains_authoritative(cache):
    path = _snapshot(cache)
    _snapshot(cache, SHA_B, {"model_type": "other"})
    _main(cache)
    assert cli._resolve_cached_hf_config(REPO, None) == (str(path), None)
    assert cli._gather_kv_cache_dtype_inputs(ALIAS)[0] == CONFIG


def test_missing_main_config_does_not_mask_with_other_snapshot(cache):
    _snapshot(cache, config=None)
    _snapshot(cache, SHA_B)
    _main(cache)
    assert cli._gather_kv_cache_dtype_inputs(ALIAS)[0] is None
    assert cli._resolve_cached_hf_config(REPO, None) == (
        None,
        f"local cache for {REPO} has no config.json at refs/main",
    )


def test_ambiguous_snapshots_disable_mtp_with_reason(cache, capsys):
    _snapshot(cache)
    _snapshot(cache, SHA_B)
    # Non-directories must not count as snapshots.
    (cache / "snapshots" / "README").write_text("ignore")
    assert cli._gather_kv_cache_dtype_inputs(ALIAS)[0] is None
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config is None
    assert capsys.readouterr().err == (
        f"Automatic MTP skipped: local cache for {REPO} has 2 snapshots "
        "and no refs/main; cannot choose one; using plain decode.\n"
    )


@pytest.mark.parametrize(
    "pin_field", ["tensorfold_target_revision", "dflash_target_revision"]
)
def test_alias_pin_resolves_ambiguity(cache, monkeypatch, pin_field):
    _snapshot(cache)
    path = _snapshot(cache, SHA_B, {**CONFIG, "pinned": True})
    pins = {"tensorfold_target_revision": None, "dflash_target_revision": SHA_A}
    pins[pin_field] = SHA_B
    profile = replace(model_aliases.resolve_profile(ALIAS), **pins)
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _: profile)
    assert cli._resolve_cached_hf_config(REPO, SHA_B) == (str(path), None)
    assert cli._gather_kv_cache_dtype_inputs(ALIAS)[0] == {**CONFIG, "pinned": True}


def test_repo_not_cached(cache):
    assert cli._resolve_cached_hf_config(REPO, None) == (
        None,
        f"{REPO} is not in the local cache",
    )


@pytest.mark.parametrize("snapshots_dir", [False, True])
def test_no_snapshots(cache, snapshots_dir):
    cache.mkdir()
    if snapshots_dir:
        (cache / "snapshots").mkdir()
    assert cli._resolve_cached_hf_config(REPO, None) == (
        None,
        f"local cache for {REPO} has 0 snapshots and no refs/main; cannot choose one",
    )


@pytest.mark.parametrize("pin", [None, SHA_B])
def test_snapshot_config_missing(cache, pin):
    _snapshot(cache, config=None)
    assert cli._resolve_cached_hf_config(REPO, pin) == (
        None,
        f"local cache for {REPO} has no config.json at revision {pin or SHA_A}",
    )


def test_cached_no_exist_is_not_a_path(cache):
    _snapshot(cache)
    marker = cache / ".no_exist" / SHA_A / "config.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    assert cli._resolve_cached_hf_config(REPO, None) == (
        None,
        f"local cache for {REPO} has no config.json at revision {SHA_A}",
    )


def test_cache_lookup_failure_is_best_effort(cache, monkeypatch):
    def fail(**kwargs):
        raise OSError("unreadable cache")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fail)
    assert cli._resolve_cached_hf_config(REPO, None) == (
        None,
        f"local cache for {REPO} could not be inspected",
    )


def test_unreadable_config_explains_auto_mtp_skip(cache, capsys):
    _snapshot(cache).write_text("invalid JSON")
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config is None
    assert capsys.readouterr().err == (
        "Automatic MTP skipped: cached target config could not be read; "
        "using plain decode.\n"
    )


def test_missing_pinned_config_explains_auto_mtp_skip(cache, monkeypatch, capsys):
    _snapshot(cache)
    profile = replace(
        model_aliases.resolve_profile(ALIAS), dflash_target_revision=SHA_B
    )
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _: profile)
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config is None
    assert capsys.readouterr().err == (
        f"Automatic MTP skipped: local cache for {REPO} has no config.json "
        f"at revision {SHA_B}; using plain decode.\n"
    )


def test_local_directory_unchanged(cache, tmp_path):
    local = tmp_path / "converted"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(CONFIG))
    assert cli._gather_kv_cache_dtype_inputs(str(local)) == (CONFIG, None)
