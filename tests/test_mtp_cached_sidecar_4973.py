"""Automatic MTP must stay local; explicit requests may provision a sidecar."""

import json
from dataclasses import replace
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub import constants

from rapid_mlx import cli, model_aliases
from tests.test_mtp_default_off_3115 import _args

ALIAS = "qwen3.6-35b-4bit"
SHA = "a" * 40
OTHER_SHA = "b" * 40
CONFIG = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}


def snapshot(cache, repo, sha=None, weight="model.safetensors"):
    sha = sha or model_aliases.resolve_revision(repo) or SHA
    root = cache / ("models--" + repo.replace("/", "--"))
    path = root / "snapshots" / sha
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CONFIG))
    if weight:
        weights = path / weight
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"weights")
    ref = root / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(sha)
    return path


@pytest.fixture
def cached_target(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    def no_fetch(*args, **kwargs):
        pytest.fail("cached startup attempted a Hub download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_fetch)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_fetch)
    profile = model_aliases.resolve_profile(ALIAS)
    snapshot(tmp_path, profile.hf_path)
    return tmp_path, profile


@pytest.mark.parametrize(
    "weight",
    [
        "model.safetensors",
        "model-mtp.safetensors",
        "mtp/model.safetensors",
        "mtp.safetensors",
    ],
)
def test_cached_alias_and_sidecar_select_local_mtp_without_fetch(cached_target, weight):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model, weight=weight)
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == str(local)
    assert args._speculative_config.model == str(local)
    assert args.mtp_max_k == profile.mtp_speculative_tokens
    # The injector receives a local directory and never enters its Hub branch.
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _resolve_sidecar_file

    assert _resolve_sidecar_file(args.mtp_sidecar) == local / weight


@pytest.mark.parametrize("state", ["absent", "config-only", "empty", "broken-link"])
def test_missing_sidecar_starts_plain_decode_without_fetch(
    cached_target, capsys, state
):
    cache, profile = cached_target
    if state != "absent":
        local = snapshot(cache, profile.mtp_draft_model, weight=None)
        if state == "empty":
            (local / "model.safetensors").touch()
        elif state == "broken-link":
            (local / "model.safetensors").symlink_to("missing-blob")
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "none"
    assert args._speculative_config is None
    assert args.mtp_sidecar is None
    guidance = capsys.readouterr().err
    assert f"rapid-mlx pull {ALIAS}" in guidance
    assert f"rapid-mlx pull {profile.mtp_draft_model}" in guidance


@pytest.mark.parametrize("payload", ['{"method":"mtp"}', None])
def test_explicit_mtp_keeps_repo_download_behavior(cached_target, payload):
    _, profile = cached_target
    args = _args(ALIAS, payload, spec_decode="mtp" if payload is None else "none")
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == profile.mtp_draft_model


@pytest.mark.parametrize("pin_cached", [True, False])
def test_sidecar_catalog_pin_never_uses_other_snapshot(
    cached_target, monkeypatch, pin_cached
):
    cache, profile = cached_target
    sidecar_profile = replace(profile, hf_path=profile.mtp_draft_model, revision=SHA)
    resolve = model_aliases.resolve_profile
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: (
            sidecar_profile if name == profile.mtp_draft_model else resolve(name)
        ),
    )
    if pin_cached:
        local = snapshot(cache, profile.mtp_draft_model)
    snapshot(cache, profile.mtp_draft_model, OTHER_SHA)
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == ("mtp" if pin_cached else "none")
    assert args.mtp_sidecar == (str(local) if pin_cached else None)


def test_missing_target_catalog_pin_explains_skip(cached_target, monkeypatch, capsys):
    _, profile = cached_target
    resolve = model_aliases.resolve_profile
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: (
            replace(profile, revision=OTHER_SHA)
            if name in (ALIAS, profile.hf_path)
            else resolve(name)
        ),
    )
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "none"
    assert f"revision {OTHER_SHA}" in capsys.readouterr().err


def test_sidecar_probe_io_failure_falls_back(cached_target, monkeypatch):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    bad = local / "mtp.safetensors"
    bad.write_bytes(b"head")
    original_stat = Path.stat

    def fail(path, *args, **kwargs):
        if path == bad:
            raise OSError("unreadable cache")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", fail)
        assert cli._cached_mtp_sidecar(profile.mtp_draft_model) is None
        from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file

        with pytest.raises(OSError, match="unreadable cache"):
            _find_mtp_weights_file(local)


def test_embedded_mtp_needs_no_sidecar(cached_target, monkeypatch):
    _, profile = cached_target
    resolve = model_aliases.resolve_profile
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: (
            replace(profile, mtp_draft_model=None)
            if name in (ALIAS, profile.hf_path)
            else resolve(name)
        ),
    )
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar is None


def test_configless_sidecar_still_stays_local(cached_target):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    (local / "config.json").unlink()
    assert cli._cached_mtp_sidecar(profile.mtp_draft_model) == str(local)


def test_unreferenced_sidecar_stays_local(cached_target):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    (local.parent.parent / "refs" / "main").unlink()
    assert cli._cached_mtp_sidecar(profile.mtp_draft_model) == str(local)


def test_unreferenced_zero_weight_is_not_ready(cached_target):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    (local.parent.parent / "refs" / "main").unlink()
    (local / "model.safetensors").write_bytes(b"")
    assert cli._cached_mtp_sidecar(profile.mtp_draft_model) is None


@pytest.mark.parametrize("kind", ["empty", "directory"])
def test_unusable_priority_head_does_not_select_later_trunk_weights(
    cached_target, kind
):
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file

    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    bad = local / "mtp.safetensors"
    if kind == "empty":
        bad.touch()
    else:
        bad.mkdir()
    assert _find_mtp_weights_file(local) == bad
    assert cli._cached_mtp_sidecar(profile.mtp_draft_model) is None


@pytest.mark.parametrize("no_ref", [True, False])
@pytest.mark.parametrize("pinned", [True, False])
def test_sidecar_subfolder_is_applied_once(cached_target, monkeypatch, no_ref, pinned):
    cache, profile = cached_target
    sidecar_profile = replace(
        profile,
        hf_path=profile.mtp_draft_model,
        subfolder="4bit",
        revision=SHA if pinned else None,
    )
    resolve = model_aliases.resolve_profile
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: (
            sidecar_profile if name == profile.mtp_draft_model else resolve(name)
        ),
    )
    local = snapshot(cache, profile.mtp_draft_model, weight="4bit/model.safetensors")
    (local / "4bit" / "config.json").write_text(json.dumps(CONFIG))
    if no_ref:
        (local.parent.parent / "refs" / "main").unlink()
    assert cli._cached_mtp_sidecar(profile.mtp_draft_model) == str(local / "4bit")


@pytest.mark.parametrize("main_revision", [None, OTHER_SHA])
def test_pinned_target_ignores_missing_or_moved_main(cached_target, main_revision):
    cache, profile = cached_target
    local = snapshot(cache, profile.mtp_draft_model)
    target_root = cache / ("models--" + profile.hf_path.replace("/", "--"))
    if main_revision is None:
        (target_root / "refs" / "main").unlink()
    else:
        moving = snapshot(cache, profile.hf_path, main_revision)
        (moving / "config.json").write_text('{"model_type": "other"}')
    args = _args(ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == str(local)
