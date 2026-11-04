# SPDX-License-Identifier: Apache-2.0
"""Catalog default MTP companions are optional provisioning dependencies."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rapid_mlx import cli, model_aliases, model_downloads
from tests.test_auto_mtp_defer_until_download import (
    serve_download as _serve_download_fixture,
)

serve_download = _serve_download_fixture

ALIAS = "qwen3.6-35b-4bit"
CONFIG = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1}
PIN = "a" * 40


@pytest.fixture
def provision(monkeypatch):
    profile = model_aliases.resolve_profile(ALIAS)
    sidecar = profile.mtp_draft_model
    state = SimpleNamespace(target=False, head=False, calls=[], sizes=[], disks=[])
    resolve = model_aliases.resolve_profile
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: (
            replace(profile, hf_path=sidecar, revision=PIN)
            if name == sidecar
            else resolve(name)
        ),
    )
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda _: (CONFIG if state.target else None, None),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_cached_hf_config",
        lambda *_: (
            None,
            f"local cache for {profile.hf_path} has no config.json at revision abc",
        ),
    )
    monkeypatch.setattr(
        cli, "_cached_mtp_sidecar", lambda _: "/cached/mtp" if state.head else None
    )
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)

    def pull(args, **kwargs):
        state.calls.append((args.model, kwargs, getattr(args, "bits", None)))
        if args.model == profile.hf_path:
            state.target = True
        elif args.model == sidecar:
            assert state.target
            assert args._original_alias == sidecar
            assert args._emit_pull_lifecycle_event is False
            state.head = True
        else:
            pytest.fail(f"unexpected dependency: {args.model}")

    monkeypatch.setattr(cli, "_pull_repository", pull)

    def size(repo, **kwargs):
        state.sizes.append((repo, kwargs))
        return 495 if repo == sidecar else 2000

    monkeypatch.setattr(cli, "_missing_download_bytes", size)
    monkeypatch.setattr(
        cli, "_check_download_disk_space", lambda n, **kw: state.disks.append(n)
    )
    return profile, state


def test_pull_cold_alias_plans_combined_bytes_and_pulls_pinned_sidecar(
    provision, capsys
):
    profile, state = provision
    cli.pull_command(SimpleNamespace(model=ALIAS))
    assert [call[0] for call in state.calls] == [
        profile.hf_path,
        profile.mtp_draft_model,
    ]
    assert state.calls[1][1] == {
        "revision_override": PIN,
        "allow_patterns_override": ["*"],
    }
    assert state.calls[1][2] is None
    assert state.disks == [2495]
    assert (
        profile.mtp_draft_model,
        {"revision_override": PIN, "allow_patterns": ["*"]},
    ) in state.sizes
    assert "495 B uncached" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [OSError("network down"), SystemExit(1)])
def test_sidecar_failure_preserves_successful_target_pull(
    provision, monkeypatch, capsys, failure
):
    profile, state = provision
    original = cli._pull_repository

    def fail(args, **kwargs):
        if args.model == profile.mtp_draft_model:
            raise failure
        return original(args, **kwargs)

    monkeypatch.setattr(cli, "_pull_repository", fail)
    cli.pull_command(SimpleNamespace(model=ALIAS))
    assert state.target and not state.head
    assert "Using plain decode." in capsys.readouterr().err


@pytest.mark.parametrize(
    "override",
    [
        {"no_spec_decode": True},
        {"speculative_config": '{"method":"suffix"}'},
        {"mllm": True},
    ],
)
def test_provisioning_respects_explicit_serving_choices(provision, override):
    _, state = provision
    state.target = True
    cli._provision_automatic_mtp_sidecar(SimpleNamespace(model=ALIAS, **override))
    assert state.calls == []


@pytest.mark.parametrize(
    "condition",
    [
        "default-off",
        "blocked",
        "unsupported",
        "same-repo",
        "offline",
        "disabled",
        "cached",
        "local-folder",
    ],
)
def test_unselected_or_unavailable_sidecar_is_not_downloaded(
    provision, monkeypatch, tmp_path, condition
):
    profile, state = provision
    state.target = True
    args = SimpleNamespace(model=ALIAS)
    if condition == "default-off":
        monkeypatch.setattr(cli, "_alias_mtp_default_enabled", lambda _: False)
    elif condition == "blocked":
        monkeypatch.setattr(cli, "_alias_continuous_mtp_tier", lambda _: "blocked")
    elif condition == "unsupported":
        monkeypatch.setattr(
            cli,
            "_gather_kv_cache_dtype_inputs",
            lambda _: ({"model_type": "llama"}, None),
        )
    elif condition == "same-repo":
        monkeypatch.setattr(
            cli, "_alias_mtp_declaration", lambda _: (profile.hf_path, 2)
        )
    elif condition == "offline":
        monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
    elif condition == "disabled":
        monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    elif condition == "local-folder":
        args.model = str(tmp_path)
        args._original_alias = ALIAS
    else:
        state.head = True
    cli._provision_automatic_mtp_sidecar(args)
    assert state.calls == []


def test_shared_implicit_provisioning_fetches_sidecar_after_target(
    provision, monkeypatch
):
    profile, state = provision
    monkeypatch.setattr(
        cli,
        "_ensure_target_model_downloaded",
        lambda *a, **kw: setattr(state, "target", True),
    )
    cli._ensure_model_downloaded(profile.hf_path)
    assert state.head
    assert [call[0] for call in state.calls] == [profile.mtp_draft_model]


@pytest.mark.parametrize("serve_download", [ALIAS], indirect=True)
@pytest.mark.parametrize("failure", [False, True])
def test_serve_first_run_external_sidecar_is_selected_or_falls_back(
    serve_download, monkeypatch, failure
):
    state, run = serve_download
    profile = model_aliases.resolve_profile(ALIAS)
    monkeypatch.setattr(cli, "_run_optional_runtime_guard", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_serve_will_run_on_mllm_lane", lambda _: False)
    state.head = False
    calls = []
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda _: (CONFIG if state.downloaded else None, None),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_cached_hf_config",
        lambda *_: (
            None,
            f"local cache for {profile.hf_path} has no config.json at revision abc",
        ),
    )
    monkeypatch.setattr(
        cli, "_cached_mtp_sidecar", lambda _: "/cached/mtp" if state.head else None
    )
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)

    def pull(args, **kwargs):
        assert state.downloaded
        calls.append(args.model)
        if failure:
            raise OSError("sidecar unavailable")
        state.head = True

    monkeypatch.setattr(cli, "_pull_repository", pull)
    run(
        deferred=True,
        enabled=not failure,
        skip_message=(
            f"Automatic MTP skipped: the MTP sidecar {profile.mtp_draft_model} is not downloaded; "
            f"run `rapid-mlx pull {profile.mtp_draft_model}` and restart to enable it. Using plain decode."
        )
        if failure
        else None,
    )
    assert calls == [profile.mtp_draft_model]


def test_shared_prefetch_opt_out_keeps_target_download(provision, monkeypatch):
    _, state = provision
    monkeypatch.setattr(
        cli,
        "_ensure_target_model_downloaded",
        lambda *a, **kw: setattr(state, "target", True),
    )
    cli._ensure_model_downloaded(ALIAS, auto_mtp=False)
    assert state.target and not state.head
    assert state.calls == []


@pytest.mark.parametrize(
    "choice", [{"no_spec_decode": True}, {"speculative_config": '{"method":"suffix"}'}]
)
@pytest.mark.parametrize("serve_download", [ALIAS], indirect=True)
def test_serve_explicit_choice_never_pulls_automatic_sidecar(
    serve_download, monkeypatch, choice
):
    from tests.test_continuous_mtp_qualification import _StopServeError

    state, _ = serve_download
    for name, value in choice.items():
        setattr(state.args, name, value)
    monkeypatch.setattr(cli, "_run_optional_runtime_guard", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_serve_will_run_on_mllm_lane", lambda _: False)
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda _: (CONFIG, None))
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **kw: pytest.fail("explicit choice fetched auto MTP"),
    )
    with pytest.raises(_StopServeError):
        cli.serve_command(state.args)
    assert state.downloaded
    assert state.args.spec_decode == "none"
    if "speculative_config" in choice:
        assert state.args._speculative_config.method == "suffix"
        assert state.args.suffix_decoding
    else:
        assert state.args._speculative_config is None


@pytest.mark.parametrize("serve_download", [ALIAS], indirect=True)
def test_serve_offline_missing_sidecar_keeps_single_plain_decode_guidance(
    serve_download, monkeypatch
):
    state, run = serve_download
    profile = model_aliases.resolve_profile(ALIAS)
    monkeypatch.setattr(cli, "_run_optional_runtime_guard", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_serve_will_run_on_mllm_lane", lambda _: False)
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda _: (CONFIG, None))
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda _: None)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
    monkeypatch.setattr(
        cli, "_pull_repository", lambda *a, **kw: pytest.fail("offline sidecar fetch")
    )
    run(
        deferred=True,
        enabled=False,
        skip_message=(
            f"Automatic MTP skipped: the MTP sidecar {profile.mtp_draft_model} is not downloaded; "
            f"run `rapid-mlx pull {profile.mtp_draft_model}` and restart to enable it. Using plain decode."
        ),
    )
