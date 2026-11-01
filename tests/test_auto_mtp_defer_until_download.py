# SPDX-License-Identifier: Apache-2.0
"""Automatic MTP retries missing target artifacts after the serve download."""

import sys
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.model_aliases import resolve_profile
from tests.test_continuous_mtp_qualification import _parsed_serve_args, _StopServeError


@pytest.fixture
def serve_download(monkeypatch, capsys):
    """Run real serve preflight, observing selection before and after download."""
    args = _parsed_serve_args("qwen3.8-27b-4bit")
    state = SimpleNamespace(downloaded=False, before=None, stderr_before="")

    def download(*_args, **_kwargs):
        state.before = (
            getattr(args, "_auto_mtp_deferred", False),
            args.spec_decode,
        )
        state.stderr_before = capsys.readouterr().err
        state.downloaded = True

    def stop(_level):
        raise _StopServeError

    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", download)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_warn_vision_text_only_degrade", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_configure_bootstrap_logging", stop)

    def run(*, deferred, enabled, skip_message=None):
        with pytest.raises(_StopServeError):
            cli.serve_command(args)
        assert state.downloaded
        assert state.before == (deferred, "none")
        stderr_after = capsys.readouterr().err
        if deferred:
            assert "Automatic MTP skipped" not in state.stderr_before
        else:
            assert state.stderr_before.count(skip_message) == 1
            assert "Automatic MTP skipped" not in stderr_after
        output = state.stderr_before + stderr_after
        if skip_message is None:
            assert "Automatic MTP skipped" not in output
        else:
            assert output.count("Automatic MTP skipped") == 1
            assert output.count(skip_message) == 1
        assert args.spec_decode == ("mtp" if enabled else "none")
        assert (args.speculative_config is not None) == enabled
        assert (args._speculative_config is not None) == enabled
        assert not getattr(args, "_auto_mtp_deferred", False)
        assert not getattr(args, "_auto_mtp_retry", False)

    return state, run


_ELIGIBLE_CONFIG = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}


@pytest.mark.parametrize("location", ["revision abc", "refs/main"])
@pytest.mark.parametrize("available_after", [True, False])
def test_absent_target_config_defers(
    monkeypatch, serve_download, location, available_after
):
    state, run = serve_download
    repo = resolve_profile("qwen3.8-27b-4bit").hf_path
    reason = f"local cache for {repo} has no config.json at {location}"
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda _: (
            _ELIGIBLE_CONFIG if state.downloaded and available_after else None,
            None,
        ),
    )
    monkeypatch.setattr(cli, "_resolve_cached_hf_config", lambda *_: (None, reason))
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda _: "/cached/mtp")

    run(
        deferred=True,
        enabled=available_after,
        skip_message=None
        if available_after
        else f"Automatic MTP skipped: {reason}; using plain decode.",
    )


@pytest.mark.parametrize(
    "problem",
    ["could not be inspected", "has 2 snapshots and no refs/main; cannot choose one"],
)
def test_uninspectable_or_ambiguous_cache_skips_immediately(
    monkeypatch, serve_download, problem
):
    _, run = serve_download
    repo = resolve_profile("qwen3.8-27b-4bit").hf_path
    reason = f"local cache for {repo} {problem}"
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda _: (None, None))
    monkeypatch.setattr(cli, "_resolve_cached_hf_config", lambda *_: (None, reason))

    run(
        deferred=False,
        enabled=False,
        skip_message=f"Automatic MTP skipped: {reason}; using plain decode.",
    )


@pytest.mark.parametrize(
    "same_repo,available_after", [(True, True), (True, False), (False, True)]
)
def test_missing_sidecar_defers_only_for_target_repo(
    monkeypatch, serve_download, same_repo, available_after
):
    state, run = serve_download
    profile = resolve_profile("qwen3.8-27b-4bit")
    assert profile.mtp_draft_model == profile.hf_path
    sidecar = profile.mtp_draft_model
    if not same_repo:
        sidecar = "example/separate-mtp-repo"
        monkeypatch.setattr(cli, "_alias_mtp_declaration", lambda _: (sidecar, 3))
    monkeypatch.setattr(
        cli, "_gather_kv_cache_dtype_inputs", lambda _: (_ELIGIBLE_CONFIG, None)
    )
    monkeypatch.setattr(
        cli,
        "_cached_mtp_sidecar",
        lambda _: "/cached/mtp" if state.downloaded and available_after else None,
    )

    enabled = same_repo and available_after
    run(
        deferred=same_repo,
        enabled=enabled,
        skip_message=None
        if enabled
        else (
            f"Automatic MTP skipped: the MTP sidecar {sidecar} is not downloaded; "
            f"run `rapid-mlx pull {sidecar}` and restart to enable it. Using plain decode."
        ),
    )
