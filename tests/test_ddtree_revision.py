# SPDX-License-Identifier: Apache-2.0
"""Offline DDTree catalog pin propagation and override contracts."""

import ast
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.model_aliases import _coerce, list_profiles, resolve_profile
from rapid_mlx.speculative.ddtree import runtime

DRAFT_REVISION = "5fc3b3d474760f18c516db87d84c37edbfd3ede6"


@pytest.mark.parametrize(
    "preflight,expected_revision",
    [
        ({}, DRAFT_REVISION),
        ({"_ddtree_drafter_repo": "user/drafter"}, None),
        (
            {
                "_ddtree_drafter_repo": "user/drafter",
                "_ddtree_drafter_revision": "a" * 40,
            },
            "a" * 40,
        ),
        ({"_ddtree_drafter_revision": "a" * 40}, DRAFT_REVISION),
    ],
)
def test_serve_ddtree_drafter_revision_fallback(
    monkeypatch, preflight, expected_revision
):
    from rapid_mlx import cli

    # Execute the actual launch branch without the unrelated serve boot path.
    function = ast.parse(inspect.getsource(cli.serve_command)).body[0]
    branch = next(
        node
        for node in reversed(function.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.enable_ddtree"
    )
    function.body = [branch]
    launch = Mock()
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.speculative.ddtree.server",
        SimpleNamespace(run_ddtree_server=launch),
    )
    profile = resolve_profile("qwen3.5-9b-8bit")
    preflight_check = Mock(return_value=("qwen3.5-9b-8bit", profile))
    namespace = {
        "__package__": "rapid_mlx",
        "_preflight_ddtree_or_exit": preflight_check,
        "_resolved_serve_port": lambda args: args.port,
        "port_explicit_for": lambda args: False,
        "cors_origins": [],
        "uvicorn_log_level": "error",
        "server": SimpleNamespace(_api_key=None),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), cli.__file__, "exec"),
        namespace,
    )
    args = SimpleNamespace(
        model="qwen3.5-9b-8bit",
        enable_ddtree=True,
        host="127.0.0.1",
        port=8000,
        served_model_name=None,
        max_tokens=32,
        no_thinking=False,
        **preflight,
    )
    namespace["serve_command"](args)
    preflight_check.assert_called_once_with(args)
    launch.assert_called_once()
    assert launch.call_args.kwargs["drafter_repo"] == preflight.get(
        "_ddtree_drafter_repo", profile.ddtree_draft_model
    )
    assert launch.call_args.kwargs["drafter_revision"] == expected_revision


@pytest.mark.parametrize("override", [None, "user/drafter"])
def test_catalog_ddtree_draft_download_uses_pair_revision(
    monkeypatch, tmp_path, override
):
    from rapid_mlx.cli import (
        _normalize_speculative_config_or_exit,
        _preflight_ddtree_or_exit,
    )

    monkeypatch.setattr(
        "rapid_mlx.speculative.ddtree.eligibility.have_runtime", lambda: True
    )
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    generator = Mock()
    monkeypatch.setitem(
        sys.modules, "dtree_mlx.api", SimpleNamespace(DFlashGenerator=generator)
    )
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("rapid_mlx.model_downloads.snapshot_download", download)
    args = SimpleNamespace(
        model="qwen3.5-9b-8bit",
        _original_alias=None,
        speculative_config='{"method":"ddtree"'
        + (f',"model":"{override}"' if override else "")
        + "}",
        enable_ddtree=False,
        enable_dflash=False,
        spec_decode="none",
        suffix_decoding=False,
        no_spec_decode=False,
    )
    _normalize_speculative_config_or_exit(args)
    _preflight_ddtree_or_exit(args)
    runtime.load_runtime(
        main_model_repo=args._ddtree_profile.hf_path,
        drafter_repo=args._ddtree_drafter_repo,
        drafter_revision=args._ddtree_drafter_revision,
        speculative_tokens=args._ddtree_speculative_tokens,
        tree_budget=args._ddtree_tree_budget,
    )
    if override:
        download.assert_called_once_with(override)
    else:
        download.assert_called_once_with(
            "z-lab/Qwen3.5-9B-DFlash", revision=DRAFT_REVISION
        )
    assert generator.call_args.kwargs["draft_model"] == str(tmp_path)


def test_local_ddtree_draft_needs_no_download(monkeypatch, tmp_path):
    download = Mock(side_effect=AssertionError("Unexpected download"))
    monkeypatch.setattr("rapid_mlx.model_downloads.snapshot_download", download)
    assert runtime._prepare_draft_model_for_dtree(
        str(tmp_path), revision=DRAFT_REVISION
    ) == str(tmp_path)
    download.assert_not_called()


@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40, "A" * 40, 123])
def test_invalid_ddtree_revision_is_rejected(revision):
    with pytest.raises(ValueError, match="ddtree_draft_revision.*40-character"):
        _coerce(
            "test",
            {
                "hf_path": "owner/target",
                "ddtree_draft_model": "owner/draft",
                "ddtree_draft_revision": revision,
            },
        )


def test_ddtree_revision_requires_drafter():
    with pytest.raises(
        ValueError, match="ddtree_draft_revision requires ddtree_draft_model"
    ):
        _coerce(
            "test", {"hf_path": "owner/target", "ddtree_draft_revision": DRAFT_REVISION}
        )


def test_ddtree_drafter_requires_revision():
    with pytest.raises(
        ValueError, match="ddtree_draft_model requires.*ddtree_draft_revision"
    ):
        _coerce(
            "test", {"hf_path": "owner/target", "ddtree_draft_model": "owner/draft"}
        )


def test_catalog_ddtree_drafters_have_revision_pins():
    drafters = {
        alias: profile
        for alias, profile in list_profiles().items()
        if profile.ddtree_draft_model
    }
    assert drafters
    for alias, profile in drafters.items():
        assert profile.ddtree_draft_revision is not None, alias


def test_catalog_ddtree_drafter_is_not_a_servable_alias():
    profile = resolve_profile("qwen3.5-9b-8bit")
    assert profile.ddtree_draft_revision == DRAFT_REVISION
    assert all(
        p.hf_path != profile.ddtree_draft_model for p in list_profiles().values()
    )
