# SPDX-License-Identifier: Apache-2.0
"""Tests for #2145 ``rapid-mlx pull --bits/--format`` variant selection.

A multi-variant repo ships every quantization side by side as top-level
folders (e.g. ``LiquidAI/LFM2.5-2.6B-MLX`` holds ``4bit/ 5bit/ 6bit/ 8bit/
mxfp4/ ...``). Without selection ``pull <repo>`` fetches ALL of them. These
tests pin that ``--bits N`` / ``--format F``:

1. narrow the HuggingFace ``snapshot_download`` to exactly that variant's
   ``allow_patterns`` (``["4bit/*"]`` etc.),
2. bypass the whole-repo R2 mirror prefetch when a variant is requested (the
   mirror has no narrow-to-variant mode; Vector's #2279 is unlanded) with a
   clear "mirror skipped" line,
3. fail loudly with the available folders listed when the variant does not
   exist — before any weight download,
4. leave the existing mirror-first behavior untouched when no selector is
   given,
5. let the user flag override catalog ``resolve_subfolder`` narrowing (said so).

The HuggingFace file listing (``list_repo_tree``), the R2 prefetch
(``_try_mirror_prefetch``), the catalog narrowing (``resolve_subfolder``) and
``snapshot_download`` are all mocked; only the selection logic in
``pull_command`` / ``_resolve_variant_allow_patterns`` is exercised — no
weights are ever touched.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest
from huggingface_hub import RepoFile, RepoFolder

from vllm_mlx import cli


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` when constructing HF errors."""

    status_code = 404
    headers = {}
    request = None
    url = ""


def _multi_variant_tree():
    return [
        RepoFolder(path="4bit", oid="a"),
        RepoFolder(path="8bit", oid="b"),
        RepoFile(path="README.md", size=100, oid="c"),
    ]


def _pull_capturing(**flags):
    """Drive ``pull_command`` with mocked listing+mirror, returning the
    ``allow_patterns`` passed to ``snapshot_download`` and the mirror call
    count via a mutable dict ``side``."""
    side = {}
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", **flags)

    def fake_snapshot(*a, **kw):
        side["allow"] = kw.get("allow_patterns")
        return "/cache/snapshot"

    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False) as mirror,
        patch("huggingface_hub.snapshot_download", fake_snapshot),
    ):
        cli.pull_command(args)
        side["mirror_calls"] = len(mirror.call_args_list)
    return side


def test_bits_narrows_snapshot_and_bypasses_mirror(capsys):
    side = _pull_capturing(bits="4", format=None)
    assert side["allow"] == ["4bit/*"]
    # An explicit variant bypasses the whole-repo mirror prefetch (option B).
    assert side["mirror_calls"] == 0
    # Refinement (1): a clear line explains that the mirror was skipped.
    out = capsys.readouterr().out
    assert "R2 mirror skipped" in out


def test_format_narrows_snapshot_by_folder_name():
    # "8bit" is both a valid format-style folder key here.
    side = _pull_capturing(bits=None, format="8bit")
    assert side["allow"] == ["8bit/*"]


def test_resolve_returns_none_without_selector():
    assert cli._resolve_variant_allow_patterns("X/Y", None, None) is None


def test_escape_glob_literal_metachars():
    """A folder name with glob metachars must match literally, not broaden."""
    assert cli._escape_glob_literal("4bit") == "4bit"
    assert cli._escape_glob_literal("[48]bit") == "[[]48[]]bit"
    assert cli._escape_glob_literal("a*b?") == "a[*]b[?]"


def test_variant_with_glob_metachars_escaped_in_allow_patterns(capsys):
    """--format on a folder named with glob metachars pins to that folder."""
    tricky_tree = [
        RepoFolder(path="[48]bit", oid="a"),
        RepoFolder(path="8bit", oid="b"),
        RepoFile(path="README.md", size=100, oid="c"),
    ]
    args = argparse.Namespace(model="X/Y", bits=None, format="[48]bit")
    with (
        patch("huggingface_hub.HfApi.list_repo_tree", return_value=tricky_tree),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", return_value="/cache/x") as snap,
    ):
        cli.pull_command(args)
    # The pattern must not glob-broaden: the literal folder with brackets.
    assert snap.call_args.kwargs["allow_patterns"] == ["[[]48[]]bit/*"]
    # The user-facing message must show the real folder name, not the escaping.
    out = capsys.readouterr().out
    assert "[48]bit" in out
    assert "[[]48[]]bit" not in out


def test_parser_accepts_single_selector():
    """The pull parser exposes --bits/--format and parses a single one."""
    parser = cli.build_parser()
    ns = parser.parse_args(["pull", "--bits", "4", "LiquidAI/LFM2.5-2.6B-MLX"])
    assert ns.bits == "4"
    assert ns.model == "LiquidAI/LFM2.5-2.6B-MLX"
    ns = parser.parse_args(["pull", "--format", "mxfp4", "LiquidAI/LFM2.5-2.6B-MLX"])
    assert ns.format == "mxfp4"


def test_parser_rejects_both_selectors():
    """--bits and --format are mutually exclusive; argparse rejects both."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["pull", "--bits", "4", "--format", "gguf", "LiquidAI/LFM2.5-2.6B-MLX"]
        )


def test_resolve_rejects_both_selectors():
    """--bits and --format pick the same single variant; both is ambiguous."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli._resolve_variant_allow_patterns("X/Y", bits="4", fmt="gguf")


def test_resolve_rejects_empty_selector():
    """An explicit-but-empty selector is a user error, not 'no selector'."""
    with pytest.raises(ValueError, match="empty"):
        cli._resolve_variant_allow_patterns("X/Y", bits=None, fmt="")


def test_resolve_reraise_repo_not_found():
    """A missing repo passes the cause through (caller surfaces it)."""
    from huggingface_hub.errors import RepositoryNotFoundError

    err = RepositoryNotFoundError("no repo", response=_FakeResponse())
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree",
            side_effect=err,
        ),
        pytest.raises(RepositoryNotFoundError),
    ):
        cli._resolve_variant_allow_patterns("nope/nope", bits="4", fmt=None)


def test_empty_selector_errors_cleanly(capsys):
    """pull --format \"\" exits with a clear message, not an unrestricted pull."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format="")
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch.object(cli.sys, "exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        cli.pull_command(args)
    assert "empty" in capsys.readouterr().out


def test_missing_variant_shows_available_folders(capsys):
    """--format gguf when the repo has no gguf/ fails loudly, listing folders."""
    args = argparse.Namespace(
        model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format="gguf"
    )
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch.object(cli.sys, "exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        cli.pull_command(args)
    out = capsys.readouterr().out
    assert "no 'gguf' variant" in out
    assert "4bit" in out  # lists an available folder
    assert "8bit" in out


def test_missing_variant_uses_original_alias(capsys):
    """Error message names the caller-supplied alias, not the resolved repo."""
    args = argparse.Namespace(
        model="LiquidAI/LFM2.5-2.6B-MLX",
        bits=None,
        format="gguf",
        _original_alias="my-alias",
    )
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch.object(cli.sys, "exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        cli.pull_command(args)
    out = capsys.readouterr().out
    assert "my-alias" in out
    assert "has no 'gguf' variant" in out


def test_missing_variant_no_folders(capsys):
    """A single-variant repo (no folders) says so instead of listing folders."""
    args = argparse.Namespace(
        model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format="gguf"
    )
    flat = [RepoFile(path="model.safetensors", size=100, oid="a")]
    with (
        patch("huggingface_hub.HfApi.list_repo_tree", return_value=flat),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch.object(cli.sys, "exit", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        cli.pull_command(args)
    out = capsys.readouterr().out
    assert "single-variant repo" in out


def test_no_selector_still_consults_mirror():
    """Without --bits/--format the mirror-first behavior is unchanged."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format=None)
    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=True) as mirror,
        patch("huggingface_hub.snapshot_download"),
    ):
        cli.pull_command(args)
    assert mirror.call_count == 1


def test_orphan_reap_announces_cleaned_files(capsys):
    """Pull announces when it reclaims abandoned download scratch files."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format=None)
    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch(
            "vllm_mlx._download_gate.reap_orphan_incomplete_blobs",
            return_value=(2, 512),
        ),
        patch(
            "vllm_mlx._download_gate._format_size",
            return_value="512 B",
        ),
        patch("huggingface_hub.snapshot_download", return_value="/cache/x"),
    ):
        cli.pull_command(args)
    out = capsys.readouterr().out
    assert "Cleaned up 2 abandoned download file(s)" in out


def test_catalog_subfolder_narrowing_when_no_selector(capsys):
    """No selector + a catalog subfolder narrows to it (existing behavior)."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format=None)
    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("vllm_mlx.model_aliases.resolve_subfolder", return_value="4bit"),
        patch("huggingface_hub.snapshot_download", return_value="/cache/x") as snap,
    ):
        cli.pull_command(args)
    # snapshot_download called with allow_patterns=["4bit/*"].
    assert snap.call_args.kwargs["allow_patterns"] == ["4bit/*"]
    assert "ships one checkpoint per quantization" in capsys.readouterr().out


def test_user_flag_overrides_catalog_subfolder(capsys):
    """--bits/--format wins over catalog resolve_subfolder narrowing (said so)."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits="8", format=None)
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("vllm_mlx.model_aliases.resolve_subfolder", return_value="4bit"),
        patch("huggingface_hub.snapshot_download", return_value="/cache/x") as snap,
    ):
        cli.pull_command(args)
    assert snap.call_args.kwargs["allow_patterns"] == ["8bit/*"]
    out = capsys.readouterr().out
    assert "8bit" in out
    assert "overrides" in out
    assert "4bit" in out


def test_user_flag_equal_to_catalog_subfolder_no_override(capsys):
    """When the flag matches the catalog subfolder, no override line is shown."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits="4", format=None)
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("vllm_mlx.model_aliases.resolve_subfolder", return_value="4bit"),
        patch("huggingface_hub.snapshot_download", return_value="/cache/x") as snap,
    ):
        cli.pull_command(args)
    assert snap.call_args.kwargs["allow_patterns"] == ["4bit/*"]
    assert "overrides" not in capsys.readouterr().out
