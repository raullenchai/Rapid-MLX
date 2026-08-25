# SPDX-License-Identifier: Apache-2.0
"""Tests for #2145 ``rapid-mlx pull --bits/--format`` variant selection.

A multi-variant repo ships every quantization side by side as top-level
folders (e.g. ``LiquidAI/LFM2.5-2.6B-MLX`` holds ``4bit/ 5bit/ 6bit/ 8bit/
mxfp4/ ...``). Without selection ``pull <repo>`` fetches ALL of them. These
tests pin that ``--bits N`` / ``--format F``:

1. narrow the HuggingFace ``snapshot_download`` to exactly that variant's
   ``allow_patterns`` (``["4bit/*"]`` etc.),
2. bypass the whole-repo R2 mirror prefetch when a variant is requested (the
   mirror has no narrow-to-variant mode; Vector's #2279 is unlanded),
3. fail loudly with the available folders listed when the variant does not
   exist — before any weight download,
4. leave the existing mirror-first behavior untouched when no selector is
   given.

The HuggingFace file listing (``list_repo_tree``), the R2 prefetch
(``_try_mirror_prefetch``) and ``snapshot_download`` are all mocked; only the
selection logic in ``pull_command`` / ``_resolve_variant_allow_patterns`` is
exercised — no weights are ever touched.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest
from huggingface_hub import RepoFile, RepoFolder

from vllm_mlx import cli


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
    assert "mirror was skipped" in out or "R2 mirror skipped" in out


def test_format_narrows_snapshot_by_folder_name():
    # "8bit" is both a valid format-style folder key here.
    side = _pull_capturing(bits=None, format="8bit")
    assert side["allow"] == ["8bit/*"]


def test_resolve_returns_none_without_selector():
    assert cli._resolve_variant_allow_patterns("X/Y", None, None) is None


def test_resolve_rejects_both_selectors():
    """--bits and --format pick the same single variant; both is ambiguous."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli._resolve_variant_allow_patterns("X/Y", bits="4", fmt="gguf")


def test_no_selector_still_consults_mirror():
    """Without --bits/--format the mirror-first behavior is unchanged."""
    args = argparse.Namespace(model="LiquidAI/LFM2.5-2.6B-MLX", bits=None, format=None)
    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=True) as mirror,
        patch("huggingface_hub.snapshot_download"),
    ):
        cli.pull_command(args)
    assert mirror.call_count == 1


def test_missing_variant_errors_with_available_folders(capsys):
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
