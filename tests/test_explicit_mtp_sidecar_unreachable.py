# SPDX-License-Identifier: Apache-2.0
"""Explicit Hub MTP sidecars fall back without fetching while offline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import constants

from rapid_mlx import cli
from rapid_mlx.model_downloads import ModelDownloadsDisabledError
from rapid_mlx.spec_decode.mtp import qwen3_5_inject

SIDECAR = "mlx-community/Qwen3.5-4B-MTP-4bit"


@pytest.fixture
def preflight(monkeypatch):
    args = SimpleNamespace(
        model="qwen3.5-4b-4bit",
        spec_decode="mtp",
        enable_mtp=True,
        mtp_sidecar=SIDECAR,
        speculative_config='{"method":"mtp"}',
        _speculative_config_explicit=True,
    )
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda _: ({"model_type": "qwen3_5"}, None),
    )
    cached = Mock(return_value=None)
    resolver = Mock(return_value=None)
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", cached)
    monkeypatch.setattr(qwen3_5_inject, "_resolve_sidecar_file", resolver)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    return args, cached, resolver


def _assert_fallback(args, capsys, reason):
    assert capsys.readouterr().err == (
        f"MTP unavailable: the MTP sidecar {SIDECAR} {reason}; using plain decode.\n"
    )
    assert args.spec_decode == "none"
    assert args.enable_mtp is False
    assert args.mtp_sidecar is None
    assert args.speculative_config is None
    assert args._speculative_config is None
    assert args._speculative_config_explicit is False


def test_offline_uncached_sidecar(preflight, monkeypatch, capsys):
    args, cached, resolver = preflight
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert cli._drop_unavailable_mtp_head(args) is True
    cached.assert_called_once_with(SIDECAR)
    resolver.assert_not_called()
    _assert_fallback(
        args,
        capsys,
        "is not downloaded and Hugging Face is offline (HF_HUB_OFFLINE=1)",
    )


@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("unreachable"), ModelDownloadsDisabledError(SIDECAR, "test")],
)
def test_online_unreachable_sidecar(preflight, capsys, error):
    args, _, resolver = preflight
    resolver.side_effect = error
    assert cli._drop_unavailable_mtp_head(args) is True
    resolver.assert_called_once_with(SIDECAR)
    _assert_fallback(args, capsys, "could not be downloaded")


def test_resolver_warning_does_not_duplicate_fallback(preflight, capsys, caplog):
    args, _, resolver = preflight
    filters_before = qwen3_5_inject.logger.filters.copy()

    def fail(sidecar):
        qwen3_5_inject.logger.warning(
            "[mtp.inject] could not resolve sidecar %r: unavailable", sidecar
        )
        return None

    resolver.side_effect = fail
    assert cli._drop_unavailable_mtp_head(args) is True
    _assert_fallback(args, capsys, "could not be downloaded")
    assert not caplog.records
    assert qwen3_5_inject.logger.filters == filters_before


@pytest.mark.parametrize("offline", [False, True])
def test_cached_sidecar_keeps_mtp(preflight, monkeypatch, capsys, offline):
    args, cached, resolver = preflight
    cached.return_value = "/cached/mtp"
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    before = vars(args).copy()
    assert cli._drop_unavailable_mtp_head(args) is False
    assert vars(args) == before
    resolver.assert_not_called()
    assert capsys.readouterr().err == ""


def test_online_resolved_sidecar_keeps_mtp(preflight, capsys):
    args, _, resolver = preflight
    resolver.return_value = Path("/cached/mtp/model.safetensors")
    before = vars(args).copy()
    assert cli._drop_unavailable_mtp_head(args) is False
    assert vars(args) == before
    resolver.assert_called_once_with(SIDECAR)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("offline", [False, True])
def test_automatic_repo_sidecar_unchanged(preflight, monkeypatch, capsys, offline):
    args, cached, resolver = preflight
    args._speculative_config_explicit = False
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    before = vars(args).copy()
    assert cli._drop_unavailable_mtp_head(args) is False
    assert vars(args) == before
    cached.assert_not_called()
    resolver.assert_not_called()
    assert capsys.readouterr().err == ""
