"""A denied cache must not masquerade as an empty successful inventory."""

import errno
import os
from types import SimpleNamespace

import pytest

from rapid_mlx import cli


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EACCES])
@pytest.mark.parametrize("json_command", [False, True])
def test_cache_permission_error_is_preserved(monkeypatch, error_number, json_command):
    from huggingface_hub import constants

    from rapid_mlx import local_models

    cache = "/fake/removable/hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", cache)
    original_stat, original_listdir = os.stat, os.listdir

    def stat(path, *args, **kwargs):
        if os.fspath(path) == cache:
            raise PermissionError(error_number, "cache permission denied", cache)
        return original_stat(path, *args, **kwargs)

    def listdir(path):
        if os.fspath(path) == cache:
            raise PermissionError(error_number, "cache permission denied", cache)
        return original_listdir(path)

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(local_models, "discover_runnable_chat_models", lambda: [])
    with pytest.raises(PermissionError) as failure:
        if json_command:
            cli.models_command(SimpleNamespace(json=True, cached=True))
        else:
            cli._scan_hf_cache_models()
    assert failure.value.errno == error_number


@pytest.mark.parametrize("error", [FileNotFoundError, NotADirectoryError])
def test_missing_cache_is_still_empty(monkeypatch, error):
    from huggingface_hub import constants

    cache = "/fake/missing/hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", cache)
    original = os.listdir

    def listdir(path):
        if os.fspath(path) == cache:
            raise error(path)
        return original(path)

    monkeypatch.setattr(os, "listdir", listdir)
    assert cli._scan_hf_cache_models() == []
