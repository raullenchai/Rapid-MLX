# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import ssl

import pytest

from rapid_mlx.share.privacy import clear_legacy_prompt_cache
from rapid_mlx.share.tls import client_context
from rapid_mlx.share.ws_tunnel import TunnelClient


def test_cleanup_only_selected_model_and_all_interrupted_snapshots(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    target = "org--model"
    names = [
        target,
        target + "--0123456789abcdef",
        target + "--0123456789abcdef.new",
        target + ".old",
        target + "-other",
        "unrelated",
    ]
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "tokens.bin").write_bytes(b"private")
    assert clear_legacy_prompt_cache("org/model") == 4
    assert (root / (target + "-other") / "tokens.bin").exists()
    assert (root / "unrelated" / "tokens.bin").exists()
    assert clear_legacy_prompt_cache("org/model") == 0


def test_cleanup_fails_closed_under_contention(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    (root / "model").mkdir(parents=True)
    with (root / "model.txlock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(OSError, match="in use"):
            clear_legacy_prompt_cache("model")
    assert (root / "model").exists()


def test_cleanup_refuses_symlink_without_touching_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "private").write_bytes(b"keep")
    (root / "model").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(OSError, match="symlink"):
        clear_legacy_prompt_cache("model")
    assert (elsewhere / "private").read_bytes() == b"keep"


def test_tls_verification_and_relay_scheme():
    context = client_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    client = TunnelClient(local_port=18765, tunnel_id="test")
    assert client._verified_connect_kwargs("wss://relay.test/up", None)[
        "ssl"
    ].check_hostname
    assert "ssl" not in client._verified_connect_kwargs("ws://localhost/up", None)
