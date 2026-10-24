# SPDX-License-Identifier: Apache-2.0
"""Startup must stop at trust failures before loading or exposing a server."""

import socket
import sys
from types import SimpleNamespace

import pytest


def test_unified_serve_rejects_public_bind_before_loading(monkeypatch, capsys):
    from rapid_mlx import cli, log_file

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(log_file, "resolve_validated", lambda _path: (None, None))
    monkeypatch.setattr(
        cli,
        "_ensure_model_downloaded",
        lambda *_a: pytest.fail("model download started"),
    )
    args = SimpleNamespace(host="0.0.0.0", api_key=None)
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 2
    assert "Non-loopback serving requires --api-key" in capsys.readouterr().err


def test_standalone_serve_rejects_public_bind_before_loading(monkeypatch, capsys):
    from rapid_mlx import server

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["rapid_mlx.server", "--host", "0.0.0.0"])
    monkeypatch.setattr(
        server, "load_model", lambda *_a, **_kw: pytest.fail("model loading started")
    )
    with pytest.raises(SystemExit) as exc:
        server.main.__wrapped__()
    assert exc.value.code == 2
    assert "Non-loopback serving requires --api-key" in capsys.readouterr().err


def test_inherited_unix_socket_remains_usable_without_network_consent():
    from rapid_mlx.server_exposure import validate_server_exposure

    first, second = socket.socketpair()
    with first, second:
        validate_server_exposure(host="0.0.0.0", api_key=None, listen_fd=first.fileno())
        second.sendall(b"still open")
        assert first.recv(10) == b"still open"


def test_prefetch_integrity_failure_is_not_downgraded_to_loader_retry(
    monkeypatch, capsys
):
    import huggingface_hub

    from rapid_mlx import cli, model_downloads
    from rapid_mlx.model_integrity import ModelIntegrityError

    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda *_a, **_kw: SimpleNamespace(sha="a" * 40, siblings=[]),
    )
    failure = ModelIntegrityError("downloaded tensor digest mismatch")

    def reject(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(model_downloads, "snapshot_download", reject)
    with pytest.raises(ModelIntegrityError) as exc:
        cli._ensure_model_downloaded("example/unregistered-model")
    assert exc.value is failure
    output = capsys.readouterr()
    assert "server will retry" not in output.out + output.err
