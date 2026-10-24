# SPDX-License-Identifier: Apache-2.0
import socket

import pytest

from rapid_mlx.server_exposure import validate_server_exposure


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "::1", "::ffff:127.0.0.1"])
def test_loopback_remains_keyless(host):
    validate_server_exposure(host=host, api_key=None)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "", "192.168.1.2", "localhost", "public.example"]
)
def test_network_requires_consent(host):
    with pytest.raises(ValueError, match="Non-loopback"):
        validate_server_exposure(host=host, api_key=None)
    validate_server_exposure(host=host, api_key="secret")
    validate_server_exposure(host=host, api_key=None, allow_unauthenticated_public=True)


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_actual_inherited_socket_checked_without_closing_it(bind):
    with socket.socket() as listener:
        listener.bind((bind, 0))
        kwargs = dict(host="127.0.0.1", api_key=None, listen_fd=listener.fileno())
        if bind == "0.0.0.0":
            with pytest.raises(ValueError, match="Non-loopback"):
                validate_server_exposure(**kwargs)
        else:
            validate_server_exposure(**kwargs)
        assert listener.getsockname()[1] > 0


def test_invalid_inherited_socket_rejected():
    with pytest.raises(ValueError, match="Cannot verify"):
        validate_server_exposure(host="127.0.0.1", api_key=None, listen_fd=-1)


def test_module_parser_exposes_consent():
    from rapid_mlx.server import _build_parser

    assert (
        _build_parser()
        .parse_args(["--allow-unauthenticated-public"])
        .allow_unauthenticated_public
    )


def test_localhost_normalization_never_resolves_dns(monkeypatch):
    from rapid_mlx.server_exposure import normalize_bind_host

    def forbidden(*args, **kwargs):
        pytest.fail("localhost must not consult DNS")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    for host in ("localhost", "LOCALHOST"):
        assert normalize_bind_host(host) == "127.0.0.1"
        validate_server_exposure(host=normalize_bind_host(host), api_key=None)
    assert normalize_bind_host("::1") == "::1"
    assert normalize_bind_host("public.example") == "public.example"


def test_unix_inherited_socket_remains_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind("listener")
        validate_server_exposure(
            host="0.0.0.0", api_key=None, listen_fd=listener.fileno()
        )
        assert listener.fileno() >= 0


def test_module_refuses_public_without_key_before_startup(monkeypatch, capsys):
    import sys

    from rapid_mlx import server

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["server", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit) as error:
        server.main.__wrapped__()
    assert error.value.code == 2
    assert "Non-loopback" in capsys.readouterr().err


@pytest.mark.parametrize(
    "options", [[], ["--api-key", "example"], ["--allow-unauthenticated-public"]]
)
def test_module_accepted_listener_reaches_configuration(monkeypatch, options):
    import sys

    from rapid_mlx import local_search_runtime, server

    class ConfiguredError(Exception):
        pass

    def configured(*args):
        raise ConfiguredError

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    host = "0.0.0.0" if options else "localhost"
    monkeypatch.setattr(sys, "argv", ["server", "--host", host, *options])
    monkeypatch.setattr(local_search_runtime, "configure", configured)
    with pytest.raises(ConfiguredError):
        server.main.__wrapped__()


def test_cli_refuses_public_before_model_setup(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.cli_parser import build_parser

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    args = build_parser().parse_args(["serve", "example", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit) as error:
        cli.serve_command(args)
    assert error.value.code == 2
    assert "Non-loopback" in capsys.readouterr().err


def test_non_socket_inherited_descriptor_copy_is_closed(tmp_path, monkeypatch):
    import errno
    import os

    from rapid_mlx import server_exposure

    duplicates = []
    real_dup = os.dup

    def duplicate(fd):
        copied = real_dup(fd)
        duplicates.append(copied)
        return copied

    monkeypatch.setattr(server_exposure.os, "dup", duplicate)
    with (tmp_path / "regular-file").open("w+b") as original:
        with pytest.raises(ValueError, match="Cannot verify"):
            validate_server_exposure(
                host="127.0.0.1", api_key=None, listen_fd=original.fileno()
            )
        assert len(duplicates) == 1
        with pytest.raises(OSError) as closed:
            os.fstat(duplicates[0])
        assert closed.value.errno == errno.EBADF
        assert os.fstat(original.fileno()).st_size == 0
