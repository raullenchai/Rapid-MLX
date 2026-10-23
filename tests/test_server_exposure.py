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
