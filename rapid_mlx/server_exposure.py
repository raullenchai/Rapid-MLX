# SPDX-License-Identifier: Apache-2.0
"""Require deliberate consent before serving unauthenticated network clients."""

import ipaddress
import os
import socket


def normalize_bind_host(host):
    """Keep localhost local without consulting mutable DNS/hosts configuration."""
    return (
        "127.0.0.1" if isinstance(host, str) and host.lower() == "localhost" else host
    )


def validate_server_exposure(
    *, host, api_key, allow_unauthenticated_public=False, listen_fd=None
):
    if api_key or allow_unauthenticated_public:
        return
    if listen_fd is not None:
        # Inspect a duplicate: never close the launcher's inherited listener.
        try:
            duplicate = os.dup(listen_fd)
            try:
                listener = socket.socket(fileno=duplicate)
            except OSError:
                os.close(duplicate)
                raise
            with listener:
                if listener.family == socket.AF_UNIX:
                    return
                host = listener.getsockname()[0]
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                "Cannot verify inherited listener; configure an API key"
            ) from exc
    try:
        address = ipaddress.ip_address(host)
        loopback = address.is_loopback or (
            isinstance(address, ipaddress.IPv6Address)
            and address.ipv4_mapped is not None
            and address.ipv4_mapped.is_loopback
        )
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(
            "Non-loopback serving requires --api-key or RAPID_MLX_API_KEY. "
            "To deliberately run behind external authentication, pass "
            "--allow-unauthenticated-public. For local use bind to 127.0.0.1 or ::1."
        )
