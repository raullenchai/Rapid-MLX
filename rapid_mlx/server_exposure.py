# SPDX-License-Identifier: Apache-2.0
"""Require deliberate consent before serving unauthenticated network clients."""

import ipaddress
import os
import socket


def _resolve_api_key(argv_value: str | None) -> str | None:
    """Resolve the effective API key with env-var fallback.

    Argv-inline (``--api-key X``) wins for backwards-compat with
    existing scripts; otherwise we fall back to the ``RAPID_MLX_API_KEY``
    env var. Explicit empty or whitespace-only values are invalid; only
    absent values disable auth. The env-var form keeps keys out of ``argv``
    (visible to ``ps -ef`` for any local user) — this is the path
    rapid-desktop's sidecar shim uses to avoid the codex BLOCKER #3
    "bearer-in-shell-history" leak.

    Exposed at module scope (not buried inside ``main()``) so the
    env-fallback contract is directly unit-testable without booting
    a model — a regression here is the bug the dogfood-v0.8.2 finding
    #3 exposed, so a test-via-the-real-code path matters.
    """
    key = argv_value if argv_value is not None else os.environ.get("RAPID_MLX_API_KEY")
    if key is not None and not key.strip():
        source = "--api-key" if argv_value is not None else "RAPID_MLX_API_KEY"
        raise ValueError(
            f"{source} must be a non-empty string; "
            + (
                "omit the flag"
                if argv_value is not None
                else "unset the environment variable"
            )
            + " to run without authentication"
        )
    return key


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
