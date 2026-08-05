# SPDX-License-Identifier: Apache-2.0
"""Cursor editor launch adapter for publicly reachable HTTPS endpoints.

Cursor routes BYOK requests through its own backend, so this adapter cannot
point Cursor directly at a server bound to the user's localhost. The launch
adapter enforces that distinction before writing any configuration. A public
HTTPS ``--server-url`` can point at a tunnel forwarding to Rapid-MLX.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from . import _common

_CONFIG_DIR_MAC = Path.home() / "Library" / "Application Support" / "Cursor" / "User"
_CONFIG_DIR_LINUX = Path.home() / ".config" / "Cursor" / "User"
_SETTINGS_FILENAME = "settings.json"


def endpoint_error(server_url: str) -> str | None:
    """Explain why ``server_url`` cannot be reached by Cursor's backend."""
    if "\\" in server_url:
        return "Cursor's --server-url cannot contain backslashes"
    try:
        parsed = urlsplit(server_url)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError:
        return "Cursor requires a valid public hostname and HTTPS port"

    if parsed.scheme.lower() != "https":
        return "Cursor requires a publicly reachable HTTPS --server-url"
    if parsed.query or parsed.fragment:
        return "Cursor's --server-url cannot contain a query string or fragment"
    if username is not None or password is not None:
        return "Cursor's --server-url cannot contain user information"
    if not hostname:
        return "Cursor requires a valid public hostname"
    if port == 0:
        return "Cursor requires a valid non-zero HTTPS port"

    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith((".localhost", ".local")):
        return "Cursor's servers cannot reach localhost or private network hosts"

    try:
        resolved = socket.getaddrinfo(
            normalized,
            port if port is not None else 443,
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        return "Cursor requires a public hostname that resolves successfully"

    addresses = {
        ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        for _family, _type, _proto, _canonname, sockaddr in resolved
    }
    if not addresses or any(
        not address.is_global
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
        for address in addresses
    ):
        return "Cursor's servers cannot reach localhost or private network addresses"
    return None


def canonical_server_url(server_url: str) -> str:
    """Validate and serialize a public Cursor URL without authority ambiguity."""
    reason = endpoint_error(server_url)
    if reason is not None:
        raise ValueError(reason)

    parsed = urlsplit(server_url)
    hostname = parsed.hostname
    assert hostname is not None
    normalized = hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        canonical_host = normalized
    else:
        canonical_host = f"[{normalized}]" if address.version == 6 else normalized
    netloc = canonical_host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit(("https", netloc, parsed.path, "", ""))


def _candidate_dirs() -> list[Path]:
    """Return per-OS Cursor user-settings directories in priority order."""
    return [_CONFIG_DIR_MAC, _CONFIG_DIR_LINUX]


def detect() -> bool:
    """Return whether Cursor appears to be installed."""
    if _common.mac_app_installed("Cursor"):
        return True
    if _common.which("cursor") is not None:
        return True
    return any(directory.exists() for directory in _candidate_dirs())


def current_config_path() -> Path | None:
    """Return Cursor's ``settings.json`` path for this platform."""
    for directory in _candidate_dirs():
        if directory.exists():
            return directory / _SETTINGS_FILENAME
    return _CONFIG_DIR_MAC / _SETTINGS_FILENAME


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Point Cursor at a public HTTPS endpoint forwarding to Rapid-MLX.

    Local, private, malformed, and unauthenticated endpoints are rejected
    before any backup or write. All unrelated Cursor settings are preserved.
    """
    canonical_url = canonical_server_url(server_url)
    if not api_key:
        raise ValueError("Cursor public endpoints require RAPID_MLX_API_KEY")

    parsed = urlsplit(canonical_url)
    path_component = parsed.path.rstrip("/")
    if not path_component.endswith("/v1"):
        path_component += "/v1"
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path_component, "", ""))

    path = config_path or current_config_path()
    assert path is not None

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    existing["cursor.aiprovider.openai.baseUrl"] = base_url
    existing["cursor.aiprovider.openai.apiKey"] = api_key
    existing["cursor.aiprovider.openai.model"] = model

    _common.atomic_write_json(path, existing)
    return path
