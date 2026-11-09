# SPDX-License-Identifier: Apache-2.0
"""Phone connection discovery and terminal QR output; no engine imports."""

from __future__ import annotations

import ipaddress
import json
import secrets
import shlex
import socket
from urllib.parse import urlencode

import psutil

from rapid_mlx.agents.server_discovery import scan_servers


def usable_ip(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
        return not (
            address.is_loopback
            or address.is_link_local
            or address.is_unspecified
            or address.is_multicast
            or address.is_reserved
        )
    except ipaddress.AddressValueError:
        return False


def lan_addresses() -> list[str]:
    """Prefer the route-selected interface; UDP connect sends no packets."""
    stats = psutil.net_if_stats()
    addresses = sorted(
        {
            entry.address
            for name, entries in psutil.net_if_addrs().items()
            if stats.get(name) and stats[name].isup
            for entry in entries
            if entry.family == socket.AF_INET and usable_ip(entry.address)
        }
    )
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            primary = probe.getsockname()[0]
        if primary in addresses:
            addresses.remove(primary)
            addresses.insert(0, primary)
    except OSError:
        pass
    return addresses


def qr_text(url: str) -> str:
    import segno

    matrix = list(segno.make(url, micro=False).matrix_iter(scale=1, border=4))
    # Dark modules on an explicit white background, independent of terminal theme.
    rows = []
    for y in range(0, len(matrix), 2):
        top = matrix[y]
        bottom = matrix[y + 1] if y + 1 < len(matrix) else [0] * len(top)
        rows.append("".join(" ▄▀█"[2 * bool(a) + bool(b)] for a, b in zip(top, bottom)))
    return "\x1b[30;47m" + "\n".join(rows) + "\x1b[0m"


def connection_info(server) -> tuple[list[str], str]:
    """Read only the selected local serve process; never print its environment."""
    proc = psutil.Process(server.pid)
    listeners = [
        c.laddr.ip
        for c in proc.net_connections(kind="inet")
        if c.status == psutil.CONN_LISTEN and c.laddr.port == server.port
    ]
    tokens = proc.cmdline()
    key = ""
    for i, token in enumerate(tokens):
        if token.startswith("--api-key="):
            key = token.partition("=")[2]
        elif token == "--api-key" and i + 1 < len(tokens):
            key = tokens[i + 1]
    if not key:
        key = proc.environ().get("RAPID_MLX_API_KEY", "")
    return listeners, key


def connect_phone(args) -> None:
    def fail(message: str, restart: str | None = None) -> None:
        payload = {"error": message}
        if restart:
            payload["restart_command"] = restart
        print(
            json.dumps(payload)
            if args.json
            else message + (f"\n{restart}" if restart else "")
        )
        raise SystemExit(1)

    servers = [
        s
        for s in scan_servers(include_lan=True)
        if s.listening and (not args.port or s.port == args.port)
    ]
    if not servers:
        fail(
            "No running Rapid server. Start one: rapid-mlx serve <model> --host 0.0.0.0 --api-key <key>"
        )
    if len(servers) > 1:
        fail(
            "Multiple servers found. Select one with --port: "
            + ", ".join(str(s.port) for s in servers)
        )
    server = servers[0]
    try:
        listeners, key = connection_info(server)
    except (psutil.Error, OSError):
        fail(
            "Cannot inspect the server's listener/key. Use Desktop's Connect phone sheet or run connect as the server's user."
        )
    reachable = [ip for ip in listeners if ip == "0.0.0.0" or usable_ip(ip)]
    if not reachable or not key:
        new_key = secrets.token_urlsafe(32)
        restart = shlex.join(
            [
                "rapid-mlx",
                "serve",
                server.model,
                "--port",
                str(server.port),
                "--host",
                "0.0.0.0",
                "--api-key",
                new_key,
            ]
        )
        fail(
            "Stop the current server, then restart it. "
            + (
                "The phone cannot reach a loopback-only listener."
                if not reachable
                else "Refusing a QR: LAN access requires an API key."
            ),
            restart,
        )
    addresses = lan_addresses()
    override = getattr(args, "host_ip", None)
    if override and not usable_ip(override):
        fail("--host-ip must be a non-loopback, non-link-local IPv4 address.")
    address = override or (addresses[0] if addresses else None)
    if not address:
        fail("No LAN IPv4 address found. Join Wi-Fi or pass --host-ip <IPv4>.")
    if "0.0.0.0" not in reachable and address not in reachable:
        fail(
            "Selected IP is not a server listener. Use --host-ip with: "
            + ", ".join(reachable)
        )
    root = f"http://{address}:{server.port}"
    url = f"{root}/connect#{urlencode({'k': key})}"
    payload = {
        "url": url,
        "base_url": f"{root}/v1",
        "api_key": key,
        "alternatives": [ip for ip in addresses if ip != address],
    }
    if args.json:
        print(json.dumps(payload))
        return
    print(qr_text(url))
    print(
        f"Scan with iPhone Camera: {payload['url']}\nBase URL: {payload['base_url']}\nAPI key: {key}"
    )
    print(
        "Keep your phone on the same Wi-Fi. HTTP traffic is unencrypted; share this QR only with people you trust."
    )
    if payload["alternatives"]:
        print("Other interfaces (use --host-ip): " + ", ".join(payload["alternatives"]))
