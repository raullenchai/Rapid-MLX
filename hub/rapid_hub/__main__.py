"""rapid-hub command line."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn

from .app import Hub, create_app


def main():
    parser = argparse.ArgumentParser(description="Local Rapid Hub")
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("up")
    up.add_argument("--engine", action="append", default=[])
    up.add_argument("--host", default="127.0.0.1")
    up.add_argument("--port", type=int, default=4800)
    up.add_argument("--data-dir", type=Path, default=Path("~/.rapid-hub"))
    up.add_argument("--plugins-dir", default="apps")
    up.add_argument(
        "--plugin-port",
        action="append",
        default=[],
        metavar="PLUGIN=PORT",
        help="Fixed UI plugin port (repeatable; e.g. research-loop=4910)",
    )
    up.add_argument("--background-concurrency", type=int, default=1)
    up.add_argument("--journal-max-mb", type=float, default=500)
    up.add_argument("--journal-queue-max-ops", type=int, default=256)
    up.add_argument("--journal-queue-max-bytes", type=int, default=8 * 1024 * 1024)
    up.add_argument("--admission", choices=("auto", "hub", "engine"), default="auto")
    up.add_argument("--queue-max-length", type=int, default=32)
    up.add_argument("--queue-max-wait", type=float, default=120)
    up.add_argument("--job-stale-seconds", type=float, default=600)
    up.add_argument("--discovery-file", type=Path)
    status = commands.add_parser("status")
    status.add_argument("--data-dir", type=Path, default=Path("~/.rapid-hub"))
    login = commands.add_parser("login")
    login.add_argument("--data-dir", type=Path, default=Path("~/.rapid-hub"))
    args = parser.parse_args()
    if args.command in ("status", "login"):
        directory = args.data_dir.expanduser()
        try:
            connection = json.loads((directory / "connection.json").read_text())
            url = connection["url"]
            instance = httpx.get(url + "/hub/v1/instance", trust_env=False).json()
            if instance["instance_id"] != connection["instance_id"]:
                raise ValueError("Stale discovery file")
            token_path = directory / "admin-token"
            if token_path.stat().st_mode & 0o777 != 0o600:
                raise ValueError("Admin token must have mode 0600")
            headers = {"Authorization": "Bearer " + token_path.read_text()}
            if args.command == "login":
                response = httpx.post(
                    url + "/hub/v1/login-nonce", headers=headers, trust_env=False
                )
                response.raise_for_status()
                print(response.json()["url"])
            else:
                response = httpx.get(
                    url + "/hub/v1/status", headers=headers, trust_env=False
                )
                response.raise_for_status()
                print(json.dumps(response.json(), indent=2))
        except (OSError, ValueError, httpx.HTTPError) as exc:
            parser.exit(1, f"Hub unavailable: {type(exc).__name__}\n")
        return
    engines = {}
    for value in args.engine:
        name, sep, url = value.partition("=")
        parsed = urlsplit(url)
        if (
            not sep
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
            or parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/v1")
            or name in engines
        ):
            parser.error(
                "--engine requires unique name=http://host:port/v1 without credentials"
            )
        engines[name] = url.rstrip("/")
    if args.background_concurrency < 1:
        parser.error("--background-concurrency must be positive")
    plugin_ports = {}
    for value in args.plugin_port:
        ident, sep, raw_port = value.partition("=")
        if (
            not sep
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", ident)
            or not raw_port.isascii()
            or not raw_port.isdecimal()
            or not 1 <= int(raw_port) <= 65535
            or ident in plugin_ports
        ):
            parser.error("--plugin-port requires unique PLUGIN=PORT with port 1–65535")
        plugin_ports[ident] = int(raw_port)
    try:
        hub = Hub(
            engines,
            args.data_dir,
            args.plugins_dir,
            args.host,
            args.port,
            args.background_concurrency,
            journal_max_mb=args.journal_max_mb,
            journal_queue_max_ops=args.journal_queue_max_ops,
            journal_queue_max_bytes=args.journal_queue_max_bytes,
            admission=args.admission,
            queue_max_length=args.queue_max_length,
            queue_max_wait=args.queue_max_wait,
            job_stale_seconds=args.job_stale_seconds,
            discovery_file=args.discovery_file,
            plugin_ports=plugin_ports,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"Rapid Hub login (one use): {hub.dashboard_url}/login?nonce={hub.login}",
        flush=True,
    )
    # Access logging is disabled so the login nonce never enters server logs.
    # Keep process supervision on the stdlib event loop used by the tests.
    uvicorn.run(
        create_app(hub),
        host=args.host,
        port=args.port,
        access_log=False,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
