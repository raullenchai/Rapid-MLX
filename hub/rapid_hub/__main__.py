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
    up.add_argument("--background-concurrency", type=int, default=1)
    up.add_argument("--journal-max-mb", type=float, default=500)
    up.add_argument("--admission", choices=("auto", "hub", "engine"), default="auto")
    up.add_argument("--queue-max-length", type=int, default=32)
    up.add_argument("--queue-max-wait", type=float, default=120)
    up.add_argument("--job-stale-seconds", type=float, default=600)
    up.add_argument("--discovery-file", type=Path)
    status = commands.add_parser("status")
    status.add_argument("--data-dir", type=Path, default=Path("~/.rapid-hub"))
    args = parser.parse_args()
    if args.command == "status":
        directory = args.data_dir.expanduser()
        try:
            connection = json.loads((directory / "connection.json").read_text())
            url = connection["url"]
            instance = httpx.get(url + "/hub/v1/instance", trust_env=False).json()
            if instance["instance_id"] != connection["instance_id"]:
                raise ValueError("Stale discovery file")
            response = httpx.get(
                url + "/hub/v1/status",
                headers={
                    "Authorization": "Bearer " + (directory / "admin-token").read_text()
                },
                trust_env=False,
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
    hub = Hub(
        engines,
        args.data_dir,
        args.plugins_dir,
        args.host,
        args.port,
        args.background_concurrency,
        journal_max_mb=args.journal_max_mb,
        admission=args.admission,
        queue_max_length=args.queue_max_length,
        queue_max_wait=args.queue_max_wait,
        job_stale_seconds=args.job_stale_seconds,
        discovery_file=args.discovery_file,
    )
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
