# SPDX-License-Identifier: Apache-2.0
"""``rmlx-web`` entry point."""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
from urllib.parse import quote

import uvicorn

from . import __version__, auth
from .app import WebConfig, create_app
from .catalog import ModelCatalog
from .connectors import ConnectorStore
from .downloads import DownloadManager
from .supervisor import (
    AttachedEngine,
    EngineSupervisor,
    SupervisorError,
    find_rapid_mlx_binary,
)

DEFAULT_PORT = 7788


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmlx-web",
        description=(
            "Serve a mobile-friendly web UI for Rapid-MLX. Point your own "
            "tunnel (cloudflared / tailscale funnel / frp) at it to reach it "
            "from a phone."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help=(
            "Model alias to load at startup. Optional: omit it to start with "
            "no model and choose one in the page."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Address to bind. Defaults to 127.0.0.1. Every tunnel connects to "
            "a local port, so loopback covers the remote-access case; "
            "0.0.0.0 additionally exposes this to everyone on the current "
            "network, which on a cafe or hotel network is effectively public."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})."
    )
    parser.add_argument(
        "--attach",
        metavar="URL",
        help=(
            "Use an already-running `rapid-mlx serve` instead of starting one. "
            "Model switching is unavailable in this mode. This cannot attach "
            "to the Rapid-MLX Desktop app's engine: that bearer is generated "
            "per launch and is not obtainable by another process."
        ),
    )
    parser.add_argument(
        "--attach-api-key",
        metavar="KEY",
        help="Bearer for the --attach target, if it was started with one.",
    )
    parser.add_argument(
        "--token",
        metavar="TOKEN",
        help=(
            "Use this access token instead of the generated token stored at "
            "~/.rapid-mlx/web-token."
        ),
    )
    parser.add_argument(
        "--new-token",
        action="store_true",
        help=(
            "Require an access token, generating and storing one at "
            "~/.rapid-mlx/web-token. Reuse it on later runs with --token. "
            "Repeating this rotates it, and existing phones must re-enter it."
        ),
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help=(
            "Permit starting model downloads from the web UI when not bound "
            "to loopback. Downloads are already enabled on loopback; this "
            "flag only lifts the restriction that applies once the port is "
            "reachable from the network."
        ),
    )
    parser.add_argument(
        "--rapid-mlx-bin",
        metavar="PATH",
        help="Path to the `rapid-mlx` command, if it is not on PATH.",
    )
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Extra argument forwarded verbatim to `rapid-mlx serve`. Repeat "
            "for each token, e.g. --serve-arg --max-model-len --serve-arg 8192."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _is_loopback(host: str) -> bool:
    """Whether binding ``host`` keeps the surface off the network.

    Not a string comparison against "127.0.0.1": the entire 127/8 block
    is loopback, and "localhost" may resolve to ::1. Getting this wrong
    in the permissive direction would silently skip the exposure
    warning, so unparseable names are treated as non-loopback.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _display_host(host: str) -> str:
    """Host to print in the banner.

    A wildcard bind is not a reachable address, so echoing "0.0.0.0" back
    at the user produces a URL that does not work. Substitute the LAN
    address they most likely want.
    """
    if host not in ("0.0.0.0", "::"):
        return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # No packet is sent: connect() on UDP only selects a route,
            # which is enough to learn the local address.
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return "localhost"


def _login_url(host: str, port: int, token: str) -> str:
    """URL that logs the browser in without retyping the token.

    The token goes in the **fragment**, never the query string: a
    fragment is not sent to the server, so it cannot land in an access
    log, a proxy log, or a tunnel provider's request history. The page
    reads it, stores it, and strips it from the address bar.
    """
    base = f"http://{_display_host(host)}:{port}/"
    return f"{base}#token={quote(token, safe='')}"


def _print_banner(*, host: str, port: int, token: str, loopback: bool) -> None:
    url = f"http://{_display_host(host)}:{port}/"
    login_url = _login_url(host, port, token)

    print()
    print("  rmlx-web")
    print(f"  URL:   {url}")
    print(f"  Token: {token}")
    print()

    # No QR code: a 25-row block pushed everything above it — including
    # the token — off a short terminal window. The link stays when there
    # is a token, since its fragment saves retyping 43 characters.
    print("  Open this link to sign in automatically:")
    print(f"  {login_url}")
    print()
    # stdout is block-buffered when not a TTY, so under `rmlx-web > log &`
    # the token would not appear until the buffer filled.
    sys.stdout.flush()


def _resolve_engine(
    args: argparse.Namespace, *, downloads_enabled: bool, connectors: ConnectorStore
):
    if args.attach:
        if args.model:
            raise SystemExit(
                "error: pass a model or --attach, not both. --attach uses the "
                "model already loaded by the running server."
            )
        # No catalog: listing aliases needs the `rapid-mlx` CLI, and an
        # attached engine may be the only rapid-mlx on this machine.
        # Switching is impossible here anyway.
        return AttachedEngine(args.attach, api_key=args.attach_api_key), None, None

    binary = find_rapid_mlx_binary(args.rapid_mlx_bin)
    engine = EngineSupervisor(
        binary=binary,
        # NOT the web token: keeping them separate means a leaked web
        # token cannot be replayed against the engine, and the web token
        # never reaches the engine's logs.
        api_key=auth.generate_token(),
        serve_args=list(args.serve_arg),
        # Re-read at every spawn, so a connector added an hour into the
        # session is armed by the next model start.
        mcp_config_path=connectors.launch_config_path,
    )
    downloads = DownloadManager(binary) if downloads_enabled else None
    return engine, ModelCatalog(binary), downloads


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    loopback = _is_loopback(args.host)

    # ON for loopback, OFF otherwise: once the port is on a network a
    # download is an endpoint a stranger can use to fill someone else's
    # disk, so it has to be asked for explicitly.
    downloads_enabled = loopback or args.allow_downloads
    if not loopback and args.allow_downloads:
        print(
            "  NOTE: downloads are enabled on a non-loopback address. "
            "Anyone who can reach this port can fill this Mac's disk.\n",
            file=sys.stderr,
        )

    # Always keep a first-party boundary. A tunnel may terminate on this
    # loopback listener, so loopback is not evidence that callers are local.
    try:
        token = auth.load_or_create_token(
            override=args.token,
            rotate=args.new_token,
        )
    except OSError as exc:
        print(
            f"error: could not read or create the token file: {exc}",
            file=sys.stderr,
        )
        return 1

    # Constructed before the engine: the supervisor reads its
    # `launch_config_path` on every spawn.
    connector_store = ConnectorStore()

    try:
        engine, catalog, downloads = _resolve_engine(
            args, downloads_enabled=downloads_enabled, connectors=connector_store
        )
    except SupervisorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = WebConfig(
        token=token,
        engine=engine,
        initial_model=args.model if not args.attach else None,
        catalog=catalog,
        downloads=downloads,
        connectors=connector_store,
    )
    app = create_app(config)

    _print_banner(host=args.host, port=args.port, token=token, loopback=loopback)
    if config.initial_model:
        print(f"  Loading {config.initial_model} in the background…")
        print("  The page is usable now; it will say when the model is ready.\n")
        sys.stdout.flush()
    elif not args.attach:
        # Without this the banner is followed by silence, which reads as
        # "still working" rather than "waiting for you".
        print("  No model loaded — choose one in the page to start it.\n")
        sys.stdout.flush()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
