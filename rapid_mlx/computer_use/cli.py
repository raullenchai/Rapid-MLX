"""`rapid-mlx computer` — model-agnostic computer-use CLI.

Any agent drives the machine through these commands; the layer never calls a
model. Output is a single JSON object on stdout: {ok: true, ...} or
{ok: false, error: {code, message, recovery[]}} so agents parse one shape.

Commands:
  capabilities / permissions                 TCC status
  list-apps                                  running regular apps
  list-windows --app <app>                   windows of one app
  get-app-state --app <app> [--window-id ID] [--no-screenshot]
  click --app <app> [--window-id ID] (--element-index N | --x N --y N)
         [--click-count N] [--mouse-button left|right|middle]
  set-value --app <app> --element-index N --text V   (read-back verified)
  type-text --app <app> (--text V | --text-stdin)
  press-key --app <app> --key Return|Escape|a|5|...
  hotkey --app <app> --key "Cmd+A"
  scroll --app <app> --direction up|down|left|right [--pages N] [--x --y]
  perform-secondary-action --app <app> --element-index N --action NAME

Prefer the stable window_id returned by list-windows for multi-window apps.
CLI actions return attempted/verified metadata and a fresh post_action_state.
Synthetic input remains unverified unless an exact Accessibility readback exists.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from . import backend
from .errors import ComputerUseError


def _emit(payload: dict, indent: int | None = None) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=indent)
    sys.stdout.write("\n")


def _snapshot_payload(snapshot: dict, include_png: bool) -> dict:
    out = {k: v for k, v in snapshot.items() if k != "screenshot_png"}
    if include_png and snapshot.get("screenshot_png"):
        out["screenshot_png_base64"] = base64.b64encode(
            snapshot["screenshot_png"]
        ).decode()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="computer", description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_window_id(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--window-id",
            default=None,
            help="opaque window_id from list-windows (for example cg:123)",
        )

    sub.add_parser("capabilities", help="Report computer-use provider capabilities")
    sub.add_parser("permissions", help="Report accessibility / screen-recording status")
    sub.add_parser("list-apps", help="List running apps available to computer-use")

    windows = sub.add_parser("list-windows", help="List windows for an app")
    windows.add_argument("--app", required=True)

    state = sub.add_parser(
        "get-app-state", help="Capture a compact AX snapshot of an app"
    )
    state.add_argument("--app", required=True)
    state.add_argument("--window-index", type=int, default=0)
    add_window_id(state)
    state.add_argument("--no-screenshot", action="store_true")
    state.add_argument(
        "--refresh", action="store_true", help="bypass the snapshot cache"
    )
    state.add_argument(
        "--png-out", default=None, help="write the window screenshot to a file"
    )

    click = sub.add_parser("click", help="Click an element or coordinate")
    click.add_argument("--app", required=True)
    add_window_id(click)
    click.add_argument("--element-index", type=int, default=None)
    click.add_argument("--x", type=int, default=None)
    click.add_argument("--y", type=int, default=None)
    click.add_argument("--click-count", type=int, default=1)
    click.add_argument(
        "--mouse-button", choices=("left", "right", "middle"), default="left"
    )

    set_value = sub.add_parser("set-value", help="Write + verify a value on an element")
    set_value.add_argument("--app", required=True)
    add_window_id(set_value)
    set_value.add_argument("--element-index", type=int, required=True)
    set_value.add_argument("--text", default=None)
    set_value.add_argument("--text-stdin", action="store_true")

    typing = sub.add_parser("type-text", help="Type literal text at current focus")
    typing.add_argument("--app", required=True)
    add_window_id(typing)
    typing.add_argument("--text", default=None)
    typing.add_argument("--text-stdin", action="store_true")

    press = sub.add_parser("press-key", help="Press a single key")
    press.add_argument("--app", required=True)
    add_window_id(press)
    press.add_argument("--key", required=True)

    hotkey = sub.add_parser("hotkey", help="Press a modifier chord, e.g. Cmd+A")
    hotkey.add_argument("--app", required=True)
    add_window_id(hotkey)
    hotkey.add_argument("--key", required=True)

    scroll = sub.add_parser("scroll", help="Scroll a window or coordinate")
    scroll.add_argument("--app", required=True)
    add_window_id(scroll)
    scroll.add_argument(
        "--direction", required=True, choices=("up", "down", "left", "right")
    )
    scroll.add_argument("--pages", type=float, default=1.0)
    scroll.add_argument("--x", type=int, default=None)
    scroll.add_argument("--y", type=int, default=None)

    secondary = sub.add_parser(
        "perform-secondary-action", help="Perform an advertised secondary AX action"
    )
    secondary.add_argument("--app", required=True)
    add_window_id(secondary)
    secondary.add_argument("--element-index", type=int, required=True)
    secondary.add_argument("--action", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "capabilities":
            _emit(
                {
                    "ok": True,
                    "provider": "rapid-mlx computer-use (pyobjc AX + CGEvent)",
                    "platform": "darwin",
                    "observation": ["get-app-state", "list-apps", "list-windows"],
                    "actions": [
                        "click",
                        "set-value",
                        "type-text",
                        "press-key",
                        "hotkey",
                        "scroll",
                        "perform-secondary-action",
                    ],
                    "snapshot_cache_ttl_s": backend.SNAPSHOT_TTL_S,
                    "protocol": 1,
                }
            )
            return 0
        if args.subcommand == "permissions":
            _emit({"ok": True, **backend.permissions()})
            return 0
        if args.subcommand == "list-apps":
            _emit({"ok": True, "apps": backend.list_apps()})
            return 0
        if args.subcommand == "list-windows":
            _emit({"ok": True, "windows": backend.list_windows(args.app)})
            return 0
        if args.subcommand == "get-app-state":
            snapshot = backend.get_app_state(
                args.app,
                window_index=args.window_index,
                screenshot=not args.no_screenshot,
                use_cache=not args.refresh,
                window_id=args.window_id,
            )
            if args.png_out and snapshot.get("screenshot_png"):
                Path(args.png_out).write_bytes(snapshot["screenshot_png"])
            _emit(
                {"ok": True, "snapshot": _snapshot_payload(snapshot, include_png=False)}
            )
            return 0
        if args.subcommand == "click":
            if args.mouse_button != "left":
                raise ComputerUseError(
                    "invalid_argument",
                    "right/middle clicks land in a follow-up revision; use left",
                )
            result = backend.click(
                args.app,
                element_index=args.element_index,
                x=args.x,
                y=args.y,
                click_count=args.click_count,
                window_id=args.window_id,
                include_post_state=True,
            )
            _emit({"ok": True, **result})
            return 0
        if args.subcommand == "set-value":
            text = sys.stdin.read() if args.text_stdin else (args.text or "")
            if not text:
                raise ComputerUseError(
                    "invalid_argument", "set-value requires --text or --text-stdin"
                )
            result = backend.set_value(
                args.app,
                args.element_index,
                text,
                window_id=args.window_id,
                include_post_state=True,
            )
            _emit({"ok": True, **result})
            return 0
        if args.subcommand == "type-text":
            text = sys.stdin.read() if args.text_stdin else (args.text or "")
            if not text:
                raise ComputerUseError(
                    "invalid_argument", "type-text requires --text or --text-stdin"
                )
            _emit(
                {
                    "ok": True,
                    **backend.type_text(
                        args.app,
                        text,
                        args.window_id,
                        include_post_state=True,
                    ),
                }
            )
            return 0
        if args.subcommand == "press-key":
            _emit(
                {
                    "ok": True,
                    **backend.press_key(
                        args.app,
                        args.key,
                        args.window_id,
                        include_post_state=True,
                    ),
                }
            )
            return 0
        if args.subcommand == "hotkey":
            _emit(
                {
                    "ok": True,
                    **backend.hotkey(
                        args.app,
                        args.key,
                        args.window_id,
                        include_post_state=True,
                    ),
                }
            )
            return 0
        if args.subcommand == "scroll":
            _emit(
                {
                    "ok": True,
                    **backend.scroll(
                        args.app,
                        args.direction,
                        pages=args.pages,
                        x=args.x,
                        y=args.y,
                        window_id=args.window_id,
                        include_post_state=True,
                    ),
                }
            )
            return 0
        if args.subcommand == "perform-secondary-action":
            _emit(
                {
                    "ok": True,
                    **backend.perform_secondary_action(
                        args.app,
                        args.element_index,
                        args.action,
                        args.window_id,
                        include_post_state=True,
                    ),
                }
            )
            return 0
        raise ComputerUseError(
            "invalid_argument", f"unknown subcommand {args.subcommand!r}"
        )
    except ComputerUseError as exc:
        _emit(exc.to_payload())
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
