"""`rapid-mlx cua` — computer-use agent with configurable slow thinking."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rapid_mlx.cua.config import (
    CUAConfig,
    load_config,
    resolve_planner,
    save_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-mlx cua",
        description="Native-accessibility computer-use agent (fast local thinking, configurable slow thinking)",
    )
    sub = parser.add_subparsers(dest="cua_command", required=True)

    run_p = sub.add_parser("run", help="run the agent loop on an app")
    run_p.add_argument(
        "--app", required=True, help="target app name, e.g. 'Google Chrome'"
    )
    run_p.add_argument("--goal", required=True, help="what the agent should accomplish")
    run_p.add_argument(
        "--planner",
        default="cloud-glm",
        help="slow-thinking preset (cloud-glm | local-27b | local-9b) or a loopback URL",
    )
    run_p.add_argument("--planner-model", default=None, help="model name override")
    run_p.add_argument(
        "--planner-vision",
        action="store_true",
        help="force screenshots to a text-capable planner (ignored for text-only presets)",
    )
    run_p.add_argument("--open-url", default="", help="URL to open before starting")
    run_p.add_argument("--allowed-domain", default="", help="hard domain guard")
    run_p.add_argument("--max-steps", type=int, default=None)
    run_p.add_argument(
        "--human-login",
        action="store_true",
        help="pause with a file sentinel when a sign-in page appears",
    )
    run_p.add_argument(
        "--no-fast-ranker",
        action="store_true",
        help="disable the local laya outcome ranker",
    )

    cfg_p = sub.add_parser("config", help="show or edit configuration")
    cfg_p.add_argument("--show", action="store_true")
    cfg_p.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="e.g. --set presets.local-9b.url http://127.0.0.1:18702/v1/chat/completions",
    )

    sub.add_parser("planners", help="list available slow-thinking planners")
    return parser


def _cmd_config(args) -> int:
    config = load_config()
    if args.set:
        for key, value in args.set:
            parts = key.split(".")
            if len(parts) == 3 and parts[0] == "presets":
                config["presets"].setdefault(parts[1], {})[parts[2]] = value
            elif key in {"fast_ranker_url"}:
                config[key] = value
            else:
                print(f"unknown config key: {key}", file=sys.stderr)
                return 2
        save_config(config)
    if args.show or not args.set:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def _cmd_planners() -> int:
    config = load_config()
    for name, preset in sorted(config["presets"].items()):
        print(
            f"{name:12s} {preset['model']:44s} {preset['url']}  # {preset.get('note', '')}"
        )
    return 0


def _cmd_run(args) -> int:
    try:
        planner_cfg = resolve_planner(args.planner, model_override=args.planner_model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.planner_vision:
        planner_cfg.text_only = False
    config = CUAConfig(
        planner=planner_cfg,
        allowed_domain=args.allowed_domain,
        human_login=args.human_login,
        fast_ranker_url="" if args.no_fast_ranker else load_config()["fast_ranker_url"],
    )
    print(
        f"[cua] app={args.app!r} planner={config.planner.describe()} "
        f"max_steps={args.max_steps or config.max_steps} domain={args.allowed_domain or '-'}",
        flush=True,
    )
    from rapid_mlx.cua.loop import run

    trace = asyncio.run(
        run(
            config,
            args.app,
            args.goal,
            open_url=args.open_url,
            max_steps=args.max_steps,
        )
    )
    status = trace.get("status", "incomplete")
    if status == "done":
        print(f"OK: {trace.get('final_summary', '')}")
        return 0
    print(
        f"NOT DONE ({status}): {trace.get('guard_stop') or trace.get('consent_stop') or trace.get('stalled') or 'see trace'}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cua_command == "config":
        return _cmd_config(args)
    if args.cua_command == "planners":
        return _cmd_planners()
    if args.cua_command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
