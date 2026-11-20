# SPDX-License-Identifier: Apache-2.0
"""CLI history readers use discovery and the private socket, even during restart."""

import http.client
import json
import sys
import time
from datetime import datetime

from .history import History
from .protocol import discover, discovery_root, request


def connection(port=None):
    statuses = discover(port)
    if not statuses:
        raise OSError("No supervised Rapid-MLX instance found.")
    # Discovery already checks directory ownership, permissions and liveness.
    for entry in sorted(discovery_root().glob("*.json")):
        try:
            data = json.loads(entry.read_text())
            status = request(data["socket"], data["secret"])
            if status["supervisor_pid"] == statuses[0]["supervisor_pid"]:
                return data
        except (OSError, ValueError, KeyError, http.client.HTTPException):
            continue
    raise OSError("Control plane unavailable")


def table_row(item, command):
    value = lambda key: "-" if item.get(key) is None else str(item[key])
    stamp = (
        datetime.fromtimestamp(item["ts"]).astimezone()
        if item.get("ts") is not None
        else None
    )
    today = datetime.now().astimezone().date()
    clock = (
        "-"
        if stamp is None
        else stamp.strftime(
            "%H:%M:%S" if stamp.date() == today else "%Y-%m-%d %H:%M:%S"
        )
    )
    if command == "events":
        details = " ".join(
            f"{k}={v}" for k, v in item.items() if k not in {"ts", "event"}
        )
        return f"{clock:<19} {value('event'):<22} {details}"
    model = value("model").rsplit("/", 1)[-1]
    integer = lambda key: "-" if item.get(key) is None else str(int(item[key]))
    tokens = f"{value('prompt_tokens')}→{value('completion_tokens')}"
    return (
        f"{clock:<19} {value('client')[:24]:<24} {model[:28]:<28} "
        f"{value('route'):<30} {value('http_status'):<6} {value('outcome'):<9} "
        f"{tokens:<11} {integer('ttft_ms'):>7} {integer('duration_ms'):>8}"
    )


def history_command(args):
    try:
        try:
            info = connection(args.port)
        except OSError:
            info = None
        offline = None
        if info is None:
            root = discovery_root().parent / "history"
            offline = History(root, readonly=getattr(args, "action", None) != "clear")

        def query(method="GET", *, target):
            if offline is not None:
                return offline.api(method, target, {})[1]
            return request(info["socket"], info["secret"], method, target=target)

        target = "/rapid/v1/" + args.command
        if getattr(args, "action", None) == "clear":
            query("DELETE", target=target)
            return 0
        cursor = 0
        limit = max(1, min(1000, args.limit))
        paging = args.command == "log" and args.follow
        if not args.json:
            print(
                "TIME                EVENT                  DETAILS"
                if args.command == "events"
                else "TIME                CLIENT                   MODEL                        ROUTE                          STATUS OUTCOME   TOKENS      TTFT ms TOTAL ms",
                flush=True,
            )
        while True:
            result = query(
                target=f"{target}?{'after' if paging else 'since'}={cursor}&limit={limit}"
                + (f"&port={args.port}" if args.port is not None else ""),
            )
            records = (
                list(reversed(result))
                if args.command == "log" and not paging
                else result
            )
            for item in records:
                cursor = max(
                    cursor,
                    item["seq"]
                    if paging
                    else item["ts"]
                    if args.command == "log"
                    else item["id"],
                )
                if args.json:
                    print(json.dumps(item), flush=True)
                else:
                    print(table_row(item, args.command), flush=True)
            if not args.follow:
                return 0
            if paging and len(records) == limit:
                continue
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(str(exc), file=sys.stderr)
        return 1
