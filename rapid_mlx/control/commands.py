# SPDX-License-Identifier: Apache-2.0
"""CLI history readers use discovery and the private socket, even during restart."""

import http.client
import json
import sys
import time

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


def history_command(args):
    try:
        info = connection(args.port)
        target = "/rapid/v1/" + args.command
        if getattr(args, "action", None) == "clear":
            request(info["socket"], info["secret"], "DELETE", target=target)
            return 0
        cursor = 0
        while True:
            result = request(
                info["socket"],
                info["secret"],
                target=f"{target}?since={cursor}&limit={args.limit}",
            )
            records = list(reversed(result)) if args.command == "log" else result
            for item in records:
                cursor = max(
                    cursor, item["ts"] if args.command == "log" else item["id"]
                )
                if args.json:
                    print(json.dumps(item), flush=True)
                else:
                    print(" ".join(f"{k}={v}" for k, v in item.items()), flush=True)
            if not args.follow:
                return 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(str(exc), file=sys.stderr)
        return 1
