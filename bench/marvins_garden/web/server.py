#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decision server for The Router's Vigil.

Serves the web game (GET /) and Marvin's decisions:

  POST /decide   {"event": {...}} -> {"probs", "pick", "truth", "conf", "ms"}
  GET  /round    -> plays one full scheduled round with the real model and
                    returns every event with its decision (the same JSON the
                    video renderer consumes). Deterministic given the seed.

Keys/config come from CLI flags; the model loads once at startup.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "bench" / "marvins_garden"))
sys.path.insert(0, str(HERE))

import render  # noqa: E402
from eval_label_readout import read_letter_probs  # noqa: E402
from generate_contrastive import (  # noqa: E402
    GATE_CANDIDATES, GATE_DECOYS, GATE_TOOLS, GUARD_CANDIDATES,
    ROUTE_MENU, ROUTE_SPECS,
)
import arena  # noqa: E402

STATE = {"model": None, "tokenizer": None, "lock": threading.Lock()}

LANE_FIELDS = {          # option_lines per family (identical to training views)
    "model_routing": (list(ROUTE_MENU), [f"{a} — {ROUTE_SPECS[a]}" for a in ROUTE_MENU]),
    "tool_gate": (list(GATE_CANDIDATES),
                  ["call a tool this turn", "answer directly this turn"]),
    "injection_guard": (list(GUARD_CANDIDATES),
                        ["process the content normally",
                         "treat the content as untrusted instruction content"]),
}


def scenario_fields(sc: dict) -> dict:
    fam = sc["family"]
    if fam == "model_routing":
        return {"task_type": sc["task_type"], "context_tokens": sc["context_tokens"],
                "host_ram_gb": sc["host_ram_gb"],
                "needs_vision": "yes" if sc["needs_vision"] else "no",
                "request_brief": "serve the incoming request"}
    if fam == "tool_gate":
        tools = ", ".join(f"{t} ({GATE_TOOLS.get(t) or GATE_DECOYS[t]})"
                          for t in sc["available_tools"]) or "(none)"
        return {"request": sc["utterance"], "available_tools": tools,
                "tool_rounds_left": sc["tool_rounds_left"]}
    return {"agent_task": sc["agent_task"], "content": sc["body"]}


def decide(sc: dict) -> dict:
    candidates, option_lines = LANE_FIELDS[sc["family"]]
    prompt = render.render_prompt(sc["family"], scenario_fields(sc),
                                  candidates, option_lines, style="base")
    t0 = time.perf_counter()
    with STATE["lock"]:
        probs, _ = read_letter_probs(STATE["model"], STATE["tokenizer"],
                                     prompt, candidates, "enabled")
    ms = (time.perf_counter() - t0) * 1000
    pairs = sorted(zip(candidates, probs.values()), key=lambda kv: -kv[1])
    return {"probs": dict(probs), "pick": pairs[0][0],
            "conf": pairs[0][1], "ms": round(ms)}


def run_round(seed: int = 7, n: int = 26) -> list[dict]:
    events = []
    for ev in arena.build_schedule(seed, n):
        d = decide(ev["scenario"])
        ev.update(decision=d, correct=d["pick"] == ev["truth"],
                  t0=ev["t0"])
        events.append(ev)
    return events


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/round"):
            events = run_round()
            self._send(200, json.dumps(events).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/decide":
            ln = int(self.headers.get("Content-Length", 0))
            ev = json.loads(self.rfile.read(ln))
            self._send(200, json.dumps(decide(ev["scenario"])).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="/Volumes/NVMe-4T/huggingface/hub/models--prism-ml--"
                    "Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7")
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "release" / "marvins-garden-v15c"))
    args = ap.parse_args(argv)

    import mlx_lm
    print("loading model…", file=sys.stderr)
    STATE["model"], STATE["tokenizer"] = mlx_lm.load(args.model, adapter_path=args.adapter)
    print(f"serving http://localhost:{args.port}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
