#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Marvin plays Slay the Spire — local demo server.

stdlib-only HTTP (same pattern as Router's Vigil). The battle runs on the
sts_lightspeed engine via its Python bindings; every decision is Marvin's
single-forward label readout. The rollout-value oracle (engine SimpleAgent)
is computed for display only — the model, not the oracle, plays.

  MARVIN_ADAPTER=... python server.py [--port 8765]
  open http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "bench" / "marvins_garden"
STS_BUILD = Path("/tmp/sts_lightspeed/build")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(STS_BUILD))

import slaythespire as sts  # noqa: E402
import generate_spire as gs  # noqa: E402
import marvin_spire  # noqa: E402

PLAYER_NORMAL = 1
NAMER = gs.Namer()

ENCOUNTERS = [e for e in ("JAW_WORM", "TWO_LOUSE", "CULTIST", "LARGE_SLIME",
                          "EXORDIUM_THUGS", "TWO_FUNGI_BEASTS", "RED_SLAVER", "BLUE_SLAVER")
              if hasattr(sts.MonsterEncounter, e)]


class Session:
    def __init__(self, seed: int | None = None):
        rng = random.Random(seed)
        self.rng = rng
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, rng.getrandbits(62), 0)
        gc.cur_hp = rng.randint(35, 80)
        for cid in rng.sample(gs.CARD_POOL, rng.randint(0, 4)):
            card = sts.Card(getattr(sts.CardId, cid))
            if rng.random() < 0.3:
                card.upgrade()
            gc.obtain_card(card)
        enc = getattr(sts.MonsterEncounter, rng.choice(ENCOUNTERS))
        self.bc = sts.BattleContext()
        self.bc.init_encounter(gc, enc)
        self.log: list[dict] = []
        self.turn_count = 0

    def snapshot(self) -> dict:
        bc = self.bc
        over = bc.outcome != 0
        entries = gs.semantic_actions(bc) if not over else []
        return {
            "over": over,
            "victory": bc.outcome == 1,
            "turn": bc.turn,
            "player": {"hp": bc.player.cur_hp, "max_hp": bc.player.max_hp,
                       "energy": bc.player.energy, "block": bc.player.block},
            "monsters": [{"name": NAMER.name(bc.monsters.get(i).id),
                          "hp": bc.monsters.get(i).cur_hp, "max_hp": bc.monsters.get(i).max_hp,
                          "block": bc.monsters.get(i).block}
                         for i in range(bc.monsters.count)],
            "incoming": sts.incoming_damage(bc),
            "hand": [{"name": gs.card_name(bc.cards.hand(i).id),
                      "upgraded": bc.cards.hand(i).upgrade_count > 0,
                      "cost": bc.cards.hand(i).cost} for i in range(bc.cards.cards_in_hand)],
            "draw": bc.cards.draw_count, "discard": bc.cards.discard_count,
            "options": [{"key": e["key"], "line": e["line"]} for e in entries],
        }

    def prompt_for(self, entries) -> tuple[str, list[str]]:
        fields = gs.facts_block(None, self.bc, NAMER)
        candidates = [e["key"] for e in entries]
        lines = [e["line"] for e in entries]
        return gs.render.render_prompt("spire_play", fields, candidates, lines), candidates

    def decide_and_apply(self) -> dict:
        if self.bc.outcome != 0:
            return {"over": True}
        # Everything touching the engine (C++) or MLX runs on the inference
        # main thread — concurrent Metal + pybind from HTTP threads segfaults.
        return marvin_spire.run(Session._step, self)

    @staticmethod
    def _step(sess) -> dict:
        bc = sess.bc
        entries = gs.semantic_actions(bc)
        prompt, candidates = sess.prompt_for(entries)
        decision = marvin_spire._decide(prompt, candidates)  # already main-thread
        # oracle for display (engine rollouts) — never decides
        values = {e["key"]: sts.rollout_value(bc, e["bits"], 16, sess.rng.getrandbits(62))
                  for e in entries}
        oracle_best = max(values, key=values.get)
        entry = next(e for e in entries if e["key"] == decision["chosen"])
        NAMER.learn_from(bc)
        sts.apply_action(bc, entry["bits"])
        rec = {"turn": bc.turn, "prompt": prompt, "candidates": candidates,
               "chosen": decision["chosen"], "confidence": decision["confidence"],
               "probabilities": decision["probabilities"], "latency_ms": decision["latency_ms"],
               "oracle_best": oracle_best, "oracle_agrees": oracle_best == decision["chosen"],
               "oracle_values": {k: round(v, 1) for k, v in values.items()}}
        sess.log.append(rec)
        return {"decision": {k: rec[k] for k in ("chosen", "confidence", "latency_ms", "oracle_agrees")},
                "probabilities": decision["probabilities"],
                "candidates": candidates,
                "state": sess.snapshot()}

    def stats(self) -> dict:
        n = len(self.log)
        agree = sum(1 for r in self.log if r["oracle_agrees"])
        lat = sorted(r["latency_ms"] for r in self.log)
        return {"decisions": n, "oracle_agreement": round(agree / n, 3) if n else None,
                "latency_ms_p50": lat[n // 2] if n else None}


class Handler(BaseHTTPRequestHandler):
    session = Session(seed=20260920)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/api/state":
            NAMER.learn_from(self.session.bc)
            self._json(self.session.snapshot())
        elif self.path == "/api/stats":
            self._json(self.session.stats())
        elif self.path == "/api/log":
            self._json(self.session.log)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/decide":
            try:
                NAMER.learn_from(self.session.bc)
                out = self.session.decide_and_apply()
                self._json(out)
            except Exception as e:  # keep the demo alive
                self._json({"error": str(e)}, 500)
        elif self.path == "/api/new":
            self.session = Session()
            NAMER.learn_from(self.session.bc)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    # MLX is main-thread-only (see marvin_spire docstring): serve HTTP from a
    # daemon thread and let the MAIN thread run the inference worker loop.
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"marvin plays the spire → http://localhost:{args.port}", flush=True)
    marvin_spire.start_main_worker()
