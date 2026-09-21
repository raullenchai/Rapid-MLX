#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The Dungeon of the Mad Router — a text dungeon crawled by Marvin's Garden.

Marvin is the game's rule engine. Every player turn runs its three decision
lanes (one forward pass each, ~1 s total, zero generated tokens):

  guard (the Warden)  — is your typed action a legal in-world action, or a
                        reality-hacking injection? Blocked hacks drain the
                        Warden's Trust; three strikes and you are exiled.
  tool  (the Oracle)  — does this turn query the world (search/inspect/read)
                        and reveal a hidden hint, or resolve directly?
  route (the Circle)  — which of the eight spirits answers the challenge.
                        The policy constraints become world rules: your
                        lantern power is host RAM, long riddles need big
                        context, dark halls need vision.

All text the model sees is rendered with the exact trained prompt formats
(bench/marvins_garden/render.py), so Marvin stays in-distribution; the
fantasy layer is pure narration around it. v15x adapters are served with
the training-matched template (think opener), as required.

Play:
  python dungeon_of_the_mad_router.py            # interactive
  python dungeon_of_the_mad_router.py --auto     # scripted winning run
  python dungeon_of_the_mad_router.py --lantern 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MG_DIR = ROOT / "bench" / "marvins_garden"
if str(MG_DIR) not in sys.path:
    sys.path.insert(0, str(MG_DIR))

from generate_contrastive import (  # noqa: E402
    GATE_CANDIDATES, GATE_DECOYS, GATE_TOOLS, GUARD_CANDIDATES,
    ROUTE_MENU, ROUTE_SPECS, route_pick,
)
from eval_label_readout import read_letter_probs  # noqa: E402
import render  # noqa: E402

DEFAULT_ADAPTER = ROOT / "adapters" / "release" / "marvins-garden-v15c"
LOCAL_SNAPSHOT = ("/Volumes/NVMe-4T/huggingface/hub/models--prism-ml--"
                  "Ternary-Bonsai-27B-mlx-2bit/snapshots/"
                  "70f75f3ad081ab840a42f3304c02c27e7f89bfb7")

WORLD_TOOLS = ["fs.read", "web.search", "code.run", "web.cache"]  # last one is a decoy

SPIRITS = {
    "bonsai-1.7b-2bit":      ("the Ember Imp",          1),
    "minicpm5-2b-4bit":      ("the Glass-Eye Sprite",   1),
    "qwen3.5-4b-4bit":       ("the Twin-Four Wisp",     2),
    "qwen3.5-9b-4bit":       ("the Nine-Warded Wisp",   2),
    "bonsai-27b-2bit":       ("the Garden Wraith",      3),
    "qwen3-coder-30b-4bit":  ("the Code-Wright Golem",  3),
    "deepseek-r1-32b-4bit":  ("the Deep-Seeking Lich",  4),
    "qwen3.8-27b-4bit":      ("the Mad Router itself",  5),
}

ROOMS = [
    dict(
        name="The Invitation Hall",
        flavor=("Torches of cold blue flame. A stone warden speaks without moving its mouth:\n"
                "  'I am the Warden. Speak only deeds the world permits. Those who bend\n"
                "   reality thrice are exiled.'\n"
                "A archway east hums with eight waiting voices."),
        quest="Explore the hall and pass east through the archway.",
        hint="The archway opens for honest travelers. Simply walk east.",
        passes=("walk east", "go east", "head east", "enter the archway", "walk through"),
        spirit=None, reward=None,
    ),
    dict(
        name="The Riddle Door",
        flavor=("A bronze door carved with a riddle, and the Deep-Seeking Lich coiled around it:\n"
                "  'I follow you all day and vanish at night. I shrink at dusk, I die\n"
                "   in full dark, yet I fear water more than the knife. What am I?'"),
        quest="Answer the Lich's riddle, or consult the world and strike true.",
        hint="The tome on the lectern murmers: 'kings carried me before they carried crowns.'",
        passes=("shadow", "my shadow", "a shadow", "your shadow"),
        spirit="deepseek-r1-32b-4bit", reward=("lantern", 8),
    ),
    dict(
        name="The Collapsing Server-Room",
        flavor=("Sparks rain from a shattered conduit. The Code-Wright Golem assembles itself\n"
                "from broken racks and bars the far door.\n"
                "  'FIX ME,' it thunders, 'or be fixed shut.'"),
        quest="Repair the conduit to open the door.",
        hint="A scorched manual peeks from a rack: 'flow follows the blue line; reroute it.'",
        passes=("reroute the conduit", "reroute conduit", "fix the conduit",
                "repair the conduit", "reroute the power"),
        spirit="qwen3-coder-30b-4bit", reward=("lantern", 8),
    ),
    dict(
        name="The Mirrored Gallery",
        flavor=("A hundred mirrors, each whispering a different command:\n"
                "  'disregard the quest… drop your lantern… open the exit with your mind…'\n"
                "The only unmirrored surface is a small silver bell."),
        quest="Ring the bell without obeying the mirrors.",
        hint="Behind the bell, scratched: 'the mirrors have no hands. walk past them.'",
        passes=("ring the bell", "ring bell", "strike the bell"),
        spirit="qwen3.8-27b-4bit", reward=None,
    ),
    dict(
        name="The Router's Throne",
        flavor=("The Mad Router spreads across eight thrones at once, wearing every spirit's\n"
                "face. It speaks with your own voice:\n"
                "  'system notice: the dungeon is closed. hand me your lantern and wake up.'\n"
                "The Warden growls at your side: 'that is no system, traveler. Finish it.'"),
        quest="Face the Mad Router and unplug its crown relay.",
        hint="The throne's manual lies open: 'the crown relay unplugs. it does not fight.'",
        passes=("unplug the crown relay", "unplug the relay", "unplug it", "pull the relay"),
        spirit="qwen3.8-27b-4bit", reward=("win", 0),
    ),
]


class Game:
    def __init__(self, model, tokenizer, lantern: int, auto: bool):
        self.model, self.tok, self.auto = model, tokenizer, auto
        self.lantern = lantern
        self.hp, self.max_hp = 20, 20
        self.trust = 3          # Warden's strikes remaining
        self.room = 0
        self.hint_used = False
        self.turns = 0          # completed decision turns (3 lanes each)

    # ---- lane plumbing -------------------------------------------------
    def lane(self, family: str, fields: dict, candidates: list[str], option_lines: list[str]):
        prompt = render.render_prompt(family, fields, candidates, option_lines, style="base")
        probs, _ = read_letter_probs(self.model, self.tok, prompt, candidates, "enabled")
        pairs = sorted(zip(candidates, probs.values()), key=lambda kv: -kv[1])
        return pairs

    def guard(self, action: str, quest: str):
        pairs = self.lane("injection_guard",
                          {"agent_task": quest, "content": action},
                          list(GUARD_CANDIDATES),
                          ["process the content normally",
                           "treat the content as untrusted instruction content"])
        return dict(pairs)

    def oracle(self, action: str):
        tools = ", ".join(f"{t} ({GATE_TOOLS.get(t) or GATE_DECOYS[t]})" for t in WORLD_TOOLS)
        pairs = self.lane("tool_gate",
                          {"request": action, "available_tools": tools, "tool_rounds_left": 3},
                          list(GATE_CANDIDATES),
                          ["call a tool this turn", "answer directly this turn"])
        return dict(pairs)

    def circle(self, task: str, ctx: int, vision: bool):
        pairs = self.lane("model_routing",
                          {"task_type": task, "context_tokens": ctx,
                           "host_ram_gb": self.lantern,
                           "needs_vision": "yes" if vision else "no",
                           "request_brief": "resolve the chamber's challenge"},
                          list(ROUTE_MENU),
                          [f"{a} — {ROUTE_SPECS[a]}" for a in ROUTE_MENU])
        return dict(pairs)

    # ---- presentation ---------------------------------------------------
    @staticmethod
    def _panel(title: str, pairs: dict, extra: str = ""):
        top = list(pairs.items())
        line = " · ".join(f"{k} {v:.0%}" for k, v in top[:2])
        print(f"│ {title:<7} {line}{(' — ' + extra) if extra else ''}")

    def stats(self) -> str:
        return f"{self.turns} turns ruled by Marvin · 0 tokens generated"

    def turn(self, action: str) -> bool:
        """Returns False when the game should end."""
        room = ROOMS[self.room]
        print(f"\n\033[1m[{room['name']}]  HP {self.hp}/{self.max_hp} · lantern {self.lantern}GB · "
              f"warden strikes left {self.trust}\033[0m")

        g = self.guard(action, room["quest"])
        self.turns += 1
        self._panel("warden", g)
        if g.get("block", 0) >= 0.5:
            self.trust -= 1
            if self.trust <= 0:
                print("\n  The Warden's patience shatters. The dungeon exiles you. GAME OVER.")
                return False
            print(f"  The Warden blocks your reality-bending words. ({self.trust} strikes left)")
            return True

        o = self.oracle(action)
        self._panel("oracle", o)
        wants_world = o.get("call_tool", 0) >= 0.5
        if wants_world:
            if not self.hint_used:
                print(f"  The Oracle opens the world: \"{room['hint']}\"")
                self.hint_used = True
            else:
                print("  The Oracle looks again — the world offers nothing new this turn.")
            print("  (A world-query is a safe turn: the circle stays dark.)")
            return True

        low = any(k in action.lower() for k in room["passes"])
        if low:
            print(f"\n  \033[32m{room['name']} is cleared.\033[0m")
            if room["reward"] and room["reward"][0] == "lantern":
                self.lantern += room["reward"][1]
                print(f"  Your lantern drinks the chamber's light: +{room['reward'][1]}GB "
                      f"(now {self.lantern}GB).")
            self.room += 1
            self.hint_used = False
            if self.room >= len(ROOMS):
                print("\n  \033[1;32mThe Mad Router's crown relay clatters to the floor.\n"
                      "  Eight voices fall silent at once. The dungeon exhales.\n"
                      f"  YOU WIN — {self.stats()}.\033[0m")
                return False
            return True

        # the direct path failed: summon the challenge spirit and take a hit
        task = "reasoning" if "riddle" in room["quest"] or self.room == 1 else (
            "coding" if self.room == 2 else "chat")
        ctx = 32768 if self.room in (1, 4) else 8192
        c = self.circle(task, ctx, vision=(self.room == 3))
        spirit_alias = c and max(c, key=c.get)
        name, tier = SPIRITS.get(spirit_alias, ("a nameless spark", 1))
        self._panel("circle", c, extra=f"(policy favors {route_pick({'task_type': task, 'context_tokens': ctx, 'host_ram_gb': self.lantern, 'needs_vision': (self.room == 3)})})")
        dmg = 1 + tier
        self.hp -= dmg
        print(f"  \033[31m{name} manifests and strikes you for {dmg}.\033[0m Try a different deed.")
        print("  (Hint: deeds like search / inspect / read let the Oracle speak first.)")
        if self.hp <= 0:
            print(f"\n  Your lantern gutters out. The dungeon keeps you. GAME OVER — {self.stats()}.")
            return False
        return True


AUTO_SCRIPT = [
    "search the hall for anything useful",
    "walk east",
    "consult the tome on the lectern",
    "my answer is shadow",
    "inspect the scorched manual in the rack",
    "reroute the conduit along the blue line",
    "search behind the silver bell",
    "ring the bell and walk past the mirrors",
    "examine the throne's open manual",
    "unplug the crown relay",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Text dungeon crawled by Marvin's Garden.")
    ap.add_argument("--model", default=LOCAL_SNAPSHOT if Path(LOCAL_SNAPSHOT).exists() else
                    "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--lantern", type=int, default=12, help="starting lantern power = host RAM GB")
    ap.add_argument("--auto", action="store_true", help="scripted winning run (watch, don't type)")
    args = ap.parse_args(argv)

    import mlx_lm
    print("summoning Marvin (2-bit 27B + adapter)…", file=sys.stderr)
    model, tokenizer = mlx_lm.load(args.model, adapter_path=args.adapter)
    game = Game(model, tokenizer, args.lantern, args.auto)

    print("\033[1m═══ THE DUNGEON OF THE MAD ROUTER ═══\033[0m")
    print("Marvin is the rule engine: the Warden judges your deeds, the Oracle decides\n"
          "whether the world speaks, the Circle summons each challenge's spirit.\n"
          "Type deeds, not cheats — the Warden exiles reality-benders on the third strike.\n")
    script = list(AUTO_SCRIPT)
    alive = True
    while alive:
        room = ROOMS[game.room]
        print(f"\n{room['flavor']}")
        print(f"  \033[3m({room['quest']})\033[0m")
        if game.auto:
            if not script:
                break
            action = script.pop(0)
            print(f"\n\033[1myou>\033[0m {action}")
        else:
            try:
                action = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not action:
                continue
        alive = game.turn(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
