#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mint Slay the Spire contrastive pairs for the marvin-garden decision model.

States come from the sts_lightspeed engine (MIT, gamerpuppy) via its pybind11
bindings (see patches/sts_lightspeed-bindings.patch). Every candidate action is
evaluated with SimpleAgent rollouts inside the engine; the value function is
(100 + player hp) on victory, -100 on loss. The argmax action is the label;
states whose best action is not separated from the runner-up by --margin are
discarded (ambiguous labels would be noise).

Only states where the engine's input state is PLAYER_NORMAL are minted, and
the menu is semantic (identical cards targeting the same monster dedupe).

Usage:
  STS_BUILD=/tmp/sts_lightspeed/build python generate_spire.py --n-states 2400
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STS_BUILD = Path(os.environ.get("STS_BUILD", "/tmp/sts_lightspeed/build"))
sys.path.insert(0, str(STS_BUILD))

import slaythespire as sts  # noqa: E402
import render  # noqa: E402

SEED = 20260920
ENCOUNTERS = [e for e in (
    "JAW_WORM", "TWO_LOUSE", "THREE_LOUSE", "CULTIST", "LARGE_SLIME",
    "SMALL_SLIMES", "TWO_FUNGI_BEASTS", "RED_SLAVER", "BLUE_SLAVER",
    "EXORDIUM_THUGS", "EXORDIUM_WILDLIFE", "GREMLIN_GANG", "LOOTER",
) if hasattr(sts.MonsterEncounter, e)]

CARD_POOL = [c for c in (
    "CLEAVE", "IRON_WAVE", "POMMEL_STRIKE", "CLOTHESLINE", "THUNDERCLAP",
    "SHRUG_IT_OFF", "BODY_SLAM", "TWIN_STRIKE", "HEAVY_BLADE", "WILD_STRIKE",
    "SWORD_BOOMERANG", "RAGE", "RAMPAGE", "HAVOC", "HEADBUTT", "SHIELD_BASH",
) if hasattr(sts.CardId, c)]


def card_name(cid: int) -> str:
    r = repr(sts.Card(sts.CardId(cid)))
    return r[len("<slaythespire.Card "):-1]


class Namer:
    """Learns monster id -> name from action descriptions (single source: engine)."""

    def __init__(self):
        self.names: dict[int, str] = {}

    def learn_from(self, bc) -> None:
        for a in sts.legal_actions(bc):
            desc = sts.action_desc(a, bc)
            if "-> (" in desc:
                try:
                    _, rhs = desc.split("-> (", 1)
                    idx_s, rest = rhs.split(")", 1)
                    name = rest.strip().rstrip(" }")
                    self.names.setdefault(bc.monsters.get(int(idx_s)).id, name)
                except Exception:
                    pass

    def name(self, mid: int) -> str:
        return self.names.get(mid, f"monster {mid}")


def semantic_actions(bc) -> list[dict]:
    """Dedupe legal action bits into semantic menu entries."""
    seen: dict[str, dict] = {}
    for bits in sts.legal_actions(bc):
        desc = sts.action_desc(bits, bc).strip()
        if desc.startswith("{ end turn }"):
            key, line = "end turn", "End turn — no more plays this turn"
        else:
            body = desc.strip("{ } ")
            head, _, tail = body.partition(" -> ")
            # "use card (3) (Defend,5,1,1)" — hand index is the first parens group
            hand_idx = int(head.split("(")[1].split(")")[0])
            ci = bc.cards.hand(hand_idx)
            name = card_name(ci.id)
            upg = ci.upgrade_count > 0
            cost = ci.cost
            if tail:
                t_idx = int(tail.strip().split(")")[0].lstrip("("))
                t_name = tail.split(")", 1)[1].strip()
                key = f"play {name}{'+' if upg else ''} -> {t_name}"
                line = f"{name}{'+' if upg else ''} — cost {cost}, targets {t_name}"
            else:
                key = f"play {name}{'+' if upg else ''}"
                line = f"{name}{'+' if upg else ''} — cost {cost}"
            del hand_idx
        seen.setdefault(key, {"key": key, "line": line, "bits": bits, "desc": desc})
    return list(seen.values())


def sample_state(rng: random.Random, namer: Namer):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, rng.getrandbits(62), rng.choice([0, 0, 0, 1, 2]))
    gc.cur_hp = rng.randint(18, 80)
    for cid_name in rng.sample(CARD_POOL, rng.randint(0, 5)):
        card = sts.Card(getattr(sts.CardId, cid_name))
        if rng.random() < 0.3:
            card.upgrade()
        gc.obtain_card(card)
    enc = getattr(sts.MonsterEncounter, rng.choice(ENCOUNTERS))
    bc = sts.BattleContext()
    bc.init_encounter(gc, enc)
    # random prefix: sometimes advance the fight a bit for turn-1+ variety
    PLAYER_NORMAL = 1  # sts::InputState::PLAYER_NORMAL
    for _ in range(rng.randint(0, 2)):
        if bc.input_state != PLAYER_NORMAL or bc.outcome != 0:  # 0 = UNDECIDED
            return None
        acts = sts.legal_actions(bc)
        if not acts:
            return None
        sts.apply_action(bc, rng.choice(acts))
    if bc.input_state != PLAYER_NORMAL or bc.outcome != 0:
        return None
    return gc, bc


def facts_block(gc, bc, namer: Namer) -> dict:
    hand = []
    for i in range(bc.cards.cards_in_hand):
        c = bc.cards.hand(i)
        hand.append(f"{card_name(c.id)}{'+' if c.upgrade_count else ''} (cost {c.cost})")
    monsters = []
    for i in range(bc.monsters.count):
        m = bc.monsters.get(i)
        monsters.append(
            f"{namer.name(m.id)} hp {m.cur_hp}/{m.max_hp} block {m.block}"
            + (f" strength +{m.strength}" if m.strength else "")
            + (f" vulnerable {m.vulnerable}" if m.vulnerable else "")
        )
    return {
        "your_hp": f"{bc.player.cur_hp}/{bc.player.max_hp}",
        "energy": f"{bc.player.energy}/3",
        "your_block": bc.player.block,
        "incoming_attack_damage_this_turn": sts.incoming_damage(bc),
        "hand": hand,
        "draw_pile": bc.cards.draw_count,
        "discard_pile": bc.cards.discard_count,
        "monsters": monsters,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=2400)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = ap.parse_args(argv)
    rng = random.Random(SEED)
    namer = Namer()

    kept, attempts = [], 0
    while len(kept) < args.n_states and attempts < args.n_states * 6:
        attempts += 1
        got = sample_state(rng, namer)
        if got is None:
            continue
        gc, bc = got
        entries = semantic_actions(bc)
        if not (2 <= len(entries) <= len(render.LETTERS)):
            continue
        namer.learn_from(bc)
        values = [sts.rollout_value(bc, e["bits"], args.rollouts, rng.getrandbits(62))
                  for e in entries]
        order = sorted(range(len(entries)), key=lambda i: -values[i])
        best, second = values[order[0]], values[order[1]]
        if best - second < args.margin:
            continue
        label = entries[order[0]]["key"]
        candidates = [e["key"] for e in entries]
        option_lines = [e["line"] for e in entries]
        fields = facts_block(gc, bc, namer)
        prompt = render.render_prompt("spire_play", fields, candidates, option_lines)
        kept.append({
            "family": "spire_play",
            "contrast_group": f"mg-spire-{len(kept):05d}",
            "flip_key": "incoming_attack_damage_this_turn",
            "candidates": candidates,
            "label": label,
            "margin": round(best - second, 2),
            "value_best": round(best, 2),
            "input": prompt,
        })
        if len(kept) % 100 == 0:
            print(f"  {len(kept)}/{args.n_states} states (attempts {attempts})", file=sys.stderr)

    # deterministic split: hold out every 4th group
    train = [r for i, r in enumerate(kept) if i % 4 != 3]
    held = [r for i, r in enumerate(kept) if i % 4 == 3]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pairs_spire_train.jsonl", train), ("pairs_spire_heldout.jsonl", held)):
        with (args.out_dir / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
