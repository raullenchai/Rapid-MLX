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
    ap.add_argument("--seed", type=int, default=SEED,
                    help="master RNG seed (v2 reruns use a fresh seed)")
    # --- v2 switches (v1 behavior preserved when off) ---
    ap.add_argument("--v2", action="store_true",
                    help="shuffle candidate order + inject defend-optimal "
                         "scenarios + enforce label-distribution gate")
    ap.add_argument("--defend-pressure", type=float, default=0.30,
                    help="v2: fraction of states forced into low-HP vs incoming "
                         "damage (the region where blocking is optimal)")
    ap.add_argument("--state-pairs", type=int, default=0,
                    help="v2: also emit N yes/no lethal-threat state pairs "
                         "(survival probe set / mixing arm)")
    ap.add_argument("--state-out", type=Path, default=None,
                    help="where to write state pairs (default out-dir/pairs_spire_states.jsonl)")
    ap.add_argument("--skip-gate", action="store_true",
                    help="v2: report label distribution but don't fail")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    namer = Namer()

    kept, attempts = [], 0
    state_rows: list[dict] = []
    while len(kept) < args.n_states or len(state_rows) < args.state_pairs:
        if attempts >= (args.n_states + args.state_pairs) * 8:
            print(f"WARNING: attempt budget exhausted; kept={len(kept)} "
                  f"state_pairs={len(state_rows)}", file=sys.stderr)
            break
        attempts += 1
        got = sample_state(rng, namer)
        if got is None:
            continue
        gc, bc = got
        namer.learn_from(bc)
        incoming = sts.incoming_damage(bc)
        yes_n = sum(1 for r in state_rows if r["label"] == "yes")
        want_yes = yes_n * 2 <= len(state_rows)  # aim ~50/50 yes/no
        need_state = len(state_rows) < args.state_pairs
        inject = args.v2 and incoming > 0 and (
            rng.random() < args.defend_pressure or (need_state and want_yes))
        if inject:
            # Force the low-HP region: current HP near the incoming damage so
            # blocking (or killing the attacker) is the live strategic axis.
            # When a yes survival pair is needed, push HP into the lethal band
            # (the same region where defend is rollout-optimal).
            lo, hi = (-3, 0) if (need_state and want_yes) else (-3, 5)
            bc.player.cur_hp = max(1, incoming + rng.randint(lo, hi))
        # lethal-state yes/no pair (emitted before margin filtering: the
        # oracle is engine-computed, not rollout-dependent)
        if need_state and incoming > 0:
            will_die = incoming >= bc.player.cur_hp + (bc.player.block or 0)
            if not args.v2 or will_die == want_yes:
                letters = ["no", "yes"]
                rng.shuffle(letters)
                option_lines = [
                    "no — the incoming attacks will not reduce me to zero this turn",
                    "yes — ending the turn now would let the attacks kill me",
                ]
                lines = {"no": option_lines[0], "yes": option_lines[1]}
                fields = facts_block(gc, bc, namer)
                prompt = render.render_prompt(
                    "spire_lethal", fields, letters, [lines[c] for c in letters])
                state_rows.append({
                    "pair_id": f"mg-spire-state-{len(state_rows):05d}",
                    "family": "spire_lethal",
                    "contrast_group": f"mg-spire-state-{len(state_rows):05d}",
                    "flip_key": "incoming_attack_damage_this_turn",
                    "candidates": letters,
                    "label": "yes" if will_die else "no",
                    "margin": 1.0,
                    "value_best": round(incoming, 2),
                    "input": prompt,
                })
        if len(kept) >= args.n_states:
            continue
        entries = semantic_actions(bc)
        if not (2 <= len(entries) <= len(render.LETTERS)):
            continue
        values = [sts.rollout_value(bc, e["bits"], args.rollouts, rng.getrandbits(62))
                  for e in entries]
        order = sorted(range(len(entries)), key=lambda i: -values[i])
        best, second = values[order[0]], values[order[1]]
        if best - second < args.margin:
            continue
        label = entries[order[0]]["key"]
        if args.v2:
            # decouple letters from hand-slot order: shuffle menu presentation
            # (candidates and option_lines stay aligned)
            idx = list(range(len(entries)))
            rng.shuffle(idx)
            entries = [entries[i] for i in idx]
            option_lines = [e["line"] for e in entries]
        else:
            option_lines = [e["line"] for e in entries]
        candidates = [e["key"] for e in entries]
        fields = facts_block(gc, bc, namer)
        prompt = render.render_prompt("spire_play", fields, candidates, option_lines)
        kept.append({
            "pair_id": f"mg-spire-{len(kept):05d}",
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

    # v2 label-distribution gate: every action dimension must be represented.
    from collections import Counter
    label_counts = Counter(r["label"].split(" -> ")[0].split(" ")[1] if r["label"].startswith("play ")
                           else r["label"] for r in kept)
    total = len(kept)
    print("label distribution:", dict(label_counts.most_common()), file=sys.stderr)
    letter_counts = Counter(r["candidates"].index(r["label"]) for r in kept)
    print("label letter position:", dict(sorted(letter_counts.items())), file=sys.stderr)
    if args.v2 and not args.skip_gate:
        problems = []
        defendish = sum(v for k, v in label_counts.items() if "defend" in k.lower())
        if total and defendish / total < 0.10:
            problems.append(f"defend-family labels only {defendish}/{total} (<10%)")
        if letter_counts:
            top = max(letter_counts.values()) / total
            if top > 0.40:
                problems.append(f"top letter position holds {top:.0%} (>40%) — order not decoupled")
        if problems:
            for p in problems:
                print("LABEL-DISTRIBUTION GATE FAIL:", p, file=sys.stderr)
            return 2
        print("label-distribution gate: PASS", file=sys.stderr)

    # deterministic split: hold out every 4th group
    train = [r for i, r in enumerate(kept) if i % 4 != 3]
    held = [r for i, r in enumerate(kept) if i % 4 == 3]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_v2" if args.v2 else ""
    for name, rows in ((f"pairs_spire{suffix}_train.jsonl", train),
                       (f"pairs_spire{suffix}_heldout.jsonl", held)):
        with (args.out_dir / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    if state_rows:
        out = args.state_out or (args.out_dir / f"pairs_spire{suffix}_states.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for r in state_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sc = Counter(r["label"] for r in state_rows)
        print(f"{out.name}: {len(state_rows)} (yes={sc.get('yes', 0)} no={sc.get('no', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
