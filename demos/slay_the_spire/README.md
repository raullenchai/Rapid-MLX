# Marvin plays Slay the Spire

Turn-based, enumerable actions, hard constraints — the perfect arena for a
decision model that is **fast AND smart**. Marvin (Bonsai 27B 2-bit + LoRA)
picks every card with **one forward pass, zero generated tokens**; the left
panel shows the live probability bars, the measured latency, and whether the
choice matches an engine-rollout oracle.

This is NOT an API-game demo: the battle runs on
[sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) (MIT), a
100% RNG-faithful headless Slay the Spire implementation, driven through
Python bindings patched on top of it (`patches/sts_lightspeed-bindings.patch`).

## Run it

```bash
bash scripts/build_sts_lightspeed.sh        # one-time engine build (~2 min)
# train/point the spire adapter (or reuse bench/marvins_garden/adapters/spire)
MARVIN_ADAPTER=../../bench/marvins_garden/adapters/spire \
  python server.py --port 8765
# → http://localhost:8765   (▶ DECIDE / auto-play / new battle)
```

## Deliverables in this directory

| file | what it is |
| --- | --- |
| `server.py` | stdlib HTTP server: battle session + Marvin decisions + oracle display |
| `marvin_spire.py` | decision engine: single-forward letter readout, latency measured |
| `index.html` | the playable page (bar chart panel on the left) |
| `play_match.py` | headless match recorder → `match.jsonl` |
| `replay_video.py` | renders match.jsonl into the deliverable mp4 |
| `match.jsonl` / `spire_demo.mp4` | recorded runs |

## How the model was taught (honest numbers)

- States minted by `bench/marvins_garden/generate_spire.py`: Act-1 encounters,
  varied HP/decks, labels from engine rollout values (SimpleAgent playouts,
  100+hp win value), ambiguous states (margin < 1.0) discarded.
- Trained as a continuation of the v15c routing adapter (LR 1e-5, batch 2,
  3×100 iters, 25% routing replay against forgetting).
- Held-out accuracy and oracle agreement: see
  `bench/marvins_garden/results/eval_marvin_spire*.json` and
  `docs/engineering/performance/`.

## Fair-play rules

- The model only sees the rendered facts the training distribution contains;
  the oracle never decides, it only grades (agreement is displayed per move).
- Model mistakes are visible and have real battle consequences (HP lost).
