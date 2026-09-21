# Marvin's Garden

High-frequency decision lanes distilled onto `Ternary-Bonsai-27B-mlx-2bit`
(Bonsai 27B): model routing, tool gating, and prompt-injection guarding —
**zero generated tokens**. One LoRA adapter, one forward pass, letter-token
softmax readout. Chat traffic never touches the adapter (two-lane serving).

## Results (192-sample held-out, same ruler for both models)

| Model | Accuracy | ECE 15-bin | Latency |
| --- | ---: | ---: | --- |
| Jev `jev-latest` (TypeSafe API, same items) | 93.23% | 0.102 | 0.19 s p50 (hosted) |
| **Marvin recipe, n=3 replications** | **94.44% ± 1.08%** | 0.019–0.031 | ~1.1 s (local, M3 Ultra) |
| Marvin v15c (release candidate) | **95.31%** | 0.031 | ~1.1 s |
| Bonsai 27B base, zero-shot | 40.1% | 0.281 | — |

- Jev's score on this ruler matches its published 93.21%, validating the
  ruler. Every Marvin replication clears it; policy-constrained routing is
  the biggest win (+7.3 pts over Jev).
- Calibration (ECE 3–5× better than Jev) makes thresholds/abstention real.
- Product profile per owner: **~1 s/decision, intelligence-first**; we do
  not race few-hundred-ms hosted products on latency.

## Layout

```
demos/                    four playable demos (see demos/*/README.md)
bench/marvins_garden/     pipeline (stdlib-only data side; mlx for train/eval)
  generate_contrastive.py  deterministic contrastive pair generator (1280 groups)
  render.py                single-source prompt rendering (base/concise/spec styles)
  validate.py              fail-closed schema validation (jsonschema)
  to_chat_sft.py           chat-SFT jsonl builder (train/valid dir layout)
  train_lora.sh            training driver (mlx-lm; --mask-prompt is mandatory)
  eval_label_readout.py    single-forward label-readout eval = /v1/classify spec
  cross_eval_jev.py        same-ruler cross-eval against TypeSafe Jev (key via env)
  iq_probe.py              general-ability A/B probe (repo eval suites, official scoring)
  data/                    pairs_{train,heldout}.jsonl + sft/ build artifacts
  results/                 committed eval JSONs for every arm
docs/engineering/         perf logs + architecture decision record
tests/                    pytest suite (runs without mlx; 14 tests)
adapters/release/         v15c release candidate + v1-base recipe start (58 MB each)
```

## Quickstart

```bash
pip install jsonschema pytest mlx-lm          # mlx-lm 0.31.3 used for results
python bench/marvins_garden/generate_contrastive.py
python bench/marvins_garden/validate.py
python bench/marvins_garden/to_chat_sft.py
pytest tests/test_marvins_garden.py -q

MODEL=<path to prism-ml/Ternary-Bonsai-27B-mlx-2bit snapshot> \
ITERS=400 LAYERS=16 BATCH=2 LR=3.0e-5 \
ADAPTER=/tmp/marvin bash bench/marvins_garden/train_lora.sh

# then the recipe: continue the converged v1-style adapter on the full set
# 150 iters, BATCH=2, LR=1.0e-5, --resume-adapter-file

# evaluate (single forward pass, no generation)
/tmp/.../python bench/marvins_garden/eval_label_readout.py \
  --model "$MODEL" --adapter adapters/release/marvins-garden-v15c \
  --temperature 1.0 --think-mode enabled
```

## Load-bearing contracts (read before serving)

1. **Adapter ⇄ serving template pairing.** mlx-lm trains with the qwen3
   think opener present; `--think-mode enabled` renders that view.
   `marvins-garden-v15c` (and all v15x) MUST be served with the
   training-matched template; mismatched pairing costs ~11 points. The
   mode must travel with the adapter in `/v1/classify`.
2. **`--mask-prompt` is mandatory** when training: without it the letter
   gradient is diluted across ~300 prompt tokens (val loss 0.048 while
   accuracy stays at base).
3. **LR 3e-5 batch 2** for base training; continuations of converged
   adapters at **1e-5 batch 2**. Batch-1 chains produce outliers (one hit
   83.3%); avoid.
4. **Two-lane serving:** decision requests load the decision adapter;
   chat stays on the bare base (adapter global-load costs 57 points of
   general ability; +8% identity-mix data recovers 47 of those points if
   a mixed lane is ever needed — see `iq_probe.py` results).

## Watch: The Router's Vigil (graphical defense game)

A canvas game where Marvin defends a realm in real time: request packets
fall from the sky and Marvin routes each into one of eight serving
portals (policy constraints are the portal specs), the Oracle gate decides
tool turns, and injection wraiths are judged by the Warden. Wrong verdicts
breach the realm's hearts; confidence bars animate every decision.

```bash
pip install mlx-lm
python demos/routers_vigil/server.py      # loads the model, serves the game
open http://localhost:8765                # ▶ Watch Marvin · or defend yourself
```

A recorded round (26 real decisions: 16 routing / 6 gates / 4 wraiths,
1 breach, ~1.6 s per verdict) ships under `docs/demo/`:
[`vigil_demo.mp4`](docs/demo/vigil_demo.mp4), with the decision log
[`vigil_round.jsonl`](docs/demo/vigil_round.jsonl) and the renderer
[`make_gameplay_video.py`](docs/demo/make_gameplay_video.py) — the video is
a frame-accurate replay of real model outputs, nothing staged.

## Play: The Dungeon of the Mad Router

A text dungeon where Marvin IS the rule engine — the same three lanes,
reskinned as game mechanics:

- **The Warden** (injection guard) judges every deed you type. Reality-hacks
  get blocked; three strikes and you are exiled.
- **The Oracle** (tool gate) decides whether your turn queries the world
  (search/inspect/read → a hidden hint) or resolves directly.
- **The Circle** (routing) summons the spirit that answers each challenge —
  your lantern power is host RAM, so an under-powered summon hurts.

A full winning playthrough is committed under [`docs/demo/`](docs/demo/) —
[`dungeon_playthrough.mp4`](docs/demo/dungeon_playthrough.mp4) (regenerable
with `make_playthrough_video.py`).

```bash
python demos/dungeon_of_the_mad_router/dungeon_of_the_mad_router.py --auto
python demos/dungeon_of_the_mad_router/dungeon_of_the_mad_router.py  # play
python dungeon_of_the_mad_router.py --lantern 12   # hard mode
```

Every turn: three forward passes, ~1 s, zero generated tokens.

## Feel it: the decision console

```bash
pip install mlx-lm
python demos/decision_console/marvin_console.py --demo  # six canned scenes
python demos/decision_console/marvin_console.py         # interactive
python demos/decision_console/marvin_console.py --ram 12 # pretend 12 GB host
```

Every turn runs all three lanes on your text — route (with the
policy-derived ground truth shown next to Marvin's answer), tool gate,
injection guard — one forward pass each, ~1.1 s per decision, zero
generated tokens. Loading the 2-bit 27B + adapter takes about a minute.

## Play: Marvin vs Slay the Spire

```bash
bash scripts/build_sts_lightspeed.sh                 # MIT headless engine (~2 min)
python demos/slay_the_spire/server.py --port 8765    # bar-chart panel on the left
```

Turn-based combat: Marvin picks every card with one forward pass (zero
generated tokens), the panel shows live probabilities, measured latency and
per-move agreement with an engine-rollout oracle. The lane was minted from
2,400 engine-labeled states (`bench/marvins_garden/generate_spire.py`) and
trained as a v15c continuation. **Status (2026-09-20 night): MVP — demo
records 3/3 Act-1 victories at 62.5% oracle agreement, but held-out accuracy
is 38.8% vs a 33.7% letter prior; candidate-order shuffling is the next
mint fix (details in `demos/slay_the_spire/README.md`).** Serving on rented
NVIDIA GPUs: `docs/engineering/operations/vast-ai-serving-runbook.md` + `serve/`.

## Cross-eval vs Jev

```bash
JEVAI_KEY=... python bench/marvins_garden/cross_eval_jev.py   # key from env only
```

Same items, same information (our letter menu becomes Jev's native
`choice` criteria), same scoring formulas. Protocol and caveats:
`docs/engineering/performance/2026-09-19-marvins-garden-mvp-signal.md`.

## How Marvin compares to Jev / Nimble

All three measured by us on the same 192-item ruler (identical information,
identical scoring). Full caveats: [COMPARISON.md](COMPARISON.md).

| | **Marvin v15c** | **Jev** (`jev-latest`) | **Nimble** (`Bespoke-Nimble-9B`) |
| --- | --- | --- | --- |
| Same-ruler accuracy | **94.44% ± 1.08%** (n=3; best 95.31%) | 93.23% | 74.48% |
| ECE 15-bin | **0.019–0.031** | 0.102 | 0.151 |
| routing / gate / guard | 91.7–94.8 / 89.6–93.8 / **100** | 87.5 / **97.9** / 100 | 56.3 / 85.4 / 100 |
| In-domain published | — | 93.21% ✓ agrees with our ruler | 90.1% on its own 324-item eval |
| Latency / decision | ~1.1 s local (50–150 ms est. cloud) | **0.19 s p50** hosted | 2.2 s here ⚠ (MPS kernel fallback) |
| Deployment | **local / on-prem / air-gapped** | hosted only | self-hosted, Apache-2.0 |
| Breadth | 3 deep lanes; mintable | **broad, productized API** | 10 curated domains |

One line: Marvin wins same-ruler accuracy and calibration with deployment
freedom; Jev wins hosted latency, API maturity and breadth; Nimble wins
openness and is the recipe ours learned from.

## License

Apache-2.0 (SPDX headers in sources).
