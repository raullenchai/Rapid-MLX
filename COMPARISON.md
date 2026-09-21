# Marvin vs Jev vs Nimble — a fair comparison

All three models below have been **measured by us on the same ruler** (the
192-sample held-out set in `bench/marvins_garden/data/pairs_heldout.jsonl`,
byte-deterministic, never trained on), with identical information and
identical scoring formulas. Published self-reports are listed separately.
Raw records: `bench/marvins_garden/results/cross_eval_{jev,nimble}.json`.

## The table

| Dimension | **Marvin v15c** (ours) | **Jev** (`jev-latest`) | **Nimble** (`Bespoke-Nimble-9B`) |
| --- | --- | --- | --- |
| **Accuracy, same ruler (measured by us)** | **94.44% ± 1.08%** (n=3 full replications; best 95.31%) | 93.23% | 74.48% |
| Calibration (ECE 15-bin, same ruler) | **0.019–0.031** | 0.102 | 0.151 |
| Per-family (same ruler) | routing 91.7–94.8% · gate 89.6–93.8% · guard 100% | routing 87.5% · gate **97.9%** · guard 100% | routing 56.3% · gate 85.4% · guard 100% |
| In-domain published accuracy | — | 93.21% (published; agrees with our ruler ✓) | 90.1% on its own 324-example eval (Jev scored 93.2% there) |
| Latency per decision | ~1.1 s local M3 Ultra; est. 50–150 ms on cloud GPU | **0.19 s p50** (hosted API incl. RTT) | 2.2 s p50 in our run ⚠ (MPS fallback kernels — not representative; CUDA/H100 is their target) |
| Base / size | Ternary-Bonsai-27B 2-bit (13.5 GB) + 58 MB adapter | closed | Qwen3.5-9B bf16 (18 GB) + 165 MB adapter, Apache-2.0 |
| Training data | 2,676 pairs, ours, byte-deterministic, committed | undisclosed | 2,676 curated examples, committed (10 domains; labels model-checked) |
| Context limit | 32k–128k requests routed by policy | undisclosed | prompts > 2,048 tokens rejected |
| Deployment | **local / on-prem / air-gapped; Apple Silicon first-class** | hosted API only | self-hosted; reference runner is CUDA-first (Mac works via community paths) |
| Task breadth | deep in 3 lanes; new families cheap to mint; OOD degrades to base (40%) | **broad classification face** | 10 curated domains + 3 typed questions; authors warn against generalizing |
| Openness | pipeline + data + evals + RC adapter + tests | closed | **open data, open weights, open recipe** (the recipe ours borrowed) |

## How to read this fairly

1. **All three columns are now same-ruler.** Nimble was run locally with its
   own prompt builder and probability math (only the device differs — its
   reference runner hard-requires CUDA; we ran Apple MPS). Jev was queried
   through its hosted SystemOne API.
2. **Domain specificity cuts both ways.** Nimble's routing collapses (56%)
   because serving-alias policy routing with hardware constraints is outside
   its curated domains — exactly what its authors warn about. The same model
   scores 90.1% in-domain on its own eval. Marvin's numbers come from its
   home distribution too; the honest claim is lane depth, not universality.
3. **The ruler was built by us.** Mitigations: held-out items were never
   trained on, and Jev's score on our ruler (93.23%) matches its published
   93.21% almost exactly. Treat sub-2-point gaps with humility.
4. **Latency is not hardware-comparable.** Jev's 0.19 s is a hosted edge;
   Marvin's 1.1 s is local inference (owner's product decision:
   intelligence-first at ~1 s); Nimble's 2.2 s here reflects missing CUDA
   kernels on Apple Silicon, not the model.
5. **Where each wins.** Marvin: same-ruler accuracy, calibration 3–5×
   better (usable thresholds/abstention), deployment freedom. Jev: hosted
   latency, API maturity, breadth — the safe call for generic tasks today.
   Nimble: openness and a great recipe; strongest in its own domains
   (90.1% in-domain), and the reason our pipeline exists.
6. **License gate.** Marvin's base is prism-ml's Ternary-Bonsai-27B; any
   hosted commercial offering needs a license review (release decision).
   Nimble is Apache-2.0 end to end.

## Reproduce

```bash
# same-ruler Jev (key via env only)
JEVAI_KEY=... python bench/marvins_garden/cross_eval_jev.py
# same-ruler Nimble (open weights; downloads ~18 GB base on first run)
python bench/marvins_garden/cross_eval_nimble.py --adapter <Bespoke-Nimble-9B snapshot>
```

## Competitive assessment — one-look table (2026-09-21)

Same 192-item ruler (our three families), same information content:

| | **Marvin v15c** (27B 2-bit, local) | **Jev-latest** (API) | **Nimble-9B** (Apache-2.0) |
| --- | --- | --- | --- |
| Accuracy | **95.31%** (n=3 reruns 94.44±1.08) | 93.23% | 74.48% |
| Calibration (ECE, 15-bin) | **0.031** | 0.102 | 0.151 |
| Latency p50 | 1.21 s (local M3 Ultra) | **0.19 s** (mean 0.82) | 2.21 s (MPS fallback — not representative) |
| Cost per 1k decisions | $0 (electricity) | API pricing | $0 |
| Deployment | open weights (base license under review) | hosted only | open weights |
| Domain breadth | 3 decision families + spire lane (MVP) | broad product surface | 10 own domains, 90.1% in-domain |
| OOD transfer measured | spire game lane 38.8% (vs 33.7% prior — MVP) | not measured by us | not measured by us |

Honest boundaries of this table:

1. Model sizes are NOT matched (27B vs unknown vs 9B); "same-ruler" refers to
   items and information, not parameters. Nimble's 74.48% is an OOD number —
   its own-domain strength (90.1% self-reported) was not re-measured by us.
2. Jev's accuracy/calibration come from its API letter-choice protocol, not
   native probability readout; its latency is hosted (us: same Mac that
   trained it — no datacenter comparison intended).
3. The spire game lane is a Marvin-only measurement at MVP quality; no
   cross-model game numbers exist yet.
4. Marvin's differentiator to defend: calibration (0.031 — thresholds and
   abstention are trustworthy) at competitive accuracy; the risk to close:
   absolute latency vs Jev and the pending base-license review.

Engineering findings from this session (MLX thread-affinity law, GPU-hang
countermeasures, spire data root-cause):
`docs/engineering/performance/2026-09-21-spire-lane-postmortem-and-mlx-threading.md`.
