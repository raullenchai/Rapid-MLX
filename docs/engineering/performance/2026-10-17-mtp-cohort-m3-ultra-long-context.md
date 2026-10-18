# Continuous self-MTP cohort vs plain batching at long context on M3 Ultra (#4795, #4794)

Date: 2026-10-17. Owner: Harbor. Follow-up to
`2026-10-17-mtp-ragged-rollback.md` (#4589 / #4777).

## Result

- **The fair plain baseline is the text-only lane.** On Qwen3.6-35B-A3B,
  `--no-spec-decode` alone serves through the vision (MLLM) lane, because only
  a speculative config triggers the automatic downgrade to the text-only lane.
  The vision lane skips the MoE gate+up fusion and is 4–22% slower end to end
  (up to 29% in steady decode) than the text-only lane that the cohort runs
  on. The comparisons below therefore
  use `--no-spec-decode --no-mllm` as "plain". The vision-lane numbers are kept
  only to explain the earlier long-context A/B.
- **Against that baseline, the cohort trails plain batching at 4 and 8
  lanes at every context measured:** −22% / −21% for short prompts, −13% / −7%
  at 8k, and −6% / −5% at 16k (end-to-end tok/s). At 8k–16k the gap is
  smaller only because prefill dominates the round. Steady-state decode trails
  by 25–29% at 8k.
- **This change caps cohorts at 3 concurrent requests.** When more than 3
  requests are live, the scheduler keeps plain batched decode. At 4 and 8
  lanes this recovers 23–24% for short prompts and 6–13% at 8k–16k, landing
  within 3% of plain batching. At 2 lanes it is unchanged from main.
- **Cohorts at 2–3 lanes still trail plain batching on this host** (−5% to
  −12% at 2 lanes). Whether to keep 2–3-lane cohorts is hardware-dependent
  and is left as a follow-up (see the last section). It is not part of this
  change.
- At 32k no cohort forms on this host. The scheduler's prefill memory gate
  admits one long prefill at a time, so the cap does not change 32k behavior.
- **#4794:** under the cap, cohorts have at most 3 lanes and form only at
  ≤16k on this host. There the O(context) part of ragged KV rollback is an
  expected 4–8% of the cycle, with a ≤12% upper bound if every cycle were
  ragged. Removing it cannot close the 24% steady-decode gap to plain batching
  at 16k × 2 lanes. It is immaterial to the routing decision, so #4794 is
  closed with these numbers. A design sketch is kept below in case wider
  long-context cohorts become possible.

## Environment

- Apple M3 Ultra, 256 GB unified memory, macOS 26.5.2, arm64. The host was
  leased exclusively for these runs. Every server boot and the micro-benchmark
  ran under the shared large-model `lockf`, so no other GPU job overlapped.
- Python 3.12.13; MLX 0.32.4; mlx-lm 0.31.3; transformers 5.15.1; httpx 0.28.1.
- Fresh editable install in an isolated venv, run from each checkout with
  `PYTHONPATH` set to it. Source revisions:
  - Same-session comparison (headline table): `main` at
    `6c3951cde1cc6d9d6315ee3f359356716981c056` for plain and main-cohort, and
    this change at `889e7474266dfef8be86aa3d7957a7decd7a0fd5` (rebased onto
    that `main`) for the capped cohort. The comment-only edit that follows
    does not change behavior.
  - Earlier long-context A/B, cohort composition and short-prompt sweep:
    `main` at `946e82d596d165ce7ad4cee09b794e655f94fb61` (includes #4777).
- Target `mlx-community/Qwen3.6-35B-A3B-4bit` snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`. MTP head
  `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` snapshot
  `0295b81421bf4d0fccca9a7c0fcfb1418dda3516`. Cohort draft depth is 2, the
  backend ceiling; the log confirms `draft_depth=2` for every cohort.

## Serve commands

Each boot is a fresh process. `<snapshot>` and `<MTP snapshot>` are the local
snapshot directories above.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
COMMON="--host 127.0.0.1 --served-model-name q36 --max-num-seqs 8 --max-tokens 4096"

# plain (text-only lane): the baseline
python -m rapid_mlx.cli serve '<snapshot>' $COMMON --port 8576 --no-spec-decode --no-mllm

# plain (vision lane): what --no-spec-decode alone selects for this model
python -m rapid_mlx.cli serve '<snapshot>' $COMMON --port 8570 --no-spec-decode

# cohort (main checkout = uncapped, this change = capped)
python -m rapid_mlx.cli serve '<snapshot>' $COMMON --port 8571 \
  --speculative-config '{"method":"mtp","model":"<MTP snapshot>","continuous_batching":true,"allow_dynamic_membership":true}' \
  --force-spec-decode
```

The cohort server logs `auto-downgraded to the text-only lane because the
requested speculative decoder is not honored by its vision lane` and
`[moe_fusion] gate+up fusion applied: 40 MoE layers`. The vision-lane plain
server logs `BatchedEngine loaded: … (mllm=True)` with no `moe_fusion` line.
The text-only plain server logs both `mllm=False` and the fusion line.

## Workload and metrics

- Prompts follow the earlier recipe: a 16-character SHA-256 nonce, then
  "Background vocabulary:", then seeded single-token words, then the fixed
  cache-explanation instruction. 350 / 8,000 / 16,000 / 32,000 words give
  410–415 / 8,059–8,065 / 16,060–16,065 / 32,059–32,065 templated tokens
  (`apply_chat_template`, `add_generation_prompt=True, enable_thinking=False`).
  Every round uses distinct fixtures. The same fixtures are reused across arms
  and boots.
- After readiness, one 664-token calibration request runs first. Then each
  point gets 1 warm round and 5 measured rounds. Requests use `model="q36"`,
  `max_tokens=300`, `temperature=0`, `enable_thinking=false`,
  `chat_template_kwargs.enable_thinking=false`, and `stream=true` with
  `stream_options.include_usage`. Every measured request in every boot
  returned exactly 300 tokens with `finish_reason=length`.
- One pooled `httpx` connection per lane. All requests in a round are
  released in the same event-loop turn.
- **e2e tok/s**: the sum of API-reported completion tokens divided by the
  round's wall time, from release until the last response finishes. This is
  the earlier recipe's metric, so it includes prefill.
- **Steady decode tok/s**: tokens emitted while every lane is decoding,
  between the last first token and the first finish, divided by that window's
  length. Tokens are counted by re-tokenizing the streamed text at the window
  edges. It is undefined ("–") when the lanes' decode phases do not overlap.
- Medians of the 5 measured rounds, with [min–max] in brackets. MTP
  acceptance comes from `/metrics`
  (`rapid_mlx_spec_decode_{attempts,accepts}_total`) before and after each
  round.
- Greedy outputs are not bit-identical across arms, or across cohort sizes
  within an arm. Batch composition changes the batched numerics. This
  comparison measures throughput, not parity.

## Same-session comparison (headline)

Four boots ran back to back on one lease: vision-lane plain, main cohort,
capped cohort, and text-only plain. Each covered short, 8k and 16k prompts
at c = 2, 4 and 8. e2e tok/s, median [min–max], N=5:

| ctx | c | plain, text lane | cohort, main | cohort, capped (this change) | plain, vision lane |
| --- | ---: | --- | --- | --- | --- |
| short | 2 | 189.2 [188.4–189.7] | 165.6 [152.9–181.2] | 179.4 [172.9–183.6] | 148.0 [144.8–152.8] |
| short | 4 | 272.2 [270.7–273.9] | 212.2 [187.8–217.2] | 263.9 [261.6–265.5] | 225.6 [225.1–227.0] |
| short | 8 | 351.1 [350.3–354.4] | 278.3 [273.8–280.7] | 342.7 [340.2–351.9] | 303.8 [301.0–306.8] |
| 8k | 2 | 60.5 [60.2–60.8] | 54.3 [54.0–54.8] | 53.6 [52.5–54.1] | 54.4 [52.1–55.6] |
| 8k | 4 | 66.0 [64.8–66.6] | 57.3 [55.5–57.4] | 64.8 [60.8–66.3] | 58.7 [57.5–59.4] |
| 8k | 8 | 64.9 [64.3–66.1] | 60.5 [59.0–60.8] | 63.9 [61.7–66.9] | 62.1 [60.6–62.6] |
| 16k | 2 | 31.0 [31.0–31.3] | 29.0 [28.7–29.1] | 28.9 [28.3–29.6] | 27.2 [27.1–27.4] |
| 16k | 4 | 31.6 [31.5–32.2] | 29.6 [29.6–29.8] | 32.4 [32.2–32.6] | 28.1 [28.1–28.2] |
| 16k | 8 | 31.9 [31.7–32.3] | 30.4 [29.8–30.5] | 32.6 [32.5–33.0] | 28.2 [28.1–28.3] |

Ratios against text-lane plain:

| ctx | c | main cohort ÷ plain | capped ÷ plain | capped ÷ main cohort |
| --- | ---: | ---: | ---: | ---: |
| short | 2 | 0.88 | 0.95 | 1.08 (main range overlaps) |
| short | 4 | 0.78 | 0.97 | 1.24 |
| short | 8 | 0.79 | 0.98 | 1.23 |
| 8k | 2 | 0.90 | 0.89 | 0.99 |
| 8k | 4 | 0.87 | 0.98 | 1.13 |
| 8k | 8 | 0.93 | 0.98 | 1.06 |
| 16k | 2 | 0.94 | 0.93 | 1.00 |
| 16k | 4 | 0.94 | 1.03 | 1.09 |
| 16k | 8 | 0.95 | 1.02 | 1.07 |

Steady decode (median tok/s; text plain / main cohort / capped): short c=4
331 / 259 / 317; short c=8 440 / 373 / 430; 8k c=4 284 / 201 / 274; 8k c=8
366 / 275 / 361; 16k c=2 178 / 135 / 131. At 16k c=4/8, lanes in the capped
arm finish prefill at different times and their decode phases do not overlap,
so steady decode is undefined there.

Capped-arm routing, from its server log and `/metrics`: 14 two-lane cohorts
formed, plus one INFO line `skipping cohort formation:
occupancy=4 exceeds cap=3`. At c=4/8 there are no MTP attempts in short and
8k c=8 rounds. 8k c=4 (36 attempts) and 16k c=4/8 (about 1,200 each over 5 rounds) show
singleton MTP on the last requests still decoding. The main-cohort log admitted `lanes=2` ×30,
`lanes=4` ×12, `lanes=5` ×6 and `lanes=8` ×6.

## Earlier long-context A/B against the vision lane (`946e82d59`)

The first sweep used `--no-spec-decode` alone as plain, which is the vision
lane. Against that baseline the cohort looked at parity or ahead at 8k–32k
with 4 and 8 lanes:

| ctx | c | plain, vision lane | cohort | cohort ÷ vision plain |
| --- | ---: | --- | --- | ---: |
| 8k | 1 | 36.5 [35.5–37.5] (b1); 35.1 [31.3–37.3] (b2) | 47.6 [41.3–53.3] (b1); 48.1 [46.1–53.2] (b2) | 1.30–1.37 |
| 8k | 4 | 53.1 [52.2–55.4] (b2); 56.2 [54.4–57.5] (b5); *36.5 [26.3–49.5] (b1)* | 54.1 [53.3–54.5] (b2); 55.4 [53.9–56.3] (b1) | 0.96–1.04 |
| 8k | 8 | 53.6 [51.9–55.0] (b2); 61.5 [60.3–62.2] (b5); 51.1 [9.4–57.5] (b1) | 55.3 [54.2–56.2] (b2); 60.2 [58.5–60.7] (b1) | 0.90–1.18 |
| 16k | 1 | 21.8 [6.6–22.6] | 27.8 [25.3–29.0] | 1.28 |
| 16k | 4 | 27.1 [26.5–27.8] (b1); 28.3 [28.1–28.7] (b5) | 28.5 [27.3–28.8] | 1.01–1.05 |
| 16k | 8 | 21.1 [19.4–21.2] (b1); 28.8 [28.5–29.1] (b5) | 27.1 [26.6–29.6] | 0.94–1.28 |
| 32k | 1 | 10.9 [10.6–12.3] | 12.6 [12.0–14.3] | 1.16 |
| 32k | 4 | 12.1 [11.9–12.2] | 12.1 [11.6–13.2] | 1.00 |
| 32k | 8 | 11.0 [9.0–12.2] | 12.9 [12.4–13.5] | 1.17 |

Plain at long context varied far more across boots than within a boot. 8k
c=4 plain b1 is a bimodal outlier (italics): 8k was the first workload after
calibration in that boot. These rows show the cohort roughly matching a
baseline that is itself 4–12% slower than the text-only lane at 8k–16k. They
do not show the cohort matching plain batching. At 32k the cohort arm never
forms a cohort (next section), so its 32k rows compare the singleton/plain
text-lane path against the vision lane. The c=1 rows compare singleton MTP
against the vision lane, and no text-lane c=1 baseline was measured.

## Cohort composition at long context

Server admission logs from the `946e82d59` sweep. The scheduler's prefill
memory gate (`[prefill-cohort]`), not the cohort policy, decides how many long
prompts may prefill together, and it applies to every arm:

| ctx | c | admitted cohorts |
| --- | ---: | --- |
| 8k | 4 | `lanes=4` every round |
| 8k | 8 | `lanes=5` (b1) / `lanes=8` (b2) every round |
| 16k | 4, 8 | `lanes=2` every round; the remaining lanes ran on the non-cohort path |
| 32k | 4, 8 | none |

At 32k the gate's estimate for the selected rows (the rows already prefilling
plus the candidate) was about 3.0e11 bytes, against a 2.15e11-byte cap, so it
deferred every additional row. The process's measured Metal peak during the
same runs, weights included, was 39–45 GB. The gate's comment calls its
workspace estimate a deliberate over-count. It is outside this change; see
the follow-ups.

## The cap and how it was validated

`_CONTINUOUS_MTP_MAX_COHORT_LANES = 3` in `rapid_mlx/scheduler.py` applies at
two points:

- Initial formation counts occupancy. Occupancy is the larger of (eligible
  queued candidates + rows already in the base generator's generation and
  prompt batches) and the scheduler's live request count. If it exceeds 3, no
  cohort forms and every request stays on plain batched decode. A large burst
  is never split into a capped cohort plus overflow.
- Dynamic joins never grow a cohort past 3 lanes. No join happens while more
  than 3 requests are live.

The router's own `max_lanes` stays `min(max_num_seqs, completion_batch_size)`.
Two earlier attempts lowered `max_lanes` itself to 3, and both measured worse
than main: 161–164 / 183–194 tok/s at short c=4 / c=8. The cause was that
`Scheduler._max_running_sequences()` also bounds scheduler admission by the
router's `max_lanes`. Lowering it to 3 capped *all* running requests at 3, so
the fourth and later requests waited for a free slot, and max TTFT rose to
4.9 s (c=4) and 9.2 s (c=8). The shipped version leaves admission width at
`max_num_seqs`, and a unit test pins `_max_running_sequences() == 8` with the
cap installed.

## Short prompts on the `946e82d59` sweep

| c | plain, vision lane | cohort (uncapped) |
| ---: | --- | --- |
| 1 | 47.2 [44.8–48.9] | 95.9 [78.4–114.1] |
| 2 | 141.4 [130.9–143.8] (b4); 143.7 [107.8–145.5] (b5) | 169.0 [156.7–183.3] |
| 3 | 184.0 [182.7–187.2] | 191.6 [154.1–197.6] |
| 4 | 219.8 [216.2–226.7] (b2); 222.4 [217.0–225.6] (b5) | 202.9 [196.0–211.8] |
| 8 | 311.7 [308.2–313.7] (b2); 305.2 [300.0–307.3] (b5) | 273.1 [259.7–274.8] |

The cohort already trailed the slower vision-lane baseline at 4 and 8 lanes.
Its 2–3-lane "wins" here disappear against the text-only lane (headline
table).

## #4794: share of ragged KV rollback in the cohort cycle

The cycle time comes from steady decode: cycle = tokens per lane per cycle ÷
per-lane steady tok/s, where tokens per cycle = 1 + 2 × accept ratio (two
drafts per lane per cycle). Rollback cost comes from the model-free rollback
micro-benchmark in the earlier document, extended with `--batch` and
`--contexts` options and rerun on this host with batch 2/4/5/8 and contexts
512/8192/16384/32768. B=8 uses the original drop and padding patterns; other
batch sizes take the first B entries of the `i % 3` drops and `i % 4` left
padding. The "O(context) part" is ragged rollback at the context minus ragged
rollback at 512 tokens. That part is what an O(verify-width) layout would
remove.

Isolated rollback medians (ms, 30 samples), uniform / ragged:

| B | 512 | 8,192 | 16,384 | 32,768 |
| ---: | --- | --- | --- | --- |
| 2 | 0.44 / 1.55 | 0.48 / 3.08 | 0.55 / 5.57 | 0.55 / 8.72 |
| 4 | 0.47 / 2.45 | 0.58 / 5.77 | 0.57 / 9.86 | 0.63 / 17.94 |
| 5 | 0.50 / 2.79 | 0.60 / 6.96 | 0.59 / 12.21 | 0.60 / 22.64 |
| 8 | 0.57 / 3.95 | 0.68 / 11.06 | 0.72 / 19.55 | 0.69 / 37.31 |

A cycle needs a ragged rollback only when the lanes accept different numbers
of drafts. With the measured per-lane accept distribution and lanes treated
as independent, about 65% of 2-lane cycles, 87% of 3-lane cycles, and nearly
all cycles at 4 or more lanes are ragged. "Upper bound" below assumes every
cycle is ragged; "expected" applies those frequencies.

| Point (cohort that formed) | Accept ratio | Cycle (ms) | O(context) part (ms) | Upper-bound share | Expected share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8k, 2 lanes (same session, main) | 0.660 | 27.8 | 1.53 | 5.5% | 3.6% |
| 16k, 2 lanes (same session, main) | 0.651 | 34.1 | 4.02 | 11.8% | 7.7% |
| 8k, 4 lanes (uncapped; no longer forms) | 0.658 | 43.8 | 3.32 | 7.6% | ≈7.6% |
| 8k, 8 lanes (uncapped; no longer forms) | 0.650 | 68.2 | 7.11 | 10.4% | ≈10.4% |

The isolated figures leave out in-situ overheads such as host
synchronization and allocator pressure. For reference, #4589 measured target
rollback in situ at 27.8 ms (B=4, 16k), before #4777.

Under the cap, the realistic cohorts on this host are 2–3 lanes at ≤16k. 3
lanes form only at 8k, where the B=2–4 interpolation puts the expected share
near 6%. Expected O(context) shares there are 4–8%, below the 10% bar. Even
the upper bound at 16k × 2 lanes (11.8%) cannot close the steady-decode gap
to text-lane plain at that point (135 vs 178 tok/s, −24%). **#4794 is
therefore closed with these numbers and no layout change.** It becomes
material again if the prefill gate is relaxed and wider cohorts form at
16k–32k: B=8 at 32k spends 33 ms on the O(context) part alone.

Design sketch, kept for that case:

- Give `BatchKVCache` a per-row valid end (`row_end[B]`) next to the shared
  physical cursor `_idx`. Rows stop being right-aligned. A ragged rewind just
  lowers `row_end` (an O(1) metadata update), and rejected tail slots stay in
  place, masked.
- Attention mask: per row, key position `p` is valid iff
  `left_padding[r] <= p < row_end[r]`. Build it from these two vectors rather
  than from `_idx` alone.
- Append: the next verify writes each row at its own `row_end[r]`. Either
  scatter the S new keys/values per row at width S (O(verify-width) per
  layer), or keep the shared `_idx` write and, only when the rows' trailing
  gap exceeds a threshold, compact with one O(context) gather. That amortizes
  the current every-cycle cost.
- RoPE offset per row = `row_end[r] - left_padding[r]` (logical position),
  independent of the physical slot.
- Merge, filter and extract (cohort join/detach, prompt-cache save) must
  respect `row_end`: extract `[left_padding[r], row_end[r])` and, on merge,
  realign once.
- Recurrent (GDN) layers are unaffected. Their rollback already selects
  per-row snapshots and is flat in context (uniform 0.4–0.7 ms above).
- Acceptance: a flat ragged rollback curve in the micro-benchmark from 512 to
  32k, plus bit-identical greedy outputs against current main on the serving
  parity set.

## Follow-ups (not in this change)

- `--no-spec-decode` on Qwen3.6-35B-A3B selects the vision lane, which skips
  MoE gate+up fusion and runs 4–22% slower end to end than the text-only lane
  for text-only traffic. Text-only serving defaults deserve their own measurement
  and decision.
- 2–3-lane cohorts trail text-lane plain by 5–12% on M3 Ultra. Earlier
  2-lane wins on other hardware were measured against `--no-spec-decode` and
  need a rerun against `--no-mllm` before deciding whether cohorts should
  form at all on a given host.
- The prefill memory gate over-estimates workspace for this hybrid model at
  long context (≈300 GB estimated for the selected rows at 32k vs 39–45 GB
  measured process peak). It serializes long prefills in every arm.
- The existing fallback tests mock `batch_gen.next`. An integration test
  with a real batch generator would cover the plain-decode handoff end to end.

## Appendix A: fixture builder

```python
import argparse, hashlib, json, random
from transformers import AutoTokenizer

TAIL = (" Write a detailed six hundred word explanation of how a reliable computer "
        "cache works. Include examples, tradeoffs, and practical design advice. "
        "Continue until the explanation is complete.")
COMMON = ("time year people way day man thing woman life child world school state family "
          "student group country problem hand part place case week company system program "
          "question work government number night point home water room mother area money "
          "story fact month lot right study book eye job word business issue side kind head "
          "house service friend father power hour game line end member law car city name "
          "team minute idea body information back parent face others level office door "
          "health person art war history party result change morning reason research girl "
          "guy moment air teacher force education river table garden market paper music "
          "window light color heart field green black white short small large early young").split()

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--out", required=True)
p.add_argument("--rounds", type=int, default=6)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
words = [w for w in COMMON if len(tok.encode(" " + w, add_special_tokens=False)) == 1]

def prompt(key, n_words):
    nonce = hashlib.sha256(key.encode()).hexdigest()[:16]
    rng = random.Random(key)
    text = f"{nonce} Background vocabulary: {' '.join(rng.choice(words) for _ in range(n_words))}.{TAIL}"
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return {"key": key, "text": text, "tokens": len(ids),
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}

fx = {"calibration": prompt("calibration", 600), "points": {}}
for ctx, n in {"short": 350, "8k": 8000, "16k": 16000, "32k": 32000}.items():
    for c in (1, 2, 3, 4, 8):
        for r in range(a.rounds):
            fx["points"][f"{ctx}/c{c}/r{r}"] = [prompt(f"{ctx}/c{c}/r{r}/l{i}", n) for i in range(c)]
json.dump(fx, open(a.out, "w"))
```

## Appendix B: load driver

Run against one server per boot (the same-session comparison used
`--points short,8k,16k --conc 2,4,8`), for example
`python drive.py --base http://127.0.0.1:8571 --fixtures fixtures.json --out cohort.jsonl --arm cohort --points 8k,16k,32k --conc 1,4,8`.
Round 0 of each point is the warm round.

```python
import argparse, asyncio, json, re, time
import httpx

BODY = dict(model="q36", max_tokens=300, temperature=0, stream=True,
            stream_options={"include_usage": True}, enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False})

async def one(client, url, fx):
    body = dict(BODY, messages=[{"role": "user", "content": fx["text"]}])
    chunks, usage, finish = [], None, None
    async with client.stream("POST", url, json=body, timeout=3600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            obj = json.loads(line[6:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                if (d := (ch.get("delta") or {}).get("content")):
                    chunks.append((time.perf_counter(), d))
                finish = ch.get("finish_reason") or finish
    return dict(key=fx["key"], sha=fx["sha"], usage=usage, finish=finish, chunks=chunks)

def metrics(base):
    txt = httpx.get(base + "/metrics", timeout=30).text
    out = {}
    for name in ("rapid_mlx_spec_decode_attempts_total", "rapid_mlx_spec_decode_accepts_total"):
        m = re.search(rf"^{name}(?:{{[^}}]*}})? (\S+)", txt, re.M)
        out[name] = float(m.group(1)) if m else None
    return out

async def main():
    p = argparse.ArgumentParser()
    for k in ("--base", "--fixtures", "--out", "--arm"):
        p.add_argument(k, required=True)
    p.add_argument("--points", default="8k,16k,32k"); p.add_argument("--conc", default="1,4,8")
    p.add_argument("--rounds", type=int, default=6)
    a = p.parse_args()
    fx, out, url = json.load(open(a.fixtures)), open(a.out, "a"), a.base + "/v1/chat/completions"
    async with httpx.AsyncClient(timeout=3600) as cal:
        await one(cal, url, fx["calibration"])
    for ctx in a.points.split(","):
        for c in map(int, a.conc.split(",")):
            clients = [httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=1)) for _ in range(c)]
            for rnd in range(a.rounds):
                m0 = metrics(a.base); t0 = time.perf_counter()
                res = await asyncio.gather(*(one(cl, url, f) for cl, f in
                                             zip(clients, fx["points"][f"{ctx}/c{c}/r{rnd}"])))
                t1 = time.perf_counter(); m1 = metrics(a.base)
                comp = sum((x["usage"] or {}).get("completion_tokens", 0) for x in res)
                out.write(json.dumps(dict(arm=a.arm, ctx=ctx, c=c, round=rnd, warm=rnd == 0,
                                          t0=t0, t1=t1, completion_tokens=comp, e2e_tps=comp / (t1 - t0),
                                          metrics_before=m0, metrics_after=m1, requests=res)) + "\n")
                out.flush()
            for cl in clients:
                await cl.aclose()

asyncio.run(main())
```

Steady decode per round: let `w0` be the latest first-chunk time across lanes
and `w1` the earliest last-chunk time. For each lane, take the concatenated
streamed text up to `w0` and up to `w1`, encode both with the target tokenizer
(`add_special_tokens=False`), and sum the differences in token count. Divide
by `w1 − w0` when `w1 > w0`.
