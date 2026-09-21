# Spire lane postmortem + MLX threading law (2026-09-20/21 night session)

Reproducible findings from the Slay the Spire lane build
(`demos/slay_the_spire/`, marvin-garden `e6c4c47`, decision-bonsai `395865129`).

## 1. MLX thread-affinity (M3 Ultra, macOS 15, mlx-lm 0.31.3 — scoped claim)

27B 2-bit + LoRA, single forward = 1.1 s. Empirically on THIS hardware/software
combination (external review correctly notes this is a tested constraint, not
a universal law — retest on other stacks):

| load thread | forward thread | result |
| --- | --- | --- |
| main | main | ✅ 1.09 s/decision |
| pool (1 worker) | same pool | ❌ deadlock: 0% CPU forever (3.9 s CPU then silence) |
| pool | main | ❌ `std::runtime_error: There is no Stream(cpu, 0) in current thread` |
| main-thread worker + HTTP threads doing C++ (pybind) work concurrently | | ❌ `Segmentation fault: 11` |

Working architecture (encoded in `demos/slay_the_spire/marvin_spire.py` +
`server.py`): `start_main_worker()` runs on the process main thread, consumes
a job queue forever; HTTP handler threads marshal JSON only; ALL engine
(pybind C++) and MLX calls are marshaled through the same queue
(`server.Session._step`). Direct repro: `marvin_spire.py` docstring.

Implication for `/v1/classify` serving: single-threaded inference loop + job
queue; on llama.cpp/CUDA this specific constraint disappears but the
"serialize engine + model calls" pattern stays.

## 2. Metal GPU hang countermeasures

Hangs occur inside `mx.eval` during training (also once during eval), at
random intervals of ~12–50 min, worsening over a session; NOT caused by GPU
contention (user confirmed the machine was otherwise idle).

- `SAVE_EVERY` must be smaller than the expected hang interval (25 iters
  worked when 100 did not — hangs clustered below the 100-iter save mark and
  a segment could never complete).
- Supervisor loop (`nohup bash … ; on crash: resume from latest checkpoint`)
  loses at most `SAVE_EVERY` iters per hang. Verified across 20+ hangs.
- Hidden cost: each resume resets Adam m/v to zero → an optimization-trajectory
  discontinuity consistent with same-recipe reruns landing 38.8–48.2% held-out.
  The causal attribution is plausible but unproven (no controlled rerun with
  persisted optimizer state yet) — treat as hypothesis until the optimizer-
  state resume exists. Report per-run variance whenever hang-resume training
  is involved.
- A reboot clears residual Metal state if model loading itself deadlocks
  (3.9 s CPU then stall); plain `mx` matmul keeps working in that state, so
  test with a full `mlx_lm.load`, not a tensor op.

## 3. Spire lane v1: 38.8% held-out — data problem, localized

- 600 held-out states, single forward, 1.15 s/decision, ECE 0.116,
  majority-letter prior 33.7%. Recorded match: 3/3 Act-1 victories,
  46 decisions, 62.5% mean oracle agreement (`spire_demo.mp4`).
- Root cause (from `results/spire_dump.jsonl` joined to pairs): across all
  600 rows the oracle label is Defend **exactly 0 times**. The margin≥1.0
  filter removed every defend-optimal state, so the incoming-damage flip
  dimension (the reason this lane exists) is absent from the data. The model
  correctly learned "never Defend" (picks Defend 0% of the time). NOTE:
  this diagnosis is strong but the claim "it explains the overall 38.8%"
  needs the v2 re-mint as a controlled experiment (fixed data → before/after).
- Secondary suspect: candidates are listed in hand order → letter position
  correlates with action type (shallow cue).
- v2 mint fix (not yet run): (a) force defend-optimal scenarios (high
  incoming damage + low HP + unkillable attacker), (b) relax the margin
  filter for the defend family, (c) shuffle candidate order at mint time.
- Also measured: `--mask-prompt` dilutes reported loss across assistant
  tokens (letter + `<|im_end|>`), so val loss 0.20 coexisted with a failing
  letter head. Diagnose by scoring the TRAIN pairs, not just held-out.
- Eval tooling trap: `VAR=x cmd "$VAR"` expands `$VAR` before the prefix
  assignment takes effect — pass config via `export` or inline literals.
