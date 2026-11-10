# Adaptive tuning framework: explicit cells and versioned profiles

Performance policy is a high-dimensional matrix. Hardware dimensions include
chip generation/tier, GPU cores and memory; software includes macOS, MLX and
Rapid-MLX versions; model dimensions include family, geometry, quantization,
attention kinds and installed capabilities; workload includes mode, width and
batch. Every cell always has a working value: its registered knob default,
which is conservative and need not be optimal. Measurements improve individual
cells over time through a shipped, versioned performance profile.

`rapid_mlx/tuning` separates these choices from execution. Profiles can select
only registered values; they cannot create code paths. Scope, capability
requirements and execution-site shape/cache/output-parity probes still run
locally. Importing tuning loads no MLX, reads no profile and starts no hardware
probe, benchmark or disk cache. `runtime_context(info, config=loaded_config,
mode=..., width=...)` snapshots existing facts into immutable profiles. Create
a replacement context when workload changes; do not infer mode from model names.
Existing chip parsing is consumed, not refactored.

Resolution order is **capability/scope block > environment > local calibration
> user overlay profile > builtin profile > knob default**. Boolean environment
spellings use the shared parser. Invalid overrides and calibration values defer.
The calibration slot is still empty. `tuning.dump(ctx)` includes every knob's
value, source (`capability`, `env`, `local`, `overlay:<id>`, `builtin:<id>`, or
`default`), reason and selected profile version (null for a non-profile choice),
plus both loaded profile versions. It excludes device names, model paths,
serials, prompts, accounts and raw environment values.

A profile entry matches only its explicitly declared hardware set. There is no
runtime nearest-neighbor distance or implicit borrowing, including within a
generation/tier. Each hardware selector names a GPU generation and tier and may
restrict cores/memory to inclusive ranges. Selectors within the list are ORed;
software, model and workload restrictions are ANDed. A missing pinned fact fails
the match. Omitted restrictions deliberately include all values, including
unknown core/memory facts. Generalizing a measurement is an explicit reviewed
entry with provenance, never a runtime guess. Matching entries in one layer
must agree on a knob value; a conflict uses the knob default, without falling
through to a lower profile. Agreeing entries use the lexically first ID for
diagnostics. Narrower selectors have no implicit priority: distillation must
produce non-conflicting cells.

The migration preserves all seven historical entries, including their original
software pins and model scope:

| Entry | Explicit hardware set | Values preserved |
| --- | --- | --- |
| `singleton-15-ultra` | GPU generation 15 / Ultra, all core/memory variants, including M3 Ultra 60 and 80 cores | bank views + residual norm on, MLX 0.32.4 |
| `singleton-16-pro` | Generation 16 / Pro, all core/memory variants | pair off, MLX 0.32.4 |
| `singleton-17-max` | Generation 17 / Max, all core/memory variants | pair on, MLX 0.32.4 |
| `affine4-m4-pro-rows` | Generation 16 / Pro, all core/memory variants | rows (2, 4), historically unpinned software |
| `affine8-m3-ultra` | Generation 15 / Ultra, all core/memory variants | affine8 on, MLX 0.32.3 and macOS 26.5.2 |
| `mtp-depth-17-max-36g-qwen35-moe` | Generation 17 / Max, exactly 36 GiB, all core variants | cap 4, MLX 0.32.3 and macOS 27.0.1, qualified model |
| `mtp-depth-15-ultra-256g-qwen35-moe` | Generation 15 / Ultra, exactly 256 GiB, including both 60/80-core variants | cap 3, MLX 0.32.4 and macOS 26.5.2, qualified model |

The first five rows explicitly retain their historical tier-wide generalization;
measured cores/memory are provenance, not selectors. Consequently no previously
covered unmeasured variant changes value in this migration. This includes
variants that formerly reached those rows through nearest-feature fallback.
There was no cross-generation/tier borrowing on main. Unlisted generation/tier
pairs, mismatched software/model scopes and unmeasured MTP memory sizes use the
default, as before. Future narrowly measured entries leave every undeclared
chip at default, even a similar one: that is the agreed explicit-cell semantics.

Builtin data lives at `rapid_mlx/tuning/profiles/builtin.json` and ships as wheel
package data. The optional overlay is
`~/.config/rapid-mlx/tuning-profile.json`, alongside `user-aliases.json`.
Both use schema version 1:

```json
{
  "schema_version": 1,
  "profile_version": "2026.11.10.1",
  "generated_at": "2026-11-10T00:00:00Z",
  "entries": [{
    "id": "example-reviewed-cell",
    "hardware": [{"gpu_generation": 15, "tier": "Ultra",
                  "gpu_cores": [60, 80], "memory_size": [274877906944, 274877906944]}],
    "software": {"mlx": "==0.32.4", "macos": "==26.5.2", "rapid_mlx": ">=1.0"},
    "model": {"family": "qwen3_5_moe_text",
              "geometry": [2048, 512, 256, 8, 4, 64], "quant_mode": "affine"},
    "workload": {"mode": "mtp", "width": 1, "batch": 1},
    "knobs": {"mtp_max_depth": 3},
    "provenance": {"evidence": "path/to/reviewed-evidence.md",
                   "hardware_scope": "Explanation of measured and generalized cells"}
  }]
}
```

Ranges include their endpoints; equal endpoints mean exact match. Memory is in
bytes. Tiers are `base`, `Pro`, `Max`, `Ultra`. Software uses version specifiers;
unknown installed versions fail pinned constraints. Optional model selectors
are `family`, six-element `geometry`, `quant_mode`, and `attention_kinds`;
workload selectors are exact `mode`, `width`, `batch`. IDs and profile versions
are short identifiers. Unknown document keys/unsupported schemas reject the
whole document. Unknown entry/selector keys, unknown knobs/capabilities,
wrong-type/out-of-range values and duplicate IDs reject entries with a debug
log, leaving valid siblings usable. Missing/unreadable/malformed profiles defer
without crashing. Row values serialize as two-element arrays and normalize to
tuples. Booleans are not integer depth values.

Provenance requires `evidence` and permits `hardware_scope`, measured `gpu_cores`
and `memory_size`, `date`, `commit`, `decision`, `plain_tps`, `mtp_tps`,
`plain_arms_tps`, `plain_ranges_tps`, `mtp_ranges_tps`, `repetitions`, and `quality`.
Statistics/quality metadata are opaque evidence, never executable selectors.
Unknown provenance fields are rejected too. Preserve evidence when distilling.
Profiles are loaded lazily once per process; restart to apply a changed file.
There is no new environment variable, file creation, or network access. A future
`rapid-mlx tune --update` will validate a downloaded profile and atomically drop
it at the overlay path; this PR implements neither download nor that command.

Affine8's runtime qualification now resides in the builtin entry's hardware and
software selectors plus `capabilities: ["affine8_attention"]`. The knob registry
requires that qualification before checking environment/calibration/overlay.
Only builtin data can qualify a runtime: overlay capability declarations cannot
expand the safe set. QSA's numerical/ISA requirements and lane ISA requirements
remain local. All execution-site correctness probes remain in force.

Community submissions never write cells directly. The process is **community
measurements → our correctness/performance gates → reviewed data-only PR**.
Record date, commit, exact facts, repetitions, evidence and the complete knob
combination. Kernel combinations require equal greedy IDs, at least three
repetitions, >=2% gain with non-overlapping spreads, and no material other-mode
regression. M3 Ultra retains its qualified pair; M4 Pro stays off. M5 Max's
measured pair qualifies on MLX 0.32.4 (+2.71% plain, MTP -0.06% within spread).
This qualifies the combination, not independent additive gains. The residual
knob stays plain-only. New knobs require registry scope, capability requirements,
working defaults, allowed values and tests before data can select them.

MTP's installed-head single-request boundary resolves `mtp_max_depth` before
building the greedy calibration profile. `RAPID_MLX_MTP_MAX_DEPTH` accepts caps
1–4; invalid values defer. Explicit speculative JSON `num_speculative_tokens`
(including normalized legacy flags) takes priority. The knob requires MTP mode
and an installed head; checkpoint limits still clamp it. The two measured rows
require the qualified 35B affine expert geometry. See the
[M5 evidence](../performance/2026-11-10-m5-mtp-depth.md) and
[M3 evidence](../performance/2026-11-10-m3-mtp-depth.md). A default decision
preserves the configured cap, ordinarily K=1, including catalog/force-opt-in
depths. The acceptance controller can select shallower rounds; calibration
measures every verify width through the cap plus the primary row. Depth
qualification permits width-dependent greedy differences using >=100 paired
fixed questions, absolute accuracy difference <=3 points and exact paired
p>=0.05 against plain and main MTP, alongside the existing speed gate. This
never relaxes kernel arithmetic probes.

Follow-ups:

- Community `--sweep` A/B mode, a tuning block in submissions, and a distillation
  script producing reviewable data-only PRs through our gates.
- Network profile updates through `rapid-mlx tune --update`.
- Local `rapid-mlx tune` measurement/calibration and persistent calibrated cells.
- MTP depth calibration stability, including the varied committed-default
  follow-up recorded in the M3 evidence.
- Unify existing chip fact parsers; acceleration-flag mapping; runtime adaptation.

Further kernel migrations need their own evidence.
