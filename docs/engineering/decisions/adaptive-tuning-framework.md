# Adaptive tuning MVP

Performance choices belong in `rapid_mlx/tuning`, separate from kernel execution
and its existing shape, cache and output-parity probes. Importing tuning loads
no MLX and starts no hardware probe, benchmark or disk cache.

Three variable classes stay distinct:

| Class | Immutable context / policy |
| --- | --- |
| Fixed facts | Existing `mx.device_info()` architecture/generation, chip tier, GPU cores and memory; installed MLX and macOS; loaded config's family, MoE geometry, attention kinds and quantization. Existing chip parsing is consumed, not refactored. |
| Workload | Explicit mode, width and batch. Create a replacement context when these change; do not infer mode from a model name. |
| Knobs | Registry entry with scope predicate, non-bypassable capability requirement, existing environment override and conservative default. |

`runtime_context(info, config=loaded_config, mode=..., width=...)` snapshots
these inputs. `resolve(name, ctx)` first checks scope/capability, then resolves
**environment > empty local-calibration hook > measured table > default**.
Boolean spellings use the shared environment parser; unknown values defer.
Overrides remain live and never bypass execution-site correctness probes.
`tuning.dump(ctx)` returns an identifier-free dict used by the startup debug
log: no model path, device name, serial, prompt, account or environment values.

Table entries describe whole configurations, including multiple knobs. Match
qualified GPU generation/tier and any pinned software versions, prefer exact
core/memory features, then nearest normalized core/memory distance within that
generation/tier. Unknown features use the tier fallback; conflicting equal
matches use the conservative default. Never borrow across generations or tiers.
Historical row tiles retain their original unpinned software behavior; affine8
and QSA retain their strict numerical qualification pins.

To add a knob, register its scope, requirement, environment name and default;
replace the consumer's inline device/OS policy with a resolver call and retain
its probes. Add boundary/override/fallback tests. To add table data, record
capability features, software pins, date, commit, evidence, repetitions and the
complete knob combination. Require equal greedy IDs, at least three repetitions,
>=2% gain with non-overlapping spreads, and no material other-mode regression.
M3 Ultra keeps its qualified pair; M4 Pro stays off. M5 Max's measured pair is
on for MLX 0.32.4: +2.71% plain (three non-overlapping reps), MTP -0.06% within
spread. This qualifies the combination, not independent additive gains. Explicit
per-knob overrides remain available. The residual knob stays plain-only.

Follow-ups: unify `chip_tier.py`, `telemetry/chip.py` and `optimizations.py`
parsers; community benchmark tuning block/schema and reviewed data ingestion;
acceleration-flag mapping; persistent local calibration; runtime adaptation.
None is implemented here. Further kernel migrations need their own evidence.
