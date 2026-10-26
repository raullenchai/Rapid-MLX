# Parser migration contract

This CPU-only suite exercises the real Chat, Responses and Messages routes with
a deterministic engine. No generated tool is executed. ASGI send interception
records first visible content by input-chunk index, before client aggregation.
A chunk is decoded text, not necessarily one tokenizer token. The engine yields
control between chunks; fixture timing is not a live-model TTFT measurement.

Run `python -m pytest tests/parser_migration`. Replay the baseline independently
with `python tests/parser_migration/replay.py CHECKOUT baseline`, where CHECKOUT
is a checkout of v0.16.9. Use `main` to replay the reviewed main observations.
The baseline and reviewed-head commit IDs are in `differential.json`.
The observations are evidence, including known defects, not desired output.
Behavior changes must update affected reviewed-head rows with an explicit reason;
the baseline rows remain immutable.

The differential corpus covers Qwen/Hermes across all six API modes, thinking
on/off, tools enabled/absent, and character/whole/seven-character partitions.
Native fixtures additionally cover every registered tool alias in all six modes.
The foreign Harmony/Gemma strings in the differential corpus are stress inputs;
they do not claim those grammars are valid Qwen output. Native-family migration
and API consolidation remain separate work.

## Defect ledger

| Contract | Tracking | Desired behavior |
| --- | --- | --- |
| R1 | #5277 | Prompt-derived channel; visible answer within eight chunks |
| R2 | #5277 | One content owner, including EOF |
| R2b | #5277 | One tool index/name/argument sequence, including EOF |
| R3 | #5273 | Reasoning-to-tool transition preserves call and following answer |
| R4 | #5273 | Near-linear full-postprocessor CPU work |
| R5 | #5274 | Foreign malformed syntax does not consume trailing answer |
| R6 | #5274 | Preceding prose establishes literal context for incomplete headers |
| Buffered ownership | #5272 | Tool wire is not also emitted as reasoning |
| Orphan native args | #5274 | Native argument syntax is not published as reasoning |
| Generated fenced echo | #5268 | Pending explicit literal/model-echo policy |

R1, R2 and R2b pass after reverting #5258 and #5267. Other known defects
remain tracked by their contracts. R3 streaming Chat and R4 retain strict
expected failures for v0.16.9 baseline parity, authorized for 0.16.10; their
upstream-aligned parser port is tracked for 0.17.0 in #5273. The unchanged R4
probe on v0.16.9 measured 22.154/46.640/142.830 microseconds per chunk for
1K/4K/16K bodies (2.105x and 3.062x growth; median of three CPU samples).
No scaling threshold or call/argument assertion is relaxed. An unexpected pass requires a deliberate
ledger change. R5 may remain deferred when migration does not reach its grammar.
The fenced-echo expectation is provisional; arbitrary code examples must remain
literal. The prompt fixture file records current-turn initial-channel contracts,
including historical markers, exact empty prefill and assistant continuation.

Run `python -m tests.parser_migration.benchmark` for 1K/4K/16K/64K argument bodies.
It measures process CPU time for complete postprocessing and EOF, excluding
fixture construction and initialization. CI stops at the first >2x per-chunk
increase while the known quadratic implementation remains; a passing run must
reach 64K. Live GPU timing and launch journeys are separate behavior-change gates.

`advertised.json` preserves observations for all 30 catalog reasoning/tool pairs
in all six modes and both thinking settings; a catalog completeness test prevents
new pairs from silently escaping the gate. Required/named/none/auto choices and
undeclared names are separate executable controls. The forced Chat fixture uses
the actual synthetic assistant prefix, as the real engine does.

`ledger.json` classifies the remaining 124 changed Qwen/Hermes focused observations
between the two fixed revisions. The revert refreshes 274 reviewed-head rows: 206
return exactly to the immutable v0.16.9 baseline. The other 68 changed rows
remain different from baseline because the rollback preserves later merged
changes; their exact observations and remaining differences stay in the ledger. Thinking-off implicit reasoning leakage and the
earlier malformed-call policy return as the explicit rollback trade-off. Three
advertised Qwen/LFM streaming rows retain their exact call and text payloads but
now publish ten content deltas starting at chunk one, instead of one at EOF. Its historical classifications include policy-sensitive
and foreign-format stress cases; an intended difference is not a correctness
claim. The test checks that every differing row has matching before/after values
and a rationale. The complete archived focused and Cartesian plain-answer runner
is `full_replay.py TREE focused|matrix OUTPUT`; Cartesian pairs are stress cases,
not a list of advertised combinations.

The normalizer retains duplicate names/arguments and uses actual Responses
argument deltas. Separate clean controls compare streamed accumulation with both
Responses completed items and the terminal response. Full cross-family terminal
parity remains an API-consolidation gate, not a claim of this evidence PR.
