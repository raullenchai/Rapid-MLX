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

Remaining expected failures use `xfail(strict=True)`: an unexpected pass requires
a deliberate ledger change. R5 may remain deferred when migration does not reach its grammar.
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

`ledger.json` classifies all 324 changed Qwen/Hermes focused observations between
the two fixed revisions. Its historical classifications include policy-sensitive
and foreign-format stress cases; an intended difference is not a correctness
claim. The test checks that every differing row has matching before/after values
and a rationale. The complete archived focused and Cartesian plain-answer runner
is `full_replay.py TREE focused|matrix OUTPUT`; Cartesian pairs are stress cases,
not a list of advertised combinations.

The normalizer retains duplicate names/arguments and uses actual Responses
argument deltas. Separate clean controls compare streamed accumulation with both
Responses completed items and the terminal response. Full cross-family terminal
parity remains an API-consolidation gate, not a claim of this evidence PR.

## Selected grammar migration

Phase four retains the frozen baseline and reviewed-main observations unchanged.
`differential_migration.json` and `advertised_migration.json` record exact
before/after values and scenario-specific reasons for intended changes. Tests
apply those explicit overrides and reject unrecorded differences. The full
4,608-case replay changes only the selected Qwen/Hermes and Qwen/GLM pairings;
other native families, including the distinct `qwen3_coder_xml` class, remain on
the legacy path. The `qwen`, `qwen3`, and `qwen3_xml` aliases use the migrated
JSON-wrapper/bracket-call grammar.

R1, R2, R2b, R3, R4, R6, and the #5272–#5274 selected-family contracts now pass.
Three strict expected failures remain: R5's two foreign-envelope route cases
and #5268's generated fenced-echo policy. Literal fenced examples are retained;
no generated-echo suppression heuristic is introduced. Preceding prose keeps
ambiguous/incomplete tool headers literal (R6). Unknown foreign controls remain
literal in the current channel; migration of their native grammars is deferred.

`test_selected_routes.py` adds delayed-engine tests on all three real routes,
both prompt modes and Qwen/Flint prompt fixtures, including first-visible input
index, multiple deltas, literal serialization, terminal lifecycle, and tool
partition parity. Indexed incomplete/invalid calls fail explicitly and cannot
become executable calls through an EOF fallback.

The CPU fixture now explicitly sets intermediate `GenerationOutput.finished`
to false. Earlier evidence inherited its terminal default and must not be used
as the comparison for this corrected full-stream benchmark. The corrected
v0.16.9 baseline loses its 64K call; record that invalid result rather than
counting it as a valid throughput sample. The current gate verifies valid calls
at every size and requires no greater than 2x per-chunk growth.
