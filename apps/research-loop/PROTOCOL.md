# Research Loop protocol 1

Protocol 1 defines three boundaries: model messages, authoritative research records,
and immutable artifacts. A new project configures a task and produces these types;
it does not supply front-end code. Protocol 1 uses `protocol: 1` on journal events
and artifact envelopes, message schemas in `protocol.SCHEMAS`, chart profile
`rl-chart/1`, and deterministic decision rule `hypotheses/2`.

## Model → harness

Only Discuss accepts free text. The harness requests each message using
`response_format: {"type":"json_schema","json_schema":{"name":"<message type>",
"schema":<schema>,"strict":true}}`. `protocol.py` contains the executable strict
JSON schemas: every object disallows extra properties and requires every declared
property. Nullable fields must be explicitly null. No code string has maxLength.

| Message | Fields |
| --- | --- |
| hypothesis | statement, why, prediction (strings) |
| experiment_plan | hypothesis_id (string or null), new_hypothesis (hypothesis object or null), change, prediction |
| edit | base_sha256, edits: [{search_lines: [string], replace_lines: [string]}] |
| conclusion | hypothesis_id, summary_md |
| run_report | summary_md, next_steps_md |

New claims originate in an experiment_plan's new_hypothesis or the hypothesis
message. Existing-claim plans require new_hypothesis=null; new-claim plans require
hypothesis_id=null. The harness checks claim identity, priority, capacity, unique
change and open status. The model never sets verdicts or hypothesis status.

Constrained decoding is not a guarantee. Every reply must finish with
finish_reason=stop, parse as JSON, pass its schema, and pass semantic checks.
Invalid replies are journaled (including rejected text and diagnostics) and retried
once with the diagnostic. Edit messages allow two retries, use 12,000 output
tokens, and include the source hash. A 400/422/500/501 rejection explicitly naming
response_format/schema/constrained decoding selects prompt-and-parse for that
client. Other HTTP errors propagate. Each attempt records strict or
prompt-and-parse; fallback itself is an event. Prompt-and-parse accepts a fenced
JSON object (including embedded JSON), validates it with the same schema and uses
the same retry rules. Legacy proposal objects and SEARCH/REPLACE edit blocks are
normalized to typed messages only in fallback mode; all edit safety checks remain.

Edits match complete lines in the original source exactly once per block. Empty
searches, embedded newlines, ambiguous matches, overlapping intervals and stale
base hashes fail. All blocks apply atomically. Python output must pass ast.parse
and every task.fixed regex before execution. No partially applied edit is saved.

## Authoritative journal and research records

`events.jsonl` is the sole append-only per-run journal. Committed events end with
a newline, have protocol, seq (strictly increasing positive integer), ts and event.
Writers hold an exclusive file lock, flush and fsync before acknowledging. A
partial trailing append is uncommitted and is removed only by a new-protocol
writer recovering its own journal. Legacy journals are never opened for writing.
Records and artifact registrations live in events, with no parallel ledger/index.
Project hypotheses.jsonl stores the cross-run claim inbox; a run journals each
selected claim. The journal is authoritative for all evidence within that run.

Before **each execution attempt**, including baseline and crash-repair attempts,
the harness appends experiment_record with record:

```
{experiment, attempt, hypothesis_id, parent_experiment, change,
 metric, direction, budget, env, seeds, aggregation, noise,
 decision_rule_version, code_sha256, model_id, prompt_id}
```

Attempts start at zero and increase within an experiment; their records never
mutate. The parent is the kept experiment whose code was edited, not simply the
previous experiment. budget contains fixed constraints and task environment;
env records Python/platform and the explicit task variables; aggregation is mean. The code hash is of the exact executed snapshot. Rejected
edits get a nonexecuted attempt with crashed seeds and the unchanged source hash;
no failed edit is presented as executed candidate code.

A result_record contains experiment, attempt, hypothesis_id, per_seed (seed ID →
finite metric value), missing_seeds, crashed_seeds, aggregate, delta, verdict,
and decision_rule_version. Aggregate averages available finite seed values; it
is descriptive if seeds are missing. Delta is the raw signed difference from
the frozen kept parent's aggregate. A complete result is improved/worse only
when its direction-adjusted delta exceeds ±noise; equality is no effect. A
complete baseline has no effect (no parent). All seeds crashed yields crashed;
any mixture of missing/crashed and measured seeds yields incomplete. Incomplete
results cannot promote the kept parent. Nonfinite and missing metrics count as
crashed seeds. stdout remains curve JSON lines plus a final `RESULT {...}`.

Hypothesis reducer **hypotheses/2** uses the latest attempt per experiment across
project runs. Intermediate repair attempts remain auditable but do not count
as separate experiments. Only complete experiments (no missing/crashed seeds)
provide improved/worse evidence:

- Both improved and worse: inconclusive, conflicting_evidence=true.
- Improved without worse: supported, even if another experiment is incomplete.
- Otherwise conclude at experiments_per_hypothesis (default 2), or when every
  associated run has ended: any worse → refuted; only no-effect, crashed,
  incomplete or unfinished evidence → inconclusive.
- Without that threshold: testing if any experiment exists, otherwise untested.

Every derived claim carries the rule version, conflicting_evidence,
missing_evidence, run_ended and a reason including the run-ended reason.
research_ended closes evidence before reports are generated; done closes SSE.
Stopping finishes the current experiment and generates reports before done.
These rules describe evidence under the configured noise threshold, not causality.
Public labels are H1, H2… in project claim order; internal IDs remain opaque.

## Artifacts

The primary kinds are **code, report, data, chart**. image, log and file ensure
that every output has a safe representation, including formats unknown to the UI.

| kind | Representation |
| --- | --- |
| code | Escaped source/diff, copy and download; parent code refs |
| report | Escaped Markdown + local KaTeX; no raw HTML |
| data | UTF-8 CSV or flat JSONL; sortable first 500 rows, column type/unit metadata |
| chart | Validated declarative JSON referencing a registered data artifact |
| image | PNG/JPEG/SVG via img, never inline artifact markup |
| log | Plain monospace text, tail and search, never Markdown |
| file | Download only |

Each registration has this envelope (columns and supersedes are optional):

```
{protocol:1, id, revision, kind, role, title, path, mime, bytes, sha256,
 producer:"harness|script|model",
 owner:{scope:"run|hypothesis|experiment|seed", run, ...related IDs},
 refs:[artifact IDs], provenance:{model_id,prompt_id,code_sha256,env}, created,
 columns?:{field:{type?,unit?}}, supersedes?:{id,revision}}
```

owner.scope is exactly one placement owner. Run owners have only run; hypothesis
owners add hypothesis_id; experiment owners add experiment and optional
hypothesis_id/attempt; seed owners also require seed. Related IDs do not create
additional owners. role describes purpose (primary_code, diff, curve, stdout,
stderr, conclusion, summary); renderers never infer behavior from artifact IDs.
Each envelope identifies one immutable revision. Same id + any registered hash is
idempotent; changed content increments revision and supersedes the preceding
revision. All revisions remain listed and bundled. Refs resolve to the latest
registered revision at registration time; harness code/curve IDs are attempt
specific, so their references do not drift. Producers should use fresh IDs for
new logical datasets and reserve revisions for corrections.

Files are copied to artifacts/<id>-r<revision>.<extension> using temp + fsync +
rename + directory fsync **before** the artifact event. Hash/size describe the
copied file. Orphan files after a crash are unregistered and never served or
listed. No generated artifact index is authoritative. The run ledger is derived
CSV, never a mutable artifact with a stale envelope.

### Chart profile rl-chart/1

This is our SVG renderer's strict, small subset of Vega-Lite field names, with
Research Loop's explicit artifact reference and band/reference extensions:

```json
{"$schema":"rl-chart/1","mark":"line","data":{"artifact":"data-id"},
 "encoding":{"x":{"field":"t","title":"Seconds","scale":{"type":"linear"}},
             "y":{"field":"energy","title":"Energy","scale":{"type":"linear"}},
             "color":{"field":"seed"}},
 "band":{"low":"lower","high":"upper"},
 "reference":{"value":-44.3268,"label":"Known minimum"}}
```

mark is line, point or bar; scale is linear or log (positive values only for log).
x/y are required; title/scale/color, band and reference are optional. layer is
an optional array of at most 12 {mark?,encoding?} overrides inheriting the data
reference. No recursive layers, external data, expressions, transforms or unknown
fields. Referenced encoding/band fields must exist in the data. Color groups
series. The task.curve configuration generates default data/chart artifacts,
preserving existing templates. Comparison places two experiments' charts of the
same role side by side. The live preview may use journal curve points before
immutable seed artifacts are finalized.

## Script → harness and safety

The script optionally writes into $ARTIFACT_DIR and lists files in
$ARTIFACT_DIR/manifest.jsonl, one object per line:

```
{"kind":"data","path":"coordinates.csv","title":"Atom coordinates","role":"coordinates"}
```

role, refs and columns are optional. Refs are registered artifact IDs; entries
are processed in manifest order. The harness also supplies $ARTIFACT_ID_PREFIX;
entry index N receives ID "<prefix>-N" (zero-based). A script producing its own
chart can reference an earlier entry by that ID. Undeclared files are ignored. Manifest paths
must be relative, contained, regular files, with no symlink component. Read size
is bounded. Kind/extension/content checks reject mismatched MIME, invalid chart
specs, nested JSONL and binary data disguised as text. Arbitrary file payloads
are permitted only as download-only file artifacts. stdout and stderr are drained
concurrently and registered per seed as logs. Oversized logs are rejected with
an event; capture memory is bounded. Experiments remain ordinary local programs,
not a security sandbox.

Registration enforces 5 MiB per artifact and 200 MiB per run including all
registered revisions. Over quota is rejected and logged, never silently stored.
A rejected manifest entry does not prevent valid following entries. Script files
are copied before hashing so later changes cannot change registered content.
Markdown is escaped; accepted links are http(s) or same-origin paths only. SVG is
served with CSP sandbox and default-src none and only embedded via img. Downloads
use attachment and application/octet-stream; all responses use nosniff. Unknown
kinds or invalid envelopes degrade to file downloads with quiet warnings. They
cannot insert HTML into the page or execute chart expressions.

## HTTP and live transport

All paths below are relative to /api/projects/<project>/runs/<run>:

- GET / → events, artifacts, records and watermark from one journal snapshot.
- GET /artifacts?experiment=7&kind=data&role=curve → envelopes + watermark.
- GET /artifacts/<id>?revision=1 → validated bytes; ?download=1 forces attachment.
- GET /artifacts/<id>?rows=500 → data rows, total, columns and watermark (max 500).
- GET /records → typed records + watermark.
- GET /ledger.csv → CSV derived from result records.
- GET /bundle.zip → all revisions, envelopes.json, records.json, journal snapshot,
  ledger.csv and this spec; unavailable/corrupt payloads get explicit warnings.
- GET /events?after=<seq> → SSE id: <seq>; Last-Event-ID overrides the query cursor.

List, then subscribe after its watermark: no event between snapshot and subscribe
is lost. Resume is exclusive (>cursor); the browser deduplicates seqs. Replay
starts at zero and retains original timing with curve compression. Evidence
transition SSE frames also carry a derived hypotheses snapshot from the Python
reducer at that seq, so replay does not apply future conclusions. This projection
is derived from the journal; it is not another stored index. Heartbeats
keep connections open; done is terminal and closed signals a dead process.

## Read-only legacy adapter

Runs without protocol events use original complete-line ordinal seqs. The adapter
projects experiment/result records, code/diffs, curve data/default charts and
loop logs with deterministic IDs. It makes no writes, repairs or migrations to
projects/*/runs. Synthetic envelopes are returned in snapshots, not inserted
into the historic journal. Legacy logs and curves can grow; fresh snapshots
reflect their current hashes, and watermark tracks only original events. The UI
refreshes the snapshot after seed/result/end events. Original task labels and
values are preserved. Missing provenance/seed results remain explicitly unknown;
legacy comparative verdicts are preserved for display. This supports an old loop
already running in memory while the server serves protocol-1 projections.
