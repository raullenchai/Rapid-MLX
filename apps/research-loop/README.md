# Research Loop

A local, project-by-project autoresearch workbench. Set a research goal, discuss
hypotheses with a local model, run fixed-budget experiments, and inspect the code and
measured results together. The browser is reusable across project templates.
No frontend build, CDN, database, or new dependency is required.

## Run

Use a Python environment with the requirements of your chosen template. Grokking
needs MLX; the sorting example uses only Python's standard library.

```sh
python apps/research-loop/server.py \
  --models q35=http://127.0.0.1:8482/v1,q27=http://127.0.0.1:8481/v1
```

This connects to models you have already started; it does not manage model
servers. The first entry is the default. Model keys are also sent as the model
name to their OpenAI-compatible endpoint. The single-model flags still work:

```sh
python apps/research-loop/server.py \
  --base-url http://127.0.0.1:8000/v1 --model my-local-model
```

Open http://127.0.0.1:4900. Other flags: `--host` (default `127.0.0.1`), `--port`
(4900), and `--rounds` (10). The loop and experiment subprocesses use this
server's Python interpreter. Keep the server on localhost: this development
tool has no authentication. Use local endpoints to keep discussions on this Mac.

## Work on a project

1. **Projects** (`#/`): create a project with a title, starting template, and goal.
   Templates are copied; each project's code and settings are independent.
2. **Discuss** (`#/p/<slug>`): edit and save the research goal, talk with a chosen
   model, and add your own claim or an assistant message as a hypothesis. The model sees the
   goal, metric, fixed budget, previous discussion, and a compact verdict ledger
   from up to five recent runs (60 entries). Discussion context is sent when you send
   a message; attaching PDFs can also request metadata from the configured chat model.
3. **Experiments** (`#/p/<slug>/experiments`): choose the number of experiments and **Start
   experiments** from Discuss. The baseline runs first; untested researcher hypotheses run
   before model proposals. The main chart shows experiment number versus metric,
   points coloured by hypothesis, labelled improvements, and the running best. It respects
   the configured direction and noise threshold. The smaller chart shows the
   selected experiment’s measurements. Discussion stays reachable in the side rail.
4. **Results** (`#/p/<slug>/results`): expand a hypothesis in the ledger to see its experiments.
   Select an experiment to inspect its claim, why, prediction, curve, final code,
   and diff against its kept parent. Download the best code, event ledger, and log. “What we learned” groups
   supported, refuted, and inconclusive hypotheses from numerical evidence.
   Model-written conclusions and the run report are attached as reports; they
   never determine status.

One run may be active per project. The goal is fixed for that run; save changes
before starting. The model picker chooses the discussion model and the model for
the next run. Existing runs keep their model. Pause and Stop take effect between
loop steps. Stop clears Pause; an ongoing model request or experiment finishes
its current step first. Closing the browser or server does not kill the loop.
Untested hypotheses remain in the project across stops and restarts. Each experiment
number is still the recorded `round` number; file names and artifact URLs keep that
contract. The baseline is experiment 0 and does not test a hypothesis.

## Hypotheses and experiments

Discussion → many **Hypotheses** → one or more **Experiments** per hypothesis →
evidence supporting or refuting the claim. A hypothesis is a testable claim about
the system, with why and prediction: “Stronger weight decay makes the network
generalise sooner.” An experiment is a concrete change testing it:
`WEIGHT_DECAY 0.01 -> 1.0`, evaluated on all seeds under the fixed budget against
the current best.

Experiment verdicts are **improved**, **worse**, **no effect**, **crashed** or
**incomplete**. Only a complete improvement beyond noise updates the best code.
Status comes from the deterministic `hypotheses/2` reducer in hypotheses.py:

- **untested**: no experiments;
- **testing**: awaiting enough evidence;
- **supported**: complete improvement with no conflicting complete worse result;
- **refuted**: worse evidence without improvement, once the experiment limit or
  end of the associated runs closes the claim;
- **inconclusive**: conflicting results, or only no-effect/crashed/incomplete
  evidence when the limit or end of the run closes the claim.

`experiments_per_hypothesis` defaults to **2**. Failed edits count as crashes;
repairs and seeds do not consume extra slots. Missing seeds cannot support or
refute a claim. Finished/stopped runs conclude open claims from available evidence
and retain the reason. Internal IDs stay opaque; the UI labels claims H1, H2….

Before each experiment the loop rereads project hypotheses across runs. Untested
researcher claims go first. Otherwise the model selects an open claim with capacity
or proposes a new claim. Concluded claims and duplicate changes are rejected with
one diagnostic retry. A new statement has at most 15 words. Typed plans use:

```json
{"hypothesis_id":"h3","new_hypothesis":null,"prediction":"Earlier generalisation","change":"WEIGHT_DECAY 0.01 -> 1.0"}
```

For a new claim, use hypothesis_id=null and new_hypothesis containing statement,
why and prediction. All model messages, including edits and reports, use strict
JSON-schema decoding with validation and an older-engine fallback; see
[PROTOCOL.md](PROTOCOL.md) for the complete contract.

## Files and reproducibility

```text
templates/<slug>/
  program.md  task.json  <editable>
projects/<slug>/                     # gitignored local research data
  project.json                      # title, template, created, model
  program.md  task.json  <editable>
  discussion.jsonl                  # role, content, ts, model
  hypotheses.jsonl                  # append-only claims: id, statement, why, prediction, source, created
  runs/<run-id>/
    run.json                        # model, rounds, task snapshot, project
    events.jsonl  loop.log           # sequenced records and artifact registrations
    artifacts/                      # immutable files, all revisions
    round0_<editable>            # exact baseline code
    roundN_<editable>            # exact experiment code, including repairs
    # Per-seed curves and logs are registered as data/chart/log artifacts.
    best_<editable>  history.json
```

An improved experiment updates the best; worse and no-effect changes are discarded.
Complete results are means across the configured
seeds; the noise threshold determines the verdict and is not a confidence
interval. The best download falls back to the kept numbered code if necessary
(including round 0 when nothing improved). Missing legacy code is reported as
unavailable. Failed edits have no fabricated code or diff.

`tasks/grokking/` moved to `templates/grokking/`. The local `grokking` project
contains the original `20261118-113651-acde31` and `recorded-q35-curves` runs,
moved without changing their contents. Unnamed legacy folders remain under the
old top-level `runs/`; the project UI does not discover that legacy directory.
Existing installations must move their runs into the desired project explicitly.

Direct loop use also accepts a project directory:

```sh
python apps/research-loop/loop.py apps/research-loop/projects/<slug> \
  apps/research-loop/projects/<slug>/runs/<new-run-id> \
  --base-url http://127.0.0.1:8000/v1 --model my-local-model --rounds 10
```

Use a fresh run directory. Local experiment code runs with your account's
permissions. Only use trusted templates and model endpoints; fixed-line checks
are experimental constraints, not a sandbox.

The project hypothesis log is append-only JSONL. Each complete row creates an
immutable claim with a sequential ID (`h1`, `h2`, …). An OS file lock serializes
ID allocation/appends between the server and loop. Status is computed on reads
by joining experiment events across runs; it is not stored as a model assertion.
Readers ignore incomplete trailing rows; a writer discards an interrupted trailing
fragment under the lock before appending. Complete rows remain immutable. Do not
hand-edit a live log.

Compatibility is read-only for runs: old `idea` events without `hypothesis_id`
derive a claim from their `hypothesis` text, deduplicated by identical text across
runs. Internal `legacy-<digest>` IDs prevent collisions with persisted `hN` IDs;
the UI presents all claims as H1, H2, etc. Legacy histories may contain more
than the new per-hypothesis limit; every recorded experiment stays visible. Old `supported/refuted` experiment
verdicts render as `improved/worse`. Old event files, curves, code, diffs, and logs
are never rewritten, including an old loop still writing events in memory.
Old project `ideas.jsonl` queues migrate idempotently when starting the new loop
or adding a claim, leaving the original file intact. Old run inboxes remain
untouched. The `/ideas` route remains an alias for adding a project hypothesis;
the old run `/idea` route is retained only for clients of a pre-v4 loop. New
loops consume project hypotheses, not run inboxes. Restart the web server to
load the new API; an already loaded old loop retains its old scheduling policy.

## Replay

```text
http://127.0.0.1:4900/#/p/grokking/experiments?run=20261118-113651-acde31&replay=1&speed=8
```

The Replay button opens this mode for a selected completed run. Change `speed`
in the URL for a different rate. Replay uses original event timestamps divided
by speed, with curve gaps capped at 150 ms. A badge remains visible and mutation
controls are disabled. SSE IDs are monotonic per-run sequence numbers (complete-line ordinals for legacy
journals); reconnection uses `Last-Event-ID` or `?after=<seq>` and ignores incomplete JSONL writes until they are complete.
After a server restart, orphaned loops are not adopted for controls; their event
files remain readable/tailable. A tracked process that exits without `done`
produces a closed event; inspect `loop.log` for the failure.

## Add a template

Create `templates/<lowercase-slug>/` with these three files, then create a new
project from it in the UI. No web edits are needed. Refresh Projects to discover
new templates.

- `program.md`: objective, fixed budget, permitted changes, and constraints.
- `task.json`: `title`, `editable` (a simple `.py` filename), `metric` (RESULT
  key), `direction` (`max` or `min`), `noise`, `seeds`, `fixed` (regexes which must
  remain present); optional `goal`, `env`, `metric_label`, `unit`, `curve`, and
  `experiments_per_hypothesis` (default 2).
- The editable script: read `SEED` from the environment and finish with
  `RESULT {"<metric>": <number>, ...}` on stdout. Flush live output. Do not change
  evaluation data or budgets between experiments.

`unit: "%"` means stored fractions are displayed as percentages; other units
are appended without scaling. Without a unit the raw metric is shown. To show
live curves, emit JSON lines with `step` and the configured series keys:

```json
{
  "title": "Improve a measured score",
  "editable": "experiment.py",
  "metric": "score",
  "metric_label": "Validation score",
  "unit": "%",
  "direction": "max",
  "noise": 0.01,
  "seeds": [0, 1],
  "fixed": ["^BUDGET = 1000$"],
  "goal": 0.95,
  "curve": {
    "x_label": "Steps",
    "series": [{"key": "score", "label": "Validation"}]
  }
}
```

Templates without curves, such as `sort-speed`, show final per-seed measurements.
The sorting example fixes 2,000 integers and 20 repetitions, measures milliseconds,
and minimizes runtime. It is a starting example, not a reviewed benchmark.
Keep a project's metric, units, and editable filename stable across its runs;
create another project for a different measurement contract.

## API

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/api/templates` | Template metadata |
| GET / POST | `/api/projects` | List / create `{title, template, goal}` |
| GET | `/api/projects/<p>` | Project, program, task, runs, discussion, hypotheses with experiments/statuses, models |
| PUT | `/api/projects/<p>/program` | `{program}` |
| POST | `/api/projects/<p>/chat` | `{message, model}` → streamed completion; saves both turns |
| POST | `/api/projects/<p>/hypotheses` | `{statement, why?, prediction?, source?}` → hypothesis and derived project hypotheses |
| POST | `/api/projects/<p>/runs` | `{rounds, model}` → starts run |
| GET | `/api/projects/<p>/runs/<r>` | Run metadata, events, available artifact names |
| GET | `/api/projects/<p>/runs/<r>/events` | SSE, optional `replay=1&speed=N` |
| POST | `/api/projects/<p>/runs/<r>/<action>` | `pause`, `resume`, `stop` |
| GET | `/api/projects/<p>/runs/<r>/files/<name>` | Whitelisted text artifact; `download=1` attachment |
| GET | `/api/projects/<p>/runs/<r>/diff/<round>` | Final code diff versus recorded parent |

Project/template slugs use lowercase letters, digits, and single hyphens.
Run IDs accept letters, digits, underscores, and hyphens for legacy data.
Traversal and symlinks are rejected. Download names are limited to numbered code,
best code, `events.jsonl`, `loop.log`, and `history.json`. Writes reject
cross-origin browser requests. An upstream chat failure saves the user message;
a successful or partial assistant reply is saved independently of the browser's
connection. The original top-level task/run/chat APIs have been replaced by the
project API.

## Check

```sh
PYTHONPYCACHEPREFIX=/private/tmp/rapid-mlx-grokdemo-v5-pycache \
  python3 -m pytest apps/research-loop/tests -q
ruff check apps/research-loop/{protocol.py,loop.py,server.py,hypotheses.py,tests,templates/lj-cluster/cluster.py,templates/no-three-ap/search.py}
ruff format --check apps/research-loop/{protocol.py,loop.py,server.py,hypotheses.py,tests,templates/lj-cluster/cluster.py,templates/no-three-ap/search.py}
node --check apps/research-loop/web/app.js
node apps/research-loop/tests/test_web.mjs
```

Tests use ephemeral stub-model ports and pure Python; they do not require Metal
or touch existing model servers. Node checks cover state and authored markup,
including both real local streams when present. They do not validate browser
layout. Review at 1920×1080 and 1440×900 in a real browser before recording.

## Protocol 1

[PROTOCOL.md](PROTOCOL.md) defines typed model messages, frozen experiment/result
records, and immutable artifacts. New runs use a sequenced `events.jsonl` journal
and strict JSON-schema decoding with validated older-engine fallback. Existing
runs use a read-only adapter; their files are never migrated.

Experiment artifacts appear under Chart · Data · Code · Log · Notes. Reports
support Markdown and KaTeX; data previews show at most 500 sortable rows. Charts
can be compared by role. Download all exports artifacts, envelopes, records,
the journal and protocol spec in one reproducibility bundle. The derived ledger
is available at `runs/<run>/ledger.csv` under the project API.

Scripts keep their curve JSON lines and final `RESULT {...}`. To add outputs,
write files into `$ARTIFACT_DIR`, then declare each in `manifest.jsonl` with kind,
path, title and optional role/refs/columns. The chemistry template demonstrates
coordinates, an SVG projection and a summary without project-specific UI code.

## Project literature

Start with an already running OpenAI-compatible embedding endpoint:

```sh
python apps/research-loop/server.py \
  --models q35=http://127.0.0.1:8482/v1 \
  --embeddings http://127.0.0.1:8483/v1 \
  --embedding-model embeddinggemma-2-4bit
```

The embedding model name defaults to `embeddinggemma-2-4bit`. Without
`--embeddings`, Literature clearly says **Not configured** and discussion and
experiments continue to work. The server never loads an embedding model or MLX.
The existing local-search extraction, chunking, semantic admission and rank-fusion
helpers are reused. PDF extraction needs the engine environment's `pypdf` and
`psutil`; missing readers produce a per-file error without blocking other files.
No new packages are installed by the workbench.

The visible **Literature** card sits directly under the goal in Discuss, and **2 Literature** in the
research strip scrolls to and highlights it. Use **Add papers** to upload PDF, Markdown, UTF-8 text, TeX or HTML files (20 MiB each),
or attach an absolute local folder path. Folder attachments are stored in
`project.json` as `literature_dirs` and are read-only: **Detach** removes the
attachment, never the original files. **Remove** deletes an uploaded copy only.
Existing uploaded filenames are replaced on upload. Hidden/sensitive files and
symlinks are excluded using the engine's local-search policy.

Uploads, folder changes and removals start background indexing; **Re-index**
rescans for changes made outside the UI. Status shows files, passages, file
progress and individual errors. Changes to attachments wait until indexing
finishes. The atomic `literature_index/index.json` snapshot holds file SHA-256,
stable chunk IDs, PDF page or section/line locators, character spans in extracted
text (per page for PDFs), passages and normalized vectors. Unchanged files reuse
vectors; changed/new files are embedded, removed/failed files lose their old
passages. Changing endpoint or embedding model requires re-indexing. Snapshots
live under gitignored `projects/`; templates are unchanged.

Discuss retrieves up to six semantic/lexical passages, removes overlapping
chunks, and supplies numbered sources with instructions to cite only supporting
claims and acknowledge silence. A reply saves `citations` only for supplied
source markers actually present in its text. Click an inline `[S#]` marker or expand a citation chip to read the
saved passage, filename, page/section and chunk ID. A Sources note explains empty
retrieval, unused sources or an unavailable endpoint. Citations are source
snapshots, so old discussions remain inspectable after a file is removed.
**Add as hypothesis** copies that reply's citations into the append-only claim's
optional `sources` field; older claims remain readable. Sources appear on cards
and in Results. Citation markers identify provided text; they do not certify that
the model's interpretation is scientifically correct.

Experiment planning receives up to three passages for the priority hypothesis
(or open claims/goal when selecting a new claim). Before implementing the chosen
experiment, retrieval uses that exact hypothesis statement again. Each supplied
context is recorded as a `literature_context` journal event with source IDs and
passage snapshots. Typed plan/edit schemas and numerical verdict rules are
unchanged. Direct `loop.py` use accepts the same embedding flags.

Paper labels use usable PDF metadata, otherwise one small strict-JSON request
with first-page text to the configured default chat model. The index caches this
metadata (including page count and filename fallbacks) by SHA-256, separately from
vectors. Edit and save a paper label on the Literature card; overrides live in
`project.json` under `literature_labels` and appear in chips, passages, hypothesis
cards and Results. Saved passage text stays immutable. Each Discuss prompt remaps
historical and retrieved citation markers into a shared table of source snapshots,
so a follow-up cannot silently bind an old marker to a different paper. Legacy
markers without saved provenance are marked unavailable in model history.
Discuss asks for a short answer and 1–2 claim/why/prediction hypotheses, aiming
under 250 words; the prompt cannot guarantee model compliance.

**Privacy:** literature indexing and retrieval send text only to the configured
embedding endpoint and (for PDF metadata extraction) the configured default chat
model endpoint; discussion and loop messages go only to the
configured model endpoints. No paper URLs are fetched, no external search service
is used, and extraction does not follow HTML links. With localhost endpoints,
passages, vectors, discussions and sources stay on this machine. An external
endpoint will receive the text provided to it. Literature adds no telemetry,
frontend dependency, CDN or build step.

Literature APIs are relative to `/api/projects/<p>`:

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/literature` | Configuration, state, files/passages, progress, errors, attached directories |
| POST | `/literature/upload` | `{name, content}` (base64 bytes); starts indexing |
| POST | `/literature/folder` | `{path}` (absolute read-only directory); starts indexing |
| POST | `/literature/remove-folder` | `{path}`; detaches and re-indexes |
| POST | `/literature/remove` | `{name}`; removes uploaded copy and re-indexes |
| POST | `/literature/label` | `{file, label}`; saves a display label without re-embedding |
| POST | `/literature/index` | `{}`; background incremental re-index |
| GET | `/literature/file?file=<encoded indexed filename>` | Source bytes from a current literature root; PDF links use `#page=N` |
| POST | `/hypotheses` | Also accepts `discussion_index` to copy a saved assistant turn's citations |

Indexing currently uses one thread per active project and exhaustive in-memory
vector search; it is intended for project-sized collections. There is no OCR for
scanned PDFs or filesystem watcher. TeX is indexed as source, not compiled.
Interrupted indexing leaves the previous complete snapshot; re-index after a
server restart to resume. Model grounding is prompt-based and requires human
review.

Source cards appear above assistant answers; display numbers follow first use
within that answer while saved `S#` identifiers keep their historical mapping.
Inline superscript buttons show a passage popover on hover or keyboard focus.
Clicking a citation or card opens a keyboard-accessible source drawer with the
section, page, full passage and adjacent passages. Escape closes the drawer and
returns focus. Quoted phrases are highlighted when they match the passage;
otherwise shared words select a passage sentence for visual orientation. This
highlight is not an entailment check. Hypotheses and Results use the same locators.

Index version 2 detects numbered/appendix and standard headings in extracted PDF,
Markdown, TeX and HTML text, including common small-cap extraction artifacts.
Sections carry across pages, but chunks split at page and heading boundaries,
so `page` and `page_end` currently match. Old indexes rebuild on re-index (vectors
are re-embedded for this version change); PDF metadata caches remain reusable.
References passages are excluded from retrieval. A named first author scopes
retrieval to that paper; section questions favor defining heading passages.
Heading detection remains heuristic and cannot infer headings missing from the
extracted text. Real-PDF tests use optional read-only fixtures in the supplied
external papers directory and skip when those files are unavailable.

Discuss streaming starts with a `data:` frame containing `retrieval` counts and
provided `sources`, followed by ordinary OpenAI content frames. Saved replies
include `retrieval` and `citation_sentences` (source number to exact claim
sentences). Only markers in the provided source table produce saved citations. Combined
markers such as `[S1, S2]` become individual markers; unknown identifiers are
discarded from saved assistant text.
The source-file API accepts only indexed filenames still within the project's
literature directory or a currently attached directory. It rejects traversal and
symlinks, bounds reads, uses source content types and sandboxes HTML without
scripts or external resources. Removed or detached files remain readable as saved
passage snapshots, but their original-file links become unavailable.

## Rapid Hub mode

The [Hub services guide](../../hub/README.md) documents starting Hub and its
minimal `rapid-plugin.toml` supervisor. `server.py --hub` reads the scoped token,
assigned port and data root from the supervisor environment. All model calls use
Hub's model-routed `/hub/v1/chat/completions` and `/hub/v1/embeddings` endpoints,
with engine-qualified model ids. Discussions are interactive; loop subprocesses
inherit background priority and project identity. Standalone flags, data paths
and unauthenticated behavior are unchanged.

In Hub mode, open Research Loop with **Open** on `hub.localhost:4800`.
The browser visits `research-loop.localhost:<assigned-port>`; a single-use,
60-second code is redeemed server-to-server for a host-only HttpOnly
SameSite=Strict plugin session. The code is stripped by redirect and responses
set Referrer-Policy: no-referrer. Pages and API require this session; health stays
public. Hub's cookie is scoped to a different hostname. Both processes bind to
127.0.0.1; browsers resolve `*.localhost` to loopback.

**For reproducibility, Research Loop enables Hub payload capture for its own
projects at startup and when creating projects (24-hour payload retention).**
This includes model-visible literature, prompts and responses. Other Hub projects
remain metadata-only unless explicitly opted in. Captures redact secret-named
fields, are bounded to 64 KiB per field and share Hub's disk cap; metadata has a
7-day maximum retention. Capture is bounded evidence, not a full archival copy.
The Hub journal API supports project deletion and capture opt-out; restarting
Research Loop enables capture again for its projects.

Jobs acknowledge cancellation as `cancelling`, terminate the loop and its
experiment process groups, wait for exit, then report `cancelled`. This is more
immediate than standalone's cooperative Stop. Jobs with no heartbeat for 10
minutes, or active jobs surviving a Hub restart, show `interrupted`; runs are not
automatically adopted. Local-search imports remain pending a Hub retrieval
service.

Plugins and experiments are **trusted local code, with no sandbox**. A browser
session protects local HTTP access, not account files from installed plugin code.
