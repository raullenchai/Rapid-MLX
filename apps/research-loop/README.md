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
   from up to five recent runs (60 entries). Nothing is sent until you send a message.
3. **Experiments** (`#/p/<slug>/experiments`): choose the number of experiments and **Start
   experiments** from Discuss. The baseline runs first; untested researcher hypotheses run
   before model proposals. The main chart shows experiment number versus metric,
   points coloured by hypothesis, labelled improvements, and the running best. It respects
   the configured direction and noise threshold. The smaller chart shows the
   selected experiment’s measurements. Discussion stays reachable in the side rail.
4. **Results** (`#/p/<slug>/results`): expand a hypothesis in the ledger to see its experiments.
   Select an experiment to inspect its claim, why, prediction, curve, final code,
   and diff against its kept parent. Download the best code, event ledger, and log. “What we learned” groups
   supported, refuted, and inconclusive hypotheses from numerical evidence; it generates no claims or model-written conclusions.

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

Experiment verdicts are **improved**, **worse**, **no effect**, or **crashed**.
Only an improvement beyond the noise threshold updates the best code. Hypothesis
status is derived, never supplied by the model:

- **untested**: no experiments;
- **testing**: experiments exist, but the conclusion rule has not been met;
- **supported**: any experiment improved beyond noise (concluded immediately);
- **refuted**: `experiments_per_hypothesis` completed experiments with no improvement,
  including any mixture of worse, no effect, and crashes with a measured result;
- **inconclusive**: the limit was reached and every experiment crashed.

The task setting `experiments_per_hypothesis` defaults to **2** (a positive integer).
Failed edits count as crashed attempts. Repairs and seeds belong to the same
experiment and do not consume additional hypothesis slots. A stopped experiment
without a final result stays testing; it reserves its slot and supplies no conclusion.

Before each experiment the loop rereads project hypotheses and their experiments
across runs. Researcher hypotheses (`you` or `discussion`) with no experiments go
first. Otherwise the model chooses an open hypothesis with capacity or states a
new claim with its first experiment. Concluded claims and duplicate changes are
rejected, with one retry. New model claims have a 15-word statement, 30-word why,
and 12-word prediction limit. The proposal protocol is:

```json
{"hypothesis_id": "h3", "why": "Less overfitting", "prediction": "Earlier generalisation", "change": "WEIGHT_DECAY 0.01 -> 1.0"}
```

Use `"hypothesis_id": null` plus `"hypothesis": "<claim>"` only for a new claim.

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
    events.jsonl  loop.log           # experiment events include hypothesis_id
    round0_<editable>            # exact baseline code
    roundN_<editable>            # exact experiment code, including repairs
    roundN_seedS_curve.json         # when written by the experiment
    best_<editable>  history.json
```

An improved experiment updates the best; worse and no-effect changes are discarded.
The verdict algorithm is unchanged. Numbers are means across the configured
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
the UI abbreviates these as `hL1`, `hL2`, etc. Legacy histories may contain more
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
controls are disabled. SSE IDs are byte offsets; reconnection uses
`Last-Event-ID` and ignores incomplete JSONL writes until they are complete.
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
PYTHONPYCACHEPREFIX=/private/tmp/rapid-mlx-grokdemo-v4-pycache \
  python3 -m pytest apps/research-loop/tests -q
ruff check apps/research-loop/{loop.py,server.py,hypotheses.py,tests}
ruff format --check apps/research-loop/{loop.py,server.py,hypotheses.py,tests}
node --check apps/research-loop/web/app.js
node apps/research-loop/tests/test_web.mjs
```

Tests use ephemeral stub-model ports and pure Python; they do not require Metal
or touch existing model servers. Node checks cover state and authored markup,
including both real local streams when present. They do not validate browser
layout. Review at 1920×1080 and 1440×900 in a real browser before recording.
