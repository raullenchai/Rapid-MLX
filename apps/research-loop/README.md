# Research Loop

A local, project-by-project autoresearch workbench. Set a research goal, discuss
ideas with a local model, run fixed-budget experiments, and inspect the code and
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
   model, and queue your own text or an assistant message. The model sees the
   goal, metric, fixed budget, previous discussion, and a compact verdict ledger
   from up to five recent runs (60 entries). Nothing is sent until you send a message.
3. **Experiments** (`#/p/<slug>/experiments`): choose the rounds and **Start
   experiments** from Discuss. The baseline runs first; queued human ideas run
   before model proposals. The main chart shows experiment number versus metric,
   grey discarded points, labelled kept points, and the running best. It respects
   the configured direction and noise threshold. The smaller chart shows the
   selected round's measurements. Discussion stays reachable in the side rail.
4. **Results** (`#/p/<slug>/results`): select a run and ledger row to inspect its
   hypothesis, why, prediction, curve, final code, and diff against its kept
   parent. Download the best code, event ledger, and log. “What we learned” groups
   recorded verdicts only; it generates no claims or model-written conclusions.

One run may be active per project. The goal is fixed for that run; save changes
before starting. The model picker chooses the discussion model and the model for
the next run. Existing runs keep their model. Pause and Stop take effect between
loop steps. Stop clears Pause; an ongoing model request or experiment finishes
its current step first. Closing the browser or server does not kill the loop.
Untried inbox ideas return to the project queue when the loop exits normally,
stops, or raises a Python exception. A forcibly killed process cannot perform
that cleanup; its original inbox remains on disk.

## Files and reproducibility

```text
templates/<slug>/
  program.md  task.json  <editable>
projects/<slug>/                     # gitignored local research data
  project.json                      # title, template, created, model
  program.md  task.json  <editable>
  discussion.jsonl                  # role, content, ts, model
  ideas.jsonl                       # queued hypotheses, source: "you"
  runs/<run-id>/
    run.json                        # model, rounds, task snapshot, project
    events.jsonl  inbox.jsonl  queued_ideas.jsonl  loop.log
    round0_<editable>            # exact baseline code
    roundN_<editable>            # exact experiment code, including repairs
    roundN_seedS_curve.json         # when written by the experiment
    best_<editable>  history.json
```

A supported change updates the best; no-effect and refuted changes are discarded.
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
  remain present); optional `goal`, `env`, `metric_label`, `unit`, and `curve`.
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
| GET | `/api/projects/<p>` | Project, program, task, runs, discussion, queue, models |
| PUT | `/api/projects/<p>/program` | `{program}` |
| POST | `/api/projects/<p>/chat` | `{message, model}` → streamed completion; saves both turns |
| POST | `/api/projects/<p>/ideas` | `{hypothesis, change?}` → pending ideas |
| POST | `/api/projects/<p>/runs` | `{rounds, model}` → starts run |
| GET | `/api/projects/<p>/runs/<r>` | Run metadata, events, available artifact names |
| GET | `/api/projects/<p>/runs/<r>/events` | SSE, optional `replay=1&speed=N` |
| POST | `/api/projects/<p>/runs/<r>/<action>` | `idea`, `pause`, `resume`, `stop` |
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
PYTHONPYCACHEPREFIX=/private/tmp/rapid-mlx-grokdemo-v3-pycache \
  python3 -m pytest apps/research-loop/tests -q
ruff check apps/research-loop/server.py apps/research-loop/tests apps/research-loop/templates/sort-speed
ruff format --check apps/research-loop/server.py apps/research-loop/tests apps/research-loop/templates/sort-speed
node --check apps/research-loop/web/app.js
node apps/research-loop/tests/test_web.mjs
```

Tests use ephemeral stub-model ports and pure Python; they do not require Metal
or touch existing model servers. Node checks cover state and authored markup,
including both real local streams when present. They do not validate browser
layout. Review at 1920×1080 and 1440×900 in a real browser before recording.
