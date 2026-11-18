# Research Loop

A local workspace for **ideas and experiments**, steps 3 and 4 of the research
cycle. Discuss hypotheses with a local model, test changes on your Mac, and
bring measured results back into the discussion. Question, literature, and
writing stay in the tools you choose. Research Loop does not write papers.

## Run

Use a Python environment with Rapid-MLX and the task's requirements (MLX for
the included grokking task). No frontend build, CDN, or new dependency is needed.

```sh
rapid-mlx serve <model> --port 8000
python apps/research-loop/server.py --base-url http://127.0.0.1:8000/v1 --model <model>
```

Open http://127.0.0.1:4900 and choose **Start research**. Server flags also include
`--host` (default `127.0.0.1`), `--port` (4900), and `--rounds` (10). The loop
and training subprocesses use the server's Python interpreter. Keep the server
bound to localhost: this is a local development tool without authentication.
Use a local model endpoint to keep ideas and data on this Mac.

The baseline runs first, followed by one hypothesis per round. **Discuss** streams
a model reply with task instructions and observed hypotheses/results as context.
**Test this idea** queues your text or a model reply for the next round. The
queue does not interrupt the experiment already in progress. Select a hypothesis
card or map node to inspect its code, curves, errors, fixes, and verdict.

Pause and Stop take effect between loop steps; Stop clears Pause so a paused
loop can exit. Running training/model requests finish their current step first.
Closing the browser or web server does not terminate the loop subprocess.

## Runs and video replay

Runs live in `apps/research-loop/runs/<id>/` (ignored by Git). Each has
`run.json`, `events.jsonl`, `inbox.jsonl`, `loop.log`, generated code, and curves.
The run selector also discovers existing run folders. Legacy runs without
`run.json` are assumed to use the grokking task and server model.

```text
http://127.0.0.1:4900/?run=<id>&replay=1&speed=8
```

Replay uses the original event timestamps divided by speed. Curve gaps are
capped at 150 ms to compress training; other event gaps retain their timing.
A replay badge stays visible and mutation controls are disabled. For recording,
use a 1920×1080 window. The SSE endpoint replays from the beginning and supports
`Last-Event-ID` byte offsets on reconnection, including partial JSONL writes.
If a loop exits without `done`, the page reports the closure; inspect `loop.log`.
Run folders remain inspectable after a server restart, but orphaned subprocesses
are not adopted for live controls (their event files can still be tailed). Restart the web server between runs.

## Add a task

Create `tasks/<name>/` containing:

- `program.md`: the goal, constraints, and what constitutes a testable idea.
- `task.json`: `title`, `editable` (filename), `metric` (RESULT key),
  `direction` (`max` or `min`), `noise`, `seeds`, `fixed` (regular expressions
  that must remain present), optional `goal`, and optional `env`.
- The editable Python file. Read `SEED` from the environment. Print JSON curve
  lines with `step`, `train_acc`, `test_acc`, and `loss` while training. Finish
  with `RESULT {"<metric>": <number>, ...}`. See `tasks/grokking/train.py`.

Only run tasks you trust: the loop executes model-edited Python with the same
permissions as your account. Task rules are experimental constraints, not a sandbox.
The chart displays train/test accuracy; verdict values use the task metric.
Restart the page to discover the new task. The loop itself can also run directly;
see its module docstring. The web server does not change its research behavior.

## Check

```sh
python -m pytest apps/research-loop/tests -q
ruff check apps/research-loop/server.py apps/research-loop/tests
ruff format --check apps/research-loop/server.py apps/research-loop/tests
node --check apps/research-loop/web/app.js
node apps/research-loop/tests/test_web.mjs
```
