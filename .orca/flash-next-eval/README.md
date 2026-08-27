# Flash-Next launch evidence harness

Reproducible evidence tooling for the final 0.13.1 wheel. This folder
does not download, publish, or start a model. Run S4 FINAL before this battery.

## Part A: correctness

`prompts.jsonl` contains 45 deterministic cases across English and Chinese,
checkable reasoning/math, strict JSON, forced and automatic tool use on both
OpenAI and Anthropic routes, code, an executable CLI todo project, 8K/32K
needle recall, multi-turn system behavior, and stop sequences.

Validate without contacting a server:

```bash
python .orca/flash-next-eval/run_eval.py \
  --model qwen3.8-flash-next-4bit \
  --dry-run
```

Run against the final candidate. `--allow-project-exec` is deliberately
required because the todo case writes the two schema-constrained files into a
fresh temporary directory and executes `test_todo.py` there with a 60-second
timeout and a temporary HOME/TMPDIR.

```bash
python .orca/flash-next-eval/run_eval.py \
  --base-url http://127.0.0.1:8464 \
  --model qwen3.8-flash-next-4bit \
  --tokenizer-path /path/to/immutable/q4/artifact \
  --allow-project-exec \
  --output /private/tmp/flash-next-final-correctness.jsonl
```

Every row records PASS/FAIL, route, latency, returned text/tool names, each
scorer verdict, and the exact error. Do not omit failures from the launch note.

## Part B: benchmark

The benchmark imports the existing `scripts/bench_service_prefill.py` prompt
builder and SSE measurement path. It enforces B=1, exact tokenizer-counted
128/2K/8K/32K prompts, 256 decode tokens, three cold-cache runs, and medians.
TTFT is first visible content/reasoning/tool output; token counts come from
server usage. RSS is the server process plus recursive children, sampled every
50 ms. Peak is the maximum; steady RSS is the median pre-request sample. Record
the serving engine's periodic `[Metal memory]` lines alongside the JSON: MLX
active memory, not RSS alone, is the unified-memory sizing measurement.

Dry validation (no server or GPU):

```bash
python .orca/flash-next-eval/benchmark.py \
  --label flash-next-q4 \
  --tokenizer-path /path/to/artifact \
  --server-pid 1 \
  --rapid-sha TRAIN_SHA \
  --dry-run
```

Quiet-window run:

```bash
python .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8464/v1 \
  --model qwen3.8-flash-next-4bit \
  --tokenizer-path /path/to/immutable/q4/artifact \
  --server-pid SERVER_PID \
  --label flash-next-q4 \
  --rapid-sha TRAIN_SHA \
  --artifact-revision dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --load-time-seconds LOAD_SECONDS \
  --output /private/tmp/flash-next-final-benchmark.json
```

Repeat the identical command against `qwen3.8-27b-4bit`, changing only model,
tokenizer path, server PID, label, artifact revision, and output. Start a fresh
server process for every model and never keep both resident. Run only after
Atlas declares a quiet GPU window; contended results are invalid.

The public result must state the machine, RAM, macOS, exact Rapid SHA,
dependency versions, artifact revision, quantization contract, RSS, and MLX
active memory. It must also state that 128 GB hardware was not physically
tested.
