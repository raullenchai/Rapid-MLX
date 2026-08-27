# Qwen3.8-Flash-Next on M3 Ultra

These are post-release correctness and performance results for Rapid-MLX
0.13.1. Qwen3.8-Flash-Next is an experimental text-only model in this release;
Flash-Next MTP and vision are outside this measurement.

> **Hardware boundary:** these results were measured on the 256 GB machine
> described below. **128 GB hardware was not physically tested.** The catalog's
> 128 GB minimum is an admission floor, not a benchmark claim for a 128 GB Mac.
> With roughly 99 GB of quantized weights plus allocator and context-cache
> headroom, 192 GB is the practical recommended tier; 128 GB is tight and
> remains untested.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`) |
| Chip | Apple M3 Ultra, 28 CPU cores |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Architecture | arm64 |
| Python | 3.12.14 |
| Rapid-MLX | PyPI 0.13.1, source `819db66767ac7e16722315122600d0855a6981c8` |
| PyPI wheel SHA-256 | `5a729b5838de42ce24a7008d113bb8f0e41923fee728adc12b279f7d841eb247` |
| MLX | 0.32.2 |
| MLX-LM | 0.31.3 |
| Transformers | 5.12.1 |
| Flash-Next artifact | `rapid-mlx/Qwen3.8-Flash-Next-4bit` at `dcf657e4acda2aae72da99cde65b6c491cd96998` |
| Flash-Next quantization | PLE q4-g32; routing gates q8-g64; remainder q4-g64 |
| 27B reference artifact | `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` at `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456` |
| 27B quantization | Affine q4-g64 |

The publication tables come from a quiet window with no other model server
resident. A process sweep immediately before each fresh model load confirmed
that the new server was the only `rapid-mlx serve` or `vllm_mlx.server`
process. The first Flash run began while four pre-existing model servers were
still resident; those measurements are preserved separately in the contended
appendix and are not used for the headline results.

## Exact setup and commands

The published release was installed into a clean Python 3.12 environment:

```bash
python3.12 -m venv /private/tmp/rapid-flash-pypi-0131
/private/tmp/rapid-flash-pypi-0131/bin/pip install 'rapid-mlx==0.13.1'
/private/tmp/rapid-flash-pypi-0131/bin/pip check
```

For each model, the server was started as a fresh process on port 8464 and
allowed to reach ready state before evidence collection:

```bash
# Flash-Next
/private/tmp/rapid-flash-pypi-0131/bin/rapid-mlx serve \
  qwen3.8-flash-next-4bit --host 127.0.0.1 --port 8464

# 27B reference, after the Flash-Next server exited
/private/tmp/rapid-flash-pypi-0131/bin/rapid-mlx serve \
  qwen3.8-27b-4bit --host 127.0.0.1 --port 8464
```

Flash-Next reached ready state in 27 seconds; the 27B reference reached it in
6 seconds. Both auto-routed to the text lane. The 27B server reported
speculative decoding off, so the reference rows do not measure speculative
decoding despite the artifact name.

The correctness battery was run against Flash-Next:

```bash
/private/tmp/rapid-flash-pypi-0131/bin/python \
  .orca/flash-next-eval/run_eval.py \
  --base-url http://127.0.0.1:8464 \
  --model qwen3.8-flash-next-4bit \
  --tokenizer-path /path/to/immutable/flash-next-artifact \
  --allow-project-exec \
  --output /private/tmp/flash-next-final-correctness.jsonl
```

Each benchmark invocation used the same harness and changed only the model
identity, immutable tokenizer path, process ID, label, artifact revision, load
time, and output path:

```bash
/private/tmp/rapid-flash-pypi-0131/bin/python \
  .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8464/v1 \
  --model MODEL \
  --tokenizer-path IMMUTABLE_ARTIFACT_PATH \
  --server-pid SERVER_PID \
  --label LABEL \
  --rapid-sha 819db66767ac7e16722315122600d0855a6981c8 \
  --artifact-revision ARTIFACT_REVISION \
  --load-time-seconds LOAD_SECONDS \
  --output OUTPUT_JSON
```

## Correctness

The deterministic battery contains 45 cases spanning English and Chinese,
checkable math/reasoning, five JSON-schema responses, automatic and forced tool
calls on both API protocols, code generation, an executable CLI todo project,
8K/32K needle recall, multi-turn/system behavior, and stop sequences.

| Result stage | Cases | Notes |
| --- | ---: | --- |
| Raw initial harness pass | 30/45 | Direct scorer result at the initial bounded budgets |
| Adjudicated initial pass | 32/45 | Two false negatives described below |
| Reasoning budget recheck | 12/12 | Passed at 4,096 and again on the user-default OpenAI path |
| Final effective pass | 44/45 | The remaining difference is the Chinese value translation below |

All 12 thinking cases initially exhausted their small harness output budgets
and returned the incomplete-reasoning fallback instead of a final answer. They
all passed when rerun with `max_tokens=4096`. They also passed through the
user-default OpenAI path with `max_tokens` omitted, including the four cases
whose system directives had to be represented as OpenAI system messages. A
literal omitted-budget request is not valid on the Anthropic Messages surface,
where `max_tokens` is a required protocol field. This is a harness budget/path
correction, not a Rapid-MLX correctness defect.

The remaining result was schema-valid JSON but translated the requested
`北京` / `中国` values to `Beijing` / `China`. That is recorded as model
behavior; no product issue was filed.

Two raw failures were harness/operator artifacts rather than model failures:

- The palindrome response lowercased its input before returning the comparison,
  but the scorer incorrectly required `lower` to appear after `return`.
- The todo response was valid two-file JSON. The first local execution could not
  resolve `python` because the venv interpreter had been invoked by absolute
  path without placing its `bin` directory on `PATH`; retesting the exact
  generated files with the activated venv passed all six unittests.

Both long-context checks passed: `MANGO-4827` at 8K in 39.363 seconds and
`ZEPHYR-9135` at 32K in 188.118 seconds. All eight tool cases passed across the
OpenAI and Anthropic routes. Stop sequences, multi-turn/system behavior, both
protocol checks, and the other four structured-JSON cases also passed.

During schema requests, the server logged that guided generation could not
construct its tokenizer adapter and fell back to unconstrained generation.
Every response remained syntactically valid JSON in this battery, but that
fallback means the run does not establish token-level schema enforcement.

## Methodology

- Batch size is one.
- The harness targets 128, 2,048, 8,192, and 32,768 prompt tokens. The server
  reported 92, 2,012, 8,156, and 32,732 tokens after applying its chat template;
  both values are shown below.
- Each request asks for 256 decode tokens. Flash-Next's 128-target response hit
  EOS after 232 tokens; every other timed response produced 256 tokens.
- Each prompt length has three cold-prefix-cache runs; the cache is cleared
  before every timed request.
- TTFT is measured at the first visible SSE content, reasoning, or tool delta.
- Prompt and completion token counts come from server-reported usage.
- Prefill rate is prompt tokens divided by TTFT, so it includes request and
  first-token overhead.
- Decode rate is completion tokens divided by elapsed time after TTFT.
- RSS includes the server process and recursive children, sampled every 50 ms.
  It does not include all Metal allocations on Apple unified memory.
- MLX active memory is the maximum allocator-active footprint reported by the
  serving engine during the three runs at that length. It is the relevant
  unified-memory sizing figure. The Flash process also reported a 148.1 GB MLX
  allocator peak inherited from model loading; the steady timed requests used
  approximately 103--105 GB active memory.
- Tables report median TTFT/rates and maximum RSS and MLX active memory across
  the three runs.

## Results

### `qwen3.8-flash-next-4bit`

| Target (server-reported) prompt tokens | Median TTFT | Median prefill tok/s | Median decode tok/s | Peak RSS | MLX active memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.380 s | 241.8 | 25.73 | 54.46 GiB | 103.0 GB |
| 2,048 (2,012) | 3.274 s | 614.6 | 22.28 | 54.74 GiB | 103.1 GB |
| 8,192 (8,156) | 37.984 s | 214.7 | 20.64 | 54.62 GiB | 103.4 GB |
| 32,768 (32,732) | 186.201 s | 175.8 | 19.65 | 54.64 GiB | 104.7 GB |

### `qwen3.8-27b-4bit` reference

| Target (server-reported) prompt tokens | Median TTFT | Median prefill tok/s | Median decode tok/s | Peak RSS | MLX active memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.429 s | 214.6 | 40.29 | 12.84 GiB | 15.6 GB |
| 2,048 (2,012) | 5.904 s | 340.8 | 39.50 | 12.84 GiB | 16.0 GB |
| 8,192 (8,156) | 24.246 s | 336.4 | 37.38 | 12.85 GiB | 17.6 GB |
| 32,768 (32,732) | 107.244 s | 305.2 | 32.58 | 12.87 GiB | 24.1 GB |

## Interpretation

Flash-Next reached the first visible token sooner at the 128- and 2K-target
prompts. The 27B reference prefills faster at 8K and 32K and decodes faster at
every measured length. At the 32K target, the reference's median TTFT is 42%
lower and its median decode rate is 66% higher. Process RSS alone materially
understates the unified-memory requirement: Flash-Next used 104.7 GB of MLX
active memory at 32K despite a 54.64 GiB RSS measurement. These are serving
measurements, not a comparison of answer quality.

The Flash model's quantized weights are approximately 99 GB before context
cache and allocator headroom. It completed the full 32K-target grid without OOM
on the 256 GB machine. **192 GB is therefore the practical recommended memory
tier.** A 128 GB Mac would have tight headroom and was not physically tested;
these results do not establish its behavior or performance.

## Appendix: contended Flash-Next run

These rows are retained for transparency but are excluded from the results
above. Four unrelated model servers were resident during this first run.

| Target (server-reported) prompt tokens | Median TTFT | Median prefill tok/s | Median decode tok/s | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.381 s | 241.5 | 25.62 | 54.87 GiB |
| 2,048 (2,012) | 3.314 s | 607.2 | 22.33 | 54.88 GiB |
| 8,192 (8,156) | 38.259 s | 213.2 | 21.17 | 54.89 GiB |
| 32,768 (32,732) | 186.433 s | 175.6 | 19.73 | 54.90 GiB |
