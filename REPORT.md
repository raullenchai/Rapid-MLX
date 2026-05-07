# Qwen3.6 35B-A3B MTPLX Benchmark Report

## Benchmark Highlight

Same prompt, same agentic workflow, one server at a time:

```text
Create the snake game using react, vite and typescript
```

| Metric | 35B base | **35B MTPLX** | **Gain** |
| --- | ---: | ---: | ---: |
| Completion | Timeout after 10 min | **Finished** | **Completed** |
| Generated app build | Failed | **Failed** | **No valid artifact** |
| All-turn avg | 27.73 tok/s | **64.85 tok/s** | **+133.9% / 2.34x** |
| Long-turn avg | 26.52 tok/s | **75.13 tok/s** | **+183.3% / 2.83x** |
| Short-turn avg | 29.18 tok/s | **52.50 tok/s** | **+79.9% / 1.80x** |
| MTP acceptance avg | N/A | **96.62%** | **MTP enabled** |

**Biggest win:** long tool-heavy turns went from **26.52 tok/s** to **75.13 tok/s**. The MTPLX run finished the agentic workflow inside the 10 minute limit; the base run did not.

## Setup

Hardware and runtime:

- Apple Silicon local MLX runtime.
- Repository branch: `convert-mtplx`.
- Server command was run once per model, with one server active at a time.
- Prefix cache was disabled for both runs.
- Tool parser: `qwen3_coder_xml`.
- Reasoning parser: `qwen3`.
- Default temperature: `0.6`.
- Default top-p: `0.95`.

Models:

- 35B base:
  - `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`
- 35B MTPLX:
  - `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed`

Agent command:

```bash
PI_OFFLINE=1 pi --provider local --model local --no-context-files --no-session \
  -p "Create the snake game using react, vite and typescript"
```

## Base Run

Command:

```bash
uv run lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0.6 \
  --default-top-p 0.95 \
  --disable-prefix-cache \
  --max-num-seqs 1 \
  --prefill-batch-size 1 \
  --completion-batch-size 1 \
  --stream-interval 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --reasoning-parser qwen3 \
  --enable-tool-logits-bias \
  --log-level INFO
```

Outcome:

- Pi exit: `142`.
- The benchmark hit the 10 minute limit.
- The model generated a React/Vite/TypeScript project in `/tmp/lmlx-35b-base-snake`.
- Build failed:
  - `src/App.tsx(190,9): error TS1109: Expression expected.`
  - `src/App.tsx(190,10): error TS1109: Expression expected.`

Observed stream completions:

| Turn | Tokens | tok/s |
| ---: | ---: | ---: |
| 1 | 63 | 21.7 |
| 2 | 121 | 59.1 |
| 3 | 41 | 38.8 |
| 4 | 28 | 24.8 |
| 5 | 34 | 1.5 |
| 6 | 1536 | 35.2 |
| 7 | 1536 | 20.6 |
| 8 | 1536 | 25.5 |
| 9 | 1536 | 27.2 |
| 10 | 1536 | 26.2 |
| 11 | 1536 | 24.4 |

Base metrics:

- All-turn avg: `27.73 tok/s`.
- Long-turn avg: `26.52 tok/s`.
- Short-turn avg: `29.18 tok/s`.
- Completion: timeout after 10 minutes.
- Generated app build: failed.

## MTPLX Run

Command:

```bash
uv run lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed \
  --log-level INFO
```

The MTPLX local-path preset automatically applied:

- MTP enabled.
- Served OpenAI model name `local`.
- Port `8010`.
- Default temperature `0.6`.
- Default top-p `0.95`.
- Prefix cache disabled.
- Single-sequence agent mode.
- `qwen3_coder_xml` tool parser.
- `qwen3` reasoning parser.
- Tool logits bias.
- Thinking enabled for Qwen3.6 35B-A3B.

Outcome:

- Pi exit: `0`.
- The agentic workflow finished.
- The model generated a React/Vite/TypeScript project in `/tmp/lmlx-35b-mtplx-snake`.
- Build failed:
  - `src/components/SnakeBoard.tsx(1,10): error TS1484: 'Position' is a type and must be imported using a type-only import when 'verbatimModuleSyntax' is enabled.`
  - `src/hooks/useSnakeGame.ts(2,48): error TS2307: Cannot find module './types' or its corresponding type declarations.`
  - `src/hooks/useSnakeGame.ts(62,14): error TS7006: Parameter 'prev' implicitly has an 'any' type.`
  - `src/hooks/useSnakeGame.ts(72,14): error TS7006: Parameter 'prev' implicitly has an 'any' type.`
  - `src/hooks/useSnakeGame.ts(106,28): error TS7006: Parameter 's' implicitly has an 'any' type.`

Observed stream completions:

| Turn | Tokens | tok/s |
| ---: | ---: | ---: |
| 1 | 64 | 31.7 |
| 2 | 358 | 79.1 |
| 3 | 61 | 27.6 |
| 4 | 70 | 54.2 |
| 5 | 1536 | 106.4 |
| 6 | 1536 | 56.0 |
| 7 | 1488 | 124.9 |
| 8 | 945 | 113.6 |
| 9 | 250 | 69.9 |
| 10 | 1536 | 37.6 |
| 11 | 1536 | 12.3 |

MTPLX metrics:

- All-turn avg: `64.85 tok/s`.
- Long-turn avg: `75.13 tok/s`.
- Short-turn avg: `52.50 tok/s`.
- MTP acceptance avg: `96.62%`.
- Completion: finished.
- Generated app build: failed.

## Interpretation

MTPLX fixed the core performance problem for Qwen3.6 35B-A3B. The 35B base run timed out after 10 minutes, while the 35B MTPLX run completed. The throughput gain was strongest on long turns, where MTPLX reached **2.83x** the base throughput.

The benchmark did not produce a valid build artifact. That is a model or agent-output quality issue for this single snake workflow, not an MTP acceptance issue: MTP acceptance averaged **96.62%** in the MTPLX run.

For serving Qwen3.6 35B-A3B, the measured default is:

- `default_temperature=0.6`
- `default_top_p=0.95`

This keeps MTP acceptance high while preserving good throughput.
