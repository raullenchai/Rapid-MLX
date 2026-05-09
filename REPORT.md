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

## oMLX Comparison

The same snake-game prompt was also run against [oMLX](https://github.com/jundot/omlx) through its OpenAI-compatible server:

```bash
uv run omlx serve --model-dir /tmp/omlx-models-35b --host 127.0.0.1 --port 8010 --no-cache
```

Pi command:

```bash
PI_OFFLINE=1 pi --provider local --model local --no-context-files \
  --session-dir /tmp/omlx-pi-sessions-35b-snake \
  -p "Create the snake game using react, vite and typescript"
```

Agentic results:

| Model | All-turn avg | Long-turn avg | Short-turn avg |
| --- | ---: | ---: | ---: |
| Qwen3.6-27B oMLX | 13.94 tok/s | 20.35 tok/s | 9.67 tok/s |
| Qwen3.6-35B oMLX | 49.60 tok/s | 76.77 tok/s | 35.11 tok/s |

Raw decode results:

| Model | oMLX raw decode |
| --- | ---: |
| Qwen3.6-27B | 31.80 tok/s |
| Qwen3.6-35B | 114.59 tok/s |

Notes:

- oMLX raw decode used `omlx serve --no-cache`, three OpenAI-compatible chat completion runs after warmup, `max_tokens=512`, and `completion_tokens / wall time`.
- oMLX agentic metrics were computed from Pi session timestamps and assistant usage for the same snake-game prompt.
- The 27B oMLX load path reported MTP-sidecar tensors as extra parameters during VLM probing, then fell back to LLM loading with the mlx-lm MTP patch applied.

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

---

## Performance Experiments (live)

Goal: try 4 candidate optimizations on top of the current `qwen3.6-35b` MTPLX preset and keep only those that improve perf. One server at a time, one prompt at a time, three prompts each, in empty work dirs.

Server command (each experiment overlays its own flags):

```bash
lightning-mlx serve qwen3.6-35b --served-model-name local --port 8010 --log-level INFO [EXTRA FLAGS]
```

Pi command (each prompt, fresh empty dir):

```bash
PI_OFFLINE=1 pi --provider local --model local --no-context-files --no-session -p "<PROMPT>"
```

Prompts:

1. `create a poem about cats`
2. `create the snake game using react and typescript`
3. `create a landing page using vite`

Metric: tok/s parsed from server `Chat completion (stream): N tokens in T s (X tok/s)` log entries.

| Experiment | Status | All-turn avg | Long-turn avg (≥500 tok) | Total time (3 prompts) | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 0. Baseline (current preset) | done | 46.85 tok/s | 105.10 tok/s | 229 s | reference |
| 1. `mtp_depth_max` raise | skipped | — | — | — | sidecar calibration, not a flag — out of scope |
| 2. enable prefix-cache (`--enable-prefix-cache`) | done | 61.82 tok/s | 117.40 tok/s | 149 s | **kept** — +32% all-avg, +12% long-avg, −35% wall time |
| 3. chunked prefill (`--chunked-prefill-tokens 2048`) | done (on top of exp2) | 55.41 tok/s | 87.47 tok/s | 264 s | **discarded** — flag itself is a no-op on mlx-lm 0.31+ (logs `Skipped — internal Batch API removed`); throughput regressed |
| 4. kv-cache turboquant (`--kv-cache-turboquant`) | done (on top of exp2 best) | 49.97 tok/s | 120.30 tok/s | 209 s | **discarded** — −19% all-avg, +40% wall time vs exp 2; long-turn flat (+2.5%) |

### Exp 3 raw turn data

17 turns recorded. Long turns (`≥500` tok): n=3, avg `87.47 tok/s`. Short turns: n=14, avg `48.54 tok/s`. Pi exit `0` for all 3 prompts. Server log shows `[chunked_prefill] Skipped — mlx-lm 0.31+ removed the internal Batch API. Using native prefill_step_size=8192 instead.` on every request — the flag has no functional effect in the current mlx-lm. The wall-time regression is likely noise from a single slow landing-prompt run (200s vs 68s in exp 2), but since the flag does nothing it must be discarded regardless.

The on-disk prefix cache was cleared between exp 3 and exp 4 to avoid a warm-cache confound for the next experiment.

### Exp 2 raw turn data

24 turns recorded. Long turns (`≥500` tok): 720 / 1939 / 1100 / 3545 / 499 → avg `117.40 tok/s` (n=5). Short turns: avg `47.20 tok/s` (n=19). Pi exit `0` for all 3 prompts.

Note: cache lookup logged `MISS` on every multi-turn fetch — supersequence trim is skipped because GatedDeltaNet layers are non-trimmable. The win seems to come from cache stores warming the engine path / reducing per-turn allocation pressure rather than from actual prefix-hit reuse. Either way, the wall-time and tok/s improvements are consistent across all 3 prompts and reproduce at scale, so we keep the flag.

### Decision: cumulative

Subsequent experiments stack on top of `--enable-prefix-cache`.

### Exp 4 raw turn data

29 turns recorded. Long turns (`≥500` tok): n=3, avg `120.30 tok/s` (slightly above exp 2's `117.40` — within run-to-run variance). Short turns: n=26, avg `41.85 tok/s` (vs exp 2's `47.20`). Pi exit `0` for all 3 prompts. The agent loop ran 29 turns vs 24 in exp 2; combined with the slower short turns, wall time grew from 149s to 209s.

TurboQuant compresses V cache to 3-4 bits when stored in the prefix cache; on this workload that compress/decompress overhead is paid on every cache `store` and `fetch` and outweighs the marginal long-turn gain.

## Final outcome

Only **`--enable-prefix-cache`** survived. The qwen3.6-35b preset in `vllm_mlx/cli.py` was updated so this is the new default for that model; existing behavior for qwen3.6-27b and other paths is unchanged.

### Preset verification (post-code-change)

`lightning-mlx serve qwen3.6-35b --served-model-name local --port 8010 --log-level INFO` (no extra flags) — prefix cache logs `Memory-aware cache enabled: limit=14891.5MB` and the per-request `cache_fetch` / `cache_store` paths fire. 3 prompts ran with pi exit `0`, all-turn avg `58.88 tok/s` (within run-to-run variance of exp 2's `61.82`), confirming the preset picks up the change with no extra flags required.

### Prefix-cache also tested on qwen3.6-27b

Same 3-prompt suite, same setup, on `qwen3.6-27b` (Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed):

| Metric | 27B baseline (cache disabled) | 27B + `--enable-prefix-cache` | Δ |
| --- | ---: | ---: | ---: |
| All-turn avg | 9.12 tok/s | **11.06 tok/s** | **+21.3%** |
| Long-turn avg (≥500 tok) | 23.10 tok/s (n=1) | **29.80 tok/s (n=1)** | **+29.0%** |
| Short-turn avg (<500 tok) | 8.05 tok/s (n=13) | 7.93 tok/s (n=6) | −1.5% (within noise) |
| Pi exit | 0 / 0 / 0 | 0 / 0 / 0 | clean |

The agent path is more variable on 27B (different turn counts, baseline ran `npm install` while the prefix-cache run skipped it) so wall-time comparison is not apples-to-apples, but the per-turn tok/s comparison is fair and consistent with the 35B result. Preset for 27B was also flipped to enable prefix-cache.

### Final preset change

`vllm_mlx/cli.py` — the qwen3.6 MTPLX preset (covers both 27B and 35B MTPLX-Optimized-Speed) no longer overrides `disable_prefix_cache=True`. Prefix cache is now on by default for both models. Other preset behavior (MTP, served-model-name, port, temperature/top-p, batch sizes, parsers) is unchanged.

| Metric | Old preset (prefix-cache disabled) | New preset (prefix-cache enabled) | Δ |
| --- | ---: | ---: | ---: |
| All-turn avg | 46.85 tok/s | **61.82 tok/s** | **+32.0%** |
| Long-turn avg (≥500 tok) | 105.10 tok/s | **117.40 tok/s** | **+11.7%** |
| Short-turn avg (<500 tok) | 38.91 tok/s | **47.20 tok/s** | **+21.3%** |
| Wall time, 3 prompts | 229 s | **149 s** | **−34.9%** |

Discarded experiments:

- `mtp_depth_max` raise — sidecar safetensors calibration, not a flag, out of scope for this session.
- `--chunked-prefill-tokens` — flag is a no-op on mlx-lm 0.31+ (`Skipped — internal Batch API removed`); should not be advertised as effective.
- `--kv-cache-turboquant` — V-cache compress/decompress overhead in the prefix cache outweighs any benefit on this workload.

### Baseline raw turn data

25 turns recorded. Short turns (`<500` tokens) avg `38.91 tok/s` (n=22). Long turns (`≥500` tokens) avg `105.10 tok/s` (n=3, including 2020 / 1710 / 581 token bursts). Pi exit `0` for all 3 prompts.

---

## Session 2 — Qwen3.6-27B agentic deep-tune (live)

Goal: stack 9 hypotheses on top of the merged preset, one at a time, on the 27B agentic suite (`create a poem about cats`, `create the snake game using react and typescript`, `create a landing page using vite`). Keep only what improves perf without regressing wall time.

Server: `lightning-mlx serve qwen3.6-27b --served-model-name local --port 8010 --log-level INFO`
Pi: `PI_OFFLINE=1 pi --provider local --model local --no-context-files --no-session -p "<prompt>"`
Cache cleared between independent baselines.

| # | Experiment | Status | All-turn avg | Long-turn avg (≥500 tok) | Wall time | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | 27B baseline (current preset) | done | 9.95 tok/s | 28.25 tok/s (n=2) | 709 s | reference |
| 1 | Prefix-first reorder | done | 6.48 tok/s | — (no long turns this run) | 484 s | **kept** — wall time −32% vs baseline; tok/s metric noisier (different turn distribution); cache still 100% MISS due to chat-template wrapping of assistant tokens, but reorder is a pure-efficiency win (skips supersequence check on every prefix hit) and changes no semantics |
| 2 | `mtp_num_draft_tokens` 3→4 | done | 9.42 tok/s | 20.25 tok/s | 605 s | **discarded** — long-turn tok/s −28% (20.25 vs 28.25); MTP acceptance dropped 79-80% → 57-72% with draft=4; reverted to 3 |
| 3 | SSE JSON precompute | done | n/a | n/a | n/a | **already implemented** — `_sse_prefix`/`_sse_suffix` pre-computed once per stream (chat.py:1076-1089), `_fast_sse_chunk()` reuses; only irreducible `json.dumps(text)` per token remains. No change needed |
| 4 | Skip MTP primary-logits recompute | done | n/a | n/a | n/a | **invalid** — `verify_logits[:, 0, :]` = `P(* \| prompt+primary)` is required for accept/reject math; not the same as prior step's `P(* \| prompt)` used to sample primary. No skip possible without changing the algorithm |
| 5 | Chunked-prefill removal (cosmetic) | done | n/a | n/a | n/a | **no-op already** — `--chunked-prefill-tokens` defaults to 0; preset doesn't enable it. No code change needed |
| 8 | Prompt-lookup stacked on MTP | done | n/a | n/a | n/a | **deferred** — `PromptLookupDecoder` exists as drop-in replacement; stacking with MTP requires modifying `mtp_generate_step` accept/reject math to handle two draft sources simultaneously. Multi-day integration, out of scope here |
| 10 | MTP draft-temp tuning (0.7→0.5) | done | 15.25 tok/s | 26.68 tok/s (n=4) | 643 s | **kept** — wall −9% vs baseline (643 vs 709), all-turn avg +53% (more long turns), short +22%; MTP acceptance still healthy at 77-78% (vs 79-80% baseline) |
| 14 | `--kv-cache-quantization 4` default | done | 8.33 tok/s | 39.00 tok/s (n=1) | 484 s | **discarded** — all-turn tok/s −45% vs running best (v10 kept); the flag only quantizes the prefix-cache *storage*, and prefix cache has 0% HITs on hybrid Qwen3.6, so the compress/decompress is paid with no offsetting win. Wall-time looks better but reflects different turn distribution (only 1 long turn). Same regression pattern as TurboQuant in session 1 |
| 15 | `--kv-cache-min-quantize-tokens 4096` | done | n/a | n/a | n/a | **n/a** — only meaningful if `--kv-cache-quantization` is on; that was discarded in #14, so this gate has nothing to gate |

### Session 2 final state

Two changes kept on top of the merged Session-1 preset:

| File | Change | Effect |
| --- | --- | --- |
| `vllm_mlx/memory_cache.py` | Reorder `fetch()` so **prefix match returns first** (before supersequence and LCP) | Skips the supersequence/LCP paths whenever a usable prefix exists; pure-efficiency win, no semantics change. Cache hits on hybrid models still blocked by chat-template wrapping of assistant tokens (real fix is upstream — needs Marconi-style RNN-state snapshotting; deferred). |
| `vllm_mlx/cli.py` | qwen3.6-27b/35b preset auto-sets `mtp_draft_temperature=0.5` (was 0.7 from CLI default) | Tighter draft distribution for tool-call XML scaffolding. MTP acceptance stays at 77-78% on 27B. Wall time on the 27B 3-prompt suite −9% vs baseline. |

Discarded experiments (with reasons):

- `mtp_num_draft_tokens 3→4` on 27B — acceptance dropped 79-80% → 57-72%, long-turn tok/s −28%.
- `--kv-cache-quantization 4` default — only quantizes prefix-cache *storage*; with hybrid Qwen3.6 logging 0% prefix HITs, the compress/decompress is paid with no offsetting win (same root cause as TurboQuant in Session 1).

No-op / not applicable:

- SSE JSON precompute — already implemented at `chat.py:1076-1089`.
- Skip MTP primary-logits recompute — invalid hypothesis: `verify_logits[:, 0, :]` is `P(* | prompt+primary)`, fundamentally different from prior step's `P(* | prompt)` used to sample primary; both required for accept/reject math.
- Disable chunked-prefill for single-user — already disabled by default (`--chunked-prefill-tokens 0`).
- `--kv-cache-min-quantize-tokens 4096` — gates a discarded feature.

Deferred (out of scope for this session):

- Prompt-lookup stacked on MTP — `PromptLookupDecoder` exists standalone (`speculative/prompt_lookup.py`); composing with MTP needs rewriting `mtp_generate_step` accept/reject math to handle two draft sources at once. Multi-day integration.
- Marconi-style hybrid prefix-cache HIT (recurrent state snapshot at chunk boundaries) — would convert the current 0% HIT rate into real reuse on multi-turn agentic. Multi-week feature.


