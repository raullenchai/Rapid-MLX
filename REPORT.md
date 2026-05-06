# Agentic MTP Speed Report

Branch: `optimize-agentic-mtp-speed`

Prompt fixture:

```text
Create the snake game using react, vite and typescript
```

Rules followed:
- Run `pi` in an empty directory.
- Run only one server at a time.
- Keep only changes that improve measured agentic performance.
- Revert changes that do not improve measured performance.
- Use generic runtime improvements only. No prompt-specific heuristics.

Base server:

```bash
.venv/bin/rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --enable-mtp \
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
  --no-thinking \
  --enable-tool-logits-bias
```

## Baseline

Directory:

```text
/tmp/rapid-agentic-snake-baseline-1778038946
```

Result:

- `pi`: `TIMEOUT`, exit code `124` after 10 minutes.
- Artifact: substantial Vite/React/TypeScript snake game created.
- Non-`node_modules` files: `28`.
- Tool-call SSE events: `23`.
- Turn count: `24`.
- All-turn average: `13.49 tok/s`.
- Long-turn average (`>=500` generated tokens): `28.02 tok/s` (`n=6`).
- Short-turn average (`<150` generated tokens): `7.42 tok/s` (`n=16`).
- MTP acceptance: avg `92.02%`, min `78.1%`, max `94.7%` (`n=18`).

Observation:

- Raw long-turn decode was acceptable, but repeated short long-context tool turns collapsed to `3-7 tok/s` near the end.

## Candidate 6: Prefix Cache Flag

Directory:

```text
/tmp/rapid-agentic-snake-prefixcache-1778039607
```

Change tested:

- Used `--enable-prefix-cache`.
- No code change.

Result:

- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Non-`node_modules` files: `25`.
- Tool-call SSE events: `12`.
- Turn count: `13`.
- All-turn average: `16.95 tok/s`.
- Long-turn average: `37.70 tok/s`.
- Short-turn average: `10.79 tok/s`.
- MTP acceptance: avg `91.59%`, min `84.4%`, max `94.0%`.
- Prefix cache evidence: `0` hits, `14` misses.

Decision:

- Not kept as code/default. Run improved, but no real prefix-cache hits happened, so safe agentic prefix caching was not proven.

## Candidate 4: Earlier Tool-Call Streaming From Reasoning

Directory:

```text
/tmp/rapid-agentic-snake-earlyreasoning-1778040034
```

Change tested:

- Routed complete tool markup found in reasoning channel through incremental tool detector immediately.
- Left incomplete `[Calling tool: name]` on keepalive path.

Validation:

- Initial focused tests passed: `53 passed`.

Benchmark result:

- `pi`: `DONE`, exit code `0`.
- Build: failed with TypeScript type-only import errors.
- Turn count: `19`.
- All-turn average: `16.43 tok/s`.
- Long-turn average: `25.04 tok/s`.
- Short-turn average: `10.53 tok/s`.
- MTP acceptance: avg `91.51%`, min `84.0%`, max `93.4%`.
- `SSE-TC`: `0`.
- `SSE-FALLBACK-TC`: `18`.

Decision:

- Reverted. It did not reduce fallback tool-call usage, worsened long-turn throughput, and artifact build failed.

## Candidate 5: Fast Tool-Call SSE Serialization

Directory:

```text
/tmp/rapid-agentic-snake-fasttoolsse-1778040656
```

Change kept:

- Added direct JSON serializer for streaming tool-call chunks in `vllm_mlx/routes/chat.py`.
- Replaced Pydantic `ChatCompletionChunk(...).model_dump_json(...)` on `SSE-TC` and `SSE-FALLBACK-TC` paths.
- Generic OpenAI-compatible output serialization. No prompt-specific behavior.

Validation:

- `tests/test_postprocessor.py`
- `tests/test_tool_calling.py`
- `tests/test_api_models.py`
- Result: `211 passed`.

Benchmark result:

- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Turn count: `9`.
- All-turn average: `15.34 tok/s`.
- Long-turn average: `33.55 tok/s`.
- Short-turn average: `10.30 tok/s`.
- MTP acceptance: avg `93.84%`, min `87.9%`, max `95.3%`.
- `SSE-TC`: `0`.
- `SSE-FALLBACK-TC`: `8`.

Decision:

- Kept. It improved end-to-end completion, long-turn speed, and short-turn speed vs baseline.

## Candidate 1: Existing `--mtp-num-draft-tokens 2`

Directory:

```text
/tmp/rapid-agentic-snake-draft2-1778040932
```

Change tested:

- Added `--mtp-num-draft-tokens 2`.
- No code change.

Result:

- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Turn count: `5`.
- All-turn average: `20.70 tok/s`.
- Short-turn average: `15.67 tok/s` (`n=3`).
- No turns reached `500` generated tokens, so long-turn evidence is weak.
- MTP acceptance: `66.8%`.

Decision:

- Not kept as default. It completed fast in this run, but acceptance collapsed and long-turn evidence was missing. Needs stronger adaptive/multi-draft implementation before enabling.

## Candidate 5b: Cheaper SSE Logging

Directory:

```text
/tmp/rapid-agentic-snake-fastlog-1778041075
```

Change kept:

- Downgraded per-chunk SSE role/tool-call logs from `INFO` to `DEBUG`.
- Kept `Chat completion (stream)` throughput logs at `INFO`.
- Reduces hot-path logging/I/O during agentic tool loops.

Validation:

- `tests/test_postprocessor.py`
- `tests/test_tool_calling.py`
- `tests/test_api_models.py`
- Result: `211 passed`.

Benchmark result:

- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Turn count: `3`.
- All-turn average: `26.47 tok/s`.
- Long-turn average: `38.60 tok/s`.
- Short-turn average: `20.40 tok/s`.
- MTP acceptance: avg `94.30%`, min `91.0%`, max `96.1%`.
- `SSE` info log count: `0`.

Decision:

- Kept. This was the best measured run and kept artifact build passing.

## Candidate 10: Disable Tool Logits Bias

Directory:

```text
/tmp/rapid-agentic-snake-notoolbias-1778041220
```

Change tested:

- Omitted `--enable-tool-logits-bias`.
- No code change.

Result:

- `pi`: `DONE`, exit code `0`.
- Build: failed with TypeScript errors.
- Turn count: `18`.
- All-turn average: `17.23 tok/s`.
- Long-turn average: `31.40 tok/s`.
- Short-turn average: `10.17 tok/s`.
- MTP acceptance: avg `88.18%`, min `84.0%`, max `90.7%`.

Decision:

- Not kept. It did not improve the kept fast-log run and generated artifact failed build.

## Candidate 9: Tool-Call Warmup

Change under test:

- Added startup warmup for chat+tools path when auto tool choice and a tool parser are enabled.
- Warmup uses a generic no-op tool schema and `max_tokens=8`.
- Goal: compile tool-call/chat/MTP kernels before `/health` returns so the first agentic turns pay less startup cost.

Validation before benchmark:

- `tests/test_config_and_middleware.py`
- `tests/test_server_utils.py`
- `tests/test_postprocessor.py`
- `tests/test_api_models.py`
- Result: `204 passed`.

Benchmark result:

- Directory: `/tmp/rapid-agentic-snake-toolwarmup-1778041762`.
- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Warmup evidence: server logged `Tool-call warmup: compiling chat+tools path` and `Warmup complete (0.9s)`.
- Turn count: `27`.
- All-turn average: `12.33 tok/s`.
- Long-turn average: `31.95 tok/s`.
- Short-turn average: `7.68 tok/s`.
- MTP acceptance: avg `90.87%`, min `84.8%`, max `93.1%`.

Decision:

- Reverted. It was slower than the kept fast-log run across all-turn, long-turn, and short-turn averages, and it increased turn count from `3` to `27`.

## Candidate 7: Lower Draft Temperature

Change tested:

- Used `--mtp-draft-temperature 0.5`.
- No code change.
- Goal: reduce speculative rejection and rollback cost with a generic less-aggressive draft policy.

Benchmark result:

- Directory: `/tmp/rapid-agentic-snake-drafttemp05-1778042580`.
- `pi`: `DONE`, exit code `0`.
- Build: passed from generated `snake-game` subdirectory.
- Turn count: `23`.
- All-turn average: `13.49 tok/s`.
- Long-turn average: `33.47 tok/s`.
- Short-turn average: `9.18 tok/s`.
- MTP acceptance: avg `91.09%`, min `88.7%`, max `94.0%`.

Decision:

- Not kept. It was slower than the kept fast-log run across all-turn, long-turn, and short-turn averages, despite completing and building successfully.

## Candidate 8: Adaptive Draft Depth

Change under test:

- Added generic adaptive MTP depth when `--mtp-num-draft-tokens > 1`.
- The policy started at depth `1`, raised depth only after high live acceptance, and lowered it after weak live acceptance.
- No prompt-specific logic.

Validation before benchmark:

- `tests/test_batching.py`
- `tests/test_config_and_middleware.py`
- `tests/test_server_utils.py`
- Result: `73 passed`, `2 deselected`.

Benchmark result:

- Directory: `/tmp/rapid-agentic-snake-adaptive-draft2-1778043139`.
- `pi`: exit code `0`, but invalid fixture result.
- Build: not run.
- Generated files: `0`.
- Turn count: `2`.
- All-turn average: `16.25 tok/s`.
- Adaptive events: `0`.

Decision:

- Reverted. The benchmark did not produce the requested snake-game artifact, so it cannot be counted as an improvement.

## Candidate 3: Batch-Size-1 Logits Processor Fast Path

Change under test:

- Added a specialized `batch_size == 1` path in the GenerationBatch MTP logits-processor flow.
- Avoided per-row list building and `mx.concatenate` for primary and verify logits processing when agentic serving runs with `--max-num-seqs 1`.

Validation before benchmark:

- `tests/test_batching.py`
- `tests/test_tool_calling.py`
- `tests/test_config_and_middleware.py`
- Result: `94 passed`, `2 deselected`.

Benchmark result:

- Directory: `/tmp/rapid-agentic-snake-batch1-fastpath-1778043270`.
- `pi`: `DONE`, exit code `0`.
- Build: failed with TypeScript type-only import errors in the generated app.
- Turn count: `15`.
- All-turn average: `17.83 tok/s`.
- Long-turn average: `26.30 tok/s`.
- Short-turn average: `13.04 tok/s`.
- MTP acceptance: avg `89.77%`, min `82.0%`, max `91.9%`.

Decision:

- Reverted. It was slower than the kept fast-log run and the generated artifact failed build.

## Candidate 2: Defer Current Logprobs Sync

Change under test:

- Changed the GenerationBatch MTP step to `mx.async_eval(gen_self._current_logprobs)` and synchronously evaluate only `input_tokens` before `tolist()`.
- Goal: reduce CPU/GPU synchronization in the hot path without changing emitted tokens.

Validation before benchmark:

- `tests/test_batching.py`
- `tests/test_tool_calling.py`
- `tests/test_api_models.py`
- Result: `152 passed`, `2 deselected`.

Benchmark result:

- Directory: `/tmp/rapid-agentic-snake-sync-defer-logprobs-1778043653`.
- `pi`: `DONE`, exit code `0`.
- Build: passed.
- Turn count: `16`.
- All-turn average: `13.96 tok/s`.
- Long-turn average: `31.07 tok/s`.
- Short-turn average: `9.93 tok/s`.
- MTP acceptance: avg `91.04%`, min `83.6%`, max `93.1%`.

Decision:

- Reverted. It was slower than the kept fast-log run across all-turn, long-turn, and short-turn averages.

## Current Kept Changes

- `vllm_mlx/routes/chat.py`: direct JSON serializer for tool-call SSE chunks.
- `vllm_mlx/routes/chat.py`: per-SSE role/tool-call logs downgraded to `DEBUG`.

## Current Best Measurement

- Directory: `/tmp/rapid-agentic-snake-fastlog-1778041075`.
- `pi`: `DONE`.
- Build: passed.
- All-turn average: `26.47 tok/s`.
- Long-turn average: `38.60 tok/s`.
- Short-turn average: `20.40 tok/s`.
- MTP acceptance average: `94.30%`.

## Requested Item Checklist

1. MTP multi-draft stronger:
   - Tested existing `--mtp-num-draft-tokens 2`.
   - Result: build passed and run was short, but acceptance fell to `66.8%` and no long-turn evidence existed.
   - Decision: not kept as default; needs a real adaptive/multi-draft redesign before enabling.

2. Reduce CPU/GPU sync in scheduler:
   - Existing branch base already includes deferred MTP `token_array` sync reduction from previous work.
   - Implemented/tested deferring `gen_self._current_logprobs` sync and synchronizing only `input_tokens` before `tolist()`.
   - Result: build passed, all-turn avg `13.96 tok/s`, long-turn avg `31.07 tok/s`, short-turn avg `9.93 tok/s`.
   - Decision: reverted.

3. Fast path for `batch_size=1`:
   - Implemented/tested specialized logits-processor path for `batch_size == 1`.
   - Result: build failed, all-turn avg `17.83 tok/s`, long-turn avg `26.30 tok/s`, short-turn avg `13.04 tok/s`.
   - Decision: reverted.

4. Tool-call streaming earlier:
   - Implemented/tested candidate for reasoning-channel tool markup.
   - Result: did not reduce fallback tool-call usage and build failed.
   - Decision: reverted.

5. Cheaper token counting/logging/serialization:
   - Implemented and kept fast tool-call SSE serialization.
   - Implemented and kept DEBUG-level per-SSE logs.
   - Result: best run improved from baseline timeout to `DONE`, build passed, all-turn avg `26.47 tok/s`, short-turn avg `20.40 tok/s`.

6. Prefix cache safe for agentic:
   - Tested `--enable-prefix-cache`.
   - Result: run improved, but observed `0` prefix-cache hits and `14` misses.
   - Decision: not kept as code/default until real repeated-turn hits are proven.

7. Speculative acceptance without expensive rollback:
   - Tested lower draft temperature with `--mtp-draft-temperature 0.5`.
   - Result: build passed, all-turn avg `13.49 tok/s`, long-turn avg `33.47 tok/s`, short-turn avg `9.18 tok/s`, acceptance avg `91.09%`.
   - Decision: not kept; slower than kept fast-log run.

8. Adaptive MTP:
   - Implemented/tested generic adaptive draft depth for `--mtp-num-draft-tokens > 1`.
   - Result: invalid benchmark; `pi` exited `0` but generated `0` files and build could not run.
   - Decision: reverted; no artifact/performance proof.

9. Better compile/warmup:
   - Implemented/tested tool-call warmup with generic no-op tool schema.
   - Result: build passed, all-turn avg `12.33 tok/s`, long-turn avg `31.95 tok/s`, short-turn avg `7.68 tok/s`.
   - Decision: reverted; slower than kept fast-log run and increased turn count to `27`.

10. Less overhead in tool logits bias:
   - Tested by omitting `--enable-tool-logits-bias`.
   - Result: build failed, all-turn avg `17.23 tok/s`, long-turn avg `31.40 tok/s`, worse than kept fast-log run.
   - Decision: not kept.
