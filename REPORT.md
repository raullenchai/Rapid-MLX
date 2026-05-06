# Qwen3.6-35B-A3B MTPLX Agentic Snake Report

Branch/workdir: `/Users/samuelfajreldines/dev/rapid-mlx`

Target prompt:

```text
create the snake game with html and typescript
```

Goal:

- Plain serve command must work for `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed`.
- Pi provider `local`, model `local` must create a functional HTML + TypeScript snake game.
- Runtime change must be generic, not prompt-specific.
- Tool-use must be efficient: no 2048-token loops, no reasoning/prolixity in stream.
- Final proof must include generated files plus build/check/browser smoke where applicable.

## Current Baseline From Live Runs

Observed before this pass:

- Plain `hi` was fixed earlier: no `reasoning_content` leaked and response was short.
- Pi snake prompt still failed.
- Runs timed out at 60-90s.
- Generated artifacts were incomplete:
  - `/tmp/rapid-mlx-pi-snake-html-ts-info`: only `index.html`, no script/build.
  - `/tmp/rapid-mlx-pi-snake-html-ts-bashhint`: `index.html` + `snake.ts`, but no package/build and HTML referenced raw `snake.ts`.
- Server logs showed first tool request could generate 480-1102 tokens and following tool turns could hit 2048 tokens.
- MTP acceptance for this workload was effectively 0 accepted / many rejected, so MTP added work without benefit in these tool-heavy turns.

## Working Hypothesis

Primary issue is not `reasoning_content` leakage anymore. The model spends too many tokens inside tool responses and often writes incomplete multi-file artifacts over several turns. Need runtime-level bounds and retry behavior for tool requests.

Next candidate changes:

- Cap tool request generation budget from the first request, not only after `last_role == tool`.
- Use lower temperature for tool requests when user did not explicitly set it.
- Add a short retry path when tool requests produce content/deferred plans instead of tool calls.
- Consider disabling MTP for tool-heavy 35B requests if acceptance remains 0 and latency worsens.

## Candidate A: Bound Tool-Turn Budgets

Change under test:

- Tool requests now cap generation at `1024` tokens from the first turn.
- Tool retry turns now cap generation at `512` tokens and retry at most `3` times.
- Tool requests default to temperature `0.2` unless the client explicitly sets temperature.

Why:

- Live Pi snake runs showed first request using `max_tokens=32768` and retry requests using `2048`, causing 60-90s stalls.
- This is generic for tool-use and not specific to snake.

Validation status:

- Unit tests passed with Candidate A.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-a`.
- `pi`: exit code `0`, but invalid artifact.
- Files: `index.html`, `style.css`, `snake.ts`, `pi.out`.
- Problem: `index.html` references raw `snake.ts`; browser will not execute TypeScript interfaces/enums directly.
- Problem: final `pi.out` was malformed/incomplete: `<think>Let</think>` then `[Calling tool=read]`.
- Server logs confirmed caps applied (`1024` first turn, `512` retry), but quality degraded and artifact was not usable.

Decision:

- Reverted Candidate A. It reduced budget but made the agent stop with incomplete/broken output.

## Candidate B: Prompt-Level Structured Tool Thinking

Change under test:

- Add structured-cot-inspired instruction to tool-use system suffix.
- This is prompt-level only, not grammar-constrained decoding, because lightning-mlx does not currently expose GBNF/logit-mask grammar support.
- Keep budgets unchanged to avoid truncating tool-call arguments.

Rationale:

- `structured-cot` reports large compression by constraining `<think>` to compact fields such as `GOAL/APPROACH/EDGE`.
- For this runtime, the nearest safe generic step is a short structured planning format for tool turns while preserving existing parser/tool behavior.

Implementation:

- Tool-use system suffix now says any necessary internal planning before a tool call must be compact structured `GOAL/ACTION/VERIFY`, one short line each, then immediately call the tool.
- Added unit coverage for this suffix.

Validation status:

- Focused tests passed: `154 passed`.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-b`.
- After 40s, no generated files except empty `pi.out`.
- Server first request still had `max_tokens=32768`.
- MTP stats stayed `accepted=0`, rejected increasing.

Decision:

- Reverted Candidate B. Prompt-level structured-cot instruction worsened live behavior. True grammar-constrained decoding may still be useful later, but prompt-only structured planning is not kept.

## Candidate C: Prefer Bash In Tool Template

Change under test:

- Preserve OpenAI response behavior, but present `bash` first in the chat template tool list when available.
- Augment only the `bash` tool description with a generic note: prefer it for multi-file project setup, build, test, and validation workflows.

Why:

- Live artifacts show the model repeatedly chooses one-file write flows (`index.html`, then `snake.ts`) and skips build/validation.
- Pi exposes a `bash` tool that can create multi-file projects and validate them in one turn.
- This is generic for agentic coding workflows and not snake-specific.

Validation status:

- Unit test added for bash ordering/description.
- Focused tests passed: `154 passed`.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-c`.
- After 35s, no generated files except empty `pi.out`.
- Server first request still ran very long with `max_tokens=32768`.
- MTP remained `accepted=0`.

Decision:

- Reverted Candidate C. Tool ordering/description did not improve live behavior.

## Candidate D: Parse Bracketed Equals Tool Calls

Change under test:

- Add generic parser support for malformed tool text like `[Calling tool=read]`.
- Parse `[Calling tool=name({...})]` when arguments are present.
- Treat no-argument `[Calling tool=name]` text as deferred tool-use intent so the retry path handles it instead of streaming it as final content.

Why:

- Candidate A finished with `pi.out` containing `<think>Let</think>` followed by `[Calling tool=read]`.
- That is tool intent, not final assistant content. If parser converts it into a real tool call, Pi can continue instead of stopping on malformed text.

Validation status:

- Unit tests added for `[Calling tool=bash({...})]` parser and `[Calling tool=read]` deferred-tool detection.
- Focused tests passed: `155 passed`.
- Live Pi validation pending.

## Candidate E: Conservative First Tool Budget Cap

Change under test:

- Cap every request with tools at `2048` generated tokens, including the first tool request.
- Keep retry count and temperature behavior unchanged.

Why:

- Baseline showed first tool requests using `max_tokens=32768`, which allowed long planning/prolixity before a tool call.
- Candidate A proved aggressive caps (`1024` first request, `512` retries) reduce budget but hurt artifact quality.
- Candidate E is the smallest conservative runtime bound: prevent runaway first turns without squeezing valid multi-file tool arguments.

Validation status:

- Unit test added for `_tool_turn_max_tokens`.
- Focused tests passed: `156 passed`.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-e`.
- `pi`: timeout after 120s (`EXIT:142`).
- Files: only `index.html` plus empty `pi.out`.
- Server confirmed first tool request was capped from `32768` to `2048`.
- But it still generated full `2048` tokens on consecutive tool turns.
- MTP stayed `accepted=0` while rejected count climbed.

Decision:

- Partially useful but incomplete. It proves first-turn cap applies, but still violates the no-2048-loop criterion.
- Superseded by Candidate F with lower cap and generic shell-tool preference.

## Candidate F: Lower Tool Cap Plus Shell Tool Preference

Change under test:

- Cap tool requests at `1536` tokens to avoid 2048-token loops.
- Present shell-style tools (`bash`, `shell`, `exec`, `run_command`) first in the chat-template tool list.
- Add a short generic shell-tool description hint for multi-file changes plus build/test/validation.

Why:

- Candidate E still hit repeated 2048-token tool turns.
- Pi created one file at a time. Shell tools can create and validate a multi-file project in one turn.
- Candidate C tested shell preference alone and failed because first request still had `32768`; Candidate F tests the combined runtime fix.

Validation status:

- Implemented.
- Focused tests passed: `157 passed`.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-f`.
- `pi`: timeout after 150s (`EXIT:142`).
- Files: `index.html`, `snake.ts`, empty `pi.out`.
- `index.html` references `snake.js`, but no `snake.js` or build/package was created.
- Server confirmed cap lowered to `1536`.
- Still multiple long tool turns: 1536, 1536, 92, then cancelled after 498 tokens.
- MTP stayed `accepted=0`, rejected exceeded 4000.

Decision:

- Partial improvement only. More files were created, but still not functional and still too slow.
- Keep shell preference as a candidate only if later validation passes; otherwise revert before final.

## Candidate G: Explicit No-MTP Comparison

Change under test:

- Add explicit `--disable-mtp` override so MTPLX presets can be compared without speculative decode.
- Run the same Candidate F runtime behavior with MTP disabled.

Why:

- All 35B tool-heavy runs show MTP `accepted=0`; every speculative step is rejected/corrected.
- Need proof whether MTP is net-negative for this workload before changing defaults.

Validation status:

- `--disable-mtp` implemented as explicit CLI override only.
- Focused tests passed: `158 passed`.
- Live result: rejected.

Live result:

- Directory: `/tmp/rapid-mlx-pi-snake-candidate-g-nomtp`.
- `pi`: timeout after 150s (`EXIT:142`).
- Files: only `index.html`, `styles.css`, empty `pi.out`.
- No MTP stats were emitted, confirming MTP was disabled.
- Decode throughput per long turn improved versus MTP, but artifact quality/regression was worse.

Decision:

- Keep `--disable-mtp` as explicit diagnostic/fallback flag unless later tests show downside.
- Do not disable MTP by default from this evidence alone.

## Candidate H: Parser Change

Change under test:

- Considered running the 35B MTPLX model with `--tool-call-parser qwen3_xml`.

Why:

- Hypothesis was that parser mismatch could cause late fallback tool calls.
- User clarified this is wrong for Qwen3.6: parser must always be `qwen3_coder_xml` with `reasoning-parser qwen3`.

Validation status:

- No parser change kept.
- Added auto-config test proving Qwen3.6 resolves to `qwen3_coder_xml` + `qwen3`.
- Existing MTPLX preset already applies `qwen3_coder_xml` + `qwen3`.
- Focused tests passed: `187 passed`.

## Conversion / Model Quality Check

Question:

- Could `convert-mtplx` have produced a bad model, or could the 35B-A3B architecture be a poor MTPLX fit?

Evidence:

- `mtplx inspect /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed --json` passed.
- Compatibility status: `verified-native`.
- Runtime contract found: `mtplx_runtime.json`.
- MTP sidecar found: `mtp.safetensors`.
- MTP tensor gate passed: `15` expected tensors, `15` present, no missing keys, no extra keys.
- Runtime contract exactness baseline reports `max_abs_diff: 0.0`.
- Converted model files match expected 4-bit base layout plus `mtp.safetensors` and `mtplx_runtime.json`.

Interpretation:

- No current evidence of a broken conversion layout.
- Still possible that this 35B-A3B 4-bit base plus BF16 MTP sidecar has poor speculative acceptance in this runtime.
- Live evidence supports that concern: MTP acceptance stayed `0` in agentic tool-heavy runs.
- Disabling MTP improved raw decode speed but worsened artifact quality, so the issue is not only MTP overhead.

Decision:

- Do not blame `convert-mtplx` yet.
- Treat this as either runtime MTP acceptance mismatch for 35B-A3B, model/tool-use weakness, or both.

## Pi `userThe` Degeneration Check

Question:

- The Pi terminal shows `userThe userThe...` for a simple `hi`. Is that MTP runtime mismatch, bad conversion, or model weakness?

Matrix:

- Base 4bit model, direct OpenAI curl, no MTP:
  - Command served `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.
  - Response: `Hi! How can I help you today? 😊`.
  - No `userThe`.
- MTPLX model, direct OpenAI curl, `--disable-mtp`:
  - Response: `Hi! How can I help you today? 😊`.
  - Same prompt/completion usage as base direct curl: `11` prompt, `212` completion.
  - No `userThe`.
- MTPLX model, direct OpenAI curl, MTP enabled:
  - Response: `Hello! How can I help you today? 😊`.
  - Usage: `11` prompt, `34` completion.
  - No `userThe`.
- MTPLX model, Pi `hi`, MTP enabled:
  - Command wrote `/tmp/rapid-mlx-pi-hi-mtp/pi.out`.
  - Output reproduced `The userThe userThe...`.
  - Server request shape: stream=true, `msgs=2`, roles `['system', 'user']`, `tools=4`, total chars around `2436`, prompt tokens `1727`.
  - MTP stats during this request: `accepted=0`, rejected climbed to `768`.
  - Completion hit `1536` tokens.
- MTPLX model, Pi `hi`, `--disable-mtp`:
  - Output file size `0`, exit `0`.
  - Server generated only `38` tokens.
  - No `userThe`.
- Base 4bit model, Pi `hi`:
  - Output file size `0`, exit `0`.
  - Server generated only `33` tokens.
  - No `userThe`.

Interpretation:

- `convert-mtplx` base copy is likely fine because MTPLX with MTP disabled behaves like base.
- Direct API with MTP enabled is also fine for the tiny no-tool request.
- Degeneration is specific to MTP enabled under Pi's agentic request shape: long system prompt plus tools plus streaming.
- This is now strong evidence of an MTP acceptance/runtime mismatch for tool-heavy 35B-A3B requests, not a generic tokenizer/model conversion failure.

## Full-Precision MTPLX Comparison

Question:

- Is the `userThe` degeneration caused by Qwen3.6-35B-A3B itself, by quantization, or by MTPLX conversion?

Additional matrix:

- Full source model `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B` served with `--enable-mtp`:
  - Runtime warning: no `model-mtp.safetensors` / `mtp.safetensors`; MTP validation failed and was disabled.
  - Pi `hi`: exit `0`, output size `0`.
  - Server generated `36` tokens.
  - No `userThe`.
- Full converted MTPLX model `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` served with MTP enabled:
  - MTP injected and validated.
  - Pi `hi`: exit `0`, output size `0`.
  - Server generated `34` tokens.
  - No `userThe`.
- Earlier 4-bit converted MTPLX model `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed` with MTP enabled:
  - Pi `hi` reproduced `The userThe userThe...`.
  - Server generated `1536` tokens and MTP accepted `0`.

Interpretation:

- Qwen3.6-35B-A3B full MTPLX can run Pi `hi` with MTP enabled without degeneration.
- The bug is not explained by Qwen3.6-35B-A3B architecture alone.
- Current strongest culprit is the 4-bit converted model path: 4-bit base plus BF16 MTP sidecar or quantized-base/MTP acceptance mismatch.
- The full source model is not a true MTP comparison because MTP did not activate there.

## Full MTPLX Express/Bun/TypeScript Pi Run

Prompt:

```text
create a REST api using express and bun and typescript.
```

Model:

```text
/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed
```

Serve:

```bash
uv run lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed --enable-mtp --disable-prefix-cache --log-level INFO
```

Result:

- Directory: `/tmp/rapid-mlx-pi-express-bun-full-mtplx`.
- Pi exit: `0`.
- No `userThe` output; `pi.out` size `0`.
- Generated files:
  - `package.json`
  - `tsconfig.json`
  - `src/index.ts`
  - `src/routes/userRoutes.ts`
  - `src/services/userService.ts`
  - `src/types/index.ts`
- `bun install` passed.
- `bun run build` passed and created `dist/index.js`.
- Smoke:
  - Server started with `bun run start`.
  - `GET /api/users` returned `200`.
  - `GET /` returned `404` with JSON `{ success: false, error: "Route not found" }`, which matches generated catch-all behavior.

Runtime evidence:

- MTP injected and validated.
- MTP acceptance remained `0` during long agentic turns.
- Despite `accepted=0`, full MTPLX avoided the `userThe` degeneration seen on the 4-bit MTPLX model.
- Some turns were still slow: examples include 73 tokens in 74s and 1036 tokens in 108s.

Interpretation:

- Full MTPLX can complete a practical agentic coding prompt with valid build and endpoint smoke proof.
- The remaining performance issue is MTP acceptance staying at `0`; full precision avoids corruption but still pays speculative overhead.
- This further isolates the severe degeneration to the 4-bit MTPLX path, while quality on full MTPLX is usable but slow.
