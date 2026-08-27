# Issue #2222 — Weather routing + strict-format: reproduction + root-cause classification (2026-08-25)

Owner: ds0732 (temporary human-owner exception; RC2 path untouched). This document
starts as the baseline reproduction + classification, and is updated to note the
implemented eval coverage (PR #2327) once it landed. Original worktree base:
`fix/issue-2222-weather-routing` from `origin/main` `bf2ff335`.

## Reported symptoms (issue #2222, filed at base `4bbca765`, build 0.12.18)
1. Explicit Weather prompt routed to `web_search`; the answer claimed no Weather tool
   was available while still listing `weather` in its own advertised inventory.
2. "Write exactly two sentences" prompt produced one long sentence.

## Baseline reproduction (current origin/main, model `ornith-1.5-9b-bf16`)

Request shape replicated the Desktop contract: `tools` = [weather, web_search] with
the exact shipped `WeatherTool` + `WebSearchTool` schemas (including their
"not web_search" / "not for current weather" cross-references), `tool_choice: "auto"`
(free-typed prompt → no `forcedTool`; the #2244 native router), temperature 0.7.

Weather prompt (`What is the current weather in Tokyo? Use the Weather tool, not web
search, and report the tool result.`):
- 5/5 samples → tool call `weather` with valid args
  (`{"location":"Tokyo","country":"Japan","units":"metric"}` variant).
- No web_search, no "Weather unavailable" messaging, no permission ask.

Strict-format prompt (`Explain why local AI can be useful. Write exactly two sentences,
no heading or bullets.`):
- 5/5 samples → exactly two sentences (correct terminal punctuation).

Control model `qwen3.5-9b-4bit` (same arch family, reference):
- Weather 4/4 → `weather`; strict format 4/4 → two sentences.

Effective system/tool-choice inputs (captured from server): the request carried both
tool schemas + `tool_choice:auto`; the rendered prompt registered ~601 tokens (tools +
system content). First-round request carries no tool-result grounding preamble — it
matches the Desktop's free-typed turn exactly.

## Root-cause classification

### Symptom 1 — Weather routed to web_search: Rapid product-contract bug, ALREADY FIXED
The old Desktop routing (present at the issue's base) forced `web_search` for any
free-typed prompt whose text contained a live-evidence phrase, including
`"current weather"` / `"当前"` (see removed `forcedToolForUserTurn` /
`promptRequiresFreshWebEvidence` in `ChatViewModel`). Because the issue's prompt
contains "current weather", the app force-dispatched `web_search` even though the
`weather` tool was enabled — the observed misroute.

PR **#2244** ("fix(mac): preserve Qwen context and weather routing") removed that
heuristic and adopted **native schema-driven tool routing** (send both schemas with
`tool_choice:auto`; the model picks; the schemas cross-reference `weather`↔`web_search`).
Verified at current `origin/main`: the exact prompt routes 5/5 to `weather` (both
Ornith 1.5 9B and the qwen3.5-9b-4bit control). No product code change required.

Note: when the Weather tool is genuinely absent from the supplied list (old
single-tool contract, `weather_only_web_search`), the model truthfully reports no
Weather tool is available and offers web search — that is correct, not a contradiction.

### Symptom 2 — strict sentence count: model capability, NOT a product defect
At current head both the reported model and the control satisfy the constraint. There
is no product code path that can enforce a strict sentence count without a prompt
rewrite/trick (explicitly a non-goal). This is an instruction-following / eval-coverage
observation, not a routing or contract defect.

## Proposed disposition (per the anti-scope-creep rule)
- The two symptoms do NOT share a product-contract defect.
- Weather routing: already fixed by #2244 on current main → verify + close/dispose as
  resolved, no product change.
- Strict format: split/disposition as a separate model-eval format evidence item
  (add a model-agnostic eval case), no product fix.

## Implemented: model-agnostic regression coverage (PR #2327)
The shipped `evals/prompts/tool_calling.json` suite could not originally exercise
weather-vs-web_search disambiguation: its global tool list had no `weather` and
scenarios could not override the tool subset. PR #2327 adds scenario
`tc31-weather-explicit` — it advertises the two Desktop-authentic schemas inline,
sends `tool_choice:auto` (native routing), and asserts `weather` for an explicit
current-weather request while forbidding `web_search`. This locks the corrected
contract, touches no product routing, and is a focused model-agnostic eval change
(approved; non-RC). See the "Verification" section for the harness-correctness
fixes (timing, forbidden-tool rejection, final-turn no-tool rule, empty-arg
handling) and their committed regression tests.

## Acceptance status
1. Baseline reproduction + root-cause classification documented: DONE (this doc).
2. Fix model-agnostic, uses existing tool contracts: Weather already uses native
   schema routing (#2244); no new product code needed.
3. Explicit Weather request → `weather`, never contradictory messaging: verified 5/5.
4. Disabled/missing Weather, auto choice, unrelated tools not regressed: verified
   (`weather_only_web_search` truthful fallback; auto + both tools healthy).
5. Strict-format: satisfied by the model family; disposed/split as separate eval item.
6. Local Release build + real-model dogfood + focused tests + review + PR validation:
   repro done via local server + real model; remaining PR/review only if Atlas approves
   the eval-coverage change.
7. No full-ci until Atlas clears RC2 lanes: respected.

## Repro commands (reproducible)

The server command is exact and reproducible. The client scripts
(`/tmp/issue2222-evidence/repro.py`, `multisample.py`) were throwaway local
diagnostics for the baseline reproduction and are NOT part of this change; the
committed, reproducible regression coverage is the `tc31-weather-explicit` eval
scenario + its harness tests in `evals/` (see "Verification"). If you need to
repro the baseline prompt routing, send the exact Desktop request shape via any
OpenAI-compatible client with both schemas + `tool_choice:auto`.

```
python3.12 -m vllm_mlx.cli serve ornith-1.5-9b-bf16 --port 8899 \
  --tool-call-parser hermes --reasoning-parser qwen3 --log-level DEBUG
# control
python3.12 -m vllm_mlx.cli serve qwen3.5-9b-4bit --port 8898 \
  --tool-call-parser hermes --reasoning-parser qwen3
```
Baseline evidence captured during investigation: local `/tmp` dumps (not committed).

## Verification of the eval-coverage change (2026-08-25)
The approved change (WEATHER_TOOL + WEB_SEARCH_TOOL + `_resolve_tools` in
`evals/run_eval.py`, scenario `tc31-weather-explicit`) was run once against the cached
`ornith-1.5-9b-bf16` model, then updated for Codex round-1 findings:
- tc31 advertises the two **Desktop-authentic** schemas inline (WeatherTool +
  WebSearchTool, including the "do not use web_search for current weather" guard) so it
  models the real Desktop two-schema contract, not a synthetic setup.
- `verify_final_text` (opt-in) runs one more non-streaming completion after the tool
  result and requires the final turn to (a) be non-empty, (b) call NO tool (any
  `tool_calls` — a repeat `weather` or a `web_search` fallback — fails the final turn
  rather than counting as an answer), (c) contain none of the scenario-declared
  `forbid_final_phrases` (a bounded phrase list asserting the weather tool is
  unavailable / use web search instead / can't use weather, etc.), and (d) reflect the
  supplied result (`require_final_terms`, any-match). This is an intentional bounded
  contract: it enforces the NAMED denial phrases and required terms, not a general
  semantic non-contradiction proof (a substring check cannot grade arbitrary
  contradictory wording). Verified end-to-end: the first call routes to **`weather`**,
  and after feeding the weather result the final completion yields a clean report that
  reflects the result ("Partly cloudy" / "18°C" / "62%") with no "web_search
  unavailable" claim; a reply denying the weather tool, suggesting web search, or
  issuing a further tool call fails the check.
- `_resolve_tools` now supports name refs OR inline schema dicts and **fails fast** on
  an unknown tool name, a non-empty-but-malformed `tools` value (`[]`, a non-list, a
  scalar), a schema dict that is not well-formed (no non-empty `function.name`), or a
  malformed entry — rather than silently dropping or broadening to the global set,
  either of which could let a routing case pass while omitting web_search. Only the
  absence of `tools` keeps the shared `TOOLS` list unchanged. Per-scenario tools are
  applied consistently across the standard, parallel, irrelevance / missing-params, and
  error-recovery branches (default is still the shared list).
- tc31 sets `first_call_stream: false`, so its first tool-detection uses the
  non-streaming completion and captures the structured `weather` call (this model
  family emits the call as streamed text under SSE). The non-streaming branch reports
  real wall-clock `elapsed` and does NOT fabricate a `ttft` (TTFT is undefined off the
  streaming path). That ensures the tool result is fed back and the `verify_final_text`
  check actually runs through the real suite path.
- tc31 sets `forbid_tools: ["web_search"]`: the standard branch checks EVERY tool call
  in the first response and fails if a forbidden tool was also called (a weather-then-
  web_search multi-call response must fail, not just grade on the first call), so
  "routes to weather, never web_search" is enforced, with a unit-level multi-call
  verification.
- Unit checks: JSON valid; `run_eval.py` parses; resolver resolves `[weather,
  web_search]`, keeps default behavior, raises on `weather`+`web_seach` and on `[]` /
  dict / scalar `tools`.
- Caveat: the eval suite's `stream_chat` auto-grades only structured
  `delta.tool_calls` in SSE. This model family emits the tool call as streamed content
  text, so the streaming auto-grade reports "no tool call" for every tool-detection
  scenario (tc01–tc17, tc21–tc31) — a PRE-EXISTING harness × model limitation, present
  on the unmodified harness, not a regression from this change. Models that emit
  structured streaming tool_calls auto-grade normally. The scenario encodes the
  corrected contract (weather over web_search) and is model-agnostic by construction.
