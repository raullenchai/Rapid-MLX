# Integration tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need a running
Rapid-MLX server on `http://localhost:8000` and a loaded model — the fixtures
`skip` cells when no server is reachable, so a naïve `pytest tests/` still
comes out green.

## Two matrices — 12 agents + 3 frameworks × 6 families

0.10.2 PR-2 pilot expanded the matrices to the 0.10.2 **Tier-1 four
families** (added DeepSeek V4) and the finalized **top-10** commercial /
open-source agents (three commercial-CLI cells added via docs-confirmed
BYOK routes). 0.11.0 adds **Hy3 (Tencent Hunyuan 3)** as the Tier-1 5th
family; the 0.12-window adds **Muse Glimmer (Meta)** as the Tier-1 6th.
Both matrices share the harness in `conftest.py`:

- `test_agents_matrix.py` — **11 Tier-1 agents × 6 families** (Qwen 3.6,
  Gemma 4, DeepSeek V4, gpt-oss, Hy3, Muse Glimmer) = 66 cells. Each cell
  is a lightweight smoke; deep flows live in the dedicated files below.
- `test_frameworks_matrix.py` — **3 Tier-1 frameworks × 6 families** =
  18 cells.

Total: **84 cells** (up from 70; +14 Muse cells, all expected-pass — see
the Muse note below).

> **Hy3 is Ultra-only and strict-xfail in always-on CI.** Hy3
> (`hy3-preview-4bit`) is a 295B/21B-active MoE whose only SKU is 166 GB
> (~156 GB peak, `min_memory_gb: 192`) — single-node-infeasible on the
> M3 Ultra under the G11 100 GB free-disk floor, exactly like DeepSeek
> V4-Flash. Rather than downgrade its 14 cells to plain `skip` (G8:
> root-cause failures, do not hide behind skips), every Hy3 matrix cell
> is `xfail(strict=True)` (applied in
> `conftest.py::pytest_collection_modifyitems`,
> reason `conftest.py::_HY3_XFAIL_REASON`). Real Hy3 inference runs only
> in the **weekly Golden Path job on real Ultra hardware**, never in
> per-PR CI. The always-on CI value-add for Hy3 is the offline
> parser-level integration test **`test_hy3_offline.py`** — it drives
> captured Hy3 wire strings through the `hy_v3` tool + reasoning parsers
> (the parsers `hy3-preview-4bit` wires) and asserts the OpenAI-API-shape
> contract (tool_calls array well-formed, `<think>` reasoning routed to
> its own channel, no leak) **without booting the 166 GB model**. That
> file runs in the normal `pytest tests/` sweep (8 tests, sub-second).

> **Muse Glimmer is always-on and expected-pass — NOT strict-xfail.**
> Muse Glimmer (`muse-glimmer-30b-4bit`) is Meta's Muse Glimmer 30B, the
> ATEM function-calls + recipient-routed-channel wire vendored in #1791
> (`muse` tool + reasoning parsers) and #1802 (text backbone). Unlike Hy3
> / DeepSeek V4-Flash it is **not** single-node-infeasible: there is no
> sub-8B Muse SKU so the 30B is the alias, but the 4-bit checkpoint is
> only ~18 GB (`min_memory_gb: 24`) — per-PR bootable on the M3 Ultra in
> the same footprint class as the gpt-oss 20B cell. Muse emits
> OpenAI-shape `tool_calls` and drives real 2-turn agent loops on
> `/v1/chat/completions`, `/v1/responses` (Codex), and `/v1/messages`
> (Claude Code), verified end-to-end on real 30B weights (sequential
> tool-calling, reasoning routed to the `to=self` channel, zero
> channel-marker leak). So — like Qwen 3.6 / Gemma 4 / gpt-oss — every
> Muse cell is a live always-on cell **expected to PASS**, carrying no
> strict-xfail.

> **Pilot scope note.** This pilot runs the Qwen 3.6 35B-A3B-8bit
> family end-to-end and leaves Gemma 4 / DeepSeek V4 Flash / gpt-oss
> 120B for sibling PRs (see PR-2c-1 / PR-2c-2 / PR-2c-3 in the parent
> issue). All four aliases resolve correctly in
> `vllm_mlx/aliases.json` today; only the pilot family is proven with
> real inference in this PR.

Support ≡ a real integration test that boots the server + real model + real
client flow, not just a YAML profile. The strict-xfail rules live in
`conftest.py`.

### Pre-flight verdict — commercial CLI top-10 finalization

Task #461 asked the pilot to verify five commercial CLIs against a
custom OpenAI base_url before wiring their matrix cells. Verdicts
recorded here (no CLI binaries were installed — verdict is based on
official BYOK docs review):

| CLI | Pre-flight verdict | Reason | Kept in top-10? |
|---|---|---|---|
| GitHub Copilot | **PASS** | `COPILOT_PROVIDER_BASE_URL` + `COPILOT_PROVIDER_API_KEY` env vars documented at [docs.github.com](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models) | ✅ (new cell `TestCopilot`) |
| Factory AI Droid | **PASS** | `~/.factory/settings.json` `customModels` array with `provider: generic-chat-completion-api`, docs at [docs.factory.ai](https://docs.factory.ai/cli/byok/overview) | ✅ (new cell `TestDroid`) |
| Moonshot Kimi Code | **PASS** | `~/.kimi/config.toml` provider block with `type = "openai"` + `base_url`, docs at [moonshotai.github.io](https://moonshotai.github.io/kimi-cli/en/configuration/providers.html) | ✅ (new cell `TestKimiCode`) |
| Cursor | **DEFERRED** | Cursor routes BYOK requests through its backend, which cannot reach a server bound to the user's localhost. A public HTTPS endpoint would no longer be a fully local integration | ❌ — fallback promoted: `qwen-code` |
| Alibaba Qoder | **DEFERRED** | Native Qoder CLI has no first-party OpenAI base_url hook — only third-party proxy wrappers (`qoder-proxy`, `qoder-cli-api`). Wire-smoke would misrepresent Qoder's native shape | ❌ — fallback promoted: `hermes-agent` |

**Final top-10** (order preserves task's ranking with fallbacks
inserted at the DEFERRED slots): `codex-cli`, `claude-code`, `opencode`,
`openhands`, `copilot`, `qwen-code` (for Cursor), `droid`, `kimi-code`,
`hermes-agent` (for Qoder), `aider`. `kilo-code` is retained from
#1030's Tier-1 list (11 cells total instead of exactly 10 — flagged
for operator scope call in the PR body).

### Tier-1 agents

| Agent | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| codex-cli | `/v1/responses` | `TestCodexCLI` | (matrix only) |
| claude-code | `/v1/messages` | `TestClaudeCode` | `test_anthropic_sdk.py` |
| opencode | `/v1/chat/completions` | `TestOpenCode` (wire smoke via OpenAI SDK) | (matrix only) |
| qwen-code | `/v1/chat/completions` | `TestQwenCode` (wire smoke via OpenAI SDK) | (matrix only) |
| openhands | shell subprocess → Docker → `/v1/chat/completions` | `TestOpenHands` (**real Docker E2E harness** — drives pinned OpenHands 0.9.0 app + runtime images with `python -m openhands.core.main -t`, asserts add.py corrected) | `test_openhands.sh` (same harness, standalone) |
| hermes-agent | `/v1/chat/completions` | `TestHermesAgent` (wire smoke via OpenAI SDK) | `test_hermes.py` (real 62-tool E2E) |
| aider | shell subprocess → `/v1/chat/completions` | `TestAider` (**real bash-CLI harness** — drives aider one-shot with `--message`, asserts add.py corrected) | `test_aider.sh` (same harness, standalone) |
| kilo-code | `/v1/chat/completions` | `TestKiloCode` (wire smoke via OpenAI SDK) | (matrix only) |
| deepseek-harness | `/v1/chat/completions` | `TestDeepSeekHarness` (wire smoke via OpenAI SDK) | `agent_smoke.sh` (Tier-1 release gate — real `dsh` binary fixes a real bug, repo test suite must go green) |
| copilot | `/v1/chat/completions` | `TestCopilot` (**wire smoke only** — real CLI needs `gh auth login`, deferred) | (subprocess cell deferred) |
| droid | `/v1/chat/completions` | `TestDroid` (**wire smoke only** — real CLI needs Factory session token, deferred) | (subprocess cell deferred) |
| kimi-code | `/v1/chat/completions` | `TestKimiCode` (**wire smoke only** — real CLI needs Moonshot auth flow, deferred) | (subprocess cell deferred) |

### Tier-1 frameworks

| Framework | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| LangChain (+ LangGraph) | `/v1/chat/completions` | `TestLangChain` | `test_langchain.py` |
| PydanticAI | `/v1/chat/completions` | `TestPydanticAI` | `test_pydantic_ai_full.py` |
| smolagents | `/v1/chat/completions` | `TestSmolagents` | `test_smolagents_full.py` |

## Running

Start the server first (positional model arg — never `--model`):

```bash
rapid-mlx serve qwen3.5-4b-4bit \
    --tool-call-parser hermes --enable-auto-tool-choice
```

Then run either matrix or a specific deep file. **Strict mode requires
one family shard per booted server** — the ``_guard_family_matches_server``
autouse fixture in ``conftest.py`` fails cells that ask for a family the
running server doesn't serve. In practice this means: pick one of the five
always-on families that matches your ``rapid-mlx serve`` alias and boot the
other four in separate server shards (or CI jobs). Hy3 remains an
Ultra-only weekly-Golden-Path lane.

```bash
# All 66 agent cells across six families. Locally, only the matching
# always-on column passes; the other four feasible columns skip
# (non-strict) or fail (strict), while Hy3 keeps its documented strict xfail.
# The always-on release lane is the 55-cell, five-family subset below.
pytest tests/integrations/test_agents_matrix.py -v

# 18-cell framework matrix across six families (same shard rule as above)
pytest tests/integrations/test_frameworks_matrix.py -v

# Strict CI — per-family shard (the always-on lane uses five jobs, one per
# feasible family — qwen36 / gemma4 / deepseek / gptoss / muse — each with
# its own booted server). Release artifact
# acceptance also sets RAPID_MLX_MATRIX_NO_SKIPS=1.
RAPID_MLX_MATRIX_STRICT=1 RAPID_MLX_AGENT_MATRIX_FAMILY=qwen36 \
    pytest tests/integrations/test_agents_matrix.py

# One agent's cells across all families (still shard-restricted)
pytest tests/integrations/test_agents_matrix.py -k QwenCode

# Deep flows (Python)
python3 tests/integrations/test_pydantic_ai_full.py
python3 tests/integrations/test_smolagents_full.py
python3 tests/integrations/test_langchain.py
python3 tests/integrations/test_anthropic_sdk.py
python3 tests/integrations/test_openwebui.py

# Deep flows (CLI / Docker)
bash tests/integrations/test_aider.sh --model qwen3.5-4b-4bit --port 8802
bash tests/integrations/test_openhands.sh --model qwen3.5-4b-4bit --port 8802
python3 tests/integrations/test_librechat_docker.py
```

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `RAPID_MLX_BASE_URL` | `http://localhost:8000/v1` | Where matrix clients point |
| `RAPID_MLX_AGENT_MATRIX_FAMILY` | (all) | Restrict to `qwen36` / `gemma4` / `deepseek` / `gptoss` / `hy3` / `muse` (`hy3` is Ultra-only, weekly Golden Path only; `muse` is always-on, ~18 GB) |
| `RAPID_MLX_MATRIX_STRICT` | `0` | If `1`, missing-server → fail (default: skip) |
| `RAPID_MLX_MATRIX_NO_SKIPS` | `0` | If `1`, the release artifact lane turns ordinary matrix skips (missing SDK, Docker, Aider, etc.) into failures; documented strict xfails remain xfails |

## Cheap-alias policy

The matrix boots the smallest available alias per family — 4B for Qwen 3.5
(3.6 has no <8B SKU), 12B for Gemma 4 (smallest text-only SKU, ~7 GB @ 4-bit),
20B for gpt-oss (no smaller SKU in the family, MXFP4-Q8 ~11 GB), 30B for
Muse Glimmer (no smaller SKU either, but the 4-bit checkpoint is only
~18 GB). The 27-35B family flagships that DO have a cheaper SKU (Qwen 3.6,
Gemma 4) keep the big variant for the weekly Golden Path job; Muse has no
cheaper SKU, so its 30B-4bit is the always-on alias (still only ~18 GB @
4-bit, gpt-oss-class). This keeps the per-process resident footprint under
the W5 OOM budget on
M3 Ultra (operator services baseline + matrix + Metal overhead ≤ 150 GB).

Family choice per matrix run:

| Family | Alias used | Rationale |
|---|---|---|
| Qwen 3.6 | `qwen3.5-4b-4bit` | 3.6 has no <27B SKU; 3.5-4B shares `hermes` / `qwen3` parsers |
| Gemma 4 | `gemma-4-12b-4bit` | Smallest text-only alias; ~7 GB at 4-bit |
| DeepSeek | `deepseek-r1-32b-4bit` | 0.10.2 PR-2 pilot swapped from `deepseek-v4-flash-8bit` (~155 GB weights, single-node-infeasible on 256 GB M3 Ultra + G11 100 GB floor). R1-Distill-Qwen-32B-4bit at ~16 GB stays above the "no cheap-alias" bar and exercises DeepSeek reasoning, but deliberately does not advertise tools because the checkpoint cannot emit calls (#1569). **Full DeepSeek V4 Flash Tier-1 tool-parser slot tracked in follow-up issue #1041** (hardware plan needed). |
| gpt-oss | `gpt-oss-20b-mxfp4-q8` | Smallest gpt-oss; ~11 GB |
| Hy3 (Hunyuan 3) | `hy3-preview-4bit` | **Ultra-only** — 295B/21B-active MoE, 166 GB weights / ~156 GB peak (`min_memory_gb: 192`). No cheap alias exists; single-node-infeasible under G11 like DeepSeek V4-Flash. All 14 Hy3 cells `xfail(strict=True)`; real inference is weekly-Golden-Path-only; always-on CI coverage is the offline `test_hy3_offline.py` (parser wire, no model boot) |
| Muse Glimmer | `muse-glimmer-30b-4bit` | **Always-on** — Meta Muse Glimmer 30B (ATEM function-calls + recipient-routed channels, #1791 parsers + #1802 backbone). No sub-8B SKU exists so the 30B is the alias, but the 4-bit checkpoint is only ~18 GB (`min_memory_gb: 24`) — per-PR bootable, gpt-oss-class footprint. Emits OpenAI-shape `tool_calls`; every Muse cell is expected-PASS, no strict-xfail |

## Current cell status (matrix through the 0.12-window Muse addition; PASS/XFAIL data from the 2026-07-06 · 0.10.2 pilot, plus the 2026-08-10 Muse column)

The PASS / XFAIL results below are from the 2026-07-06 serial pilot run on
the 0.10.2 four-family matrix. The 0.11.0 Hy3 column is `xfail(strict=True)`
across the board (Ultra-only — see the Hy3 note above); its always-on
coverage is `test_hy3_offline.py`, not these live cells. Empty (🔲) cells
have no live coverage yet.

> **Muse column provenance (2026-08-10).** The Muse Glimmer column below
> is from a real 30B-4bit boot on the M3 mini, not the 2026-07-06 pilot.
> The nine wire cells that share the OpenAI `/v1/chat/completions`,
> `/v1/responses` (codex), and `/v1/messages` (claude-code) tool paths
> were driven end-to-end through 2-turn agent loops (tool call → result
> fed back → clean final, reasoning routed to `to=self`, zero
> channel-marker leak) and PASS. The `aider` (bash-CLI) and `openhands`
> (Docker) cells and the three framework cells reuse those same
> tool-call wires and are **expected-PASS**, marked `exp` until run under
> their CLI/Docker/SDK harnesses in a Golden Path pass.

### Agent × Family matrix (11 × 6 = 66)

Pilot execution 2026-07-06 — **serial 3-family run** against real
inference under `RAPID_MLX_MATRIX_STRICT=1`. All PASS cells exercised
real tool-call routing (function name = `get_weather`, arg parses as
JSON, `city == "Tokyo"`, no `<think>` leak, no `<|channel|>analysis`
leak). PydanticAI + smolagents cells assert the tool implementation
itself was invoked (closure counter check), not just that the model
produced final text. Aider now runs a real bash-CLI harness (drop-out
of the previous structural XFAIL): the matrix cell shells out to
`test_aider.sh`, which seeds a scratch `add.py` with
`return a - b  # BUG`, drives `aider --message "Fix the bug ..."`
one-shot against `/v1/chat/completions`, and asserts the file was
rewritten to `return a + b`. Aider's own edit format
(`SEARCH ... REPLACE ...` in plain text) does NOT require OpenAI
tool_calls, so R1-Distill drives it successfully alongside the
other three families.

OpenHands 2026-07-07 update — the four `openhands` cells previously
strict-`XFAIL`'d as a "Docker E2E harness required" placeholder now run
the real harness `test_openhands.sh` which pulls the pinned OpenHands
0.9.0 app + runtime images (`ghcr.io/all-hands-ai/openhands:0.9.0` +
`ghcr.io/all-hands-ai/runtime:od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22`),
invokes `docker run ... python -m openhands.core.main -t "Fix the bug
in add.py — ..." -d /workspace` with the sandbox docker-in-docker
sock passthrough, then asserts the file was rewritten. The correctness
gate is a **strict AST whitelist on `add.py`'s return expression** —
same scratch-file scaffolding as Aider (`return a - b  # BUG` seed) but
a stronger, non-executing gate: parse the file, first require the
module's top level to be **only** a `def add` (optionally preceded by
a docstring — codex #1048 round 6 finding #3 rejected the previous
"allow arbitrary other top-level statements" behaviour that would let
an injected `import os; os.system(...)` slip past a good `def add`),
find `def add(a, b): …`, and require the returned expression to be one
of `{a + b`, `b + a`, `sum([a, b])` / `sum((a, b))`}` after an optional
docstring, with the signature pinned to `(a, b)` (no `*args` /
`**kwargs` / extra positional). The previous `operator.add(a, b)`
branch was removed in round 5 because the whitelist did not verify
`import operator` at module top level, so it would accept a file that
`NameError`d at import time. Strictly stronger than a runtime
pair-sweep (no `a - b + k`, `(a - b) + k`, `return CONST`, or `if …:
return 5` cheat can satisfy the AST shape) AND safer — zero code
execution, so the LLM's output never touches the host process (codex
#1048 rounds 1 / 3 / 4 / 5 / 6). OpenHands'
CodeActAgent parses `<execute_ipython>` / `<execute_bash>` text-action
tags from plain-text LLM output, NOT via OpenAI tool_calls, so
R1-Distill drives it successfully (same pattern as Aider). ONE family
still `XFAIL`s: gpt-oss + OpenHands. The rapid-mlx wire-level bug PR
#1051 fixed (harmony parser channel-scoping of user-supplied `stop=...`
so analysis-channel CoT can no longer trigger a premature stop) has
landed on `main` (commit `e7e4668a`, v0.10.3) and is pinned at the unit
level by `tests/test_harmony_stop_final_channel_only.py`. What still
blocks end-to-end is a format mismatch: gpt-oss's native harmony output
format (analysis + final channels, plain markdown code in the final
channel) does not emit the `<execute_bash>` / `<execute_ipython>`
text-action XML tags that CodeActAgent parses. OpenHands treats the
reply as an empty `MessageAction` → prompts for user input → EOFError
on non-interactive stdin → 300 s wall-clock timeout, `add.py` never
rewritten. This is an upstream OpenHands parser gap, not a rapid-mlx
bug (see `conftest.py::_GPTOSS_OPENHANDS_XFAIL_REASON` block); filed
as an informational note in OpenHands issue
[#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167).

DeepSeek family Tier-1 rep was **swapped** from
`deepseek-v4-flash-8bit` (~155 GB, single-node-infeasible on M3 Ultra
+ G11 100 GB floor) to `deepseek-r1-32b-4bit`
(`mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit`, ~16 GB) after HF-API
size verification showed every complete V4 Flash quant is > 96 GB. The
swap preserves parser coverage (same `deepseek` tool-call parser +
`deepseek_r1` reasoning parser) while fitting the pilot's disk budget.
Full V4 Flash coverage tracked in follow-up issue **#1041**
("hardware plan needed").

| Family | Boot alias | Boot time | Wall time (14 cells) | Result |
|---|---|---|---|---|
| Qwen 3.6 | `qwen3.6-35b-8bit` (MoE, 3 B active) | ~15 s | 11.32 s + ~3 s aider + ~32 s openhands | 14 PASS / 0 XFAIL |
| Gemma 4 | `gemma-4-31b-4bit` (dense) | ~10 s | 17.45 s + ~7 s aider + ~48 s openhands | 14 PASS / 0 XFAIL |
| DeepSeek | `deepseek-r1-32b-4bit` (R1-distilled Qwen 32B, dense) | ~18 s | 191.29 s + ~22 s aider + ~72 s openhands | 5 PASS / 9 XFAIL (9 arch-XFAIL R1-Distill tool-call gap; OpenHands passes because it parses text-action tags, not tool_calls) |
| gpt-oss | `gpt-oss-120b-mxfp4-q8` (MoE) | ~15 s | 14.61 s + ~3 s aider + XFAIL openhands | 13 PASS / 1 XFAIL (OpenHands XFAIL — gpt-oss harmony format vs CodeActAgent text-action parser mismatch, upstream OpenHands [#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167); rapid-mlx wire-level harmony parser bug fixed in #1051) |

> **Aider row added post-pilot 2026-07-07.** The pilot times above are the
> 12-cell subset (aider was structural XFAIL). Re-running with the real
> bash-CLI harness adds ~3–22 s per family (see `test_aider.sh` — aider
> spawns a Python subprocess, boots LiteLLM, then does one
> ``/v1/chat/completions`` round-trip; the DeepSeek row is the outlier
> because R1-Distill emits a long ``<think>`` block before the
> SEARCH/REPLACE edit). Family-by-family verification:
> Qwen 3.5-4B-4bit (`qwen36` rep) 2.81 s, Gemma-4-31B-4bit 7.14 s,
> DeepSeek R1-Distill-32B-4bit 22.26 s, gpt-oss-20B-MXFP4-Q8 3.15 s —
> all four PASS with add.py rewritten to ``return a + b``.

> **OpenHands row added 2026-07-07.** `test_openhands.sh` runs OpenHands
> 0.9.0 inside Docker with a docker-in-docker sandbox — cold-cache
> boot pulls two images (~3.4 GB + ~9 GB uncompressed) and builds a
> hash-tagged runtime layer (first run ~5 min on a warm apt mirror);
> subsequent runs reuse the hash-tagged image and take 30-75 s
> per cell. Family-by-family (2026-07-07, cached-image path, 8802 port,
> `--tool-call-parser <family>` + `--enable-auto-tool-choice`):
> Qwen 3.5-4B-4bit (`qwen36` rep) 32.14 s (2 CodeAct steps: read →
> `edit_file_by_replace` → finish), Gemma-4-31B-4bit 47.87 s, DeepSeek
> R1-Distill-32B-4bit 72.08 s (long analysis-channel CoT before the
> edit action but still one-shot), gpt-oss-20B-MXFP4-Q8 **XFAIL** at
> the harness 300 s wall-clock timeout — reason is the harmony-format
> vs CodeActAgent text-action-parser mismatch documented in the
> paragraph above and pinned in
> `conftest.py::_GPTOSS_OPENHANDS_XFAIL_REASON`, not a rapid-mlx bug.
> Empirical rerun of the other three families under the harness digest
> fix (this PR) deferred to CI.

| Agent | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss | Hy3 | Muse |
|---|---|---|---|---|---|---|
| codex-cli | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) | ✅ |
| claude-code | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) | ✅ |
| opencode | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| qwen-code | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| openhands | ✅ | ✅ | ✅ | XFAIL (format) | XFAIL (Ultra) | exp |
| hermes-agent | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| aider | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) | exp |
| kilo-code | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| deepseek-harness | ✅† | ✅ | XFAIL (arch) | ✅† | XFAIL (Ultra) | ✅ |
| copilot | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| droid | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |
| kimi-code | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | ✅ |

† `deepseek-harness` joined the matrix on 2026-08-17. Its Qwen 3.6 and gpt-oss
cells were run strict (`RAPID_MLX_MATRIX_STRICT=1`) against real weights that
day — `qwen3.6-35b-8bit` and `gpt-oss-20b-MXFP4-Q8` respectively. The Gemma 4
and Muse cells are recorded from the shared code path rather than an
independent run: `TestDeepSeekHarness.test_smoke` is a one-line call to the
same `_run_openai_tool_smoke` helper that backs `kilo-code`, `copilot`,
`droid` and `kimi-code`, and the helper's only per-agent input is a label used
in failure messages — so those columns measure the server's tool-call wire for
the family, which those four identical cells already record. The next full
matrix run resolves them for real; if any disagrees, this footnote is the bug
report.

### Framework × Family matrix (3 × 6 = 18)

| Framework | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss | Hy3 | Muse |
|---|---|---|---|---|---|---|
| LangChain (+ LangGraph) | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | exp |
| PydanticAI | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) | exp |
| smolagents | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) | exp |

Legend: ✅ passing (real inference · real tool call · semantic assertion;
or for aider / openhands: real bash-CLI drive · real file rewrite)
· **exp** = Muse expected-PASS: the cell reuses a tool-call wire already
verified end-to-end on real Muse 30B weights (see the Muse column
provenance note above), but has not yet been run under its CLI / Docker /
SDK harness — it is a live always-on cell, NOT a strict-xfail
· **XFAIL (arch)** = R1-Distill architectural tool-emission gap (see next
paragraph and issue #1041) · **XFAIL (format)** = gpt-oss native harmony
output format (analysis + final channels, plain markdown code in the
final channel) does not emit `<execute_bash>` / `<execute_ipython>`
text-action XML tags that OpenHands' CodeActAgent parses; upstream
OpenHands parser gap tracked at
[All-Hands-AI/OpenHands#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167).
The rapid-mlx wire-level harmony bug that used to underlie this cell is
fixed in PR #1051 (see `conftest.py::_GPTOSS_OPENHANDS_XFAIL_REASON`).
· **XFAIL (Ultra)** = Hy3 (`hy3-preview-4bit`) is 166 GB / ~156 GB peak,
single-node-infeasible in per-PR CI under G11 (like DeepSeek V4-Flash);
real inference runs weekly on M3 Ultra. Always-on parser coverage is the
offline `test_hy3_offline.py`. See `conftest.py::_HY3_XFAIL_REASON`.

**Totals across the 4 always-on families in the 2026-07-06 pilot**
(Muse, the 5th always-on family, was added later — see its own paragraph
below): 56 cells run → **46 PASS · 10 XFAIL · 0 FAIL** (9 XFAIL are the
R1-Distill architectural
tool-emission cells listed in
`conftest.py::_DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS`; 1 XFAIL is the
gpt-oss+OpenHands cell — gpt-oss harmony format vs CodeActAgent
text-action parser mismatch, tracked upstream at
[All-Hands-AI/OpenHands#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167)).

**Hy3 (5th family, 0.11.0)**: +14 cells, all `xfail(strict=True)` in
always-on CI (Ultra-only, weekly Golden Path). The CI-runnable coverage
is the 8-test offline `test_hy3_offline.py` (parser wire, no model boot)
— **8 PASS** in the normal `pytest tests/` sweep.

**Muse Glimmer (6th family, 0.12-window)**: +14 cells, all live always-on
(no strict-xfail). Verified 2026-08-10 on a real `muse-glimmer-30b-4bit`
boot: the 9 wire cells that share the OpenAI / codex / claude-code
tool-call paths PASS through real 2-turn agent loops; the `aider`,
`openhands`, and 3 framework cells reuse those same wires and are
expected-PASS (`exp`), to be exercised under their harnesses in a Golden
Path pass. Muse is `gpt-oss`-class footprint (~18 GB, `min_memory_gb: 24`),
so unlike Hy3 it boots per-PR.

**DeepSeek family — architectural tool-emission gap.** The 9 DeepSeek
tool-call cells (7 agents + LangChain + PydanticAI) are marked
`xfail(strict=True)` via `pytest_collection_modifyitems` in
`conftest.py`. Root cause verified on **both** 4bit (16 GB) and 8bit
(34.8 GB) R1-Distill-Qwen-32B weights: R1's post-training was
reasoning-only per DeepSeek's own paper (arXiv 2501.12948 §2.3.3), and
distillation into Qwen 32B lost the base model's tool-emission
behavior. The refusal
`"I cannot provide the current weather in Tokyo as I cannot access the get_weather tool."`
reproduces deterministically at both quant levels — not a rapid-mlx
parser bug, not a quant artifact. Text-only cells (CodexCLI +
ClaudeCode) and smolagents' code-execution routing PASS on the same
booted server, proving the wire is healthy. Full tool-trained coverage
for the family needs V4-Chat / V4-Coder / V4-Flash weights, all
> 96 GB and single-node-infeasible on M3 Ultra under G11 — tracked in
follow-up issue #1041 (hardware plan).

## Historical deep-file coverage (pre-0.10.2)

For reference — this is what the deep flows historically covered on the
2026-06 M3 Ultra baseline before the matrix restructure:

| Test | Plain | Stream | Tool | Multi-tool | Structured | Notes |
|---|---|---|---|---|---|---|
| `test_pydantic_ai_full.py` | x | x | x | x | x | + multi-turn |
| `test_smolagents_full.py` | x | — | x | x | — | CodeAgent + ToolCallingAgent |
| `test_langchain.py` | x | x | x | x | x | + system prompt |
| `test_anthropic_sdk.py` | x | x | x | — | — | `/v1/messages` endpoint |
| `test_openwebui.py` | — | x | — | — | — | Docker: register, login, models, chat |
| `test_aider.sh` | — | — | — | — | — | CLI edit-and-write workflow |
| `test_openhands.sh` | — | — | — | — | — | Docker E2E: CodeActAgent edit-and-write against a running server (2026-07-07) |
| `test_librechat_docker.py` | — | — | — | — | — | Docker: register, login, endpoints, models |
| `test_hermes.py` | x | x | x | x | — | 62-tool Hermes Agent E2E + API stress test |

Model is auto-detected from the running server (`/v1/models` endpoint).

Run all agent tests automatically via:

```bash
rapid-mlx agents hermes --test
rapid-mlx agents                    # list all supported agents
```
