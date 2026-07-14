# Agent × Family support matrix

This page renders the Tier-1 **agent × model-family** integration matrix
truthfully from the authoritative test suite in
`tests/integrations/` — the matrix cells
(`test_agents_matrix.py`), the family aliases and strict-xfail rules
(`conftest.py`), and the pilot run recorded in
`tests/integrations/README.md`.

> **"Support" means a real integration test**, not just a YAML profile: a
> cell PASSes only when the server boots the real model and a real client
> flow succeeds (real tool-call routing, or — for aider / OpenHands — a
> real file rewrite). A cell is never marked PASS if the test skips or
> xfails. See `workflow.md` W3 taxonomy §B.3.

## Legend

- **✅ PASS** — real inference, real tool call (or real file rewrite for the
  bash-CLI / Docker harnesses), semantic assertion.
- **XFAIL (arch)** — the model architecturally cannot emit OpenAI-shape
  `tool_calls`. Applies to the DeepSeek Tier-1 rep
  (`deepseek-r1-32b-4bit`): R1's post-training was reasoning-only
  (arXiv 2501.12948 §2.3.3) and distillation into Qwen 32B lost the base
  model's tool-emission behavior. Root-caused (G8), not a parser bug.
  Tracked in issue #1041 (V4-Flash hardware plan). Marked
  `xfail(strict=True)` so it flips to a red XPASS if a future change unlocks
  tool calls.
- **XFAIL (format)** — gpt-oss × OpenHands only. gpt-oss's native harmony
  output (analysis + final channels, plain-markdown code) does not emit the
  `<execute_bash>` / `<execute_ipython>` text-action tags that OpenHands'
  CodeActAgent parses. Upstream OpenHands parser gap
  ([#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167)); the
  rapid-mlx wire-level harmony-stop fix landed in PR #1051.
- **XFAIL (Ultra)** — the Hy3 (Tencent Hunyuan 3) family. Its only SKU,
  `hy3-preview-4bit`, is a 295B/21B-active MoE at 166 GB / ~156 GB peak
  (`min_memory_gb: 192`) — single-node-infeasible in per-PR CI under the G11
  100 GB free-disk floor, exactly like DeepSeek V4-Flash. Every Hy3 cell is
  `xfail(strict=True)`; real inference runs only in the **weekly Golden Path
  job** on M3 Ultra hardware. Always-on parser coverage lives in the offline
  `test_hy3_offline.py` (parser wire, no model boot).

## Families

| Column | Boot alias (matrix rep) | Tool-call parser | Reasoning parser |
|---|---|---|---|
| Qwen 3.6 | `qwen3.6-*` (matrix reps via `qwen3.5-4b-4bit`, shares parsers) | `qwen3_coder_xml` (3.6) / `hermes` (3.5 rep) | `qwen3` |
| Gemma 4 | `gemma-4-12b-4bit` | `gemma4` | `gemma4` |
| DeepSeek | `deepseek-r1-32b-4bit` | `deepseek` | `deepseek_r1` |
| gpt-oss | `gpt-oss-20b` | `harmony` | `harmony` |
| Hy3 | `hy3-preview-4bit` (Ultra-only) | `hy_v3` | `hy_v3` |

> The matrix boots the smallest available alias per family to stay under the
> per-process memory budget. The 27–35B family flagships are exercised in the
> weekly Golden Path job. Qwen 3.6 has no <27B SKU, so the small-alias matrix
> uses `qwen3.5-4b-4bit` as a stand-in — it shares the `qwen3` reasoning
> parser and `hermes` tool-call family with Qwen 3.6, exercising the same
> wire without loading a 15 GB weight blob per test process.

## Agent × Family (11 agents × 5 families)

Source: `tests/integrations/README.md` "Current cell status" (pilot run
2026-07-06, four always-on families; Hy3 column is `xfail(strict=True)` from
0.11.0).

| Agent | Wire | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss | Hy3 |
|---|---|---|---|---|---|---|
| <a id="agent-codex-cli"></a>[codex-cli](codex-cli.md) | `/v1/responses` | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) |
| <a id="agent-claude-code"></a>[claude-code](claude-code.md) | `/v1/messages` | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) |
| <a id="agent-opencode"></a>[opencode](opencode.md) | `/v1/chat/completions` | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-qwen-code"></a>[qwen-code](qwen-code.md) | `/v1/chat/completions` | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-openhands"></a>[openhands](openhands.md) | Docker → `/v1/chat/completions` | ✅ | ✅ | ✅ | XFAIL (format) | XFAIL (Ultra) |
| <a id="agent-hermes-agent"></a>[hermes-agent](hermes-agent.md) | `/v1/chat/completions` | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-aider"></a>aider | bash CLI → `/v1/chat/completions` | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) |
| <a id="agent-kilo-code"></a>kilo-code | `/v1/chat/completions` | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-copilot"></a>copilot | `/v1/chat/completions` (wire smoke) | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-droid"></a>droid | `/v1/chat/completions` (wire smoke) | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| <a id="agent-kimi-code"></a>kimi-code | `/v1/chat/completions` (wire smoke) | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |

Notes on cell shape:

- **codex-cli / claude-code** use text-only routes (`/v1/responses`,
  `/v1/messages`), so they PASS on DeepSeek R1-Distill where the tool-calling
  agents XFAIL — R1 answers inline instead of emitting `tool_calls`.
- **aider / openhands** drive a real CLI / Docker harness and assert a file
  was rewritten; aider's `SEARCH/REPLACE` edit format and OpenHands'
  text-action tags don't require OpenAI `tool_calls`, so aider PASSes on
  DeepSeek and OpenHands PASSes on DeepSeek (but XFAILs on gpt-oss — format
  mismatch, see legend).
- **copilot / droid / kimi-code** are **wire-smoke only** in CI: their BYOK
  routes are documented and verified, but driving the real CLI binaries is
  blocked on vendor OAuth / first-run onboarding. The wire smoke still
  catches server-side tool-call regressions that would break BYOK users.

## Framework × Family (3 frameworks × 5 families)

Source: `test_frameworks_matrix.py` + `tests/integrations/README.md`.

| Framework | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss | Hy3 |
|---|---|---|---|---|---|
| LangChain (+ LangGraph) | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| PydanticAI | ✅ | ✅ | XFAIL (arch) | ✅ | XFAIL (Ultra) |
| smolagents | ✅ | ✅ | ✅ | ✅ | XFAIL (Ultra) |

smolagents PASSes on DeepSeek because its `ToolCallingAgent` uses a
code-execution routing style that does not require the OpenAI `tool_calls`
shape.

## Totals

Across the four always-on families (Qwen 3.6, Gemma 4, DeepSeek, gpt-oss);
the Hy3 column adds 11 more agent cells, all `xfail(strict=True)`:

- **Agents:** 11 agents × 4 families = 44 cells → **36 PASS · 8 XFAIL ·
  0 FAIL**. The 8 XFAIL are the 7 DeepSeek arch cells (opencode, qwen-code,
  hermes-agent, kilo-code, copilot, droid, kimi-code) + 1 gpt-oss × OpenHands
  format cell.
- **Frameworks:** 3 frameworks × 4 families = 12 cells → the two
  `tool_calls`-dependent frameworks (LangChain, PydanticAI) XFAIL on DeepSeek;
  smolagents PASSes everywhere.
- **Combined always-on run** (56 cells excluding Hy3): **46 PASS · 10 XFAIL
  · 0 FAIL** — 9 XFAIL are the DeepSeek R1-Distill architectural
  tool-emission cells, 1 is gpt-oss × OpenHands.
- **Hy3 (0.11.0):** +14 cells, all `xfail(strict=True)` (Ultra-only). The
  CI-runnable coverage is the 8-test offline `test_hy3_offline.py`
  (**8 PASS** in the normal `pytest tests/` sweep).

## Reproducing

Strict CI runs one family shard per booted server:

```bash
# Boot the family you want to test (positional alias — never --model)
rapid-mlx serve qwen3.5-4b-4bit --tool-call-parser hermes --enable-auto-tool-choice

# Run the agent matrix for that family, strict
RAPID_MLX_MATRIX_STRICT=1 RAPID_MLX_AGENT_MATRIX_FAMILY=qwen36 \
    pytest tests/integrations/test_agents_matrix.py
```

Valid `RAPID_MLX_AGENT_MATRIX_FAMILY` values: `qwen36`, `gemma4`,
`deepseek`, `gptoss`, `hy3` (`hy3` is Ultra-only, weekly Golden Path only).
Without a running server every cell **skips** (non-strict) so a naive
`pytest tests/` stays green.

## See also

- Per-agent setup: [codex-cli](codex-cli.md) · [claude-code](claude-code.md)
  · [opencode](opencode.md) · [qwen-code](qwen-code.md) ·
  [openhands](openhands.md) · [hermes-agent](hermes-agent.md)
- [AI client compatibility](../guides/ai-clients.md) · [Tool calling](../guides/tool-calling.md)
