# Atlas handoff — Personal Intelligence productization

- **Owner:** Atlas
- **Branch / PR:** `feat/personal-intelligence` / #3490
- **Base:** `origin/main` at the v0.14.2 release commit
- **Status:** Draft for the next release; deliberately not queued

## Completed and verified

- Desktop has the approved composer toggle, first-use copy, per-conversation
  state, attachment boundary, cancellation, and profile-mismatch handling.
- `/v1/models/{id}` is the server-owned source of truth for the selected
  model's nullable `personal_intelligence_profile`; Desktop has no model
  allowlist and never switches or downloads another model.
- Qualification checks the public identity, backing repository identity, live
  parser, and expected harness. Parser opt-out/override and alias reuse fail
  closed.
- Q4 and Q8 MiniCPM artifacts have distinct, versioned, evidence-backed
  qualification records even though they share one harness. Q8 is a 16 GB
  candidate and does not inherit Q4's 8 GB recommendation.
- Codex review found and closed parser, backing-weight, cross-quant, and
  repository-evidence containment gaps; the last review had no finding.
- Relevant Python regression suite: 387 passed. Qualification suite after the
  receipt refactor: 107 passed. Affected Swift suites: 72 passed. Ruff,
  accessibility identifier gate, and `git diff --check` passed.

## Unresolved product blocker

The GUI currently creates `execution: server` runs. With no user-configured MCP,
the server exposes only calculator helpers; it does not expose Desktop's
existing web search, browse, weather, memory context, or native tool executor.
That does not fulfill the first-use promise to use the Mac's tools and local
context, so #3490 must remain Draft.

## Next concrete action

Implement the thin Desktop client-tool adapter using the existing
`execution: client` protocol:

1. Define a server-owned allowlist/schema for the official Desktop built-ins;
   clients select names but cannot submit arbitrary schemas or risk labels.
2. Have `AgentSessionController` execute `awaiting_tool_result` calls through
   the existing `NativeToolCallExecutor` and permission stores, then submit the
   typed result to the server.
3. Pass a bounded snapshot of existing custom instructions and MemoryStore
   context without persisting it in public agent events.
4. Run physical-Mac GUI dogfood with MiniCPM Q4 before changing the 8/16 GB
   recommendation catalog or marking the PR ready.

## Evidence

- `docs/engineering/performance/2026-09-13-minicpm5-small-agent-harness-ab.md`
- `docs/engineering/decisions/2026-09-13-rapid-agent-runtime.md`
- `docs/guides/agent-runtime.md`
