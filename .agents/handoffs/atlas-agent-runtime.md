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
- Desktop now executes the three existing read-only built-ins (`web_search`,
  `browse`, `weather`) for client-owned Agent runs and returns results under the
  exact opaque call ID. Schemas and risk labels remain server-owned.
- Custom instructions, MemoryStore, and the last eight completed chat messages
  are available as bounded transient context and never enter public Agent
  events.
- Intent routing and deterministic search-to-browse staging reduce the visible
  tool surface and remove mechanical argument generation from small models.
- Live 15/15 receipts now qualify Qwen3.5 4B Q4, Qwen3.5 9B Q4, Qwen3.6 35B
  A3B Q8, and LFM2.5 1.2B Q4 in addition to MiniCPM.
- Codex review found and closed parser, backing-weight, cross-quant,
  repository-evidence, explicit no-network, pagination, multi-source, and
  ranked-URL selection gaps.
- Current affected Python suite: 191 passed. Affected Swift suites: 90 passed.
  Ruff format/lint and `git diff --check` passed.

## Remaining qualification work

The Desktop execution/context blocker is closed. #3490 remains Draft because
the maintainer's usage-table union is the product support target and the exact
remaining builds have not all completed physical qualification.

## Next concrete action

Run the exact-build matrix in
`docs/engineering/performance/2026-09-15-personal-intelligence-top-model-qualification.md`.
Only add a `PersonalIntelligenceQualification` after a 15/15 JSON receipt.
Qwen3.6 35B Q4, the Qwen3.8 variants, Bonsai, Qwen3-Coder, Ling, Qwen3.6 27B,
GPT-OSS, and the Qwen3.5 Q8 builds remain pending.

## Evidence

- `docs/engineering/performance/2026-09-13-minicpm5-small-agent-harness-ab.md`
- `docs/engineering/decisions/2026-09-13-rapid-agent-runtime.md`
- `docs/guides/agent-runtime.md`
- `docs/engineering/performance/2026-09-15-personal-intelligence-top-model-qualification.md`
