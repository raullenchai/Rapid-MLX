# Atlas handoff — Personal Intelligence productization

- **Owner:** Atlas
- **Branch / PR:** `feat/personal-intelligence` / #3490
- **Base:** `origin/main` at the v0.14.2 release commit
- **Status:** Qualification-complete; two unstable builds deliberately held out

## Completed and verified

- Desktop has the approved composer toggle, first-use copy, per-conversation
  state, attachment boundary, cancellation, and profile-mismatch handling.
- `/v1/models/{id}` is the server-owned source of truth for the selected
  model's nullable `personal_intelligence_profile`; Desktop has no model
  allowlist and never switches or downloads another model.
- Qualification checks the public identity, backing repository identity, live
  parser, and expected harness. Parser opt-out/override and alias reuse fail
  closed.
- MiniCPM Q4 has a versioned, evidence-backed qualification record; MiniCPM Q8
  is a 16 GB candidate and must not inherit Q4's 8 GB recommendation or Q4's
  receipt. Q8 remains held out until it is requalified on a committed revision.
- Desktop now executes the three existing read-only built-ins (`web_search`,
  `browse`, `weather`) for client-owned Agent runs and returns results under the
  exact opaque call ID. Schemas and risk labels remain server-owned.
- Custom instructions, MemoryStore, and the last eight completed chat messages
  are available as bounded transient context and never enter public Agent
  events.
- Intent routing and deterministic search-to-browse staging reduce the visible
  tool surface and remove mechanical argument generation from small models.
- Live 15/15 receipts now qualify Qwen3.5 4B Q8, Qwen3.5 9B Q8, Qwen3.6 35B Q4,
  Qwen3.6 27B Q4, all four Qwen3.8 27B variants, Bonsai 27B, Ling 3.0 Tiny,
  in addition to the prior support set.
- Codex review found and closed parser, backing-weight, cross-quant,
  repository-evidence, explicit no-network, pagination, multi-source, and
  ranked-URL selection gaps.
- Current affected Python suite: 233 passed. Ruff format/lint and
  `git diff --check` passed.

## Remaining qualification work

The September usage-table union is complete. Qwen3-Coder 30B reached 13/15
after the contradictory-URL gate, GPT-OSS 20B reached 14/15, and both are
deliberately disabled; their failure receipts are committed, but no
qualification record was added.

## Next concrete action

Review the evidence-backed enablement and hold the two unstable builds out.
Retry Qwen3-Coder 30B or GPT-OSS 20B only after a model/serving behavior
change, with a fresh canonical 15/15 matrix and exact identity checks.

## Evidence

- `docs/engineering/performance/2026-09-13-minicpm5-small-agent-harness-ab.md`
- `docs/engineering/decisions/2026-09-13-rapid-agent-runtime.md`
- `docs/guides/agent-runtime.md`
- `docs/engineering/performance/2026-09-15-personal-intelligence-top-model-qualification.md`
- `docs/engineering/performance/2026-09-16-personal-intelligence-remaining-builds-qualification.md`
