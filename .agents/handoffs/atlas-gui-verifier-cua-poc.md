# Atlas handoff: GUI verifier Computer Use POC

- Receiving role: Atlas
- Branch: `atlas/gui-verifier-cua-poc`
- Host: Studio
- Status: POC implemented and dogfooded; product integration remains open

## Verified facts

- Qwen3.5-9B 4bit can plan from a screenshot plus compact DOM context when
  Rapid strict JSON schema is enabled.
- GUI-Actor-Verifier-2B loads unmodified through mlx-vlm 0.7.2 on Apple Silicon.
- The verifier correctly separated oracle search/reviews targets from obvious
  wrong points, but misranked an Amazon carousel arrow above the correct product.
- The main end-to-end failures were Qwen planning, coordinate candidate
  generation, goal retention, and repeated recovery actions.
- GLM-5.3-Flash with low reasoning completed the same task both with the search
  bootstrap (4 model steps) and fully unguided (11 model steps). Unguided mean
  planning latency was 7.65 seconds versus Qwen's 38.68 seconds in the
  controlled run.
- GLM's four redundant search-field clicks exposed an action/state-contract
  failure: a screenshot cannot reliably report keyboard focus, `type` carries
  no semantic target, and reflection therefore reinforced an unnecessary retry.
- No cart, account, checkout, payment, or purchase action was executed.

## Risks

- GUI-Actor-Verifier is a local coordinate verifier, not a goal-level policy or
  safety model.
- Amazon's dynamic and sponsored layout makes results non-repeatable.
- Raw screenshots and temporary Chrome profiles stay under `/private/tmp` and
  must not be committed.
- Product claims require a repeated benchmark with stable fixtures and an
  independent result oracle.

## Next concrete action

Add atomic `fill(target_node_id, text)` and `submit(target_node_id)` actions and
record the focused Accessibility/DOM node after execution. Then replace raw
coordinate alternatives with Accessibility/DOM candidates, batch verifier
scores, and run a held-out suite that labels planner, candidate-generation,
verifier, executor, reflection, and terminal-verification failures separately.
