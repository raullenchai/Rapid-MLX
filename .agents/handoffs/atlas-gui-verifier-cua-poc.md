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

Replace Qwen-generated coordinate alternatives with candidates compiled from
macOS Accessibility/DOM nodes, batch the verifier scores, and run a held-out
suite that labels planner, candidate-generation, verifier, executor, reflection,
and terminal-verification failures separately.

