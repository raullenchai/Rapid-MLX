# Atlas handoff — GUI verifier → productized CUA

## State
- `rapid_mlx/cua/` product package LIVE: config presets (cloud-glm/local-27b/local-9b
  in ~/.rapid-mlx/cua-config.json), planner (strict schema + repair, injectable),
  fast path (laya + NoProgressTracker fixation gate), consent gates, agent loop,
  CLI (`rapid-mlx cua run|config|planners`). E2E verified with local-9b on
  Wikipedia (3 steps, grounded summary). Targeted CUA and tool-layer tests pass.

## Verified facts
- Qualification matrix (see decisions doc): 9B ok short/planner-only, fixates
  long-horizon; 27B premature-done risk; GLM still strongest long-horizon brain.
- Chrome active-tab AXValue is empty → URL guard fails closed when AX cannot
  expose the active URL; browser fallback requires the host app to have
  Automation permission.

## Next concrete actions
1. User grants Automation→Chrome to the host app; then verify domain guard bites
   (run with --allowed-domain wikipedia.org, navigate away, expect stop).
2. Try vision for small models (Qwen3-VL-8B in HF cache) as planner-vision path.
3. Port Amazon pre-ranking (shopping-fast-path) onto AX snapshots for the
   shopping flow under rapid_mlx.cua.
4. Optional: AXObserver eventing, right/middle+drag, per-action consent prompts.

## Risks
- Planner-only flows: validator rejects hallucinated files but NOT omissions
  (9B left installer.pkg unfiled) — add completeness check vs directory listing.
- 9B on flows ≥10 steps: only with the fixation gate; expect stalls on hard UI.
