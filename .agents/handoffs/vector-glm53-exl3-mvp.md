# Vector handoff: GLM-5.3-Flash-Next EXL3 MVP

Receiving role: Atlas
Branch: `vector/glm53-exl3-mvp`
Base: `origin/main@233439e88`
Status: experiment complete; EXL3 product integration is a no-go

## Verified facts

- A real tiny `glm5_next` checkpoint completed EXL3 Metal forward and cache
  generation after the KDA fusion compatibility guard.
- PonyExl3 direct K=4/K=2 did not beat MLX q4 end-to-end quality.
- A K=5-body/q4-head hybrid reduced mean KL by about 70%, but decode throughput
  fell from 92.9 to 48.7 tok/s on the tiny graph.
- Production-shape K=4 expert projection GEMVs were about 1.6x slower than MLX
  q4. Lower-bit experiments did not recover both quality and speed.
- PonyExl3 does not currently convert GLM's individual routed-expert tensor
  layout and requires Python 3.14.
- Full-model validation was impossible under the enforced Hugging Face cache
  quota; the full checkpoint was not downloaded or redirected.

Detailed evidence and reproduction constraints are in
`docs/engineering/performance/2026-09-11-glm53-exl3-mvp.md`.

## Recommendation and next action

Accept the small packed-linear compatibility guard if its focused and broader
tests remain green. Do not advertise or enable EXL3 for GLM. Re-open only after
the five gates in the performance note are met, beginning with upstream GLM
expert-layout support and a production-shape MoE kernel win.
