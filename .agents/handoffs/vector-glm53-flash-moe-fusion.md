# Vector handoff: GLM-5.3-Flash MLLM MoE fusion

- Owner: Vector
- Receiving owner: Atlas
- Branch: `vector/glm53-flash-perf`
- Worktree: `/private/tmp/CommunityBenchMark-glm53-perf`
- Base: `origin/main@cae3412e4e45b07ee6fbcb9426fb9dba64a2791b`

## Goal and scope

Compare Rapid's published GLM-5.3-Flash results with oMLX and mlx-vlm, then
recover a low-risk decode optimization that Rapid already applies to text MoE
models but omitted from the forced MLLM lane. No MTP enablement, quantizer
change, native-extension import, alias change, or other VLM enrollment belongs
in this task.

## Verified facts

- Rapid's checked-in official-checkpoint campaign reports 27.20-32.38 decode
  tok/s across 128-32K prompts. oMLX's separately published oQ4e campaign
  reports 23.5-24.1 tok/s decode and stronger long-prefill numbers, but the
  campaigns are not directly comparable.
- GLM-5.3 routes through `MLXMultimodalLM`, so it missed the existing post-load
  gate/up fusion. mlx-vlm also owns a distinct `SwitchGLU` class family, which
  the helper did not recognize.
- The change enrolls only `model_type == "glm5_next"`, discovers mlx-vlm only
  when already loaded, keeps the environment kill switch, and fails closed if
  upstream changes the verified `(self, x, indices)` call contract.
- A production-shape q4-g64 layer microbenchmark measured a 1.037x median paired
  speedup across five fresh processes (range 1.000x-1.119x), with byte-identical
  output in every run. This is not an end-to-end model claim.
- The 189 MB architecture fixture fused 2/2 sparse MoE layers and completed a
  real OpenAI-compatible server request.
- The official 181.7 GB checkpoint is not currently present in the fixed HF
  cache or warm tier. With only about 6 GB quota free, storage policy correctly
  prevented a new full-model download.

## Risks and next action

The rewrite is already qualified in the text lane, but the whole-model benefit
for GLM must be measured when the exact cached snapshot becomes available.
Atlas should review the architecture-scoped MLLM load hook and merge only after
the focused/full test gates and PR validation pass. The follow-up campaign and
stop criteria are recorded in
`docs/engineering/performance/2026-09-11-glm53-flash-runtime-comparison.md`.
