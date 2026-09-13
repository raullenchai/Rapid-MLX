# Vector → Atlas: MiniCPM5 small-agent harness

- Owner: Vector
- Receiving owner: Atlas
- Branch: `vector/minicpm-harness-ab`
- Host: M2 Pro Mac mini, 32 GB
- Base: `origin/main@7a665912`

## Verified facts

- A deterministic 12-task, three-seed harness compares MiniCPM5-2B Q4,
  Qwen3.5-4B Q4, and Qwen3-VL-4B Q4 under raw and enhanced conditions.
  Model-written code is never executed; file paths stay inside a temporary
  workspace.
- Under strict completion criteria, MiniCPM Q4 improved from 19/36 to 25/36
  with a compact plan, task-scoped tools, and tool-result task state. Qwen3.5
  held 28/36 in both arms after strict rescoring; its effect-aware diagnostic
  score improved from 0.936 to 0.946.
- Enhanced MiniCPM averaged 4.39 seconds per task versus 11.71 seconds for
  enhanced Qwen.  The speed/footprint advantage is real, but the three-run
  reliability gap does not justify changing the shipping default yet.
- MiniCPM Q8 enhanced reached 23/36 and averaged 4.78 seconds, worse than Q4 in
  this sampled run, while its process peaked at 5.91 GB.  It is not an 8 GB
  recommendation or a current product priority.
- Qwen3-VL moved from 24/36 raw to 26/36 enhanced and averaged 15.34 seconds in
  the enhanced arm. It is not a faster or more reliable cognition core than
  MiniCPM; its differentiated value is multimodality.
- Qwen3-VL passed 9/9 ordinary GUI, chart, and receipt extraction runs, but
  followed an instruction embedded in the image in 3/3 adversarial trials.
  Default desktop-agent use requires an untrusted-vision boundary.
- A first prototype injected task state as a new user message and regressed
  coding/organization.  State attached to the latest tool result avoided that
  role-confusion failure.
- The existing Desktop runtime already owns the tool loop, a three-external-tool
  ceiling, context trimming, and durable user memory.  The missing product
  layers are task-scoped tool projection and concise per-turn task state; this
  belongs in the GUI/agent runtime, not the inference server.

## Risks and unresolved questions

- This is a synthetic M2 Pro 32 GB qualification, not an end-to-end physical
  8 GB GUI run.  It measures explicit creative constraints, not blinded prose
  preference.
- Search-source opening and multi-file evidence completeness remain the largest
  strict failures.  A planner alone does not enforce completion.
- The visual suite is deliberately narrow and synthetic. Qwen VL still needs
  real GUI-path, natural-image, small-font OCR, and physical 8/16 GB testing.
- Persisting active task state in the existing memory library would mix
  temporary work with durable user facts.  Atlas should reuse the storage
  substrate or patterns, but keep a separately typed task-ledger namespace and
  lifecycle.
- Planning roughly doubles MiniCPM task latency, so direct chat and obvious
  single-tool requests must bypass it.

## Next concrete action

Implement an opt-in MiniCPM product slice in a separate Atlas branch:
deterministic tool projection, compact task state attached to tool results,
bounded completion checks, and trace capture. In parallel product planning,
add Qwen3-VL-4B as an opt-in multimodal qualification candidate and design the
untrusted-vision boundary before enabling visual tool actions. Re-run the same
strict suites through the actual Desktop loop, then qualify on physical 8 GB
and 16 GB Macs before changing onboarding or default recommendations.
