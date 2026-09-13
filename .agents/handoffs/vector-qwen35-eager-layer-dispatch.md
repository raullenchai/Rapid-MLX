# Vector handoff: Qwen3.5/3.6 eager layer dispatch

## Scope and outcome

The scoped implementation submits narrow, qualified Qwen3.5-family 35B-A3B
decoder-layer outputs with `mx.async_eval`. Ordinary Qwen3.6-35B-A3B decoding
improved 84.32 to 88.05 tok/s (+4.42%) in same-load interleaved A/B, with all
token streams exact. Other family shapes fail closed to stock scheduling.
Native and continuous MTP were neutral and non-regressing; those gains are not
combined in the public claim.

## Reference check

- Primary GPU-serving precedents do not have an analogous model-layer
  submission mechanism for this lazy execution runtime.
- One MLX server precedent pipelines whole lazy decode steps and explicitly
  submits pending token/cache arrays. That is a promising separate scheduler-
  level direction, but materially larger than this model-layer PR and should
  be tested as its own spike.
- The upstream model runtime submits selected generation outputs, not
  individual Qwen3.5 decoder layers.
- A unified MLX runtime prototype supplied the model-layer precedent and the
  same-boot measurement lesson. Rapid adapted only the narrow scheduling idea;
  its own mixed-workload and server gates determine the shipping boundary.

## Rejected/deferred items from the same review

- The QSA/NAX kernel prototype failed Rapid's non-zero dense-oracle comparison
  despite passing its compile probe. Do not port without a corrected numerical
  contract.
- APC v2 is currently useful architectural research, but its required-layer
  restore remains whole-entry fail-closed rather than arbitrary segment
  recomposition. Porting it now would duplicate Rapid's mature prefix cache
  without demonstrated user benefit.
- ANE verifier offload is greedy-only, cannot preserve general logits APIs, and
  requires approximate commit to use its quantized head actively. Keep it an
  experimental shadow-mode candidate, not a default backend.
- Static B4 batching is narrower than Rapid's dynamic continuous scheduler.
  The reusable lesson is lifecycle/fairness instrumentation, not the scheduler
  replacement.

## Closed follow-up exploration

A scheduler-level one-step-lookahead spike found no independent overlap to
capture: the generation batch already keeps the current token lazy while it
constructs the next forward pass, explicitly submits the next token and cache,
and only then materializes the current token. A second pending-job state
machine would duplicate that dependency chain while adding cancellation,
grammar/logit-processor, MTP transaction, and cache-ownership risk. Do not
productize it without a new trace showing a real idle gap under both B=1 and
B=4 quality gates.
