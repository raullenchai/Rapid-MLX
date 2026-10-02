# Vector handoff: GLM-5.3 TensorFold Desktop compatibility

- Owner: Vector, with Pixel coordination; Atlas retains release priority.
- Branch/worktree: `vector/glm53-desktop-compat` in
  `/private/tmp/harbor-desk-glm53-desktop-compat`.
- Base: exact main `d51ee1e0b21198a9ab5ada92842f95bb326f37c4`.
- Scope: make the built-in `glm5.3-flash-tensorfold` preset accept an ordinary
  Desktop chat turn without weakening TensorFold's unsupported-feature gates.
- Non-goals: TensorFold sampler implementation, tools/media/grammar support,
  catalog redesign, unrelated optimization, release metadata, or queue action.

## Verified cause and compatibility boundary

`ChatStreamClient` always serialized Desktop's general
`repetition_penalty=1.1`, neutral frequency/presence penalties, and the boolean
thinking switch. The TensorFold boundary rejected every present value, so a
first turn could fail before the asynchronous server profile replaced the
general repetition default with the profile's neutral `1.0` recommendation.

The fix keeps semantics explicit:

- only the exact GLM TensorFold alias maps Desktop's implicit `1.1` default to
  neutral `1.0`; other aliases and caller-selected nondefault values are
  untouched;
- the server accepts only penalty identities (`1.0`, `0.0`, `0.0`);
- the shared prompt renderer's qualified boolean `enable_thinking` switch is
  accepted, while other chat-template kwargs still fail closed;
- tools, media, response grammar, non-neutral penalties, and other unsupported
  request features continue to return HTTP 400.

## Reference-first check

Rapid's existing model-profile path already declares `repetition_penalty=1.0`
for this alias and applies recommendations without overwriting user intent. vLLM
and SGLang define `1.0` repetition plus `0.0` presence/frequency as the neutral
API defaults. MLX-LM and oMLX likewise treat those identities as disabled/no-op
processors. The adopted boundary therefore normalizes only implicit client
defaults and accepts only mathematically neutral values; it does not pretend
TensorFold implements unsupported logits processors.

## Verification and next action

- Python HTTP/adapter contracts cover the real post-normalization Desktop body,
  prompt-renderer thinking propagation, and negative penalty/template/tool/
  media/grammar cases.
- Swift URLProtocol coverage captures the body emitted by the production
  `ChatStreamClient.send` path and proves other aliases plus nondefault values
  remain unchanged.
- Independent exact-head review is required after the PR is opened. Do not
  queue from this lane.
