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

- `SamplingConfig` carries persisted provenance so only an untouched implicit
  Desktop `1.1` maps to neutral `1.0` for the exact GLM TensorFold alias;
  explicitly selected `1.1`, other values, and ordinary aliases stay on the
  wire for normal server validation;
- the Desktop wire boundary omits its independently populated tool registry and
  tool choice only for the exact built-in TensorFold alias; ordinary aliases
  retain both, and externally supplied TensorFold tools still fail closed at
  the server;
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

- Python HTTP/adapter contracts cover the production `stream=true` body through
  the route, including content, terminal framing, usage, prompt-renderer
  thinking propagation, and negative penalty/template/tool/media/grammar cases.
- Swift URLProtocol coverage captures a production-shaped request with a
  non-empty tool registry and proves exact-alias omission, implicit-default
  normalization, explicit `1.1` preservation, and ordinary-alias behavior.
- Verified locally: 60 focused Python tests; 92 focused Swift tests across
  request-body, sampling, profile, and deterministic-wire suites; Ruff,
  compileall, diff checks, and the package test build.
- Independent exact-head review is required after the PR is opened. Do not
  queue from this lane.
