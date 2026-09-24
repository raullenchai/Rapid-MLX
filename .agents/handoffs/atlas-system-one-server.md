# System One server handoff

Receiving role: Vector

Owner: Atlas

Host: Studio for implementation; idle Mac required for real-model qualification

Branch: `atlas/system-one-server`

PR: #3728

## Verified facts

- The server-only product boundary is implemented as `rapid-mlx system-one`;
  no Desktop or GUI code is changed.
- The service exposes TypeSafe-compatible `/v1/systemone`, `/v1/rank`,
  `/v1/models`, and `/health` routes with bearer auth and the existing request
  size/depth protections.
- Laya uses the Apache-2.0 `laya-mlx` 0.2 optional runtime.
- CLM uses Qwen3 last-token hidden states plus native MLX state/action heads.
  The serving process reads safetensors only; a separate converter loads the
  official `.pt` checkpoint with `weights_only=True`.
- Unit tests exercise wire validation, auth, ranking, CLM projection loading,
  state/action caching, CLI parsing, and optional dependency packaging without
  downloading model weights.

## Unresolved questions and risks

- Real Laya output has not yet been compared byte-for-byte with `laya-mlx`
  direct calls through an HTTP request because Studio GPU capacity is occupied.
- CLM needs a BF16 parity run against the official vLLM server using the same
  states/actions. Quantized Qwen3 encoders are rejected until measured.
- The initial CLM encoder evaluates cache misses one text at a time for exact
  padding-free last-token semantics. Vector should measure grouped prefill and
  only batch it if output parity holds.

## Next concrete action

On an idle Apple Silicon host, convert the official 75 MB CLM head, run a fixed
System One corpus through official CLM and Rapid-MLX BF16, and record maximum
probability/ranking drift plus cold/warm latency. Then benchmark Laya through
the HTTP service and attach the reproducible commands/results under
`docs/engineering/performance/`.
