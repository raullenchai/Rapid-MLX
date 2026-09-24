# System One server handoff

Receiving role: Vector

Owner: Atlas

Host: Studio

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
  state/action caching, artifact validation and rollback, CLI parsing, and
  optional dependency packaging without downloading model weights. The current
  focused suite has 53 passing tests; the last no-MLX simulation passed 23 with
  16 MLX-specific skips, and changed production lines had 100% local coverage.
- Real cached Laya weights passed CPU server dogfood. Typed answers matched a
  direct `laya-mlx` call, restart-to-ready was about one second with warm file
  pages, and warm latency was about 52–56 ms for one question and 170 ms for
  three. The reproducible screen is recorded under `docs/engineering/performance/`.
- The official CLM-v0.1 head and BF16 Qwen3-8B encoder passed native CPU server
  dogfood. Converted MLX head projections matched the upstream PyTorch head to
  at most 1.49e-7 absolute error for identical hidden states. Cold cache misses
  took 1.0–5.5 seconds in the sampled requests; exact cache hits were below one
  millisecond. The run found and fixed top-level `clm-latest` CLI rejection and
  ignored CLM `--device cpu` selection.

## Unresolved questions and risks

- CLM still needs an end-to-end BF16 comparison against the official vLLM
  pooling server using the same states/actions. Quantized Qwen3 encoders are
  rejected until measured.
- The released zero-shot CLM head was confidently wrong on some simple routing
  and software-remediation cases. Release notes should call CLM support
  experimental and avoid broad quality or calibration claims.
- The initial CLM encoder evaluates cache misses one text at a time for exact
  padding-free last-token semantics. Vector should measure grouped prefill and
  only batch it if output parity holds.

## Next concrete action

On an NVIDIA host, run the fixed corpus through the official vLLM pooling
server and this native MLX BF16 path, then record end-to-end probability and
ranking drift. Separately profile grouped prefill on Apple Silicon before
changing the serial padding-free cache-miss path.
