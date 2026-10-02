# Native System One server boundary

Date: 2026-09-24  
Owner: Atlas  
Status: Accepted for implementation

## Context

Jev-compatible decision models use a typed `POST /v1/systemone` contract, but
their runtimes differ. Laya has a small native MLX implementation. CLM-8B
ships a CUDA/vLLM reference server whose projection head is a PyTorch pickle,
although its Qwen3-8B backbone and projection layers can run on MLX.

The first product scope is server-only. Desktop and GUI behavior remain
undecided.

## Decision

- Expose a dedicated `rapid-mlx system-one` process with `/v1/systemone`,
  `/v1/rank`, `/v1/models`, and `/health`.
- Keep it separate from the generative `rapid-mlx serve` lifecycle. A decision
  model is an encoder/classifier and must not occupy chat model routes or
  advertise generation capabilities.
- Integrate Laya through the optional Apache-2.0 `laya-mlx` runtime.
- Run CLM's Qwen3 encoder and projection heads natively in MLX. Convert the
  upstream `.pt` head once with safe tensor-only loading; production serving
  reads safetensors and does not depend on PyTorch or vLLM.
- Preserve CLM's state/action split and cache projected vectors independently.
- Use Rapid-MLX bearer authentication, body-size limits, and JSON-depth limits.

## Consequences

Laya is ready for low-memory Apple Silicon serving. CLM requires the original
BF16 Qwen3-8B encoder until quantized parity is measured, and its converted
head is an explicit operator input until an upstream safetensors artifact is
published. The API can later support other compatible decision backends
without changing clients.

No GUI surface or automatic Desktop model management is introduced by this
decision.
