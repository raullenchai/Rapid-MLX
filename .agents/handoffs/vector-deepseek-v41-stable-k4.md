# Vector handoff: DeepSeek V4.1 fixed-K4 stability

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GiB unified memory)
- Branch: `vector/deepseek-v41-stable-k4`
- Base: PR #3313
- Worktree: `/private/tmp/rapid-mlx-deepseek-v41-stable-k4`
- Status: implementation and validation in progress

## Scope and contract

This task resolves the long-output qualification question without changing
server, catalog, GUI, artifact download, or default inference behavior. Fixed
K4 is target-authoritative and deterministic for a fixed execution shape. It
does not promise bitwise identity with sequential AR, which uses a different
target batch shape.

## Evidence

- Four prompt domains, 128 output tokens, and two consecutive repeats produce
  identical K4 token streams for 4/4 prompts.
- The sustained A/B run reaches 10.72 weighted tok/s versus 9.16 tok/s AR with
  the same direct-QMV target kernels (+17.1%). An isolated two-repeat K4 run
  reaches 11.76 tok/s.
- Domain results are uneven: code +74.1%, structured +13.6%, Chinese +34.1%,
  and low-acceptance reasoning -17.7%. Fixed K4 is therefore not safe as an
  unconditional global default.
- Peak MLX memory is 218.07 GB and does not grow on the second repeat.
- The layer-zero trace finds the first 5.96e-8 batch-shape difference in
  hyper-connection mixing, followed by 0.001953125 at the quantized query
  projection and 4.7786 at final logits. This is target batch numerics, not an
  unverified draft-token escape.

## Private reference check

vLLM and SGLang were checked first for target-authoritative speculative
verification and fallback boundaries. oMLX was then checked for MLX-specific
DSpark and adaptive-acceptance precedent. The adopted boundary keeps the target
authoritative and treats poor draft acceptance as an admission/fallback
problem. Serializing all target token-local matrix operations was rejected
because it removes the batch work that creates the speedup.

## Next action

After this scoped stability PR, pin and validate the artifact/runtime delivery
chain with unified-memory admission. A later measured controller change may
fall back to AR when acceptance cannot repay verification overhead; do not add
an unmeasured heuristic to this PR.
