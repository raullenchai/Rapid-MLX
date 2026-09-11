# Vector handoff: product-owned DeepSeek V4.1 DSpark runtime

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GiB unified memory)
- Branch: `vector/deepseek-v41-native-dspark`
- Base: PR #3311 (`vector/deepseek-v41-dspark-adapter`)
- Worktree: `/private/tmp/rapid-mlx-deepseek-v41-native-dspark`
- Status: implementation and full-model dogfood complete; PR/review pending

## Scope

Replace the default benchmark dependency on checkpoint-bundled Python with a
minimal Rapid-owned DSpark runtime. Server/catalog/GUI integration, artifact
download, sampling, and resolution of the 128-token batch-numerics divergence
are explicit non-goals.

## Verified facts

- The owned loader reads data only and rejects path traversal, symlinked MTP
  shards, index/shard mismatch, incomplete three-stage tensor contracts, and
  allocations that violate host or Metal headroom.
- The target owns and shares the quantized embedding/head; the sidecar does not
  load their linked target shards.
- Actual-sidecar A/B against the explicitly trusted oracle is bit-exact: all six
  observed KV windows, proposal tokens, and five confidence logits match, with
  maximum absolute difference 0.
- The final full-model K4 run reaches 13.42 tok/s versus 7.82 target AR (+71.7%),
  with identical 32-token greedy output and 218.00 GB peak MLX memory.
- K5 reaches 14.61 tok/s but diverges and remains excluded.
- Adversarial dogfood caught and fixed two P1 integration defects: a provider
  class-name mismatch and 4.25 GB of retained unpacked expert tensors. The
  final pack releases all 3,456 replaced arrays and restores the 218 GB peak.
- Focused tests: 21 passed. Ruff and diff whitespace checks pass.

## Reference check (internal only)

- vLLM and SGLang use product-owned draft architectures with shared target I/O
  and explicit verification state rather than loading executable checkpoint
  code; this boundary was adopted.
- Current oMLX also owns its V4.1/MTP runtime, but its converted tensor namespace
  is incompatible with this 212.93 GB target, so no runtime or artifact was
  copied.
- Existing Rapid native RMS/RoPE/fake-quant/Hyper-Connection behavior was reused
  where shapes matched; draft-specific rank conventions remain isolated here.

## Next concrete action

Open the scoped stacked PR, run author-owned adversarial review and PR validation,
then address 128-token target batch numerical stability in a separate PR.
