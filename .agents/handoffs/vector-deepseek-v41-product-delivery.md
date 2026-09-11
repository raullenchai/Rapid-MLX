# DeepSeek V4.1 Flash product delivery

Owner: Vector

Branch: `vector/deepseek-v41-product-delivery`

## Outcome

- Moved the data-only V4.1 runtime into the shipped `vllm_mlx` package.
- Added a serial OpenAI-compatible K4 lane because the model-level compressed
  cache cannot satisfy the current scheduler's per-layer merge/extract ABI.
- Pinned target and MTP commits. The sidecar pull is narrowed to three MTP
  shards plus index/config and checks sizes plus SHA-256 before load.
- Added a hard 224 GiB catalog admission floor; 256 GiB is the supported SKU.
- Product-path dogfood: 300.81 s load, correct factual response, 217.97 GB peak.
  The first run exposed a literal EOS marker; fixed and regression-tested.
- Self-review also caught and closed three product-boundary failures: the
  generic 32,768-token server default exceeded this lane's 4,096-token cap;
  artifact-only CLI imports eagerly loaded the model package; and `info`
  advertised generic architecture/speculation guesses instead of DSpark K4.
- The offline size manifest reports 212,937,705,666 target bytes. Pinned,
  filtered disk admission now prices uncached target and sidecar files even
  when a partial snapshot already contains `config.json`.

## Reference check

- vLLM and SGLang speculative controllers confirm the target-authoritative
  accept/rollback contract, but their paged per-layer cache ABI does not map to
  V4.1's model-owned window/compressed/index/Engram state.
- The existing Rapid DFlash dedicated server supplied the proven serial
  deadline, cancellation, backpressure, auth, and admission boundary.
- MLX-native V4.1 implementations were checked for data layout and load
  behavior. Rapid retains a strict data-only loader and executes no checkpoint
  Python.

## Explicit limits

- No continuous batching or prefix-cache claim.
- Greedy only; input <= 8,192, output <= 4,096.
- No image, tools, MCP, or structured-output claim.
- K5 remains excluded; adaptive selection and new drafter training are future
  performance work, not part of this delivery PR.
