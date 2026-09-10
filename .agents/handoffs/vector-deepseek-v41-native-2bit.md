# Vector handoff: DeepSeek V4.1 native 2-bit

- Owner: Vector
- Host: Studio (256 GiB unified memory)
- Branch: `raullenchai/deepseek-v41-native-2bit`
- Worktree: `/Users/raullenstudio/orca/workspaces/rapid-mlx/deepseek-v41-native-2bit`
- Status: qualification complete; product gate failed; no upload or exposure

## Intention and gates

Build a native packed affine 2-bit DeepSeek V4.1 Flash text checkpoint from the
already-cached Vontra checkpoint, then expose it through Rapid-MLX only if a
real 256 GiB sustained-greedy benchmark reaches at least 12 token/s with
acceptable parity/coherence. The optimization target is 20+ token/s.

Non-goals are MTP, visual input, cloud execution, downloading the raw 754B
checkpoint, and product exposure below the 12 token/s gate.

## Verified facts

- Source checkpoint: 238.796 GB indexed, affine 2-bit/group-64.
- Text-only native repack estimate: 234.183 GB without pruning.
- Routed experts occupy 174.116 GB; REAP12.5 would reduce the candidate to
  approximately 212.4 GB.
- Layer-0 real-weight repack is byte-preserving and loads all 21 MoE tensors.
  Against the source execution path, output cosine is 0.999988, max absolute
  difference 0.0625, and mean absolute difference 0.00708.
- Single-token layer-0 MoE compute does not improve from packing alone:
  1.81 ms packed versus 1.11 ms source direct-QMV after warmup. Full-runtime
  qualification is therefore mandatory.
- The REAP12.5 artifact contains 212,930,051,680 tensor bytes (~199 GiB on
  disk), strict-loads with zero missing/unexpected parameters, and peaks at
  213.51 GB MLX memory when resident.
- Correct V4.1 chat framing generates `The capital of France is Paris.` and
  EOS. Raw completion probes without the model's role tokens are invalid.
- Standard `(N-1)/elapsed` decode is 7.31 tok/s at evaluation interval 4 and
  7.92 tok/s in the one-graph short-decode ceiling. The latter is not a
  long-context safety claim.
- Gate/up fusion is rejected: it reaches only 7.87 fixed model steps/s and
  raises the load transient to 263.72 GB.
- The complete tiny-config parity suite passes: <=1.4e-6 relative differences,
  100% argmax agreement, bit-exact quant/dequant cases, strict float/quantized
  layouts, and bit-equal streaming.
- The qualification runtime lives under `scripts/`, outside the production
  model package and registry. The pinned Python 3.11 mypy budget and 16 focused
  tests pass. A local full-suite run reached 23,679 passes; its remaining four
  tokenizer errors and one missing optional image dependency were unrelated
  offline-environment gaps, not failures in this change.
- The 12 tok/s product floor is missed by 51.5%; the artifact was not uploaded
  and Rapid catalog/server/GUI support was not added.

## Reference check (internal only)

- Primary engine precedents were checked for reusable model/runtime support;
  neither currently exposes this architecture.
- MLX-native implementations were inspected next. The active native port at
  revision `8bdd543c7160800f3451d9595c996129b7abecdf` provides strict parity,
  shared compressed attention, Engram, and packed SwitchGLU. Its public builds
  require at least 275 GB, so Rapid adapts the runtime and reuses the cached
  2-bit tensors without re-quantization.
- A separate multimodal integration is draft-only and lacks a complete load and
  generation path; it is not a viable dependency for this task.

## Storage log

2026-09-10 — task `DeepSeek V4.1 native 2-bit qualification`; model
`DeepSeek-V4.1-Flash native packed affine 2-bit`; planned generated build
234.2 GB at
`/Volumes/RTL-2T/models-cold/local/DeepSeek-V4.1-Flash-native-2bit-unpruned`.
This is a local byte-preserving conversion, not a download. No Hugging Face
cache variables or alternate download directories are used.

2026-09-10 — same task; calibrated REAP12.5 candidate; expected generated
size 212.93 GB at
`/Volumes/RTL-2T/models-cold/local/DeepSeek-V4.1-Flash-native-2bit-reap12_5`.
The selection uses 2,048 real causal tokens and keeps an expert when either
disjoint corpus half assigns material normalized routing weight.

## Disposition

Keep this as a bounded, reproducible negative qualification. Do not productize
the model unless a future exact target-forward implementation independently
passes correctness, long-context stability, and at least 12 tok/s on the same
hardware class. More routed-expert count pruning or a storage-only fractional
BPW format does not reduce the six active expert widths per token and is not a
credible way to close the gap.
