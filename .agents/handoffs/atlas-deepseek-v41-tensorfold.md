# Atlas -> Vector: DeepSeek V4.1 TensorFold feasibility

Branch: `atlas/deepseek-v41-tensorfold-mvp` (based on `origin/main` at
`e8e582022`). Owner: Atlas; performance follow-up: Vector. Host: Studio.

Verified: Rapid already serves V4.1 Flash in an experimental 256 GiB native
profile, with pinned local REAP 2-bit target and DSpark sidecar. Its 42 focused
artifact tests pass. Installed TensorFold 0.5.0 rejects that checkpoint at
`detect()` because no `deepseek_v41` family exists. Upstream `main` at
`609ca419` has a V4 Flash Metal family only; issue #299 and PR #300 are CUDA
V4.1 work on two DGX Sparks. In issue #299, `gilby` also reports a separate
Mac MLX V4.1 prototype: 35.1-38.1 tokens/s code
and 26.3-28.0 prose with DSpark on an M3 Ultra 512 GB, using an oQ4e checkpoint
that needs 287 GiB resident weights. The family, weight format, and
attention/cache path cannot be bridged by a profile alias. Full model
measurements were not run because the host had 13.1 GiB swap in use and other
large model processes.

Risk: treating the DGX Spark throughput numbers as a Mac speedup, or wiring the
V4 Flash family to V4.1, would produce a false qualification claim.

Next action: review the published Mac family source, then identify whether its
row kernels or Engram prefetch can be adapted to the cached 2-bit checkpoint
without changing outputs.
Use the existing four-prompt DSpark suite, the same cached target/sidecar,
alternating warm paired runs, exact-output checks, and a clean memory gate.
Only productize after a positive whole-model speed result.

2026-10-04 update: The four Mac branches are now public, open stacked PRs
ashhart/TensorFold #369-#372, based on 0.6.5. Inspect #372 as the complete
stack. The oQ4e checkpoint still needs >256 GiB, and no same-checkpoint Rapid
comparison exists. Upstream has not merged the stack.

2026-10-04 MVP update: ported TensorFold #372's concurrent Engram page-read
idea to Rapid's existing 2-bit mmap table, default off. Added a standalone
cached-shard I/O probe and a qualification-suite flag. Real shard row reads
improved from 3.621/3.156 s to 0.366/0.362 s in two alternating pairs;
identical-index arrays matched, and 54 focused tests pass. This is I/O-stage
evidence only. Full-model paired prefill/decode remains outstanding until the
Studio has clean memory pressure and zero used swap.
