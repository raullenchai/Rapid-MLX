# Atlas -> Vector: DeepSeek V4.1 TensorFold feasibility

Branch: `atlas/deepseek-v41-tensorfold-mvp` (based on `origin/main` at
`e8e582022`). Owner: Atlas; performance follow-up: Vector. Host: Studio.

Verified: Rapid already serves V4.1 Flash in an experimental 256 GiB native
profile, with pinned local REAP 2-bit target and DSpark sidecar. Its 42 focused
artifact tests pass. Installed TensorFold 0.5.0 rejects that checkpoint at
`detect()` because no `deepseek_v41` family exists. Upstream `main` at
`609ca419` has a V4 Flash Metal family only; issue #299 and PR #300 are CUDA
V4.1 work on two DGX Sparks. In issue #299, `gilby` also reports a separate
Mac MLX V4.1 prototype on four unpublished branches: 35.1-38.1 tokens/s code
and 26.3-28.0 prose with DSpark on an M3 Ultra 512 GB, using an oQ4e checkpoint
that needs 287 GiB resident weights. No public family code or compatible
256 GiB checkpoint is available yet. The family, weight format, and
attention/cache path cannot be bridged by a profile alias. Full model
measurements were not run because the host had 13.1 GiB swap in use and other
large model processes.

Risk: treating the DGX Spark throughput numbers as a Mac speedup, or wiring the
V4 Flash family to V4.1, would produce a false qualification claim.

Next action: inspect the Mac family's source if its four branches become
public, then adapt it to the cached 2-bit checkpoint; or port one measured
TensorFold row/lane technique into Rapid's native V4.1 implementation.
Use the existing four-prompt DSpark suite, the same cached target/sidecar,
alternating warm paired runs, exact-output checks, and a clean memory gate.
Only productize after a positive whole-model speed result.
