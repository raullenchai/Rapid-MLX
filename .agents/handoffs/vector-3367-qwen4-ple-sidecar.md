# Vector handoff: Qwen4 PLE file-backed sidecar (#3367)

- Owner: Vector
- Source PR: #3367
- Studio integration branch: `integration/pr-3367-studio`
- Studio worktree: `/Volumes/NVMe-4T/rapid-worktrees/pr3367`
- Immutable model revision: `rapid-mlx/Qwen3.8-Flash-Next-4bit@dcf657e4acda2aae72da99cde65b6c491cd96998`

## Scope

Port Pierre Lamy's opt-in Qwen4 q4/group32 PLE reader from the retired
`vllm_mlx` package path to current `rapid_mlx`, preserve attribution, and
measure its user-visible memory and latency tradeoff on real inference. This
work must not trigger or prepare a release.

## Correctness and usability fixes

The public checkpoint stores the source tensors under
`model.language_model...`, while the vendored model sanitizes them to
`language_model.model...`. The original PR required the runtime prefix in the
source index and rejected its target checkpoint. Validation now binds both
exact prefixes.

Hugging Face snapshots use symlinks into the same repository cache's sibling
`blobs/` directory. Validation now accepts that immutable layout while still
rejecting arbitrary paths outside the model repository.

The real checkpoint also has one trained RMSNorm anchor mean at 0.765 while the
other 47 anchors are in the zero-centered band. The detector now requires 95%
producer consensus plus a robust median band; the deliberately mixed contract
is still rejected.

A production builder is available as:

```sh
python -m rapid_mlx.models.qwen4_ple_build \
  --model /path/to/immutable/snapshot \
  --output /path/to/ple_rows.bin
```

It streams tensor chunks, verifies all source geometry, writes per-shard
SHA-256 values, fsyncs and atomically publishes through partial files, and
validates shard edges plus random rows before publication.

## Studio A/B

Host: Apple Silicon Mac Studio with 256 GiB unified memory. The host had
background load, so cold load and first-token timings were noisy. Each lane ran
in a separate process with offline loading, greedy sampling, a 64 MiB bounded
row cache, and the exact revision above. Token IDs were identical in every
paired run.

The source index SHA-256 is
`1ffe41e4484e7dc137900b12b14ae29cf92d1737e89fbe72c4fa2dd1b42ee7f1`.
The model index reports 104,681,488,408 tensor bytes. The sidecar contains 128
shards, 320,001,536 rows, 160 dimensions, and exactly 32,000,153,600 bytes.

After one warmup:

| Measured request | Resident PLE | File-backed PLE | Delta |
| --- | ---: | ---: | ---: |
| Peak MLX memory, short prompt | 102.978 GB | 70.975 GB | -32.003 GB |
| TTFT, short prompt / 64 output | 226.3 ms | 305.5 ms | +79.2 ms |
| Decode, short prompt | 25.97 tok/s | 25.54 tok/s | -1.65% |
| Peak MLX memory, 1024-token prompt | 106.025 GB | 74.022 GB | -32.003 GB |
| TTFT, 1024-token prompt / 32 output | 1.442 s | 1.487 s | +44.3 ms |
| Decode, 1024-token prompt | 27.31 tok/s | 26.59 tok/s | -2.64% |

Cold timings varied substantially in both directions because of background
load and page-cache state; they are retained in
`/Volumes/NVMe-4T/pr3367-artifacts/{on,off,on2,off2}.log` and are not used as
the speed claim.

The production builder recreated all 32,000,153,600 bytes in 25.04 seconds.
`cmp` confirmed that its output was byte-identical to the independently built
A/B artifact, and validation checked 512 rows spanning every shard edge plus
256 random samples.

## Verification

- 36 focused sidecar CPU/load/production-lane contracts pass.
- 13 Qwen4 norm-convention CPU contracts pass.
- Ruff check, Ruff format check, and `git diff --check` pass.
- Earlier Mac mini verification reached 114/115 existing Qwen4 tests; its only
  failure was an unchanged MTP test calling unavailable `mx.full_like` in
  that host's older MLX build.
