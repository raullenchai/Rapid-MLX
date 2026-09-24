# Qwen3.8 MLLM fused-GDN — 32 GB qualification

## Decision

**NO-GO on Apple M2 Pro with 32 GiB unified memory.** The frozen experiment
harness's mandatory 32-step real-weight parity setup failed before any
measured prompt stratum ran. The single model-loading attempt increased system
swap use by 8,658.94 MiB (8.46 GiB). It was not rerun and no qualification gate
was relaxed.

This result provides no wall-time, decode-speed, media-recovery, or VLM
correctness conclusion: there are no measured strata. It only proves that no
candidate threadgroup passed the mandatory setup on this host under the
frozen runtime and that the attempt caused unacceptable memory pressure.

The sanitized durable record is
[2026-09-23-qwen38-mllm-fused-gdn-32gb-no-go.json](../../benchmarks/results/2026-09-23-qwen38-mllm-fused-gdn-32gb-no-go.json).

## Bound provenance

- Experiment commit: `1c1c6573ab0353aa94bfb2b04e18208d5ddf2466`
- Production canary source statically checked, before its B0 rebase:
  `e8e1ef6c4d9e6092d30b2a0c93b202b3a8243360`
- Exact artifact: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` revision
  `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`
- B0 verification ID:
  `hf-snapshot-sha256:360a8c76fc60c254c595442d16157eb76868033d328d5df9f41344c83bf77a66`
- Methodology SHA-256:
  `5eec2e6dbcd67e5b9ef2df856880eef468ea5394c7d43a3fbad62150394bb5b3`
- Kernel SHA-256:
  `73deff77202039597b77cb75cd4e91dd91dfc05575acdca53a1b2937031b2db8`
- Raw receipt SHA-256:
  `cd42c401eb7c6e0a37b24c12e1b3b71ca16a3a82b833371fda7603a87b5bef73`
- Runtime: mlx 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.1, rapid-mlx
  0.15.0; offline execution.

Artifact preflight verified the exact config, index, three canonical shard
identities, and zero missing shards. The raw benchmark receipt still records
`artifact.verified=false` because the setup exception occurred before the
benchmark populated its success receipt. These fields describe different
stages and must not be conflated.

Only the experiment harness loaded the model. Production canary validation was
static/read-only; no production engine was started and no production `active`
status was observed.

## Outcome and memory

- Host: Apple M2 Pro (`Mac14,12`), 32 GiB unified memory, macOS 26.5.2
  (`25F84`).
- Experiment-harness model-loading attempts: one; qualification reruns: zero.
- Exit: status 2 after 18.53 seconds, with
  `RuntimeError: 32-step real-weight fused GDN parity probe failed`.
- Aggregate and setup gates: false. Prompt strata run: zero.
- Swap used: 1,892.62 MiB before and 10,551.56 MiB after, a delta of
  8,658.94 MiB (8.46 GiB).
- Process maximum RSS: 3,510,812,672 bytes. Process peak footprint:
  16,588,009,712 bytes. These process counters do not cover all system-wide
  pressure; the swap delta is therefore part of the decision.

The preflight log reported the size of each shard *symlink*, not its resolved
target. This report intentionally makes no total checkpoint-byte claim.

## Diagnostic limit

The frozen probe tries threadgroup sizes 32, 16, 8, and 4. It collapses output
or cache mismatch, metadata mismatch, incomplete hit counts, and exceptions
into the same setup failure. The receipt therefore proves that no candidate
passed, but not which first comparison or exception caused the failure. An
instrumented diagnosis would require another model run and was deliberately
not attempted after the swap increase.

## Product consequence

Keep the production canary default off on 32 GB hosts. This result does not
qualify 48 GB or 64 GB hosts and does not justify changing an alias or public
default. Those hardware classes still require independent qualification.
