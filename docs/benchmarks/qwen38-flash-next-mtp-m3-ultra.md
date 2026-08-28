# Qwen3.8 Flash-Next native MTP on M3 Ultra

This follow-up measures the default-off native MTP path for
`rapid-mlx/Qwen3.8-Flash-Next-4bit`. The comparison uses the same candidate
engine, immutable checkpoint, prompts, process isolation, and benchmark script
on both sides; only the MTP configuration changes.

> **Hardware boundary:** the measurements were made on a 256 GB M3 Ultra.
> **128 GB hardware was not physically tested.** The q4 weights are roughly
> 99 GB before the MTP head, allocator, and context-cache headroom. A 192 GB
> machine remains the practical recommended tier; 128 GB is tight and
> unverified.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`) |
| Chip | Apple M3 Ultra, 28 CPU cores |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Architecture | arm64 |
| Python | 3.12.14 |
| Rapid-MLX implementation | `279426b0` |
| MLX | 0.32.2 |
| MLX-LM | 0.31.3 |
| Transformers | 5.12.1 |
| Artifact | `rapid-mlx/Qwen3.8-Flash-Next-4bit` at `dcf657e4acda2aae72da99cde65b6c491cd96998` |
| Quantization | PLE q4-g32; routing gates q8-g64; remainder q4-g64 |
| MTP | Native one-layer head from the same immutable checkpoint; fixed K=1 for measurement |

No other model server was resident during either run. Each variant used a
fresh server process. Prefix cache was cleared before every timed request.

## Exact commands

The baseline and MTP servers ran from the same worktree and environment. The
placeholder below is the immutable local snapshot for the artifact revision in
the environment table.

```bash
export SNAPSHOT=/path/to/dcf657e4acda2aae72da99cde65b6c491cd96998

# Baseline
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" python3.12 -m vllm_mlx.cli serve \
  "$SNAPSHOT" --host 127.0.0.1 --port 8465 --no-thinking

# MTP, in a fresh process after the baseline server exits
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" python3.12 -m vllm_mlx.cli serve \
  "$SNAPSHOT" --host 127.0.0.1 --port 8465 --no-thinking \
  --speculative-config '{"method":"mtp","disable_auto_k":true}'
```

The same command was used for each benchmark variant, changing only the label,
server PID, and output path:

```bash
python3.12 .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8465/v1 \
  --model "$SNAPSHOT" \
  --tokenizer-path "$SNAPSHOT" \
  --server-pid SERVER_PID \
  --label VARIANT_LABEL \
  --rapid-sha 279426b0 \
  --artifact-revision dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --output OUTPUT.json
```

The MTP correctness run used the repository's 45-case Flash-Next battery:

```bash
python3.12 .orca/flash-next-eval/run_eval.py \
  --base-url http://127.0.0.1:8465 \
  --model "$SNAPSHOT" \
  --tokenizer-path "$SNAPSHOT" \
  --allow-project-exec \
  --output /private/tmp/rapid-qwen4-mtp-final-correctness.jsonl
```

## Methodology

- Batch size is one; thinking is disabled and temperature is zero.
- Prompt targets are 128, 2,048, 8,192, and 32,768 tokens. The server reports
  92, 2,012, 8,156, and 32,732 tokens after chat templating.
- Every request asks for 256 decode tokens. Each prompt length has three runs;
  the table reports median TTFT and rates plus maximum sampled RSS.
- TTFT is the first visible SSE content, reasoning, or tool delta. Prefill rate
  is reported prompt tokens divided by TTFT. Decode rate excludes TTFT.
- RSS is process memory and does not account for all Metal/unified-memory
  allocations. MLX active memory is taken from the serving engine's allocator
  telemetry and is the relevant sizing measurement.
- The benchmark pins K=1 and disables adaptive depth so it measures the native
  MTP path on every eligible step. Normal opt-in serving leaves the expected-
  value controller enabled, so it can park speculation when a workload does
  not benefit.

## Results

| Target (reported) prompt tokens | Baseline TTFT | MTP TTFT | TTFT delta | Baseline prefill | MTP prefill | Baseline decode | MTP decode | Decode speedup | Baseline / MTP peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.393 s | 0.381 s | -3.0% | 234.2 tok/s | 241.3 tok/s | 25.17 tok/s | **34.85 tok/s** | **1.38x** | 54.54 / 56.43 GiB |
| 2,048 (2,012) | 3.346 s | 3.515 s | +5.1% | 601.4 tok/s | 572.4 tok/s | 23.64 tok/s | **33.53 tok/s** | **1.42x** | 54.82 / 56.45 GiB |
| 8,192 (8,156) | 13.643 s | 14.581 s | +6.9% | 597.8 tok/s | 559.4 tok/s | 22.82 tok/s | **32.20 tok/s** | **1.41x** | 54.72 / 56.47 GiB |
| 32,768 (32,732) | 62.844 s | 67.707 s | +7.7% | 520.8 tok/s | 483.4 tok/s | 21.16 tok/s | **28.82 tok/s** | **1.36x** | 54.74 / 56.54 GiB |

The baseline used 103.1 GB MLX active memory at the short prompts and 104.7 GB
at 32K. The MTP process used approximately 107 GB at the shorter prompts,
107.2--107.5 GB at 8K, and 109.2--111.3 GB across the three 32K runs. Its
largest observed active footprint was therefore 6.6 GB above the corresponding
baseline. Both processes reported a 148.1 GB allocator peak inherited from
model loading; that historical peak is not the steady active footprint.

## Correctness and qualification gate

The MTP run completed all 45 cases covering English and Chinese, math,
structured JSON, OpenAI and Anthropic tool calls, code generation, an
executable CLI todo project, multi-turn/system behavior, stop sequences, and
8K/32K needle recall. All 45 functional outcomes matched the ordinary-decode
baseline. The raw scorer marked 43/45 because of two pre-existing adjudication
artifacts shared by the baseline: one probability answer differs from the
fixture's expected value, and one correct lowercase-before-return function does
not match the fixture's order-sensitive regular expression.

Across the correctness battery and four-length benchmark, fixed K=1 recorded
1,844 proposals and 1,409 accepts, an aggregate **76.41% accept ratio**. The
release-qualification floor for this exact family and workload is 70% after at
least 256 proposals. That is a reproducible performance gate, not a universal
runtime cutoff: acceptance is workload-dependent, and normal serving uses the
expected-value controller's observed committed-token and round-cost signals.

Every proposal is checked by the target model before it is emitted, and a
rejection restores the coupled GDN, PLE, QSA, and KV state to one atomic token
boundary. Multi-token target verification can nevertheless choose a different
temperature-zero token at a near-tied logit because its floating-point
accumulation order differs from serial one-token decode. The real-checkpoint
outputs were therefore functionally equivalent but not promised byte-identical
to the baseline. MTP remains default-off and activates only through an explicit
`--speculative-config` request.

## Interpretation

Native MTP is a decode optimization. It raises median sustained generation by
36--42% over the same optimized engine and checkpoint. It does not improve
long-prompt prefill: at 2K--32K it adds 5--8% to TTFT, and it adds several GB of
unified-memory pressure. Users whose workload is dominated by long prefill and
short answers should leave it off; coding agents and other workloads that emit
long responses are the stronger fit.

The production path is deliberately limited to the checkpoint's single native
draft layer. An explicit deeper request fails at startup instead of silently
running a different depth. Failed tensor validation also fails attachment
rather than serving with partially loaded draft weights.

Two candidate optimizations were rejected during profiling: splitting the QSA
verify block reduced throughput, and a custom short-row quantized matrix-vector
kernel regressed decode. Neither is included in the implementation.
