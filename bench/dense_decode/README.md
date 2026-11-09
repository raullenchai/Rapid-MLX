# Dense decode ablations

Development experiments for singleton, greedy text decode on cached checkpoints.
These helpers run only in the benchmark process; they do not install a serving
optimization or change a runtime default. Graph ablations currently accept the
loaded Qwen3.5 and LFM2 language classes. `stock` and memory-budget arms can run
other models supported by the loader.

Use the same pinned snapshot, user-message file and output length for each arm:

```bash
python -m bench.dense_decode.benchmark \
  --model /path/to/cached/snapshot --prompt-file /path/to/prompt.txt \
  --arm stock --reps 3 --tokens 256 > stock.jsonl
python -m bench.dense_decode.benchmark \
  --model /path/to/cached/snapshot --prompt-file /path/to/prompt.txt \
  --arm all --reps 3 --tokens 256 > candidate.jsonl
```

The tool acquires the existing shared GPU lock, including legacy `lockf`
compatibility and stale-inode protection. It waits for CPU ≤15%, GPU ≤12%, and
aggregate VM CPU ≤20% before loading and before each sample, rejects samples
contaminated at the post-check, and stops after eight rejected attempts. It
never stops a competing process. Checkpoint resolution is local-cache-only;
no model download or cache redirection is performed.

| Arm | Change |
|---|---|
| `stock` | Dependency graph and maximum recommended wired-memory limit |
| `gu` | Join singleton dense SwiGLU gate/up projections |
| `both` | Also join same-input QKV and GDN input projections |
| `resnorm` | Fuse singleton residual addition with the following RMS norm |
| `all` | Grouped projections plus the residual/RMS seam |
| `fullnorm`, `allfull` | Also fuse the seam into the next layer’s input norm, alone or with grouped projections |
| `compile` | Compile stateless dense MLP calls |
| `ladder1`, `ladder4` | Submit each layer, or each fourth layer, asynchronously |
| `wired0`, `wiredfit` | Disable wiring, or wire only the loaded active set |
| `cap128`, `cap512` | Set the command-buffer size budget before MLX initialization |
| `gdn` | Apply the existing guarded GDN input-projection helper |
| `gdnfull` | Also enroll the guarded, FP32-state fused recurrence for compatible dense Qwen geometry |

Each sample starts with fresh prompt caches and an excluded prefill. A short
warmup precedes all samples. The timed loop builds and submits the next token
before reading the previous token, then drains the final submission. It has no
HTTP, detokenizer, logits processors, or stream callbacks. The last submitted
step is included, so its absolute rate is not an exact subtraction from a
server's client-measured decode rate. The output always has the requested
length, including EOS; this is a throughput probe, not a conversational client.

JSON records contain the resolved checkpoint path, device/runtime metadata,
actual templated prompt length, prompt digest, timing boundaries, token digest,
allocator memory, and load gates. No prompt, generated text or token IDs are
printed. Check the token digest against stock for every arm. The three boundary
timers include backend waits and must not be presented as a CPU/GPU partition.
Use fresh processes and alternate arm order for a final qualification campaign.

The grouped-projection helper preserves original row views for wider calls and
keeps its temporary result only until all projection branches consume it. It is
an experimental inference-only mutation, with no speculative verification or
model-lifetime qualification. Do not call `install` repeatedly with different
arms in one process or turn this helper into a production loader patch.

The residual/RMS kernel retains the sum's dtype rounding and the dependency's
four-values-per-thread reduction and norm output casts. The implementation is
limited to singleton Qwen3.5/LFM2 decoder seams of width ≤4096. Its measured
accuracy evidence is checkpoint/chip-specific; it is not a general bitwise
compatibility guarantee. See [LICENSE-RMS](LICENSE-RMS) for the reduction notice.
No captured GPU dispatch replay is included in this tool.

The full-seam variants cache the following norm's computed value by array identity
inside a serialized forward. Concurrent calls, cancellation, training and
pending-output teardown are unqualified; they remain benchmark experiments.
