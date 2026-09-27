# Agent-session prefix reuse on 16-32 GB Macs

Date: 2026-09-27  
Owner: Vector (measurements) / Atlas (policy)  
Status: Accepted

## Context

A Claude-Code-like session (about 23k tokens of system prompt plus 30 tool
schemas, then 10 turns) on an M3 Pro 18 GB Mac running Qwen3.5-9B-4bit
re-prefilled the whole prompt on every turn, at about 70 s TTFT per turn.
oMLX, Ollama and mlx-lm reused the prefix at 0.4-0.9 s. About half of the
installs we see have 32 GB or less.

The server logs showed four separate failures:

1. **The budget was smaller than one session.** `MemoryCacheConfig` sized the
   prefix cache at 20% of `psutil.virtual_memory().available`, measured when
   the cache was built. At that point the weights are already resident and the
   desktop apps are running, so the budget came out at 0.81-1.2 GB. One
   23k-token entry for this model is about 958 MiB:
   - 8 full-attention layers × 2 (K and V) × 4 KV heads × 256 × 2 bytes =
     32 KiB per token, stored in bf16. That is about 712 MiB at 22.8k tokens.
   - 24 GatedDeltaNet recurrent states of about 2 MiB each. The entry holds
     the live state plus up to 4 hybrid checkpoints, about 250 MiB in total.

   Any entry larger than the whole budget is dropped at store time
   (`Cache entry too large`).
2. **Cache-self pressure evicted the only entry.** On a boot where the budget
   came out at about 1.1 GB, the entry fit. Then the R6-H6 cache-self trigger
   evicted down to 90% of the cache's own budget. That trigger fires on every
   engine tick, independent of Metal pressure, so it removed the entry right
   after the store.
3. **Hybrid requests stored two entries.** Each request stored two
   non-trimmable entries of about 1 GB each: the message-boundary snapshot and
   the N-token prompt. The next turn can only extend the boundary entry,
   because the history re-renders the generation prompt differently. A
   recurrent-state exact hit cannot trim one token, so the N-token entry is
   useless. Storing it second also LRU-evicted the boundary entry.
4. **The shutdown save skipped the entry.** The shutdown flush predicts each
   write before starting it. With no sample yet, it assumed 150 MB/s, which
   predicts 6.4 s for a 1 GB entry. It skipped the entry under the 3.5 s
   SIGTERM budget, even though the SSD writes it in about 1 s.

PFlash is not involved. Requests that carry tools skip compression
(`skip_when_tools=True`), and the logs show no `[pflash]` line for any agent
turn.

## Decision

- **Session floor for the budget.** The heuristic budget becomes
  `max(percent × available, floor)`:
  - The floor is `min((Metal cap − resident weights) / 3, 4 GiB)`
    (`memory_cache.session_floor_bytes`). The remaining two thirds of the
    headroom stay free for live KV and activations.
  - The floor applies only when a Metal cap is configured. The engine's
    default auto cap counts.
  - An explicit `--cache-memory-mb` or `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` is
    never raised.
  - The 4 GiB cap means large-RAM hosts keep their existing budget.
- **Cache-self pressure keeps the newest entry.** The cache-self trigger still
  trims older entries, but never the most recently used one. Metal-active
  pressure may still evict every entry.
- **Admission reclaims the cache before rejecting.** When a new request would
  cross the Metal cap, the D-METAL-CAP admission gate evicts memory-aware
  prefix-cache entries in LRU order before returning 503. A larger cache
  therefore never turns into backpressure.
- **Hybrid prompts store one entry per turn.** A non-trimmable cache that has
  already stored its message-boundary snapshot skips the N-token prompt entry.
  Trimmable caches are unchanged.
- **Shutdown saves use measured throughput.** A budgeted (shutdown) save
  measures a 16 MiB fsynced write and halves the result to cover serialization
  overhead. It uses that throughput for the first prediction when it beats
  the 150 MB/s floor. The 3.5 s SIGTERM budget is unchanged.

## Alternatives considered

- **Raise `--cache-memory-percent`.** The budget would still track whatever
  RAM happens to be free at boot, which is 0.8-1.2 GB on the same machine
  across boots. It would also grow the cache on large hosts, where R6-H6
  already bounded it.
- **Store prefix entries with quantized KV by default.** The engine supports
  8-bit storage, which halves the KV part of an entry. It is lossy on every
  cache hit for every model, so it stays opt-in until its quality impact is
  measured.
- **Admit single entries larger than the budget.** That is unbounded on long
  contexts and replaces a sizing problem with a safety problem.

## Memory safety

The floor sits inside the existing Metal allocation cap and does not raise it.

On the measured 18 GB Mac:

- The cap is 11.6 GB and the resident weights are 5.2 GB.
- That makes the floor 2.1 GB.
- A cold 23k-token prefill peaks at about 3.5 GB above the weights.

On a 16 GB Mac:

- The working set is about 10.7 GB, so the cap is about 9.6 GB.
- The floor is 1.5 GB, which still holds one 23k-token entry.
- Weights + a full cache + a cold prefill come to about 10.2 GB, which is
  above the cap.

Three guards cover that case:

- The Metal-pressure evictor fires at 90% of the cap.
- The admission gate now evicts cache entries before a request is admitted.
- The per-request KV projection still rejects requests that cannot fit.

No guard was removed.

## Consequences

Turn N+1 of an agent session reuses turn N's boundary entry on 16-32 GB Macs.
Measured before/after numbers are in the PR that introduced this decision.

Restart persistence now completes for a realistic session. The newest
boundary entries are written within the SIGTERM budget and reloaded on start,
so a client that resumes the same conversation extends them.

Replaying a session from turn 1 after a restart is still a cold prefill for
turn 1. That turn needs the shorter turn-1 prefix, and loaded entries carry no
hybrid checkpoints. The follow-up is to persist hybrid state checkpoints with
the entry, so a restarted turn 1 can snap to the newest checkpoint below the
shared prefix. The other option is a separate system+tools boundary snapshot.
