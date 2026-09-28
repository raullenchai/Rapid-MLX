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
  - An explicit `--cache-memory-percent`, `--cache-memory-mb` or
    `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` is never raised. An explicit percent
    below the floor is kept, and one warning names the floor.
  - The 4 GiB cap means large-RAM hosts keep their existing budget.
- **Pressure below the Metal cap keeps the newest entry.** The cache-self
  trigger and the soft Metal zone (90% of the cap up to the cap) still trim
  older entries, but never the most recently used one. In the 16 GB-class run,
  a hit turn's transient state (live KV plus the snapshot copy) peaked at
  9.3 GB against a 9.6 GB cap. Evicting there dropped the entry on every hit
  turn. At or over the cap itself, every entry may still be evicted.
- **Admission reclaims the cache before rejecting.** When a new request would
  cross the Metal cap, the D-METAL-CAP admission gate evicts memory-aware
  prefix-cache entries in LRU order before returning 503. A larger cache
  therefore never turns into backpressure.
- **The message-boundary entry is the one a hybrid turn keeps.** This applies
  to non-trimmable caches once the request has stored its message-boundary
  snapshot. Trimmable caches are unchanged.
  - The N-token prompt entry is skipped when it would add at most 64 tokens of
    reuse over the boundary entry. That is the normal case. A fallback
    boundary further back keeps both entries.
  - The prompt + output completion entry is skipped when it and the boundary
    entry would not both fit, either in the byte budget or under the
    hybrid-entry count bound. On non-thinking templates that re-render the
    output verbatim, the next turn then re-prefills the previous output. The
    boundary entry still covers everything before it.
  - Otherwise the completion entry is stored and the boundary entry is moved
    back to most recently used, so budget or pressure trims take the
    completion entry first.
- **Long prefills reclaim the cache first.** Before a prefill starts, the
  request's projected peak is compared against the Metal pressure threshold
  (90% of the cap). The projected peak is:
  - the recurrent state and the in-flight reservations, plus
  - the remaining prompt's KV, plus 3.5 × that KV for transients. A measured
    23k-token cold prefill on Qwen3.5-9B peaked 3.2 GB above the weights, with
    0.75 GB of that being the prompt's KV.

  Decode growth (`max_tokens`) is left out, because it is not part of the
  prefill spike; the pressure tick covers it as it accrues. The KV dims are
  read from the model itself (`.config`, or `.args` on mlx-lm models) through
  the same hybrid-aware estimator as the `/v1/models` context ceiling. The
  config-only admission projection reads 0 for mlx-lm models; that gap
  predates this PR, and changing it is out of scope.

  While the peak does not fit, prefix-cache entries are evicted in LRU order.
  This is the case the pressure tick cannot cover: it runs every 16 engine
  steps, and a 12-chunk prefill may finish inside one interval.
  - Reclaim applies only to hybrid (recurrent-state) models. That is where the
    transient was measured: head_dim-256 attention materialises each chunk's
    score matrix. On dense models KV dominates the peak, so they get no
    transient allowance here.
  - Reclaim stops as soon as an eviction frees no Metal memory, such as a
    lazily loaded entry or buffers the request still shares. So it never
    empties the cache for nothing.
- **Shutdown saves use measured throughput.** A budgeted (shutdown) save
  measures a 16 MiB fsynced write and halves the result to cover serialization
  overhead. For the first prediction it uses that throughput, bounded to
  150-600 MB/s. A page-cache-speed probe cannot over-promise, because real
  1 GB entry writes measured 700-1300 MB/s. Later entries use the throughput
  observed on the real writes. The 3.5 s SIGTERM budget is unchanged.
- **Hybrid checkpoints are persisted.** An entry's recurrent-state checkpoints
  are written to an `entry_K_ckpt.safetensors` sidecar. On load, the sidecar
  is re-attached only if every array matches its layer's state slot in shape
  and dtype, every position lies inside the entry, and every recurrent layer
  is covered. Any mismatch, including a truncated file, drops the checkpoints
  and keeps the entry. After a restart, a session replayed from its first
  turn snaps to the newest checkpoint below the shared prefix instead of
  re-prefilling everything.

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

The prefill reclaim covers that case. Before the prefill starts, it evicts
cache entries until the projected peak fits under 90% of the cap. The existing
guards stay in place:

- The Metal-pressure evictor fires at 90% of the cap. It trims older entries
  there and empties the cache at the cap.
- The admission gate now evicts cache entries before rejecting a request.
- The per-request KV projection still rejects requests that cannot fit.

No guard was removed.

## Consequences

Turn N+1 of an agent session reuses turn N's boundary entry on 16-32 GB Macs.
Measured before/after numbers are in the PR that introduced this decision.

Restart persistence now completes for a realistic session. The newest
boundary entries are written within the SIGTERM budget and reloaded on start,
so a client that resumes the same conversation extends them.

Replaying a session from turn 1 after a restart originally resumed from the
newest persisted 2048-token stride checkpoint below the turn-1 prefix. The
#3796 follow-up below adds the exact message-boundary anchor.

Known limits:

- The floor is computed once, when the scheduler is built. It does not
  subtract models loaded later, and with `--disk-stream` the resident weights
  are near zero.
- The multimodal (MLLM) lane keeps its own budget and gets no floor.
- An explicit `--cache-memory-percent` is never raised by the floor. The
  operator's value is kept, and one warning at startup names the floor. The
  floor applies only when the flag is not passed. `--cache-memory-mb` and
  `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` are likewise never raised. Other
  front-ends that build `SchedulerConfig` directly must set
  `cache_memory_percent_explicit` to get the same treatment.

## Follow-up: stable boundary checkpoint

Issue #3796 closes the replay gap by recording the recurrent state at every
real message-boundary snapshot, independent of the 2048-token stride. The
first boundary reached by a cold session is marked as its anchor. Checkpoint
thinning keeps that anchor while still enforcing
`RAPID_MLX_HYBRID_CHECKPOINT_MAX`; newer stride and message-boundary samples
compete for the remaining slots. The anchor position is stored in the
checkpoint sidecar metadata, so later turns after a restart cannot thin it
away. Internal N-1 snapshots are not session anchors.

Measured 2026-09-28 on an M3 Pro 18 GB with
`mlx-community/Qwen3.5-9B-4bit`, BF16 KV, MTP, PFlash off, 30 synthetic tool
schemas, and a fresh cache home:

| Phase | Prompt / cache | Engine TTFT |
|---|---:|---:|
| Cold turn 1 | 17,220 tokens / miss | 50.3 s |
| Live turns 2-3 | 38-41 tokens remaining | 0.4 s |
| Replayed turn 1 after SIGTERM/restart | 17,205 cached, 15 remaining | 0.7 s |

The shutdown saved one 17,231-token boundary entry under the 3.5-second
budget. Restart loaded it with its sidecar, and the fetch log reported
`LCP snapped to checkpoint: shared=17216 entry_len=17231 resumed_at=17205`.
This is a smaller synthetic prompt than the 22,819-token issue harness, so it
confirms the boundary mechanism and sub-second result on the target hardware;
it does not replace the issue's exact 30-tool harness measurements.
