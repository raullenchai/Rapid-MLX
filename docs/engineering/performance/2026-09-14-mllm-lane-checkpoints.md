# Recurrent-state checkpoints for the MLLM lane (Qwen3.5 4B / 9B)

Date: 2026-09-14. Follow-up to
[2026-09-13-hybrid-cache-checkpoints.md](2026-09-13-hybrid-cache-checkpoints.md),
which covered the text lane only.

## Why the MLLM lane needed its own port

Qwen3.5 4B and 9B (the two most-used desktop models) are served through
`MLLMBatchGenerator`, not the text scheduler. Their prefix cache is mlx-vlm's
exact APC (`mlx_vlm.apc`): a stored entry is a deep clone of the whole prompt
cache, reusable only when it is a *prefix* of the new prompt, and the manager
keeps just `APC_EXACT_CACHE_ENTRIES` (default 2) of them with no byte
accounting. Measured on `main` before this change (Qwen3.5-4B-4bit, `--mllm`,
~5 700-token document, `max_tokens=48`, greedy):

| scenario | Studio M3 Ultra | mini M2 Pro 32 GB |
|---|---|---|
| cold | 2.72 s | 12.73 s |
| exact repeat | 0.18 s | 0.33 s |
| same document, new question | 2.63 s (MISS) | 12.64 s (MISS) |
| late edit (section 60/65) | 2.68 s (MISS) | — |
| mid edit (section 30/65) | 2.67 s (MISS) | — |
| follow-up after two edits | MISS (2-entry LRU dropped the conversation) | 0.37 s |

Every edited or re-asked document was a full cold prefill, and a conversation
lost its own boundary snapshot as soon as two other prompts came through.

## What changed

* `hybrid_state_checkpoints.is_recurrent_layer` now also positively
  identifies mlx-vlm's `ArraysCache` (Qwen3.5's VLM implementation builds its
  prompt cache from mlx-vlm's classes, and mlx-vlm's clone adapters rebuild
  entries as those classes).
* The MLLM lane owns its chunked text prefill, so it records recurrent-state
  checkpoints there (`record_checkpoints`, same `RAPID_MLX_HYBRID_CHECKPOINT_*`
  knobs and stride/thinning rules as the text lane) and attaches them to the
  boundary snapshot mlx-vlm stores (`attach_checkpoints`, truncated to the
  stored length). A warm exact hit seeds the resumed request with the
  snapshot's checkpoints so positions keep extending one set.
* When the exact lookup falls short, `_snap_exact_text_prefix` scans mlx-vlm's
  in-memory entries for the longest common prefix, picks the entry whose
  newest checkpoint below the divergence is furthest along, rewinds a copy
  (KV layers via mlx-vlm's own `trim`, recurrent layers via
  `restore_recurrent_layer`) and clones it through mlx-vlm's adapters exactly
  like a native hit. Logged as `[mllm_apc] … SNAP …`; the exact path still
  logs `HIT`.
* The exact-entry LRU defaults to 8 (`_MLLM_EXACT_CACHE_ENTRIES_DEFAULT`,
  `APC_EXACT_CACHE_ENTRIES` still wins when set) and is bound by the shared
  prefix-cache byte budget (`MemoryCacheConfig.compute_memory_limit`, i.e.
  `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` or 20 % of available RAM). Oldest entries
  are evicted first; the newest always survives. A 5.7k-token 4B snapshot is
  ~375 MB (eight full-attention layers of bf16 KV, capacity-padded), so on a
  32 GB mini with the desktop app running the budget resolved to 1.5 GB and
  held four entries.

## Results (this branch)

Sequence on one server: D cold (section 31 edited) → C (section 61 edited)
→ A (original) → E (follow-up on A) → B (same document, new question) →
A again. `cached` is the resumed position reported by the server.

| step | resumed at | Studio M3 Ultra | mini M2 Pro 32 GB |
|---|---|---|---|
| D cold | — | 3.37 s | 17.37 s |
| C edit 61/65 (snap from D) | 2560 | 2.00 s | 8.85 s |
| A original (snap from C) | 4608 | 1.25 s | 4.39 s |
| E follow-up (exact hit) | 5777 | 0.27 s | 0.56 s |
| B new question (snap from A) | 4608 | 0.90 s | 3.59 s |
| A repeat (exact hit) | 5777 | 0.57 s | 1.60 s |

Cold baselines on fresh servers: A 3.66 s / 16.11 s, C 3.36 s (Studio).
Checkpoints land at 512, 2560, 4608 because the adaptive first chunk is 512
tokens and the stride is 2048 from there.

Correctness: the greedy 48-token answers for A (resumed at 4608) and C
(resumed at 2560) are byte-identical to the cold answers on both machines.

## Not done here

* The final chunk position (the boundary itself) is only recorded when it is
  a full stride past the previous checkpoint, so "same document, new
  question" resumes at 4608 rather than ~5 760. Recording the boundary
  unconditionally would cost one more recurrent-state copy per prompt.
* Qwen3.5's chat template renders an empty think block on assistant rows
  only while they are the last turn, so a tool round's prompt is not a prefix
  of the next turn's (shares 442 of 496 tokens). That is a template
  normalisation, tracked separately.
