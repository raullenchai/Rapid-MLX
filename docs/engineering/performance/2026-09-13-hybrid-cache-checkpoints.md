# Recurrent-state checkpoints for hybrid prompt caches

Date: 2026-09-13

## Decision

Record the recurrent (GatedDeltaNet / Mamba) state at prefill chunk
boundaries and let the prefix cache resume a *divergent* prompt from the
newest checkpoint below the divergence, instead of refusing every
trim-requiring match for hybrid models.

The idea is lifted from mlx-lm-unified's APC v2 (`state_checkpoint` /
`trim_to_position` in its `models/cache.py`): keep a bounded number of
checkpoints per layer, always keep the newest, thin by dropping the one with
the smallest gap to its predecessor. We did **not** take its layered /
segmented cache, batching or ANE work — see *What we did not absorb*.

## Why this is the gap, and what it is not

A KV cache rewinds by slicing `offset`. An `ArraysCache` is a running
summary, so a stored hybrid entry can only be resumed at the exact length it
was captured at. `MemoryAwarePrefixCache` therefore served hybrids only on
exact and prefix-extension matches, and logged this on everything else:

```
[cache_fetch] LCP unavailable: shared=5679 entry_len=5760 requested_len=5710 non_trimmable=True
[cache_fetch] request=chatcmpl-e67 MISS prompt_tokens=5709
```

Before building anything I checked what the desktop app actually sends on
consecutive turns (a TCP tee in front of the sidecar on a 32 GB mini). Three
things fall out, and only one of them is this PR:

1. **Follow-up turns are already append-only** since #3335 (date-only system
   row, per-message send-time trailers, sorted JSON keys). Those hit the
   prefix-extension path and need no trimming.
2. **The tail of a prompt is already covered.** The scheduler snapshots the
   cache at the tile-aligned message boundary of every request
   (`_snapshot_boundary_segments`), so "same document, different question at
   the end" resumes from that boundary today — measured 1.5 s vs 20 s cold
   with checkpoints *off*. Nothing to gain there.
3. **Head divergence cannot be helped by the engine.** The app inserts a
   ~400-token tool-guidance preamble at the *front* of the system row on
   every turn that carries a tool result, and removes it on the next. Each
   flip re-prefills the whole conversation. That is an app change (make the
   guidance ride on the tool row instead), tracked separately.

What remains for the engine is **divergence in the middle of a long
prompt**: the user edits a paragraph of a pasted document, edits an earlier
message of a conversation whose boundary entry has been evicted, or a client
re-orders / rewrites part of a long context. Those were cold prefills; with
checkpoints they resume from the last chunk boundary before the edit.

## Design

* `vllm_mlx/hybrid_state_checkpoints.py` — `StateCheckpoints`, an immutable
  holder of `(position, arrays)` attached to each recurrent layer under
  `_rapid_state_checkpoints`. MLX arrays are immutable, so a checkpoint is a
  reference to the superseded state array; nothing is copied. The holder
  returns itself from `__deepcopy__`, so the `deepcopy` a fetch performs
  shares it instead of duplicating it.
* Recording happens in the scheduler on the per-chunk prompt responses
  mlx-lm 0.31+ already returns (`progress[0]` = tokens processed for that
  uid), through the public `BatchGenerator.extract_cache` API. No model or
  cache class is patched, batched prefill rows are handled by row index, and
  the prompt end is left to the existing boundary / prompt-cache-save
  entries. A checkpoint is recorded every `RAPID_MLX_HYBRID_CHECKPOINT_STRIDE`
  tokens (default 2048, i.e. every prefill chunk) up to
  `RAPID_MLX_HYBRID_CHECKPOINT_MAX` per layer (default 4; `0` disables).
* Checkpoints are attached to the cache right before each of the three
  store sites, are carried into the next request on a hit (`_seed_hybrid_
  checkpoints`), and are charged to the entry's byte ledger.
* Fetch: in the supersequence and LCP branches the former refusal becomes
  `_snap_hybrid_trim`, which rewinds KV layers exactly (array-sharing, as
  before) and restores recurrent layers from the checkpoint at the snapped
  position. If any non-trimmable layer is not an `ArraysCache` shape (a
  trim-liar class, a wrapper, an unknown type) the snap is refused and the
  old behaviour stands. A boundary entry that reaches further than the snap
  still wins (prefix path is checked first).

Only recorded when `--hybrid-cache-entries > 0` (auto-set to 8 for hybrid
models with the prefix cache on), so dense models and the drop-at-store
policy are untouched.

## Environment

* Mac Studio M3 Ultra 256 GB, Qwen3.8-27B-4bit (`rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`),
  text lane, `--enable-prefix-cache` (hybrid entries auto 8), MTP on.
* Serve: `python -m vllm_mlx.cli serve qwen3.8-27b-4bit --port 8123 --enable-prefix-cache`
  under `lockf -k /var/tmp/rapid-mlx-large-model.lock`; "off" =
  `RAPID_MLX_HYBRID_CHECKPOINT_MAX=0`.
* Prompt: 65 generated sections, ~5 700 tokens, plus a one-line question.
  Each row rewrites one section and sends the whole document again.
  TTFT = wall clock to first streamed content token, `max_tokens=4`.

## Results

| edited section (of 65) | divergence (tokens) | off: cold TTFT | on: TTFT | resumed at |
|---|---|---|---|---|
| 65 (last) | ~5 680 | 19.5 s | **7.5 s** | 4096 |
| 59 | ~5 150 | 19.6 s | **7.1 s** | 4096 |
| 46 | ~3 990 | 20.6 s | **15.1 s** | 2048 |
| 31 | ~2 660 | 20.0 s | **15.3 s** | 2048 |
| 13 | ~1 070 | 19.7 s | 22.4 s (miss, see below) | — |

Server log for the first row:

```
[cache_fetch] LCP snapped to checkpoint: shared=5679 entry_len=5760 resumed_at=4096 remaining=1614
[cache_fetch] request=chatcmpl-5e0 HIT prompt_tokens=5710 cached=4096 remaining=1614
```

Correctness: greedy 48-token answers for two edited documents ("which
section was rewritten, and what does it talk about?") are byte-identical
between the snapped path and a cold prefill on the checkpoint-off server
(`parity_on_*.txt` == `parity_off_*.txt`), and both name the right section.

Cold-prefill overhead of recording (four distinct ~5 700-token documents,
every one a MISS):

| document | off | on |
|---|---|---|
| seed 11 | 19.9 s | 20.1 s |
| seed 12 | 19.5 s | 20.8 s |
| seed 13 | 19.7 s | 20.8 s |
| seed 14 | 19.5 s | 20.0 s |

Mean 19.6 s vs 20.4 s (+4 %, 0.8 s over ~5 700 tokens, i.e. ~0.25 s per
recorded chunk on a shared machine whose cold runs earlier in the day spread
19.5–20.6 s). Recording is three `extract_cache` calls plus `mx.eval` of the
recurrent arrays per prompt; the per-uid stride gate skips the extract when a
chunk cannot produce a checkpoint.

## What we did not absorb from mlx-lm-unified

* **Layered / segmented cache and its batching**: our scheduler already has
  boundary snapshots, hybrid entry bounds and continuous batching through
  mlx-lm's `BatchGenerator`; the segmented design would replace, not add.
* **ANE offload**: no path to a matrix unit that helps a 27B decode on M3/M4
  (see the 2026-09-12 MTP depth-ceiling note).
* **QSA+NAX, qwen4 prefill**: separate track (#3055).

## Follow-ups

* MLLM lane (Qwen3.5 4B/9B, dense GDN served through `mlx_vlm.apc` exact
  entries): the same chunk loop is ours (`mllm_batch_generator.py`), so the
  cheapest port is to store an exact entry per stride there; needs a byte
  budget first because that cache is entry-counted (default 2).
* Desktop: move the tool-guidance preamble off the head of the system row
  so a tool turn stops re-prefilling the whole conversation twice.
