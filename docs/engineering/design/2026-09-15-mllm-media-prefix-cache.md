# Serialized-MLLM media-aware exact prefix reuse: boundary design

Date: 2026-09-15 · Series: (1) media sampling parity (landed), (2) singleton no-rebatch (`perf/mllm-singleton-no-rebatch`, head `9d639eabc`), (3) this PR — `perf/mllm-media-prefix-cache` · Target lane: serialized hybrid MLLM (`MLLMBatchGenerator` under the structural B=1 policy) · Qualified checkpoint: Qwen3.6-35B-A3B-4bit, revision `38740b84…` (offline resolution).

Authoritative spec: `vector-desk/.agents/handoffs/vector-mlx-vlm-runtime-independence.md` — "Media-aware full-prefix reuse ceiling", "Fable follow-up disposition", "Multi-turn media boundary spike", "PR ordering" (this PR is third, after singleton converged).

## Problem

Multi-turn media conversations re-prefill the entire processor-expanded prompt every turn: image patches, prior question, rendered assistant turn, new question. Two reuse layers exist and are not enough:

* the projected-vision-feature cache (`mllm_batch_generator.py:2019-2034`) skips the vision encoder on a repeated image, but language prefill still runs end to end;
* the exact text-prefix APC (`_lookup_exact_text_prefix`, `mllm_batch_generator.py:1148`) deliberately bypasses image-bearing requests: media placeholders depend on pixel content, and its identity is keyed on text-lane semantics.

Spike evidence (below) shows the language-prefill work left after a warm feature-cache hit is large enough that verified prior-turn boundary reuse is worth a separate cache — under a strict, fail-closed contract.

## Spike evidence (evidence-only, not production numbers)

Three disposable spike scripts live in `/private/tmp/rapid-mlx-qwen36-image-native-cache-spike/scripts/`. They are monkeypatch harnesses, not durable tools; this section preserves their findings. The durable harness is `scripts/bench_qwen36_media_prefix.py` + `evals/prompts/qwen36_media_prefix_conversations.json` (see Qualification).

**Identical-request ceiling** (`spike_qwen36_full_media_replay_ceiling.py`) — clones the complete post-prefill language state and replays a byte-identical request with all preprocessing and prompt-forward work bypassed, restoring `_position_ids`/`_rope_deltas` on the language module. Across 4 paired repeats: TTFT 0.20446 s → 0.01636 s (−92.0%), end-to-end 0.29757 s → 0.10822 s (−63.6%), output 4/4 byte-exact. The baseline already had projected-vision-feature reuse warm, so this is incremental to the existing repeat-image win. It is an upper bound for an identical request, not a resumability result.

**Strict-extension eligibility** (`spike_qwen36_media_strict_extension.py`) — captures the processor-expanded input IDs the worker actually prefills, then compares each next-turn prompt against `prior_prompt + generated_ids`. Direct terminal-history reuse is **not** safe: 0/6 next-turn prompts had the prior terminal sequence as a strict prefix. All six diverged four tokens before the end of the previous generation prompt, where the Qwen chat template's assistant-generation marker is replaced by the rendered assistant turn. The longest common processor-expanded prefix was stable in 6/6 transitions (previous prompt length minus four), covering 75.6%–91.6% of the next prompt; per image the `rope_delta` was identical across all three turns.

**Boundary snapshot and resume** (`spike_qwen36_media_boundary_resume.py`) — splits turn-one prefill immediately before the stable marker boundary, evaluates and clones the hybrid cache plus `rope_delta`, then on turn two requires the new full token sequence to begin with the stored IDs, clones the stored state, restores the MRoPE delta, and prefills only the suffix. A mismatch raises instead of trimming or guessing. Results (3 screenshots × 3 pairs, feature cache warm in both phases):

| Metric | Warm current path | Boundary resume | Change |
| --- | ---: | ---: | ---: |
| Turn-one TTFT | 0.17932 s | 0.15603 s | −13.0% (ordering noise; no tax) |
| Turn-one end-to-end | 0.34680 s | 0.33707 s | −2.8% |
| Turn-two TTFT | 0.16577 s | 0.06869 s | **−58.6%** |
| Turn-two end-to-end | 0.19582 s | 0.08755 s | **−55.3%** |
| Deterministic output | baseline | candidate | 9/9 exact both turns |

A 2,296-token synthetic long-context turn reused 2,292 of 2,329 prompt tokens: turn-two TTFT −92.7% (1.046 s absolute), end-to-end −91.2%, output 3/3 token-exact — but a harmless phrasing near-tie flipped in 3/3 runs (`"Pick a model"`/`"Next" button` vs unquoted), confirming a different prefill partition can reorder floating-point reductions. Byte identity is required only where the execution graph is numerically identical; elsewhere task-quality plus first-divergence margin analysis governs. A 20-pair replacement soak produced 20/20 token-exact outputs on both turns with active Metal memory +376,832 bytes (~0.002%) — no per-snapshot ratchet — and median turn-two TTFT −72.1%. The soak measured snapshot sizes via `leaf.nbytes`; entries are tens to hundreds of MiB.

All numbers above are spike evidence on one checkpoint and must be re-derived by the tracked harness before any production claim.

## Design

### 1. Single-pass processor-derived boundary identity

Cache identity derives from the **processor-expanded input tokens plus a stable media digest**, never the unexpanded tokenizer prompt. That is the known defect of the legacy `MLLMPrefixCacheManager` (`mllm_cache.py:180`): it hashes the unexpanded prompt and leaves `skip_prompt_processing=False`; it is precedent and removable debt, not a wiring target.

Identity components, all collected in **one pass** inside `_process_prompts` (`mllm_batch_generator.py:2196`) / `_preprocess_request` (`:1673`) while the processor outputs are already materialized — no re-render of the conversation:

* the full processor-expanded token IDs, stamped as `MLLMBatchRequest.full_prompt_token_ids` (`:397` — the same field the text APC stamps at `:1172`);
* the ordered media content digest — reuse `compute_images_hash` / `request.vision_feature_key` (`:419`), already content-keyed and order-sensitive;
* model revision, quantization/adapter identity, processor geometry and pixel bounds — the semantics `_apc.semantic_extra_hash(model=…, processor=…)` packs into `_prefix_cache_extra_hash` today (`:970`), extended with the dynamic-vision pixel bounds this lane mutates (`_temporary_vision_pixel_bounds`, `:92`);
* chat-template identity and thinking/tool settings — `enable_thinking`, converted tool schemas (`convert_tools_for_template`), and `chat_template_kwargs` as consumed by the engine's template application (`engine/batched.py:2288-2298`).

Anything that can change token or position layout is part of identity; the token-prefix check below is the second line of defense, not a substitute.

### 2. Request-owned MRoPE save/restore transaction

MRoPE bookkeeping (`_position_ids`, `_rope_deltas`) lives on the language module, not in the cache. Precedent: `Qwen36NativeCacheTextWrapper` (`engine/batched.py:885-920`) saves the previous values, installs lane-local ones around the complete forward, and restores them in a `finally` — with a `missing` sentinel so an attribute that was absent is deleted again rather than set to `None`. This PR owns the same transaction for the media lane:

* at snapshot time, the evaluated `rope_delta` for the prefix is recorded into the entry (spike evidence: identical per image across turns);
* on a resume hit, the entry's delta is installed around the suffix forward and the previous model fields are restored on every exit;
* on a miss, error, or cancellation, the pre-existing model fields are restored unchanged — an aborted request never leaves its position state behind for the next request.

**Invariant.** The serialized lane is structurally B=1 (`max_num_seqs` / `prefill_batch_size` / `completion_batch_size` == 1, plus the `_next()` no-active-batch admission rule the singleton PR relies on), and all MLX work runs on the single model-owning worker thread. Exactly one request can be between transaction-open and transaction-close, and the worker thread is the only writer of the model fields — the same atomicity argument the native-text wrapper documents. If any future change admits a second concurrent request on this lane, the media store/lookup path must refuse (fail closed) rather than share the transaction; eligibility re-asserts B=1 explicitly. Failure to restore model-global position state on all exit paths is a hard no-go (handoff gate).

### 3. Exact token-prefix verification at the boundary

The boundary stores the **processor-expanded token IDs of the strict prefix**, not generated IDs and not a re-rendered assistant message. Spike evidence is explicit: `generated_ids` appended to the prior prompt is *not* a prefix of the next prompt (0/6), because the template replaces the generation marker with the rendered assistant turn. An entry holds:

* `token_ids` — the processor-expanded prefix (prior prompt up to and including the stable boundary);
* cloned hybrid cache leaves at exactly that position (a complete prior-turn boundary; recurrent `ArraysCache` state is never trimmed generically — the design restores a complete boundary and prefills a strict suffix, per the handoff);
* the `rope_delta` recorded for that prefix;
* byte size, identity digest, and insertion/recency for the LRU.

On the next turn the newly processed full token sequence (`full_prompt_token_ids`) must begin with the stored `token_ids` exactly — a full prefix check, not an LCP heuristic. Any mismatch is a **clean miss**: cold prefill through the existing path, no trimming, no guessing, no partial resume. The spike raises on mismatch; production downgrades that to a counted miss so a template or processor change degrades to today's behavior instead of an error.

**Decided**: the split point is computed from processor-expanded tokens at store time — the generation-marker offset derives from template metadata gathered in the single identity pass, never a hardcoded four-token constant (the spike hardcodes it; the PR must not). **Open**: the exact generic derivation across template revisions (see Open questions).

### 4. Small global byte-bounded LRU

The request surface has no reliable per-session identity, and entries are tens to hundreds of MiB, so reuse is a **global** store bounded by bytes, not entry count or session. The engine-wide byte-budget pattern already exists for the text exact cache and is reused directly: `_configure_exact_cache_capacity` (`mllm_batch_generator.py:1286`) resolves the budget via `MemoryCacheConfig().compute_memory_limit()`; `_exact_entry_bytes` (`:1519`) charges cache arrays plus attached checkpoints (conservative double-charging); `_enforce_exact_cache_budget` (`:1530`) evicts oldest-first until snapshots fit, always keeping the newest entry, and increments the budget-eviction counter surfaced by `get_prefix_cache_stats` (`:1562`). `APC_EXACT_CACHE_ENTRIES` remains the operator count override; media entries ride the same store and accounting rather than introducing a parallel cache.

### 5. Fail-closed rules

* **Detachment.** Terminal extraction must yield arrays that stay valid no matter what the live batch does afterwards. The singleton PR's `_extract_detached_singleton_leaf` (`:496`) is the convention: fresh leaves from explicit `mx.contiguous` copies, evaluated on the worker stream before returning, KV slabs trimmed to `offset`. Boundary clones follow the same rule — no aliasing of live state, no lazy slices.
* **Clone failure.** Cloning covers only the qualified leaf types (`_singleton_regular_cache_leaves`, `:458` — exact-type matching on `KVCache`/`ArraysCache`). An unknown, wrapped, subclassed, or quantized leaf fails the clone, no entry is stored, and the request continues cold. The spike's `raise RuntimeError("uncloneable cache leaf")` becomes a logged miss.
* **Cancellation/error.** The MRoPE transaction (§2) restores prior model state on every non-success exit; a request aborted mid-prefill or mid-decode never stores a boundary. Mid-generation cancellation recovery is re-qualified by the harness lifecycle scenario.
* **Rollback lever.** A config flag mirroring `mllm_singleton_fastpath` (`scheduler.py:601`, CLI `--mllm-singleton-fastpath`, generator check `:917-922`): `auto` (default) / `off`. `off` disables lookup *and* store, so a rolled-back build behaves byte-for-byte like today's lane.
* **Image cold path stays the fallback.** The projected-vision-feature cache and pixel cache work unchanged; a media-prefix miss falls through to exactly the current warm-image path, so the PR only adds reuse on top.

### 6. Integration points

* **Lookup** at the per-request cache creation site in `_process_prompts` (`:2261-2263`): before `make_prompt_cache`, an eligible image-bearing request consults the media store; on a verified hit the cloned leaves become `request_cache`, `request.cached_tokens` is stamped, and the suffix (boundary → end) is what `_run_vision_encoding` (`:1970`) forwards with `pixel_values=None` and the restored `rope_delta` — mirroring the spike's candidate path.
* **Store** at terminal extraction: only `singleton_regular` batches (the serialized lane this PR targets; `MLLMBatch.cache_layout`, `:565`, `extract_cache`, `:644`) may store. The batched merge path is out of scope; its `extract` lifecycle is not qualified for boundary snapshots.
* The text-only APC gate (`_is_text_only_request`) is untouched; media lookups are a separate branch so neither path corrupts the other's counters or entries.

## Non-goals

* No families beyond the qualified Qwen3.6 hybrid serialized lane; dense Qwen results are not a gate for this PR.
* No MTP and no draft/target interaction (a later MTP spike may *reuse* the request-owned position transaction, but nothing here anticipates it).
* No processor/cache-ownership refactor and no loader replacement — the lane keeps its mlx-vlm processor and cache leaf classes.
* No native-text-engine path; text-only traffic routed to the shared-weight native text engine (#3400) is out of scope, as in the singleton PR.
* No removal of `MLLMPrefixCacheManager` in this PR (separate debt cleanup).

## Qualification gates

Harness: `scripts/bench_qwen36_media_prefix.py` (tracked, ported and hardened from the disposable boundary-resume spike) over a separate manifest `evals/prompts/qwen36_media_prefix_conversations.json` — **20 manually constructed, privacy-safe conversations × 3 turns**: small UI screenshots, a real long-document screenshot, multi-image follow-up, repeated media, changed-media misses, template/settings misses, structured output, and OCR/grounding follow-ups designed to expose position errors. No model paths, user traces, or raw outputs are committed; private App traces may only inform aggregate strict-extension hit rates.

* Paired **cold-vs-resume** per conversation; deterministic token equality on the same exact build as the primary gate, with per-turn semantic checkers (required/forbidden terms, JSON shape) so two equally wrong outputs cannot both pass.
* Sampled fixed-seed A/B on ≥10 conversations (sampling parity is in the base, so the real request API is exercised).
* **50-turn bounded-memory soak**: no active-memory ratchet beyond allocator noise, byte accounting on every entry, eviction under the global budget exercised.
* Cancellation at random prefill/decode points with byte-exact post-abort recovery probes; concurrent-session MRoPE isolation assertion.
* **Absolute-value gate:** a real long-document or multi-image case with ≥1 s absolute turn-two TTFT saving (the synthetic 2,296-token spike run cleared a preliminary 1.046 s signal but does not satisfy this gate).
* Where outputs are not byte-identical: first-divergence margin analysis plus task-quality checks. Any incorrect deterministic answer, any entry without byte accounting, or any un-restored position state is a hard no-go.
* Real-model runs follow the manual qualification-machine contract (Mac Studio M3 Ultra, offline resolution, recorded environment); CI validates harness schema and deterministic contract paths only.

## Open questions

1. **Generic boundary derivation.** The spike used the template's generation-marker offset (four processor-expanded tokens in the fixtures). The PR derives it from single-pass template metadata; if that proves template-fragile, the fallback is the text lane's checkpoint-snap resume (`_snap_exact_text_prefix` / `_rewind_exact_entry`, `mllm_batch_generator.py:1438` / `:1408`) — but recurrent leaves without a checkpoint at the divergence point must miss cleanly, not trim.
2. **Vision-feature-cache composition on resume.** A resumed suffix forwards `pixel_values=None`, so the projected-feature cache is bypassed on the suffix by construction; whether the stored boundary should additionally pin feature-cache liveness (entries share the `_vision_feature_cache` LRU) needs an explicit policy during implementation.
3. **Turn-one cost.** The −13% turn-one TTFT delta was attributed to ordering noise; the harness must confirm no turn-one regression before the lever defaults to `auto`.
4. **Disk-backed entries.** Out of scope (`APC_DISK_ENABLED` stays excluded on this lane), but the byte-budget interface should not preclude a later cold tier.
