# Tier 0 + Pinned Prefix Cache Implementation Plan

**Status**: Planned
**Target**: MiniMax-M2.5-MLX-4bit on M3 Ultra 256GB (serving OpenClaw)
**Branch**: `feat-minimax-parser`
**Baseline**: See [README benchmark section](../../README.md#baseline-benchmark--minimax-m25-on-m3-ultra)

## Context

After establishing a baseline benchmark for MiniMax-M2.5-MLX-4bit on M3 Ultra 256GB, we're implementing the first round of optimizations:

- **Tier 0**: GC control, detokenizer caching, grammar error handling (low-risk, high-impact quick wins)
- **Pinned Prefix Cache**: Prevent system prompt eviction under memory pressure

Goal: measurable improvement in TTFT, throughput stability, and error resilience. Results will be benchmarked against the baseline and packaged as a PR.

---

## 1. GC Control During Generation

**Problem**: Python's GC runs unpredictably during streaming, causing latency spikes (especially with 120GB+ model).

**File**: `vllm_mlx/server.py` -- `lifespan()` + `stream_chat_completion()`

**Implementation**:
- At server startup in `lifespan()`: `gc.set_threshold(100_000, 50, 50)` to dramatically reduce GC frequency (vLLM upstream uses similar approach, see [PR #33575](https://github.com/vllm-project/vllm/pull/33575))
- In `stream_chat_completion()` and `chat_completion()`: wrap generation with `gc.disable()` / `gc.enable()` + `gc.collect()` after completion
- Add `--gc-control` CLI flag (default: enabled) for easy toggle

**Key pattern** (in streaming path):
```python
gc.disable()
try:
    async for chunk in engine.stream_chat(...):
        yield chunk
finally:
    gc.enable()
    gc.collect()
```

**Expected impact**: Eliminates GC-induced latency spikes during generation. Most noticeable with large models (120GB+) where GC cycle scanning is expensive.

---

## 2. Detokenizer Result Caching

**Problem**: `mlx_lm.stream_generate()` returns text deltas, but `server.py` re-encodes the full accumulated text to count tokens on every chunk. This is O(n^2) over the response.

**File**: `vllm_mlx/server.py` -- `stream_chat_completion()`

**Implementation**:
- Track `token_count` incrementally instead of re-encoding each chunk
- Use tokenizer's `encode()` only on the delta text, accumulate count
- This avoids the quadratic re-tokenization pattern

**Key change**: In the streaming loop, replace any `len(tokenizer.encode(accumulated_text))` with an incrementally updated counter.

**Reference**: vLLM upstream [PR #20413](https://github.com/vllm-project/vllm/pull/20413) achieves 13.7x speedup with similar approach.

**Expected impact**: Reduces per-chunk overhead from O(n) to O(1), significant for long generations (2K+ tokens).

---

## 3. Grammar/Schema Error Handling Hardening

**Problem**: Bad JSON schemas from clients (OpenClaw sends diverse schemas) can crash the guided generation path, potentially killing the server.

**Files**:
- `vllm_mlx/api/guided.py` -- `json_schema_to_pydantic()` already has try/except but returns None
- `vllm_mlx/server.py` -- Schema extraction and guided generation call

**Implementation**:
- In `server.py`: When guided generation returns None or raises, fall back to standard generation with a warning (not a 500 error)
- Log the problematic schema at DEBUG level for debugging
- Ensure the `_disconnect_guard` error handler properly catches schema-related errors

**Reference**: vLLM upstream [PR #30346](https://github.com/vllm-project/vllm/pull/30346)

**Expected impact**: Server no longer crashes on malformed schemas. Graceful fallback to unconstrained generation.

---

## 4. Pinned Prefix Cache

**Problem**: OpenClaw's system prompt (~2K tokens) gets evicted under memory pressure, causing repeated re-computation. With MiniMax-M2.5 on 256GB, the first request after eviction adds 0.5-1s TTFT.

**Files**:
- `vllm_mlx/paged_cache.py` -- `CacheBlock`, `FreeKVCacheBlockQueue.popleft()`, `_maybe_evict_cached_block()`
- `vllm_mlx/prefix_cache.py` -- `PrefixCacheManager`, `BlockAwarePrefixCache`
- `vllm_mlx/server.py` -- New CLI flag + auto-pin logic

### 4a. CacheBlock pinning (`paged_cache.py`)

Add `is_pinned: bool = False` to `CacheBlock` dataclass.

In `FreeKVCacheBlockQueue.popleft()`: skip pinned blocks when popping:
```python
block = self.fake_head.next_free_block
while block is not self.fake_tail and block.is_pinned:
    block = block.next_free_block
```

In `_maybe_evict_cached_block()`: refuse to evict pinned blocks:
```python
if block.is_pinned:
    return False
```

### 4b. Pin/unpin API (`paged_cache.py`)

Add `pin_blocks(block_ids: list[int])` and `unpin_blocks(block_ids: list[int])` methods to `PagedCacheManager`. These set `is_pinned = True/False` on specified blocks.

### 4c. Prefix-level pinning (`prefix_cache.py`)

Add `pin_prefix(token_ids: list[int])` to `PrefixCacheManager` / `BlockAwarePrefixCache`. Finds blocks covering the given token prefix and pins them. Add `unpin_prefix(token_ids: list[int])` for cleanup.

### 4d. Auto-pin system prompt (`server.py`)

- Add `--pin-system-prompt` CLI flag (default: False)
- When enabled: after first request with a system message, pin the prefix cache blocks for that system prompt
- Track pinned prefix hash to avoid re-pinning on every request

**Reference**: vLLM upstream [PR #27097](https://github.com/vllm-project/vllm/pull/27097)

**Expected impact**: Turn4/Turn1 TTFT ratio drops from 2.09x toward 1.0x. System prompt KV cache stays warm indefinitely.

---

## 5. CLI Flags

**File**: `vllm_mlx/cli.py`

New flags:
- `--gc-control / --no-gc-control` (default: enabled)
- `--pin-system-prompt` (default: False)

---

## File Change Summary

| File | Changes |
|------|---------|
| `vllm_mlx/server.py` | GC control in lifespan + generation paths, schema fallback hardening, auto-pin system prompt |
| `vllm_mlx/paged_cache.py` | `is_pinned` field, skip-pinned eviction, pin/unpin API |
| `vllm_mlx/prefix_cache.py` | `pin_prefix()` / `unpin_prefix()` methods |
| `vllm_mlx/cli.py` | `--gc-control`, `--pin-system-prompt` flags |
| `vllm_mlx/api/guided.py` | Minor: ensure schema errors logged at DEBUG |

---

## Verification Plan

1. **Correctness**: Run the server with all new flags, send test requests (tool calling, reasoning, long gen) to confirm no regressions
2. **Benchmark**: Run `benchmark_minmax.py` with same parameters as baseline
3. **Compare**: Verify improvements in:
   - TTFT (should improve with GC control + prefix pinning)
   - Throughput consistency (GC control reduces spikes)
   - Prefix cache turn ratio (pinning should keep Turn4/Turn1 <= 1.5x)
   - Error resilience (bad schemas no longer crash)
4. **PR**: Create PR with baseline vs post-optimization comparison table
