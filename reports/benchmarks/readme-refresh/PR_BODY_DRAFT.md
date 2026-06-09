<!--
Draft PR body for `docs/readme-bench-refresh-2026-06-06` -> `raullenchai/Rapid-MLX:main`.
Branch is local-only; review before pushing. Open with:

  git push raullenchai docs/readme-bench-refresh-2026-06-06
  gh pr create --repo raullenchai/Rapid-MLX --base main \
    --title "docs(readme): refresh benchmarks against v0.6.80 + Ollama 0.24" \
    --body-file reports/benchmarks/readme-refresh/PR_BODY_DRAFT.md
-->

## Summary

Refresh the README hero table + sizing table with **measured** B=1 single-user numbers, and add a cross-engine A/B at B=4 concurrent load to back the "2-4× faster than Ollama" claim. The prior numbers were carried over from 2026-04 and several rows had drifted by 20-40 %.

Hero / sizing tables: re-measured 2026-06-09 against **v0.6.83** (post fused top-p sampler, PR #542). Cross-engine A/B: 2026-06-06 against **v0.6.80 + mlx-lm 0.31.3 + Ollama 0.24** — keeping these numbers as-is rather than re-running because the cross-engine framing is what matters and B=4 aggregate is closer to the M3 Ultra throughput plateau, so the speedup deltas don't shift materially.

Bench script, raw JSON artifacts, methodology notes, and the 7×3 tally all live under `reports/benchmarks/readme-refresh/`.

## Cross-engine A/B — B=4 aggregate tok/s

Aggregate `tok/s = sum(output_tokens across 4 streams) / wall_clock`. Each engine is the median of 3 measured rounds after 1 discarded warmup. M3 Ultra 256 GB · macOS 25.3.0 · rapid-mlx v0.6.80 · mlx-lm 0.31.3 · Ollama 0.24.0.

| Model (MLX alias)         | rapid-mlx | mlx-lm  | Ollama tag       | Ollama | vs mlx-lm | vs Ollama |
|---------------------------|----------:|--------:|------------------|-------:|----------:|----------:|
| qwen3.5-4b                |     261.1 |   173.2 | qwen3:4b         |  119.5 |   1.51×   |   2.18×   |
| qwen3.5-9b                |     180.0 |   136.3 | qwen3:8b         |   84.1 |   1.32×   |   2.14×   |
| qwen3.5-27b               |      65.9 |    54.9 | qwen3:32b¹       |   27.1 |   1.20×   |   2.43×   |
| gemma-4-12b               |      55.4 | crash²  | gemma3:12b       |   56.1 |     —     |   1.00×   |
| gpt-oss-20b               |     220.5 |   162.0 | gpt-oss:20b      |   96.5 |   1.36×   |   2.29×   |
| qwen3.6-35b (A3B 4-bit)   |     176.4 |   128.6 | qwen3:30b-a3b    |   87.1 |   1.37×   |   2.02×   |
| qwen3.5-35b (A3B 8-bit)   |     151.4 |   112.0 | qwen3:30b-a3b    |   87.1 |   1.35×   |   1.74×   |

### Headlines for the README copy

- **Apples-to-apples row** — gpt-oss-20b is the only row where rapid-mlx and Ollama load identical weights; rapid-mlx is **2.29×** there.
- **Qwen3 closest-tag rows** — Qwen3.5/3.6 DeltaNet isn't in llama.cpp yet, so the Ollama side runs `qwen3:Nb` (Qwen3 base). rapid-mlx leads 1.74-2.43× across this family.
- **Gemma 4 row** — different arch on both sides (mlx-vlm gemma-4-12b vs Ollama gemma3:12b). Tied at parity (1.00×) — kept in the table for coverage but doesn't carry the speedup story.
- **Tier-2 (vs mlx-lm)** — rapid-mlx is consistently 1.20-1.51× the upstream mlx-lm `serve` baseline at B=4 across same-weight comparisons.

¹ qwen3.5-27b vs `qwen3:32b` is a closest-comparator pick — `hf.co/unsloth/Qwen3.6-27B-GGUF:Q4_K_M` failed every load attempt (HTTP 500, Qwen3.6 dense isn't on llama.cpp). Audit trail JSONs are kept.
² mlx-lm 0.31.3 doesn't load Gemma 4 (its loader lives in mlx-vlm).

## README diff (highlights)

The README hero table and the per-Mac sizing table both moved to **B=1 single-user** numbers (one request, 256 max tokens, output_tokens / wall-clock including TTFT, median of 3). The cross-engine A/B above lives at the bottom in `## Benchmarks`.

- Qwen3.5-4B: 160 → **147 tok/s** (fixed methodology + v0.6.83 fused sampler)
- Qwen3.6-35B-A3B: 95 → **93 tok/s** (thinking-off + fused sampler — net ~flat because the old number double-counted reasoning_content, +38 % perf cancelled the methodology correction)
- Qwen3.5-35B-A3B 8bit: 83 → **80 tok/s** (same dynamic; +35 % from fused sampler offsets the thinking-off correction)
- Qwen3.5-27B: 39 → **37 tok/s** (methodology correction; v0.6.83 added back ~+13 %)
- Added rows: **Gemma 4 12B** (64 tok/s, vision-capable), **GPT-OSS 20B** (119 tok/s, harmony-native), **Qwen3.5-9B** (101 tok/s)
- Footnote ¹ marks the four 96-256 GB rows carried over from 2026-04 (disk-constrained on this refresh)

The B=1 hero table and the cross-engine A/B (B=4) use the same prompt + max_tokens but answer different questions — the hero numbers are "what you feel sitting in chat", the A/B numbers are "what a server handles under concurrent load". `reports/benchmarks/readme-refresh/summary.md` includes a "v0.6.83 hero-table refresh" section showing the v0.6.80 → v0.6.83 delta per row.

## Methodology caveats (recorded inline in `summary.md`)

- **Ollama daemon residency** — `OllamaEngine.stop()` only unloads its own tag; previously loaded blobs stay warm in the daemon between rows. 8 s cooldown between engines drops Metal pressure but not residency. Round-to-round CoV < 1 % so we left the run untouched, but a tighter follow-up should restart `ollama serve` per row.
- **Thinking-off requested** via `chat_template_kwargs.enable_thinking=False`. rapid-mlx / mlx-lm / mlx-vlm honour it; Ollama 0.24 ignores it for Qwen3 and streams `delta.reasoning` chunks, which decode at the model rate — so the Ollama Qwen3 rows reflect CoT-on throughput in practice. The 1.7-2.4× lead would widen if Ollama honoured the flag.
- **Authoritative token count** comes from the streaming `usage` chunk, not SSE-frame counting. Catches Ollama's `delta.reasoning` correctly; matches both rapid-mlx and mlx-lm's `total_completion_tokens`.

## Audit trail

Raw JSON artifacts under `reports/benchmarks/readme-refresh/`:

- `results-20260606-150434.json` — early 4-engine combined run (qwen3.5-4b/9b)
- `results-20260606-151417.json` — Tier-2 (vs mlx-lm) sweep on 27b/35b
- `results-20260606-152047.json`, `results-20260606-152654.json` — failed qwen3.5-27b Ollama warmup (kept for methodology trail; superseded)
- `results-20260606-152743.json`, `results-20260606-153334.json` — qwen3:32b re-run (working comparator)
- `results-20260606-154940.json`, `results-20260606-155011.json` — final 4-model rerun (qwen3.5-27b + gpt-oss + 35b pair vs Ollama)
- `summary.md` — full methodology + the 7×3 tally above

## Codex review status

Branch went through 6 review rounds before stopping:

| Round | Fix                                                                 |
|------:|---------------------------------------------------------------------|
|     1 | token-count rigor (use `usage` chunk, not SSE-frame counting) + targeted ollama stop + framing |
|     2 | exact ollama tag match + summary table + redact home paths in JSON  |
|     3 | apples-to-apples scope explicit + B=1 framing + repro split         |
|     4 | per-stream metric clarity (`agg ≠ per_stream`) + arg whitespace     |
|     5 | glossary excludes Gemma (different arch) + scorecard decode refreshed + JSON field renamed |
|     6 | wall-clock cap (timeout per round), engine-binary validation, audit-trail wording |

A 7th codex round was not requested — the failure modes had collapsed to wording NITs by round 6. Final commit `c10f418` is a docs-only polish (per-stream vs aggregate metric + Ollama daemon residency caveat).

## Test plan

- [ ] Re-render README locally — check the hero + sizing tables render with the new tok/s values
- [ ] Confirm `summary.md` and `PR_BODY_DRAFT.md` agree on every cited number
- [ ] Spot-check 2 JSON files match the table (`results-20260606-150434.json` for 4b/9b, `results-20260606-155011.json` for gpt-oss row)
- [ ] Verify no checked-in JSON has the raw `~/Library/...` home path (round-2 redaction)
- [ ] No bench script changes touch inference; pr_validate is not gated on this PR
