# Performance Experiments — Qwen3.6-27B Agentic

Branch: `performance-enhance`
Test: `lightning-mlx serve qwen3.6-27b --served-model-name local --port 8010` + Pi on 3 prompts
- `create a poem about cats`
- `create the snake game using react and typescript`
- `create a landing page using vite`

Decision rule: keep if improves all-turn or long-turn tok/s **without regressing wall time**, else revert.

## Progress

- [x] **Baseline** — current preset (prefix-cache on, MTP, draft_tokens=3, draft_temp=0.7) — 9.95 all / 28.25 long / 7.14 short / 709 s
- [x] **#1 Reorder cache-fetch: prefix-first** — kept (wall −32%, structural cleanup)
- [x] **#2 Raise `mtp_num_draft_tokens` 27B 3→4** — discarded (acceptance dropped 80%→70%, long-turn tok/s −28%)
- [x] **#3 Pre-compute SSE JSON templates** — already implemented (no-op for this session)
- [x] **#4 Skip MTP primary-logits recompute in verify** — invalid (different distribution required for accept/reject math)
- [x] **#5 Disable chunked-prefill for single-user** — already disabled by default; no-op
- [x] **#8 Suffix / Prompt-Lookup stacking** — deferred (multi-day integration; out of session scope)
- [x] **#10 MTP draft-temp tuning (0.7→0.5)** — kept (wall −9%, acceptance 77-78%)
- [x] **#14 `--kv-cache-quantization 4` default for 27B/35B** — discarded (tok/s −45%, same pattern as TurboQuant: store-only compress overhead with 0% HITs)
- [x] **#15 `--kv-cache-min-quantize-tokens 4096`** — n/a (gates a discarded feature)

## Status legend

- [ ] pending / running
- [x] done — see REPORT.md row for kept/discarded
