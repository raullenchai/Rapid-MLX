# Spec Decoding Validation Notes

## 2026-07-06 MTP A/B: Gemma 4 assistant sidecar

Target:

- Base: `mlx-community/gemma-4-12B-it-4bit`
- Sidecar: `google/gemma-4-12B-it-assistant`
- Config: `{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":1,"disable_auto_k":true}`
- Server controls: `temperature=0`, `--disable-prefix-cache`, same prompt set, single request at a time

Result:

- Correctness failed: 3 of 4 greedy HTTP prompts diverged from baseline text.
- MTP metrics on the failed run showed activity (`attempts=102`, `accepts=71`, accept ratio about `0.696`), and some prompts were faster, but this is not acceptable because greedy output was not lossless.
- Offline probes:
  - Injecting the assistant did not change fresh target logits (`max_abs_diff=0.0` on a direct target-logit check).
  - Forced-all-reject drafter matched baseline, so the reject rollback path is basically sound.
  - Forced-correct-draft all-accept matched baseline, so the simple all-accept path can be sound.
  - The real assistant still caused divergence under the vendored generator/server path; root cause remains open.

Decision:

- Gemma 4 assistant-sidecar MTP is not supported or advertised.
- Detection and dispatch fail closed for Gemma 4 MTP until a future implementation passes greedy-lossless correctness, stability, and performance validation end to end.

Next validation targets:

- Qwen3.5 / Qwen3.6 native MTP checkpoints with `mtp_num_hidden_layers >= 1`.
- Confirm pre/post behavior for:
  - greedy correctness: exact token/text equality vs baseline
  - performance: TTFT, decode tok/s, acceptance ratio
  - stability: repeated runs, multiple prompt classes, no mid-stream fallback corruption
