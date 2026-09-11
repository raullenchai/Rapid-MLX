# Spec Decoding Validation Notes

## 2026-07-06 MTP A/B: Gemma 4 assistant sidecar

Target:

- Base: `mlx-community/gemma-4-12B-it-4bit`
- Sidecar: `google/gemma-4-12B-it-assistant`
- Config: `{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":1,"disable_auto_k":true}`
- Server controls: `temperature=0`, `--disable-prefix-cache`, same prompt set, single request at a time

Result:

- Three of four greedy HTTP prompts diverged from baseline text.
- MTP metrics on the run showed activity (`attempts=102`, `accepts=71`, accept
  ratio about `0.696`), and some prompts were faster. At the time, the text
  divergence was treated as a correctness failure.
- Offline probes:
  - Injecting the assistant did not change fresh target logits (`max_abs_diff=0.0` on a direct target-logit check).
  - Forced-all-reject drafter matched baseline, so the reject rollback path is basically sound.
  - Forced-correct-draft all-accept matched baseline, so the simple all-accept path can be sound.
- The real assistant still caused divergence under the vendored generator/server path.

Later Qwen3.5 investigation in #3295 established that byte equality with stock
single-token AR is stronger than the engine's batched-consistency contract.
Quantized near ties can flip when the target forward changes from `q_len=1` to
`q_len>=2`; that numerical difference is not, by itself, an acceptance or
rollback defect. The Gemma 4 result therefore cannot be classified as a
correctness failure from text divergence alone.

Decision:

- Gemma 4 assistant-sidecar MTP is not supported or advertised.
- Detection and dispatch continue to fail closed until the family is
  re-evaluated under the batched-consistency contract, with separate stability
  and performance validation. This note does not reclassify the old result as
  sufficient evidence to enable the family.

Next validation targets:

- Qwen3.5 / Qwen3.6 native MTP checkpoints with `mtp_num_hidden_layers >= 1`.
- Confirm pre/post behavior for:
  - greedy correctness: K=0 same-generator reference, speculative activity,
    and first-divergence source; do not use stock-AR byte equality alone
  - performance: TTFT, decode tok/s, acceptance ratio
  - stability: repeated runs, multiple prompt classes, no mid-stream fallback corruption

## Status (2026-08-18)

Qwen3.5 / Qwen3.6 native MTP has since shipped: `qwen3_5` and
`qwen3_5_moe` are in the MTP support allowlist
(`_SUPPORTED_MODEL_TYPES` in `vllm_mlx/spec_decode/mtp/detect.py`),
enabled via `--speculative-config '{"method":"mtp"}'`. The Gemma 4
assistant-sidecar decision above still stands — it remains unsupported
and fails closed.
