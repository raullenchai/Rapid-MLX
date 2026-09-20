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
(`_SUPPORTED_MODEL_TYPES` in `rapid_mlx/spec_decode/mtp/detect.py`),
enabled via `--speculative-config '{"method":"mtp"}'`. The Gemma 4
assistant-sidecar decision above still stands — it remains unsupported
and fails closed.

## Requalification (2026-09-20)

Gemma 4 assistant-sidecar MTP was re-evaluated with the supported server and
same-loaded-model parity harness. The target was
`unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit`; the assistant was
`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`. Both artifacts were pinned
to the revisions recorded in the durable performance report.

The requalification establishes a narrower, explicit opt-in contract:

- the ordinary multimodal wrapper call remains unchanged;
- only a named assistant sidecar makes the allowlisted Gemma 4 wrapper
  eligible;
- every proposed token is verified by the target;
- `K=0` exactly matched stock autoregressive decoding on all eight prompts;
- all `K=0` and `K=3` arms completed without fallback, stream corruption, or
  rollback failure;
- `K=3` matched `K=0` token-for-token on two of eight prompts. Five first
  divergences came from a target/non-draft token and one from a draft token
  that the batched target verification also selected. These are classified as
  the already-documented quantized `q_len=1` versus `q_len>=2` numerical fork,
  not as unverified draft acceptance.

The pooled run improved decode throughput from 57.88 to 67.12 tok/s (1.160x)
at 64.64% draft acceptance. Per-prompt results varied: low-acceptance prose
could be slower under fixed `K=3`, so fixed depth is not a universal speedup.
Automatic depth selection remains the recommended serving configuration.

Decision:

- Support Gemma 4 assistant-sidecar MTP only when the operator explicitly
  names a sidecar.
- Fail closed for shared-K/V target layouts until their shortened producer-only
  cache has a separately qualified assistant-layer mapping.
- Keep ordinary Gemma 4 inference and automatic model defaults unchanged.
- Preserve the batch-size-one runtime gate until a separate continuous-batch
  qualification exists.
- Treat exact `K=0`/`K>0` token equality as diagnostic evidence, not the sole
  correctness gate; target verification, completion, rollback integrity, and
  first-divergence classification are authoritative.

Full environment, revisions, commands, and per-prompt measurements are in
[`docs/engineering/performance/2026-09-20-gemma4-assistant-mtp.md`](engineering/performance/2026-09-20-gemma4-assistant-mtp.md).
