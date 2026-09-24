# Qwen3.6 native text-cache negative qualification on mlx-vlm 0.7.1

Date: 2026-09-23

Target: `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`

Host: Mac Studio, Apple M3 Ultra, 256 GiB unified memory

Runtime: macOS 26.5.2, Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3,
mlx-vlm 0.7.1, rapid-mlx 0.15.0

## Decision

Do not enroll the shared-weight native text companion on the pinned 0.7.1
runtime. The existing MLLM/VLM path remains authoritative, including when an
operator explicitly selects `--mllm`.

The performance result alone is a decisive NO-GO. Across six adjacent,
alternating-order pairs, native text was slower in all six. The fallback median
was 108.50 tok/s, the native median was 79.19 tok/s, and the paired median ratio
was **0.729x**. All six paired outputs were exact, so output drift does not
explain the regression.

This supersedes the 1.482x result recorded for mlx-vlm 0.6.17 in
`2026-09-13-qwen36-mllm-native-text-cache.md` for current-runtime enrollment.
That older receipt remains historical evidence only; it is not evidence that
the optimization is beneficial on 0.7.1.

## Separate media result

The screenshot check also failed: the model returned `Performance on Macs like
yours` instead of the configured expected phrase `Community pulse`. The image
still used the MLLM path, and the following text-recovery request passed.

This checker mismatch is separate from the performance decision. Correcting or
replacing the media expectation would not change the NO-GO: zero of six native
pairs improved and the independent paired performance ratio was 0.729x.

## Scope and integrity

All five compact behavioral cases passed on both lanes. MMLU was intentionally
not run because the requested first gate was the six-pair performance and media
campaign. The benchmark process exited, and a fresh post-run MLX probe reported
zero active and cache bytes.

The sanitized machine-readable receipt is
`2026-09-23-qwen36-native-text-vlm071-negative.json`. Its
`raw_receipt_sha256` binds the preserved scratch receipt and sidecar without
persisting absolute model or image paths. The raw SHA-256 is
`fb6af37e0a8f7aeaef0131e3a0c5f6f4d24761af253c533e69fe74b9fed3d227`.

The run used the already-cached immutable snapshot with Hugging Face,
Transformers, and datasets offline modes enabled. No model was downloaded.
