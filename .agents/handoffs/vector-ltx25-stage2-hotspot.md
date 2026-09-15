# Vector handoff: LTX-2.5 stage-2 hotspot

Receiving role: Atlas

Date: 2026-09-13

Runtime branch: `vector/ltx25-stage2-hotspot`, stacked on the dequantized
matmul commit from upstream PR `MrMoferFRAN/ltx-2-mlx#1`.

## Verified facts

- At 6144 video tokens, video FFN is 39.1%, video self-attention 35.2%, and
  video text cross-attention 14.3% of the isolated block component sum.
- FP16 activations, fused QKV, Flash query chunking, and persistent
  dequantized-weight caches all failed the performance/memory gate.
- Raw middle-step residual replay improves full latency by 17.5% but reduces
  same-seed output to PSNR 30.07 dB and SSIM 0.8809. Least-squares scaling does
  not improve quality. No approximate inference control flow was retained.
- All runs completed without swap growth or a thermal warning.

## Decision and next action

There is no new product default from this pass. Atlas should choose between:

1. an exact, higher-effort native fused dequant-matmul-activation FFN kernel;
2. a separately labeled quality/speed tier, starting with multi-prompt
   calibration of a distilled stage-2 residual predictor/TeaCache controller.

Vector recommends the native FFN kernel first unless product explicitly wants
an approximate fast mode. Do not ship the tested previous-residual replay.
