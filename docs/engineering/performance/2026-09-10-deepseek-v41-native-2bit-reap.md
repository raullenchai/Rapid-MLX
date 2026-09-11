# DeepSeek V4.1 Flash native 2-bit REAP qualification

Date: 2026-09-10

## Decision

Do not publish the generated artifact or expose DeepSeek V4.1 Flash in the
Rapid catalog. A native affine 2-bit text build pruned from 384 to 336 routed
experts fits in 256 GiB unified memory and passes a correctly framed greedy
smoke test, but its best observed decode result is 7.92 tok/s. An opt-in gate/up
fusion reaches only 7.87 model steps/s while transient memory rises to 263.72
GB. Neither passes the agreed 12 tok/s product floor, and the fused load is too
close to physical memory capacity for a safe 256 GiB default.

This work therefore stops before model upload, catalog registration, server
routing, or GUI exposure. The 20+ tok/s target remains unproven.

## Environment and source

- Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory
- MLX 0.32.2
- Source checkpoint revision:
  `802f1a00982705d81b79ad1c83aa0ccc0b863ebc`
- Source indexed size: 238,796,133,496 bytes
- Source quantization: MLX affine 2-bit, group size 64
- Native runtime reference revision:
  `8bdd543c7160800f3451d9595c996129b7abecdf`
- Calibration: 2,048 causal corpus tokens, split into two disjoint halves
- Product floor: at least 12 decode tok/s; target: 20+ tok/s

The checkpoint was already present in the standard Hugging Face cache. The
experiment did not redirect the cache or download another model. Generated
artifacts were written only below the Studio warm tier's `models-cold/`
directory.

## Artifact construction

The repacker streams safetensors bytes without materializing the full source.
It preserves existing packed affine values exactly, stacks split routed expert
tensors along an expert axis for the native runtime, and drops the unqualified
vision and MTP components. It does not claim to re-quantize original weights.

The calibration records normalized routing-weight saliency independently for
each corpus half. The conservative score is the maximum normalized saliency
from either half, so an expert is considered low value only when both halves
assign it little mass. Keeping 336 of 384 experts per layer produced:

- mean retained robust routing mass: 99.9949%
- worst-layer retained robust routing mass: 99.8564%
- split-half keep-set overlap: 92.1354%
- never-routed layer/expert cells: 4,910

The final text artifact contains 212,930,051,680 tensor bytes (approximately
199 GiB on disk), compared with 234,183,391,840 tensor bytes for the unpruned
native repack. Every routed projection and router row was checked for the
expected 336-expert leading dimension, and the strict loader reported no
missing or unexpected parameters.

## Correctness

Raw completion text is not a valid V4.1 quality probe. The required chat
framing is:

```text
<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>
```

With that framing, the prompt `What is the capital of France? Answer in one
short sentence.` generated `The capital of France is Paris.` followed by EOS.
All tested evaluation intervals produced the same greedy token chain.

The native runtime also passes the upstream tiny-config parity battery:

- prefill, decode, and chunked-prefill relative differences at or below
  1.4e-6;
- 100% argmax agreement;
- bit-exact fake-quant and dequant unit cases;
- strict float and quantized release-layout round trips;
- layer-at-a-time streaming equal to the loaded model.

That battery exposed and led to fixes for two bugs that a one-token smoke test
did not catch: grouped `wo_a` must retain its logical group axis when affine
quantized, and the original flat float layout must still work for multi-token
prefill and streaming loads.

This remains a smoke/parity result, not a broad task-quality, tool-use, vision,
or long-context evaluation.

## Performance results

The standard-metric run materialized the 199 GiB build in 239.44 seconds. Peak
MLX memory was 213.51 GB. Decode throughput uses Rapid's `(N-1)/elapsed`
definition: the 17-token correctly framed prompt produces the first token, then
seven measured target transitions produce the remaining tokens through EOS.

| Runtime path | Evaluation interval | Decode tok/s | Steady last-half tok/s | Peak MLX GB |
| --- | ---: | ---: | ---: | ---: |
| Native packed, watchdog-conservative | 4 | 7.31 | 7.43 | 213.51 |
| Native packed, short-decode ceiling | 20 | 7.86 | 7.89 | 213.51 |
| Native packed, one short decode graph | 40 | **7.92** | **7.90** | 213.51 |
| Gate/up fusion, model-step diagnostic | 20 | 7.87 steps/s | 7.84 steps/s | 263.72 |

The 40-layer graph completed this short decode, but it is not declared safe for
long-context serving: the cold unpruned build previously triggered the Metal
watchdog. The fused run's first prompt also paid 15.91 seconds of graph/fusion
warmup; later prefills were 0.25-0.26 seconds. Fusion is rejected because it is
not faster than the non-fused ceiling and its transient is only about 11 GB
below physical memory, leaving insufficient operational margin. Its older
diagnostic counted fixed model steps rather than Rapid's token-transition
metric, so the row is intentionally labeled separately.

## Why more REAP does not close the speed gap

REAP removes routed experts that are unlikely to be selected. It reduces the
stored model and router dimensions, but each decode token still executes six
experts of the same 2,304-wide shape in every layer. More aggressive expert
count pruning therefore does not proportionally reduce active expert compute.
It can also weaken routing coverage while leaving the main per-token kernels
unchanged.

Likewise, changing 2-bit affine group size or introducing a fractional-BPW
storage format primarily changes bytes and metadata. It does not create a
faster MLX execution kernel. A new packing format without a qualified native
kernel would either be decoded back into the current layout or run slower.

The remaining gap from 7.92 to 12 tok/s is 51.5%; the gap to 20 tok/s is
152.5%. Reaching either target needs a materially cheaper exact target forward,
not another catalog alias or a more aggressive storage-only prune.

## Reproduction

Calibration and repacking are explicit local operations. The calibration
command executes checkpoint-bundled code and therefore requires an affirmative
trust flag:

```shell
python scripts/calibrate_deepseek_v41_reap.py \
  --model <cached-source-snapshot> \
  --output /private/tmp/rapid-mlx-deepseek-v41-calibration/routing-2048.npz \
  --tokens 2048 --keep-experts 336 \
  --trust-checkpoint-runtime <calibration-corpus>

python scripts/repack_deepseek_v41_native.py \
  --source <cached-source-snapshot> \
  --destination <warm-tier-output> \
  --keep-experts 336 \
  --saliency /private/tmp/rapid-mlx-deepseek-v41-calibration/routing-2048.npz

DSV41_WIRED_GB=235 python scripts/qualify_deepseek_v41_native.py \
  <warm-tier-output> --tokens 16 --intervals 1 2 4
```

Do not use whole-process Metal capture for this model. Use bounded operation or
layer probes; a full capture can consume enough storage to endanger the system
volume.
