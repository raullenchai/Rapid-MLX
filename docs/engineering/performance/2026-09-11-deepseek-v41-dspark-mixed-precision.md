# DeepSeek V4.1 Flash DSpark mixed-precision experiment

Date: 2026-09-11

Status: the 4-bit-dense/2-bit-expert head passes the performance and memory
experiment and is published as a revision-pinned experimental sidecar. It is
not an unconditional default; broader output-quality qualification remains a
separate product gate.

## Question and boundary

The shipped candidate uses an affine 2-bit three-stage DSpark head. This
experiment asks whether selectively preserving more precision in the small
dense paths improves draft acceptance enough to raise end-to-end throughput,
without making a 256 GiB target fail to load.

Only drafter weights and offline extraction tooling change. The target model,
target kernels, verifier, catalog, server, GUI, and default K remain unchanged.

## Environment and provenance

- Apple M3 Ultra, 256 GiB unified memory (`hw.memsize=274877906944`)
- macOS 26.5.2 (25F84)
- Rapid commit `9abd21d9dd3b85091aaa0c1018f232fbf79953c9`
- Target: local 212.93 GB REAP12.5 native affine 2-bit checkpoint
- Low-precision head source: `Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP`
  revision `802f1a00982705d81b79ad1c83aa0ccc0b863ebc`
- Official source: `deepseek-ai/DeepSeek-V4.1-Flash` revision
  `dba1be0a40aa45a94ad051997016db3960a90277`

The official DSpark tensors occupy shards 44 through 46. Each source shard was
read and converted separately because the Hub cache did not have enough free
space for all three simultaneously. Source shard bytes and SHA-256 values:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| 44 | 2,652,728,736 | `9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05` |
| 45 | 2,573,998,176 | `0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5` |
| 46 | 2,706,402,896 | `e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d` |

The extraction tool requires revision-pinned files already present in the
global cache and exposes no download or cache-redirection option. A typical
per-shard conversion and final composition is:

```shell
python3.12 scripts/extract_deepseek_v41_dspark_precision.py convert-shard \
  --source <cached-official-shard> \
  --source-index <pinned-official-index> \
  --destination <affine4-head-dir> \
  --bits 4 --group-size 64

python3.12 scripts/extract_deepseek_v41_dspark_precision.py finalize \
  --destination <affine4-head-dir> \
  --source-config <pinned-official-config> \
  --source-index <pinned-official-index> \
  --source-revision dba1be0a40aa45a94ad051997016db3960a90277

python3.12 scripts/extract_deepseek_v41_dspark_precision.py compose-mixed \
  --low <pinned-affine2-checkpoint> \
  --high <affine4-head-dir> \
  --destination <mixed-head-dir> \
  --low-revision 802f1a00982705d81b79ad1c83aa0ccc0b863ebc \
  --high-revision dba1be0a40aa45a94ad051997016db3960a90277
```

## Precision choice and capacity result

The full affine 4-bit head is 8,015,180,738 bytes (7.47 GiB). Two attempts were
killed while the 212.93 GB target and head were becoming resident. It is
rejected for the 256 GiB product boundary.

The selected mixed head keeps the bandwidth-dominant routed expert matrices at
the proven affine 2-bit precision. Attention, shared experts, main projection,
Markov, and confidence paths use affine 4-bit. Target-shared embedding and LM
head metadata remain explicitly 2-bit.

- Total: 4,617,792,648 bytes (4.30 GiB)
- Increase over the 4,460,082,088-byte 2-bit head: 157,710,560 bytes (3.5%)
- Tensors: 3,584 across three independently hashed stage shards
- Strict DSpark loader: passed
- Peak MLX memory in qualification: 218.232 GB

Stage shard SHA-256 values are:

| Stage | Bytes | SHA-256 |
| --- | ---: | --- |
| 0 | 1,556,346,856 | `381a7fbc8758cd86baab55e8f1ae3020e49e2977f067d4269416889c1aee8118` |
| 1 | 1,512,099,424 | `2ca7dc77528ebe3220e6e1633733925ca3420e346b4475163e1639541333876e` |
| 2 | 1,549,346,368 | `1ed4663f0487e13372e753b2a5e8b760ae8fd43ba661fae891dfdff62384e64b` |

## Multi-domain result

The fixed suite covers code, arithmetic reasoning, JSON-only structured output,
and Chinese. Each row is a weighted result over two consecutive 128-token
passes in one loaded process. Chinese K4 reached EOS after 41 output tokens.

```shell
python3.12 scripts/qualify_deepseek_v41_dspark_suite.py \
  --target <reap12.5-target> \
  --overlay <mixed-head-dir> \
  --tokens 128 --repeats 2 --verify-k 4 \
  --collect-target-margins
```

| Domain | Mixed K4 tok/s | Accepted tokens/block | Repeat stable |
| --- | ---: | ---: | --- |
| Code | 22.52 | 2.37 | yes |
| Reasoning | 16.27 | 1.42 | yes |
| Structured | 25.93 | 2.88 | yes |
| Chinese | 11.87 | 0.74 | yes |

Across all eight prompt-runs, mixed K4 produces 842 transitions in 43.425
seconds: **19.39 tok/s**, versus **9.58 tok/s** for same-process AR
(**2.02x**). Mean acceptance is 1.85 extra draft tokens per block. Compared
with the prior 2-bit head's identically shaped 10.72 tok/s conservative A/B
result, throughput rises by **80.9%**, while the head grows by only 3.5%.

K5 was also tested with the same four prompts and two repeats. It reaches 18.49
tok/s (1.93x AR) and accepts 2.14 extra tokens/block, but the larger verification
window costs more than the added acceptance repays. K4 is the measured sweet
spot; this experiment does not justify changing the default K.

## Correctness and quality limits

All four outputs are token-for-token stable across two repetitions for both K4
and K5. No K4 output is bitwise identical to the sequential AR stream; first
differences occur at indices 79, 7, 15, and 7. The verifier remains
target-authoritative: every emitted draft token is accepted by target logits,
and corrections come from the target. As previously isolated, different target
batch shapes can cross low-margin decisions, so the supported contract is
determinism for the fixed execution shape rather than equality with sequential
AR.

Manual inspection also finds that only the Chinese K4 sample is clearly good.
The code sample becomes repetitive, the reasoning answer misreads a complete
word problem, and the structured sample violates JSON-only output. Similar
repetition exists in the target-only and prior-head probes, so this experiment
does not establish an instruction-quality regression, but it also cannot clear
the artifact for broad publication. A larger task-quality gate must compare the
mixed head, current head, and AR outputs before productization.

## Decision

Mixed precision is the right head direction for this 256 GiB target: it raises
the conservative multi-domain DSpark result from about 10.72 to 19.39 tok/s and
stays within the existing memory envelope. Full 4-bit and fixed K5 are rejected.
The sidecar is published at revision
`rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX@9530d6d2bf59e0d05177bd538095d5704ded1488`.
Rapid pins and verifies that byte set. Broader quality qualification remains
required before removing the experimental label or widening admission.
