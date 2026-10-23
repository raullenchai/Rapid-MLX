# Nemotron 3.5 Bolt E4B text serving

E4B is a Gemma 4 derivative, not the Nemotron-H/Puzzle architecture used by
Bolt 74B-A9B. Reuse the existing Gemma 4 shared-KV text loader and `gemma4`
tool/reasoning parsers. Recognized Gemma architectures take precedence over
the legacy tokenizer fallback triggered by the publisher's Nemotron name.
The old fallback could repeat `<turn|>` during direct generation.

## Checkpoint contract

The BF16 source is `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` at
`625857316b2a8b12022378a2cb2469a03cd72295`: `Gemma4ForConditionalGeneration`,
42 dense transformer layers, width 2560, 18 shared-KV layers, sliding window
512, and 65,536 text positions. The source contains vision/audio components;
these aliases deliberately qualify text only. The approximately 8B total / 4B
effective parameter designation does not mean a 4B MoE.

The separate “4GB” release is GGUF Q4_K_M, not an MLX packed checkpoint.
Source/model terms are OpenMDW 1.1, with Apache 2.0 terms for the underlying
Gemma 4 model. The source repository was created October 5; its model card
announces October 23. Repository creation is not proof of public release date.

Preserve the checkpoint-owned template, including `conversational_mode` and
`enable_thinking`. The actual template defaults thinking to **false**, although
the card describes true as the default. Explicit true inserts `<|think|>`;
reasoning uses `<|channel>thought` / `<channel|>`. Tools use
`<|tool_call>call:name{...}<tool_call|>`, not the large sibling's XML grammar.
Preserve generation defaults: temperature 1.0, top-p 0.95, top-k 64, and EOS
IDs 1, 106, 50. Greedy qualification overrides temperature to zero.

## Catalog and local conversion

| Alias | Published artifact | Immutable revision |
| --- | --- | --- |
| `nemotron-3.5-bolt-e4b` | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit` | `601cd937da1e168dfe0dcf184b95204344c683a9` |
| `nemotron-3.5-bolt-e4b-8bit` | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-8bit` | `72263b814727693c874bf765c657dfb32102e870` |

The published Rapid-MLX conversions use
BF16 source weights, affine group-64 quantization at four/eight bits, and
unchanged tokenizer/template/generation files. The text config carries the full
generation EOS set (1, 106, 50) and excludes vision/audio configs. Final conversion runs on GPU using
MLX 0.32.4 and mlx-lm 0.31.3. Only the main source weight shard is loaded:
the separate frozen shared-KV sidecar contains unused borrower K/V projections.
The main shard alone matches the native text model's expected tensors exactly.

All **1,355 tensors per quant** in the final local GPU conversions match the
temporary artifacts byte-for-byte (dtype, shape, and SHA-256 of tensor payload).
The tokenizer and template are preserved. This establishes equal model quality
for the published aliases. Both artifacts match
the independent native implementation's greedy outputs and score 12/12 on the
exact-answer smoke set alongside BF16.

An earlier CPU conversion experiment produced different quantization rounding:
665/1,355 tensors matched. Its 215-token raw-text mean NLL was 7.058 versus
7.240 for GPU four-bit, and 7.737 versus 7.775 for GPU eight-bit. Those CPU
artifacts are retained as experiments, not the staged publication candidates.
No broad benchmark quality equivalence to BF16 is claimed.
See the [qualification record](../performance/2026-10-23-nemotron-bolt-e4b.md).

Owner-approved publication on October 23 preserves both license files and the
upstream attribution. Every published file is checked against the staged SHA-256
manifest after download into a clean cache. The aliases pin the immutable
publication revisions above. Rollback is a revert of this alias publication
change; retain the published revisions for reproducibility. Vision/audio support
and broader quality evaluation remain outside this text qualification.

## DFlash

Leave speculative decoding disabled. The 4GB companion is GGUF-only. The BF16
companion at `913d13c49fbbb2b477dce78552a31ab650d5b693` is ModelOpt NVFP4,
with its own output head and five sliding-attention draft layers. The existing
loader rejects its metadata: `checkpoint quantization metadata is missing
required fields: bits`. No compatible draft ran, so acceptance rate, greedy
identity, and speedup are **not measured**. A new conversion/runtime port would
need its own correctness and net chat speedup qualification.
