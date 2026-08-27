# Qwen3.8-Flash-Next text-lane M1

Status: experimental text lane. The immutable release artifact is published as
`rapid-mlx/Qwen3.8-Flash-Next-4bit` at revision
`dcf657e4acda2aae72da99cde65b6c491cd96998`. MTP and vision remain separate
milestones.

## Architecture and conversion contract

The checkpoint declares `Qwen4ExpForConditionalGeneration` / `qwen4_exp`: 48
hybrid decoder layers, 512 routed experts with 10 active, Qwen Sparse Attention,
Gated DeltaNet, four-stream gated residuals, and PLE n-gram embeddings. Rapid's
text lane ports the architecture from typed checkpoint metadata; it does not
remap the model to an older family or infer behavior from a repository name.

The converted q4 artifact uses:

- q4 group 64 for eligible decoder matrices;
- q4 group 32 for PLE shards, whose width 160 is not divisible by 64;
- q8 group 64 for routing gates;
- no quantization for one-dimensional state, norms, buffers, or widths that do
  not satisfy the declared group contract.

`scripts/qwen38_streaming_convert.py` streams source shards, emits MLX sibling
`weight` / `scales` / `biases` keys, preserves the fused MoE gate-up/down
contract, and writes a complete index plus SHA-256 manifest. The verified
release artifact contains 28 shards / 97.51 GiB. The immutable repository is
104,695,605,424 bytes including metadata; all 34 entries in `SHA256SUMS.txt`
(manifest hash prefix `826f00b2`) match the remote revision. Strict loading
reports zero missing and zero unexpected parameters. Its model card records
source revision `f5d08274`, the Qwen Apache-2.0 license, this mixed-group
quantization contract, and the generated whole-tree manifest.

## Reference and numerical gate

The GDN and QSA forward math is adopted from the MIT-licensed implementation at
commit `ecf1aa0a62958ea770bc25c35e173effe142aa3c`. Source comments pin the same
commit at the adopted equations and Metal RoPE kernel. Rapid retains its own
batched, prefix-persistent QSA cache lifecycle.

On the same q4 weights, sequential processes produced:

| Probe | max abs diff | mean abs diff |
| --- | ---: | ---: |
| GDN, uncached | 0 | 0 |
| GDN, cached | 0 | 0 |
| QSA | 0 | 0 |
| PLE | 0 | 0 |
| MoE | 0 | 0 |
| all 48 captured layer outputs | 0 | 0 |
| 248,320 final logits | 0 | 0 |

The component tolerance is `1e-3`, approximately one BF16 quantization step at
unit scale; greedy-token agreement is required in addition to the tolerance.
The fixed short panel has 16/16 identical 32-token greedy sequences. Four
bounded 4.9K recall cases cross the 2,048-token sparse budget and also match
exactly, for 20/20 panel parity.

The pinned reference materializes a `[B,L,topk,K]` boolean intermediate and
requests 191,427,018,752 bytes on the original 19.3K recall prompts. That dense
intermediate is intentionally not copied. Rapid computes the identical batched
QSA scores and selected block sets, then materializes its bounded token mask.
All 12 QSA selected masks on the 4,906-token probe match bit-for-bit.

Run the opt-in artifact gate with sequential residency:

```bash
RAPID_MLX_QWEN4_EXP_ARTIFACT=/path/to/q4-checkpoint \
RAPID_MLX_QWEN4_EXP_REFERENCE=/path/to/pinned-reference \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest -q tests/test_qwen4_exp_artifact_parity.py
```

The gate covers the fixed short component/logit probes, then a 2,052-token
probe that crosses the 2,048-token sparse-QSA budget by one complete compressed
block, followed by the first cached decode token. Both logit probes require the
same greedy token in addition to the component tolerance.

## Real-model evidence

The original 20-case panel covers reasoning, code, Chinese, tool calls, and four
19.3K-token recall prompts. A 256-token run completed in 703.61 seconds with
maximum RSS 96,328,220,672 bytes, peak footprint 170,744,359,528 bytes, and zero
swap. The four long prefills completed in 130.209, 130.739, 145.181, and 136.935
seconds. Answers included the exact four recall facts and well-formed tool calls
for weather, stock price, meeting scheduling, and local-document search.

Both an explicitly served local checkpoint and the
`qwen3.8-flash-next-4bit` alias advertise the `experimental` capability. The
alias resolves to the immutable release artifact and declares a 128 GB minimum
memory floor; the exact 104,695,605,424-byte repository footprint is part of the
model-size manifest. The measured peak means 128 GB remains a tight operator
tier, while 192 GB is recommended for evaluation headroom. M1 is CLI/API only;
Desktop catalog exposure remains a later product milestone.

## Remaining milestones

- M1 PR gate: proportional engine/server tests, review convergence, and PR
  validation.
- M2: MTP head extraction and batched-consistent lossless verification before
  choosing any speculative-token default.
- M3: vision tower/processor integration and real-image correctness.
- M4: Desktop catalog exposure and mirror integration for the already
  immutable CLI/API artifact.
