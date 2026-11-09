# Dense Qwen3.5 GDN fusion enrollment

Dense Qwen3.5 checkpoints served through the multimodal loader previously
skipped the guarded GDN input-projection fusion and the fused FP32-state
recurrent decode kernel. The recurrent kernel consumes projected tensors, so
the `hidden_size == 2048` enrollment check only reflected the MoE checkpoint
it was first qualified on. Plain (no-spec) dense loads now opt in with
`allow_dense=True`; other callers keep their prior scope. Speculative loads keep
the split projections that verification and rollback patching read.

Rollback: `RAPID_MLX_GDN_IN_PROJ_FUSION=0`, `RAPID_MLX_QWEN35_FUSED_GDN_DECODE=0`.

## Qualification

Qwen3.5 4B 4-bit, 564 prompt tokens, greedy, 256 client-counted output tokens,
three fresh-server repetitions per arm, idle-gated, median decode tok/s.
MLX 0.32.4.

| Chip | Stock (both fusions off) | Default | Change |
|---|---:|---:|---:|
| M3 Ultra | 166.89 (166.48–166.97) | 187.86 (187.29–188.23) | +12.6% |
| M4 Pro | 81.76 (81.56–81.78) | 85.68 (85.58–85.73) | +4.8% |

Four fixed prompts (story, arithmetic, JSON, code) match stock token-for-token
on both chips, 282 emitted positions including EOS; cold and repeated response
digests match across all runs. No arithmetic change: the recurrent state stays
FP32 and the existing device parity probes still gate enrollment.

Per-token M3 Ultra decode graph: 868 → 460 dispatches, 249 → 177 quantized
matvecs, 153 → 81 standalone RMS norms, 19 → 13 command buffers.

## Remaining dense-decode work

Grouping same-input projections and fusing residual-add with RMS norm gained
another 2–7% in raw-loop experiments on M3 Ultra and M4 Pro, on both Qwen3.5
and LFM2.5. They are not in this change: shared intermediate lifetimes under
concurrent serving, cancellation, speculative rollback and teardown are not yet
qualified.

```bash
python -m pytest tests/test_dense_gdn_loader.py \
  tests/test_gdn_in_proj_fusion.py tests/test_qwen35_fused_gdn_decode.py -q
```
