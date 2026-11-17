# M5 Max MTP lookup copy ceiling

Measured A/B evidence: Apple M5 Max, 36 GB, GPU generation 17,
MLX 0.32.4, macOS 27.0.1, Qwen3.8-27B-4bit MTP. Greedy code rewrite,
2021 output tokens, three repetitions per arm.

| Arm | Decode tok/s | Copy proposals | Dominant verify widths (including base row) |
| --- | --- | --- | --- |
| Existing ceiling 31 | 191.7 | 68 | 32 rows: 62 |
| History only | 192 | Not supplied | Not supplied |
| Ceiling 47 | Same as 31 | Same as 31 | Tile-floor rounding retains 31 copied tokens |
| Ceiling 63 | 267–288 (+45–50% reported) | 38 | 64 rows: 30 |
| History + 63 | 287 | Not supplied | Same as 63 alone |

All arms produced output byte-identical to plain decode, reported SHA prefix
`84961702d631392c`. Chat was unchanged in every arm. History brought no gain,
and 47 cannot earn a wider tile; only ceilings 31 and 63 remain selectable.

The builtin cell selects 63 only on generation 17, Max tier, exactly 36 GB,
MLX 0.32.4 and macOS 27.0.1, for singleton MTP with an installed head and
existing lookup qualification. As with existing cells, core count is not a key.
The model schema cannot key on checkpoint name or revision: the narrowest
available model key is family `qwen3_5_text`, geometry
`hidden_size=5120, bits=4, group_size=64` (live text args do not expose the
intermediate size, so it is not a key), and quantization mode `affine`. This identifies
the dense 27B affine4/group64 family, including checkpoints with the same facts.
Installed module formats supply quantization facts because loaded model args
omit them; mixed or unknown formats do not qualify. Other chips, model facts
and software retain ceiling 31 (or their existing family policy limit).

The global default remains 31. The adaptive gate earns larger copies after
successful smaller copies, and rollback admission and output budget still limit
width. This evidence qualifies the measured rewrite and unchanged chat control;
it does not establish performance for other workloads or configurations.
