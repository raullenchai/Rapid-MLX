# TensorFold Ternary Bonsai 2 profile qualification

Date: 2026-10-06. Hosts: Mac Studio M3 Ultra, 256 GB, and a Mac with M4 Pro,
48 GB. Every run reused complete snapshots from the default Hugging Face
cache. One request at a time, temperature 0.

## Immutable inputs

- Runtime: TensorFold 0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`,
  MLX 0.32.3.
- Target: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` at
  `fcba37d2117a7077eac6b613b2668d14d9779edd`; 2-bit affine, groups of 128,
  Hadamard-rotated pack. 16.0 GiB resident with the drafter.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` at
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, packed to 4 bits, blocks of 8.

## Exactness

The tracked fixture
[`fixtures/glm53-tensorfold-ttlcache.json`](fixtures/glm53-tensorfold-ttlcache.json)
(768 completion tokens, 256-token thinking budget) was sent to
`tensorfold serve <target> --context 8192 --parallel 1` as tracked and again
with `"draft": false`.

| Drafted | `draft: false` | Token hash | Identical |
|---:|---:|---|---|
| 90.5 tok/s | 32.2 tok/s | `82d05fe8a0db` | yes |

## Speed through the product alias

`rapid-mlx serve bonsai2-27b-tensorfold` against a 2.6k-token prompt and a
170-token reply, median of four replies, beside `bonsai2-27b-2bit` on the same
host.

| Host | Profile | Ordinary alias | First token (profile / ordinary) |
|---|---:|---:|---|
| M3 Ultra, 256 GB | 53.1 tok/s | 37.3 tok/s | 7.76 s / 9.16 s |
| M4 Pro, 48 GB | 14.1 tok/s | 17.7 tok/s | 27.9 s / 29.0 s |

The M4 Pro row is upstream `tensorfold serve` with the same pair, taken before
the alias existed. Resident weights were 16.0 GiB on both hosts, so the loss
there is not a memory shortfall: without drafts TensorFold decodes this pack
slower than Rapid's ordinary lane on both machines (32.2 against 37.3 tok/s on
the M3 Ultra, 9.3 against 17.7 on the M4 Pro), and only the M3 Ultra recovers
more than that through drafting. The alias therefore requires 96 GB, the
smallest M3 Ultra configuration, and makes no claim for other chips.

## Behaviour checks on the alias

- Thinking on and off, streaming and non-streaming chat answered correctly.
- A request with `tools` returned HTTP 400.
- `/v1/models` reported `method: dflash`, `backend: tensorfold`, the pinned
  target and drafter revisions, and the fallback alias.
- `--no-spec-decode` exits before loading and names `bonsai2-27b-2bit`. The
  repository is shared with that alias, which serves it on the multimodal
  lane; this profile is text-only and has no ordinary mode of its own.

Swap on the M3 Ultra was unchanged across the runs.
