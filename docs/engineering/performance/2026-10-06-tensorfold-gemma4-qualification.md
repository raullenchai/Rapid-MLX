# TensorFold Gemma 4 26B profile qualification

Date: 2026-10-06. Hosts: Mac Studio M3 Ultra, 256 GB, and a Mac with M4 Pro,
48 GB. Every run reused complete snapshots from the default Hugging Face
cache. One request at a time, temperature 0. Other workloads shared the M3
Ultra during its runs.

## Immutable inputs

- Runtime: TensorFold 0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`,
  MLX 0.32.3.
- Target: `mlx-community/gemma-4-26b-a4b-it-4bit` at
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`; 4-bit affine, groups of 64,
  8-bit router. 13.8 GiB resident.
- No drafter. TensorFold's suffix lookup is the only speculation.

## Exactness

The tracked fixture
[`fixtures/glm53-tensorfold-ttlcache.json`](fixtures/glm53-tensorfold-ttlcache.json)
(768 completion tokens, 256-token thinking budget) was sent to
`tensorfold serve <target> --context 8192 --parallel 1` as tracked and again
with `"draft": false`.

| Host | Drafted | `draft: false` | Token hash | Identical |
|---|---:|---:|---|---|
| M3 Ultra, 256 GB | 125.8 tok/s | 126.7 tok/s | `45b4648394c2` | yes |
| M4 Pro, 48 GB | 79.0 tok/s | 80.6 tok/s | `45b4648394c2` | yes |

Suffix lookup adds nothing on this fixture, so the profile's gain over the
ordinary alias is the kernels'.

## Speed through the product alias

`rapid-mlx serve gemma-4-26b-tensorfold` against a 2.6k-token prompt and a
170-token reply, median of four replies, beside the same alias with
`--no-spec-decode` (Rapid's ordinary text engine on the same checkpoint) on the
same host, back to back.

| Host | Profile | Ordinary engine | First token (profile / ordinary) |
|---|---:|---:|---|
| M3 Ultra, 256 GB | 130.5 tok/s | 125.3 tok/s | 1.29 s / 1.34 s |
| M4 Pro, 48 GB | 84.8 tok/s | 80.1 tok/s | 3.91 s / 3.98 s |

The gain is 4% on the M3 Ultra and 6% on the M4 Pro.

## The paired drafter was not adopted

With `z-lab/gemma-4-26B-A4B-it-DFlash` at
`77d4202772dfe50b2396ec7bac9cfffc7b9e7057` packed to 8 bits, the same alias
measured 151.5 tok/s on the M3 Ultra but 76.3 tok/s on the M4 Pro, below both
the target-only profile and the ordinary engine there. The profile therefore
ships target only, with a 48 GB floor that matches the smaller measured host.

## Behaviour checks on the alias

Run on both hosts.

- Thinking on and off, streaming and non-streaming chat answered correctly.
- A request with `tools` returned HTTP 400.
- `/v1/models` reported `method: suffix`, `backend: tensorfold`, and the
  fallback alias `gemma-4-26b-4bit`.
- `--no-spec-decode` served the checkpoint through the ordinary text engine.
