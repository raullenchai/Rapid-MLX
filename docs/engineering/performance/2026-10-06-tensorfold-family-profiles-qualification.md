# TensorFold family profiles qualification: Nemotron 3.5 Lightning and Qwen3.8 Flash Next

Date: 2026-10-06. Hosts: Mac Studio M3 Ultra, 256 GB, and a Mac with M4 Pro,
48 GB. Every run reused complete snapshots from the default Hugging Face
cache. One request at a time, temperature 0.

## Immutable inputs

- Runtime: TensorFold 0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`,
  MLX 0.32.3.
- `nemotron-3.5-lightning-tensorfold`:
  `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` at
  `d9d758fb83953437f7263256b0d96157e2a348b8`; 4-bit affine, groups of 64; MTP
  head shipped beside the weights. 17.3 GiB resident.
- `qwen3.8-flash-next-tensorfold`:
  `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at
  `2b170fa6309d5d1ee380b35636075fac7945f286`; 4-bit affine, groups of 32;
  embedded MTP head. 104.9 GiB resident.

## Exactness

The request is the tracked fixture
[`fixtures/glm53-tensorfold-ttlcache.json`](fixtures/glm53-tensorfold-ttlcache.json):
768 completion tokens with a 256-token thinking budget. It was sent to
`tensorfold serve <target> --context 8192 --parallel 1` twice, once as
tracked and once with `"draft": false`. The two replies must carry the same
message and the same token hash.

| Profile | Drafted | `draft: false` | Token hash | Identical |
|---|---:|---:|---|---|
| Nemotron 3.5 Lightning | 269.1 tok/s | 223.6 tok/s | `3aafba21a377` | yes |
| Qwen3.8 Flash Next | 119.6 tok/s | 87.7 tok/s | `b6c73213f6b7` | yes |

## Speed through the product alias

`rapid-mlx serve <alias>` against a 2.6k-token prompt and a 170-token reply,
median of four replies. "Ordinary" is the existing Rapid alias for the same
model on the same host.

| Alias | Host | Decode | First token | Ordinary alias |
|---|---|---:|---:|---:|
| `nemotron-3.5-lightning-tensorfold` | M3 Ultra | 245.5 tok/s | 1.23 s | 151 tok/s |
| `nemotron-3.5-lightning-tensorfold` | M4 Pro | 113.4 tok/s | 3.79 s | 93.2 tok/s |
| `qwen3.8-flash-next-tensorfold` | M3 Ultra | 112.3 tok/s | 2.61 s | 25.5 tok/s |

The M4 Pro row ran the profile's backend through Rapid's HTTP boundary before
the alias existed; the M3 Ultra rows ran the alias. Flash Next was not run on
the smaller host: its weights need the 192 GB floor the alias enforces.

## Behaviour checks on the alias

- Thinking on and off, streaming and non-streaming chat all answered
  correctly, with reasoning separated by the alias's parser.
- A request with `tools` returned HTTP 400.
- `/v1/models` reported `method: mtp`, `backend: tensorfold`, the fallback
  alias and the memory floor.
- `--no-spec-decode` on the Nemotron alias served the same checkpoint through
  the normal path at 138 tok/s on the M3 Ultra.
- `--no-spec-decode` on the Flash Next alias first loaded 105 GB and then
  failed with incompatible weights, because the checkpoint's 76 MTP tensors
  are unknown to the normal loader. The alias now exits before loading and
  names `qwen3.8-flash-next-4bit`.

Swap on the M3 Ultra was unchanged across the runs (15,968 MB used before and
after, left by earlier unrelated work).
