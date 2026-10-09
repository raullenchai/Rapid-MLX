# TensorFold DeepSeek V4 Flash profile qualification

Date: 2026-10-07. Host: Mac Studio M3 Ultra, 256 GB. Every run reused complete
snapshots from the default Hugging Face cache. One request at a time,
temperature 0.

## Immutable inputs

- Runtime: TensorFold 0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`,
  MLX 0.32.3.
- Target: `mlx-community/DeepSeek-V4-Flash-4bit` at
  `38c0bd20a6fba70f22c5ee2940ec0092b36ab936`; 4-bit affine, groups of 64,
  routed experts in mxfp4.
- Draft head: `TensorFold/DeepSeek-V4-Flash-DSpark-MLX` at
  `31fb9a6eeca93fe3e19aef8c9406fd42d16bb5e7`. 151.3 GiB resident with the
  target.

## Exactness

The tracked fixture
[`fixtures/glm53-tensorfold-ttlcache.json`](fixtures/glm53-tensorfold-ttlcache.json)
(768 completion tokens, 256-token thinking budget) was sent to
`tensorfold serve <target> --drafter <head> --context 8192 --parallel 1` as
tracked and again with `"draft": false`.

| Drafted | `draft: false` | Token hash | Identical |
|---:|---:|---|---|
| 56.8 tok/s | 43.2 tok/s | `88bcf2cd9644` | yes |

## Choice of draft head

Upstream publishes two heads for this target. Three short prompts (a class to
write, a 300-word explanation, a function to rewrite), thinking off, median of
three replies each, through `tensorfold serve`; the last row is
`deepseek-v4-flash-4bit` on Rapid's ordinary engine.

| Configuration | Code | Prose | Edit |
|---|---:|---:|---:|
| DSpark head | 66.9 tok/s | 45.5 tok/s | 69.4 tok/s |
| MTP head (`TensorFold/DeepSeek-V4-Flash-MTP-MLX`) | 60.0 tok/s | 49.8 tok/s | 62.0 tok/s |
| `draft: false` | 44.8 tok/s | 45.0 tok/s | 44.2 tok/s |
| Ordinary alias | 33.4 tok/s | 33.6 tok/s | 33.5 tok/s |

The MTP head is ahead on prose and about 15 GiB lighter; DSpark is ahead on
code and edits, on the thinking fixture above (56.8 against 43.2 tok/s for the
MTP head), and is upstream's default. The profile pins DSpark and refuses the
MTP head.

## Speed through the product alias

`rapid-mlx serve deepseek-v4-flash-tensorfold`, same three prompts, was ready
54 s after launch with warm files.

| Code | Prose | Edit |
|---:|---:|---:|
| 67.4 tok/s | 46.8 tok/s | 71.3 tok/s |

Against a 2.6k-token prompt with a 170-token reply the alias reached its first
token in 5.8 s and decoded at 50.9 tok/s (median of four replies).

## Behaviour checks on the alias

- Chat answered correctly, streaming and non-streaming; a repeated request
  returned the same message.
- Thinking follows the checkpoint's default and is off unless the request sets
  `chat_template_kwargs.enable_thinking`; with it set, reasoning arrived in
  `reasoning_content` and the answer in `content`.
- Requests with `tools` or `reasoning_effort` returned HTTP 400.
- `/v1/models` reported `method: mtp`, `backend: tensorfold`,
  `runtime_state: active`, the fallback alias, and the 256 GB floor.
- `--no-spec-decode` served the same checkpoint on the ordinary text lane.
