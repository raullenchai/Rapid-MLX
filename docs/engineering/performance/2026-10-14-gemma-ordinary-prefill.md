# Gemma 4 ordinary prefill: implementation validation

Ordinary Gemma 4 text and image prefills can request only the last output
position while still filling the complete producer KV cache. This avoids
unused vocabulary projection; models with trailing KV-sharing layers can
also reduce those layers' work. This optimization requires neither speculative
decoding nor a particular chip generation. Its benefit depends on model
architecture, prompt length, and the existing serving path.

## Scope and compatibility

The generator qualifies the known upstream Gemma 4 wrapper once at model load:
both forwarding wrappers must expose the option and the trunk must explicitly
accept it. Missing source, old wrappers, other model families, vendored and
unified implementations retain their existing behavior. Only ordinary cached
text/image forwards with known processor metadata qualify. Video, explicit
output overrides, hidden-state capture and unknown extra kwargs retain their
existing forward. Existing chunked-prefix/final-token calls are unchanged.
There is no retry after a model exception that may have mutated its cache.

The DFlash server owns a separate generation path; this change does not modify
its verification or hidden-state capture. No sampler or scheduling policy changes.
Floating-point reductions can differ with output shape, so equal logits or
identical stochastic samples across revisions are not promised.

## Exact-source M5 dogfood

Baseline: `5c70e238f6068d373553c48d92786a5cf333d05b`.
Candidate runtime: `70602f631a01bfebae29fbd0118e89287b218ef5`;
`rapid_mlx/mllm_batch_generator.py` SHA-256:
`ff66177beb6affc19511bbc4c3154ca7b90d17a814261b670c3ad1566f94cc74`.
The candidate used the production gate; observation did not inject model kwargs.

Environment: Apple M5 Max, 36 GiB, macOS 27.0.1, Python 3.12.13,
MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers 5.15.1,
NumPy 2.5.3, HTTPX 0.28.1. Model: `mlx-community/gemma-4-e2b-it-4bit`,
revision `238767527555cb75a05732a84dff5d6ba0dd6809`; weights SHA-256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.

Four fresh processes ran baseline/candidate/candidate/baseline. Each warmed
four input shapes, then ran three alternating-order repetitions per shape:
48 measured streams and 16 warmups. TTFT is local HTTP request start to first
nonempty content delta, with model already loaded. Prefix/disk caches and
speculation were disabled. Every response had the expected answer, usage,
finish reason and stream terminator; reported cached tokens were zero.

| Input (processor tokens) | Baseline median | Candidate median | Reduction |
| --- | ---: | ---: | ---: |
| Text, 220 | 50.65 ms | 26.84 ms | 47.0% |
| Text, 796 | 121.57 ms | 47.07 ms | 61.3% |
| Text, 2,424, already chunked | 117.67 ms | 116.89 ms | 0.7% |
| Image + text, 280 | 112.73 ms | 85.04 ms | 24.6% |

Forward traces confirmed one output row for eligible candidate calls and full
rows for baseline. The 2,048/375-token chunk-prefix calls and final one-token
call retained their existing arguments. All four servers exited cleanly.
One-second host samples found no other Python process over 512 MiB; system swap
stayed constant at 545,783,808 bytes. This is a small synthetic warm-serving
measurement, not a cold-start or universal model-size speedup claim.

Separate baseline/candidate correctness runs passed 42/42 checks with prefix
cache disabled and 42/42 with prefix cache enabled: arithmetic, long retrieval,
multi-turn, JSON, finite logprobs, repeated seeded requests within each revision,
cancellation/drain/recovery, concurrent image/text, readiness, clean exit,
Chat Completions, Messages and Responses routes, text streaming and image input.

## Reproduction

Use separate checkouts of the baseline and candidate, the same pinned dependencies
and cached model. Start each process in ABBA order:

```sh
rapid-mlx serve /path/to/cached/model --host 127.0.0.1 --port 18646 \
  --served-model-name hotpath --no-spec-decode --disable-prefix-cache \
  --disable-disk-caches --disable-model-downloads
```

Wait for `/health/ready`. Send streaming `POST /v1/chat/completions` requests
with `model="hotpath"`, `temperature=0`, `enable_thinking=false`, `max_tokens=16`,
`stream=true`, `stream_options={"include_usage":true}`. Use a single user message:

```python
content = (
    "Access code: 7319.\n"
    + "".join(f"Entry {n}: apples, pears, oranges.\n" for n in range(rows))
    + "The access code is 7319. Reply with the four-digit code only."
)
```

Use `rows=16,64,192`. For the image case, send a 224×224 solid red PNG and
`What color is this image? Reply with one English color word.` as text.
Change the first pixel to `(255, request_index, request_index)` per request to
avoid identical-image memoization. Warm each shape once, then measure three
repetitions per shape per process, reversing shape order on alternating
repetitions. Record monotonic request start and the first nonempty SSE content
delta; consume the whole stream and verify `7319` or `red`, usage and `[DONE]`.
Pool the six measured requests per variant/shape and compare medians.

## Regression evidence and limits

`tests/test_mllm_gemma4_prefill.py`: 24 passed on M5. Disabling the new
`logits_to_keep=1` assignment made the ordinary text and image tests fail.
The real tiny Gemma test verifies complete producer cache, selected token,
and the next decode, including a trailing KV-sharing layer. Retained prefill
logits are also bounded at absolute tolerance 0.002 / relative 0.0001:
shape-dependent float32 differences were approximately 0.001 on M5 and below
0.0000004 on M3. Exact cache and next-decode assertions remain stricter.
The final M3 focused suite adds six uninspectable-signature cases (both standard
inspection errors at all three model hops): 30/30 passed.

The focused six-file suite produced 306 passed and three failures in existing
chunked-prefill numerical assertions (`kv`, `rot:64`, `rot:16`). All three
reproduced with identical values on the unmodified baseline in the same
environment. No existing tolerances were changed. Default validation remains
required; these baseline failures must not be counted as a green full suite.

Earlier cross-chip and model-size experiments are recorded at
[the research revision](https://github.com/raullenchai/Rapid-MLX/blob/5643d543b/docs/engineering/performance/2026-10-14-gemma-cross-chip-validation.md).
Those experiments are distinct from this exact-source implementation run.
