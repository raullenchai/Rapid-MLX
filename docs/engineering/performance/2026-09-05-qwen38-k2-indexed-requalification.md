# Qwen3.8 Flash-Next K=2 indexed-QSA requalification

Date: 2026-09-05

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB, `applegpu_g15d`

Branch: `vector/qwen38-mtp-k2-indexed`, PR #3087, rebased onto `main` after
prerequisite PR #3055 merged.

## Outcome

Indexed M=3 QSA changes the conclusion of the 2026-08-28 K=2 no-go, but only at
ultra-long context. A fully isolated same-code A/B found K=2 7.1% slower at 16K,
within 0.7% of K=1 at 32K, and 15.4% faster at 64K. The old gathered-QSA
long-context collapse is gone; the benefit has a measured crossover rather than
being universal.

The branch deliberately keeps implicit Qwen3.8 MTP at K=1. It admits K=2 only
when the operator explicitly requests `num_speculative_tokens=2`, and continues
to reject K=3. That explicit-only contract is important: K=2 is a 64K-oriented
operator choice and must not become the family default from this evidence.

## Why the old no-go can be reopened

The earlier experiment at `a20936703` had already qualified chain-of-K cache
correctness, but its gathered QSA verify path made K=2 decode fall from 34.41
tok/s at 2K to 17.93 tok/s at 32K. K=1 reached 33.14 tok/s at 32K in the same
campaign. The current indexed split-K path reads compact QSA selections without
materializing gathered K/V and is qualified for K=2's target verify width M=3
from 16K onward.

The implementation change in this branch is intentionally narrow:

- Qwen3.8 implicit depth remains K=1.
- Explicit K=1 or K=2 is accepted; every other explicit depth fails startup.
- The injected model capability ceiling is two.
- Existing generic chain-of-K and per-position GDN/PLE/QSA/KV rollback code is
  reused unchanged.

## Environment and method

| Component | Value |
| --- | --- |
| macOS | 26.5.2 (25F84) |
| Python | 3.12.14 |
| MLX / MLX-LM | 0.32.2 / 0.31.3 |
| Transformers | 5.16.1 |
| Artifact | `rapid-mlx/Qwen3.8-Flash-Next-4bit` at `dcf657e4acda2aae72da99cde65b6c491cd96998` |
| Decode | temperature zero, thinking disabled, fixed K, 256 requested tokens |
| Cache | prefix cache cleared before every request |

The normal mlx-lm load intermittently failed in
`mx.eval(model.parameters())` with a Metal GPU watchdog timeout before either
candidate code path could execute. Both K=1 and K=2 servers therefore used the
same local-only launcher: call mlx-lm with `lazy=True`, then materialize eight
parameter leaves per `mx.eval`. This changes only load synchronization; it does
not set MLX command-buffer environment overrides or alter inference scheduling.

The measured K=2 process explicitly set the kernel environment variable:

```bash
export RAPID_MLX_QSA_INDEXED_SPLITK=1
python3.12 /private/tmp/rapid_chunked_launch.py serve "$SNAPSHOT" \
  --host 127.0.0.1 --port 8465 --no-thinking \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":2,"disable_auto_k":true}'
```

The 16K and 32K K=1 runs used the identical command without the indexed
environment variable and with `num_speculative_tokens` set to one. The strict
64K K=1 comparator additionally exported
`RAPID_MLX_QSA_INDEXED_SPLITK=1`, matching the K=2 kernel environment and
isolating speculative depth. Timed requests used the repository's
`.orca/flash-next-eval/benchmark.py` with one requested prompt length per run.
The service emitted `QSA indexed split-K attention enabled for narrow
decode/verify` on the first 16K K=2 request, proving that the candidate route
was constructed rather than silently falling back.

Adversarial review identified that a public explicit K=2 request must not
silently depend on an undocumented environment variable. The final CLI uses
`setdefault` to activate indexed QSA for explicit Qwen3.8 K=2 while preserving
an operator's explicit `RAPID_MLX_QSA_INDEXED_SPLITK=0` fallback/debug override.

Because another validation supervisor repeatedly started a 38 GB model worker
despite the reserved host, every overlapping 64K K=1 replacement was excluded.
The detached supervisor and its launcher were stopped before the accepted
replacement. Excluded JSON files remain in `/private/tmp`; the accepted rows
below require both a cold-cache MISS and an interval with no competing model
process. This is contamination filtering, not performance outlier trimming.

## Clean results

| Prompt target | K=1 clean decode samples | K=1 median | K=2 indexed clean decode samples | K=2 median | Delta |
| ---: | --- | ---: | --- | ---: | ---: |
| 16K | 35.38, 35.34, 35.40 | 35.38 tok/s | 34.45, 31.54, 32.88 | 32.88 tok/s | -7.1% |
| 32K | 33.31, 33.21, 33.14 | 33.21 tok/s | 32.97, 33.03, 32.57 | 32.97 tok/s | -0.7% |
| 64K | 28.89, 29.45, 29.23 | 29.23 tok/s | 33.82, 33.72, 33.28 | 33.72 tok/s | +15.4% |

The strict comparison reverses the provisional one-sample 32K signal recorded
earlier in the day. K=2 is neutral within noise at 32K and a regression at 16K;
its material value begins somewhere between 32K and 64K. At 64K every K=2
sample beats every K=1 sample, with a 4.12 tok/s gap even between the slowest
K=2 and fastest K=1 runs. The final 64K K=1 comparator also had indexed QSA
enabled, removing the M=1 kernel toggle as a confounder.

The 32K K=2 median is still 1.84x the old 17.93 tok/s gathered-QSA result,
although that historical comparison crosses Rapid and MLX revisions and is
directional rather than a strict A/B.

MLX active memory reported approximately 107.1--107.3 GB at 32K and 109.7--109.8
GB at 64K. The historical allocator peak remained 148.1 GB.

## Correctness

Focused suites passed before real-model dogfood:

```text
tests/test_mtp_spec_decode.py                         132 passed
tests/test_qwen4_exp_vendored.py                     101 passed
tests/test_qsa_indexed_splitk.py + CLI wiring        123 passed
ruff on all changed Python files                     passed
```

The Qwen4 synthetic generation test now runs fixed K=1 and K=2 twice and
requires deterministic output at each depth. Generic K=3 tests continue to
cover the stronger partial-accept rollback geometry. Real K=2 captures at 128
and 2K completed 256 tokens coherently. They diverged from K=1 at token 37 and
token 9 respectively into coherent alternatives, consistent with the already
documented near-tied-logit accumulation boundary; no unverified draft or cache
failure was observed.

The isolated K=2 process also ran the 45-case Flash-Next release battery. It
retained the established K=1 baseline vector at 42/45: 8K/32K needle recall,
JSON schema, forced and automatic tool use, OpenAI and Anthropic protocols,
multi-turn behavior, and stop sequences passed. The three existing failures
were reproduced exactly: the probability answer, the narrow palindrome regex
scorer, and the generated project invoking unavailable `python` instead of
`python3`. There were no new K=2 failures.

A real streaming request then ran K=2 with `temperature=0.8`, `top_p=0.9`, and
`max_tokens=1024`. The client disconnected after two seconds and 26 generated
tokens. Server logs showed `CancelledError`, deferred abort, removal from the
running batch, and cleanup. The immediately following greedy request returned
exactly `RECOVERED`. Two further sampled requests at temperature 0.8/top-p 0.9
each completed 128 tokens through the fused top-p sampler and produced distinct
coherent paragraphs.

## Promotion gate

Completed:

1. Isolated fresh-process K=1/K=2 A/B with three accepted samples per arm at
   16K, 32K, and 64K.
2. Material 64K win beyond the observed noise band, with no claim of a shorter-
   context win.
3. The 45-case release battery with no new failure relative to K=1.

Remaining before merge:

1. Complete independent review and repository PR validation.
2. Have Atlas accept explicit-only K=2 as a supported ultra-long-context
   compatibility surface.

Do not enable K=2 by default from this result. The supported claim, if approved,
is limited to an explicit opt-in whose measured payoff is at 64K context.
