# Qwen3.8 Flash-Next K=2 indexed-QSA requalification

Date: 2026-09-05

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB, `applegpu_g15d`

Branch: `vector/qwen38-mtp-k2-indexed`, stacked on the indexed split-K spike
`f620f0d847207a17d7c92f808230bd010fae8be6` and therefore on PR #3055 head
`8ef208607ececfaf9646da18575a1c7b95f7f4da`.

## Outcome

Indexed M=3 QSA changes the conclusion of the 2026-08-28 K=2 no-go enough to
justify continued qualification. It removes the old long-context collapse and
produced a small positive 32K signal versus a same-code K=1 run. It is **not yet
eligible for a production PR or validation queue**: repeated unrelated model
services interrupted the host, leaving only one clean same-code K=1 comparator
and one clean 64K K=2 sample.

The branch deliberately keeps implicit Qwen3.8 MTP at K=1. It admits K=2 only
when the operator explicitly requests `num_speculative_tokens=2`, and continues
to reject K=3. The product default must not change until a fully isolated A/B
and the existing release correctness battery pass.

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

K=2 additionally used:

```bash
export RAPID_MLX_QSA_INDEXED_SPLITK=1
python3.12 /private/tmp/rapid_chunked_launch.py serve "$SNAPSHOT" \
  --host 127.0.0.1 --port 8465 --no-thinking \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":2,"disable_auto_k":true}'
```

K=1 used the identical command without the indexed environment variable and
with `num_speculative_tokens` set to one. Timed requests used the repository's
`.orca/flash-next-eval/benchmark.py` with one requested prompt length per run.
The service emitted `QSA indexed split-K attention enabled for narrow
decode/verify` on the first 16K K=2 request, proving that the candidate route
was constructed rather than silently falling back.

Because other agents repeatedly started 6--38 GB model workers despite the
reserved host, a timestamped watchdog recorded and terminated competitors.
Any request whose wall-clock interval overlapped a watchdog event was excluded.
The exclusions explain why the clean sample count is uneven; they are not
performance outlier trimming.

## Clean results

| Prompt target | K | Clean decode samples | Median | Peak RSS |
| ---: | ---: | --- | ---: | ---: |
| 16K | 2 indexed | 33.45 | 33.45 tok/s | 55.97 GiB |
| 32K | 2 indexed | 34.79, 35.64, 36.30 | 35.64 tok/s | 55.96 GiB |
| 32K | 1 native gather | 34.29 | 34.29 tok/s | 55.87 GiB |
| 64K | 2 indexed | 37.80 | 37.80 tok/s | 56.06 GiB |

At 32K, the three-run K=2 median is 3.9% above the one clean same-code K=1
sample. More importantly, it is 1.99x the old 17.93 tok/s K=2 gather result,
although that historical comparison crosses Rapid and MLX revisions and is
therefore directional rather than a strict A/B. The clean 64K request did not
show the old context-length collapse, but a single sample is not a promotion
receipt.

MLX active memory reported approximately 107.1--107.3 GB at 32K and 109.7--109.8
GB at 64K. The historical allocator peak remained 148.1 GB.

## Correctness

Focused suites passed before real-model dogfood:

```text
tests/test_mtp_spec_decode.py                         132 passed
tests/test_qwen4_exp_vendored.py                     101 passed
tests/test_qsa_indexed_splitk.py + CLI wiring        107 passed
ruff on all changed Python files                     passed
```

The Qwen4 synthetic generation test now runs fixed K=1 and K=2 twice and
requires deterministic output at each depth. Generic K=3 tests continue to
cover the stronger partial-accept rollback geometry. Real K=2 captures at 128
and 2K completed 256 tokens coherently. They diverged from K=1 at token 37 and
token 9 respectively into coherent alternatives, consistent with the already
documented near-tied-logit accumulation boundary; no unverified draft or cache
failure was observed.

## Promotion gate

1. Reserve a genuinely isolated M3 Ultra window with no PR validation or
   Desktop model supervisor.
2. Collect at least three clean samples per arm at 16K, 32K, and 64K from fresh
   K=1 and K=2 processes on this exact stack.
3. Require K=2 to beat K=1 beyond run-to-run noise at the intended crossover;
   a one-sample 3.9% delta is not sufficient.
4. Run the 45-case Flash-Next release battery, cancellation/EOS coverage, and
   sampled decoding before changing the family capability in production.
5. Have Atlas decide whether explicit-only K=2 is a supported compatibility
   surface or whether the indexed kernel should land first while K=2 remains an
   experimental follow-up.

Until those gates pass, do not enable K=2 by default, open a production PR, or
enqueue this branch in PR validation.
