# Absorbed MLA for multi-token verification

Date: 2026-09-05

Owner: Vector

## Decision

Carry the pending mlx-lm absorbed-MLA crossover as an experimental,
fail-closed compatibility patch. It is disabled by default and is limited to
post-update caches of at least 1024 tokens. Do not promote it to a default
until a supported speculative route has a larger trajectory-quality battery.

The patch covers the exact mlx-lm 0.31.3 implementations of DeepSeek V3,
GLM-4 MoE Lite, Kimi Linear, and LongCat Flash MLA. A source-body mismatch is
skipped. DeepSeek V3.2 remains owned by Rapid's indexed-attention patch and is
not replaced. The patch retires when mlx-lm exposes its upstream
`max_absorbed_queries` implementation.

## Mechanism

For latent rank `r`, combined non-positional query/value width `d`, post-update
cache length `T`, and query width `L`, absorbed MLA is selected only when:

```text
L < r*d*T / (r*d + T*(2*r - d))
```

The implementation evaluates the strict crossover with integer arithmetic.
Cold prefill stays on mlx-lm's materialized path; `L=1` delegates to mlx-lm's
existing absorbed decode branch. Rapid additionally requires `T >= 1024`,
based on the end-to-end qualification below.

Enable it explicitly with:

```bash
RAPID_MLX_MLA_ABSORBED_VERIFY=1 rapid-mlx serve MODEL
```

## Environment and artifact

- Mac Studio, Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2 arm64
- Python 3.12.14
- MLX 0.32.2; mlx-lm 0.31.3
- `mlx-community/GLM-4.7-Flash-4bit`
- immutable model revision `1454cffb1a21737e162f508e5bc70be9def89276`
- seed 7, greedy decoding where applicable

The Mini was not used because an unrelated 30B service owned its memory during
qualification. No process was stopped for this benchmark.

## Warm M=3 forward

Each arm used the same loaded weights and a cloned warm cache. Six samples per
arm were collected in ABBA order after compilation. The measured unit is one
full-model three-token forward.

| Context | Stock median | Absorbed median | Speedup | Final argmax |
| ---: | ---: | ---: | ---: | :---: |
| 1,024 | 50.795 ms | 20.315 ms | 2.500x | equal |
| 4,096 | 144.810 ms | 21.961 ms | 6.594x | equal |
| 16,384 | 517.628 ms | 31.992 ms | 16.180x | equal |

The output is not bit-exact. Final-logit RMS deltas were 2.1763, 1.0086, and
2.1409 respectively. This is why the switch remains opt-in despite the large
kernel win.

Reproduction:

```bash
python3.12 scripts/bench_mla_absorbed_verify.py \
  --model mlx-community/GLM-4.7-Flash-4bit \
  --revision 1454cffb1a21737e162f508e5bc70be9def89276 \
  --contexts 1024 4096 16384 --width 3 --repeats 6 \
  --oracle-context 4096 --oracle-cases 12 \
  --suffix-repeats 4 --suffix-max-tokens 128 \
  --seed 7 --json /tmp/mla-verify.json
```

## Sequential-decode oracle

At a 4K cache, twelve deterministic random token triples compared both M=3
paths with three sequential `L=1` target calls:

| Path | Oracle argmax matches | Mean logit RMS to oracle |
| --- | ---: | ---: |
| Stock materialized M=3 | 9/12 | 1.3222 |
| Absorbed M=3 | 11/12 | 1.4416 |

Absorbed MLA matched the sequential argmax more often but had 9.0% higher mean
RMS. The experiment rejects both “bit-exact” and “always numerically closer”
claims.

## End-to-end suffix dogfood

Four runs per arm used the existing suffix-decoding harness for 128 generated
tokens and the benchmark's deterministic 4457-token repeated code-edit prompt.

| Path | Median TPS | Delta vs stock | Diffs vs stock |
| --- | ---: | ---: | ---: |
| Stock materialized | 36.02 | baseline | 0 |
| Absorbed | 146.02 | +305.4% | 0/128 |

Both suffix arms differed from sequential vanilla on 103/128 tokens, while
they remained identical to each other. This isolates a 4.054x attention-path
gain without claiming that GLM-4.7 suffix decoding itself is correct. No
validated short-cache speculative route exists in Rapid today, so the initial
compatibility patch uses a conservative 1024 cache floor.

## Product scope and risks

Rapid's native MTP allowlist does not currently include GLM-4.7; the direct
benefit today is limited to explicit long-context suffix use and compatible
external speculative callers. HY3 uses ordinary attention, not MLA, and does
not benefit. This is a reusable MLA lever, not an all-model optimization.

The main risks are numerical path changes and drift in copied upstream model
methods. Default-off behavior installs no wrappers, exact source hashes block
unknown implementations, and mechanism counters expose absorbed,
materialized, single-token, and short-cache routing.

Upstream reference: <https://github.com/ml-explore/mlx-lm/pull/1817>
