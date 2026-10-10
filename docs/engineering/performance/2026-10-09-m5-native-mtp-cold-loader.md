# Native MTP cold-loader repair on M5 Max

Date: 2026-10-09. This report qualifies the loader repair and the measured
greedy workload; it does not change runtime defaults or expand model eligibility.

## Problem and repair

A fresh process loading the qualified Qwen3.6 target and MTP head failed with
`AttributeError: module 'mlx_vlm.models.qwen3_5_mtp' has no attribute 'TextConfig'`.
The served-architecture shim exposed `Model` and `ModelConfig`, but the nested
configuration loader also resolves `TextConfig` from that same module.

The shim now binds the vendored `TextConfig` adapter when the served package
exports one. Its reuse check includes that adapter, repairing missing or stale
adapters even when the two top-level classes already match. Unrelated canonical
exports, package search paths and module metadata remain preserved. This applies
to the Qwen and GLM MTP packages; drafter families without that export keep their
existing binding contract. Using the vendored adapter also preserves corrected
Qwen3-Next MoE config selection when the nested loader runs a second conversion.

## Environment and provenance

- Apple M5 Max Mac Studio, 36 GB unified memory, macOS 27.0.1 (26A434).
- Python 3.12.13, MLX/Metal 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2,
  Transformers 5.15.1, NumPy 2.5.3.
- Tested loader/harness commit: `a792c7412` (base
  `c36183c4338b8b89e4b69bf3ae76bf7786f73fd4`). Full commit and exact changed
  loader/harness SHA256 hashes are in the fixture provenance.
- Target: `mlx-community/Qwen3.6-35B-A3B-4bit`, revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- Head: `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, revision
  `0295b81421bf4d0fccca9a7c0fcfb1418dda3516`.

Both snapshots already existed in the default HF cache. No additional model
download or cache relocation was needed. Benchmarks ran serially, with two
independently launched processes and no canonical-module preload workaround.
Raw observations are in
[fixtures/m5-native-mtp-2026-10-09](fixtures/m5-native-mtp-2026-10-09/).
Only filesystem paths were normalized.

## Real-model evidence

Each process loads one target and head through the production vendored drafter
registry and uses the same generation function as the native server. Both modes
are warmed on the coding prompt with 32 output tokens in AR/MTP/MTP/AR order.
Five prompts then run three greedy pairs each, reversing mode order in round 1.
Requests use fresh generation caches, a 192-token cap, normal EOS termination
and a draft block size of 3. Final token-ID receipts are validated against the
reported token count; repeated terminal streaming frames do not duplicate IDs.
The MTP arm must expose acceptance receipts or the harness fails.

| Fresh process | Exact token pairs | Median paired decode speedup | Positive pairs | Peak active MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Run 1 | 15/15 | 1.3984x | 12/15 | 19.76 GiB |
| Run 2 | 15/15 | 1.4015x | 14/15 | 19.76 GiB |

All **30/30** token comparisons pass. The repeated process medians support a
roughly 1.40x decode gain for this particular workload mix. They do not establish
a 40% gain for every prompt. Per-prompt paired medians are:

| Prompt | Output tokens | Run 1 | Run 2 |
| --- | ---: | ---: | ---: |
| Coding | 192 | 1.3984x | 1.4015x |
| Reasoning | 192 | 1.4335x | 1.4366x |
| Creative | 184 | 0.9973x | 1.0007x |
| JSON | 66 | 1.1937x | 1.1926x |
| Tool arguments | 15 | 1.4064x | 1.4083x |

The creative prompt receives essentially no benefit. The 15-token tool output
is especially short, so its decode-only speed ratio should not be extrapolated
to request latency. These are warmed, single-request decode measurements, not
end-to-end service, concurrency, sampled decoding or long-context claims.
No M3/M4-versus-M5 hardware comparison is claimed. GLM receives loader contract
coverage only; no new real-model GLM performance qualification is asserted.

## Regression verification

Before the repair, all nine new loader regression cases failed: missing/stale
nested exports for both MTP families, and the real nested-config loader boundary
for dense Qwen, MoE Qwen and Qwen3-Next. After the repair, **105 tests passed**
across the drafter parity/binding suite, native-MTP routing/runtime suite and
benchmark token-receipt tests. The byte-parity inventory records the binding
deviation explicitly. Ruff lint, formatting and diff whitespace checks pass.

A fresh CLI service also loaded the immutable pair through the normal native
backend. `/healthz` returned 200; streaming and non-streaming chat requests
both returned HTTP 200, `READY`, identical content digests and `stop`, with
completion-token usage and a terminal `[DONE]` frame for streaming. The owned
child server was then terminated. The smoke receipt is `http-smoke.json`.
The repository resolver needed HF metadata access with these copied cache
snapshots; benchmark runs used local paths fully offline. Shutdown logged a
semaphore cleanup warning, so this smoke is not a resource-leak qualification.

## Reproduction

Use the dependency versions and existing immutable cache snapshots above.
Run the following command twice as separate processes from the checkout root,
using `run1.json` and `run2.json` output names:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAPID_MLX_TELEMETRY=0 PYTHONPATH=.
out=/private/tmp/Pierre-native-mtp-loader
mkdir -p "$out"
python scripts/benchmark_m5_native_mtp.py \
  --target "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46" \
  --drafter "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit/snapshots/0295b81421bf4d0fccca9a7c0fcfb1418dda3516" \
  --rounds 3 --max-tokens 192 --output "$out/run1.json"
python -m pytest -q tests/test_mlx_vlm_vendored_drafters.py \
  tests/test_native_mtp.py tests/test_m5_native_mtp_benchmark.py
```

For the service smoke, allow HF metadata resolution and start the standard
loopback service with the cached qualified alias:

```sh
HF_HUB_OFFLINE=0 python -m rapid_mlx.cli serve qwen3.6-35b-4bit \
  --speculative-config '{"method":"mtp","backend":"native"}' \
  --host 127.0.0.1 --port 8618 --served-model-name native-mtp-smoke \
  --no-mllm --max-tokens 32
```

Vector owns loader follow-ups. The separate continuous-MTP token mismatch,
fused-GDN exactness investigation and checkpoint-disabled warm/cold differences
remain outside this repair and its qualification claim.
