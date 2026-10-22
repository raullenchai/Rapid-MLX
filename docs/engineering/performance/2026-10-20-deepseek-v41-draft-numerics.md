# DeepSeek V4.1 draft numerics and decode overhead

CPU-verifiable phase, 2026-10-20. Owner: Vector. Branch:
`vector/dsv41-draft-numerics`; baseline:
`59dfabfa2955a5eeb24821bcd3e1133a91f96ea5`.
Scope: DSpark, installed draft helpers, post-prefill speculative loop,
benchmark, regression tests and Apple CI registration. Target modules and the
512-token prefill loop are unchanged. No GPU qualification is implied.

## Findings

| Hypothesis | Evidence | Disposition |
| --- | --- | --- |
| Draft attention/RoPE/KV disagree with target math | Adjacent pairs, tail-only rotation, inverse output rotation, head-dimension scaling, denominator-only sink and FP8/e8m0 blocks of 32 agree. Draft attention is intentionally noncausal across its noise block, as in the released draft. Sparse target attention and dense draft attention differ in softmax reduction order; CPU tests bound this with fp32/bf16 tolerances. | Reuse target RoPE. Preserve block visibility, sink and cache quantization. Do not substitute causal target attention. |
| Grouped output projection changes precision | Target casts attention output to fp32 before grouped `wo_a`, then casts back before `wo_b`. Both draft implementations previously kept the lower input precision through `wo_a`. | Match both casts in scalar and installed vectorized attention. Test identical weights/inputs in both precisions. |
| Routing/clamped activation disagree | Release uses sqrt-softplus, selection-only bias, normalized top-3 with denominator `sum + 1e-20`, scale 1.5, upper-only gate clamp and two-sided up clamp at 10. Draft scores and selections agree with native target on non-tied synthetic inputs. Target and draft use different top-k ordering; exact ties remain backend-dependent. | Preserve release routing policy and reuse target clamped activation. No temperature/scoring or sampling policy changes. |
| Routed scaling is numerically interchangeable across down projection | Native target scales fp32 **after** down projection. Draft scaled before the activation's bf16 cast. Linearity does not preserve equality across that cast. Released source also scales before down projection; native target and release therefore already differ at this boundary. | Match the actual native target. Both scalar and packed draft now scale after down projection. This is native-target alignment, not a claim of bitwise reproduction of release training kernels. |
| Scalar and packed projections preserve native precision | Scalar `linear` rounded dense inputs to weight dtype and forced all outputs back to input dtype. Native projections promote mixed inputs/weights normally. The native routed activation stays fp32 into down projection; the shared expert explicitly casts it back. These boundaries differ from the old scalar and packed draft. | Remove premature scalar casts, keep routed down inputs fp32, retain the shared-expert cast, and scale routed outputs afterward. Full native MoE versus scalar/packed tests use identical affine weights, fp32/bf16 inputs and fp32/bf16 projection storage. Dense projections match native layers bit-exactly in all four input/weight dtype combinations. |
| HC mixing or Sinkhorn differs | Staggered pre-mixes and residual orientation agree. The duplicate draft projection lacked the target's explicit weight promotion; separate einsum/reduction implementations risked drift. | Reuse target HC mixing, pre-collapse and post-expansion, including dtype handling and Sinkhorn implementation. CPU component tests are bit-exact. |
| Hidden-state source or positions are wrong | Release and native configuration select zero-based layers `[37, 38, 39]`. Native captures the arithmetic mean over HC copies at block entry, before RMSNorm and before learned HC collapse; concatenation follows configured layer order and preserves stream dtype. Release captures after any Engram update; native captures before it. Release Engram layers are `[1, 14]`, so this does not affect the selected layers. | No capture change. Tests verify block-entry means and deliberately reordered layer IDs. Batch observations preserve absolute contiguous positions and the final sliding window. |
| Rejection observes or retains invalid state | Existing loop only observes target hidden states through `mismatch - 1`, correctly excluding the replacement until it is actually forwarded. Offset rollback alone did not restore older ring rows overwritten by a rejected suffix; on the tiny model, rejection at index 1 diverged from plain greedy as early as output index 2. | Preserve the observation boundary (now one `observe_many` call over the verified prefix). Ring-row restoration and the verify-length cap come from the parent sliding-window rollback change, which this change builds on unmodified. |
| Five-token verify differs from sequential target calls | Seeds 7, 11, 42, float32: max absolute logit differences `6.5565109e-7`, `7.1525574e-7`, `2.9802322e-7`; all five argmaxes agree per seed. Small CPU reduction differences exist, with no greedy divergence in these measurements. | No target kernel changes. Assert a `2e-5` CPU bound and test whole-generation equality and deterministic draft proposals. Full bf16 target execution cannot run on this CPU backend: GatherMM supports float32 only. GPU near-tie behavior remains unverified. |
| Per-round host overhead is excessive | Old Markov chain performed five scalar argmax readbacks; verifier performed up to four more. Observations evaluated once per accepted position; installed packed MoE still dispatched once per draft position. | Keep Markov dependencies on device and read all proposal IDs together; one batched verifier argmax readback; one observation evaluation per verified chunk (rows projected individually so draft state does not depend on chunking); one packed MoE dispatch per stage. Keep the short host acceptance loop to preserve mismatch/EOS ordering. |

Release configuration was read as data from the cached official 1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` and compared with the native
checkpoint/sidecar configuration: block size 5, draft experts 128/top-3,
HC multiplier 4, 20 Sinkhorn sweeps, HC epsilon `1e-6`, RMS epsilon `1e-20`,
RoPE theta 10000, sliding window 128. No checkpoint Python was executed.

An additional release/native difference remains intentionally unchanged:
released attention fake-quantizes the grouped output activation, whereas native
target attention does not. Adding that only to the draft would introduce a new
native-target mismatch. The real checkpoint's quantized weights, input dtypes,
and release-kernel acceptance cannot be assessed from synthetic CPU weights.

## CPU evidence

Environment: Apple arm64 host, CPU only; Python 3.12.13;
CPU-built MLX `0.32.1.dev20261019+3a62199`. No model download or Metal execution.
Tiny setup: seed 7, actual native target, 3 layers, width 32, 2 heads, head width
32, window 8, 4 target/draft experts, top-2, 3 draft stages, K4 proposals,
synthetic affine 4-bit grouped/expert matrices. This draft is **untrained**.

Commands from the repository root (activate the CPU-capable test environment):

```sh
python -m scripts.bench_deepseek_v41_draft --tokens 64 --repeats 3 --seed 7 --label new
python -m pytest -q -s tests/test_deepseek_v41_draft_numerics.py
python -m pytest -q tests/test_deepseek_v41_*.py
```

Old/new use the same benchmark script and synthetic weights. The old serving
and DSpark modules were loaded from the baseline commit via
`--runtime-module baseline_native.serving --label old`; dependencies and target
were unchanged. The provider must expose `DSpark`, `DSparkRuntime`,
`load_product_runtime`, both draft installers, `_match`, and `stream_generate`.
Each measured run uses fresh caches; both paths are warmed first. Prefill is
excluded. Verify/draft timings force evaluation; host time is the round residual,
including observation, rollback, seed readback and streaming/detokenization.
The first seed readback precedes the timer. Instrumentation adds overhead.

Median of three 64-token CPU runs (sequential runs, not a hardware speed claim):

| Measurement | Baseline | Changed |
| --- | ---: | ---: |
| Plain target tok/s | 543.31 | 506.32 |
| Speculative tok/s | 211.29 | 222.38 |
| Draft ms/round | 2.841 | 2.037 |
| Verify ms/round | 1.869 | 2.060 |
| Host ms/round | 0.422 | 0.477 |
| Total ms/round | 5.133 | 4.568 |
| Accepted draft tokens/round | 0.0678 | 0 |
| Acceptance histogram, each repeat | `{0: 55, 1: 4}` | `{0: 63}` |
| Plain/speculative token equality | false in all repeats | true in all repeats |

Changed speculative throughput is about 5.2% above baseline in this smoke;
draft time falls about 28%. Plain-target throughput also varies about 7% between runs, so this is noisy evidence. Host time increases slightly, including the required
ring restoration work. The incorrect baseline follows a different token/round
mix; this comparison cannot isolate an acceptance benefit. The untrained draft
still loses to plain target decoding on CPU. Acceptance and GPU speed cannot be
predicted from these numbers. JSON includes methodology hash, schema version,
per-run equality, throughput, histogram and timing splits.

Verification results:

- New numerics suite: 35 passed.
- Entire `tests/test_deepseek_v41_*.py`: 205 passed, 12 skipped, 2 failures.
  The same two failures were reproduced **before editing** on unchanged sources
  (170 passed, 12 skipped):
  `test_target_qmv_install_and_small_route_execution` requires Metal;
  `test_disk_engram_bypass_paths_handle_empty_and_uncached_rows` hits unsupported
  empty-array dequantization on this CPU build.
- Benchmark metadata, Apple coverage union and no-MLX marker contracts:
  59 passed.
- Changed executable lines: 214/214 covered (100%), including the benchmark
  entrypoint and checkpoint-provider dispatch. Coverage runs preload NumPy
  before starting pytest instrumentation to avoid an environment-specific
  extension reimport error.
- Ruff check and format check pass for all changed Python files.
- Python 3.11 isolated environment, exactly `config/mypy-requirements.txt`:
  `python scripts/check_mypy_error_budget.py` passes, 821 grandfathered errors
  across 158 files, no growth.
- `git diff --check` passes. No target-model files or prefill-loop lines changed.

## Pending GPU measurement and handoff

Vector owns checkpoint acceptance/numerical qualification; Harbor owns the
Apple CI follow-through. This document records the intention, scoped changes,
evidence, risks and next action.

`DSpark.observe`/`observe_many` are textually identical to the prefill change, so
the two compose without a merge conflict. Rows are projected one at a time and
evaluated together: a batched projection rounded differently from a single-row
one (CPU dense GEMM and Metal), which made draft state depend on how verified
rows happened to be chunked.

On an isolated M3 Ultra, set `TARGET_MODEL` and `DRAFT_MODEL` to existing local
checkpoint directories. Use the same revisions, prompt, token count, precision,
QMV installation, hardware, and benchmark script in separate old/new processes.
Set `BENCH_BASELINE_DIR` to a policy-compliant scratch directory. This prepares
an importable baseline provider without changing either checkout or target:

```sh
export BASELINE_REV=59dfabfa2955a5eeb24821bcd3e1133a91f96ea5
python - <<'PYBASELINE'
import os
import subprocess
from pathlib import Path

root = Path(os.environ["BENCH_BASELINE_DIR"]) / "baseline_native"
root.mkdir(parents=True, exist_ok=True)
(root / "__init__.py").write_text(
    "from pathlib import Path\n"
    "import importlib, sys\n"
    "import rapid_mlx.models.deepseek_v41_native as native\n"
    "__path__.append(str(Path(native.__file__).parent))\n"
    "sys.modules[__name__ + '.load'] = importlib.import_module("
    "'rapid_mlx.models.deepseek_v41_native.load')\n"
)
for name in ("dspark.py", "serving.py"):
    source = subprocess.check_output(
        ["git", "show", os.environ["BASELINE_REV"] +
         ":rapid_mlx/models/deepseek_v41_native/" + name], text=True
    )
    (root / name).write_text(source)
PYBASELINE
export PYTHONPATH="$BENCH_BASELINE_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

Keep all other native dependencies identical. Exact measured commands:

```sh
python -m scripts.bench_deepseek_v41_draft \
  --model "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
  --runtime-module baseline_native.serving --label old \
  --tokens 256 --repeats 5 --seed 7 > draft-old.json
python -m scripts.bench_deepseek_v41_draft \
  --model "$TARGET_MODEL" --draft "$DRAFT_MODEL" --label new \
  --tokens 256 --repeats 5 --seed 7 > draft-new.json
```

Prompts are rendered through the serving chat template and every built-in
class (QA, code, structured output, summarization) is reported separately with
tokens actually emitted per round (normally accepted draft tokens + 1, fewer
when a token limit cuts the round short). Runs live under `classes[i].runs`.
The top-level `output_format_version` is 2 for this per-class layout; version 1 used flat `runs`.
`--component-timers` adds synchronized draft-component timings to each run's
`draft_on.mean_round_ms`; these diagnostic boundaries perturb latency.
Add `--mixed-target-precision`
for checkpoints the server loads that way. `--raw` encodes text without the
chat template; its output is labelled `"prompt_format": "raw"` and must not be
quoted as a chat result, because repetitive raw text accepts nearly every draft
token. On full checkpoints draft-on output is exact with respect to the verify
forward but may fork from single-token greedy decode (`first_difference`),
because batched verify reduces in a different order. Inspect token equality rather than
assuming it: five-token versus one-token GPU kernels may disagree near ties.
Measure real bf16/affine projection equivalence, target logits, acceptance
histograms, memory impact of live ring snapshots, draft/verify/host times, and
end-to-end throughput. Snapshots retain at most five KV rows per layer; allocator
and lazy graph lifetime still require measurement. Verify compressed-state and
Engram rollback with the full checkpoint; the CPU smoke has neither subsystem.

Expected outcome remains a **target**, not a result: +10–30% single-stream
speculative throughput over today's draft path. No GPU speedup or acceptance
increase has been established. The bounded ring restoration is required for
correctness independently of performance.
