# Continuous self-MTP rollback constant factors (#4589)

Date: 2026-10-17. Owner: Vector.

## Result and scope

Reuse transaction-local validation plans across target/draft rollback, read each
KV offset vector once, and select uniform recurrent snapshots by reference.
For ragged KV rollback, share one full-capacity gather index across keys and
values, preserving masked padding and the reserved tail exactly.

Retain the original row-slice/concatenate algorithm for ragged recurrent
snapshots: every full-state vectorized candidate was slower on the representative
30-layer workload. Snapshot preflight now checks metadata without constructing
device selections. Both cache groups finish validation before either is applied;
standalone `trim_ragged_cache(validate=True)` retains that atomic validation.
Custom adapter hooks retain their existing call contract.

The rollback microbenchmarks below are model-free; end-to-end serving results
are reported separately.
Routing, admission, draft depth, defaults, dependency bounds, and cache layout
are outside this change.

## Environment and workload

- Apple M5 Max, 36 GiB unified memory; macOS 27.0.1, arm64.
- Python 3.12.13; MLX 0.32.4; mlx-lm 0.31.3.
- Baseline: `76b8296cd1cf3a009de045d719dd4823974afae2`; both arms run in
  fresh processes using the module extraction described in the appendix.
- Measured product source SHA-256 (before the validation-only compatibility fix):
  `b0fb2986c9d68a1ca4eafef1f93332b2145b2262557fe7f27ddd42b921cbaef7`.
- Cached Qwen3.6-35B-A3B-4bit config snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`; no weights loaded or downloaded.
- B=8, verify width=3, 40 target layers: 30 recurrent layers with slots
  `[8, 3, 8192]` BF16 and `[8, 32, 128, 128]` FP32; 10 KV layers with
  `[8, 2, capacity, 256]` BF16 keys and values. Capacity rounds context+3 up
  to a multiple of 256; initial left padding is `[0,1,2,3,0,1,2,3]`.
- Uniform drops `[1]*8`; ragged drops `[0,1,2,0,1,2,0,1]`. Draft drops are
  zero, so the draft cache is skipped as in the existing adapter contract.
- Independent contiguous, evaluated buffers per layer/boundary. Cache setup
  occurs outside timing. Each sample includes `adapter.rollback`, evaluation
  of resulting cache states, and `mx.synchronize()`; 5 warmups, median of 30.

## Full rollback medians (ms)

| Context | Uniform before | Uniform after | Ragged before | Ragged after |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4.383 | 0.353 | 6.252 | 4.099 |
| 4,096 | 4.375 | 0.345 | 19.809 | 9.024 |
| 16,384 | 4.345 | 0.347 | 70.532 | 25.977 |
| 32,768 | 4.366 | 0.358 | 136.138 | 48.348 |

## End-to-end serving (M5 Max)

Measured on the same M5 Max (36 GiB) with Qwen3.6-35B-A3B-4bit and its
4-bit MTP model. Two boots per arm ran in order plain → cohort-main →
cohort-patched → cohort-patched → cohort-main → plain. Each boot used a
single-request calibration, then one warm and three measured rounds per point.
Values are mean aggregate completion tok/s including prefill and HTTP;
ratios compare matching boot numbers. Short prompts contain 411–416 templated
tokens; every request returns 300 completion tokens with temperature=0,
stream=false, enable_thinking=false and chat_template_kwargs.enable_thinking=false.

| c | Boot | Plain tok/s | Cohort-main tok/s | Cohort-patched tok/s | Patched÷main |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 93.3 | 150.7 | 150.8 | 1.001× |
| 1 | 2 | 93.3 | 150.7 | 150.7 | 1.000× |
| 2 | 1 | 185.1 | 182.3 | 187.6 | 1.029× |
| 2 | 2 | 184.8 | 182.2 | 187.5 | 1.029× |
| 4 | 1 | 265.7 | 220.2 | 228.2 | 1.037× |
| 4 | 2 | 265.4 | 220.3 | 228.3 | 1.036× |
| 8 | 1 | 328.2 | 254.9 | 264.1 | 1.036× |
| 8 | 2 | 328.8 | 256.1 | 270.0 | 1.055× |

At c=1 the cohort arm cannot form a multi-lane cohort. All 48 short c=2/4/8
rounds admitted depth-2 cohorts; c=8 M2 admitted seven initial lanes in all
four cohort boots, versus eight in the other c=8 rounds. Patched serving gains
at c=2/4/8 were 2.9%/3.6–3.7%/3.6–5.5% over main; c=4/8 remained below plain.

Required parity comparisons were **64/64 identical**, with no divergence in
full messages, completion-token usage, or completion text re-encoded with the
target tokenizer. The API does not expose internal generated token IDs.
All non-timing speculative counters were identical across all four cohort
boots: 21,564 attempts, 13,426 accepts, acceptance ratio 0.6226.

The long workload (8,063–8,066 templated prompt tokens, c=2/4) formed no
continuous cohort in any of its 32 warm/measured rounds on 36 GiB. The existing
prefill memory gate estimated approximately 130 GB against a 27.1 GB cap.
Its throughput therefore does not measure continuous-cohort rollback gains.
**M3 Ultra long-context serving remains to be measured.**

Exact serve commands below preserve the measured flags and environment, with
checkout, executable and model paths generalized. `<snapshot path>` is the
target snapshot; `<MTP snapshot path>` is the draft snapshot. Run each command
in its indicated checkout with a Rapid-MLX dev install on PATH. The loopback
host is written as `localhost`; all other flags and environment values are unchanged.
Boot 2 repeats the patched, main and plain commands in that order, changing
only their ports to 18913, 18914 and 18915 respectively.

```bash
# plain, patched checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host localhost --port 18910 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 --no-spec-decode

# cohort-main, baseline checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host localhost --port 18911 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 \
  --speculative-config '{"method":"mtp","model":"<MTP snapshot path>","continuous_batching":true,"allow_dynamic_membership":true}' \
  --force-spec-decode

# cohort-patched, patched checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host localhost --port 18912 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 \
  --speculative-config '{"method":"mtp","model":"<MTP snapshot path>","continuous_batching":true,"allow_dynamic_membership":true}' \
  --force-spec-decode
```

The load driver uses `httpx` with one pooled connection per lane, releasing all
requests in a batch in the same event-loop turn against `/v1/chat/completions`.
Each user prompt starts with a deterministic 16-character SHA-256 nonce, then
"Background vocabulary:" and seeded random space-prefixed common words that
each encode to one token with the target tokenizer. Use 350 words for short
prompts, 8,000 for long prompts, and 600 for calibration; append: "Write a detailed
six hundred word explanation of how a reliable computer cache works. Include
examples, tradeoffs, and practical design advice. Continue until the explanation
is complete." Build fixtures once and reuse identical prompts across arms/boots;
verify served prompt counts against `apply_chat_template` with
`add_generation_prompt=True, enable_thinking=False`.

After readiness, send one calibration request (at least 512 templated tokens),
then short c=1/2/4/8 and long c=2/4 batches. At each point run one warm round and
three measured rounds, using distinct deterministic fixtures per round. Every
request uses `model="q36"`, `max_tokens=300` and the decoding settings above.
For each round, aggregate tok/s is the sum of API-reported completion tokens
divided by wall time from releasing the batch until all responses finish; report
the arithmetic mean of the three measured rates. Compare cohort-main and
cohort-patched by matching boot, workload, concurrency, round and request key;
require matching prompt hashes, full response messages, completion-token counts,
and completion text re-encoded with the target tokenizer
(`add_special_tokens=False`). Check server admission logs per round and
speculative counters before/after each workload.

These serving measurements used the product source hash recorded above, before
the later validation-only compatibility fix; serving was not rerun for that fix.

## Item decisions

For the default adapter, preflight calls fall from three to one per applied
cache; KV offset host reads fall from four to one. Recurrent snapshot selection
is absent from preflight, and uniform restores build no gather/concatenate.

One KV layer, ragged drops, synchronized 30-sample medians (ms):

| Context | Original roll | Full-index gather (retained) | Selected-row roll/scatter |
| ---: | ---: | ---: | ---: |
| 512 | 0.772 | 0.310 | 0.535 |
| 4,096 | 2.223 | 0.857 | 2.470 |
| 16,384 | 7.785 | 2.531 | 8.984 |
| 32,768 | 15.580 | 4.760 | 17.778 |

Snapshot selection across 30 independent recurrent layers, both slots,
alternating candidate order, synchronized 30-sample medians:

| Method | ms |
| --- | ---: |
| Original row slices + concatenate (retained) | 2.920 |
| Concatenate candidates + take | 10.496 |
| Stack candidates + take_along_axis | 13.225 |
| Mask/where | 7.900 |
| Compiled mask/where | 5.311 |

The compiled mask's isolated small-slot result was 0.399 ms versus 0.406 ms
for the original, while the large recurrent-state slot regressed from 2.761
to 5.154 ms. This does not justify adding a shape-dependent selection policy.
An initial full rollback candidate using compiled masks regressed the 512-token
ragged case; the final table above uses the retained original selection.

## Verification and reproduction

All real MLX work ran on the M5 Max. Final focused suite: **171 passed**;
final broad suite: **1,456 passed, 11 skipped**, 40,301 deselected (153.53 s).
Bit-exact comparisons cover random ragged/positive-ragged, uniform and zero KV
drops, left padding, reserved tails, differing K/V dimensions, BF16/FP32, and
recurrent boundary-list/legacy snapshots. Failure cases prove validation precedes
mutation; fake-cache instrumentation proves once-per-cache validation.

From the patched checkout, using `python` from a Rapid-MLX dev install:

```bash
export HF_HUB_OFFLINE=1
python -m pytest -q tests/test_mtp_ragged_cache.py \
  tests/test_continuous_self_mtp_mlx_backend.py tests/test_continuous_mtp_runtime.py \
  --cov=rapid_mlx.spec_decode.mtp.ragged_cache --cov-report=xml:coverage.xml
python -m pytest -q -k 'mtp or ragged or cohort or continuous'
diff-cover coverage.xml --compare-branch=origin/main \
  --include=rapid_mlx/spec_decode/mtp/ragged_cache.py --fail-under=100
```

After the validation-only compatibility fix, local fake-only verification of
`tests/test_mtp_ragged_cache.py` passes **58 tests**, including validation-only
leaves and rejection before mutation for standalone trim and adapter rollback.
That historical fake-only run disabled MLX imports and pytest plugin autoload
with an import-blocking wrapper; no GPU was used. The real MLX results above
precede this final fix. The focused command above runs the same test file from
the dev install without requiring that wrapper.

Current diff coverage is **80/80 changed product lines (100%)**. Local `ruff check`,
`ruff format --check`, and `mypy --follow-imports=silent` pass on all three
changed Python files. The added MLX tests live in the existing Apple CI-listed
`tests/test_continuous_self_mtp_mlx_backend.py`.

The appendix reproduces the full rollback workload without model files. Run
before/after in separate foreground processes with no concurrent test or
benchmark load. The tables above record the final full rollback and isolated
candidate medians; the appendix emits source hashes and raw timing samples for
new full rollback runs.

## O(verify-width) follow-up

The retained KV gather remains O(context). For post-uniform cursor `s`, row
residual `r`, and physical column `p`, it reads `(p-r) % s` when `p < s` and
`p` otherwise. Zero-residual rows and the reserved tail therefore retain every
bit. Starting at minimum left padding would change masked-padding bits and was
rejected under this equivalence contract.

Vector owns the layout follow-up: decouple per-row write/valid-end positions
from the shared right-aligned cursor, coordinating attention masks, append
positions, RoPE offsets, and merge/filter/extract contracts. Increasing left
padding alone leaves accepted history at incorrect physical positions. That
layout change is required for O(verify-width) rollback and is outside this task.

## Appendix: rollback micro-benchmark

Save the following as `bench_rollback.py` in the patched checkout root. It needs
`mlx`, `mlx-lm` and a Rapid-MLX dev install on Apple hardware with Metal support.
Extract the baseline module from the repository, then run each arm in a fresh
process using the same dev install and patched checkout:

```bash
git show 76b8296cd1cf3a009de045d719dd4823974afae2:rapid_mlx/spec_decode/mtp/ragged_cache.py > baseline_ragged_cache.py
python bench_rollback.py --mode before --baseline baseline_ragged_cache.py --repeats 30
python bench_rollback.py --mode after --repeats 30
```

The workload uses 30 recurrent layers and 10 KV layers, with every fourth layer
KV. Setup and template evaluation are outside timing; each sample includes
rollback, output evaluation and synchronization. Five warmups precede 30 measured
samples per context/pattern. JSON output includes the loaded module's source
hash, dependency versions, medians and individual samples.

```python
import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform
import statistics
import sys
import time
from importlib.metadata import version

import mlx.core as mx
from mlx_lm.models import cache as cm
from rapid_mlx.spec_decode.mtp.continuous_engine import SelfMTPCachePair

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['before', 'after'], required=True)
p.add_argument('--repeats', type=int, default=30)
p.add_argument('--baseline', default='baseline_ragged_cache.py')
a = p.parse_args()
if a.mode == 'before':
    name = 'rapid_mlx.spec_decode.mtp._baseline_ragged_cache'
    spec = importlib.util.spec_from_file_location(name, a.baseline)
    r = importlib.util.module_from_spec(spec)
    sys.modules[name] = r
    spec.loader.exec_module(r)
else:
    from rapid_mlx.spec_decode.mtp import ragged_cache as r
r.install_ragged_cache_rollback(cache_module=cm, qwen4_state_cls=None, qsa_cls=None)
assert cm.BatchKVCache.trim_ragged.__module__ == r.__name__
assert cm.ArraysCache.trim_ragged.__module__ == r.__name__
# Recorded model dimensions; no config, tokenizer, or weights are needed.
b = 8
shapes = [(b, 3, 8192), (b, 32, 128, 128)]
print(json.dumps(dict(mode=a.mode,
    source_sha256=hashlib.sha256(pathlib.Path(r.__file__).read_bytes()).hexdigest(),
    platform=platform.platform(), python=platform.python_version(),
    mlx=version('mlx'), mlx_lm=version('mlx-lm'))), flush=True)
adapter = r.RapidRaggedCacheAdapter()
for ctx in (512, 4096, 16384, 32768):
    # Independent, evaluated buffers for every layer and boundary. No weights loaded.
    templates = []
    for layer in range(40):
        if (layer + 1) % 4 != 0:
            states = [[mx.contiguous(mx.full(s, boundary+1, dtype=mx.bfloat16 if slot == 0 else mx.float32)) for slot,s in enumerate(shapes)] for boundary in range(3)]
            mx.eval(states)
            templates.append(('gdn', states))
        else:
            shape = (b, 2, ((ctx+3+255)//256)*256, 256)
            kv = [mx.contiguous(mx.full(shape, v, dtype=mx.bfloat16)) for v in (1, 2)]
            mx.eval(kv)
            templates.append(('kv', kv))
    mx.synchronize()
    for pattern, drops in [('uniform', [1]*b), ('ragged', [0,1,2,0,1,2,0,1])]:
        timings = []
        for iteration in range(5+a.repeats):
            group = []
            for kind, arrays in templates:
                if kind == 'gdn':
                    cache = cm.ArraysCache(size=2)
                    cache.cache = list(arrays[2])
                    cache.rollback_state = [tuple(arrays[0]), tuple(arrays[1])]
                else:
                    cache = cm.BatchKVCache(left_padding=[0,1,2,3,0,1,2,3])
                    # Fresh array wrappers prevent slice assignment from mutating templates.
                    cache.keys, cache.values = [mx.array(x) for x in arrays]
                    cache._idx = ctx+3
                    cache.offset = mx.array([ctx+3-x for x in [0,1,2,3,0,1,2,3]])
                group.append(cache)
            pair = SelfMTPCachePair(group, [cm.BatchKVCache(left_padding=[0]*b)])
            mx.eval([x.state for x in group])
            mx.synchronize()
            start = time.perf_counter_ns()
            adapter.rollback(pair, target_drops=drops, draft_drops=[0]*b, verify_width=3)
            mx.eval([x.state for x in group])
            mx.synchronize()
            elapsed = (time.perf_counter_ns()-start)/1e6
            if iteration >= 5:
                timings.append(elapsed)
        print(json.dumps(dict(mode=a.mode, context=ctx, pattern=pattern, median_ms=statistics.median(timings), samples_ms=timings)), flush=True)
        del pair, group
    del templates
    mx.clear_cache()
```
