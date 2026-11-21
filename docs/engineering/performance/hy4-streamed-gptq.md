# Hy4 Flash layer-streamed GPTQ

`scripts/hy4_gptq.py` builds native MLX affine text checkpoints from an
immutable local `hy_v4_flash` block-FP8 checkpoint. It never imports Python
from the checkpoint. The adapter uses the repository's Hy4 decoder, routing,
clamping, attention, Engram and hyper-connection code.

The shared `streamed_gptq_quant.py` contains the existing column-ordered GPTQ,
weighted Gram accumulation, damping and packing implementation unchanged.
The original builder imports compatibility exports. Both builders share the
RSS watchdog and MLX allocator limits in `streamed_gptq_memory.py`.

| Tensor family | `hy4-gptq-v1` |
| --- | --- |
| Routed expert gate/up/down | Q4, group 64, router-weighted GPTQ |
| Shared expert, dense MLP, latent expert projections | Q8, group 64, RTN |
| MLA, sliding attention, indexer projections | Q8, group 64, RTN |
| Router, expert bias, norms, hyper-connections, sinks, pooling gate | Source floating precision |
| Token embedding, output head | Q8, group 64; `--keep-embeddings-bf16` preserves source precision |
| Engram tables | Q4, group 32, RTN |
| Other Engram parameters | Source precision |
| Vision and auxiliary draft layers | Omitted |

`hy4-gptq-lowbit` is an unqualified configuration tier: gate/up use Q2 and
down uses Q3. `--gate-up-3bit-layers 1,12,22` promotes whole projection banks
in the specified zero-based layers to Q3. Choose promotions from measurements;
the tool does not invent pilot results. All experts in each bank share the
same native specification. Packed gate/up input banks are split consistently.

Engram weights use row-major native `weight`/`scales`/`biases` tensors,
with no alternate codec or interleaved row records. Conversion reads bounded
row chunks. Candidate calibration loads each layer's quantized table resident.
The reference instead reads selected FP8-dequantized rows, keeping full source
tables out of memory. SSD prefetch integration remains a separate runtime task.

Calibration manifests use the existing `{"slices": {"name": ["file"]}}`
format, with paths relative to the manifest. Include `english`, `chinese`,
`code`, and `tools` slices. Plain text becomes a user message; a JSON file
contains `messages` and optionally `tools`, allowing complete tool-call turns.
Each conversation is rendered locally using the source chat template twice,
with `reasoning_effort=high` and `reasoning_effort=no_think`. Each rendered
conversation supplies at most `--tokens` tokens. Provide separate held-out
files: overlapping content hashes and identical token windows are rejected.
The tokenizer uses local files only and disables checkpoint Python execution.

Set `PYTHON` to the approved contributor interpreter. Set `SRC` to the existing
read-only source snapshot; set `TRAIN` and `HELDOUT` to calibration manifests.
Set `SMOKE`, `OUT`, and `QUALITY` to separate writable artifact directories.
These commands require native MLX and do not download weights:

```bash
export RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1 PYTHONPATH=.

# Three-layer checkpoint, including both attention types and Engram.
"$PYTHON" -m scripts.hy4_gptq build --source "$SRC" \
  --destination "$SMOKE" --slices "$TRAIN" --recipe hy4-gptq-v1 \
  --only-layers 0,1,2 --tokens 512 --memory-limit-gb 55

# Full checkpoint. Repeat the identical command after an interruption.
"$PYTHON" -m scripts.hy4_gptq build --source "$SRC" \
  --destination "$OUT" --slices "$TRAIN" --recipe hy4-gptq-v1 \
  --tokens 512 --memory-limit-gb 55

# Full-vocabulary KL and top-1 agreement against streamed source forward.
"$PYTHON" -m scripts.hy4_gptq quality --source "$SRC" \
  --destination "$OUT" --slices "$HELDOUT" --recipe hy4-gptq-v1 \
  --work "$QUALITY" --tokens 512 --memory-limit-gb 55
```

The default cap is 55 decimal GB; values at or above 60 GB are rejected.
The shared watchdog exits on excess RSS or MLX active/cache bytes, including
inside long native operations. This bounds execution by failing closed; it is
not a measured full-checkpoint memory qualification. Inspect the reported peak
RSS from an actual run before claiming that qualification.

The source inventory hashes shard payloads, config and tokenizer metadata on
each invocation, so the initial scan is substantial. Build receipts bind the
source, recipe, calibration hashes, token windows, tensor payloads and residual
boundaries. A lock excludes concurrent writers. Only fully committed layers
are reused; an interrupted layer is rebuilt. A truncated build must use its own
destination, because the layer prefix is part of the resume identity.

`size-report.json` records tensor count, actual safetensors file bytes, layers,
recipe, identity and observed memory peaks. `build-state/` holds resume receipts
and residuals and must be retained for resume and qualification. Top-level
checkpoint files can be loaded with the repository's Hy4 model implementation.
The quality command writes `quality.json` with mean/p99 KL, top-1 agreement,
token count and whether the result is truncated. It uses the existing ruler's
KL/top-1 metric functions and finishes reference inference before candidate
inference. Truncated logits are diagnostic and do not qualify a full model.

Synthetic native checks live in `tests/test_hy4_gptq_native.py`; CPU naming,
FP8, corruption, numerical and transaction contracts live in
`tests/test_hy4_gptq.py`. Native loading/forward, RSS and real-source quality
must be checked on a host where MLX can initialize its device.

Reproduce the scoped checks with the approved interpreter:

```bash
changed_python=(
  scripts/deepseek_v41_gptq.py scripts/deepseek_v41_gptq_quant.py
  scripts/hy4_gptq.py scripts/hy4_gptq_adapter.py
  scripts/streamed_gptq_io.py scripts/streamed_gptq_quant.py
  scripts/streamed_gptq_memory.py
  tests/test_hy4_gptq.py tests/test_hy4_gptq_native.py
)
"$PYTHON" -m ruff check "${changed_python[@]}"
"$PYTHON" -m ruff format --check "${changed_python[@]}"
"$PYTHON" -m pytest tests/test_hy4_gptq.py -q
"$PYTHON" -m pytest tests/test_hy4_gptq_native.py -q
"$PYTHON" -m pytest tests/test_deepseek_v41_gptq*.py -q
```

On a machine without a device, run the CPU subset separately. Excluding its
one native reservation test is an environment-limited subset, not a replacement
for passing all original builder tests:

```bash
"$PYTHON" -m pytest tests/test_hy4_gptq.py tests/test_hy_v4_flash.py \
  tests/test_deepseek_v41_gptq_fullrun.py \
  -k 'not remote_reservation_and_standalone_cli_share_lock' -q
```

## Opt-in v2 diagnosis recipe

`hy4-gptq-v2` addresses sparse expert calibration with a larger offline token
budget, squared routing weights in the Hessian, and a width-relative low-token
policy. It is unqualified: no full-model quality or resident-memory measurement
is implied. The legacy v1 and Engram8 recipe fields, specifications, weighting,
and damping policies are unchanged. Existing receipt and quality identity checks
remain strict; modified tool hashes still invalidate an older build epoch.

```bash
"$PYTHON" -m scripts.hy4_gptq build --source "$SRC" \
  --destination "$OUT" --slices "$TRAIN" --recipe hy4-gptq-v2 \
  --tokens 512 --memory-limit-gb 55
```

Calibration first takes the normal high/no_think windows from all manifest
records, then round-robins through successive contiguous windows from the same
rendered offline records. The default budget is four times the actual v1 token
count for this manifest and window size. `--calibration-token-budget 131072`
sets an explicit hard budget; the final window is trimmed to fit it. Sources
are never repeated to fill a budget. Insufficient text produces an explicit
`source_exhausted` report. The CLI reports each manifest path's existence before
calibration; missing sources still fail rather than silently changing the mix.
For the local `calib-train.json`, all 26 files exist: English 9, Chinese 6,
code 2, tools 1, multilingual 8. No sources or weights are downloaded.

Router-aware adaptive top-up is not implemented. The layer-major streamer
commits residual boundaries for a fixed window set at every layer. A new
sequence introduced at layer N needs replay through the entire already-quantized
prefix; replacing that window set would also invalidate committed boundary
receipts and resume checks. V2 therefore uses the permitted fixed larger budget
and prints layer coverage (fraction of all routed expert IDs with at least
`--min-tokens`, default 1024), including unseen experts and whether the 98% goal
was met. Expert counts persist in the existing layer receipts. Coverage is a
diagnostic, not a guarantee or an adaptive stopping condition.

Routing weights contribute as `w² * x xᵀ` by default, including the fitted
intermediate activations used for down projections. Use
`--no-routing-weight-squared` to retain linear weighting for diagnosis. Raw
routed-token counts determine the low-token policy independently of weight mass:
GPTQ relative damping is 0.1 at or above the input width, increases linearly to
0.2 at half the width, and switches to RTN below half the width. Unseen experts
also use RTN. The coverage threshold does not change this width-relative policy.

The default map uses routed down Q6/g64 in every sparse layer, and gate/up
Q6/g64 in zero-based layers 20–26. Remaining tensors follow v1, including Q4/g32
Engram tables. `--bits-map '{"20":{"gate_proj":4,"up_proj":4}}'` or
`--bits-map /path/to/map.json` replaces selected layer/projection entries; omitted
entries keep the v2 defaults. Values may be 2, 3, 4, 5, 6 or 8, with group 64.
Overrides must name existing sparse layers. Each bank has one specification for
all experts; separate banks and layers may differ. Use the same recipe options
for the quality command; held-out calibration windows retain v1 truncation.
All v2 options are serialized into its recipe identity.

Use `--rtn-projections down_proj` to quantize routed down projections with plain
affine RTN in every sparse layer, skipping their Hessian accumulation. The option
accepts a comma list from `gate_proj,up_proj,down_proj` (default empty); unlisted
projections retain GPTQ and its low-token fallback. Bits and group size still
come from the v2 bits map. To use down Q6 with gate/up Q4 everywhere, override
both gate/up entries to 4 for each sparse layer in 20–26 via `--bits-map`.
Pass the same RTN option to `quality`; it is part of the recipe identity and
changing it invalidates resume receipts.

Mixed-bank loading uses the existing Hy4 load predicate, which returns
`config["quantization"][module_prefix]` to `nn.quantize`. Output entries use
`{"bits":6,"group_size":64,"mode":"affine"}` under the canonical module path
without `.weight`. `QuantizedSwitchLinear` stores bits/group per module and
passes them to `gather_qmm` on each call. MLX's affine quantization supports
[6-bit weights](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html);
its [gather_qmm API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)
accepts the specification per call. No loader change is needed. CPU synthetic
checks cover config entries, Hessians, fallback dispatch and six-bit packing;
native kernel execution remains unverified by these checks.

These estimates count stored tensor payloads. File headers, loader precision
promotions, SSD row caches, KV state and runtime workspace add overhead.

Metadata-only payload estimates for the cached 42-layer source snapshot
`003107fa248f5892064ab1c43af0a7003dd68c9b` are:

| Recipe | On disk (decimal GB) | Resident weights with Engram offloaded (GB) |
| --- | ---: | ---: |
| v1 | 130.79 | 91.19 |
| v2 default | 147.40 | 107.80 |
| Delta | +16.61 | +16.61 |

These include packed weights and BF16 scales/biases, exclude safetensors headers,
build-state residuals, KV cache, runtime workspace and SSD row caches, and are
not measured RSS. The requested default map exceeds the approximately 100 GB
resident target even before runtime overhead. The CLI prints estimates for the
chosen source, map and layer prefix before starting inference and records them
in `size-report.json`. Use a reduced promotion map if 100 GB is a hard limit.
