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
