# DeepSeek V4.1 1017 build and evaluation tooling

These offline tools build and evaluate derived 1017 checkpoints. Serving uses
existing runtime support; the tools do not change aliases, pins, request limits
or model loading. The dated performance reports record historical experiments,
not fresh qualification of a new build.

Create an isolated development environment with Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
# Needed only for dataset preparation / task prefetch from Parquet:
uv pip install --python .venv/bin/python pyarrow
```

The numerical tools require MLX. A sandbox without access to Metal needs a
matching CPU MLX library; set `DYLD_LIBRARY_PATH` to its build directory for
CPU tests only. GPU build and qualification runs require a Metal-capable host.

Export `DSV41_SCRATCH_ROOT` to a task scratch directory permitted by the host's
storage policy. The HTTP evaluation harness and full-run driver use that
variable; their default is a task directory below the system temporary root.
Set `RD_DATA` to approved local dataset exports, `SOURCE` to the immutable
revision-pinned source snapshot, and choose separate immutable build outputs.
No launcher or model checkpoint is bundled with these tools.

| Stage | Entry points | Contract |
| --- | --- | --- |
| Corpus and scoring | `deepseek_v41_rd_corpus_prepare.py`, `calibrate_deepseek_v41_reap.py`, `deepseek_v41_merge_calibration.py` | Local provenance checks, explicit trust for checkpoint code, split stability |
| Allocation and packing | `deepseek_v41_expert_quant_error.py`, `deepseek_v41_rd_allocate.py`, `deepseek_v41_rd_layer_allocate.py`, `repack_deepseek_v41_native.py` | Measured cost/error inputs; explicit `--costs` for variant allocation; source expert identity retained |
| Source reference and KL | `deepseek_v41_streaming_teacher.py`, `evaluate_deepseek_v41_rd.py`, `deepseek_v41_public_ruler.py` | Layer streaming, source hashes, full or compact held-out reference checks |
| GPTQ pipeline | `deepseek_v41_gptq.py`, `deepseek_v41_gptq_scan.py`, `deepseek_v41_gptq_quality.py` | Bounded memory, committed checkpoint resume, authenticated recipes and artifacts |
| Full-run coordination | `deepseek_v41_gptq_fullrun.py` | Explicit host, remote workspace, remote interpreter, source and destination; export verification before cleanup |
| Incremental assembly | `deepseek_v41_gptq_assemble_stage.py` | Separate staging locks and hash verification; input run state stays immutable |
| HTTP qualification | `eval_deepseek_v41_tasks.py` | Connects to an already-running endpoint; never starts a server or loads weights |

HTTP `prefetch` writes a frozen manifest and an independent sibling named
`<manifest filename>.tokenizer.json`. Move both files together. Smoke and speed
commands resolve that sibling relative to the manifest and verify its digest;
`--tokenizer` can supply another local copy with the same digest. Interrupted
manifest publication can be retried; prefetch replaces its unpublished tokenizer
sidecar. Speed probes reserve 128 tokens for chat framing and qualify a result
only when the server-reported input count stays within the target budget.

All entry points live under `scripts/`; use `--help` for required inputs.
See the dated build reports for recipe-specific command sequences and the
[amendment procedure](gptq-recipe-amendment.md) before changing a recorded recipe.
Full-run coordination and recipe amendment must never run concurrently against
the same state. A launcher is maintained outside the repository; the optional
launcher contract test runs only when `DSV41_LAUNCHER` explicitly points to it.

For CPU verification, select the CPU device before pytest collection:

```sh
VECLIB_MAXIMUM_THREADS=2 .venv/bin/python - <<'PY'
from pathlib import Path
import mlx.core as mx
mx.set_default_device(mx.cpu)
import pytest
files = sorted(str(p) for p in Path('tests').glob('test_*deepseek_v41*.py'))
raise SystemExit(pytest.main(['-q', *files]))
PY
```

Metal-only cases skip with a CPU library. CPU evidence does not establish GPU
quality, throughput or memory use. Existing runtime failures should be reproduced
on the same main revision before being attributed to the extracted tooling.
CI enforces 100% changed-line coverage using the union of Linux and Apple
`rapid_mlx` coverage. That gate does not measure these `scripts/` modules;
focused script coverage must be inspected separately.
