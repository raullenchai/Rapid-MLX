# Qwen3.5-family GDN qualification on M4 Pro

This supplements the earlier [long-history qualification](2026-10-10-qwen35-native-gdn-qualification.md)
for issue #4448 with a separately leased M4 Pro run. The production adapter,
kernel, admission rules and defaults are unchanged. No newly admitted numerical
defect was established, and no performance or universal hardware claim is made.

## Source and runtime

- Clean source: `ff7a54403aa9f256460602194059c2896941c13d`.
- Apple M4 Pro, 48 GiB unified memory, macOS 26.5.1 (25F80).
- Python 3.12.13, MLX/Metal 0.32.3, native model package 0.31.3, NumPy 2.4.4.
- Transformers 5.15.1; the complete package inventory is retained with raw evidence.
- Two immutable checkpoints copied from the existing cache into the worker's
  default Hugging Face cache. Snapshot entries such as
  `model-00001-of-00004.safetensors` are symlinks to `../../blobs/<64-hex-SHA256>`.
  Before loading, the diagnostic streams every shard through SHA-256 and checks
  it against the resolved blob's filename. The verified content digest is also
  recorded in the supplement; ordinal snapshot filenames are not hash inputs.

The shared kernel source includes the admission check merged since the earlier
capture. These results use a new source inventory; the earlier evidence has not
been relabeled as a run of current code.

## Numerical contract

Each checkpoint runs in a fresh process using the existing diagnostic command.
Each trajectory has fresh native prefill and independent persistent native and
fused caches, 128 or 8192 prompt tokens, and 256 decode steps. Both stock-first
and fused-first orders run serially. Every eligible layer/step compares the
normalized output, projected output, convolution cache and FP32 recurrent state
by storage bits, and retains hashes and differing-element counts. Silent fallback,
non-finite values and incomplete execution fail qualification.

## Results

Both checkpoints admitted all 30 GDN layers with threadgroup Y=32. Each completed
four trajectories of 256 steps and 7,680 unique layer rows. All 61,440 rows and
245,760 tensor comparisons passed: finite values, zero differing storage elements,
and equal native/fused hashes. Generated token sequences agree between both
orders for each checkpoint/history. The audit checked the full row key space,
not only totals or final token equality.

| Checkpoint | Immutable revision | Histories | Layer rows | Result |
| --- | --- | --- | ---: | --- |
| Qwen3.5-35B-A3B-4bit | `1e20fd8d42056f870933bf98ca6211024744f7ec` | 128, 8192 | 30,720 | exact |
| Qwen3.6-35B-A3B-4bit | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | 128, 8192 | 30,720 | exact |

The [machine-readable supplement](2026-10-10-qwen35-m4-gdn-qualification.json)
records source/runtime identities, immutable shard hashes, trajectory/token
digests and raw-artifact hashes. This establishes native/fused equality within
the M4 matrix; equality to the earlier M3 run's individual tensors is not claimed.

## Execution evidence and limits

The machine was reserved through an active, non-stale resource lease. The model
processes run serially. An external observer retains timestamped process lists
and the accelerator's GPU-client registry before, during and after each model
process. A separate observer retains lease state and expiry samples; the
acquisition receipt identifies the lease and epoch. These records support an
audit of competing model workloads, rather than inferring exclusivity from a
single idle-utilization reading or the diagnostic's cooperative lock.

| Checkpoint | UTC model window | Successful samples | Maximum observed gap |
| --- | --- | ---: | ---: |
| Qwen3.5 | 21:40:10–21:43:31 | 196 | 1.081 s |
| Qwen3.6 | 21:43:32–21:46:47 | 189 | 1.082 s |

Both external observers exited successfully with zero capture errors. The only
GPU client added relative to each run's desktop baseline was that qualification
process. The process inventory also contains a short-lived Python 3.9 process
whose workload was not attributed and which did not appear in the sampled GPU
registry. These records do not assert that no other CPU process ran.

Periodic lease observations began at 21:41:10 UTC, about one minute after the
first model started. The original acquisition receipt's expiry covered that
interval and the first observation. Subsequent observations and the receipt
after both model runs show the same active, non-stale lease/epoch and unexpired
heartbeats. The model processes and their GPU clients were absent at cleanup;
the worker lease was then released.

Normal macOS desktop GPU clients remain present. The observer does not enforce
GPU isolation, and finite sampling cannot exclude short unobserved work. The
diagnostic's `hardware_exclusive=false` remains unchanged. The bounded execution
claim is a dedicated leased model workload with no competing model compute
observed; hardware-wide exclusivity is not proved by these records. Under the
issue's strict hardware-exclusive interpretation, this supplement alone does
not close #4448.

One initial launch reached the diagnostic before all snapshot shards had finished
copying. It failed with `snapshot has no complete local weights`, completed no
trajectories, and is retained separately from numerical qualification. The
successful run began after the complete checkpoint was available. Startup errors
are not counted as passing trajectories or numerical mismatches.

## Reproduce

Acquire a dedicated worker lease, check for competing model workloads, and use
the pinned runtime and clean source above. Retain execution inventories throughout
the run. The diagnostic itself does not certify external hardware exclusivity.

```bash
python -m scripts.qualify_qwen35_fused_gdn \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec" \
  --histories 128 8192 --steps 256 \
  --output /private/tmp/validate-qwen35-long-2-exclusive/reproduce-35

python -m scripts.qualify_qwen35_fused_gdn \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46" \
  --histories 128 8192 --steps 256 \
  --output /private/tmp/validate-qwen35-long-2-exclusive/reproduce-36
```

86 focused tests passed on this worker, covering the qualification diagnostic,
production adapter, shared kernel and benchmark contracts. The final artifact
audit checks unique row identities, all four metrics, complete trajectories,
token equality between orders, source cleanliness and execution receipt hashes.

Every row, generated continuation, full process/GPU-client sample, sanitized
lease receipt, runtime package list, focused-test log and the external
collector/audit scripts are retained on Studio under
`/Volumes/RTL-2T/scratch-archive/rapid-mlx-4448/ff7a54403-m4-20261010/`.
The archive's `artifact-manifest-v2.json` hashes all 33 retained files; its SHA-256 is
`80146be3ff73dfc84a795b9df28ba6bed03649e5ea8193f1ab67df5fdc3aa466`.
The original summary/manifest remain intact; `summary-v2.json` and `audit-v2.py`
add per-trajectory exactness counts derived from the same immutable row streams.
Credentials and lease secrets are excluded from the archive and this supplement.
