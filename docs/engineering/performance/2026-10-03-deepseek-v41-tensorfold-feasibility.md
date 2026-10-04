# DeepSeek V4.1 Flash and TensorFold MLX feasibility

Date: 2026-10-03. Host: Mac Studio M3 Ultra, 256 GiB. This is a compatibility
probe, not a throughput result.

## Rapid baseline

Rapid serves DeepSeek V4.1 Flash through the experimental
`deepseek-v41-flash-reap-2bit` profile. Its immutable target is
`rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX@a25fec277b9e7cedc0e9f3f15da874a5cf9d491b`;
the DSpark sidecar is
`rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX@9530d6d2bf59e0d05177bd538095d5704ded1488`.
Both snapshots were already in the default Hugging Face cache. The native
runtime uses a 40-layer, hidden-size-5120 target with affine 2-bit weights,
Engram, staggered hyper-connections, and a fixed K4 DSpark verification path.
The existing four-workload qualification reported 19.39 tokens/s versus 9.58
tokens/s autoregressive; this was a *Rapid native* result, not TensorFold.

## TensorFold compatibility probe

The installed TensorFold distribution identifies as 0.5.0. Its `detect()`
rejected the cached Rapid V4.1 target before loading any weights:

```
ValueError: TensorFold has no recipe for model_type 'deepseek_v41' yet
```

Reproduce without a model download or model load:

```sh
python - <<'PY'
from pathlib import Path
from tensorfold.families import detect
snapshot = (Path.home() / '.cache/huggingface/hub/'
            'models--rapid-mlx--DeepSeek-V4.1-Flash-REAP-2bit-MLX/'
            'snapshots/a25fec277b9e7cedc0e9f3f15da874a5cf9d491b')
print(detect(snapshot))
PY
```

TensorFold upstream `609ca419abecebdc5a059498a613680bd3aa847f` exposes
`deepseek_v4` for DeepSeek V4 Flash. That family expects 43 layers, hidden
size 4096, 4-bit affine group-64 base weights, and mxfp4 routed experts. Its
compression ratios and attention/cache semantics also differ from V4.1.
Changing only the model type does not fix the weight and architecture mismatch.
The upstream V4.1 contribution plan, issue #299, targets CUDA on two DGX
Sparks. PR #300 introduces CUDA interfaces; it does not add a V4.1 family or
Metal kernels. A later comment on issue #299 reports a **separate Mac MLX
V4.1 prototype**. Its authors measured 35.1-38.1
tokens/s on code and 26.3-28.0 tokens/s on prose with DSpark on an M3 Ultra
512 GB host, using `Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp`. They report 287 GiB
resident weights and about 305 GiB peak RSS. This is real prototype evidence,
but the code was not in public `main` at this probe's date, and that checkpoint
cannot fit this 256 GiB Studio. It is a different checkpoint from Rapid's
REAP 2-bit target, so its speed numbers are not a same-weights comparison.

## Update: 2026-10-04

The Mac implementation is now public as four open, stacked TensorFold PRs:
[#369](https://github.com/ashhart/TensorFold/pull/369) adds the serial MLX
family, [#370](https://github.com/ashhart/TensorFold/pull/370) adds DSpark,
[#371](https://github.com/ashhart/TensorFold/pull/371) adds exact decode-row
kernels, and [#372](https://github.com/ashhart/TensorFold/pull/372) shares
weight reads and reads Engram pages concurrently. None is merged into `main`.
The latest PR reports upstream-tool greedy decode results on the 512 GiB M3
Ultra of 29.1 tokens/s for code and 27.3 for chat; its cold Engram read-ahead
probe reports 82 -> 244 prompt tokens/s. These are the authors' measurements
on their oQ4e checkpoint, not a comparison with Rapid's REAP 2-bit checkpoint.
The first PR's checklist still leaves real-checkpoint resumed-prompt and cold
prefill tool checks open. A separate CUDA V4.1 family is open as draft #342.

## Read-ahead MVP: 2026-10-04

Rapid already starts Engram table reads on a background worker before the
corresponding model layer runs. TensorFold #372 adds another step for long
prompts: it reads selected mmap pages concurrently before gathering the rows.
The experimental Rapid port applies that idea to the existing affine 2-bit
disk table, preserving the current checkpoint and tensor layout. It is off by
default. The qualification suite enables it with
`--engram-ssd-offload --experimental-engram-page-read-ahead`.

The standalone I/O probe used the pinned cached REAP 2-bit target, layer 1's
actual Engram shard, 12,288 uniformly selected rows per trial, two alternating
serial/parallel pairs, and no model load. Run it with:

```sh
python scripts/benchmark_deepseek_v41_engram_pages.py \
  --snapshot <cached-target-snapshot> --layer 1 --rows 12288 --repeats 2
```

Serial selected-row reads took 3.621 and 3.156 seconds; parallel reads took
0.366 and 0.362 seconds. The median ratio is 9.3x **for file I/O only**.
Both modes returned 983,040 bytes of rows per trial; a separate identical-index
check found all three packed arrays equal. On a repeated warm lookup, the
parallel path takes about 0.005 seconds because already-prefetched pages are
skipped. The focused Engram and native-load suites pass (54 tests). This does
not establish a model-level prefill or decode gain. The Studio still has over
13 GiB swap in use and other large model processes, so a 200+ GiB model run
would violate the existing clean-memory qualification gate.

The current Studio has 13.1 GiB swap in use and other large model processes.
Under the large-model qualification policy, this invalidates a new 200+ GiB
performance capture. No V4.1 TensorFold speed claim was made.

## Engineering decision

An honest MLX MVP needs the reported V4.1-specific TensorFold family to be
published and adapted to the 2-bit REAP checkpoint, or a focused port of its
row-kernel/lane ideas into Rapid's existing V4.1 runtime. A V4 profile
alias would load the wrong architecture and must not ship. The first real
speed gate is a same-checkpoint, same-prompt, same-host comparison against
Rapid's current K4 path, with serial and speculative output agreement, warmup,
alternating run order, clean memory pressure, and separate prefill and decode
measurements. Keep the V4.1 TensorFold profile out of the catalog until that
gate passes.

Upstream: https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f/src/tensorfold/families/deepseek_v4
and https://github.com/ashhart/TensorFold/issues/299 .
