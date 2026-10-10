# GLM-5.3 Flash load-window memory

Issue #3709 concerns loading the ordinary 4-bit GLM target on a shared
256 GiB Studio. A successfully resident model does not establish that the
loading transient fits: checkpoint pages, source tensors, and repacked
destination tensors can coexist.

## Two materialization boundaries

Both the ordinary multimodal loader (`--no-spec-decode`) and the default
native-MTP target loader now use the same Darwin-only loading adapter:

1. Read and evaluate one safetensors shard, then request file-cache eviction
   through the existing read-only UBC helper before reading another shard.
2. Evaluate the sanitized parameter tree in batches totaling at most 1 GiB,
   or one larger tensor, clearing unused allocator cache between batches.

Both steps are necessary. An intermediate implementation that evaluated shards
eagerly but retained the final whole-tree evaluation still produced substantial
compression during expert-weight repacking. Evicting only after the loader
returns is also too late to bound the load window.

The adapter binds the pinned loader's functions to private per-call namespaces.
It does not replace process-global functions. The existing architecture loader
still owns sanitization, quantization, strict missing-key validation, processor
construction, immutable revisions, and remote-code policy. Other architectures
and non-Darwin platforms keep their existing paths. A structurally incompatible
loader fails explicitly instead of silently reverting to the unbounded path.

Eviction is best effort. A successful syscall reports requested invalidation
bytes, not a measurement of pages actually reclaimed. This change does not make
a model fit when its steady footprint and other resident workloads exceed RAM.

## Real-checkpoint verification

Environment: Apple M3 Ultra, 256 GiB, macOS 26.5.2, Python 3.12.14,
MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.2. The existing cached target was
`Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`:
43 shards, 181,709,451,790 bytes, largest shard 4,295,149,840 bytes.
No checkpoint download or background-service shutdown was needed.

For the ordinary multimodal path, the full target loaded, `/health` returned
healthy, and a 24-token prompt generated a 32-token HTTP 200 completion. During
the startup samples, file-backed memory peaked at 15.02 GiB, physical compressor
memory stayed at approximately 30.47 GiB, and swap stayed at 8.833 GiB. These are
whole-machine samples, not per-model allocation measurements. Existing background
workloads remained running; there was no controlled constant-40-GB co-tenant in
this successful run. An earlier extra-40-GiB stress attempt was stopped on memory
pressure and is not claimed as a pass.

The default native-MTP path also needs the cold-sidecar loader correction already
merged in #4420. The initial older-base run completed target materialization but
then failed in sidecar configuration, independently of memory. Validation of the
default command must therefore use a base containing that correction.
After updating to `226f6bcddfd87360def0f4c70d3f119e1697c2ea`, the default
native-MTP command loaded the target and sidecar successfully, `/v1/models`
returned the served alias, and the same short prompt produced a 32-token
HTTP 200 completion. File-backed memory peaked at 18.89 GiB; compressor
memory and swap again remained stable. The request reached its token limit
while reasoning, so this is a generation smoke rather than an answer-quality
assertion. The distilled samples and source digests are in
[the verification fixture](fixtures/glm53-load-window-2026-10-10.json).

These are startup and inference smoke checks, not a throughput comparison or a
guarantee of success under arbitrary co-residency. The real-model processes were
terminated after validation, and their machine lease was released after cleanup.

## Reproduce

Use a source checkout containing this change and the repository-pinned vision
runtime. Keep the ordinary model cache and ensure the target and MTP sidecar are
already present for an offline run. On a shared host, obtain its machine lease
first and monitor memory while loading:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  python -m rapid_mlx.cli serve glm5.3-flash-4bit \
  --host 127.0.0.1 --port 18709 --disable-model-downloads
```

Repeat with `--no-spec-decode` for the ordinary multimodal path. In another
terminal, sample `vm_stat` and `sysctl vm.swapusage` throughout startup. The
`GLM shard materialized` log must occur between successive shard reads. The
native server exposes `/healthz`; the ordinary server exposes `/health`. Both
expose `/v1/models` and `/v1/chat/completions`.

Regression checks cover ordering, failed evaluation, isolated concurrent calls,
platform/family routing, immutable revisions and trust-policy forwarding,
bounded final evaluation, and exact quantized tensor equality with the pinned
loader on a small local GLM checkpoint. A missing parameter still fails strict
loading. Rollback is to revert the adapter and its two call sites; doing so
restores the original transient risk.
