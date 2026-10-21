# DeepSeek V4.1 Flash 1017 serving qualification

Paths in command examples use owner-supplied environment variables. Set
`BUILD_DIR` to a complete local 1017 build (target plus co-located `dspark/`).

## Lane contract

The serial V4.1 DSpark K4 product lane selects its artifacts from
`PRODUCT_ARTIFACTS` in `rapid_mlx/models/deepseek_v41_native/artifacts.py`.
The released REAP entry retains its target/draft repositories, revisions,
draft byte manifest, identity digest, memory floor and request limits. Its
existing test requiring QMV installation on every target layer is now enforced
at startup. The 1017 entry explicitly admits mixed target projection precision;
only compatible 2-bit down projections receive that optimization.

Both lanes remain greedy, text-only, without tools/MCP, with at most 8,192
input tokens and 4,096 output tokens. The default output limit is 4,096.

## Memory and read-only artifact inspection

The 1017 physical-memory floor is **224 GiB**, enforced by alias metadata and
programmatic server startup. This retains the conservative capacity reasoning
of the released lane: resident weights alone do not account for draft packing,
attention caches, prefill temporaries, allocator overhead and OS memory.

Header-only inspection of an early T256 qualification build found (the
published build differs; its pins and sizes live in `artifacts.py` and
`rapid_mlx/model_sizes.json`):

- Target tensors excluding SSD Engram tables: **191,363,969,248 bytes**.
- Draft tensors: **8,386,383,636 bytes** (the file is 8,386,765,492 bytes).
- Combined resident tensor budget: **199,750,352,884 bytes**, about **186.03 GiB**.
  At the floor this leaves about 38 GiB for the remaining allocations; a 256 GiB
  machine has about 70 GiB beyond these tensors. This is a capacity guard,
  not a measured peak-memory guarantee.
- Offloaded Engram tensors: **122,883,656,000 bytes**, affine 4-bit, group 32.
  The metadata-only `supports_engram_ssd_offload` probe returns true.
- Forty target layers; the supplied build retains 384 experts per layer.
  The explicit module map contains 102 expert projections at 2 bits, 16 at
  3 bits, and two at 4 bits. Existing strict native loading already supports
  the per-layer counts and projection-specific map.
- `dspark/` contains `config.json`, `model.safetensors.index.json`, and
  `model-mtp.safetensors`. Its 3,580 tensor headers use BF16/F32/U32, with
  three stages, 128 routed experts and top-k 3. Draft experts are 4-bit;
  shared experts and attention projections are 8-bit, group 64.

The owned draft loader accepts the target root or its explicit `dspark/`
directory, retaining config/index/tensor-contract checks. Hub shard symlinks
inside `dspark/` may resolve only into the same repository's blob directory.
No checkpoint Python code is imported.

## Prompt framing finding

**The supplied `tokenizer_config.json` has no `chat_template` key.** There is
also no separate template asset in that build. Therefore exact template equivalence
cannot be established from this build. The existing hand-built renderer is
preserved; there is no evidence supporting a per-artifact change.

The tokenizer's framing tokens agree with that renderer: BOS 0, EOS 1,
`<｜User｜>` 128803, `<｜Assistant｜>` 128804, `<think>` 128821 and `</think>`
128822. Non-thinking generation ends its prompt with
`<｜Assistant｜></think>`. Thinking generation ends at `<｜Assistant｜>`.
The upload owner should provide the authoritative template if exact
multi-turn/template equivalence is required before public qualification.

## Serve a local, unpinned build for qualification

```sh
RAPID_MLX_V41_MODEL_DIR=$BUILD_DIR \
  python -m rapid_mlx.cli serve deepseek-v41-flash-1017 \
  --host 127.0.0.1 --port 8000 --served-model-name dsv41-local --max-tokens 4096
```

`RAPID_MLX_V41_MODEL_DIR` follows the established model-directory override
pattern. It is explicit, accepted only with the 1017 registry entry, checks
required target/draft metadata and files, and emits an
`UNPINNED LOCAL QUALIFICATION` warning. It bypasses downloads and revision/file
hash pinning, with runtime revisions reported as `local-unpinned`. It preserves
memory checks and every request limit. Missing/incomplete paths fail without
falling back to Hub. The Engram token map must already exist so this read-only
build does not need a generated cache beside its tokenizer.

The served endpoint is `http://127.0.0.1:8000/v1`, model `dsv41-local`, suitable
for `scripts/eval_deepseek_v41_tasks.py` smoke/speed/bench after startup.
Community Benchmark publication and `pull` use pinned artifacts and do not
accept this qualification override.

## Serve the published alias

The published repository is `rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX`.
`FLASH_1017` in `artifacts.py` pins one immutable 40-character commit for both
target and draft, with complete `ArtifactFile(name, size, sha256)` manifests for
the target assets and every indexed shard, and for the co-located `dspark/`
draft. Remote files are size/hash checked; there is no second draft repository.
Updating the published build means regenerating those pins and the
`model_sizes.json` entry from the new commit.

```sh
env -u RAPID_MLX_V41_MODEL_DIR \
  python -m rapid_mlx.cli serve deepseek-v41-flash-1017 \
  --host 127.0.0.1 --port 8000 --max-tokens 4096
```
