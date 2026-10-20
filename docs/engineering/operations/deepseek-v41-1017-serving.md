# DeepSeek V4.1 Flash 1017 serving qualification

Paths in command examples use owner-supplied environment variables: export
`DSV41_SCRATCH_ROOT` for task scratch and `RD_DATA` for approved dataset exports.
Where used, set `SOURCE` to the immutable model snapshot, `GPTQ_OUTPUT` and
`RD_OUTPUT` to build destinations, `GPTQ_HOST`, `REMOTE_WORK`, `REMOTE_PYTHON`
for the remote worker, and `MLX_CPU_LIBRARY_DIR` to a matching CPU MLX library.
These paths must exist on the host running the corresponding command.

Owner: Vector. Branch/worktree: `vector/dsv41-reap`, `dsv41-reap`.
Phase 7 changes are intentionally uncommitted and unpushed. No model server or
GPU job was started, and `.venv-metal` was not inspected or modified.

The existing serial V4.1 DSpark K4 product lane now selects its artifacts from
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

Header-only inspection of the supplied T256 build found:

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
also no separate template asset in T256. Therefore exact template equivalence
cannot be established from this build. The existing hand-built renderer is
preserved; there is no evidence supporting a per-artifact change.

The tokenizer's framing tokens agree with that renderer: BOS 0, EOS 1,
`<｜User｜>` 128803, `<｜Assistant｜>` 128804, `<think>` 128821 and `</think>`
128822. Non-thinking generation ends its prompt with
`<｜Assistant｜></think>`. Thinking generation ends at `<｜Assistant｜>`.
The upload owner should provide the authoritative template if exact
multi-turn/template equivalence is required before public qualification.

## Serve the local T256 build tonight

Run from this worktree **after the GPU sweep finishes**, using the existing
Metal interpreter. These are owner commands; they were not executed in phase 7.
Tests used the separate CPU-only `.venv/bin/python` interpreter.

```sh
RAPID_MLX_V41_MODEL_DIR=$RD_BUILDS/T256 \
  .venv-metal/bin/python -m rapid_mlx.cli serve deepseek-v41-flash-1017 \
  --host 127.0.0.1 --port 8000 --served-model-name dsv41-t256 --max-tokens 4096
```

`RAPID_MLX_V41_MODEL_DIR` follows the established model-directory override
pattern. It is explicit, accepted only with the 1017 registry entry, checks
required target/draft metadata and files, and emits an
`UNPINNED LOCAL QUALIFICATION` warning. It bypasses downloads and revision/file
hash pinning, with runtime revisions reported as `local-unpinned`. It preserves
memory checks and every request limit. Missing/incomplete paths fail without
falling back to Hub. The Engram token map must already exist so this read-only
build does not need a generated cache beside its tokenizer.

The served endpoint is `http://127.0.0.1:8000/v1`, model `dsv41-t256`, suitable
for `scripts/eval_deepseek_v41_tasks.py` smoke/speed/bench after startup.
Community Benchmark publication and `pull` use pinned artifacts and do not
accept this qualification override.

## Activate the published alias after upload

The new repository is `rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX`. Remote serving,
pull and benchmark execution deliberately fail closed until the upload owner
fills `FLASH_1017.target_revision`, `mtp_revision`, `target_files`, and
`mtp_files`. Both revisions must be the same immutable 40-character commit.
Populate complete `ArtifactFile(name, size, sha256)` manifests: target config,
index, tokenizer assets, Engram token map and every indexed target shard;
draft config, index and `model-mtp.safetensors` with names relative to
`dspark/`. The manifests must cover all shards referenced by their indexes.
The target revision includes the co-located draft; no second repository is
resolved. Remote files are size/hash checked, with the existing same-repo Hub
CAS shortcut retained.

After filling those pins and making the checkpoint available:

```sh
env -u RAPID_MLX_V41_MODEL_DIR \
  .venv-metal/bin/python -m rapid_mlx.cli serve deepseek-v41-flash-1017 \
  --host 127.0.0.1 --port 8000 --served-model-name dsv41-t256 --max-tokens 4096
```

## Verification and handoff

CPU tests cover registry selection, unchanged REAP pins/identity, unset and
partial pins, manifest/index coverage, local gating and warnings, co-located
Hub shard confinement, a tiny complete synthetic indexed draft with actual
4-bit affine projection execution, mixed target precision, and mocked CLI and
serial server dispatch. Existing lane, loader, server callbacks, CLI, alias,
Community Benchmark and provenance contracts are included. The CLI parser
snapshot passes unchanged; no new CLI option or generated snapshot is needed.

The unrestricted broader run produced **4,539 passed, two failed and one
Metal test deselected**. Both failures are existing process-group cleanup
tests in `test_community_benchmark_workspace.py`: they discover detached
children using `pgrep`, which cannot enumerate processes in this sandbox
(`sysmond service not found; Cannot get process list`). They remain a Harbor
follow-up to run on an unrestricted host. The Metal QMV execution test remains
for GPU qualification after the sweep.

Ruff lint/format and `git diff --check` pass. The full mypy budget gate was run
with `.venv/bin/python`; it fails in this Python 3.13 environment. A complete
comparison using `--shadow-file` for all changed Python files against `HEAD`
found **1,418 errors on each side and identical counts in every file**. There
is no new mypy error growth; do not update the checked-in Python 3.11 budget
from this environment. Small missing test dependencies were installed only in
`.venv`; `.venv-metal` was untouched. Logs and private implementation notes are
in `${DSV41_SCRATCH_ROOT}/phase7/`.

Atlas/Pixel/Harbor/Echo awareness handoff is recorded in private scratch notes;
Orca agent messaging was unavailable in this session. No PR was created, and no
commit or push was performed, as requested.

Next owner actions: upload owner fills and verifies remote pins; Vector runs
local smoke/speed/bench and measures peak memory after the sweep; Harbor runs
the two process-group tests without sandbox process restrictions. **Tools,
inputs beyond 8k, and sampling remain separate follow-ups and are not enabled
by phase 7.** Full-model correctness, speed and peak memory are unmeasured here.

Final verification after the last changes: **4,539 passed, three deselected**
(the two sandbox-limited process tests and the Metal execution test). The
**25 phase-7 tests** also passed separately. The released draft identity was
compared directly with `HEAD` and remains
`sha256:eb94ba9766612111d89b89676a7485d612ad534b2d8b0d37b78c137ff8994d75`.

Changed files:

- `rapid_mlx/models/deepseek_v41_native/artifacts.py`: registry, pin gate,
  explicit local artifact resolution, per-artifact identity and memory check.
- `rapid_mlx/models/deepseek_v41_native/dspark.py`: co-located discovery and
  nested Hub shard confinement.
- `rapid_mlx/models/deepseek_v41_native/serving.py`: explicit mixed target
  precision support with the original REAP QMV invariant enforced.
- `rapid_mlx/models/deepseek_v41_native/server.py`: registry-selected artifacts
  and truthful local runtime identity.
- `rapid_mlx/cli.py`: registered lane dispatch and pinned/local download policy.
- `rapid_mlx/aliases.json`: 1017 alias using the existing closed-key schema.
- `rapid_mlx/community_bench/local_runner.py`: registry-selected loading and
  co-located draft provenance.
- `rapid_mlx/community_bench/workspace.py`: registered lane readiness/pin gates.
- `tests/test_deepseek_v41_phase7.py`: 25 CPU qualification contract tests.
- `tests/test_no_out_of_band_routing.py`: allow the explicit source-directory
  override after registry routing.
- `docs/engineering/operations/deepseek-v41-1017-serving.md`: this handoff.
