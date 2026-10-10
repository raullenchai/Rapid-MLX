# Apple offline unit validation

Issue #4438 tracks failures that were already present in Apple full-unit runs.
The full suite stays fail closed: a focused pass or green CI does not turn a
failed full-unit result into a pass.

## Run against an isolated installation

Use Python 3.12 on Apple Silicon and install the declared test dependencies in
an isolated environment. Activate that environment before running the suite so
subprocess CLI checks find its installed entry points instead of another
checkout's executables. Use the existing default Hugging Face cache; do not
create another model cache or fetch weights to make unit tests pass.

The canonical suite also executes JavaScript workflow harnesses and validates
MCP command discovery. Put a real Node.js installation (including `npx`) on
the noninteractive runner's `PATH`; a Python environment alone is insufficient.
Check `node --version` and `npx --version` in the same shell that will launch
pytest. Missing executables are validation prerequisites, not reasons to skip
the workflow or security tests.

For the complete numerical qualification, launch pytest with
`MLX_ENABLE_TF32=0` as well as the offline flags. Several models compare
matrix-based prefill with matrix-vector decode at float32 tolerances; on M5
the default arithmetic policy gives those paths different precision. Set the
flag at process launch, before MLX initializes its cached precision policy.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ \
  --ignore=tests/integrations --ignore=tests/test_event_loop.py \
  -q --no-header --tb=line --junitxml=full-unit.xml
```

Record the exact commit, platform, Python and dependency versions, command,
exit code and complete JUnit result. A run stopped before completion is not a
full-suite result. To classify remaining failures, repeat every failed/error
node at an immutable base checkout in the same environment, preserving class
names, statuses and messages. Never collapse duplicate method names across
classes or treat a timeout as a baseline match.

## Self-contained unit fixtures

Streaming detokenizer tests construct a real byte-complete BPE tokenizer with
merged word and space-prefix tokens. They serialize it and use mlx-lm's actual
tokenizer loader to verify optimized decoder selection. ASCII and Unicode
round trips check decoded content as well as streaming/batch parity. They need
neither network access nor trained model weights.

Model registry tests serialize a seeded two-layer Qwen3 model into pytest's
temporary directory, then load it through the real model loader. Ownership,
transfer, garbage collection, sequential generation, batching and cache
recovery assertions remain active. The temporary random model tests those
mechanisms; it makes no claim about pretrained model quality or large-model
performance. Model initialization restores the caller's MLX random state.

Agent config-isolation tests stub both version-discovery import locations so
host OpenCode installation and version cannot affect the config scenario or
create operator state. Adapter failure classification likewise uses the
supplied profile instead of an installed-version override. The interactive
preflight golden-output test uses the existing uncached fixture to clear both
offline environment flags while its Hub calls remain mocked and the shared
network guard remains active.

## Glimmer cache-parity precision

The M5 Max seed-0 Glimmer full-prefill/cache discrepancy reproduces at main
`9bb5110d6`: `0.003024384379386902` against the original `0.002` tolerance.
With Python 3.12.13, MLX 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2,
transformers 5.15.1 and NumPy 2.5.3, all seeds 0–31 fail under the default
precision policy (worst difference `0.006861530244350433`).

The cause is unequal arithmetic precision: M5 float32 matrix multiplication
uses MLX's default TF32 path, while single-token matrix-vector multiplication
uses full float32. Even the first query projection, before attention or cache
updates, differs by `0.0014522075653076172` for seed 0. Launching the same
control with `MLX_ENABLE_TF32=0` reduces that projection difference to
`5.960464477539062e-7` and the logit difference to `1.3634562492370605e-6`.
All 32 seeds pass, with worst logit difference `4.023313522338867e-6`.
See [MLX's numerical precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

The strict cache-parity tests therefore launch a bounded child process with
`MLX_ENABLE_TF32=0` set before importing MLX. They keep the original tolerance,
model shapes, sequence length and assertions, and retain seeds 0–31 as named
regressions. A forward smoke test also runs under the caller's precision policy.
The parent test process and serving runtime keep their existing policy; this
does not establish a `0.002` parity guarantee for default TF32 inference.

This precision prerequisite also explains the retained numerical failures in
Bailing, DeepSeek V4, G9V3, K2 Horizon, MLA absorbed verification, MLLM batch
generation and Qwen4 tests. On M5, the unchanged seven modules pass all
373 tests with full float32. Use the process-launch flag for complete-suite
qualification; the isolated Glimmer tests also work when invoked from an
already initialized default-precision parent.

To repeat the strict seeded test without changing the suite's precision:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
  tests/test_muse_glimmer_model.py -q
```

Harbor owns reconciliation of other environments' baseline failure lists with
the landed fixture repairs and isolated prerequisites. Cached real-tokenizer
and real-weight qualifications retain their own prerequisite contracts.
