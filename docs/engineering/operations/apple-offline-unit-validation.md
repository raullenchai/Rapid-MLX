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

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ \
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

## Residual numerical investigation

The original M5 Max seed-0 Glimmer full-prefill/cache discrepancy was
`0.003024384379386902` against the original `0.002` tolerance. M3 Ultra controls
at the issue's current investigation base passed seeds 0–2. This is evidence
of an environment-dependent result, not a correctness certification or a
reason to relax the tolerance. Vector retains numerical investigation
ownership; Harbor assists with matching dependency and hardware environments.
Preserve both the reported failing seed and the original assertions when
investigating that boundary. Cached real-tokenizer/real-weight qualifications
elsewhere in the suite retain their own prerequisite contracts.
