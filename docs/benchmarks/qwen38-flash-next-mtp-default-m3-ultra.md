# Flash-Next native MTP default qualification, 2026-11-10

Enable embedded native MTP at **K=1** for `qwen3.8-flash-next-4bit`.
K=1 improves median registered decode throughput by **32.79%** at
pp512-tg128 and **43.20%** at pp2048-tg512. TTFT changes by -3.30% and
-0.45%, respectively. All 24 functional quality outcomes match ordinary
decode. K=2 remains explicit-only: it is slower than K=1 at every measured
length and regresses against ordinary decode at 32K.

## Environment

| Component | Value |
| --- | --- |
| Host | Mac Studio Mac15,14; M3 Ultra; 28 CPU / 60 GPU cores |
| Unified memory | 256 GiB; zero swap usage during qualification |
| macOS | 26.5.2 (25F84), arm64 |
| Python | 3.12.14 |
| MLX / MLX-LM / MLX-VLM | 0.32.4 / 0.31.3 / 0.7.2 |
| Transformers / NumPy | 5.15.1 / 2.5.4 |
| Measurement source | `280fdb04793225c9651902fa06c7269733a376af` |
| Checkpoint | `rapid-mlx/Qwen3.8-Flash-Next-4bit` |
| Immutable revision | `afeb96d530bf07aee45b2f6f428e18ba464eb88a` |
| Architecture / cache | `qwen4_exp`; BF16, full-precision KV; prefix cache disabled |
| Quantization | Checkpoint-defined mixed affine 4-bit; default group size 64 |

Only the already-cached checkpoint was used. Hugging Face and Transformers
were offline. Every weight load, inference run, and final test run used
`lockf /var/tmp/rapid-mlx-large-model.lock`. No listening server or port 8451
was used. No other process was terminated.

## Measurements

Registered `rapid-community-speed` v2 uses exact synthetic token IDs,
concurrency one, greedy sampling, ignore-EOS, one discarded warmup, and five
measured rounds for each short case. The extension uses the same runner and
token corpus at 8,192/32,768 prompt tokens and 256 output tokens, also with
one warmup and five measured rounds. Decode is `(output_tokens - 1)` divided
by the interval after the first emitted token. TTFT includes prefill.

Each arm used a fresh process. The comparison order was K=1, K=2, ordinary
decode. An initial ordinary-decode pass overlapped briefly with synthetic
tests and was discarded; all baseline rows below come from the fresh pass.
The speculative configurations used the normal serving policy; greedy decode
uses reproducible depth, rather than adapting depth from round timings.
The public K=2 option also selects its existing indexed-QSA verification
setting (`RAPID_MLX_QSA_INDEXED_SPLITK=1`); it was not a private kernel override.

Values are median **[minimum, maximum]** over five measured rounds. Memory
is MLX peak active allocation, in GiB, rather than RSS. The short-case peak
resets after warmup; subsequent peaks are cumulative through the indicated
bucket and include long-context warmup. Model-loading peaks are excluded.

| Prompt / output | MTP | Decode tok/s [min, max] | TTFT seconds [min, max] | Peak GiB |
| --- | --- | ---: | ---: | ---: |
| 512 / 128 | off | 32.875 [32.811, 33.099] | 0.733 [0.711, 0.743] | 98.493 |
| 512 / 128 | K=1 | 43.654 [43.603, 43.875] | 0.709 [0.691, 0.712] | 99.860 |
| 512 / 128 | K=2 | 41.798 [41.630, 41.984] | 0.717 [0.712, 0.723] | 99.860 |
| 2048 / 512 | off | 30.429 [30.410, 30.491] | 2.250 [2.249, 2.286] | 98.493 |
| 2048 / 512 | K=1 | 43.575 [43.334, 43.749] | 2.240 [2.230, 2.250] | 99.860 |
| 2048 / 512 | K=2 | 41.708 [41.182, 42.157] | 2.263 [2.257, 2.267] | 99.860 |
| 8192 / 256 | off | 29.700 [29.567, 29.752] | 9.457 [9.428, 9.479] | 99.073 |
| 8192 / 256 | K=1 | 41.434 [40.868, 41.755] | 9.345 [9.319, 9.449] | 100.440 |
| 8192 / 256 | K=2 | 39.928 [39.646, 40.164] | 9.403 [9.343, 9.460] | 100.440 |
| 32768 / 256 | off | 26.862 [26.586, 27.079] | 45.757 [45.621, 45.822] | 102.016 |
| 32768 / 256 | K=1 | 35.314 [35.284, 35.590] | 45.832 [45.683, 45.983] | 103.383 |
| 32768 / 256 | K=2 | 22.734 [22.524, 23.323] | 45.858 [45.718, 45.965] | 103.383 |

K=1 adds 1.367 GiB to peak allocation at every measured length. Its 8K/32K
decode gains are 39.51%/31.47%; TTFT changes are -1.18%/+0.17%. There is no
observed memory blowup or TTFT regression exceeding 10%. This does not qualify
128 GiB hardware or other checkpoints, sampling policies, or concurrency.

## Quality

The 24 cases are selected from `.orca/flash-next-eval/prompts.jsonl`:
four English/multi-turn, four Chinese, four math/logic, five JSON, four tool
calls, and three code cases. Requests use temperature zero and thinking off.
The direct-engine JSON prompts explicitly include their response schemas;
they do not rely on an HTTP grammar constraint. Output token IDs and the
target's top-two log-probability margins are recorded for every emitted token.
The difference between two log probabilities is the corresponding logit margin.

Both K=1 and K=2 match **23/24 token sequences exactly**. In the remaining
English water-freezing explanation, the first difference is at generated
token index 16 (zero-based): baseline token 506 versus speculative token 1332.
The baseline top-two margin is **0.0**, and both speculative arms' margin is
**0.125**, below the 0.25 limit. The wording changes from “temperature at
which” to “temperature where”; the explanation has the same meaning.

All **24/24 functional outcomes match and pass** for each arm. JSON values
match after parsing. XML tool names and arguments match the expected cities
and calculator operations. Generated Python palindrome and JavaScript sum
functions pass concrete input/output checks; generated SQL returns the same
ordered rows from an in-memory fixture. The old Python regex scorer rejects
the correct lowercase-before-return implementation, so execution adjudicates
that case. The old tool scorer recognizes JSON calls, while this checkpoint
emits XML; XML arguments are parsed and checked explicitly.

Two initial extended jobs completed all timing rounds but failed afterward
because the quality wrapper passed a tokenizer `BatchEncoding` mapping instead
of its `input_ids`. Timing data was saved before quality began. Quality was
rerun separately after correcting that wrapper. The five JSON cases were
rerun on every arm with explicit schemas after discovering that their original
fixture stored schemas outside the messages. Only matching final prompts are
used in the quality comparison.

## Commands and evidence

Commands ran from the repository root. `PY` was
`/Users/raullenstudio/.rapid-mlx/bin/python`. Qualification scripts and raw
rounds are retained in `/private/tmp/flashmtp-qualification` and archived under
`/Volumes/RTL-2T/scratch-archive/flashmtp-qualification-2026-11-10`.
The wrapper imports the production community runner and serving loader;
`*-standard.json` saves the registered results before the extension runs.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD"
PY=/Users/raullenstudio/.rapid-mlx/bin/python
Q=/private/tmp/flashmtp-qualification

# Extended measurements; separate processes, same lock.
lockf /var/tmp/rapid-mlx-large-model.lock env QUAL_LABEL=k1 "$PY" \
  "$Q/extended.py" benchmark run qwen3.8-flash-next-4bit \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --json --progress
lockf /var/tmp/rapid-mlx-large-model.lock env QUAL_LABEL=k2 "$PY" \
  "$Q/extended.py" benchmark run qwen3.8-flash-next-4bit \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --json --progress
lockf /var/tmp/rapid-mlx-large-model.lock env QUAL_LABEL=off-extended "$PY" \
  "$Q/extended.py" benchmark run qwen3.8-flash-next-4bit \
  --no-spec-decode --json --progress

# Corrected quality-only retries: repeat with k2/2.
lockf /var/tmp/rapid-mlx-large-model.lock env QUAL_LABEL=k1 "$PY" \
  "$Q/quality_only.py" 1
# Schema retries: repeat with k1/1 and k2/2.
lockf /var/tmp/rapid-mlx-large-model.lock env QUAL_LABEL=off-extended \
  QUAL_SCHEMA_ONLY=1 "$PY" "$Q/quality_only.py" 0
python3.12 "$Q/adjudicate.py"
python3.12 "$Q/summarize.py"

# Final production default smoke: no speculative configuration.
lockf /var/tmp/rapid-mlx-large-model.lock "$PY" -m rapid_mlx.cli \
  benchmark run qwen3.8-flash-next-4bit --json --progress
lockf /var/tmp/rapid-mlx-large-model.lock "$PY" -m rapid_mlx.cli \
  benchmark run qwen3.8-flash-next-4bit --no-spec-decode --json --progress
```

The final tests use the serving environment's MLX/NumPy, with already-installed
pytest tooling appended by `$Q/runtime/sitecustomize.py` so subprocesses can
also import pytest. No dependency was downloaded or added to the project.

```bash
lockf /var/tmp/rapid-mlx-large-model.lock env \
  PYTHONPATH="$PWD:$Q/runtime" "$PY" -m pytest \
  tests/test_qwen4_exp_vendored.py tests/test_qwen4_exp_reference_parity.py \
  tests/test_flash_next_memory_gate.py tests/test_mtp_cli_wiring.py \
  tests/test_mtp_spec_decode.py tests/test_mtp_greedy_determinism.py \
  tests/test_native_mtp.py tests/test_mtp_lossless.py \
  tests/test_continuous_mtp_routing.py tests/test_mtp_default_off_3115.py -q

lockf /var/tmp/rapid-mlx-large-model.lock env \
  PYTHONPATH="$PWD:$Q/runtime" "$PY" -m pytest \
  tests/test_flash_next_mtp_default.py tests/test_aliases_contract.py \
  tests/test_model_aliases.py tests/test_model_auto_config.py \
  tests/test_decode_expectations.py tests/test_mtp_cli_wiring.py \
  tests/test_mtp_default_off_3115.py tests/test_community_bench_acceleration.py -q
```

The automatic-default smoke completed both cases with five measured rounds:
44.137/44.108 tok/s median decode, 0.686/2.241 seconds median TTFT, and
99.860 GiB peak allocation. Its execution metadata confirms `method: mtp`,
`assistant_source: embedded_head`, and `max_draft_tokens: 1` on the pinned
checkpoint.
The post-change explicit opt-out also completed both cases: metadata reports
`method: none`, decode medians are 33.498/30.510 tok/s, TTFT medians are
0.739/2.273 seconds, and peak allocation returns to 98.493 GiB.

The engine/MTP selection passed 640 tests with one skip across its main run and
four subprocess retries. The affected default/catalog selection passed 4,843
tests before the final snapshot-policy regression was added; that final test
passed in the six-test final regression run. The skipped module is
`test_qwen4_exp_reference_parity.py`: its optional external reference is not
installed. The initial six rollback-fixture failures under an older MLX
environment did not recur with serving MLX 0.32.4. Lint and format
checks cover every changed Python file.

## Policy and sibling alias

`experimental:true` remains. Commit `2063517f8` introduced it for the
experimental text-only launch; `docs/release-notes/v0.13.1.md` records the
limited release scope, and the original benchmark report explicitly leaves
128 GiB hardware untested. This single-host MTP experiment does not establish
broader hardware, vision, or concurrent-serving qualification.

The embedded head must use the ordinary engine injector. Automatic selection
previously mapped every native-capable alias to `backend:"native"`, whose
separate serial runtime does not qualify `qwen4_exp`. The default selector now
reserves that backend for aliases declaring an external native drafter. Tests
cover both alias and repository spellings, K=1, explicit K=2, and opt-out.
The offline decode snapshot records method `mtp` with no qualified default
configuration cohort; historical ordinary-decode cohorts remain unchanged.
Its two policy fields were generated from
`scripts.refresh_decode_expectations.build_snapshot([], existing_refreshed_at)`;
only this alias's `default_method` and `default_config_key` were copied into
the existing snapshot. Observation values and the export date were preserved.

`qwen3.8-flash-next-tensorfold` needs **no corresponding change**: its
`tensorfold_mtp:true` profile already selects MTP through a distinct checkpoint
and runtime. Its existing qualification is documented in
`docs/engineering/performance/2026-10-06-tensorfold-family-profiles-qualification.md`.
Its tool and opt-out boundaries differ; these native-head results cannot be
transferred to it.

AI assistance: Codex wrote the focused alias, selector, snapshot-policy, test,
and documentation changes and ran the measurements and checks reported here.

## Files changed

| File | Change |
| --- | --- |
| `rapid_mlx/aliases.json` | Enable this alias's K=1 MTP default; retain experimental status. |
| `rapid_mlx/cli.py` | Select the embedded-head engine path for the automatic default. |
| `rapid_mlx/decode_expectations.json` | Synchronize the default method/configuration policy. |
| `tests/test_flash_next_mtp_default.py` | Default, repository spelling, opt-out, explicit K=2, and snapshot regressions. |
| `tests/test_aliases_contract.py` | Assert the default and its capability label. |
| `tests/test_model_auto_config.py` | Expect the default-enabled capability label. |
| `docs/benchmarks/qwen38-flash-next-mtp-m3-ultra.md` | Link historical evidence to this decision. |
| This report | Record protocol, rounds, quality, decision, commands, and limitations. |
