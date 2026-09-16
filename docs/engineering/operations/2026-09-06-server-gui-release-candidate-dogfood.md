# Server and GUI release-candidate dogfood — 2026-09-06

## Verdict

The release-mode Desktop build and a source-tree server both completed their
real-weight acceptance walks on an Apple M3 Ultra. Dogfood found one P1 release
blocker: a single unresponsive external-volume symlink in the Hugging Face
cache could hang `rapid-mlx models --cached --json` indefinitely. That command
is also the Desktop model inventory probe, so the same condition could leave
the GUI picker waiting forever.

The candidate fix bounds all relocated cache probes under one two-second
deadline, preserves exact sizing for responsive relocated entries, and omits
only entries whose backing volume does not respond. With three unresponsive
entries present, the source command went from **more than two minutes without
returning** to **3.05 seconds**, and the packaged sidecar completed in **4.17
seconds** on both the original and latest upstream bases. Server and
isolated-GUI dogfood passed after the fix.

This is a commit-bound local qualification receipt, not authorization to
publish a release. The initial qualification base was
`caafd08df9e8d7242b081a3fc9fb7c9f6807eca0`. The branch was then rebased onto
`cc7d849821643706da15a9cdc52545ea28a05ace` and the complete release app plus
the newly merged Community Benchmark surface were requalified at
`f09a761a26de553f0e87f66c30bd733759f6f4b2`. A final release audit rebased onto
`2eac979c66fedee5cd9767b416b4fabdc9cf8d00`; its exact candidate
`444eb395400b57516765f81536125d02387eb9c4` passed a fresh release build,
packaged inventory, Community Benchmark identity run, and the focused tests
for the newly merged benchmark and Desktop visual-recovery changes. PR merge
gates remain mandatory before Atlas uses the fix for release integration.

## Candidate and host

- Rapid-MLX: 0.13.4, latest tested head
  `444eb395400b57516765f81536125d02387eb9c4`
- Latest tested upstream base:
  `2eac979c66fedee5cd9767b416b4fabdc9cf8d00`
- Branch: `vector/release-dogfood-20260906`
- Host: Apple M3 Ultra, 256 GiB unified memory, macOS 26.5.2
- Source runtime: Python 3.11.16, MLX 0.32.2
- Desktop sidecar: Python 3.12, MLX 0.32.2, mlx-lm 0.31.3,
  mlx-vlm 0.6.17, mlx-audio 0.4.3, mflux 0.19.0
- Release-mode sidecar: 482 MiB raw, 160 MiB compressed, 173 Mach-O files
- Exact-head sidecar tarball SHA-256:
  `bd4cfe48f4a74a71a4d9565353578b66f549acf14439282b67f02706b0bfbcaf`
- Desktop signing: deep ad-hoc signing for local dogfood only

The production app at `/Applications/Rapid-MLX Desktop.app` remained running
and untouched. Initial GUI work used bundle ID
`com.rapidmlx.rapid.dogfood-8dbdee13` and port 59381. Exact-head Community
Benchmark qualification used `com.rapidmlx.rapid.dogfood-2ec525f1`, another
throwaway HOME, and port 59382.

## P1 reproduced and fixed

The default HF cache contained three model-entry symlinks into an external
ExFAT volume that was mounted but pathologically slow and 99.8% full:

- `mlx-community/Qwen3.5-9B-4bit`
- `mlx-community/Qwen3.5-35B-A3B-4bit`
- `mlx-community/Qwen3.6-35B-A3B-4bit`

Before the fix, `uv run rapid-mlx models --cached --json` remained blocked for
more than two minutes in `_dir_size_bytes -> os.scandir -> __opendir2`. A
sampled installed-sidecar process showed the same stack after more than 42
minutes. The CLI had no filesystem deadline, and Desktop's `ModelCatalog`
waited on that child without its own timeout.

The fix keeps local cache directories on the existing exact synchronous path.
It detects only relocated root symlinks, scans them concurrently in daemon
workers, and applies one shared two-second deadline to the whole relocated set.
Responsive links retain their true size and modification time. Timed-out links
are excluded from the inventory pass so later runnability checks cannot block
on them again; stderr names every skipped repo.

Measured with the real unhealthy cache still mounted:

| Probe | Before | Fixed source | Fixed packaged sidecar |
| --- | ---: | ---: | ---: |
| `models --cached --json` | >120 s, manually interrupted | 3.05 s | 4.17 s |
| Exit status | none | 0 | 0 |
| Healthy rows returned | none | 43 | 43 |
| Unresponsive rows skipped | none | 3 | 3 |

No cache was deleted, migrated, or unmounted. The external-volume capacity
problem remains a host-operations issue; the product fix prevents it from
freezing unrelated healthy inventory.

## Source server dogfood

`qwen3.5-4b-4bit` loaded from the local cache on loopback port 8000. Warmup
completed, health reported `ready=true`, and the following real inference
paths passed:

- `/health` and `/v1/models`
- OpenAI Chat, non-streaming: exact `RAPID_RELEASE_OK`
- OpenAI Chat SSE: exact `STREAM_OK` followed by `[DONE]`
- Responses API, non-streaming: completed output item with exact
  `RESPONSES_OK`
- Responses API SSE: ordered events through `response.completed`, exact
  `RESPONSE_STREAM_OK`
- Anthropic Messages: exact `ANTHROPIC_OK`
- Named no-argument function call: `finish_reason=tool_calls`, function
  `ping`, arguments `{}`
- Eight simultaneous Chat requests: all eight returned their distinct exact
  `CONCURRENT_<n>_OK` values
- Client disconnect after 200 ms: scheduler aborted the orphaned request;
  immediate health stayed ready and the next request returned exact
  `AFTER_CANCEL_OK`
- Graceful Ctrl-C: prefix cache saved and the engine, scheduler, and server
  stopped without a remaining listener

One deliberately strict forced-tool request requiring a `city` object made the
4B model emit malformed scalar/XML arguments. The server returned HTTP 400
under the existing fail-closed schema contract. A named no-argument tool call
passed immediately afterward. This is a small-model output-quality limitation,
not justification to weaken validation before release.

## Desktop GUI dogfood

The fixed source was rebuilt into the full release-mode sidecar before GUI
testing. The isolated app completed these user-visible paths:

- Fresh-persona onboarding appeared with stable AX selectors and correctly
  reported Apple M3 Ultra / 256 GB hardware.
- Skipping onboarding entered the main chat surface without mutating the
  production bundle's preferences or Application Support.
- Model inventory completed despite the three unhealthy external symlinks.
  Model Management displayed 184 available models and a completed storage
  summary of 30 models on disk; controls and rows exposed stable AX identifiers.
- `qwen3.5-4b-4bit` started from the GUI. The owned packaged sidecar listened
  only on isolated port 59381 and reported `ready=true`.
- A message entered and sent through the GUI returned exact `GUI_RELEASE_OK`.
  Captured stats were 2.17 s time-to-first-token and 2.23 s total for three
  completion tokens on the first GUI request.
- After app termination and relaunch with the same isolated persona,
  onboarding did not repeat. The conversation and both persisted messages
  reappeared after selecting its sidebar row.
- Both shutdowns reaped the owned sidecar and released port 59381. The
  throwaway run was marked finished for host hygiene.

### Exact-head Community Benchmark dogfood

After rebasing onto upstream `cc7d84982` (Desktop Community Benchmark result,
model-grouping, and run-progress changes), the full release app was rebuilt and
launched as a second isolated persona. The model menu rendered the three new
sections in order: Recommended for this Mac, Downloaded, and All models. The
recommended group included the fitting Flux 2 Klein, Gemma 4, and Qwen focus
models; the cached Qwen 3.5 4B model appeared under Downloaded.

A first real run was stopped after the UI showed the model, workload scope,
expected duration, and advancing elapsed timer. The benchmark process group
was gone within the five-second observation window, the control returned to
Run locally, and the UI reported that no incomplete result was shared.

A second real `qwen3.5-4b-4bit` run completed all ten measured rounds and
flowed from the packaged CLI's JSON record into the GUI:

| Case | Measured rounds | GUI / independently recomputed result |
| --- | ---: | ---: |
| `pp512-tg128` | 5 | 171.0 tok/s, TTFT 266 ms |
| `pp2048-tg512` | 5 | 169.6 tok/s, TTFT 947 ms |

The independent calculation used the leaderboard formula
`(output_tokens - 1) / decode_duration`. It matched both GUI rows exactly.
The completion stamp `2026-09-07T05:29:39.745233Z` rendered in the local time
zone as `Today 10:29 PM`. The exact-head app then terminated cleanly, released
its host lock, and was marked finished. No P0 or P1 issue was found in the new
Community Benchmark surface.

After the subsequent upstream benchmark-identity hardening, the newly rebuilt
packaged CLI ran the same registered protocol against the real cached
`qwen3.5-4b-4bit` snapshot. All ten measured rounds completed. The stored model
record now reflects the artifact actually loaded: affine weights, four bits
(`weight_bits_x2=8`), group size 64, and resolved Hugging Face revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`. Its measured medians were 170.5
tok/s with 264 ms TTFT for `pp512-tg128`, and 168.6 tok/s with 944 ms TTFT for
`pp2048-tg512`. This validates both the packaged inference path and the new
quantization/revision provenance rather than trusting catalog metadata.

## Automated verification

- `uv run pytest -q tests/test_cli_models.py`: 86 passed
- `uv run pytest -q tests/test_cli_models.py tests/test_cli_models_json.py tests/test_first_run_guide.py`:
  124 passed
- Expanded post-fix selection including cache inventory, first-run, the
  Transformers 5.15 offline-wrapper regression, and packaged BF16 image
  construction: 125 passed, 1 sanctioned offline skip
- `uv run ruff check vllm_mlx/cli.py tests/test_cli_models.py tests/integrations/test_default_on_deep.py`:
  passed
- `git diff --check`: passed
- `apps/rapid-mac/scripts/desktop-test-timeout.sh`: 3,542 tests in 306
  suites passed in 90.61 seconds
- Exact-head Community Benchmark Swift selection: 29 passed
- Latest-base Python cache and Community Benchmark selection: 321 passed
- Latest-base Swift Community Benchmark, draft-post, and visual-grounder
  selection: 72 tests in 3 suites passed; live-only fixtures skipped
- Exact-head cache/first-run/offline Python selection: 27 passed
- Changed production lines under targeted coverage: 100% (40/40)
- Initial release build (`candidate-caafd08e`): passed
- Exact-head release build (`candidate-f09a761a`): passed
- Latest-base release build (`candidate-444eb395`): passed
- Packaged `rapid-mlx models --cached --json` under a ten-second outer guard:
  passed in 4.17 seconds initially, 4.19 seconds at the first exact head, and
  4.17 seconds at the latest base; every run returned 43 healthy rows and
  skipped the same three unresponsive entries

An initial direct `swift test` invocation produced timing failures because it
allowed Swift Testing to parallelize suites. The repository and hosted CI
explicitly require `desktop-test-timeout.sh` / `--no-parallel`: synchronous
child-process waits can starve the cooperative pool otherwise. The exact same
built test bundle passed completely through the supported serial gate above.
The first complete Python collection selected 23,225 tests and finished with
22,963 passed, 256 skipped, 6 expected failures, 1 expected pass, and two
environment-path failures. The packaged BF16 construction test required the
optional `mflux` dependency and passed once the release-pinned mflux 0.19.0 was
installed. The constrained-tool negative control exposed a real test-harness
compatibility gap: Transformers 5.15 wraps a typed Hub offline cache miss in
`OSError`, so the intended sanctioned skip did not fire. The test now walks
the exception chain for the typed cause while continuing to fail on unrelated
disk `OSError`s; its positive and negative guards pass. Hosted exact-head
checks remain mandatory.

## Release boundary and remaining risk

Atlas should merge the P1 fix before cutting the candidate, then rebuild and
run artifact-bound smoke on the final signed/notarized output. This run did not
create a tag, GitHub Release, notarized artifact, deployment, or updater
mutation. The nearly-full external ExFAT volume is still unsuitable for the
large-model fleet gate, so a healthy cache volume remains necessary for a
complete multi-family `release-check-m3` receipt.
