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
seconds**. Server and isolated-GUI dogfood passed after the fix.

This is a commit-bound local qualification receipt, not authorization to
publish a release. The source base was `caafd08df9e8d7242b081a3fc9fb7c9f6807eca0`;
the P1 fix was tested as an uncommitted candidate on top and must pass the PR
merge gates before Atlas uses it for release integration.

## Candidate and host

- Rapid-MLX: 0.13.4, base `caafd08df9e8d7242b081a3fc9fb7c9f6807eca0`
- Branch: `vector/release-dogfood-20260906`
- Host: Apple M3 Ultra, 256 GiB unified memory, macOS 26.5.2
- Source runtime: Python 3.11.16, MLX 0.32.2
- Desktop sidecar: Python 3.12, MLX 0.32.2, mlx-lm 0.31.3,
  mlx-vlm 0.6.17, mlx-audio 0.4.3, mflux 0.19.0
- Release-mode sidecar: 482 MiB raw, 160 MiB compressed, 173 Mach-O files
- Sidecar tarball SHA-256:
  `22dd46425296d6d3644c5b4eab2fd11220e4301a2afb0ef45b451b24b33fb44d`
- Desktop signing: deep ad-hoc signing for local dogfood only

The production app at `/Applications/Rapid-MLX Desktop.app` remained running
and untouched. GUI work used a copied app with bundle ID
`com.rapidmlx.rapid.dogfood-8dbdee13`, a throwaway HOME, and port 59381.

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

## Automated verification

- `uv run pytest -q tests/test_cli_models.py`: 85 passed
- `uv run pytest -q tests/test_cli_models.py tests/test_cli_models_json.py tests/test_first_run_guide.py`:
  123 passed
- Expanded post-fix selection including cache inventory, first-run, the
  Transformers 5.15 offline-wrapper regression, and packaged BF16 image
  construction: 125 passed, 1 sanctioned offline skip
- `uv run ruff check vllm_mlx/cli.py tests/test_cli_models.py tests/integrations/test_default_on_deep.py`:
  passed
- `git diff --check`: passed
- `apps/rapid-mac/scripts/desktop-test-timeout.sh`: 3,542 tests in 306
  suites passed in 90.61 seconds
- `RAPID_CANDIDATE_IDENTITY=candidate-caafd08e bash scripts/build.sh`:
  passed
- Packaged `rapid-mlx models --cached --json` under a ten-second outer guard:
  passed in 4.17 seconds

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
