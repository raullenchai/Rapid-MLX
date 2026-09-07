# Vector -> Atlas: server + GUI release dogfood P1

- Owner/host: Vector, Studio (Apple M3 Ultra, 256 GiB, macOS 26.5.2)
- Branch: `vector/release-dogfood-20260906`
- Latest tested upstream base: `2eac979c66fedee5cd9767b416b4fabdc9cf8d00`
- Exact tested branch head: `444eb395400b57516765f81536125d02387eb9c4`
- Durable report:
  `docs/engineering/operations/2026-09-06-server-gui-release-candidate-dogfood.md`

## Verified facts

- A real external-cache failure made `models --cached --json` hang for more
  than two minutes; an installed-sidecar instance had remained blocked for
  more than 42 minutes. This also blocks Desktop model inventory.
- The candidate bounds all relocated symlink scans under one shared two-second
  deadline while preserving exact local and healthy-relocated sizing.
- With three unresponsive external entries still present, source inventory
  finishes in 3.05 seconds and the packaged sidecar finishes in 4.17 seconds;
  both return 43 healthy rows and skip only the three named bad entries.
- Source server real-weight Chat, Responses, Anthropic, both SSE paths,
  function calling, eight-way concurrency, disconnect abort/recovery, and
  graceful shutdown passed on `qwen3.5-4b-4bit`.
- The isolated release-mode GUI passed onboarding, model inventory, GUI model
  start, exact `GUI_RELEASE_OK` chat, shutdown, relaunch, and conversation
  persistence. The production app and its state were untouched.
- Focused Python tests (124), Ruff, diff check, the supported serialized
  Desktop gate (3,542 tests), full release-mode app/sidecar build, and packaged
  inventory regression passed. The 23,225-test Python collection had only two
  environment-path failures: optional mflux absence (passed after installing
  the release-pinned version) and a Transformers 5.15 wrapped offline miss.
  The latter test-harness compatibility fix and its narrow fail-closed guard
  pass. Hosted exact-head results must still be read from the PR.
- After rebasing onto the Community Benchmark UI changes in `cc7d84982`, a
  second exact-head release build passed. Its packaged inventory returned 43
  healthy rows in 4.19 seconds and its sidecar tarball SHA-256 is
  `108556bdbea7550296863dd72a855dbddb682781401bdefbec19e2ae38cd80ba`.
- Exact-head isolated GUI dogfood verified the Recommended / Downloaded / All
  models grouping, live run scope and elapsed status, cancellation/reaping,
  and a complete real Qwen 3.5 4B benchmark. The GUI and independently
  recomputed medians matched: 171.0 tok/s + 266 ms TTFT for `pp512-tg128`,
  169.6 tok/s + 947 ms TTFT for `pp2048-tg512`. No P0/P1 was found in that
  newly merged surface.
- After the final rebase, `candidate-444eb395` passed a complete release app
  and sidecar build. Its sidecar tarball SHA-256 is
  `bd4cfe48f4a74a71a4d9565353578b66f549acf14439282b67f02706b0bfbcaf`.
  Focused latest-base verification passed 321 Python tests and 72 Swift tests
  covering cache inventory, Community Benchmark, draft-post, and bounded
  visual recovery.
- The latest packaged Qwen 3.5 4B benchmark completed all ten rounds at 170.5
  tok/s + 264 ms TTFT (`pp512-tg128`) and 168.6 tok/s + 944 ms TTFT
  (`pp2048-tg512`). Its stored identity was read from the actual cached model:
  affine four-bit weights, group size 64, resolved revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3`.

## Risk and next action

Atlas should treat this as a release P1 and merge it before building the final
candidate. The external ExFAT cache is still 99.8% full and too unhealthy for
the large-model fleet gate; no cache data was deleted or moved. After merge,
Harbor should rebuild and repeat exact-artifact smoke on the signed/notarized
candidate. Publishing still requires explicit human authorization.
