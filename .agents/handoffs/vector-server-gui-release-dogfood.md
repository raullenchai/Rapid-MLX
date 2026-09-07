# Vector -> Atlas: server + GUI release dogfood P1

- Owner/host: Vector, Studio (Apple M3 Ultra, 256 GiB, macOS 26.5.2)
- Branch: `vector/release-dogfood-20260906`
- Latest tested upstream base: `cc7d849821643706da15a9cdc52545ea28a05ace`
- Exact tested branch head: `f09a761a26de553f0e87f66c30bd733759f6f4b2`
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

## Risk and next action

Atlas should treat this as a release P1 and merge it before building the final
candidate. The external ExFAT cache is still 99.8% full and too unhealthy for
the large-model fleet gate; no cache data was deleted or moved. After merge,
Harbor should rebuild and repeat exact-artifact smoke on the signed/notarized
candidate. Publishing still requires explicit human authorization.
