# Vector: legacy Computer Use workflow cleanup

## Status

Implementation complete on `vector/cua-legacy-workflow-cleanup`, stacked on the
general-task-only Desktop branch at `b03f0dc4`. Draft PR:
https://github.com/raullenchai/Rapid-MLX/pull/3859

## Scope and evidence

- Removed the retired starter catalog, draft-and-post workflow, downloads cleanup
  workflow, and their private macOS observation/actuation/visual-grounding stack.
- Removed tests that exercised only those retired implementations.
- Kept the general Computer Use client, sidecar manager, view model, task panel,
  HTTP API, Python executor, discovery, approval, cancellation, and lifecycle
  tests intact.
- Symbol-reference audit found no production users of the removed generic-named
  macOS helpers outside the retired workflow stack.
- Python source/server/CLI searches found no legacy workflow implementation, so
  this change intentionally makes no Python edits.
- README and current product guides contain no retired workflow claims. One
  historical operations record remains accurate and was retained.

## Verification

- `swift test --disable-sandbox --filter CUA`: 64 tests passed.
- `swift build --disable-sandbox`: passed.
- Python 3.12 CUA contract suite: 137 passed, 2 deselected.
- Legacy production-symbol search: zero matches.
- `git diff --check`: passed.

## Coordination

Pixel owns the stacked general-task UI removal. This cleanup does not edit its
UI files. Atlas received the start scope and ongoing evidence. The role FYI
messaging channel was not available from this task environment; this handoff
records the equivalent start/completion context for the remaining roles.

## Reference check

Rapid's general Computer Use client/server path and task-level session contract
were inspected first. Open WebUI's Functions separation, Cherry Studio's agent
and Computer Use design material, Jan's current public repository, and LM
Studio's tool-use and external-integration documentation were checked. Their
relevant shared pattern is a generic tool/session boundary whose product entry
points do not require hard-coded task workflows. Rapid already has that boundary
in `CUAClient`, `CUAViewModel`, and `/v1/cua`, so the cleanup retains it and does
not introduce a replacement abstraction. No reference implementation provided a
reason to retain the disconnected starter-specific Swift stack, and no external
code or assets were copied.
