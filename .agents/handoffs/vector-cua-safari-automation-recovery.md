# Vector handoff: Safari Automation recovery

- **Owner:** Vector, coordinated by Atlas
- **Branch:** `vector/cua-safari-automation-recovery`
- **Worktree:** `/private/tmp/harbor-desk-cua-url-tcc`
- **Base:** integration aggregate `20b113017ae14a0ba8b3c96f682f63b9591ddf64`
- **Scope:** trusted browser URL read and focused backend tests

## Finding

The macOS helper already carries the Apple Events usage description and
automation entitlement. Safari's first AppleScript URL read can remain blocked
while macOS presents the Automation consent dialog. The backend used a five
second timeout and swallowed `subprocess.TimeoutExpired`, so the domain guard
correctly failed closed but Desktop received only an empty URL instead of the
existing `automation_permission_required` recovery flow.

## Change

For an allowed-domain read, the first trusted URL request for a browser bundle
now has a bounded 30-second consent window. A successful AppleScript response
marks that bundle ready in the helper process; later per-step reads retain the
five-second timeout. Denial and initial timeout both surface
`automation_permission_required`. Reads without a domain policy retain the
short timeout and best-effort empty-string behavior. No page-controlled AX URL
is trusted, and no action occurs without a validated URL.

This adapts the existing typed backend error and `CUAViewModel` recovery path;
no new UI contract is introduced. Pixel should confirm the existing Automation
settings prompt remains visible and understandable during Mini dogfood.

## Verification

- `tests/test_computer_use.py`: 183 passed
- `tests/test_cua.py::test_loop_surfaces_typed_browser_automation_denial`: passed
- Ruff format and lint: passed on both changed Python files

Focused tests cover the longer initial timeout, transition to the short steady
timeout, explicit Apple Events denial, timeout recovery, and unchanged
best-effort behavior without an allowed-domain policy.

## Remaining live check

On the Mac mini, clear or reset the helper-to-Safari Automation decision, start
an allowed-domain Safari task, respond to the system dialog within 30 seconds,
and confirm the task proceeds only after a trusted URL is returned. Also verify
denial and no-response cases show the existing actionable Desktop recovery.
