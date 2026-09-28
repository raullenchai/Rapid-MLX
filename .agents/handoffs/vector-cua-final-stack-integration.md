# Vector: final Computer Use stack integration

## Status

Integration complete on `vector/cua-final-stack-integration`, based on the
legacy workflow cleanup at `b2f82b1f` and merging final server head `70050d43`.
Draft PR: https://github.com/raullenchai/Rapid-MLX/pull/3860

## Why this integration is required

The Desktop branch and server branch diverged from their shared base. The
Desktop branch contained the current general task UI and lifecycle but none of
the server API, model-free serving, packaging, or grounding commits. Applying
only the native macOS dependency commit would make imports succeed while
leaving the GUI's required endpoint and lifecycle contracts absent.

## Resolution

- Merged the complete linear server stack rather than copying selected files.
- The merge had no textual conflicts. The three overlapping files
  (`rapid_mlx/routes/cua.py`, `tests/test_cua.py`, and
  `tests/test_cua_server.py`) merged automatically and were reviewed for the
  combined brain configuration, authentication, selected-window, observation,
  approval, idempotent-create, and model-free server contracts.
- Kept the Desktop general-task UI, ambiguous-create recovery, session
  lifecycle, and retired-workflow cleanup unchanged.
- Preserved the native dependency constraints, staged MIT notice, runtime import
  smoke, and Mach-O baseline checks from the server stack.
- Applied the repository's current formatter to two expressions in
  `tests/test_computer_use.py`; behavior is unchanged.

## Verification

- Combined Python CUA suites: 235 passed, 2 deselected.
- Post-format computer-use regression: 68 passed.
- Swift CUA suites: 64 passed.
- Full Swift target/test compilation completed as part of the focused run.
- Ruff check and format check passed for all integrated Python CUA files.
- Packaging tests are included in the 235-test run.
- Existing release-shaped sidecar artifact imports `objc`,
  `ApplicationServices`, and `Quartz` at version 12.2.2 and contains the staged
  MIT notice. A fresh full app/sidecar build is assigned to Atlas for final
  release-shaped signing and bundle verification.
- `git diff --check` passed.

## Risks and next action

The integration itself adds no new design. Its main risk is release packaging:
a fresh signed build must confirm the expected Mach-O count and every nested
binary signature in the final app bundle. Atlas will run that build and the GUI
dogfood after this Draft PR is available.

## Coordination

Atlas and Pixel were notified when the missing full server ancestry was found.
The environment does not expose the full role messaging channel; this handoff
records the equivalent completion FYI for other roles.
