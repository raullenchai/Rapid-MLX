# Vector handoff: CUA window occlusion

- Owner: Vector
- Branch/worktree: `vector/cua-window-occlusion`, `/private/tmp/harbor-desk-cua-window-occlusion`
- Base: model-free CUA server stack `aec75f34115cc58cf11e51299366307f8d4c0372`
- Goal: ignore non-normal macOS system overlay layers when determining which real application window is topmost at an action coordinate, while preserving fail-closed checks for normal-layer blocking windows.
- Scope: `_topmost_window_id_at` and focused regression tests only. No AX collection, node-cap, GUI, planner, or action API changes.
- Verified dogfood fact: filtering the topmost query to visible layer 0 in a temporary process allowed the same selected TextEdit run to press Bold; its AXValue changed from 0 to 1. Product code was not modified during that check.
- Local live recheck limitation: the Studio shell's CoreGraphics window-list call returned null, consistent with Screen Recording/TCC not being granted to this shell, so it could not enumerate the still-open TextEdit window. Automated fixtures reproduce the observed layer 21 overlay ahead of a layer 0 blocking panel and target.
- Reference check: existing Rapid discovery already accepts only normal-layer application windows. The occlusion probe now applies the same macOS window-layer boundary rather than owner-name exceptions, which would be brittle and incomplete.
- Follow-up: blank editable AX controls and the AX node budget remain a separate grounding task.
