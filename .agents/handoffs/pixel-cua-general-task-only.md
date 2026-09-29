# Pixel handoff: general Computer Use task surface

- Owner: Pixel
- Branch: `pixel/cua-general-task-only`
- Base: #3857 `b16d5ab0`
- Scope: make the general goal-driven CUA form the only task entry in the
  Computer Use tab by removing legacy workflow cards, placeholder creation UI,
  embedded sheets, presentation state, and ContentView runtime wiring.
- Non-goals: CUA API/view-model behavior and deletion of the underlying legacy
  workflow types or standalone files. Vector owns the stacked code cleanup.

## User-visible result

The tab opens directly to the existing general task form. Goal, brain, process,
window, permissions, Start/Stop, approval, recovery, and event history remain.
The old flow grid, unavailable cards, creation placeholder, and workflow sheets
are absent. The page-owned CUA sidecar remains independent of chat model state.

## Private reference check

Reviewed the current task entry patterns in the required desktop references and
Rapid's existing `CUASection`. The consistent pattern is a single free-form task
composer with model/tool configuration nearby and progressive run state after
submission. Retained Rapid's native form and approval structure rather than
presenting parallel template cards. No reference code or assets were copied.

## Follow-up

Vector should stack on this branch, confirm remaining references, then remove
unused legacy workflow implementation files and their dedicated tests. Do not
remove `CUASection`, `CUAViewModel`, client contracts, or native CUA execution.
