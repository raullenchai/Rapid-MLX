# Pixel handoff: CUA navigation session

- Owner: Pixel
- Branch: `pixel/cua-navigation-session`
- Base: `pixel/cua-model-free-server` at `bbec0bab`
- Scope: keep the native Computer Use form, active run, polling, Stop, and
  approval state alive while the user navigates between Desktop sections.
- Non-goals: server run discovery, process persistence across app relaunch,
  server or action behavior changes.

## Verified facts

- `ComputerUseView` previously created its own `CUAViewModel`. Leaving the
  section destroyed the only local run identity while the app-owned sidecar
  and server run could continue.
- The app-owned `CUAServerManager` now owns one view model for the exact
  authenticated sidecar session. Navigation reuses it. A stopped, failed, or
  replaced sidecar clears it, so old run controls cannot address a new
  endpoint.
- Focused CUA tests pass, including preservation of goal, planner, active
  approval, and object identity across repeated panel presentation, plus a
  fresh idle model after a sidecar restart.

## Private reference check

Reviewed the existing native app session ownership patterns and the analogous
task/session flows in the required desktop references. Adopted the established
pattern of owning long-lived task state above transient navigation content,
while binding it to one authenticated local service session. A persisted
run-only cache was rejected because it cannot safely reconstruct approval and
ambiguous-create state from the current API.

## Remaining validation

- Rebuild the combined Desktop stack and dogfood: start a run, navigate to
  chat, return, approve or stop it, then restart only the CUA sidecar and
  confirm the old controls disappear.
