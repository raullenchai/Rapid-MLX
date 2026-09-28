# Pixel handoff: CUA starter runtime availability

- Owner: Pixel
- Branch: `pixel/cua-starter-availability`
- Base: `pixel/cua-navigation-session` at `dbc98519`
- Scope: make the Draft and Post starter accurately unavailable when there is
  no authenticated chat runtime. Free up space and custom high-level CUA runs
  remain available.
- Non-goals: planner migration, server/API work, downloading or starting a
  chat model, and starter execution changes.
- Verification: focused starter catalog regression passes; the broader suite's
  unrelated opt-in test observed contaminated host defaults, while the targeted
  test passes alone.
- Private reference check: followed the native progressive availability pattern
  used by the required desktop references: show the known workflow and its
  current availability, but do not expose an action that can only fail after
  opening. No reference code or assets were copied.
