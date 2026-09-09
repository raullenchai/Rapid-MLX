# Pixel handoff: MCP connectors under Experimental

- Owner: Pixel
- Branch: `pixel/mcp-connectors-experimental`
- Worktree: `/private/tmp/rapid-mlx-mtp-connectors-experimental`
- Base: `origin/main` at `0d25f3bd`
- Goal: remove the standalone Connectors settings category, expose MCP as an
  explicit Experimental opt-in, rename the Share Compute provider section to
  Inference Pool, and prevent pool models that do not fit the host's unified
  memory from being offered as ready/actionable.
- Non-goals: server port ownership/crash recovery, MCP protocol/runtime
  behavior, QuickSilver registration or routing behavior, model catalog or
  download changes.

## Decisions and verified facts

- The user confirmed the feature is MCP (Model Context Protocol), not MTP.
- MCP expands the model's available tools/data; it does not improve the model
  itself and can add latency. The Experimental toggle says this directly and
  preserves approval-by-default behavior.
- Disabling MCP still clears the live catalog immediately, so tools stop being
  advertised before an eventual engine restart.
- Pool-model compatibility now consumes the centralized
  `ModelSizing.isAvailable` verdict. The same verdict is used by onboarding,
  picker start, auto-start, and cache-aware default selection; it combines the
  footprint classifier with the verified RAM-tier override.
- All three current pool aliases are unavailable on an 18 GB fixture. An
  unavailable selection cannot download, register, or join from Share Compute.
- Removing the Experimental container accessibility identifier was necessary:
  SwiftUI was overwriting every child toggle identifier with the parent's.
  Runtime AX validation now sees five independent switches with readable values.

## Reference check

- Searched the available Orca workspace for an established Settings /
  Experimental placement pattern; no reusable source was present.
- Reused Rapid Desktop's existing Experimental opt-in row and settings section
  components rather than introducing a new navigation or disclosure pattern.
- Reused the existing RAM-tier/footprint compatibility policy and consolidated
  duplicate call sites into its shared verdict.

## Verification

- 218 focused Swift tests passed across MCP, Settings, Share Compute,
  ModelSizing, RAM tiers, model selection, accessibility, and command palette.
- GUI golden flows passed: `settings-persistence`, `settings-mtp`, and
  `no-dead-controls`.
- The no-dead-controls flow toggled MCP on and back off through AX and verified
  the state round trip.
- Light and Dark Experimental + MCP-enabled snapshots were generated. Visual
  inspection confirmed the new hierarchy, copy, scrolling, and empty state.
- The broader legacy snapshot matrix generated the relevant images, then later
  hit a pre-existing missing `ServerManager` environment on an unrelated
  post-Settings fixture. Directly encountered Settings/Images/Tools fixture
  dependencies were repaired; remaining unrelated harness cleanup is follow-up.
- `git diff --check` passed.

## Coordination

The required PR-start FYI channel was not available in this session. This
handoff records the same intention, boundaries, affected areas, and validation
for Atlas, Vector, Harbor, and Echo. Send the matching PR-complete FYI when the
messaging channel is available.
