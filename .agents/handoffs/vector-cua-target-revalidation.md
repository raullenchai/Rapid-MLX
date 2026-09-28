# Vector handoff: selected-window target revalidation

- Owner: Vector
- Branch/worktree: `vector/cua-target-revalidation`, `/private/tmp/harbor-desk-cua-target-revalidation`
- Base: bounded priority grounding stack `baae705be42e8a121b22d40d398dde0b3c065db7`
- Goal: allow a non-gated selected-window action when unrelated dynamic sibling text changes during planning, while preserving fail-closed target, window, and domain binding.
- Design: pre-action re-observation compares the planned index's complete target identity (index, role, label, geometry, center, subrole, actions) and selected window identity/bounds. It no longer requires the entire tree text to match for non-gated actions. URL/domain is still re-read immediately before dispatch.
- Safety: an index shift that places another control at the planned index fails `target_stale`; target label, role, action, or geometry replacement also fails. Window movement/replacement remains `window_stale`. Approval-gated actions retain the stricter whole-tree, target, window, URL, and consent revalidation after the human pause.
- Reference check: existing Rapid target/window identity helpers and last-moment domain check were retained; only the redundant whole-tree equality requirement was removed from the non-gated selected-window branch.
- Verification: regression changes live table values across initial, pre-action, and post-action observations while keeping CPU target stable and proves one click. A second swaps CPU/Memory indices and proves no click. Approval and domain regressions remain green.
- Risk: target identity is based on exposed AX metadata rather than an OS-persistent element token. Identical replacement controls at exactly the same geometry are not distinguishable across fresh AX snapshots; backend snapshot/window validation still runs at dispatch.
