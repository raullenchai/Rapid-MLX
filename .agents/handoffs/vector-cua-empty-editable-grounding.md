# Vector handoff: blank editable grounding

- Owner: Vector
- Branch/worktree: `vector/cua-empty-editable-grounding`, `/private/tmp/harbor-desk-cua-empty-editable-grounding`
- Base: window-occlusion stack `37f9d849c042490e3dc04fe69655fe4680328e42`
- Goal: include empty editable accessibility controls in CUA snapshots so blank documents and form fields can be filled.
- Scope: AX collection inclusion predicate and focused regressions. No node-budget, planner, GUI, server API, or action behavior changes.
- Safety: only known editable roles bypass the label/action requirement. Empty structural/static nodes remain excluded. Secure fields still skip all label/value reads at collection and retain the redacted marker.
- Reference check: existing Rapid fill roles and native accessibility semantics were reused; the change records the same field roles already accepted by fill instead of broadening collection to every unlabeled node.
- Live verification: automated regression reproduces a blank `AXTextArea` with geometry and no actions. The task shell could not launch the installed TextEdit bundle (`kLSNoExecutableErr`) and no TextEdit process was available, so an independent app-owned sidecar dogfood should re-observe a new blank document after packaging this commit.
- Node-cap investigation: `MAX_NODES=600` limits appended interesting targets globally in depth-first child order. Large table rows/cells can exhaust that output budget before later siblings such as top tabs are visited. A separate design should reserve per-region budget or traverse high-value navigation/toolbars before repetitive table content, while preserving deterministic target IDs and truncation reporting; do not merely raise the cap.
