# Vector handoff: bounded priority CUA grounding

- Owner: Vector
- Branch/worktree: `vector/cua-priority-grounding`, `/private/tmp/harbor-desk-cua-priority-grounding`
- Base: blank-editable grounding stack `da982e4ef654a19d6153ae4ee076c0c508f06791`
- Goal: retain high-value top controls in bounded AX snapshots even when an earlier long table can consume all 600 targets.
- Design: stable-sort small sibling regions so toolbars/tab groups/menu bars and direct controls precede repetitive tables/lists. Preserve original AX order within each priority. Sibling lists over 64 are not prescanned, avoiding one extra AX role IPC per row in large virtualized collections. The 600-target and depth bounds remain unchanged.
- Safety: target IDs are assigned after deterministic traversal and retain the exact live element reference used by actions. Selected-window collection, secure-field redaction, and truncation reporting are unchanged.
- Reference check: existing bounded collector and platform AX region roles were reused. A global budget increase was rejected because it expands prompt size and collection time without ensuring later window regions appear.
- Verification: regression places a long table before CPU/Memory controls and proves controls receive the first stable IDs within a four-target budget; another proves a 65-child collection is not role-prescanned beyond visited targets.
- Performance risk: small sibling groups add at most 64 role reads per node to prioritize regions. Large row collections stay in native order and stop at the existing output cap. App-owned Activity Monitor dogfood should confirm its concrete hierarchy exposes the top controls with this ordering.
- Studio dogfood: Activity Monitor window observation remained bounded at 600 with `truncated: true`; the first 17 targets included Stop, Inspector, Actions, plus CPU, Memory, Energy, Disk, and Network radio buttons. CPU and Memory were actionable at indices 4 and 5. Collection took approximately 9 seconds, inside the existing 20-second collection timeout but material enough to monitor.
- Draft PR: https://github.com/raullenchai/Rapid-MLX/pull/3853 (not Ready and not merged).
