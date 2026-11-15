# Permission justifications

- `debugger`: reads accessibility trees, inspects controls, and performs ordered
  browser actions on explicitly shared or task-created tabs. Also supports
  screenshots when the user enables a vision model. Detaches on Stop, completion,
  or user takeover. The browser displays its debugging indicator.
- `tabs`: identifies the current tab for explicit sharing and tracks navigation,
  closure, and ownership. Tab metadata is not uploaded in bulk.
- `scripting`: installs an isolated status border and Stop control in owned tabs
  where access permits. The debugger isolated-world path handles other shared pages.
- `tabGroups`: groups task-created tabs and detects when the user moves a tab out
  of the task group, relinquishing ownership.
- `storage`: local settings and API key; session transcript, shared-tab ownership,
  and debug trace. There is no synchronized storage.
- `sidePanel`: hosts the conversation, settings, approvals, and Stop button.
- Optional `https://*/*` and `http://*/*`: declare support for user-selected model
  hosts. Save requests only the selected origin, never a wildcard host. The
  browser's host grants cover all ports on that host. Local discovery checks
  ports 8000 and 7659 on 127.0.0.1. HTTP is limited to loopback, private IPv4 LAN,
  and 100.64.0.0/10 addresses, with a visible unencrypted-transport warning.
  HTTPS is required for other hosts. Website control uses explicit tab ownership
  and the debugger, not a blanket optional host grant.
