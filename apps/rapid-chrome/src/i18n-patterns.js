// Render-time templates for extension diagnostics; page and model text stay unchanged.
export const patterns = [
  ["$1 older steps collapsed. Copy debug trace for the full record.", "olderSteps"],
  [
    "Rapid MLX for Chrome",
    "extensionName"
  ],
  [
    "Chat with your chosen model endpoint and operate explicitly shared browser tabs.",
    "extensionDescription"
  ],
  [
    "Open Rapid side panel",
    "openPanel"
  ],
  [
    "Your browser assistant",
    "tagline"
  ],
  [
    "Settings",
    "settings"
  ],
  [
    "Connecting…",
    "connecting"
  ],
  [
    "Connected",
    "connected"
  ],
  [
    "Reconnecting…",
    "reconnecting"
  ],
  [
    "Get started with Rapid",
    "welcome"
  ],
  [
    "1. Connect a model",
    "stepConnect"
  ],
  [
    "1. Model connected — settings",
    "stepConnected"
  ],
  [
    "2. Share this tab, or open a tab and visit a page, then share it.",
    "stepShare"
  ],
  [
    "3. Try an example task below.",
    "stepTry"
  ],
  [
    "Approvals let you allow one site visit or consequential action at a time.",
    "approvalsHelp"
  ],
  [
    "Stop cancels the task and releases browser control.",
    "stopHelp"
  ],
  [
    "Model connection",
    "connection"
  ],
  [
    "Model endpoint",
    "endpointChoice"
  ],
  [
    "Rapid on this Mac",
    "localEndpoint"
  ],
  [
    "Custom OpenAI-compatible endpoint",
    "customEndpoint"
  ],
  [
    "Endpoint URL (origin only)",
    "baseURL"
  ],
  [
    "Warning: HTTP is unencrypted. Page content and your API key can be read by others on the network; use only a trusted network.",
    "httpWarning"
  ],
  [
    "API key (optional)",
    "apiKey"
  ],
  [
    "Model",
    "model"
  ],
  [
    "First available model",
    "firstModel"
  ],
  [
    "Reasoning effort",
    "reasoning"
  ],
  [
    "Off",
    "off"
  ],
  [
    "Low",
    "low"
  ],
  [
    "High",
    "high"
  ],
  [
    "Model supports vision",
    "vision"
  ],
  [
    "Maximum tool steps",
    "stepCap"
  ],
  [
    "Save & connect",
    "save"
  ],
  [
    "Test saved connection / refresh models",
    "testConnection"
  ],
  [
    "Keys stay in local extension storage. Page data goes only to your chosen model endpoint.",
    "storageHelp"
  ],
  [
    "Use this tab",
    "share"
  ],
  [
    "Open a tab",
    "openTab"
  ],
  [
    "Stop",
    "stop"
  ],
  [
    "Shared tabs",
    "sharedTabs"
  ],
  [
    "New chat",
    "newChat"
  ],
  [
    "Copy debug trace",
    "copyTrace"
  ],
  [
    "Conversation",
    "conversation"
  ],
  [
    "Tool steps",
    "toolSteps"
  ],
  [
    "Approvals",
    "approvals"
  ],
  [
    "Complete the check or resolve the page problem in the tab, then continue.",
    "pausedHelp"
  ],
  [
    "Continue",
    "continueTask"
  ],
  [
    "Message Rapid",
    "messageLabel"
  ],
  [
    "Ask Rapid to help with your tabs…",
    "messagePlaceholder"
  ],
  [
    "Send",
    "send"
  ],
  [
    "attached",
    "attached"
  ],
  [
    "shared",
    "shared"
  ],
  [
    "Release",
    "release"
  ],
  [
    "Working…",
    "working"
  ],
  [
    "Allow once",
    "allowOnce"
  ],
  [
    "Deny",
    "deny"
  ],
  [
    "Debug trace copied",
    "traceCopied"
  ],
  [
    "Could not copy debug trace. Check clipboard access.",
    "traceFailed"
  ],
  [
    "Shopping",
    "shopping"
  ],
  [
    "Travel",
    "travel"
  ],
  [
    "Reservations",
    "reservations"
  ],
  [
    "Mail",
    "mail"
  ],
  [
    "Compare the main prices of these products; ask before purchasing.",
    "shoppingExample"
  ],
  [
    "Find travel options on this page and compare times and prices.",
    "travelExample"
  ],
  [
    "Find available reservations for two people; ask before booking.",
    "reservationsExample"
  ],
  [
    "Summarize the messages on this page and draft a reply; ask before sending.",
    "mailExample"
  ],
  [
    "Visit a page in the new tab, then choose Use this tab.",
    "openTabHelp"
  ],
  [
    "Use an HTTP(S) origin without a path, credentials, query or fragment.",
    "invalidEndpoint"
  ],
  [
    "HTTP is allowed only on loopback, private LAN, or 100.64.0.0/10 addresses. Use HTTPS for other hosts.",
    "insecureEndpoint"
  ],
  [
    "Endpoint permission missing. Save settings and allow access to this origin.",
    "permissionMissing"
  ],
  [
    "Endpoint unreachable or blocked by CORS. Check that the server is running, the URL and network are correct, and CORS allows this extension origin.",
    "unreachable"
  ],
  [
    "HTTP 401: API key rejected. Check your saved API key.",
    "unauthorized"
  ],
  [
    "HTTP 403: access denied. Check endpoint access rules and CORS configuration.",
    "forbidden"
  ],
  [
    "The endpoint returned an HTTP error. Check its server logs.",
    "endpointHTTP"
  ],
  [
    "The endpoint returned invalid JSON. Check the endpoint URL.",
    "invalidResponse"
  ],
  [
    "The endpoint has no loaded models. Load a model and test again.",
    "noModels"
  ],
  [
    "No tool-calling support detected: the model did not return the requested test tool call. Choose a model with tool support.",
    "noTools"
  ],
  [
    "Ready",
    "ready"
  ],
  [
    "running",
    "running"
  ],
  [
    "Done",
    "done"
  ],
  [
    "error",
    "error"
  ],
  [
    "Rapid is working",
    "overlayWorking"
  ],
  [
    "Stopped. No further actions will run.",
    "stoppedNotice"
  ],
  [
    "A task is already running",
    "alreadyRunning"
  ],
  [
    "Stop the task before starting a new chat",
    "stopNewChat"
  ],
  [
    "Stop the task before changing settings",
    "stopSettings"
  ],
  [
    "Stop the task before sharing another tab",
    "stopSharing"
  ],
  [
    "No active tab",
    "noTab"
  ],
  [
    "Stop first",
    "stopFirst"
  ],
  [
    "Debugger detached. Keep working or stop?",
    "debuggerDetached"
  ],
  [
    "Shared tab: $1",
    "sharedTab"
  ],
  [
    "Open $1 in a new tab?",
    "openApproval"
  ],
  [
    "Go to $1?",
    "goApproval"
  ],
  [
    "Add another one? (already added once)",
    "repeatAdd"
  ],
  [
    "Click '$1' on $2?",
    "clickApproval"
  ],
  [
    "$1 '$2' on $3?",
    "actionApproval"
  ],
  [
    "$1 could not load. Reload or resolve the error in that tab, then press Continue. Rapid will verify the page again.",
    "loadError"
  ],
  [
    "$1 requires a human check. Complete the verification in that tab, then press Continue. Rapid will verify that the check is gone.",
    "humanCheck"
  ],
  [
    "Overlay unavailable: $1. Panel Stop remains available.",
    "overlayUnavailable"
  ],
  [
    "Rapid engine HTTP $1. Check API key and --cors-origins.",
    "modelHTTP"
  ],
  [
    "Rapid engine not reachable at $1 — start it with rapid-mlx serve <model>. $2",
    "modelUnreachable"
  ],
  [
    "Rapid engine returned no message",
    "noMessage"
  ],
  [
    "The model returned no final answer; task completion is unconfirmed.",
    "emptyReply"
  ],
  [
    "Task completion is unconfirmed: the model did not inspect or act on the browser.",
    "unconfirmed"
  ],
  [
    "Stopped at step cap ($1). Review progress before continuing.",
    "stepLimit"
  ],
  [
    "Working",
    "workingStatus"
  ],
  [
    "Error",
    "errorStatus"
  ],
  [
    "Stopped",
    "stoppedStatus"
  ],
  [
    "Stopping…",
    "stoppingStatus"
  ],
  [
    "User takeover: tab $1 $2. It is no longer available.",
    "tabTakeover"
  ],
  [
    "released by user",
    "releasedReason"
  ],
  [
    "closed",
    "closedReason"
  ],
  [
    "moved out of its group",
    "movedReason"
  ],
  [
    "detached ($1)",
    "detachedReason"
  ],
  [
    "Cannot map cross-origin frame coordinates",
    "cannotMapFrame"
  ],
  [
    "paused: debugger detached by user",
    "pausedDebugger"
  ],
  [
    "tab_taken_over: user owns this tab now",
    "takenOver"
  ],
  [
    "Task stopped",
    "taskStopped"
  ],
  [
    "unowned_tab: use open_tab or ask the user to share a tab",
    "unownedTab"
  ],
  [
    "tab_detached: ask the user to keep working",
    "detachedTab"
  ],
  [
    "stale_ref: call read_page for fresh refs",
    "staleRefs"
  ],
  [
    "stale_ref: call read_page",
    "staleRef"
  ],
  [
    "DOM inspection failed",
    "domInspection"
  ],
  [
    "blocked_sensitive_field: ask the user to enter passwords or payment details",
    "sensitiveField"
  ],
  [
    "blocked_sensitive_field",
    "sensitiveFieldShort"
  ],
  [
    "stale_ref: target changed during approval; call read_page",
    "approvalChanged"
  ],
  [
    "offscreen_target: no visible area; inspect dialogs or choose another control",
    "offscreenTarget"
  ],
  [
    "obscured_target: hit test failed and approved target is not visible/enabled",
    "obscuredTarget"
  ],
  [
    "stale_ref: approved target changed before DOM activation",
    "changedTarget"
  ],
  [
    "invalid_arguments: choose must contain an option label",
    "chooseEmpty"
  ],
  [
    "ambiguous_option: multiple options match choose; use the returned outline to select an exact option",
    "ambiguousOption"
  ],
  [
    "option_not_found: text was filled, but no unique autocomplete option matched choose; inspect the fresh outline before continuing",
    "missingOption"
  ],
  [
    "Effect observation unavailable",
    "effectUnavailable"
  ],
  [
    "stale_ref: navigation during read; call read_page",
    "navigationDuringRead"
  ],
  [
    "stale_ref: page changed while reading forms; call read_page again",
    "formChanged"
  ],
  [
    "No history entry",
    "noHistory"
  ],
  [
    "Waited $1ms",
    "waited"
  ],
  [
    "Opened tab",
    "openedTab"
  ],
  [
    "Screenshot captured",
    "screenshotCaptured"
  ],
  [
    "$1 completed",
    "completed"
  ],
  [
    "$1 completed. Fresh observation:\n$2",
    "freshObservation"
  ],
  [
    "denied: $1 '$2'",
    "deniedAction"
  ],
  [
    "denied: Go to $1?",
    "deniedGo"
  ],
  [
    "stale",
    "staleShort"
  ],
  [
    "Navigation blocked: $1",
    "navigationBlocked"
  ],
  [
    "needs_human: $1",
    "needsHuman"
  ],
  [
    "invalid_tool: unknown tool or arguments",
    "toolInvalid"
  ],
  [
    "invalid_arguments: missing $1",
    "argsMissing"
  ],
  [
    "invalid_arguments: $1",
    "argsInvalid"
  ],
  [
    "invalid_arguments: choose must be nonempty and cannot combine with submit",
    "chooseInvalid"
  ],
  [
    "invalid_arguments: choose url OR action",
    "urlChoice"
  ],
  [
    "invalid_arguments: choose ref OR direction",
    "directionChoice"
  ],
  [
    "repeated_call: identical call already attempted without progress; do not repeat it. Use the last observation, change arguments, or give a final answer.",
    "repeatedCall"
  ],
  [
    "unobserved_url: URL was not supplied by the user or seen as an href in an outline in this task. Click a link from the page or use site search; do not invent product or venue URLs.",
    "unobservedURL"
  ],
  [
    "blocked_url: browser internals, Web Store, credentials, files, and the engine are unavailable",
    "blockedURL"
  ],
  [
    "blocked_url: malformed URL or host; use an absolute HTTP(S) URL without spaces in the host",
    "malformedURL"
  ],
  [
    "blocked_url: malformed host; use a valid HTTP(S) hostname without encoded spaces",
    "malformedHost"
  ],
  [
    "Error: $1",
    "errorDetail"
  ],
  [
    "Read page",
    "toolReadPage"
  ],
  [
    "Click",
    "toolClick"
  ],
  [
    "Fill",
    "toolFill"
  ],
  [
    "Press key",
    "toolPress"
  ],
  [
    "Navigate",
    "toolNavigate"
  ],
  [
    "Open tab",
    "toolOpenTab"
  ],
  [
    "Scroll",
    "toolScroll"
  ],
  [
    "Wait",
    "toolWait"
  ],
  [
    "List tabs",
    "toolListTabs"
  ],
  [
    "Screenshot",
    "toolScreenshot"
  ],
  [
    "ok",
    "ok"
  ],
  [
    "URL: $1",
    "outlineURL"
  ],
  [
    "Title: $1",
    "outlineTitle"
  ],
  [
    "Page data (untrusted):",
    "outlineData"
  ],
  [
    "Product: $1",
    "outlineProduct"
  ],
  [
    "Form controls (use refs in outline):",
    "outlineForms"
  ],
  [
    "[outline truncated: $1 entries/cards omitted; use read_page(query) to narrow results.]",
    "outlineTruncated"
  ],
  [
    "[More below/outside viewport; click known refs directly, or read_page(query) for omitted content.]",
    "outlineMore"
  ],
  [
    "[No matching text for the query; showing the normal viewport-first outline. Use these fresh refs for site search or other controls.]",
    "outlineNoMatch"
  ],
  [
    "$1. Fresh observation:\n$2",
    "freshResult"
  ],
  [
    "Fresh observation unavailable: $1. Wait then read_page; do not repeat the action.",
    "observationUnavailable"
  ],
  [
    "Submit form",
    "submitForm"
  ],
  [
    "Press $1",
    "pressKey"
  ]
];
