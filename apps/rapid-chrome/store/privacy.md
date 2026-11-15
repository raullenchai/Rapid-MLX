# Rapid Chrome extension privacy policy

The extension has no telemetry, analytics, advertising, or developer-operated data
collection service. All model processing goes to the model endpoint you choose.
No prompts, page data, API keys, or diagnostic traces are sent anywhere else by
the extension. Browser actions visit websites as directed by your task; those
websites receive ordinary navigation and form submissions under their policies.

The current tab’s URL and title are omitted from model requests until the tab
is explicitly shared or browser control is granted. The tab identifier alone may
be included to route a current-page request.

Tasks, conversation history, accessibility page text, tool results, and optional
screenshots are sent to your chosen model endpoint to carry out your request.
The API key is sent only to that endpoint in its Authorization header. It is
stored in local extension storage, never synchronized, logged, or included in
debug traces. HTTP endpoints are unencrypted: use a trusted network or HTTPS.
A remote endpoint may retain data under its own policy; review that policy.

Settings and the API key remain locally until replaced or the extension is
removed. Recent conversations (up to 60 messages each and eight archived chats)
and the current run checkpoint remain in local extension storage across browser
sessions. Checkpoints include partial answers and bounded step details, which
can contain page excerpts and action arguments. Tab ownership and diagnostic
traces use browser session storage. New chat clears the current conversation,
run checkpoint, and trace when you choose “Delete all chats and activity”; this
also deletes archived chats. “Keep in history” archives the previous conversation
locally. Cancel leaves the conversation in place. Settings and tab sharing are
kept in either case. Stored tool arguments are limited to 4 KiB, results to 8 KiB,
and diagnostic events to 16 KiB each (UTF-8 JSON bytes), with at most 200 events.
Debug traces may contain your task, page excerpts, and model output; copying
them is an explicit local clipboard action. Review them before sharing.
There is no automatic diagnostic upload. Extension removal clears its storage.

You choose which tabs to share and approve individual consequential actions.
Stop cancels the task and releases browser control. Password and payment input
are blocked from automated filling; enter sensitive details yourself. Known web
vaults, listed banking login sites, and pages with visible password or payment
fields require human control before page reads, screenshots, or actions. These
heuristics are conservative and cannot identify every sensitive page. Approval
cards show bounded form details with sensitive values masked; values are checked
again before dispatch. Financial actions without inspectable form values require
human completion. Do not
share pages containing information you do not want your chosen endpoint to see.

This policy applies to extension version 0.1.0. Publish this document at the
store listing's privacy policy URL before submission.
