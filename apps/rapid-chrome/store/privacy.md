# Rapid Chrome extension privacy policy

The extension has no telemetry, analytics, advertising, or developer-operated data
collection service. All model processing goes to the model endpoint you choose.
No prompts, page data, API keys, or diagnostic traces are sent anywhere else by
the extension. Browser actions visit websites as directed by your task; those
websites receive ordinary navigation and form submissions under their policies.

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
run checkpoint, and trace, and archives the previous conversation locally.
Debug traces may contain your task, page excerpts, and model output; copying
them is an explicit local clipboard action. Review them before sharing.
There is no automatic diagnostic upload. Extension removal clears its storage.

You choose which tabs to share and approve individual consequential actions.
Stop cancels the task and releases browser control. Password and payment input
are blocked from automated filling; enter sensitive details yourself. Do not
share pages containing information you do not want your chosen endpoint to see.

This policy applies to extension version 0.1.0. Publish this document at the
store listing's privacy policy URL before submission.
