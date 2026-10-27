# Chat context and presentation ownership

The chat owner controls token budgets, context compaction, and tool presentation.
CLI and Desktop clients consume the same additive wire-v1 fields.

## Context accounting

Default chat context is the minimum of 32,768 tokens, the checkpoint's native
window, and the server's advertised `max_model_len`. An explicit context request
is preserved subject to those serving limits. A spawned inference server receives
the selected context length.

Before each model request the owner counts the complete chat template, dynamic
instructions and exposed tools through `/v1/messages/count_tokens`. Repeated
identical requests reuse cached counts; immutable text-part estimates are cached
separately. Actual generation input-token usage calibrates template differences.
If counting is unavailable, a conservative Unicode-aware token estimate is used
and marked `estimated` in context events. Transport and tool-output byte quotas
remain independent and explicitly named in bytes.

The owner reserves output space and compacts when needed:

1. Replace older tool output with short evidence excerpts and omission notices,
   preserving tool names and call/result IDs. Original results remain durable.
2. Summarize older complete turns through tools-disabled, bounded model requests.
   A system message with `summary: true` and `covers_through` on its text part is
   appended atomically to the branch. Projection replaces the covered prefix
   with that checkpoint as untrusted user-reference content, avoiding multiple
   leading system messages on templates that reject them. History and export
   retain the originals. A running
   turn's action revision fence advances in the same transaction.
3. If the current turn still cannot fit, disable further tools and bound its
   evidence for synthesis. Context pressure never produces a budget-error
   terminal. After executed tools, a visible answer or explicit evidence-gap
   answer completes the turn. Budget-exhaustion state persists across synthesis
   requests even when the smaller final context no longer requires compaction.
   Unexpected native tool calls are rejected before dispatch once tools are
   disabled, including when the initial request already exceeds the budget.

Manual `/compact [instructions]` uses the same checkpoint path with branch
revision checks. The CLI does not automatically retry this mutation after an
ambiguous transport failure. Summary generation is cancellable. Summary failure
during automatic compaction falls through to bounded synthesis.

`context` events carry used/limit/free tokens, owner-calculated percentage,
system+tools/conversation/tool-output attribution, estimate status, and compaction
notices. Full-request counts are measured; category attribution is proportional
because template token boundaries can cross categories. A final refresh includes
the answer, so the next prompt indicator reflects the latest context.

The harness can merge adjacent request envelopes before generation. Segment
recovery uses its native `new_messages()` boundary, including pending tool
returns, instead of slicing by the original history length. This preserves every
executed call/result pair when a summary and user message become one envelope.

Synthesis reminders stay in system instructions; no synthetic English user turn
can replace the actual human language. Regeneration fallback language uses the
durable user message because a regenerate request has no new text.

Parent turns have no default request-count or tool-count ceiling. Embedders may
provide explicit `UsageLimits`; helper quotas, context limits, web limits, and
the 16,000-byte tool-result allowance remain separate. When a limit ends the tool
phase, the owner requests a summary of completed work, verified results, and
unfinished work in the user's language with tools disabled. This final request
has a separate allowance and one retry for an empty response. Unexpected tool
calls during summary generation are never dispatched. A durable visible notice
explains that the turn's budget is exhausted and the user can say “continue”. If
the summary itself fails, the owner explicitly reports that failure and retains
completed tool pairs and the continuation notice.

If context preparation disables tools, the owner switches to the summary agent
before generation. Tool-shaped summary text is rejected even when truncated,
and is excluded from the retry context and durable history.

## Presentation

The owner produces tool `label`, `target`, and one-line `summary`, optional bounded
`details`, `details_kind`, and `omitted_lines`. Raw evidence remains in the event
journal; renderers do not parse it to derive summaries. Approval events include a
question title and preview language. Client rendering preserves event order and
explicitly associates interleaved parallel results with their tool.

The terminal displays context at or above 50%, a warning at or above 90%, and dim
compaction notices. Approval selection uses the independent input-reader thread,
arrow/number selection and expandable previews; it does not depend on the main
response loop draining terminal input. Color is optional through `NO_COLOR`.

The base instruction matches the user's language and script, including Simplified
versus Traditional Chinese, unless the user requests otherwise.

## Output truncation and failed turns

The harness defaults to a 32,768-token output ceiling for both thinking modes,
matching the local serving ceiling. Explicit `max_tokens` values remain caps.
Before each model request, the context manager clamps output to the measured
remaining context minus 128 tokens. Compaction still reserves up to a quarter of
the context window; this reserve does not cap the actual response length.

A `length` finish reason on any tool response stops dispatch of the entire
response, including arguments that happen to be valid JSON. The runtime discards
unexecuted calls, preserves completed earlier rounds and visible text, and asks
for complete, shorter arguments. There are at most four retries within the same
request/tool usage budget. Exhaustion produces a visible continuation notice.
Plain-text length stops retain the answer and add a localized “continue” notice.

On generation errors, completed call/result pairs, visible partial text, and a
localized failure answer are committed together with the error terminal. The
store applies the same complete-pair validation as successful turns, so the next
turn sees actions that already happened. Owner logs record the exception class,
never its message or traceback. Process crashes and cancellation retain their
existing recovery behavior; this change covers generation failures in a live
owner.
