# Two additional English journey rounds

The core journey is: **open → ask → get something useful → refine → use the result
→ return later without losing work**. These two rounds exercise that chain across
real native controls and real local inference, rather than treating each screen
or completed response as a successful user journey.

The release remains Desktop **0.16.3** (`76b8296cd1cf3a009de045d719dd4823974afae2`).
Tests ran October 18, 2026, approximately 15:00–15:15 UTC, on the same M3 Ultra /
256 GB / macOS 26.5.2 host. The prior English-only isolated persona was reopened
to test returning-user behavior. Shared existing models were reused; no models
were downloaded. The main workflow model was explicitly selected as
`qwen3.5-4b-4bit`. Release-copy identity/signing and warm-cache limitations remain
as described in the [main audit](../README.md).

The most important new findings are **draft ownership and recovery**, plus a
**misleading unsupported-file paste path**. The existing tool-mode issue also now
has an English local-file-write reproduction in an ordinary follow-up. No app
behavior was changed by this audit, and the journeys are not yet all smooth.

## Round A: open, ask, refine, use and find

The task was to plan a small workshop: suggest three titles, use the second title
in an invitation, copy the result, leave an unfinished revision, switch tasks,
and find the original conversation again.

| Checkpoint | Observation | Disposition |
| --- | --- | --- |
| Reopen a prepared app | Existing history remained available; no onboarding repeat | Works in this persona; cold start is outside this round |
| Start with the intended model | Following a prior explicit unload, an initial launch chose a different cached model; after selecting Qwen, a later ordinary quit/relaunch retained it | Unload/relaunch selection needs a separate controlled check; keep #4854 selection acceptance |
| Type a multiline message | Shift+Return inserted a newline | Verified on the first launch |
| Send using Return | Created the workshop thread and received three titles | Verified on the first launch |
| Refine the previous answer | The reply used the requested second title and Room Cedar | Context worked, but a local_write approval interrupted the task: #4853 |
| Copy a useful reply | Clipboard contained the response Markdown and Room Cedar | Verified; copied content was synthetic |
| Switch to New Chat | An unfinished workshop draft remained in the new chat | [#4862](https://github.com/raullenchai/Rapid-MLX/issues/4862) |
| Switch between existing chats | The other conversation's draft appeared in the selected thread; returning showed the replacement text | [#4862](https://github.com/raullenchai/Rapid-MLX/issues/4862), independently reproduced |
| Find a chat by message content | Searching “Room Cedar” returned the workshop even though its title did not contain that full phrase | Verified |
| Rename a chat | “Cedar workshop invitation” persisted after relaunch | Verified |
| Export Markdown | Native Save panel reached; an 850-byte file was written to scratch, containing the renamed title and follow-up room | Verified output, not just menu acknowledgement |
| Copy a code block | Native Copy button copied only the 199-character example code block, without the surrounding explanation | Verified using the earlier English code thread; execution was not tested |
| Open New Chat and immediately type | AX focus was on the window in some states; later negative and positive Unicode-injection controls both failed after relaunch | **Inconclusive as a lost-keystroke defect**; retain manual typing/focus acceptance in #4855, do not file a confirmed new bug from this |

## Round B: attach, recover, revise, stop and reopen

The document task used generated local fixtures only. The brief stated 12 attendees,
a $480 budget, and the distinctive code `CEDAR-7821`; the PDF used `CEDAR-PDF-8246`.
The CSV contained four cost rows summing to 480.

| Checkpoint | Observation | Disposition |
| --- | --- | --- |
| Paste a Word document | Its local file path appeared as ordinary draft text, without an attachment chip or unsupported-format explanation | [#4863](https://github.com/raullenchai/Rapid-MLX/issues/4863); Word parsing is outside the recovery fix |
| Attach TXT after that failure | TXT chip appeared and remained removable | Recovery with a supported file worked |
| Paste the same TXT again | One chip remained and “That file is already attached.” appeared | Verified |
| Ask about the TXT | Answer correctly stated $480 and 12 attendees | Verified against fixture |
| Ask a document follow-up | Answer was `CEDAR-7821` | Verified; user did not need to attach again |
| Edit the sent follow-up | Revised question returned `$480`, with “Version 2 of 2” | Verified; editor focus was exposed correctly |
| Retry the answer | Returned `$480`, with “Response 2 of 2” | Verified |
| Stop before the first visible answer | Returned to Send and showed Stopped | Verified UI recovery; no content existed to preserve |
| Stop after visible answer content | Partial journal text remained; Stopped and Send appeared | Verified; no claim about cancellation latency is made |
| Ask a new question after stopping | Returned “Ready for the next question.” | Verified |
| Ask about a text PDF | Returned `CEDAR-PDF-8246` | Verified against fixture; OCR/scanned PDFs untested |
| Ask about a CSV | Returned `480` | Verified against fixture; large/complex spreadsheets untested |
| Reopen document history | Filename, answer, edit version and retry version were retained | Verified |
| Reopen an unfinished draft | “Unsent Cedar draft: Please RSVP by Thursday afternoon.” disappeared, including after reopening its conversation | [#4862](https://github.com/raullenchai/Rapid-MLX/issues/4862), same draft lifecycle owner |
| Locate voice input | Audio → Speech to Text / Text to Speech controls were present | Discovery only; no microphone recording, audio-model start, permission grant or transcription was tested |

The positive behaviors are worth preserving: follow-up context, supported-document
reuse, duplicate feedback, editable/regenerable branches, copy, history search,
rename, export, stopped-response recovery, and saved transcript restoration.
They should form the basic release journey checks alongside the fixes below.

## Fix handoff and product order

1. **Deliver a useful first answer.** Fix English drafting confusion in #4859 and
   the chat/tool boundary in #4853. The invitation follow-up asked to draft text;
   a file write was not requested. Denial was respected and a chat answer followed.
2. **Protect unfinished work.** Scope drafts to their conversation and recover
   them after normal quit. New Chat must not inherit another thread's message.
3. **Keep starting ordinary.** Complete #4854/#4855/#4857: honest setup, one
   understandable model path, one start/send action and keyboard-ready task entry.
4. **Explain file recovery.** Recognize unsupported file paste as an attachment
   attempt. Preserve the existing message, name the unsupported type, and offer
   PDF/CSV/TXT recovery. Do not silently substitute a filesystem path for a file.
5. **Polish the quiet shell.** Keep #4856 and the precise privacy boundary #4858
   aligned with the same everyday journey.

Filed handoffs:

- [P1 #4862: drafts cross conversations and disappear after quitting](https://github.com/raullenchai/Rapid-MLX/issues/4862).
- [P2 #4863: unsupported document paste silently inserts a local path](https://github.com/raullenchai/Rapid-MLX/issues/4863).
- [English local_write follow-up evidence added to #4853](https://github.com/raullenchai/Rapid-MLX/issues/4853#issuecomment-6251585222).

Pixel owns draft and attachment UI fixes; Vector supports tool/request boundaries;
Atlas owns cross-feature compatibility and sequencing. The user will arrange
implementation. The role FYI channel was unavailable; these are durable handoffs.

## Evidence and proposals

- Draft: [new chat](evidence/draft-in-new-chat.png),
  [switch to document](evidence/draft-crossed-to-document.png),
  [switch back](evidence/draft-overwrote-workshop.png),
  [before quit](evidence/draft-before-quit.png),
  [after relaunch](evidence/draft-after-relaunch.png).
- File recovery: [Word path pasted into composer](evidence/unsupported-word-paste.png),
  [supported TXT](evidence/txt-attachment.png), [correct TXT answer](evidence/txt-answer.png),
  [document follow-up](evidence/document-followup.png).
- Lifecycle: [edited prompt](evidence/edited-prompt.png),
  [retry](evidence/retried-answer.png), [stopped partial response](evidence/stopped-response.png),
  [new reply after stop](evidence/after-stop.png),
  [search](evidence/search-history.png), [restored document](evidence/restored-document.png).
- Formats: [PDF answer](evidence/pdf-answer.png), [CSV answer](evidence/csv-answer.png).
- [Selected semantic observations](evidence/selected-observations.json),
  [distilled export verification](evidence/export-check.json),
  [synthetic fixtures](evidence/fixtures/).
- Proposed UI: [conversation-owned drafts](../mockups/drafts.png),
  [unsupported-file recovery](../mockups/unsupported.png),
  [keyboard-ready entry target](../mockups/focus.png).
  Editable source: [design.html](../mockups/design.html), with `?screen=drafts`,
  `unsupported`, or `focus`. These are authored proposals, not shipped behavior.

Raw AX transcripts, long generated text, native Save dialog history, unrelated
permission prompts, operator files and clipboard contents are excluded. A native
AX action once returned a stale-element error after successfully opening the edit
view; this was verified by observing the actual editor, not filed as an app bug.
Native Save-panel shortcuts required a global key-event fallback; the saved file
was checked afterward. Streaming made some AX reads slow, so this audit does not
infer user input latency or model performance from automation duration.

## Acceptance for the everyday journey

An English-speaking user should be able to open a prepared app, type immediately,
receive a relevant answer, refine it using context, copy it, start another task,
and return to the first with the right draft and history. A supported document
should attach, answer, and remain usable across follow-ups and relaunch. Failure,
cancellation and unsupported formats must leave a clear next action and intact
work. These are acceptance targets; the unfixed release does not pass all of them.

Next independent verification: Pixel should repeat manual focus/typing and the
fixed draft/file journeys on a common 16/24/32 GB Mac; Pixel and Harbor should own
cold-cache admission/download, offline recovery, microphone/voice and image-input
runs. Those are explicit remaining coverage gaps, not implied passes or new
performance claims. No product telemetry was added and no retention cause proven.
