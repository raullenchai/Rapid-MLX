# Main HEAD: English desktop journey regression

**The English drafting failure is fixed in this local build. Draft ownership and
recovery remain the most urgent unfinished journey.** Ordinary invitation drafting
also stays in chat instead of requesting a local file write. These results qualify
the current source build; they do not change the historical release observations.

## Build and method

- Main frozen at `329fed00dde9f5e123f3c72ecf0a5842b90e9165`, including #4860 and
  #4861. Built October 18, 2026; GUI testing approximately 16:06–16:13 UTC.
- Full native release-configuration build and newly built bundled engine using
  `apps/rapid-mac/scripts/build.sh`; `BUNDLE_MODEL=0` reused an existing model.
  No source changes, substituted release engine, or new model downloads.
- Candidate identity `candidate-329fed00`; version fields still say 0.16.3 (186).
  The version pill in screenshots therefore **does not identify the published
  0.16.3 release**. The engine stamp independently records `distribution: source`
  and the same full Git revision. Deep/strict code-signature verification passed;
  this is an ad-hoc local build, not a notarized distribution test.
- M3 Ultra, 256 GB, macOS 26.5.2; native 1440 × 900 window. Fresh isolated bundle
  identifier, HOME, preferences, sessions and port; same shared model cache.
  Explicitly selected `qwen3.5-4b-4bit` and kept default Tools settings.
- Native Accessibility actions, observed resulting state and app-window captures;
  synthetic English prompts and fixtures only. Telemetry disabled by launch
  environment. Host work was concurrent; no latency or throughput conclusions.

Reproduction build settings, with a clean main checkout and scratch caches:

```sh
RAPID_CANDIDATE_IDENTITY=candidate-329fed00 BUNDLE_MODEL=0 \
  bash apps/rapid-mac/scripts/build.sh
bash apps/rapid-mac/scripts/dogfood-isolate.sh \
  'apps/rapid-mac/build/Rapid-MLX Desktop.app' /private/tmp/dogfood-main-persona
```

Launch the generated `launch.sh`, preserving the isolated environment. Do not
launch the copy through LaunchServices. The audit explicitly bounded the run to
the 4B model, using the helper's 12 GiB working-set estimate; no larger model was
started. Scratch TMPDIR, pip and compiler caches were used for the build.

## Regression results

| Everyday checkpoint | Current main observation | Disposition |
| --- | --- | --- |
| Fresh app → first chat | Cached-model choice → Start → ready → Start chatting works | Warm-cache path passes; choice density and setup prose remain #4857/#4855 |
| Default local tool boundary | All five local file/command toggles Off; old mode control absent | #4853 fix verified for this fresh persona |
| Draft a landlord email | Exact release prompt repeated in two fresh threads; both produce email drafts | #4859 fixed reproduction; no already-sent claim |
| Draft a cover letter | Returns a letter with company/name placeholders | #4859 fixed reproduction; no internal message stamp exposed |
| Three-bullet meeting summary | Three bullets with the supplied owners and dates | Pass in this sample |
| Refine workshop titles | Uses the second title, Friday 2 PM and Room Cedar; no file-write approval | Context/tool boundary passes; requested two sentences became one |
| Weekend packing checklist | Produces clothing/toiletries/documents checklist, no tool chip or approval observed | Previous interruption did not recur; this does not establish an offline mode |
| Local date control | Correct October 18, 2026 and PDT / America/Los_Angeles | Clock context remains useful after framing fix |
| Switch unfinished work | Workshop draft appears in summary thread; summary draft appears back in workshop | #4862 still reproducible |
| New Chat | Inherits the other conversation's draft | #4862 still reproducible |
| Paste a Word document | Composer contains the local `.docx` path; no attachment chip or unsupported notice | #4863 still reproducible |
| Recover with TXT | Attach succeeds; answer gives $480 and 12 attendees; follow-up gives `CEDAR-7821` | Pass against fixture |
| Edit and retry | Edited question gives $480 / Version 2 of 2; retry gives $480 / Response 2 of 2 | Pass |
| Focused keyboard entry and copy | Shift+Return adds newline; Return submits; Copy puts `Keyboard works.` in clipboard | Pass with explicitly focused composer; immediate typing after New Chat remains a manual check |
| Stop and continue | Partial journal response remains with Stopped; next question receives the requested reply | Pass; cancellation latency was not measured |
| Quit normally and reopen | Transcript, TXT filename, edit/retry versions and chosen model restored | Pass; model warms up automatically |
| Recover unfinished draft | Exact pre-quit draft is empty after reopening its thread | #4862 still reproducible |
| Draft with model unloaded | Draft remains, Send disabled; separate Start then Send required | #4854 remains; starting eventually sends correctly with a second action |

This is exploratory regression evidence, not a model benchmark or estimated
success rate. The two-sentence instruction miss and an email adding an unstated
detail also mean “drafting regression fixed” should not be read as universal
writing-quality acceptance.

## Privacy and entry design

The setup completion now distinguishes local inference from online tools; Privacy
settings qualify the local-data promise with web search/browsing exceptions.
[#4858](https://github.com/raullenchai/Rapid-MLX/issues/4858) is verified at the UI
copy level. Search, browsing and weather remain enabled by default. This round
does not verify every network payload or claim all default chats are offline.

The first chat still presents a seeded model/setup explanation. New Chat still
shows a promotional action, raw model identity, residency and diagnostic metrics.
The existing entry/hierarchy handoffs #4855/#4856 remain relevant; the general
shell has not been redesigned by the tool and timestamp fixes.

## Owner and next action

1. **Pixel: fix #4862 first.** Scope drafts to their conversation, give New Chat
   its own draft, and recover unsent content after ordinary quit. The same main
   build reproduces both cross-thread contamination and restart loss.
2. **Pixel: complete #4854/#4855/#4857.** Make readiness part of task entry, reduce
   beginner model-choice burden and put a useful first action in the empty state.
3. **Pixel: fix #4863 recovery.** Explain unsupported file paste and offer supported
   formats while preserving the user's existing message. Word parsing is not
   required to fix this misleading path substitution.
4. **Pixel/Harbor: validate the repaired journeys on common-memory hardware.**
   Cold downloads, offline recovery, manual immediate typing/focus, actual voice
   capture, image input and media generation remain outside this pass.

The user will arrange fixes. Atlas owns sequencing; Vector supports request/tool
contracts if needed. No implementation PR was created. Role messaging was
unavailable; this report and issue comments are the durable handoff. Findings
were appended to existing issues rather than duplicated. The test app was quit
and its isolated server stopped after verification.

## Selected evidence and existing designs

- Fixed writing: [first email](evidence/email-first.png),
  [repeated email](evidence/email-repeat.png), [cover letter](evidence/cover-letter.png),
  [invitation follow-up](evidence/invitation-followup.png), [clock control](evidence/date-control.png).
- Boundaries: [tool defaults](evidence/tool-defaults.png),
  [setup completion](evidence/setup-complete.png), [Privacy](evidence/privacy.png).
- Unfinished work: [draft in another thread](evidence/draft-crossed-to-summary.png),
  [replacement draft back in workshop](evidence/draft-crossed-to-workshop.png),
  [New Chat](evidence/draft-in-new-chat.png), [before quit](evidence/draft-before-quit.png),
  [restored thread with empty composer](evidence/restored-document.png).
- Files: [unsupported Word paste](evidence/unsupported-word.png),
  [TXT answer](evidence/txt-answer.png), [follow-up](evidence/txt-followup.png),
  [edited question](evidence/edited-answer.png), [retried answer](evidence/retried-answer.png).
- Entry: [model choice](evidence/model-choice.png), [first chat](evidence/first-chat.png),
  [unloaded draft](evidence/unloaded-draft.png), [started but unsent](evidence/started-unsent.png).
- Recovery: [stopped partial reply](evidence/stopped.png),
  [next reply after stop](evidence/after-stop.png),
  [keyboard-submitted reply](evidence/keyboard-send.png).
- [Distilled observations](evidence/observations.json); fixtures remain in the
  [earlier journey evidence](../journeys/evidence/fixtures/).
- The existing English design proposals remain applicable:
  [conversation drafts](../mockups/drafts.png),
  [unsupported-file recovery](../mockups/unsupported.png),
  [start and send](../mockups/sleeping.png),
  [task-first entry](../mockups/empty.png).
  [Editable mock source](../mockups/design.html). These are proposals, not shipped UI.
