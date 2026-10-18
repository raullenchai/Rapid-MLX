# Desktop 0.16.3: English-first UX dogfood

[Two additional end-to-end journey rounds](journeys/README.md) cover follow-ups,
drafts, history, attachments, editing, stopping, export and relaunch.

Rapid's primary audience is English-speaking users looking for an approachable
local AI assistant. The primary evidence, reproduction steps, screenshots, design
mocks, and priorities in this audit now follow that English product journey.
Earlier Chinese tests are retained as secondary localized observations only.

The product recommendation is to make everyday conversation dependable and easy
to start. Ordinary English drafting requests were mistaken for messages already
sent; a simple packing checklist triggered web tools and a blocking approval while
Personal Intelligence showed Off. These interruptions deserve attention alongside
the entry state, model readiness, and information hierarchy.

These are observed usability risks, **not proven causes of churn**. No customer
interviews, retention data, or funnel data was available. Priority is proposed
product priority, not incident severity.

## Release and method

- Tested [published Desktop `rapid-mac-v0.16.3`](https://github.com/raullenchai/Rapid-MLX/releases/tag/rapid-mac-v0.16.3),
  published October 17, 2026; release commit `76b8296cd1cf3a009de045d719dd4823974afae2`.
- DMG SHA-256: `66b5fedff777fab4d3942b780a74413f13b91105937c9425af205e4dd48ee8d0`.
- Apple M3 Ultra, 256 GB memory, macOS 26.5.2; 1440 × 900 and 820 × 720 windows.
  English retest: October 18, 2026, approximately 14:42–14:46 UTC.
- Used a separate fresh persona with unique bundle identifier, temporary HOME,
  isolated sessions/preferences and a separate port. The release copy was ad-hoc
  signed for isolation. The existing shared model cache was reused; no model was
  downloaded. This is fresh app state with a warm cache, not a clean-machine
  download, Gatekeeper, or notarization test. Existing installation was untouched.
- Ran the selected cached `qwen3.5-4b-4bit` starter with default Tools preferences
  and Personal Intelligence Off. No permission to write files, run commands, or
  send external messages was granted. A browsing request was denied.
- Used native Accessibility actions, checked resulting UI state, and inspected
  screenshots. Primary screenshots show only English test conversations.
- Tested English task entry/replies, onboarding, new chat, compact layout, and
  stopped-model draft readiness. Did not test cold downloads, document processing,
  heavy media generation, microphone permissions, offline networking, updates,
  long-term retention, or representative small-memory hardware.

## English task observations

Nine submitted turns across eight distinct everyday prompts were tested. Each
started in a fresh conversation. All nine streams completed; stream completion
alone does not establish that the user task succeeded. This small exploratory
sample is not a model benchmark or an estimated failure rate.

| Task | Observed result |
| --- | --- |
| Rewrite a work message, three options | Returned three English options without tools; phrasing was somewhat awkward |
| Draft a polite email about a leaking faucet | Twice claimed a message/email was already sent instead of drafting it; no send tool or actual email action occurred |
| Summarize meeting notes in exactly three bullets | Returned one paragraph, omitting the requested bullet format |
| Explain compound interest in under 100 words | Supplied an explanation/example but exceeded the requested length |
| Plan a relaxed Saturday at home | Invoked three web_search calls despite no request for current information; returned a plan and sources |
| Write a Python deduplication function | Returned code and examples for hashable values without tools; execution was not tested |
| Draft a junior designer cover letter | Exposed reasoning about an internal `[MESSAGE SENT]` section and did not deliver the requested letter |
| Create a simple weekend packing checklist | Invoked search and browse; a webpage approval interrupted the task; denial was followed by an answer |

[Task prompts and selected observations](evidence/english-task-observations.json)
contain synthetic test content only. The English writing failure is materially
more actionable than the original language-specific quality concern: the repeated
already-sent claims and exposed `[MESSAGE SENT]` wording point to a context boundary
worth investigating. The release's `ChatViewModel.clockContext` appends a user-turn
timestamp trailer with this marker. That is a **causal lead**, not proof without a
controlled comparison. Preserve correct current-time behavior when fixing it.

## Findings and ownership

| Issue | Priority | English-first finding | Suggested owner / next action |
| --- | --- | --- | --- |
| [#4853](https://github.com/raullenchai/Rapid-MLX/issues/4853) | P1 | Everyday English chat invokes online tools while Personal Intelligence says Off | Pixel + Vector: align visible chat/tool mode with actual offered tools |
| [#4854](https://github.com/raullenchai/Rapid-MLX/issues/4854) | P1 | A stopped downloaded model disables Send for a written English draft | Pixel: offer one explicit Start & send action |
| [#4855](https://github.com/raullenchai/Rapid-MLX/issues/4855) | P1 | Setup prose and central promotion compete with useful task entry | Pixel: provide actionable English starts and a focused composer |
| [#4856](https://github.com/raullenchai/Rapid-MLX/issues/4856) | P2 | Model operations and diagnostics dominate everyday chat space | Pixel / Atlas: disclose operational detail on demand |
| [#4857](https://github.com/raullenchai/Rapid-MLX/issues/4857) | P2 | Model selection demands catalogue knowledge before first value | Pixel: lead with one understandable starter and secondary comparison |
| [#4858](https://github.com/raullenchai/Rapid-MLX/issues/4858) | P1 | Absolute local privacy promises omit online-tool boundaries | Pixel + Vector / Atlas: precise copy consistent with actual configuration |
| [#4859](https://github.com/raullenchai/Rapid-MLX/issues/4859) | P1 | English drafting requests are mistaken for messages already sent | Vector + Pixel: isolate timestamp/context effects and restore useful drafting |

The [two additional workflow rounds](journeys/README.md) added
[P1 #4862: conversation draft isolation and recovery](https://github.com/raullenchai/Rapid-MLX/issues/4862)
and [P2 #4863: unsupported-document paste recovery](https://github.com/raullenchai/Rapid-MLX/issues/4863),
with additional English local-file-tool evidence on #4853.

Prioritize the repeated English drafting failure, the chat/tool contract, and the
first-reply path. The source of the task-quality finding and its priority were
updated after the English retest; the initial Chinese-only examples are no longer
its primary reproduction or evidence. Existing issues were revised rather than
creating duplicate tickets.

## Evidence and design handoff

| Journey | Actual English release UI | Proposed UI |
| --- | --- | --- |
| English tool interruption | [Webpage approval](evidence/unexpected-tool-approval.png), [packing task](evidence/english-checklist.png) | [Explicit chat/tool mode](mockups/tools.png) |
| Model readiness | [Stopped model and draft](evidence/stopped-model-draft.png) | [Start & send](mockups/sleeping.png) |
| First value | [First chat](evidence/first-chat.png), [new chat](evidence/new-chat.png) | [Task-oriented entry](mockups/empty.png) |
| Visual hierarchy | [Compact shell](evidence/compact-chat.png) | [Optional diagnostics](mockups/shell.png) |
| First model | [Model chooser](evidence/model-choice.png) | [Starter first](mockups/setup.png) |
| Privacy | [Welcome](evidence/welcome.png), [Privacy](evidence/privacy.png), [English task using search](evidence/english-plan.png) | [Precise local promise](mockups/privacy.png) |
| English drafting | [Email](evidence/english-email.png), [repeat](evidence/english-email-repeat.png), [cover letter](evidence/english-cover-letter.png) | [Useful email draft](mockups/answer.png) |

All mock conversation content is now English. Editable source:
[design.html](mockups/design.html), with `?screen=empty`, `tools`, `sleeping`, `shell`,
`setup`, `privacy`, or `answer`. These are static visual proposals with authored
sample answers, not shipped behavior, model outputs, or final pixel specifications.
They share a restrained warm palette, amber brand accents, readable conversation
width, familiar history, and one primary action per state.

The privacy mock anticipates a real Chat only mode. Until that exists, copy must
accurately describe current tools and cannot imply the proposed mode has shipped.
Telemetry review preserves the user's saved preference; collection policy is not
being changed. Implementation must retain hardware/disk guards, source disclosure,
cancellation/retry, localization, keyboard access, VoiceOver, contrast, and compact
layouts. No reference-product code or assets are included.

[Selected English Accessibility records](evidence/semantic-observations.json)
support mode/readiness observations. [Secondary localized observations](evidence/secondary-zh/README.md)
retain the earlier Chinese findings without driving the primary English-user
assessment. No raw operator screens, credentials, unrelated permission prompts,
or unfiltered diagnostic logs are published.

## Delivery

This branch changes evidence, design proposals, and handoffs only. App/engine
behavior and production release state are untouched. Fixes belong to the named
receiving roles. Role FYI messaging was not available; ownership and next actions
are recorded in the issues. Validate improvements with English-speaking everyday
users on common 16/24/32 GB Macs, using existing approved activation signals where
available. Define any new instrumentation separately.
