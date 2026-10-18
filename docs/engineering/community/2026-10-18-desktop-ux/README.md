# Desktop 0.16.3: conversation-first dogfood

Rapid's Desktop goal is an approachable everyday local AI assistant. This audit
found avoidable interruptions between typing a simple request and receiving a
useful answer. The central product recommendation is to make conversation the
default experience, with model operations and agent capabilities available through
deliberate, understandable choices.

These are observed usability risks and design proposals, **not proven explanations
of user churn**. No retention data, customer interviews, or funnel data was available
in this audit. Priority means proposed product priority, not incident severity.

## Release and method

- Tested the [published Desktop release `rapid-mac-v0.16.3`](https://github.com/raullenchai/Rapid-MLX/releases/tag/rapid-mac-v0.16.3),
  published October 17, 2026. Release commit: `76b8296cd1cf3a009de045d719dd4823974afae2`.
- DMG SHA-256: `66b5fedff777fab4d3942b780a74413f13b91105937c9425af205e4dd48ee8d0`.
- Host: Apple M3 Ultra, 256 GB memory, macOS 26.5.2. Window sizes: 1440 × 900 and
  820 × 720. Session: October 18, 2026, starting around 14:25 UTC.
- Ran a copy of the signed release with a unique bundle identifier and temporary
  HOME using `scripts/dogfood-isolate.sh`; that copy was ad-hoc signed. Preferences,
  sessions, and server port were isolated; the existing shared model cache was reused.
  Existing installation was version 0.15.6 and was not used or modified.
- This is **fresh app state with a warm model cache**, not a clean-machine download
  or Gatekeeper/notarization test. No new model was downloaded.
- Used real native Accessibility operations and inspected screenshots. Assertions
  describe observed state after an action, rather than trusting an AXPress success
  message. Screenshots of sheets use their separate window IDs.
- Tested onboarding, a cached starter, Chinese and English rewriting requests,
  permission denial/recovery, new chat, draft survival across conversation changes,
  model unload/restart, compact layout, and Tools/Privacy settings. Did not test
  heavy media generation, microphone permissions, cold downloads, offline networking,
  document processing, update installation, or long-term retention.

## Findings and ownership

| ID | Priority | Observed problem | Proposed owner | Next action |
| --- | --- | --- | --- | --- |
| [UX01 / #4853](https://github.com/raullenchai/Rapid-MLX/issues/4853) | P1 | Ordinary Chinese rewriting requests trigger local file actions while Personal Intelligence says Off | Pixel + Vector | Make a visible conversation tool mode agree with the actual tool surface |
| [UX02 / #4854](https://github.com/raullenchai/Rapid-MLX/issues/4854) | P1 | A stopped, downloaded model disables Send and requires a separate Start | Pixel | Queue the draft behind one explicit Start & send action |
| [UX03 / #4855](https://github.com/raullenchai/Rapid-MLX/issues/4855) | P1 | Seeded setup prose and the new-chat empty state do not help users reach their first useful task | Pixel | Replace setup-as-transcript and central promotion with actionable starts |
| [UX04 / #4856](https://github.com/raullenchai/Rapid-MLX/issues/4856) | P2 | Diagnostics and model operations occupy permanent primary-chat space | Pixel; Atlas for navigation scope | Put chat/history first; move operational detail into an accessible inspector |
| [UX05 / #4857](https://github.com/raullenchai/Rapid-MLX/issues/4857) | P2 | First model selection exposes many overlapping technical choices before first value | Pixel | Lead with one explained starter and a secondary comparison route |
| [UX06 / #4858](https://github.com/raullenchai/Rapid-MLX/issues/4858) | P1 | Absolute local privacy promises omit enabled online-tool boundaries | Pixel + Vector; Atlas for copy | Make local-processing and optional-network behavior precise everywhere |
| [UX07 / #4859](https://github.com/raullenchai/Rapid-MLX/issues/4859) | P2 | The selected starter misreads the Chinese task even after local file tools are disabled | Vector + Pixel | Investigate app instructions/model behavior and validate the starter on everyday tasks |

## Evidence

The `evidence/` directory contains only app screenshots from this synthetic test and
selected semantic records. Raw logs, window inventories, operator screens, private
paths, credentials, and unrelated permission prompts are excluded.

- [Initial welcome](evidence/welcome.png), [model selection](evidence/model-choice.png),
  [first chat](evidence/first-chat.png), [new chat](evidence/new-chat.png).
- [Unexpected file approval](evidence/unexpected-file-approval.png),
  [default tools](evidence/default-tools.png), [privacy settings](evidence/privacy.png).
- [Compact chat](evidence/compact-chat.png),
  [stopped model with a written draft](evidence/stopped-model-draft.png).
- [Answer with local file tools disabled](evidence/local-tools-disabled-answer.png),
  [selected Accessibility observations](evidence/semantic-observations.json).

The same Chinese rewrite request produced `local_write` approvals on two fresh
conversations. Personal Intelligence read Off both times. Denial allowed a response
to follow, but it changed the intended task: asking the coworker to provide three
reports rather than offering three phrasings. Disabling only `local_write` led to
`local_read` and `local_search` attempts. Disabling all five local file/command tools
removed that interruption in the next sample, but did not fix task understanding:
the answer mixed English and Chinese, supplied four numbered options instead of
three, and still asked the coworker for three versions of the report. A successful
`web_search` chip was also visible: disabling local file tools did not make this
request local-only. No wire payload was captured. These are
small-sample observations; root cause and model-family generality remain unproven.

A further all-tools-disabled sample still treated the three requested rewrites as
three report versions and added irrelevant date/time commentary. An explicitly
quoted control prompt that specified exactly three Chinese outputs and no new
report-count requirement did produce three Chinese variants without tool use.
That limits the finding: natural task interpretation and default experience need
evaluation; this does not establish a universal Chinese/model failure. See
[all-tools-disabled sample](evidence/all-tools-disabled-answer.png) and
[explicit prompt control](evidence/explicit-prompt-control.png).

An English rewrite request explicitly asking for a reply in chat completed normally:
the UI reported 130 tokens/sec, 1.0 seconds to first token, and 1.9 seconds generation
time. Those are app-reported values for one run on this host, not general performance
benchmarks. Draft text survived a conversation switch and model restart. The selected
cached starter loaded successfully. These working behaviors should be preserved.

## Design handoff

| Proposal | Preview |
| --- | --- |
| Task-oriented new chat | [empty.png](mockups/empty.png) |
| Explicit Chat only / Use tools | [tools.png](mockups/tools.png) |
| Start & send | [sleeping.png](mockups/sleeping.png) |
| Quiet shell and optional diagnostics | [shell.png](mockups/shell.png) |
| Recommended starter first | [setup.png](mockups/setup.png) |
| Precise privacy promise | [privacy.png](mockups/privacy.png) |
| Expected Chinese rewrite result | [answer.png](mockups/answer.png) |

Editable source: [design.html](mockups/design.html). Open it locally with
`?screen=empty`, `tools`, `sleeping`, `shell`, `setup`, `privacy`, or `answer`.
These are static visual proposals with illustrative copy and values, not a working
replacement app or pixel specifications. They share a warm neutral palette,
restrained amber brand accents, a consistent sidebar, readable conversation width,
and one primary action per state. Existing branding need not change.

The privacy mock anticipates a real Chat only mode; until UX01 is implemented, copy
must describe current tool behavior and cannot imply this mode already exists.
Its telemetry review link preserves the user's actual setting and does not propose
a change to collection policy. Download size, hardware-fit warnings, source choice,
cancel/retry, accessible controls, keyboard navigation, contrast, localization,
and compact layouts must remain part of implementation acceptance.

## Delivery and follow-up

This branch contains evidence and design handoffs only. No application behavior,
engine code, telemetry policy, model selection policy, or release was changed.
The user authorized filing findings as GitHub issues with mock designs. All seven
linked findings are now filed and ready for owner triage; fixes are for the receiving
owners to implement. Atlas should sequence UX01/UX06, UX02/UX03,
before optional visual reorganization; investigate UX07 alongside starter validation. Keep each implementation scoped to
its own issue rather than turning the audit into a wholesale rewrite.

The requested Orca role FYI messaging channel was not exposed in this session.
Ownership and next steps are recorded here and in the linked GitHub handoffs;
no substitute agents were spawned or unsolicited messages sent.

To measure whether fixes help, use existing privacy-approved aggregate activation
milestones where available: setup completion, first successful reply, and return
usage. Define missing measurements separately before adding instrumentation.
Validate with everyday users on common 16/24/32 GB Macs; this large-memory host is
not representative evidence of their first-load experience.
