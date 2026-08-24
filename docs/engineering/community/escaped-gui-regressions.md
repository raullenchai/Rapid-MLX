# Escaped GUI regression ledger

This ledger records user-visible Desktop regressions that reached dogfood or a
released build, and the regression layer added after discovery. Its purpose is
to make repeated gaps visible and keep each fix at the cheapest layer that can
prove the behavior.

This is evidence tracking, not a bug counter or quality score. Add a record only
when the escaped symptom and regression coverage are linked to repository
evidence. Do not infer coverage from a PR description when the named test no
longer exists on `main`.

## Record template

Copy this block for each independently testable root boundary. Split reports
that contain multiple unrelated causes.

```markdown
### <short symptom> — <issue or dogfood date>

- **Escaped symptom:** What the user observed and where it escaped.
- **Root boundary:** The state, ownership, lifecycle, or rendering boundary
  that allowed the behavior.
- **Cheapest effective regression layer:** Unit, integration/contract,
  XCUITest, visual/geometry, or real-hardware dogfood, with why that layer can
  prove the fix.
- **Evidence:** Issue/dogfood report, merged fix PR, and test file or journey.
- **Remaining gap:** A boundary the added test does not prove, or `None known`
  when the evidence closes the reported behavior.
```

When a remaining gap is later covered, update the existing record with the
merged PR and test link. Do not replace historical evidence or mark an open PR
as coverage already added.

## Verified records

### Onboarding skipped a visible step — issue #2033

- **Escaped symptom:** A cached first model advanced the first-run counter from
  step 2 directly to step 4, making it appear that the user missed a step in
  the released novice journey.
- **Root boundary:** The onboarding coordinator moved directly from idle to
  starting without representing the skipped-download transition.
- **Cheapest effective regression layer:** Swift unit tests for the coordinator
  phase transition; the failure is deterministic state logic and does not
  require a launched app to prove.
- **Evidence:** [dogfood report #2033](https://github.com/raullenchai/Rapid-MLX/issues/2033),
  [fix PR #2169](https://github.com/raullenchai/Rapid-MLX/pull/2169), and
  [onboarding completion tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/OnboardingCompletionBehaviorTests.swift).
- **Remaining gap:** The unit layer does not prove that the intermediate state
  remains visually readable at runtime; a native onboarding journey would own
  that presentation boundary.

### Onboarding showed noisy cached-model variants — issue #2033

- **Escaped symptom:** The first-model chooser exposed near-duplicate
  quantization variants as separate beginner choices in a released build.
- **Root boundary:** Cached inventory entries were rendered directly instead
  of first passing through a family/variant presentation policy.
- **Cheapest effective regression layer:** Swift unit tests for grouping and
  selection semantics, plus the existing onboarding golden flow for the
  accessible disclosure control.
- **Evidence:** [dogfood report #2033](https://github.com/raullenchai/Rapid-MLX/issues/2033),
  [fix PR #2196](https://github.com/raullenchai/Rapid-MLX/pull/2196),
  [cached-model policy tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/QuickstartCachedModelTests.swift),
  and the `cached-variant-collapse` journey in
  [the GUI harness](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/scripts/gui-golden-flows.sh).
- **Remaining gap:** The structural golden flow does not prove row spacing,
  clipping, or disclosure layout across window sizes.

### App launch showed an unsolicited low-memory sheet — PR #2053

- **Escaped symptom:** Resuming the last model on a memory-tight Mac presented
  a crash-risk confirmation sheet immediately on app launch, before the user
  requested a model load.
- **Root boundary:** Launch restoration and explicit user-start intent shared
  the same memory-warning side effect.
- **Cheapest effective regression layer:** Swift integration tests around
  `ServerManager.start`, because the contract is whether launch intent enqueues
  a warning while explicit intent still does.
- **Evidence:** [fix PR #2053](https://github.com/raullenchai/Rapid-MLX/pull/2053)
  and [launch auto-start memory tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/LaunchAutoStartMemoryTests.swift).
- **Remaining gap:** No native relaunch journey currently proves that the sheet
  stays absent with a deterministic low-memory fixture.

### Dictation's first use silently downloaded a model — PR #2188

- **Escaped symptom:** Dictation appeared ready before its weights were on disk;
  the first hotkey press then looked hung during a large, silent download.
- **Root boundary:** Readiness did not include local model availability, and
  server startup was allowed to become an implicit download path.
- **Cheapest effective regression layer:** Swift unit/contract tests for the
  `modelOnDisk` readiness decision, plus the `audio-readiness` golden journey
  for the explicit Download state and no-start-on-download behavior.
- **Evidence:** [fix PR #2188](https://github.com/raullenchai/Rapid-MLX/pull/2188),
  [dictation tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/DictationTests.swift),
  [audio catalog tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/AudioCatalogTests.swift),
  and the `audio-readiness` journey in
  [the GUI harness](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/scripts/gui-golden-flows.sh).
- **Remaining gap:** Cancellation, failure, retry, and relaunch across a real
  first download still need native journey and real-hardware coverage.

### Dictation hotkey armed before the model was ready — PR #2193

- **Escaped symptom:** The first shortcut press could begin recording while
  transcription weights were still loading, so transcription appeared to hang
  with no clear readiness boundary.
- **Root boundary:** Process-global hotkey registration was not gated by the
  current model's completed warmup identity.
- **Cheapest effective regression layer:** Table-driven Swift policy tests for
  every prerequisite and stale asynchronous completion; the `dictation` golden
  journey separately proves that Loading precedes Ready in the UI.
- **Evidence:** [fix PR #2193](https://github.com/raullenchai/Rapid-MLX/pull/2193),
  [policy extraction PR #2256](https://github.com/raullenchai/Rapid-MLX/pull/2256),
  [dictation policy tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/DictationEnablePolicyTests.swift),
  and the `dictation` journey in
  [the GUI harness](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/scripts/gui-golden-flows.sh).
- **Remaining gap:** A native journey has not yet exercised the real global
  shortcut before and after readiness, including failure and retry.

### Async Chat attachments crossed conversation ownership — PR #2265

- **Escaped symptom:** An attachment import started in conversation A could
  finish while conversation B was visible and appear in or be sent from the
  wrong composer.
- **Root boundary:** Pending attachments and asynchronous import completion
  shared one mutable composer bucket instead of being keyed by conversation.
- **Cheapest effective regression layer:** Swift unit tests for the
  conversation-owned draft store, import generations, immutable submission,
  and late completion rejection.
- **Evidence:** [fix PR #2265](https://github.com/raullenchai/Rapid-MLX/pull/2265)
  and [Chat attachment draft tests](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/Tests/RapidTests/ChatAttachmentDraftTests.swift).
- **Remaining gap:** Unit coverage does not prove native picker, paste, and
  drag/drop events preserve the same attachment identity through the composer
  and sidecar request.
