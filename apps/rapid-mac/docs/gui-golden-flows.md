# AX-first GUI golden flows

The AX suite remains the broad semantic regression net. A native XCUITest
target now complements it under `Tests/RapidUITests`: its first migrated
journey is `image-generation`, where XCTest captures each thumbnail itself,
crops away selection chrome, and proves the two rendered interiors differ.
That pixel assertion is intentionally impossible in an `AXUIElement` dump.
Run it with full Xcode after building the app:

```bash
./scripts/run-xcui-tests.sh
```

The target is additive; do not remove an AX journey until its semantic and
structural assertions have equivalent native coverage.

`scripts/gui-golden-flows.sh` runs the release journeys against a built
Rapid-MLX Desktop app without loading a real model.

**Journeys** — a user walking a path end to end:

1. fresh install, consent, onboarding, and steady-state shell; its
   pre-consent checkpoint waits without interacting and proves the fake
   sidecar receives no `models` or `ls` cache probe (#1560); its
   `cached-quickstart` companion proves a first-run user can select and start
   an existing chat model without a second download (#1793);
   `download-progress` drives a deliberately undersized download estimate and
   proves the first-run card falls back to truthful “bytes downloaded” copy
   instead of displaying more bytes downloaded than its total (#1550);
2. Settings mutation and persistence across an app relaunch;
3. basic chat, persisted conversation row, and restored transcript;
4. a deliberately slow stream and semantic **Stop generating** action;
5. model start, a one-shot sidecar crash, automatic respawn, and ready state.
6. a memory-constrained user can see and select an honestly labelled sub-1B
   fallback instead of being sent back to a chooser whose smallest visible
   model is the one that just failed the live-memory guard.
7. “Browse all models” lowers the onboarding sheet, opens Model Management in
   Settings, accepts a foreground interaction, and returns to the wizard with
   the user's original model selection intact. A final full-screen capture
   records the state a person actually sees.
**Invariants** — properties that must hold, not paths a user walks. These were
added after a release where every escaped defect landed on a surface no journey
covered, and each one names the defect it would have caught:

8. `update-state` — Settings → App must name the version the app actually is;
   a restored updater window on that same release must show one coherent
   up-to-date state (no install CTA or false missing-DMG warning).
9. `no-dead-controls` — every Settings panel must expose controls of its own.
10. `catalog-integrity` — a model that cannot chat must never be offered as one.
    Now covers image aliases too: `rapid-mlx models` tags them `[image:gen]`,
    `[image:edit]`, or `[image:both]` in their own section (mirroring
    `[video:gen]`), and the
    chat catalog's `hasNonChatKindTag` drops `image` alongside `audio`/`video`,
    so a 24 GB FLUX/Qwen-Image checkpoint can never surface in the chat picker.
    The same flow opens Model Management and pins its always-visible disk
    overview, including the largest app-managed model; read-only external
    runtime entries remain visible but are excluded from that calculation.

11. `image-generation` — a journey, not an invariant (listed here to keep the
    numbering stable): the Images tab turns a text prompt into a picture and
    lets the user iterate by re-prompting (see **Image generation** below). The
    instruction-edit path is available as a cancellable action and the same
    journey continues through generated-result editing and iterative editing.
12. `chat-document-attachment` — a vision-language model accepts a PNG through
    the Chat composer, renders it in the user turn, and sends typed
    `text` + `image_url` content; the same composer keeps its attachment
    control visible but disabled for a text-only alias and rejects paste/drop.
13. `window-close-prompt` — the first native main-window close reaches the
    Dock-visibility prompt, exposes both decisions plus “Don't ask again”, and
    choosing No completes a normal close. This pins the SwiftUI-to-NSWindow
    installation seam that #1590 found entirely disconnected.
14. `chat-restore` also exercises the formerly unmounted #1588 recovery and
    utility surface: the status-footer log toggle opens and closes the real
    drawer, and a restored assistant message opens the cross-paragraph
    “Select text” sheet.

The distinction matters. A journey answers *"can someone do this?"*; an
invariant answers *"is this still true everywhere?"*. The three defects below
were all invisible to journey-shaped tests:

| Flow | Would have caught | Why a journey missed it |
| --- | --- | --- |
| `update-state` | [#1612](https://github.com/raullenchai/Rapid-MLX/issues/1612) — the fallback update manifest sat at 0.11.0 for four releases | Nothing in a journey compares what the panel *says* to what the bundle *is* |
| `no-dead-controls` | [#1595](https://github.com/raullenchai/Rapid-MLX/pull/1595) dead recovery buttons, [#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) toggles that reported success without changing value, [#1605](https://github.com/raullenchai/Rapid-MLX/issues/1605) a tray item that reported nowhere | A journey visits the controls it needs; these were the ones nobody scripted |
| `catalog-integrity` | [#1603](https://github.com/raullenchai/Rapid-MLX/issues/1603) — eight video-generation models offered as chat models, dead-ending *after* a download of up to 64 GB | The picker renders them perfectly; the bug is that they are there at all |

### Full flow roster

The numbered narrative above is selective. The authoritative, complete set of
flows the harness can run is the dispatch table in `scripts/gui-golden-flows.sh`
(`case "$FLOW" in …`). As of this checkout that is 26 flows: `fresh-install`,
`cached-quickstart`, `cached-curated-tradeup`, `download-progress`,
`settings-persistence`, `settings-mtp`, `chat-restore`, `message-actions`,
`restored-tools`, `tool-loop-budget`, `chat-depth`, `math-rendering`,
`slow-stream-stop`, `model-crash-recovery`, `low-memory-choice`, `update-state`,
`window-close-prompt`, `no-dead-controls`, `catalog-integrity`,
`browse-all-destination`, `chat-document-attachment`, `image-generation`,
`dictation`, `audio-readiness`, `resident-load-rejected`, `launch-integrations`.
`--flow all` runs them in that order.

### Current baseline

**2026-08-09** — the whole suite (`gui-golden-flows.sh`, no `--flow`) passes on
a build of this checkout, twice in a row, the second run against the committed
baselines rather than the run that wrote them.

Getting there meant refreshing every structural baseline, and the reason is
worth recording: they were last updated **2026-08-07** (#1666), while
`Sidebar.Images` landed in #1705 and the attachment control (then photo-only)
landed in #1723, both on **2026-08-09**. The suite had therefore been red on
`main` through two merges
and nobody knew, because at that time it ran by hand and was wired into no
workflow. It is no longer only run by hand: a `gui-golden-flows` job in
`.github/workflows/rapid-mac-ci.yml` now builds the real `.app` and runs each
flow as its own step against `scripts/fake-rapid-mlx.sh` (every flow except
`chat-depth`, which is explicitly excluded there — its full-transcript
assertion is invalid on the runner's small window). Every line in the
refresh diff is one of exactly three things:

| Added | Source |
| --- | --- |
| `Sidebar.Images` (×12) | #1705 |
| `ChatView.AddAttachments` (×17) | document attachments |
| `Settings.ModelManagement.CapabilityTabs` + its Chat/Image segments (×4) | this change — the fake now emits an `[image:gen]` alias, so the capability tabs have a second capability to show |

One line changed rather than appeared: the sidebar conversation menu gained
`title="More"`, an AXTitle AppKit synthesises for a borderless `Menu` (our code
sets only `accessibilityLabel`). It held across the verification run; if it
ever flickers it is noise the normalizer should scrub, not a UI change.

The per-flow results below were recorded on 2026-08-07, on a build of that
checkout:

| Flow | Result |
| --- | --- |
| `update-state` | **PASS** — the panel's version matches `CFBundleShortVersionString` (0.12.6 when recorded; the assertion compares the two, it does not pin a literal) |
| `catalog-integrity` | **PASS** — `fake-video-alias` reaches neither the chat surface nor Model Management |
| `no-dead-controls` | **PASS** — all six Settings panels expose controls of their own; see the red → green note below |

The first two were measured against `main`. `no-dead-controls` was red on `main`
and is green as of the identifier work described below; the run recorded here is
the one that made it green.

#### `no-dead-controls`: red → green

This flow shipped red, on purpose, and has since been driven green by fixing
the product rather than the assertion. Worth recording, because a gate that has
never moved is a gate nobody has evidence about.

It first failed on **`tools`**: Settings → Tools rendered three tool toggles, a
backend radio group and a browsing toggle, and **none of them carried an
identifier**. The controls worked — real `AXCheckBox`/`AXRadioButton` with
correct values — they were simply unaddressable. Naming them took that panel
`0 → 8`, and the flow then failed one panel further along, on **`privacy`**,
which had the same gap: a telemetry toggle and three policy `Link`s, all
unnamed. Naming those took `privacy` `0 → 4`. Final measured run:

```
[gui-golden]   models: 2 identified controls
[gui-golden]   modelManagement: 16 identified controls
[gui-golden]   tools: 8 identified controls
[gui-golden]   appearance: 3 identified controls
[gui-golden]   privacy: 4 identified controls
[gui-golden]   app: 5 identified controls
[gui-golden] PASS — no-dead-controls
```

`app` was never bare — the loop simply died at `privacy` before reaching it.
Confirmed rather than assumed: `Settings.App.{UpToDate,RecheckCTA,
ExportDiagnostics,HideDockOnCloseToggle,ResetDockOnboardingCTA}`.

#### What this flow does NOT prove

It counts identifiers; it does not press them. A panel can be fully addressable
and still contain a control that does nothing observable, so green here means
"reachable", not "works".

That was not hypothetical, and the worked example is worth keeping now that it
has been fixed — because the flow stayed green through the whole of it.

`Settings.Privacy.TelemetryToggle` is addressable and is a real `AXCheckBox`.
`AXPress` on it flipped the stored preference
(`com.rapidmlx.rapid.telemetry.enabled` `0 → 1`, a client ID minted, the shared
`~/.rapid-mlx/telemetry-client-id` written) — and the switch did not re-render,
so its AX value stayed `0` until you left the panel and came back.
`SettingsView.telemetryEnabledBinding`'s getter read `TelemetryConfig.isEnabled`,
a plain `static var` over `UserDefaults.standard`, which gives SwiftUI no
dependency to invalidate on. To a user: a consent switch that appears to refuse
their choice while they are in fact opted in. Same family as
[#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) in the table above.

Fixed in [#1623](https://github.com/raullenchai/Rapid-MLX/issues/1623), measured
both ways on real builds:

| build | AX value after press | preference |
| --- | --- | --- |
| before | `0 → 0` (stuck) | `0 → 1` |
| after | `0 → 1` | `0 → 1` |

**`no-dead-controls` was green for every one of those runs.** It counts
identifiers; it does not press them. The gap it leaves is exactly this: a
control that is reachable and inert. A successor that presses each control and
asserts the observable value moved is what closes it — until then, read a green
here as "reachable", never as "works".

Two notes on writing assertions here, both learned the hard way while adding
these:

- The first version of `no-dead-controls` counted every `Settings.*` identifier
  on the panel. The six `Settings.Category.*` buttons appear on *every* panel,
  so the count was never below six and the flow went green on a completely
  unlabelled Tools panel. Count the panel's **own** controls.
- `catalog-integrity` proves it discriminates rather than trivially passing:
  the non-video `fake-alias` appears 9 times in the same tree where
  `fake-video-alias` appears 0 times. A test that asserts an absence must show
  that the corresponding presence is detectable.

Every journey gets a unique bundle identifier and throwaway `HOME` through
`dogfood-isolate.sh`. The fake sidecar emits deterministic SSE and JSONL
lifecycle evidence, so the suite does not download a model or put meaningful
pressure on unified memory.

`chat-depth` also streams CSS and Makefile fences with punctuation-bearing
configured tokens (`background-color`, `@font-face`, `.PHONY`, `filter-out`),
split across SSE chunks. The AX contract proves the source survives intact in
separate rendered code blocks; `SyntaxHighlighterTests` owns the colour-run
assertion because foreground colours are not exposed by `AXUIElement`.
Its comparison response must also expose the native macOS SwiftUI `Table`
shape (`AXOutline` with `AXRow`, `AXCell`, titled `AXColumn` children), not
merely six sibling text nodes, so VoiceOver retains table navigation and
header association (#1689).

### Low-memory recovery

The normal model picker intentionally hides sub-1B models: they fall below the
default quality and tool-use floor, and presenting them beside normal choices
without context makes a faster but worse answer look like a product failure.
That policy cannot govern a recovery path. If the live memory guard says the
starter is unsafe and tells the user to “pick a smaller model,” onboarding must
actually contain one.

`low-memory-choice` pins the visible half of that contract through AX:

1. open fresh onboarding and advance to **Choose your first model**;
2. find `Quickstart.Choice.qwen3-0.6b-4bit` under **LOWEST MEMORY**;
3. assert that the card says **less accurate** and **not recommended for
   tools**, so lower memory is not presented as equivalent quality;
4. select it through AX and retain the before/after trees as evidence.

The warning-to-switch half is deterministic Swift coverage rather than a host-
RAM-dependent GUI trick. `QuickstartView.lowMemoryRecoveryChoice(for:)` replays
the original live-memory snapshot against the fallback footprint and exposes
`Quickstart.Memory.SwitchToLowMemory` only when the replacement falls below the
85% danger line. Under heavier pressure the button is absent, avoiding a false
promise or a warning loop; **Cancel** still returns to the chooser where the
low-memory category remains visible.

### Chat attachments

`chat-document-attachment` covers image input inside the normal Chat tab. This is
separate from `image-generation`: the former asks a VLM to understand an image;
the latter asks a diffusion model to create one.

The deterministic lane should use a fake VLM alias and a small fixture PNG, and
walk both halves of the capability boundary:

1. select the fake VLM alias and assert `ChatView.AddAttachments` is present and
   enabled;
2. add the fixture through the standard open panel, then assert the thumbnail's
   `ChatView.Attachment.Remove.<filename>` control is present;
3. enter a caption question, send, and assert the user bubble contains the
   attachment while the fake sidecar records an `image_url` data URI alongside
   the typed text part;
4. retry or regenerate the turn and assert the replay request still contains
   the attachment;
5. switch to a text-only alias and assert `ChatView.AddAttachments` remains
   visible and enabled for PDF/CSV/TXT input;
6. attempt image paste and drop, assert no thumbnail appears and no image bytes
   reach the fake sidecar. Historical image turns must also be reduced to text
   before a text-only request is encoded.

The standard file picker itself remains outside the structural AX baseline, as
with the Images-tab save panel. The flow drives it only to supply the fixture;
the product assertions begin at the composer thumbnail and end at the recorded
wire request.

Real-weight dogfood on 2026-08-09 used the locally built
`rapid_mlx-0.12.7` wheel with its `[vision]` extra and
`gemma-4-e2b-4bit`. The GUI accepted `cheetah-logo-96.png`, displayed it in
the user bubble, and the model described the spotted feline in the fixture at
5.8 tok/s. The paired `qwen3.5-4b-4bit` text-only run kept the add button
visible and disabled and accepted no pasted attachment. A base-wheel-only run
is not valid evidence for this flow: without `[vision]` / `mlx-vlm`, the engine
intentionally rejects or text-degrades VLM serving.

## Image generation

The Images tab is a dedicated text→image / image-edit surface, reached from
`Sidebar.Images`. It is decoupled from chat on purpose: rapid-mlx serves **one
model per process**, so an image-gen alias (e.g. `flux2-klein-4b`) cannot be
loaded alongside the chat LLM — selecting one reloads the sidecar, exactly the
stop/start path a chat model-switch already takes.

**The interactive golden flow is text→image generation.** The tab imitates the
fast half of ChatGPT's image experience: type a prompt, get a picture in
seconds, refine by re-prompting. `image-generation` walks that through AX
identifiers, no real diffusion weights required (the fake sidecar answers
`/v1/images/*` with a 1×1 PNG). Demonstrated live with real weights below.

1. open the Images tab via `Sidebar.Images`; assert the `Images.EmptyState`
   hero (the cheetah mark + "Draw anything") is present;
2. the composer's `Images.ModelPicker` lists image models whose CLI capability
   includes generation (`[image:gen]` or `[image:both]`) from
   `rapid-mlx models`, never a chat alias (see `catalog-integrity`); set the
   aspect ratio with `Images.Aspect` (1:1 / 3:4 / 4:3);
3. **Load the model.** rapid-mlx serves one model per process, so when the
   server is on a different (e.g. chat) model the tab shows a readiness banner
   ("<model> isn't running"); press `Readiness.Action` to switch the server to
   the image model. `Images.Generate` stays disabled until it is ready — the
   same `ModelReadiness` gate chat uses;
4. **Generate.** Type into `Images.Prompt`, press `Images.Generate`; assert the
   in-flight progress card appears (a true `step / total` bar, elapsed, ETA, and
   an `Images.Cancel` control), then a result appears in `Images.Stage` with a
   thumbnail under `Images.Gallery` (and `Images.EmptyState` is gone);
5. **Refine by re-prompting.** Adjust the prompt and press `Images.Generate`
   again; each render lands as its own thumbnail in the `Images.Gallery`
   filmstrip, clickable to revisit its prompt;
6. `Images.Result.Save` (a hover control on the focal image) opens the standard
   save panel — not asserted through the modal `NSSavePanel`, out of AX scope
   like every other file-picker in the app.

### Instruction edit

The Images tab exposes `/v1/images/edits` in two places: import a PNG/JPEG from
the composer, or press the pencil action on a generated result. Entering edit
mode preserves the source image, filters the model picker to edit-capable
checkpoints (`[image:edit]` or `[image:both]`), and changes the prompt to an
edit instruction.
The result remains in edit mode so multiple changes can be applied in sequence;
Exit returns to the previous text-to-image model. The normal progress and Cancel
controls remain available for the full request.

The built-in `flux2-klein-4b` alias is shared by generation and editing, so a
user who already downloaded it needs no second checkpoint.

The `image-generation` golden journey enters editing from a generated result,
submits an instruction through the multipart edit endpoint, verifies the source
image reached the wire, verifies the returned image becomes the next edit
source, then exits back to generation mode. It then drives the second entry —
`Images.Edit.Import` — all the way through a real file import to a regenerate:
the journey presses `Images.Edit.Import` and drives the app through a
deterministic test seam: the harness's launcher sets `RAPID_GUI_GOLDEN_MODE=1`
plus `RAPID_SIMULATED_IMPORT_PATH` (a golden-harness-only switch, like
`RAPID_BIN`/`RAPID_GUI_WEB_SEARCH_FIXTURE`), so the button imports exactly that
fixture through the same post-pick path a real picker would. The explicit
golden-mode gate means a real user's launch — which never sets it — always opens
`NSOpenPanel` even if an unrelated process leaked an import path into the
environment. This is deliberate: the native `NSOpenPanel`'s file browser publishes no
accessibility identifiers, so neither AX actions nor injected keyboard events can
drive its "Go to Folder" sheet on an unattended CI runner. The journey then
asserts the app entered edit mode keyed to the imported file's name, submits an
instruction, and verifies the fixture's decoded pixel payload reached the wire as
a multipart edit (compared by RGBA hash rather than raw bytes, because the app
legitimately re-encodes imports). Pressing the button itself, entering edit mode
keyed to the file name, and the fixture's bytes on the wire are all still exercised
end-to-end; what the seam cannot cover is Apple's own OS file-browser dialog. As
with `Images.Result.Save`'s save dialog, the native picker UI itself is out of AX
scope — the "import an image → edit it" contract it feeds is proven through the
app-level path above, which no tree dump can witness on its own.

### Model realities the UX has to design around

Verified on an M2 Pro 32 GB with the 4-bit mflux checkpoints:

* **FLUX.2 Klein uses four steps for both operations.** The same 4-bit,
  approximately 4.3 GiB checkpoint backs text-to-image and image-conditioned
  editing. The server swaps between mflux's generation and edit variants so
  only one copy remains resident in unified memory.
* **Edit output uses the backend default canvas.** The OpenAI-compatible edit
  request omits `size`; `Flux2KleinEdit` therefore uses its 1024×1024 default.
  Sequential rounds continue from the newest result.
* **Editing is image-conditioned generation.** It handles object changes,
  background replacement, styles, and multiple references, but a compact 4B
  checkpoint may preserve exact text or identity less reliably than a much
  larger dedicated edit model.

The **model-vs-endpoint contract** is enforced server-side and covered by
hermetic tests rather than a live flow: generation-only and edit-only models
still reject the wrong endpoint, while `flux2-klein-4b` accepts both. The
Images tab selects the endpoint from the explicit `[image:gen]`,
`[image:edit]`, or `[image:both]` capability carried by the CLI catalog.

### What the journey does and does not drive

`gui-golden-flows.sh --flow image-generation` is runnable, and its structural
baseline is committed as `image-generation.generated.txt`.

Two things the scripted journey deliberately does **not** do, so the list above
is not read as a coverage claim:

* **Aspect ratio** — asserted present (`Images.Aspect`), not exercised. The
  individual options inside the menu carry no identifiers of their own, so
  driving them would mean clicking coordinates, which is the one thing this
  harness exists to avoid.
* **`Images.Result.Save`** — out of AX scope like every other file picker in
  the app; the flow stops at the focal image.

Writing it surfaced two defects that no tree dump could show, both now fixed:

| Defect | Why nothing caught it |
| --- | --- |
| The prompt editor announced itself as `rapid.chat.compose` — ``ComposeField``'s default, shared with chat | `Images.Prompt` sits on the SwiftUI wrapper and resolves to the placeholder `AXStaticText`. `set-value` on it returns `{"success":true}` and changes nothing: the binding never updates, `Images.Generate` stays disabled, and the press is silently dropped. A green type step, then a render that never happened. |
| Filmstrip thumbnails lacked a stable, baseline-compatible identifier | Their whole label is an image, so they reached VoiceOver — and the flow — as unnamed buttons. (#1725 later gave them a per-render `Images.Thumb.<uuid>`, but a UUID changes every run and cannot anchor a repeatable baseline; the positional `Images.Gallery.Thumb.<n>` can.) "A second render produced a second thumbnail" was unassertable except by counting anonymous buttons. |

The editor now carries `rapid.images.compose` (pinned by
`ChatComposeAccessibilityTests`) and each thumb carries
`Images.Gallery.Thumb.<n>`, newest first.

`type_prompt` in the flow is the guard against the first defect returning: it
requires the composer to hold the text **and** `Images.Generate` to be enabled
before anything is pressed, because either signal alone can lie.

The fake sidecar answers `/v1/images/generations` with a real 1×1 PNG after a
scripted number of steps (`FAKE_IMAGE_STEPS`, `FAKE_IMAGE_STEP_MS`), so the
in-flight card is observable rather than a frame between two polls, and the
bytes differ per render, and the flow compares their SHA-256s — which proves
the SIDECAR returned two different images, not that the app drew two different
ones. AX exposes no bitmap, so that last link is out of reach here; see #1719.
`tests/test_fake_sidecar_image_catalog.py` pins the fixture's catalog row
against the shape `ModelCatalog.parseImageRows` indexes — the two files never
import each other, and the drift shows up on a Mac disguised as a product bug.

## Run

Build the current checkout, then run all flows:

```bash
cd apps/rapid-mac
SKIP_SIDECAR=1 BUNDLE_MODEL=0 ./scripts/build.sh
./scripts/gui-golden-flows.sh
```

Run one journey or retain its isolated persona for diagnosis:

```bash
./scripts/gui-golden-flows.sh --flow slow-stream-stop
./scripts/gui-golden-flows.sh --flow low-memory-choice
./scripts/gui-golden-flows.sh --flow chat-restore --keep
./scripts/gui-golden-flows.sh --flow math-rendering
./scripts/gui-golden-flows.sh --flow browse-all-destination
./scripts/gui-golden-flows.sh --flow no-dead-controls
```

The suite needs an active **GUI login session**. A command launched directly by
`sshd` or tmux does not inherit Terminal's Screen Recording and Accessibility
grants. Remote runs are supported by asking the logged-in Terminal app to run
the command (for example with `osascript ... do script ...`); the test process
then has the same TCC identity as that Terminal session.

The screen must also stay awake: when the session goes idle,
`CGSSessionScreenIsLocked`
flips to `Yes`, every app reports zero windows through AX, and `screencapture`
returns wallpaper. That looks exactly like a broken app. Hold the session with
`caffeinate -dimsu -t <seconds>` for the length of the run — `-u` is the
load-bearing flag, since plain `-d` stops display sleep but not the idle path —
and re-read the lock state before trusting any window assertion.

Set `RAPID_GUI_SOURCE_APP` to test a release candidate bundle and
`RAPID_GUI_GOLDEN_OUT` to choose the artifact directory. Each run records AX
trees, actions, fake-sidecar events, logs, and a top-level `result.json`.

## AX structural baselines

Settled states across the journeys are also fingerprinted as
**structural baselines**, committed under
`Tests/GUIGoldenFlows/__Snapshots__/<flow>.<state>.txt` — currently 32 baseline
files spanning 12 baseline names (`audio-readiness`, `chat-depth`,
`chat-restore`, `fresh-install`, `image-generation`, `launch-integrations`,
`model-crash-recovery`, `onboarding-direction-d`, `settings-mtp`,
`settings-persistence`, `slow-stream-stop`, `update-state`). All but one are
runnable flows from the dispatch roster above; `onboarding-direction-d` is a
baseline-only specimen — it has no `case` entry in `gui-golden-flows.sh` (its
snapshots are emitted by `baseline` calls, not a `--flow onboarding-direction-d`
journey). `scripts/ax-baseline.py`
normalises a raw AX dump into an indented tree and the suite fails on any
difference, so a PR that removes a button, reparents a control, renames an
identifier, drops an icon or flips an enabled state produces a reviewable diff
instead of passing silently.

**This is the cheap layer of appearance testing and it is structural only.** It
cannot see colour, spacing, typography or anything else that never reaches the
accessibility layer; the PNG snapshots in `Tests/RapidTests/__Snapshots__` stay
the pixel-level check.

The normaliser keeps hierarchy, role, subrole, `accessibilityIdentifier`,
`AXTitle`/`AXDescription`/`AXHelp`, enabled state, sibling order below the
window level, and the *kind* of each value (`bool:true`, `bool:false`,
`number`, `text`, `empty`). It drops or rewrites everything that is legitimately
volatile: screen coordinates and sizes, pids, top-level window z-order, value
contents, and version numbers, byte sizes, token rates, durations, dates, clock
times, UUIDs, `/Users/<name>` paths and the fake model alias wherever they
appear in text. `Settings.App.UpToDate` carries the release version and a
conversation row identifier carries a fresh UUID — recording those verbatim
would make the baselines flap every release and every run.

Two further things are dropped because they flap without any product change,
both found by comparing real recorded baselines rather than by reasoning:

- **Everything below a window-control button.** The traffic lights are AppKit's,
  and their anonymous `AXGroup` descendants are realized lazily: two dumps taken
  seconds apart in the *same* run recorded one group under `AXZoomButton` in
  `settings-root` and two in `models-idle`. The buttons themselves stay, so a
  missing close box is still a diff; their private innards do not.
- **Relative day headings.** A transcript is filed under `Today` — until a run
  straddles local midnight, at which point the identical UI says `Yesterday`
  and every baseline holding one goes red at 00:00 for no reason.

An intended UI change is a deliberate commit:

```bash
./scripts/gui-golden-flows.sh --update-baselines
git diff apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__
```

Recording is **only** ever done by `--update-baselines`. A missing baseline is a
failure, not a free pass: recording on absence would mean a typo'd snapshot name,
or one somebody forgot to `git add`, sails through CI green while comparing
against nothing. (This deliberately diverges from the PNG convention in
`Tests/RapidTests/SnapshotHelpers.swift`.)

Inspect a single normalised tree without running a journey:

```bash
python3 scripts/ax-baseline.py normalize --scrub fake-alias /tmp/…/steady.json
```

## Why this is not coordinate automation

The checked-in `rapid-ax.swift` helper talks directly to macOS Accessibility.
It finds controls by stable `AXIdentifier`, performs `AXPress`, sets native text
values, and serializes roles/descriptions/values for assertions. Peekaboo is
kept for permission checks, window discovery, menu interaction, and screenshots.
Coordinates are used in exactly two places, both deliberate and both derived
from AX bounds. One is the first-run SwiftUI consent-sheet fallback, for older
accessibility stacks. The other is `browse-all-destination`, where a coordinate
click is the *point*: `AXPress` reaches a window trapped behind a modal sheet
just as well as a usable one, and so does Peekaboo's default background click,
so proving that a person could use the Settings window the wizard opened takes
a real `--foreground` mouse event at a real position. Bounds are re-read after
focusing, since focusing can raise or move the window.

This makes normal actions independent of window position, resolution, theme,
and most layout changes. It also avoids Peekaboo snapshot publication failures
seen while the ready-state UI is updating. The app does not need to own the
foreground for ordinary AX actions; menu operations and screenshots may briefly
activate it, so release CI should still use a dedicated logged-in macOS session.

## Adding a flow

Prefer a stable `.accessibilityIdentifier(...)` in product code, then assert
observable user state rather than sleeps or pixels. Keep model behavior behind
the fake sidecar unless the purpose of the flow is real inference quality. A
real-model dogfood pass remains a separate, explicitly memory-budgeted release
stage.

Fingerprint only *settled* states. Baselines taken mid-transition flap: the
crash-recovery tree captured while the sidecar was still restarting contained a
transient "Starting …" banner in one run and not the next. `wait_send_idle`
exists for this — `ChatView.SendOrStopButton` publishes `AXHelp` only while the
readiness gate is closed, so the absence of that attribute is a
copy-independent "ready and not streaming" signal. If a new state turns out to
be irreducibly unstable, exclude it and say why rather than loosening the
comparison.

**Never assert that text appears *somewhere* in the tree when a specific place
is what you mean.** `chat-restore` failed roughly one run in two for a reason
worth repeating. `start_model` gated on `SendOrStopButton.description ==
"Send message"`, which is the button's label for the whole startup — its hint
still read "… is still starting." So the flow pressed Send into a closed
readiness gate, the press was dropped, and the draft stayed in the composer.
`assert_tree_text "golden restore marker"` then *found* the prompt — in the
composer — and reported success for a message that was never sent. The run only
failed later, on the reply that never came, which is why it looked like a
flake rather than a broken assertion.

Both halves are now fixed: `start_model` waits on `wait_send_idle`, and
`send_prompt` requires the composer to actually drain. The general rule: an
assertion that a string is present anywhere is satisfied by the input field,
the placeholder, the tooltip and the sidebar. Say *which element*.
## The identifier gate

Because every flow above finds its target by `AXIdentifier`, this suite's
ceiling is exactly the set of controls that carry one — and that ceiling drops
silently every time a feature ships an unlabelled control, because the app still
works by hand. `scripts/check_rapid_mac_ax_identifiers.py` (wired into the
`accessibility-identifiers` job in `.github/workflows/rapid-mac-ci.yml`) fails a
PR that **adds** an interactive control under `apps/rapid-mac/Sources/` with no
`.accessibilityIdentifier(...)`.

It is scoped to lines the diff added. The pre-existing backlog is deliberately
out of scope — a gate that failed on it would be un-landable, and a disabled
gate is worse than none. `--audit` lists that backlog when you want to chip at
it:

```bash
python scripts/check_rapid_mac_ax_identifiers.py --audit
python scripts/check_rapid_mac_ax_identifiers.py --base-ref origin/main   # what CI runs
```

Name new identifiers with the existing `<Surface>.<Thing>` convention (the
inventory lives in `docs/userflows.md`), and put them on the control itself —
an identifier on the enclosing `HStack` does not give `AXPress` anything to
press.

### Opting out

There is currently **no** known control on this surface that cannot carry an
identifier. `confirmationDialog` / `alert` buttons were the standing suspicion —
`docs/userflows.md` carried "Approval dialogs lack identifiers" as an open item
for several releases — and the suspicion was measured rather than inherited: the
presented dialog is an `AXSheet` whose `AXButton` children *do* carry the
identifiers declared at the call site. So the escape hatch exists for a case
nobody has produced yet, and `rg ax-exempt apps/rapid-mac` returning nothing is
the expected state. If you find a real one, opt out explicitly, with a written
reason, on the control's line or the line directly above it:

```swift
// ax-exempt: <what you measured that shows the identifier cannot be reached>
Button("Allow once") { approve() }
```

The reason is mandatory — a bare `// ax-exempt:` fails the gate just like a
missing identifier — and every opt-out is greppable, so the true manual-only
surface stays countable instead of invisible:

```bash
rg ax-exempt apps/rapid-mac
```
