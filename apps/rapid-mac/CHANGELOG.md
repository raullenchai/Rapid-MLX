# Changelog

All notable user-facing changes ship here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely
([Semantic Versioning](https://semver.org/spec/v2.0.0.html) for
version numbers), with one section per release.

The motivation for hand-curating this file rather than relying on
GitHub auto-generated notes: a non-technical user reading "Tier 1
#I.2 pop-out conversation" has no idea what shipped. This file
translates each commit subject into something a release-day reader
can actually understand.

> **Note:** pre-0.9 entries link their issue/PR and compare/release URLs to
> the app's archived `machinefi/rapid-desktop` repository, where that work
> shipped before the app moved into this `raullenchai/Rapid-MLX` monorepo.

## [Unreleased]

## [0.12.17] — 2026-08-19

Dictation actually works now — 0.12.16's release build shipped without the
microphone permission, so the "Allow…" button did nothing. Update, and macOS
will ask for the microphone like it should.

### Fixed

- **Speech to Text can be enabled again.** The signed app now carries the
  microphone entitlement macOS requires. 0.12.16's release build shipped
  without it, so the permission prompt never appeared and dictation
  dead-ended at setup — development builds were unaffected, which is why
  it slipped through. The build pipeline now refuses to sign a release
  that is missing it.

## [0.12.16] — 2026-08-19

Dictate into any app from a global hotkey, web search works out of the box,
chat typesets math, and conversations can be filed into folders and
exported.

### Added

- **Dictate into any app.** Tap right Option, speak, and the words land at
  your cursor — in any app, even with Rapid's window closed. Transcription
  runs through your local server, so audio never leaves the Mac.
- **Math renders as math.** Chat typesets inline and block LaTeX the way
  models actually write it, so equations show up as equations instead of
  raw markup.
- **Web search that just works.** Chat's web search defaults to Keenable's
  keyless service — real query-relevant snippets with no account and no API
  key, replacing the DuckDuckGo scrape that rate-limited after a few
  searches. Want the best measured quality? Paste a free Parallel key in
  Settings → Tools (it tops the Artificial Analysis Search Index; the free
  tier covers about 1,000 searches a month). Brave's entry now says plainly
  that its API requires a card on file and auto-bills overage.
- **File conversations into folders, and export them.** Create a folder from
  any conversation's row menu, rename it, and export conversations as
  Markdown from the same place.

### Changed

- **Model recommendations follow the Artificial Analysis Intelligence
  Index.** Every Mac from 32 GB up is offered Qwen3.8-27B — GPT-5.6-class
  intelligence at ~40 tok/s with multi-token prediction on — replacing
  larger models that score far lower on the same index. Smaller Macs keep
  their best-in-class picks.
- Getting a model is now two explicit steps — Download, then Start when
  you're ready — and the launch memory modal is gone.
- Setup fills the window instead of floating as a small card over a dimmed
  chat, so every step is reachable on every display.
- Opt-in telemetry now includes the chip family and a rounded memory tier —
  never raw bytes — and the privacy policy spells that out.

### Fixed

- The main window can no longer shrink below a usable size.
- "Jump to latest" works after an answer finishes.
- The same file can no longer be attached twice.
- Code blocks no longer flicker between plain and code styling while an
  answer streams.
- Markdown code blocks and tables are visible again in chat.
- Voice notes in Apple formats (M4A, CAF, …) transcribe directly — the app
  transcodes them for the server on the fly.

## [0.12.15] — 2026-08-18

Fixes a bug that could silently truncate files your coding agent writes, adds a
Developer section to Settings, and brings MTP acceleration to the Qwen 3.8
models that ship with it.

### Fixed

- **Coding agents no longer get truncated file writes.** When an agent asked a
  Qwen 3.5 or 3.6 model to write a file, the file could be cut off at the first
  line break — a 700-byte file arriving as 11 bytes — with no error shown
  anywhere. Short values could come back subtly misspelled (`Tokyo` as `Toyo`).
  This affected versions 0.12.5 through 0.12.14; if you ran a coding agent on
  those, it is worth re-checking files it wrote.
- Downloading a model no longer crawls when our mirror slows down. Each file
  now watches its own speed and switches to Hugging Face if the transfer
  collapses, instead of waiting it out — one 335 MB model took 8 minutes before.
- Voice transcription no longer invents words when a clip is short or silent.
  A brief silent recording now returns nothing instead of a made-up sentence.
- Images and other visual input work correctly again on hybrid vision models,
  which could previously answer from the wrong cached state.
- Media-only models (image, audio, video) no longer appear as things you can
  launch for chat.

### Added

- **Settings → Developer**, in development builds only: rehearse the first-run
  experience without touching your real setup. You choose what gets erased —
  conversations, settings, and the telemetry decision are each opt-in, and the
  confirmation names every item before anything is removed.
- **MTP acceleration for Qwen 3.8**, opt-in from Settings → Performance. It is
  off by default because the speedup varies by machine. Models that ship an MTP
  head now carry everything needed to run it, so turning it on just works; a
  model without one refuses to start rather than quietly running unaccelerated.
- DeepSeek Harness joins Claude Code, Codex, Hermes and Aider as a fully
  supported coding agent — including in the gate that every release must pass.

### Security

- Model files downloaded from the internet are treated as untrusted by default:
  code shipped inside a model repository no longer runs implicitly, downloaded
  Python components are checksum-verified, and the audio checkpoint loader
  refuses formats that can execute code unless you opt in.

### Changed

- The status footer drops readouts it cannot fit instead of squeezing six
  indicators into the width of four.
- Installing on Apple Silicon now insists on a matching Python and repairs a
  half-built environment, instead of leaving an installation that never worked.

## [0.12.14] — 2026-08-15

Adds a Rapid-built version of Qwen3.8-27B, lets you give models your own names,
and moves app updates fully onto the signed background updater.

### Added

- **Qwen3.8-27B, Rapid mixed-precision build** — a version of Qwen3.8-27B we
  quantized ourselves, at 13 GB of weights instead of the standard build's
  15 GB. It answered every coding task and 25 of 30 tool-calling scenarios in
  our release testing. Plan for a 48 GB Mac or larger; on 32 GB it is a tight
  fit. Maths is its weak spot — for arithmetic-heavy work Qwen 3.6 35B or
  Gemma 4 26B remain the safer picks.
- Qwen 3.8 models now show their real family name in the model list, and their
  verified tool support shows as a capability instead of "unknown".
- Give any model a name of your own from the command line —
  `rapid-mlx alias set fast qwen3.5-4b-4bit`, then `rapid-mlx serve fast`.
  Names can point at a built-in model or any Hugging Face repository.
- DeepSeek Harness joins the list of coding agents Rapid can set up for you.

### Changed

- App updates now run entirely through the signed background updater. The old
  in-app installer is gone, which removes the case where an update could
  download but fail to install. Versions 0.12.12 and newer update themselves;
  older versions are pointed at the download page once.

### Fixed

- The model information shown for hybrid models is now read live instead of
  from a stale snapshot.
- Closing a chat mid-answer no longer records a false error in the logs.

## [0.12.13] — 2026-08-14

Adds Qwen's newest 27B model and document attachments in chat, plus a round of
Mac app reliability and polish fixes.

### Added

- **Qwen3.8-27B** — Qwen's latest 27B model is now one click/command away
  (`rapid-mlx chat qwen3.8-27b-4bit`). It's a strong general-purpose and
  tool-using model, and it can also describe images when you start it with the
  vision option turned on.
- Normal chats now accept PDF, CSV, and TXT attachments. Rapid extracts document
  text locally, keeps it with the conversation for follow-up questions, and
  clearly marks large partial extracts. Scanned PDFs without selectable text
  report that OCR is required instead of sending an empty prompt.

### Fixed

- The model list no longer shows a phantom "No" model on a machine that has
  nothing downloaded yet.
- The Launch page's Codex and Hermes buttons, which did nothing when clicked,
  now start the right thing.
- Hovering a button no longer covers its label with a grey block.
- Long answers no longer slow the app down as their links pile up.
- The message-box placeholder text now stays out of the way while you type in
  another language.
- Audio models are downloaded and checked before the app tries to use them, so
  they no longer fail at the moment you press play.

## [0.12.12] — 2026-08-13

A vision and reliability release. Images get their first response noticeably
sooner, a class of image chats that used to come back empty now answer
correctly, and the app can update itself in the background.

### Added

- Signed background updates: the app now checks for, downloads, and installs
  new versions on its own, so you stay current without visiting a download
  page.
- Support for a new small model, Ling 3.0 tiny.

### Changed

- Sending an image now gets its first words back much faster — on the Gemma
  vision models the wait before the reply starts is now on par with, and
  often ahead of, every other way to run them on a Mac.

### Fixed

- Some image chats (the Gemma 3 family) could reply with nothing at all;
  they now answer normally.
- Starting a model no longer stalls when the network is slow or a lookup
  hangs — the app falls back to what it already has on disk.
- Image and video models no longer pile up in memory; only one stays
  resident at a time.


## [0.12.11] — 2026-08-12

A speed release. The engine underneath the app got a full tuning pass
measured against every other way to run these models on a Mac, and it is
now the fastest of them at answering several conversations at once —
with single-conversation typing speed matched to the best of them too.

### Changed

- Several chats, agents, or tool calls running at the same time now share
  the Mac far better. On a large mixture-of-experts model with eight
  conversations in flight, total throughput went up about 40%; every
  model size we measure now runs concurrent work faster than the
  reference implementation rather than behind it.
- A single conversation types out faster as well, including long ones —
  the engine no longer pays for machinery it only needs when several
  conversations are active.
- Plain chats no longer carry the extra per-word processing that only
  tool-using requests need.

### Fixed

- A tool call that the app asked a model to make could come back with
  malformed arguments and either break the client reading it or, when
  streaming, end the turn with nothing at all. Those calls are now
  repaired to a valid shape, or reported clearly when the tool's own
  requirements cannot be met.

## [0.12.10] — 2026-08-11

Settings got a full visual refresh: every category now shares one design —
consistent cards, buttons, switches, and spacing, in both Light and Dark —
and tools and connectors introduce themselves with plain names and short
descriptions instead of wire identifiers. The menu bar now shows the official
Rapid "R" mark, drawn the native way so macOS keeps it crisp in Light, Dark,
and while the menu is open. Update availability lives in the menu's
"Update available" row and in Settings → App.

### Added

- A new model in the catalog: Ling 3.0 tiny, a fast hybrid model that runs
  in about 5 GB of memory.
- Very large mixture-of-experts models can now stream their expert weights
  from disk instead of holding everything in memory (`--disk-stream` for
  advanced setups).

### Changed

- Settings: unified visual system across Model Management, Tools,
  Connectors, Appearance, Privacy, and App — amber selection, consistent
  card padding and control heights, native switches and forms, and layouts
  that adapt down to small windows.
- The Audio tab's mode selector uses the same segmented control as
  Settings, fixing the hard-to-read white-on-amber selection.
- Model recommendations now share one RAM-tier source, so Chat and
  first-run suggestions agree about what fits on your Mac.
- Cached-model defaults prefer the best model you already have, not just
  the most recent one.

### Fixed

- Downloads report honest progress and failures instead of optimistic
  placeholders, and built-in web tools are harder to misuse.
- First-launch no longer probes the model catalog before you've picked
  anything, and onboarding language is clearer.
- The storage overview adds up correctly, and Markdown tables read
  properly with VoiceOver.
- Third-party license texts ship inside the app bundle as required.

## [0.12.9] — 2026-08-10

The desktop app can hear and speak now: a new Audio tab turns recordings into
text and text into speech, in a voice you pick — all on your Mac. Underneath,
Chat and Images learned to keep more than one model warm at once so switching
back to a recent one is instant, a new Muse Glimmer model joins the roster, and
math in chat finally renders as math instead of raw markup. The rest is the app
smoothing rough edges around updates, the main window, memory prompts, and
reusing models you already have.

Bundles the **Rapid-MLX 0.12.9** engine.

### Added

- **A new Audio tab: turn recordings into text, and text into speech.** Drop in
  or record audio and get a transcript back; or type a line, choose a voice, and
  hear it spoken aloud. Each built-in voice can be previewed before you pick it,
  and everything runs locally on your Mac — no upload, no account.

- **Chat and Images keep more than one model ready at once.** Within a memory
  budget you control, the models you have been using stay loaded, so switching
  back to a recent one responds immediately instead of pausing to reload it.
  When the budget is tight, the least-recently-used model steps aside on its own.

- **New model: Muse Glimmer 30B.** A native text model with tool-calling and
  step-by-step reasoning, available in a compact 4-bit build or full precision —
  no extra runtime needed.

### Fixed

- **Math in chat renders as math in the shipped app.** Equations now display as
  formatted mathematics rather than the raw LaTeX they briefly showed in
  downloaded copies.

- **A fresh install reuses models you already downloaded** — both from an earlier
  Rapid install and from another MLX app on your Mac — so first launch does not
  re-download gigabytes you already have.

- **Updates and image previews no longer get stuck in a stale state**, the main
  window closes and reopens cleanly, and overlapping "free up memory?" prompts no
  longer pile on top of each other.

- **When a model wrongly insists it cannot access real-time information, Chat
  quietly asks once more** so you still get a real answer instead of a refusal.

- Under the hood: vision models that mix text and images now run through a single
  ordered lane for stability, and model-status metrics are exposed for
  diagnostics.

## [0.12.8] — 2026-08-10

The desktop app makes pictures now: a new Images tab renders locally, and Chat
can read images you attach. Everything else is the engine and app catching the
rough edges — the Images tab that could not actually generate in a downloaded
copy, a chat speed number that disagreed with itself, and a handful of
server-side correctness fixes underneath.

Bundles the **Rapid-MLX 0.12.8** engine.

### Added

- **Connectors: give the model tools from MCP servers, from Settings.** A new
  **Settings → Connectors** panel adds, edits and removes MCP servers, shows
  whether each one connected — and the reason when it didn't — and lists the
  tools it exposes with an on/off switch each. The first time the model calls
  one, Rapid asks: **Allow once / Always allow / Don't allow**, showing which
  connector it came from and exactly what arguments it was given. "Always
  allow" applies to that one tool, and every remembered answer can be reviewed
  and reset. Connectors are off by default, edits apply without restarting the
  model, and a connector that won't start now reports why instead of taking
  the whole local server down with it. Previously this was command-line only:
  you had to hand-write `~/.config/rapid-mlx/mcp.json` and had no way to see
  what connected or what it was doing.

- **A new Images tab generates pictures on your Mac.** Choose an image model,
  follow the readiness prompt to load it, type a prompt and generate. Each
  render joins a filmstrip below the picture; selecting an earlier one brings
  its prompt back so you can adjust and re-run. Aspect ratio is 1:1, 3:4 or
  4:3, and the save button writes a PNG wherever you choose.

- **Chat accepts image attachments.** With a model that can read images, attach
  one to a message and ask about it. Models that cannot read images show the
  attach button disabled with "This model doesn't support images", rather than
  letting you attach one and failing later.

- **Web-page approvals now include “Always allow.”** Choose it once to let the
  model read future public web pages without interrupting you for every URL.
  Private and local addresses remain blocked, and the permission can be turned
  off again in Settings → Tools.

### Changed

- **The speed test is gone.** It was a card on the empty chat screen that
  measured your Mac against the model you had open. It needed two fixes in
  the week it existed, and the number it produced still disagreed with the
  one the chat itself reported for the same model in the same minute — so it
  created more doubt than confidence. The chat caption below, which is now
  correct, is the better answer to "how fast is this".

- **The chat tab opens on the conversation, not on two ads for itself.**
  "Connect tools" and "Speed on Mac" sat on the empty screen as the first
  things a new user saw. Tools still live in Settings → Tools; the speed test
  is gone for the reason above.

- **The model menu focuses on models.** The "N small models hidden" footer is
  gone, and a menu with models in it no longer ends with "Refresh catalog" and
  "Type a model name…" — those are now offered only when there is nothing to
  choose yet. What you get is the quickstart picks, the recommendations, and
  the model list, with the ones you have downloaded first in each group.
  Models under 1B stay hidden unless you turn them on in
  Settings → Model Management, except for the one you are currently using,
  which always shows.

### Fixed

- **The Images tab actually makes an image in the installed app.** In a
  downloaded copy the picture engine was missing, so loading an image model
  failed to start and nothing could render — the tab was broken the first time
  it would ever reach anyone. The engine now ships inside the app, and both
  built-in image models produce pictures from `/Applications`.

- **The chat's tok/s is now the model's actual writing speed.** It timed the
  whole turn, including the time the model spent reading your prompt before
  writing anything. On a long or tool-carrying prompt that is most of the
  turn, so the same model was captioned at 13 tok/s while generating at ~131.
  The caption now measures only the writing, and reports time-to-first-token
  separately — which is what the wait actually was. (The model picker still
  shows its own estimate for a model you have not run yet; that is a
  different measurement, on different hardware assumptions, and is unchanged
  here.)

- **"Browse all models" in the setup wizard opens the model catalogue.** It used
  to close the wizard instead — your chosen model was discarded and you landed
  on the chat surface pinned to a model you never picked. It now opens
  Settings → Model Management with the wizard still behind it, so closing
  Settings puts you back on your selection.

## [0.12.7] — 2026-08-07

The privacy switch does what it looks like it does, and the models Rapid
suggests for your Mac are now the ones that were actually measured on hardware
rather than the ones that seemed reasonable.

From this release Rapid and its engine share **one** version number. 0.12.6 is
skipped on purpose: the app already shipped as 0.12.6 before this work landed,
and reusing that number would have made it mean two different things.

Bundles the **Rapid-MLX 0.12.7** engine.

### Fixed

- **The "Send anonymous usage data" switch stays where you put it.** Turning it
  on recorded your choice correctly, but the switch itself sprang back to off
  until you left Privacy and came back — so a setting about your privacy looked
  like it had refused you. The natural response is to press it again, which
  left people unsure what had actually been saved.
- **The link to get a Brave search key goes to the right page.**

### Changed

- **The suggested models for your Mac are now measured, not estimated.** Each
  memory size offers exactly two: a faster one and a smarter one, with the
  memory each really uses and how quickly it answers taken from runs on real
  hardware. Two exceptions: on Macs with 64 GB or more, the *smarter* choice is
  a previous recommendation carried over without a measurement, because those
  models are bigger than the machine the benchmarks ran on. The faster choice
  is measured on every size.
- **Models and Model Management are one screen** instead of two that overlapped.

### Added

- **A recovery path when a model is too big for your Mac.** If the memory check
  says the suggested model will not fit, you are offered a smaller one that is
  honestly labelled as less capable — rather than being sent back to a list
  whose smallest option is the one that just failed.
- **Code and tables render properly in chat** — syntax highlighting, and
  markdown tables that look like tables.
- **Models with no compatible benchmark now say "Untested"** instead of showing
  an empty dashed track that read as a score of zero. A missing measurement and
  a poor result should not look the same.

## [0.12.6] — 2026-08-07

Rapid can now look things up. The headline is a set of built-in tools —
weather, web search, and fetching a page — that a model can use mid-answer,
with you deciding what it is allowed to reach. The rest of the release is
about the app telling the truth: a first question that got a real answer, a
Rename box that took what you typed, buttons that did what they said, and a
model list that stopped offering models that can never chat.

Bundles the **Rapid-MLX 0.12.5** engine.

### Added

- **Built-in tools: weather, web search, and reading a web page.** Ask about
  the weather or something recent and the model can go and find out, showing
  a card for each lookup that you can open to see exactly what it asked for
  and what came back. Fetching a page asks your permission first, naming the
  site, and "Don't allow" is a normal answer rather than an error. Search
  works out of the box; you can switch the backend in Settings → Tools, and
  private and local addresses are refused.
- **Conversations can be pinned, renamed and archived.** Right-click a chat in
  the sidebar, or use the ··· button that appears when you hover it. Pinned
  chats sit in their own group above everything else; archived ones move into
  a collapsed "Archived" group you can reopen at any time. Renaming a chat
  stops Rapid from re-titling it for you afterwards.
- **Deleting a conversation asks first**, names the chat it is about to
  delete, and says plainly that it cannot be undone.
- **Every message has its own actions.** Copy any message; edit a question you
  already sent; regenerate an answer you did not like.

### Fixed

- **The first question in a chat gets a real answer again.** Because Rapid's
  web tools are on by default, every opening message quietly carried an
  instruction telling the model that anything not in "the tool result" was
  unknown to it — when there was no tool result at all. Asked *what is the
  capital of France?*, the starter model answered "I don't have access to
  current or external data". That instruction now only travels with a message
  that actually has a tool result behind it.
- **Regenerating a reply replaces it instead of piling up.** The retry button
  under an answer used to append a second answer below the first.
- **Renaming a chat works.** Picking Rename put a text box on the row, but the
  keyboard never moved to it: what you typed went into the message box at the
  bottom of the window instead, so the title you meant to set became a draft
  message, and neither Return nor Escape could get you out of the row. The
  same fault affected editing a message you had already sent.
- **The setup wizard appears for everyone again.** If your Mac already had any
  model downloaded — from an earlier version, from the command line, from
  anything — Rapid started that model the instant it launched, and starting a
  model was enough to suppress the first-run wizard entirely. New users on
  such a Mac never saw it, and instead of the small starter model the wizard
  offers, they silently got whichever model happened to sort first
  alphabetically. Now nothing starts until either the wizard is done or you
  have used Rapid before — and nothing starts before you have answered the
  "Help improve Rapid-MLX" question, either.
- **Models that can only make video are no longer offered as chat models.**
  Eight of them sat in the model picker and in Model Management looking like
  ordinary chat models, with nothing to distinguish them. They cannot hold a
  conversation at all, so picking one led to "Couldn't start… Try again" —
  advice that would fail every time — and you could reach that after
  downloading up to 64 GB.
- **Buttons in Settings that did nothing now do something.** Several recovery
  buttons — including the one on the first-run card you see when a download or
  a model start has just failed — highlighted when you hovered them, accepted
  the click, and then did nothing at all. They now open the Settings page they
  name. The one that offered to open a "Permissions" screen has been removed:
  there is no such screen.
- **"Open-source credits" and "Privacy policy" no longer open a 404.** Both
  links pointed at addresses that do not exist. The credits list itself was
  wrong as well — it named two libraries Rapid does not use, left out three it
  does, and said the engine's Python dependencies are not bundled, which
  stopped being true in 0.6.6.
- **Model Management and the readiness banner stop saying untrue things** about
  what is installed, what is running, and what a button is about to do.
- **Settings switches work with VoiceOver again.** The toggles in Settings
  rendered as inert text to assistive technology: VoiceOver announced them as
  labels rather than checkboxes, and activating one reported success without
  changing anything. They are native checkboxes again, and the sidebar's New
  Chat, Launch and conversation rows now carry stable identifiers.
- **A model that wedges while starting no longer leaves the app stuck.** Rapid
  now notices and recovers instead of waiting forever.
- **Web search now says what actually went wrong.** DuckDuckGo — the free
  backend Rapid uses out of the box — starts refusing searches from your Mac
  after the first few, and Rapid read that refusal as an ordinary success with
  no results. The tool card said "Web search couldn't finish. Check its
  settings, then try again," which sent you to a Settings page where nothing
  was wrong. It now names the situation ("DuckDuckGo is rate-limiting web
  searches from this Mac") and offers a one-click jump to Settings → Tools,
  where you can switch to Brave Search or Tavily. The Settings caption for
  DuckDuckGo no longer claims it "works out of the box."
- **Turning down a web page is no longer treated as a breakage.** When a model
  asked to fetch a page and you chose "Don't allow", Rapid showed a red card
  reading "The tool couldn't finish. Check its input, then try again" — telling
  you something had gone wrong, blaming what you typed for it, and offering to
  retry the thing you had just refused. Declining is now shown as what it is: a
  quiet note saying "You didn't allow this, so there's nothing to show. Ask
  again if you change your mind," with no error styling and no retry button.
  Stopping a reply while the permission box is open is no longer recorded as
  you having said no, either.

## [0.12.5] — 2026-08-05

A stability release. The headline is a crash: a reply with a formula in it
took the whole app down, and an ordinary maths question was enough to
produce one. The starter model that shipped with 0.12.1 also turned out to
be unusable, so it has been replaced.

### Fixed — crashes and hangs

- **Maths no longer crashes the app.** When a reply contained a formula, the
  app quit outright with a macOS crash dialog — a plain `2 + 2 = 4` was
  enough. Formulas now show as their plain-text source instead of taking the
  app down. (Typesetting them properly is still to come.)
- **A silent server no longer hangs the chat.** Sending a message used to sit
  there indefinitely if the model stopped responding; it now fails after a
  bounded wait and the message stays retryable.
- **Quitting no longer stalls or crashes**, and starting the app no longer
  risks shutting down a Rapid-MLX server you started yourself in a terminal.

### Fixed — first run

- **The starter model has changed.** The previous one fell apart on ordinary
  multi-step questions — doubling words, then looping until it ran out of
  room. Measured on the same Mac and the same question: 0/4 correct before,
  16/16 after, with the first answer arriving in about a second.
- **Two models that could fail to answer are temporarily hidden.** Ministral 3
  and Gemma 4 E2B looked like ordinary small chat models in the picker, but on
  some Macs sending them a message produced no reply at all, or a reply that
  made no sense. They stay hidden until that is fixed.
- **8 GB Macs get a recommendation instead of a rejection**, and every RAM
  size now gets a "smart" and (where it helps) a "fast" pick.

### Fixed — chat and models

- **Conversation history keeps its order.** Older chats could jump around in
  the sidebar.
- **The compose box grows with what you type**, and the transcript stops
  yanking you back to the bottom while you are reading further up.
- **Loading a model that will not fit is now blocked** with a warning based on
  free memory at that moment, rather than freezing or panicking the Mac.
- **"Speed on this Mac" shows live progress and an estimate** instead of an
  unmoving spinner, and the size column no longer mixes download size with
  memory use.

### Fixed — connecting your tools

- **Your API key is no longer shown in the clear** in the copyable setup
  snippets on the Launch page.
- **Cursor no longer gets a configuration that cannot work.** Cursor routes
  requests through its own servers, so a `localhost` address was never
  reachable; the app now says so and points at Claude Code, Cline, or
  Continue for a local connection.

### Changed

- A refreshed visual system for the model list and its states.
- Bundles the **Rapid-MLX 0.12.4** engine. What you should notice: replies
  are faster on a long conversation because the model no longer re-reads
  what it has already read; models that think before answering keep their
  reasoning out of the answer; the app can tell whether a model is busy or
  idle; and several models that used to fail to start — including the
  Gemma 4 family — now load. LFM2.5 models are newly available.

## [0.12.1] — 2026-08-03

The first **signed + notarised** release — it installs and opens without the
Gatekeeper "unverified developer" warning that the 0.12.0 internal build hit.
Two dogfood fixes and the latest engine.

### App

- **First-run onboarding no longer depends on the shared Hugging Face cache.**
  A brand-new user who happens to have unrelated models on disk (e.g. Whisper
  from another tool) now still gets the one-click Quickstart instead of being
  dropped into the raw model picker. Onboarding keys on the app's own state.
- **The menu-bar icon is now the brand cheetah**, matching the app icon (it
  was a lightning bolt).

### Engine

- Bundles the **Rapid-MLX 0.12.1** inference engine: agent / DeepSeek
  stability, prefix-cache correctness, Gemma 4 fixes, a validated audio stack,
  and community-benchmark submission over HTTP.

## [0.12.0] — 2026-08-02

The first release of Rapid-MLX Desktop as an **open-source** app in the
`raullenchai/Rapid-MLX` monorepo (Apache-2.0). An internal-testing build
ahead of 1.0.

### App

- **Ollama / ChatGPT-style layout.** A single-column chat with an inline
  model picker in the compose box; the model starts on your first message
  (no start/stop buttons). A sidebar with New Chat, a Launch page of
  connect-your-tools cards, and your conversation history.
- **Conversation history** persists privately on device (owner-only file
  permissions) and survives quit.
- **"Speed on this Mac"** benchmarks the current model on your hardware, with
  an optional submit to the community leaderboard.
- Fixed a dead close button on the Launch page.

### Engine

- Bundles the **Rapid-MLX 0.11.9** inference engine (plus the latest merged
  fixes), including the freeform `bench` fix so "Speed on this Mac" reports a
  real number.

### Privacy

- Telemetry stays **opt-in** and anonymous; the app and the embedded engine
  share one client ID so an install is never double-counted. A new
  loopback-only `RAPID_MLX_TELEMETRY_ENDPOINT` lets you audit exactly what is
  sent by pointing it at a local server.

## [0.11.0] — 2026-07-28

This release upgrades the bundled inference engine to **Rapid-MLX 0.11.1**
and rounds out a broad reliability, privacy, and first-run polish pass.

### Engine

- **Bundled engine upgraded to Rapid-MLX 0.11.1** (from 0.10.8). The wins
  you'll feel most: **longer conversations in the same memory** — the
  running model's key/value cache is now quantized live, so large contexts
  fit where they used to run out of room; a **quieter, faster first run** —
  cold-start "dead air", leaked download progress bars, and the
  unauthenticated-download advisory are gone, and the starter model's real
  size and progress are shown correctly; **more reliable tool and JSON
  output** — tool calls are grammar-constrained by default; and **better
  connector support** — the in-chat agent loop can now talk to multiple
  Model Context Protocol (MCP) servers. Large reasoning models also serve
  more coherently (a Qwen3.6 normalization bug was corrected), and a new
  output-coherence release gate guards against garbage generations.

### Privacy

- **Telemetry is now explicit opt-in, off by default.** On first run the
  app asks once whether to share anonymous usage data; nothing is sent
  unless you agree, and the choice is shared with the bundled engine so
  you are never counted twice or asked twice.
- **In-app feedback.** You can now send feedback, with optional
  diagnostics, from inside the app.

### Models & downloads

- **Downloaded models come first.** The model picker and the All-models
  list now surface the models you already have at the top.
- **No more accidental re-downloads.** A default-quantization alias you
  already downloaded is recognized as cached instead of fetching a second
  copy.
- **Reusable chat presets.** Save a system-prompt-and-settings combination
  once and reuse it across chats.

### Reliability & fixes

- **Clearer failure messages with a way out.** When something fails — a
  model won't load, a tool errors, a download stalls, permission is denied
  — the app now shows a plain-language reason and a single recovery action
  instead of a generic error.
- **Stopping mid-startup is clean.** Pressing Stop while a model is still
  starting no longer wedges the app or shows a false failure.
- **More small fixes:** safer message editing, offline browse re-reads,
  weather lookups for accented and non-Latin city names, steadier system
  indicators, and an onboarding window that can no longer strand itself
  off-screen.

## [0.10.8] — 2026-07-21

- **Fixed: audio-only models no longer clutter the model list.** Text-to-speech
  and speech-to-text models (Whisper, Kokoro, Parakeet, and their variants)
  were showing up in the model picker and in Model Management even though the
  app has no microphone or audio input to use them — picking one and pressing
  Start would fail with a confusing setup error. They are now hidden
  everywhere they used to appear, so every model you can choose is one the app
  can actually run.

## [0.10.7] — 2026-07-21

- **Smaller download and install — about 29 MB lighter.** The bundled
  inference engine shed three chunks of weight that never ran on your
  Mac: a second, redundant copy of `pip` embedded in the runtime,
  NumPy's bundled test suites, and the orphaned Tcl/Tk GUI toolkit
  (left behind after its Python bindings were already stripped). Nothing
  about how the app works changes — same models, same speed, same tools
  and connectors — it is simply a lighter package to download and to
  auto-update.

- **Locator slot classifier trimmed to its three live slots (internal,
  no user-facing change).** The v0.8.10 cutover stopped
  `ServerLocator.find()` from discovering the `rapid-mlx` CLI via PATH /
  Homebrew / pipx / uv, which left the matching `classify()` branches —
  and the `.homebrewAppleSilicon` / `.homebrewIntel` / `.pipx` / `.uv` /
  `.path` `ResolvedSource` cases — unreachable (a real `RAPID_BIN`, even
  a Homebrew symlink, already classifies as `.rapidBin`). Removed the
  dead branches and cases plus the unwired `SettingsView.formatSource`
  helper. The About-panel source label for every real install
  (RAPID_BIN / in-app update / bundled) is unchanged. Supersedes the
  "`classify()` left intact" note in the 0.8.10 entry below.

- **Dead code removed from the app (internal, no user-facing change).**
  Deleted unused Swift helpers, two fully-dead source files, and a
  264 KB image asset with no remaining references. Build-hygiene only;
  no behaviour changes.

## [0.10.6] — 2026-07-18

A feel-and-accessibility pass. The interface now scales with your system
text size, animates with interruptible springs that honor Reduce Motion,
and every button acknowledges your press. Onboarding gained a Skip,
Settings is keyboard-navigable, deleting a chat can be undone, and file
permissions are scoped more tightly.

### Added

- **Press feedback on the controls you tap most.** Buttons, chips, the
  send button, and the onboarding model cards now depress the instant you
  press them, with a subtle spring — the interface finally feels alive
  under the pointer. Reduce Motion keeps the cue without the movement.
- **Undo a chat delete.** Deleting a conversation now shows a brief Undo,
  so an accidental delete is one click from being restored.
- **Skip the welcome screen.** First-run onboarding now has a clear Skip /
  Escape exit.

### Changed

- **App content scales with your text size.** The chat transcript and ~80
  other text sites now follow the macOS Dynamic Type setting instead of
  ignoring it. (The compact CPU/GPU/RAM readouts stay fixed-size by
  design.)
- **Calmer, interruptible motion.** Animations moved from fixed-duration
  curves to interruptible springs, and Reduce Motion is now honored across
  the whole app — previously it was respected on only one screen.
- **Keyboard-navigable Settings.** The Settings category rail is now a
  native, arrow-key-navigable list with proper VoiceOver semantics.
- **Clearer recommended models.** Recommended model cards now sit at a
  distinct elevation from the dense model table below them.

### Fixed

- **File permission prompts show the real target,** and access you grant
  is scoped to the conversation that asked for it rather than persisting
  for the whole app session.
- **No stray machine text.** When a model that can use tools doesn't
  actually use one, raw internal text no longer appears in the reply.

## [0.10.5] — 2026-07-15

Rapid can now act, not just answer. The model can read and write files,
run commands, and read web pages — each one gated the same careful,
ask-first way Connectors already were. Nothing runs until you allow it,
and a new **Permissions** screen lets you decide, per capability, what may
run without asking.

### Added

- **File tools.** The model can read a file, list a folder, create a file,
  and edit an existing file — but only after you allow access to that
  folder. Reading and editing are separate permissions: letting the model
  read your files never lets it change them.
- **Run commands.** The model can run shell commands in a restricted
  sandbox — no network, and it can only change files in folders you've
  allowed, though it can still read files on your Mac. Rapid shows the exact
  command and asks before the first run.
- **Browse the web.** The model can open a web page and read it as clean
  text. Private and local network addresses are always refused, so a page
  can't be used to reach devices on your network.
- **Permissions (Settings → Permissions).** One screen to decide which of
  these — read, edit, run, browse, and connector tools — may run without
  asking. Everything is off by default, so Rapid asks the first time. A
  single **Auto-approve everything** switch is there for unattended use,
  for when you trust what you're running.

## [0.10.4] — 2026-07-13

Connectors arrive. Rapid can now use tools from Model Context Protocol
(MCP) servers — a local filesystem or database bridge, a web-search
service, and more — with the same careful, ask-first approach the rest of
the app takes with your data.

### Added

- **Connectors (Settings → Connectors).** Add, edit, enable, and remove
  MCP servers from a form — the same tools a command-line user wires up by
  hand, but with a per-connector on/off switch, live connection status, and
  a consent step that shows the exact command before it ever runs. Off by
  default; nothing runs until you turn it on.
- **Per-tool approval.** The first time the model wants to run a given
  connector tool, Rapid asks: **Allow once**, **Always allow** (remembers
  that tool), or **Don't allow**. A prominent **Auto-approve all** switch is
  there for unattended use — the only way to skip the prompts, and only if
  you trust every connector you've added.

### Fixed

- Hardened how Rapid reads output from the local model process, closing a
  rare crash that could happen if a background process shut down at just the
  wrong moment.
- In-app update checks are more reliable: Rapid now checks for new versions
  through an additional path and falls back cleanly if one is unreachable.

## [0.10.3] — 2026-07-11

A polish pass on the very first thing you see. The first-launch setup
screen has been redesigned to match the rest of the app, and the
download size it reports is now accurate.

### Changed

- Redesigned the first-launch setup screen. While the starter model
  downloads, you now get a cleaner, on-brand screen: a clearer progress
  bar with a live percentage, and a note that everything runs privately
  on your Mac — no cloud, no account.

### Fixed

- The setup screen no longer over-states the download size for the
  compact starter model. It previously showed roughly double the real
  size (about 957 MB for a model that is really ~495 MB); the figure now
  matches what actually downloads, and the bar corrects itself to the
  true size as soon as the download begins.

## [0.10.2] — 2026-07-11

A brand-new first impression. The model that greets you on a fresh
install is now a genuinely capable one — it holds a real conversation
and can use tools — while keeping the download small and the launch
instant.

### Changed

- The first-run starter model is now Bonsai 1.7B instead of the old
  0.6B model. It's still a small (~0.5 GB) download that starts almost
  immediately, but unlike the previous starter it stays coherent across
  a full back-and-forth and can actually use tools. The old 0.6B model
  tended to fall apart after the first couple of turns.
- The friendly "trade up to a larger recommended model" nudge now
  appears a little sooner — after about three messages instead of five —
  so you find your way to a stronger model faster.
- Mistral and Devstral models can use tools again. The updated engine
  reads their tool calls correctly, so the Tools switch works for them
  once more (this reverses the temporary limitation noted in 0.10.1).

### Under the hood

- Updated the bundled engine to rapid-mlx 0.10.8.

## [0.10.1] — 2026-07-09

A reliability pass on model recommendations and tool use. Every model
we recommend for a role (Default / Speed / Quality / Coding) is now one
we've personally tested end-to-end, and models that can't reliably use
tools no longer pretend they can.

### Changed

- Recommended models are now vetted per role and RAM size. Three picks
  that looked good on paper but fell apart in practice were replaced
  with models we verified handle tools cleanly: the "Speed" pick on
  16 GB machines, and the "Coding" pick on both 17–24 GB and 25–36 GB
  machines.

### Fixed

- Models that can't reliably call tools — phi-4-mini, deepseek-coder-v2-lite,
  the deepseek-r1 8B distill, and the Mistral / Devstral family under the
  current bundled engine — no longer show a Tools switch that silently
  does nothing or spills raw tool-call text into the chat. They now chat
  normally, and the model picker labels them "no tools" so the limitation
  is clear before you pick one.

## [0.10.0] — 2026-07-09

A big first-run and model-picking refresh, plus a major jump in the
bundled inference engine (rapid-mlx 0.9.12 → 0.10.5). New users get a
guided welcome that starts them chatting in a couple of clicks;
everyone gets a redesigned model browser, the option to keep models in
a folder of their choosing, more and better models to choose from, and
a round of reliability fixes.

### Added

* **A guided first-run welcome.** Opening Rapid for the first time now
  walks you through three simple screens — a short welcome, a "choose
  your first model" step, and the download — instead of dropping you in
  front of a single dense setup card. It recommends a tiny starter
  model that downloads in seconds and starts instantly, and lets you
  trade up to a larger, higher-quality model right there if you'd
  rather.

* **A redesigned model browser.** Model Management is now a proper
  browser: models are grouped by what they're good at, with at-a-glance
  Accuracy and Speed ratings, an estimated download size, and a clear
  "recommended" pick for each job, so choosing the right model no
  longer means guessing from a flat list of names.

* **Keep your models anywhere.** You can now point Rapid at a folder of
  your choice for downloaded models — for example an external drive —
  instead of being locked to the default location, which is handy when
  larger models start filling up your disk.

### Changed

* **More and better models to choose from.** The updated engine adds
  first-class support for the latest Qwen 3.6, Gemma 4, and gpt-oss
  model families, so the model browser now surfaces stronger options
  across the quality-and-size range.

* **Reasoning effort now carries through.** Requests that ask for a
  specific reasoning effort (low / medium / high) are honored correctly
  by reasoning-capable models, so you can dial how much a model "thinks"
  before it answers.

* **Faster answers on more models.** The updated engine streams replies
  out noticeably quicker across a wider set of models, at the same
  quality.

### Fixed

* **The menu-bar icon shows reliably on the latest macOS.** On recent
  macOS versions the tray icon could fail to appear, hiding the main
  way to reopen Rapid or quit it. It now uses a more dependable
  mechanism and shows consistently across macOS versions.

* **No more chats stuck on "Thinking…".** If Rapid was closed while a
  reply was still streaming, that conversation could reopen frozen on
  "Thinking…" forever. Interrupted replies are now cleaned up on load
  so the chat is usable again.

* **Clearer message when the model is busy or low on memory.** When a
  request can't be served because the model is at capacity or your Mac
  is low on memory, Rapid now explains what happened in plain language
  and what to try, instead of showing a raw error.

* **Gemma 4 works on a fresh install.** A missing dependency could stop
  Gemma 4 models from loading right after install; the engine now
  bundles what it needs so Gemma 4 works out of the box.

## [0.8.20] — 2026-07-03

Updates the bundled inference engine (rapid-mlx 0.9.9 → 0.9.12) and
ships a round of app polish: a single menu-bar icon, clearer first-run
setup progress, safer chat-history storage, smoother image attachments,
and VoiceOver support for streaming replies.

### Changed

* **Gemma 4 replies are faster.** The engine now drafts several tokens
  ahead and verifies them in one pass (speculative decoding) for Gemma
  4 models, so answers stream out noticeably quicker at the same
  quality.

* **One menu-bar icon, not two.** Rapid used to install two separate
  menu-bar icons with split menus, which looked broken. It now shows a
  single icon with everything in one place — open, new chat, the live
  model status, settings, and quit.

* **First-run setup shows one clear progress total.** While Rapid
  downloads the engine and your first model, the setup screen now shows
  a single combined amount (for example "172 / 413 MB") that scales up
  to GB for larger models, instead of a cramped two-part line whose
  units could get cut off.

### Fixed

* **Tool-using chats are more reliable.** Two rough edges are smoothed:
  a coding model could stall mid-tool-call while streaming, and a tool
  call whose arguments arrived in an unusual shape could be mishandled.
  Both now parse cleanly, so tool calls complete consistently across a
  conversation.

* **Reusing the exact same context no longer wastes memory.** When a
  follow-up message repeated a prompt the engine had already cached, an
  internal bookkeeping slip could add a duplicate cache entry. The
  engine now trims it, keeping long and repeated conversations steady.

* **One unreadable saved chat no longer erases your whole history.** A
  single corrupt or newer-format conversation on disk could previously
  empty the entire sidebar. Rapid now keeps every chat it can still
  read, tells you how many were skipped, and — if the whole file is
  unreadable — lets you restore a backup from Settings → Storage.

* **Image attachments no longer make the app stutter.** A photo attached
  to a message was re-read and re-decoded at full size on every screen
  update, which could hitch scrolling and typing while a reply streamed.
  Thumbnails are now prepared once in the background and reused.

* **Quickstart models use a cleaner on-disk layout.** A Quickstart model
  was stored one folder deeper than intended. New downloads now land in
  the right place; models you already have keep working untouched.

### Accessibility

* **VoiceOver reads streaming replies.** As an assistant reply streams
  in, VoiceOver now announces it — with a start cue and a completion
  notice — instead of staying silent until you navigate to it by hand.

## [0.8.19] — 2026-06-30

Ships the updated inference engine (gemma-4 now loads on 18 GB Macs;
tool-using chats no longer error on repeat calls) and finishes the
plain-language cleanup of error, crash, and loading messages plus the
response-length control.

### Fixed

* **gemma-4 models now load on 18 GB Macs.** A gemma-4 model that
  genuinely fit in memory could fail its very first message with an
  "Internal error during streaming" — the engine over-estimated how
  much memory the model needed and rejected the request before it ran.
  It now measures correctly, so gemma-4 starts and chats normally on
  smaller Macs.

* **Tool-using chats no longer fail with an error on a later message.**
  On the default Qwen models, a chat that called a tool could return an
  error once a follow-up message reused the same context — the engine's
  memory-saving cache hit an internal bug. Tool calls now keep working
  across a conversation.

* **Error, crash, and loading messages are now plain English.** When
  something goes wrong (the model can't start, a chat is interrupted, a
  port is busy), the message you see is a short, actionable recovery
  step instead of engine internals or raw error codes. The loading
  screen and menu-bar / picker / settings / download states got the
  same treatment.

### Changed

* **The "Max Tokens" control in Settings now takes effect.** The slider
  that caps how long a single response can get is wired through to the
  model, so lowering it produces shorter answers (and frees the model
  up sooner) as expected.

## [0.8.18] — 2026-06-29

A model-recommendation quality fix plus a cosmetic cleanup of the
update window. No change to the bundled inference engine.

### Changed

* **The "Speed" model recommendation no longer points at a model that
  can't actually hold a conversation.** Every Mac's recommended-models
  list has a "Speed" pick. It used to be a 1-billion-parameter model
  (`gemma3-1b-qat-4bit`) chosen purely for raw tokens-per-second — but
  on-device testing showed it is too small to be usable: it scores 17
  out of 100 on general reasoning (around random-guess level), gets
  basic arithmetic wrong, and falls apart after the first follow-up
  question. The "Speed" pick is now `qwen3.5-4b-4bit` on most Macs (and
  `phi-4-mini-4bit` on 16 GB Macs), which run at essentially the same
  speed (~158 tokens/sec) but score roughly four times higher on
  reasoning and stay coherent in multi-turn chat. `gemma3-1b-qat-4bit`
  is still available in the full model list for anyone who wants it.

### Fixed

* **The in-app "Update available" window now shows formatted release
  notes instead of raw Markdown.** Previously the notes appeared with
  literal `##`, `**`, and backtick characters dumped on screen. They now
  render with proper headings, bold text, bullet lists, and inline code,
  and the redundant version line at the top (already shown in the window
  header) is hidden. (This formatting takes effect for updates offered
  *after* you are on v0.8.18.)

## [0.8.17] — 2026-06-29

Dogfood hotfix on top of v0.8.16. On-device dogfood surfaced that the
**Gemma 3** family of models produced garbled output — every space in a
reply was rendered as `__`, so a sentence came back as
`The__search__results__mention…`. The bug lived entirely in the bundled
inference engine (rapid-mlx), so this release fixes it by bumping the
engine from v0.9.7 to its latest release, v0.9.8. No desktop UI code
changed.

### Fixed

* **Gemma 3 models now render spaces correctly instead of `__`.** The
  bundled engine has a repair pass that rescues a small class of models
  whose tokenizer ships a mismatched decoder. That pass identified a
  broken decoder by scanning the vocabulary for byte-level space markers
  — but Gemma 3's vocabulary contains a *genuine* character that looks
  like one of those markers, so the pass wrongly concluded Gemma 3's
  (correct) decoder was broken and swapped it out, which destroyed the
  space mapping for the whole tokenizer. The engine fix
  ([rapid-mlx#959](https://github.com/raullenchai/Rapid-MLX/pull/959))
  replaces the vocabulary-scan heuristic with a behaviour-based check
  that observes how a space is actually encoded, so it only repairs a
  decoder that genuinely leaks markers and leaves Gemma 3 untouched.

### Changed

* **Bundled inference engine updated to rapid-mlx v0.9.8.** Beyond the
  Gemma 3 fix above, this brings additional community-curated quantization
  variants for the Qwen3.5 / Qwen3.6 families and internal stability fixes
  for speculative decoding and KV-cache quantization. No change to the
  default model or to how existing models behave on this machine.

## [0.8.16] — 2026-06-28

Dogfood hotfix on top of v0.8.15. On-device dogfood (a notched 18 GB
MacBook running macOS 14/15) surfaced a layout collapse that v0.8.15's
#459 width fix did not address — once a conversation grew taller than the
window, the message composer dropped off the bottom of the screen and
could not be recovered. Root-caused and confirmed fixed on the affected
device via a temporary on-screen layout probe (removed before this
release); codex review converged at 0 BLOCKING / 0 MAJOR.

### Fixed

* **The message composer no longer disappears once a conversation fills the window** ([#459](https://github.com/machinefi/rapid-desktop/issues/459) follow-up). On macOS 14/15 a `NavigationSplitView` nested inside a `VStack` proposes an *unbounded* height to its detail content. The outer `VStack` still clamps the split's own frame correctly, but inside it the detail column's flexible chat `ScrollView` grew to its full content height instead of scrolling — so a transcript taller than the window pushed the compose bar far below the window's bottom edge (windowed: the composer vanished; full-screen: the chat pane was empty but for its header). macOS 26 happens to bound this, which is why it never reproduced on the development machine and slipped past the v0.8.15 fix. The fix measures the height the outer `VStack` actually allocates to the split (a value that *is* reliably bounded) and hard-caps the detail column to it with `.frame(maxHeight:)`, which converts the unbounded proposal into a concrete bound so the transcript scrolls and the composer stays in view. The compose bar is additionally pinned as a bottom safe-area inset on the transcript so chrome (download strip, banners, a larger notch safe-area, full-screen tiling) can never push it off-screen. No behavioural change on macOS 26, where the detail column was already bounded.

## [0.8.15] — 2026-06-27

Dogfood hotfix on top of v0.8.14. Real-device dogfood of the shipped
v0.8.14 build surfaced two P0s the three-persona automated pass missed —
one that blocks every in-place upgrader at first launch, and one that only
manifests on a built-in MacBook screen — plus the first slice of a
user-facing-jargon cleanup. All three are fixed with no behavioural
regression in the replay; codex review converged at 0 BLOCKING / 0 MAJOR.

### Fixed

* **v0.8.13 → v0.8.14 upgraders no longer hit a permanent "Setup didn't finish" splash** ([#458](https://github.com/machinefi/rapid-desktop/issues/458)). On an in-place upgrade the Quickstart model is already on disk, so the bootstrapper's model-install leg threw `.alreadyInstalled`. Because both install legs run inside one `withThrowingTaskGroup` (first error cancels its siblings), that error tore down the sibling sidecar leg and surfaced as `.modelDownloadFailed("model already present…")` — the "Setup didn't finish" splash every upgrader saw, with Retry permanently stuck (the file is still there on the next attempt). Fix: resolve the model's on-disk state *before* the task group and, when it is already present, skip the model leg entirely (and drop its byte budget from the progress bar) while the sidecar leg installs normally. A corrupt/partial model directory (exists, no `config.json`) is removed first so the download can re-stage cleanly; if that removal fails (read-only mount, permission flip), the install now fails fast with an actionable filesystem error ("Couldn't clear an incomplete model download… Free up disk space or check folder permissions, then Retry.") instead of silently re-triggering the cancel cascade (codex r1 MAJOR). First unit coverage for the whole path: disk-state classification, the skip-leg upgrader case, and the corrupt-undeletable fail-fast case.
* **Green-button maximise / full-screen tiling no longer collapses the whole layout on MacBook screens** ([#459](https://github.com/machinefi/rapid-desktop/issues/459)). The main window scene is `.windowResizability(.contentMinSize)`, which makes `ContentView`'s `.frame(minWidth:)` the HARD minimum macOS enforces — including when the green traffic-light button snaps the window to one half of the display. The old 880-pt floor exceeds half the width of every built-in MacBook screen (~640–860 pt), so tiling forced the content to overflow the tiled frame: the brand header collapsed, the chat transcript blanked, and the composer was clipped off the right edge (the window became unusable, with no way to type). Fix: lower the floor to 640 pt (half of a 1280-pt display, the smallest realistic Mac screen), extracted to `ContentView.minWindowWidth` with a regression test pinning it ≤ half the smallest supported display. Default window size stays 1200×820; this only changes how narrow the window may be driven.
* **First-run download UI and the throughput pill drop internal jargon** ([#461](https://github.com/machinefi/rapid-desktop/issues/461), partial). The bootstrapper splash no longer says "Downloading / Installing / Fetching rapid-mlx engine…" (now "Downloading Rapid-MLX…" / "Installing Rapid-MLX…" / "Preparing setup…"), the dual-leg progress readout reads "Engine X / Y MB · Model X / Y MB" instead of "Sidecar …", and the footer throughput pill reads "<n> tok/s" instead of the "TPS" abbreviation (consistent with the per-message caption and the LM Studio convention). This is the high-visibility first-run slice; the broader `rapid-mlx`-leak sweep across menu / picker / error surfaces plus a forbidden-string lint rule, the truncated-units fix, and the dual-task single progress bar remain tracked in #461.

### Internal

* `CFBundleShortVersionString` 0.8.14 → 0.8.15, `CFBundleVersion` 130 → 131.

## [0.8.14] — 2026-06-26

Quickstart, onboarding, and returning-user resilience release. Combines the original v0.8.14 user-experience cleanup (five UX issues from a 16-userflow dogfood walk on top of v0.8.13) with four P1 fixes surfaced by a follow-on three-persona dogfood pass (fresh-install Maya / power-user Alex / returning-heavy Sam) that ran against the v0.8.14 prep build. None are P0 install-blockers (v0.8.13 already cleared that), but each one blocks or visibly degrades real-user paths — first-run, API integrations, and returning-user upgrade. v0.8.14 fixes all nine with no behavioural regressions in the dogfood replay; the bundled `rapid-mlx` sidecar also advances from v0.8.18 to v0.9.7 + PR #948 head, picking up the v0.9.0 series of routing / API hardening as a side-effect of the bump required to deliver the #447 fix.

### Fixed

* **Forced `kill -9` on the desktop no longer leaves a 30 GB orphan `rapid-mlx` sidecar holding the loopback port** ([#449](https://github.com/machinefi/rapid-desktop/issues/449), [vllm-mlx#942](https://github.com/raullenchai/Rapid-MLX/pull/942)). Persona 3 (Sam) dogfood caught: when the desktop dies under SIGKILL (force-quit, OOM kill, kernel panic), macOS re-parents the bundled `rapid-mlx serve` subprocess to launchd (PID 1) and the sidecar keeps running indefinitely — holding 20-30 GB of model weights in unified memory and the port the next launch needs to bind. Atexit on the desktop side cannot fire under SIGKILL, and the PortSweep reaper added in [#170](https://github.com/machinefi/rapid-desktop/pull/170) only runs on the *next* launch (so if the user closes the lid and walks away, the orphan persists forever). Fix: the bundled `rapid-mlx` now ships a parent-PID watchdog (vllm-mlx PR #942) that polls `os.getppid()` every 2 s and self-terminates the moment the live PPID stops matching the supervisor's stamp; `ServerManager.serveEnvironmentAdditions` now stamps `RAPID_MLX_WATCHDOG_PPID=<launcher PID>` on the spawned sidecar env (Layer 2, alongside `RAPID_MLX_API_KEY`). Two new test suites (one Swift, one Python) pin both halves of the protocol so a future refactor that drops the stamp cannot silently re-introduce the orphan-sidecar bug.
* **Streaming chat with `tool_choice=auto` / `tool_choice=required` no longer returns an empty body on hermes-parser aliases** ([#447](https://github.com/machinefi/rapid-desktop/issues/447), [vllm-mlx#948](https://github.com/raullenchai/Rapid-MLX/pull/948)). Persona 2 (Alex) power-user dogfood caught: any streaming chat that opted into tool-calling via `tool_choice=auto` / `tool_choice=required` (the default form for OpenAI SDK / LangChain / LiteLLM) terminated the SSE stream cleanly on Qwen-family aliases but never emitted a tool_calls delta — the user saw an empty assistant bubble and the chat appeared "stuck thinking forever" until they cancelled. The fix lives in the bundled rapid-mlx submodule: three-layer guard (envelope-aware split-prefix routing + 1-chunk hold-forward buffer for ambiguous `<` / `<t` prefixes + streaming synth fallback when forced `tool_choice` produces zero parser-detected calls) preserves the non-streaming `_synthesize_forced_tool_call` parity. Explicit `tool_choice:{"type":"function",...}` path remains regression-free.
* **Corrupt `sessions.json` no longer silently empties the sidebar** ([#450](https://github.com/machinefi/rapid-desktop/issues/450), [#453](https://github.com/machinefi/rapid-desktop/pull/453)). Persona 3 (Sam) dogfood caught: when `sessions.json` failed to parse (truncated by crash, disk corruption, schema breakage), the existing recovery path correctly backed up the file as `sessions.corrupt.<timestamp>.json` but the UI gave zero indication anything had happened — the user saw an empty sidebar and assumed weeks of conversations had been wiped. Fix: `SessionStore.lastLoadError` (new `SessionLoadError` value, Observable + Sendable) is populated by the backup path; a new `SessionLoadFailureBanner` renders above the model picker (orange, not red — the bytes are preserved and recoverable) with "Show in Finder" (opens the `.corrupt` backup) and "Dismiss" actions. Dismissal is keyed by SHA-256 digest of the corrupted bytes (codex r1 catch: the original time-stamp-based dismissal would re-fire on every relaunch because the canonical file regenerates with a fresh timestamp; a content-digest key fingerprints the actual corruption, so the same bytes silently dismiss but genuinely-new corruption re-pesters the user). Legacy time-stamp key preserved as a one-version upgrade fallback.
* **Restored sessions no longer have their model alias silently overwritten by the picker default** ([#451](https://github.com/machinefi/rapid-desktop/issues/451), [#453](https://github.com/machinefi/rapid-desktop/pull/453)). Persona 3 (Sam) dogfood caught: sessions saved with `qwen3.6-27b-8bit` reopened pointing to whatever the picker's current default was, and if the session's original alias was no longer in the v0.8.14 catalog (e.g. `qwen3.5-4b` got renamed in an upstream rapid-mlx bump) the substitution happened silently — the user lost the ability to tell which model their stored conversation was actually produced with. Fix: `SessionAliasRestore.resolve` (pure decision helper) returns one of `useSessionAlias` / `staleSessionAlias` / `noChange`; when the session's alias is genuinely absent from the live catalog the new `StaleSessionAliasBanner` surfaces a `rapid-mlx pull <alias>` hint while the on-disk session bytes remain untouched. Codex r1 caught a race in the catalog-probe path that would have false-fired the banner before catalog finished loading; a `Task.isCancelled` + binary-path equality guard now blocks the stale-catalog write-back.
* **First-launch chat detail pane no longer renders blank on a fresh `HOME`** ([#438](https://github.com/machinefi/rapid-desktop/pull/438), [#435](https://github.com/machinefi/rapid-desktop/issues/435)). `BootstrapCoordinator.state` initialised to `.checking`, so `BootstrapGateView.body` always rendered `SplashView` on the first SwiftUI body evaluation, then flipped to `.installed` a few ms later when `start()`'s async detect Task settled. The `Group { switch state }` case-swap raced `NavigationSplitView`'s first layout pass on a cold launch and the detail pane occasionally never received a body invalidation, leaving the user staring at a blank right column until they clicked something. Fix: `BootstrapCoordinator.init` now performs an eager bundled-marker check before publishing `.checking`, so when a bundled sidecar is present the coordinator publishes `.installed` immediately and `SplashView` is never inserted on first pass.
* **AutoStart prefers a cached, runnable model over the RAM-bucketed default** ([#437](https://github.com/machinefi/rapid-desktop/pull/437), [#436](https://github.com/machinefi/rapid-desktop/issues/436)). Post-Quickstart UX cliff: a 256 GB M3 Ultra user who had just pulled `qwen3-0.6b-4bit` via the bootstrapper still saw "Download & start qwen3.6-35b-4bit (~4.4 GB)" because `ModelPickerBar.recommendedDefault` consulted the RAM-bucketed default table without ever checking `ModelEntry.cached`. The new `CacheAwareDefault` pure-function helper applies a four-step ladder (cached-and-runnable preferred match → cached-and-runnable any match → fallback to RAM-bucketed default → safe minimum) so anyone who has already paid the download cost for a model never sees a stale large-download prompt at the AutoStart surface.
* **Quickstart download no longer stuck at `99% · <1 min left` after the subprocess exits cleanly** ([#441](https://github.com/machinefi/rapid-desktop/pull/441), [#440](https://github.com/machinefi/rapid-desktop/issues/440)). `DownloadManager.Job` was a plain `final class` (not `@Observable`), so SwiftUI's Observation framework didn't track terminal status mutations. The existing `handleExit` callback correctly flipped `job.status = .completed`, but the Quickstart card's `.task(id: job.status)` modifier never saw the change and stayed pinned to the last-rendered progress value. Marking `Job` `@Observable` makes the status flip observable, so the card re-renders into its "Ready to chat" state the instant the subprocess exits.
* **Onboarding tour Skip / Next / Esc now actually work** ([#443](https://github.com/machinefi/rapid-desktop/pull/443), [#439](https://github.com/machinefi/rapid-desktop/issues/439)). When `OnboardingTour` appeared on a parent window that wasn't the macOS `keyWindow` (canonical case: post-Quickstart relaunch where the user's foreground app is still Chrome / Finder / Slack), the SwiftUI `.sheet` couldn't intercept Esc (no key window), the 26×16 pt Skip button below the rounded-rect frame edge silently absorbed clicks without firing, and `acceptsFirstMouse` was false on the non-key-window sheet so even direct clicks were swallowed by the activation transition. Three coordinated fixes: `NSApp.activate(ignoringOtherApps: true)` + `makeKeyAndOrderFront` when the sheet appears, the Skip / Next hit-area widens from 26×16 pt to 50×32 pt, and the sheet hosts on a window that opts into `acceptsFirstMouse(for:)`.
* **`missingOverlay` swaps primary CTA to "Download update vX.Y.Z" when the updater has resolved a newer release** ([#433](https://github.com/machinefi/rapid-desktop/pull/433), [#432](https://github.com/machinefi/rapid-desktop/issues/432)). The v0.8.12 → v0.8.13 incident exposed the gap: every fresh slim-DMG user of v0.8.12 hit `missingOverlay` because of the ServerLocator off-by-one ([#430](https://github.com/machinefi/rapid-desktop/issues/430)), and even after v0.8.13 shipped the patch within hours, stuck users had no in-app discovery path back to a working release. The overlay now promotes "Download update vX.Y.Z" to the full-width primary CTA the moment `UpdateChecker.latestPublishedVersion` resolves a release newer than `CFBundleShortVersionString`, with Recheck and Quit Rapid-MLX kept as secondaries on the row below (so the recovery surface never loses its escape hatches). When no update is available the original two-button Quit + Recheck layout is preserved verbatim. The update URL is enforced to `https://` and the GitHub release host allowlist, so a poisoned `latest.json` cannot redirect a stuck user to an attacker-controlled binary.

### Internal

* `CFBundleShortVersionString` 0.8.13 → 0.8.14, `CFBundleVersion` 129 → 130.
* **Bundled `rapid-mlx` submodule advanced from v0.8.18 to PR #948 head** ([#452](https://github.com/machinefi/rapid-desktop/pull/452), [#454](https://github.com/machinefi/rapid-desktop/pull/454)) covering v0.8.19 → v0.9.7 + the issue #449 watchdog fix + the issue #447 streaming-with-tools fix. Picks up the v0.9.0 series of routing / API hardening as a side-effect of the bump.
* **Dogfood test harness now isolates CFPreferences per persona** ([#446](https://github.com/machinefi/rapid-desktop/pull/446)). `scripts/dogfood-isolate.sh` rewrites `CFBundleIdentifier` to a per-run `com.rapidmlx.rapid.dogfood-<8hex>` variant + ad-hoc re-signs, fixing the prior leak where `HOME=/tmp/...` didn't isolate `cfprefsd` (which keys on bundle-id, not HOME) — so every prior "fresh install" dogfood was actually a returning-user test in disguise, AND those tests were silently writing `rapid.install.lastSeenVersion=0.8.14` back into the user's prod plist. Test-only change, no user impact.
* **Formal release SOP documented** ([#445](https://github.com/machinefi/rapid-desktop/pull/445)) at `docs/release/release-sop.md` — 11-gate pre-tag checklist consolidating the lessons from the v0.8.10 → v0.8.13 slim-DMG saga. Codex r1-r3 review hardened it against the actual ad-hoc failure modes (notary `403`, R2 paths, slim DMG 2-step UDRW→UDZO).

## [0.8.13] — 2026-06-25

P0 hotfix for v0.8.12. **Every fresh slim-DMG install of v0.8.12 hits the "Setup didn't finish" overlay immediately after the bootstrap pipeline completes, with no in-app recovery path.** v0.8.13 patches the off-by-one in `ServerLocator` so the sidecar the bootstrapper just installed is actually findable.

**Root cause** ([#430](https://github.com/machinefi/rapid-desktop/issues/430)): `Sources/Rapid/Server/ServerLocator.swift` looked for the runtime-override slot at `runtime-override/bin/rapid-mlx`, but the bootstrap install pipeline publishes the sidecar at `runtime-override/rapid-mlx/bin/rapid-mlx` (the `rapid-mlx/` wrapper is the top-level arcname of `scripts/build-sidecar-tarball.sh`'s tarball, preserved through extract → atomic publish). The mismatch has been latent since the runtime-override slot was added in PR #36 — never exposed because v0.8.10/v0.8.11 silently fell back to the canonical full DMG (which uses the bundled slot, which IS shaped `rapid-mlx/bin/rapid-mlx` under `Contents/Resources/`). v0.8.12 was the first release where the slim DMG actually went live on `latest.json` (see [0.8.12] below), making this the first release where the wrong path was actually consulted at runtime.

The locator returning nil cascades through `ServerManager.refreshBinary()` (called from the `BootstrapGateView.onInstalled` callback) → `binaryPath = nil` → `state = .missing` → `ContentView` renders `missingOverlay` ("Setup didn't finish" / "Quit Rapid-MLX" / "Recheck"). The Recheck button re-runs `find()` against the same wrong path, so the dialog is a permanent dead-end for end users. Existing v0.8.x bundled-DMG users are unaffected — they keep hitting the bundled slot, which has always been wrapped correctly.

### Fixed

* **`ServerLocator` runtime-override path now matches the bootstrap install layout** ([#431](https://github.com/machinefi/rapid-desktop/pull/431), [#430](https://github.com/machinefi/rapid-desktop/issues/430)). Both the `find()` candidate (`Sources/Rapid/Server/ServerLocator.swift:80`) and the `classify()` slot-identifier (`:203`) move from `runtime-override/bin/rapid-mlx` to `runtime-override/rapid-mlx/bin/rapid-mlx`. The new shape is now symmetric with the bundled slot (`Contents/Resources/rapid-mlx/bin/rapid-mlx`) — both candidate paths end in `rapid-mlx/bin/rapid-mlx`, so a future fourth lookup slot stays consistent by construction.
* **Two new regression tests pin both directions of the fix** ([#431](https://github.com/machinefi/rapid-desktop/pull/431)). `Tests/RapidTests/ServerLocatorTests.swift` now asserts (a) a binary planted at the OLD flat path returns nil — locks the fix direction so any refactor that silently re-introduces the flat shape fails closed, and (b) a binary at the wrapped path resolves AND classifies back to `.runtimeOverride` — proves the locator and the install pipeline agree on the on-disk shape end-to-end.
* **Fixture-path sweep across the bootstrap + extractor test surface** ([#431](https://github.com/machinefi/rapid-desktop/pull/431)). `Tests/RapidTests/ServerLocatorTests.swift` (`rapidBinWins`, `overrideBeatsBundled`, `classifyMatchesPriorityChainSlot`), `Tests/RapidTests/BootstrapCoordinatorTests.swift` (default `makeCoordinator` extractorLayout + the 3 inline layouts at `L430/L867/L1314`), `Tests/RapidTests/BootstrapCoordinatorConcurrentInstallTests.swift` (3 inline layouts at `L259/L697/L840`), and `Tests/RapidTests/SidecarExtractorTests.swift` (happy-path `StubTarExtractor` map + 2 publish assertions) all planted their override-slot / extracted-tree fixture at the flat `bin/rapid-mlx` shape — agreeing with `ServerLocator`'s old wrong path AND failing to mirror what `scripts/build-sidecar-tarball.sh` actually produces. Fixtures and assertions updated to the wrapped `rapid-mlx/bin/rapid-mlx` shape so production code AND test scaffolding now agree on a single on-disk layout that matches reality.

### Internal

* `CFBundleShortVersionString` 0.8.12 → 0.8.13, `CFBundleVersion` 128 → 129.

## [0.8.12] — 2026-06-25

Release-pipeline patch. **Slim bootstrapper DMG is _actually_ back on the `latest.json` hot path.** v0.8.10/v0.8.11 silently fell back to the canonical full ~157 MB DMG; v0.8.12 ships the ~5–6 MB slim DMG to `dl.rapidmlx.com/latest.json` and the GitHub Release with both a stapled inner `.app` ticket and a stapled DMG-level ticket.

**Root cause** (took 4 wrong hypotheses + ~2 weeks to find — see [[gotcha_local_verify_artifact_before_blaming_external]] for the methodological lesson and [[project_rapid_desktop_slim_dmg_2step_udzo_2026-06-25]] for the technical lesson): `hdiutil create -srcfolder -fs HFS+ -format UDZO` (the 1-step pack form scripts/build-bootstrapper-dmg.sh used through v0.8.10/v0.8.11/v0.8.12-a/b/c) **deterministically produces a UDIF whose internal BLKX compressed-block table is unreadable** when the staged source contains a stapled slim `.app` (= strip + re-codesign + inline-notarise + `xcrun stapler staple` → ~5 MB). The koly trailer + XML offsets look structurally sound, but `hdiutil verify` rejects with `"corrupt image"` and Apple Notary then rejects with `"could not be extracted"` (faithfully reporting that Apple's extractor can't unzip the corrupt bytes either). The canonical full DMG (scripts/dmg.sh) escapes this because its 156 MB payload doesn't trip the same codepath.

**The fix**: scripts/build-bootstrapper-dmg.sh now uses a **2-step UDRW → UDZO pack** (Apple TN3119 / the same pattern `dmgbuild`, `create-dmg`, `pkgbuild` use for stapled `.app` DMGs). Create an empty UDRW (read-write) image at fixed 64 MB capacity, mount it, `cp -R` the staged `.app` + Applications symlink into the mount, detach, then `hdiutil convert -format UDZO` to recompress into the final UDZO DMG. This isolates the staged-tree-to-HFS+-layout step from the UDZO compression step, sidestepping the 1-step codepath that fails on stapled `.app` metadata. Verify-run 28196766421 confirmed end-to-end: Apple Accepted both the inline `.app` zip submit AND the outer DMG submit; `stapler staple` ran successfully on both.

### What changes for users

New downloads from `rapidmlx.com` / `dl.rapidmlx.com/latest.json` get the slim ~5–6 MB bootstrapper DMG. Both `spctl --type exec` (first-launch Gatekeeper) and `spctl --type open` (download-time Gatekeeper) read a stapled ticket — same trust posture as the canonical full DMG. First-launch sidecar + Quickstart model download UX (slice γ/δ/ε.1, shipped v0.8.6+) is unchanged. Already-installed v0.8.x users are unaffected — both the slim and full path land on the same installed app.

### Fixed

* **2-step UDRW → UDZO pack on the slim DMG path** ([#429](https://github.com/machinefi/rapid-desktop/pull/429), [#427](https://github.com/machinefi/rapid-desktop/issues/427)). `scripts/build-bootstrapper-dmg.sh` replaces `hdiutil create -srcfolder -fs HFS+ -format UDZO` (1-step) with `hdiutil create -size 64m -fs HFS+ -ov` (empty UDRW) → `hdiutil attach` → `cp -R` → `hdiutil detach` → `hdiutil convert -format UDZO` (2-step). The 1-step form fails BLKX-table validation on small stapled `.app` payloads; the 2-step form passes `hdiutil verify` cleanly and survives Apple Notary submission.
* **Inline `.app` notarisation via zip submission BEFORE the DMG wrap** ([#429](https://github.com/machinefi/rapid-desktop/pull/429)). After the strip + `xattr -cr` + re-codesign, `ditto -c -k --keepParent` the scratch `.app` into a temporary zip and submit via `scripts/notarize.sh` (`notarytool submit` only accepts `.zip / .dmg / .pkg`). On success, `xcrun stapler validate` defensively asserts the ticket persisted before the DMG wrap. The outer `Notarise + staple bootstrapper DMG` step then adds a DMG-level ticket on top (belt-and-braces — either ticket alone satisfies Gatekeeper).
* **DMG envelope codesign stays dropped on the slim path** ([#429](https://github.com/machinefi/rapid-desktop/pull/429)). The original v0.8.10 corruption signal that prompted this investigation. Canonical full DMG (`scripts/dmg.sh`) keeps envelope codesign because the ~156 MB payload absorbs the trailing CMS blob; on ~5 MB UDZO the CMS append overlaps the koly trailer and corrupts the image. The outer slim notarise relies on `NOTARYTOOL_FORCE=1` to bypass notarytool's local validator (which gates on envelope codesign presence); Apple's server then runs the full pipeline on the cleanly-packed bytes and accepts.
* **`--deep` removed from the re-codesign** ([#429](https://github.com/machinefi/rapid-desktop/pull/429)). v0.8.11 added it defensively; the rapid-desktop `.app` has no nested signed Mach-Os post-strip (`Contents/Resources/rapid-mlx/` is the only nested signed tree, and the strip removes it), so `--deep` was a no-op at best and a gratuitous departure from `scripts/build.sh`'s explicit "Apple discourages `--deep` for distribution signing" rationale at worst.
* **v0.8.11's `hdiutil verify` retry loop deleted** ([#429](https://github.com/machinefi/rapid-desktop/pull/429)). Misdiagnosed as a race; three back-to-back failures on identical bytes proved deterministic corruption. With the 2-step UDZO pack the underlying corruption is gone, no retry needed.

### Internal

* `CFBundleShortVersionString` 0.8.11 → 0.8.12, `CFBundleVersion` 127 → 128.
* `Tests/RapidTests/BootstrapperNotarizeIntegrationShapeTests.swift` + `Tests/RapidTests/BootstrapperDMGShapeTests.swift` updated to assert the lowercase-hyphenated `build/rapid-mlx-desktop-bootstrapper.dmg` path the workflow uses since v0.8.11 PR #428.

## [0.8.11] — 2026-06-25

Release-pipeline patch. **Slim bootstrapper DMG is back on the `latest.json` hot path.** v0.8.10 silently regressed at the publish gate: Apple Notary rejected the slim DMG with `Invalid` (per the `notarytool log` fetched on the v0.8.10 verify-run: "could not be extracted" + "no signed executables or bundles"), the `latest.json` publish step fell back to `slim_available=false`, and `dmg_url` quietly served the full 157 MB DMG to every new download for v0.8.10. v0.8.11 fixes the slim DMG build pipeline so Apple Notary returns `Accepted` and the 5–6 MB slim install path is the default again.

### What changes for users

**New downloads from `rapidmlx.com` / `dl.rapidmlx.com/latest.json` get the slim ~5–6 MB DMG again** (was silently the full ~157 MB DMG for v0.8.10). First-launch sidecar + Quickstart model download UX (slice γ/δ/ε.1, shipped v0.8.6+) is unchanged. Users who installed v0.8.10 are unaffected — both the slim and the full path land on the same installed app.

### Fixed

* **Slim DMG notarisation is `Accepted` again** ([#428](https://github.com/machinefi/rapid-desktop/pull/428), [#427](https://github.com/machinefi/rapid-desktop/issues/427)). Root cause: `scripts/build-bootstrapper-dmg.sh` copies the canonical `.app` (which has already been notarised + stapled), strips `Contents/Resources/rapid-mlx/`, and re-codesigns. The post-strip `.app` was inheriting the canonical pipeline's `com.apple.notary.ticket` extended attribute, whose embedded CodeDirectory hash referenced the pre-strip bundle layout — Apple Notary's extractor saw the hash mismatch and rejected the archive as un-extractable. Fix: `xattr -cr "$SCRATCH_APP"` after the sidecar strip, before the re-codesign. Added `--deep` to both the ad-hoc and Developer ID branches of the re-codesign as defensive coverage for any future nested signed Mach-Os.
* **Slim DMG renamed to `rapid-mlx-desktop-bootstrapper.dmg`** (was `Rapid-MLX Desktop-bootstrapper.dmg`). Eliminated as the direct cause of the extractor failure but kept as a hygiene improvement — lowercase-hyphenated DMG filenames sidestep a class of pipeline-tool brittleness around shell-quoted spaces.
* **`scripts/notarize.sh` now parses the `--output-format json` `status` field directly** instead of trusting `notarytool`'s exit code, which is `0` on `Invalid` (the exit code reflects whether the *upload* succeeded, not the notarisation verdict — Apple quirk that was masking the v0.8.10 regression). On failure, auto-fetches `notarytool log <id>` so the operator sees Apple's actual reason without manual re-query. Adds `NOTARYTOOL_FORCE=1` env flag (skips notarytool's local UDIF/zip pre-validator; Apple's server still runs the full pipeline) for the slim DMG path, which `notarytool`'s local validator rejects with "must be a zip archive, flat installer package, or UDIF disk image" even though `hdiutil verify` accepts the same bytes.
* **`scripts/build-bootstrapper-dmg.sh` hdiutil verify retry loop** (3 attempts, 2 s backoff) after the DMG envelope codesign. Defensive against an "image not recognized" race observed once between the in-place trailer rewrite and the immediate `hdiutil verify`.

### Internal

* `CFBundleShortVersionString` 0.8.10 → 0.8.11, `CFBundleVersion` 126 → 127.

## [0.8.10] — 2026-06-25

Sidecar isolation + setup-UX cleanup. The slim bootstrapper DMG (v0.8.9 ε.2) made `BootstrapCoordinator` the canonical install path for the rapid-mlx sidecar; this release closes the Phase 1 legacy escape hatch that let a user-installed sibling `rapid-mlx` on `PATH` / homebrew / pipx / uv silently shadow whichever sidecar the desktop release pipeline shipped — a path that would silently lie about "Rapid · up to date" because the desktop and the shadowing CLI versions could drift independently.

### What changes for users

**Spawn behaviour**: Rapid-MLX Desktop now executes ONLY the sidecar this release shipped (or, for power users, an explicit `RAPID_BIN` override). If you previously had `brew install rapid-mlx` on disk, the desktop no longer routes through it — it routes through the sidecar in `~/Library/Application Support/Rapid/runtime-override/` (populated by the bootstrapper on first launch) or, for full-bundle DMGs, the copy inside `.app/Contents/Resources/rapid-mlx/`. You can keep the brew install on disk for terminal use; the desktop just doesn't depend on it any more.

**Sidecar path no longer shown in the bottom bar**. The raw `/opt/homebrew/Cellar/…` (or `~/Library/Application Support/Rapid/…`) string that used to render as a tertiary monospace tail next to the version pill was both crowding the version pill on long paths and surfacing an implementation detail every chat session. The path moves to the About panel (Rapid-MLX → About), where it stays accessible to anyone who actually needs to inspect it.

**"Set up Rapid-MLX" overlay**: when the engine isn't installed (bootstrapper hadn't run, install was interrupted, the runtime-override tree was manually deleted), the overlay copy now reads "Setup didn't finish — Quit Rapid-MLX and reopen to re-run the one-time setup" instead of the previous "Install rapid-mlx via brew" button. Reopening the app re-enters `BootstrapCoordinator` and downloads the sidecar from `dl.rapidmlx.com`. The brew-install button surfaced an install path the v0.8.9 cutover intentionally retired.

### Changed

* **`ServerLocator.find()` priority chain collapsed from 6 slots to 3.** Was `RAPID_BIN → runtime-override → bundled → PATH → /opt/homebrew/bin → /usr/local/bin → ~/.local/bin`; now `RAPID_BIN → runtime-override → bundled` only. When all three miss, returns `nil` and callers surface the missing-install UX so `BootstrapCoordinator` can re-run, rather than scavenging for a sibling rapid-mlx of unknown provenance.
* **`Sources/Rapid/Server/BrewDetector.swift` deleted.** Its only caller was the `missingOverlay` brew-install button, which is no longer wired up.
* **`Sources/Rapid/Server/InstallScript.swift` deleted.** Same reason — it generated the `brew install raullenchai/rapid-mlx/rapid-mlx` `.command` script that the missingOverlay primary button used to write to disk and hand to Terminal. The bootstrapper-driven install path replaces it end-to-end.
* **`ContentView.missingOverlay` reworked.** Headline `"Setup didn't finish"`, primary button `"Quit Rapid-MLX"`, secondary `"Recheck"`. No brew detection, no install one-liner, no `brew.sh` link.
* **`ContentView.statusFooter`** no longer renders `server.binaryPath` as a tertiary monospace tail.

### Internal

* **Test invariants flipped to match the 3-slot chain.** `ServerLocatorPriorityTests.bundledResolvesWhenOnlyBundled` (was `bundledBeatsLegacyChain`) drops the PATH-plant fixture because the locator no longer consults PATH. `ServerLocatorPriorityTests.pathInstallIsIgnored` (was `emptyBundleFallsThroughToPath`) flips the Phase 1 fall-through guarantee — a binary planted on PATH must NOT resolve when override + bundle are empty. New `ServerLocatorPriorityTests.fixedFallbackPathsIgnored` exercises the `~/.local/bin/rapid-mlx` fixed slot the locator used to walk. `ServerLocatorPriorityTests.allSlotsEmptyReturnsNil` (was `allCustomSlotsEmptyClassifiesFallback`) asserts `nil` directly instead of the previous "any host-Cask fallback OR nil" hedge that was forced by the host's real brew install bleeding into the test.
* **`Tests/RapidTests/InstallScriptTests.swift` deleted** alongside the source file. `ApplicationSupportLocatorTests.allConsumersUseLocator` drops the `InstallScript.swift` entry from its source-introspection expected-references list.
* **`ServerLocator.classify()` left intact.** Its `.homebrewAppleSilicon` / `.homebrewIntel` / `.pipx` / `.uv` / `.path` branches become unreachable from `find()` but stay useful as diagnostic labels when a power user points `RAPID_BIN` at a binary that happens to live in one of those locations.
* **No `DownloadManager` brew-install code paths existed** in the first place — `augmentedEnv` only appends `/opt/homebrew/bin` to the subprocess `PATH` so spawned `rapid-mlx pull` children can locate Git/openssl/etc. Left untouched.
* `CFBundleShortVersionString` 0.8.9 → 0.8.10, `CFBundleVersion` 125 → 126.

## [0.8.9] — 2026-06-25

P3 cutover slice ε.2 — `latest.json.dmg_url` now points at the slim bootstrapper DMG (~5.6 MB) instead of the canonical full DMG (~157 MB). **New users downloading from rapidmlx.com get a 96% smaller initial download.** The bootstrapper splash then pulls the sidecar tarball (~126 MB) + Quickstart `qwen3-0.6b-4bit` model tarball (~307 MB) from `dl.rapidmlx.com` on first launch with progress / resumable / cancellable / retry-on-failure UX (slice γ delivered the BootstrapCoordinator state machine; slice δ added the R2 mirror; slice ε.1 added notarize + R2 publish; this slice flips the manifest pointer so they all become user-facing).

### What changes for users

**New install (rapidmlx.com / direct download)**: download is now ~5.6 MB instead of ~157 MB. First launch shows a splash window with progress bars for the sidecar + model download; transitions to the chat surface when both complete. Subsequent launches skip the splash entirely (sidecar + model live in `~/Library/Application Support/Rapid/` once installed). Same end-state as the bundled DMG; just network-deferred.

**Already-installed v0.8.x users** (in-app UpdateChecker): the auto-update download is also now ~5.6 MB instead of ~157 MB. After install, the bootstrapper splash detects the existing `~/Library/Application Support/Rapid/` install (sidecar + model from any prior bundled-DMG version are preserved across the upgrade) and skips the download, transitioning to chat immediately — same launch time as before. If the prior install was bundle-only (sidecar inside `.app/Contents/Resources/`, never copied to Application Support — e.g., a fresh user who never touched the bootstrapper), the bootstrapper splash runs the full download on this one upgrade and then every subsequent patch is the 5.6 MB shape.

**Net win across the v0.8.x → v1.0 cadence**: every patch release after this one is ~152 MB smaller for everyone. The one-time upgrade cost for bundle-only users is the same bytes they would have downloaded anyway via the full DMG.

### Changed

* **`latest.json.dmg_url` flips to slim bootstrapper DMG** (P3 cutover slice ε.2). New `Pre-publish slim bootstrapper DMG to dl.rapidmlx.com (R2)` step in `.github/workflows/release.yml` mirrors the notarised + stapled slim DMG to `rapid-mlx-desktop-bootstrapper-<VERSION>.dmg` (versioned, immutable) + `rapid-mlx-desktop-bootstrapper.dmg` (unversioned alias) BEFORE the canonical `Mirror DMG + publish latest.json` step composes the manifest — atomicity invariant (same as the sidecar + model legs from slice γ/δ). The canonical mirror step now reads `${{ steps.slim_prepublish.outputs.* }}` to set `LATEST_DMG_KEY` / `LATEST_DMG_SHA256` / `LATEST_DMG_SIZE` shell vars; the `jq` invocation that writes `dmg_url` / `dmg_sha256` / `dmg_size` into `latest.json` uses these conditional vars. The unversioned canonical alias (`rapid-mlx-desktop.dmg`) still mirrors the full DMG for direct-URL bookmarks; only `latest.json` (which is what `rapidmlx.com/api/desktop-download` and the in-app UpdateChecker both read) flips.

* **GH Release attach step renamed** from `(P3 slice ε.1 — preview asset, dormant)` to `(P3 slice ε.2 — load-bearing)`. The slim DMG remains attached to every GH Release as a discoverable asset alongside the canonical full DMG.

* **Release-pipeline-safety fallback** preserved: when the slim DMG is missing on disk (slice α build skipped / failed), is not stapled (slice ε.1 notarise step failed), or either wrangler put fails / times out, the pre-publish step emits `slim_available=false` and the canonical mirror step falls back to writing `dmg_url` for the full DMG — preserving release-pipeline integrity. A failure of slim ε.2 turns ε.2 off for that one release; it never breaks the release pipeline.

### Internal

* **3 flipped test invariants** + **1 new isolation invariant**. `BootstrapperDMGShapeTests.releaseYamlFlipsLatestJsonDmgUrlToSlimWithCanonicalFallback` (was `releaseYamlDoesNotChangeLatestJsonDmgUrlComposition`) pins the new `${LATEST_DMG_KEY}` conditional shape + the fallback shell vars. `BootstrapperDMGShapeTests.releaseYamlOrdersSlimPrepublishBeforeCanonicalLatestJson` (was `releaseYamlOrdersSlimR2PublishAfterCanonicalLatestJson`) flips the ordering invariant (slim BEFORE canonical, for atomicity). `BootstrapperNotarizeIntegrationShapeTests.sliceEpsilon2KeepsLatestJsonSchemaUnchanged` (was `sliceEpsilon1DoesNotChangeLatestJsonSchema`) keeps the schema_version=1 + no-new-keys invariants. `BootstrapperNotarizeIntegrationShapeTests.slimR2PrepublishIsADedicatedStepWithTimeoutAndContinueOnError` (was `slimR2PublishIsADedicatedStepWith…`) pins the `id: slim_prepublish` + `timeout-minutes` + `continue-on-error: true` + `!cancelled()` + tag-gate isolation shape. `slimR2PrepublishStepLivesBeforeCanonicalLatestJson` (was `slimR2StepLivesAfterCanonicalLatestJson`) flips the ordering pin sibling.

* **No new runtime dependencies. No BootstrapCoordinator / UpdateChecker / SplashView changes** — slice γ's state machine (download / verify / install / retry / resume / cancel) was already user-facing for slim-DMG installs and is now exercised by every install path. Standard patch version bump: `CFBundleShortVersionString` 0.8.8 → 0.8.9, `CFBundleVersion` 124 → 125.

## [0.8.8] — 2026-06-24

Patch release. **Zero user-visible behavior change for installed v0.8.x users.** Fixes a single P0 in the slim-bootstrapper-DMG first-launch path discovered by the v0.8.7 release-day dogfood agent ([#414](https://github.com/machinefi/rapid-desktop/issues/414), F-DGF-V087-01). v0.8.6's P3 slice γ landed concurrent sidecar + Quickstart model download but never closed the loop between the on-disk model install root (`~/Library/Application Support/Rapid/quickstart-models/<alias>/`) and the `rapid-mlx ls` enumeration path that `AutoStartDecision` reads to decide `.start` vs `.promptDownload`. `rapid-mlx ls` walks only the HuggingFace Hub cache (`<HF_HUB_CACHE>/models--*/snapshots/`); the Quickstart install was invisible to it. Slim-DMG users consequently completed the bootstrapper splash, saw the model on disk, and were then asked to download it again. Bundled-DMG users (which is everybody today via the `latest.json.dmg_url` cutover still pending in slice ε.2) were never reachable to this bug because they take the `.installed(.bundled)` short-circuit AND the bundled snapshot is laid down in HF Hub cache layout end-to-end.

### Fixed

* **Slim-DMG Quickstart install now resolves through `rapid-mlx ls`** ([#414](https://github.com/machinefi/rapid-desktop/issues/414)). New `Sources/Rapid/Server/QuickstartModel.swift` fabricates a HuggingFace Hub cache stub (`<HF_HUB_CACHE>/models--<owner>--<name>/refs/main` + `snapshots/quickstart/<file>` symlinks pointing back into the flat Quickstart install dir) on every launch + on the BootstrapCoordinator post-commit hook, mirroring the idempotent-relink shape `BundledModel.installBundledSnapshotSymlink` already uses for the bundled snapshot. `AutoStartDecision.decide` now sees the alias as cached and returns `.start`, the sidecar auto-spawns on first launch, and the slim-DMG happy path matches the bundled-DMG happy path bit-for-bit (smooth download/install + Quickstart model + start running inference, no manual re-download prompt). Stub-vs-real-cache discrimination via the unique combination `refs/main` content (40-char hex SHA from real HF Hub vs literal `"quickstart"` from our stub); a user's pre-existing real download always wins. Stub is idempotent across launches (intact stub returns `.alreadyPresent` without touching disk), self-healing across Quickstart re-extractions (drifted leaf symlinks trigger a stub-only rebuild that never touches adjacent caches), and resilient to the F-DGF-V087-03 tarball-strip bug (probes both flat `<root>/<alias>/<files>` and nested `<root>/<alias>/<alias>/<files>` layouts via `config.json` sentinel). 16 new `QuickstartModelTests` (including two codex-r1 regression pins against the cache-dir-symlink data-loss hazard). Wired into `ContentView.task` (alongside the existing `BundledModel.installBundledSnapshotSymlink` call) and into `BootstrapCoordinator` right after `modelInstaller.commit` succeeds so the stub is live before the splash transitions to `.installed`.

### Internal

* No release-pipeline / CI changes. No new runtime dependencies. The fix is desktop-side only — rapid-mlx's HF-cache scan remains the single source of truth across every downstream tool (LM Studio export, HF Hub web links, sidecar's own `snapshot_download` resolution); we just make the Quickstart install LOOK like an HF Hub cache entry, the same trick `BundledModel` already uses. Standard patch version bump: `CFBundleShortVersionString` 0.8.7 → 0.8.8, `CFBundleVersion` 123 → 124.

## [0.8.7] — 2026-06-24

Patch release. **Zero user-visible behavior change for installed v0.8.x users.** Fixes a single P0 in the slim-bootstrapper-DMG publish pipeline introduced by v0.8.6's rehearsal: `latest.json.sidecar_version` shipped as a 7-character git short SHA (`26ac5b4`) instead of a SemVer-shaped string (`0.8.18`). The bootstrapper's defensive manifest validator (added in #403, on purpose, to prevent path-injection via the on-disk `VERSION` marker) correctly rejected the bad value — but that meant 100% of slim-DMG installs hard-failed at the "Setup didn't finish" splash card with no retry path. Bundled-DMG users (which is everybody today, including the rapidmlx.com/desktop CTA — slice ε.2 is still dormant) were never reachable to this bug because they take the `.installed(.bundled)` short-circuit; the bootstrapper splash never fires. Discovered by the v0.8.6 release-day dogfood agent ([#411](https://github.com/machinefi/rapid-desktop/issues/411)).

### Fixed

* **`latest.json.sidecar_version` is now strictly dotted-digit end-to-end** ([#411](https://github.com/machinefi/rapid-desktop/issues/411)). Root cause: `actions/checkout`'s submodule pull only fetches the pinned SHA, not the tag list, so `git describe --tags --always` in `scripts/build.sh` fell back to `--always` and produced a SHA. That SHA propagated into the bundled `Contents/Resources/rapid-mlx/VERSION` → into `scripts/build-sidecar-tarball.sh`'s manifest emit → into `latest.json`. Fixed at four layers: (1) **`release.yml`** now explicitly fetches submodule tags after checkout (`Fetch submodule tags (#411)` step). (2) **`scripts/build.sh`** derives the bundled VERSION from `git tag --points-at HEAD --list 'v[0-9]*'` first (works for both lightweight and annotated tags — relevant because rapid-mlx's v0.8.15/v0.8.16/v0.8.18 are lightweight), strips the leading `v`, falls back to `git describe --tags --abbrev=0 --match 'v[0-9]*'`, hard-fails if neither yields a dotted-digit value. (3) **`scripts/build-sidecar-tarball.sh`** refuses to emit a manifest with a non-dotted-digit `sidecar_version` (defense-in-depth — the floor under any upstream regression). (4) **`release.yml`** at `latest.json` compose-time runs the same regex against the value read from the manifest BEFORE upload, AND after publish reads the object directly from R2 origin via `wrangler r2 object get` and re-validates — hard-fails the release on mismatch (not warn), because every minute a bad manifest is live = more bricked bootstrapper installs. All four regexes are strictly `^[0-9]+(\.[0-9]+)+$` (no pre-release / build suffix), matching `BootstrapCoordinator.isValidVersionString` exactly so a future rapid-mlx tag like `v0.8.19-rc.1` is rejected at build time instead of bricking installs at runtime. R2-origin authoritative check (not CDN curl) so a 300-second Cloudflare cache window can never false-fail a good release.

### Internal

* No source-layer changes beyond the four-layer regex gate described above; all P3 plumbing from v0.8.6 remains dormant exactly as shipped. Slice ε.2 cutover is gated on the v0.8.7 slim-DMG dogfood passing.
* **Bundle size delta gate now filters previous-release DMG by exact canonical name.** The first v0.8.7 release attempt failed at the "Bundle size delta gate" step because v0.8.6 introduced a second `.dmg` asset on the GH Release (the slim bootstrapper DMG, ~5.6 MB) and the legacy `select(.name | endswith(".dmg")) | head -1` filter picked it non-deterministically — making v0.8.7's 166 MB full DMG appear to grow by +160 MB vs the previous release. The fix pins the filter to `select(.name == "rapid-mlx-desktop.dmg")` so any future preview/variant DMG cannot be picked by mistake. Regression pin: `Tests/RapidTests/DeltaGateAssetSelectionTests.swift`. Release-pipeline fix only — no runtime / shipped-binary impact.

## [0.8.6] — 2026-06-24

Patch release. **Zero user-visible behavior change for installed v0.8.x users** — this release is a real-CI rehearsal of the bootstrapper P3 plumbing that landed in main since v0.8.5 (seven merged PRs: two P3 follow-up bug fixes from v0.8.5 dogfood, plus five slices of the bootstrapper-DMG architecture). The slim "bootstrapper-only" DMG is produced, notarized, stapled, and mirrored to `dl.rapidmlx.com/rapid-mlx-desktop-bootstrapper-0.8.6.dmg` for the first time, but `latest.json.dmg_url` still resolves to the full bundled DMG — so existing in-app `UpdateChecker` clients continue to update to the full DMG and rapidmlx.com/desktop CTA continues to serve the full DMG. The cutover flip (ε.2) and old-Quickstart cleanup (P4) and Sparkle migration (P5) ship in a future release once this rehearsal validates the pipeline.

### Fixed

* **Bootstrapper `VERSION` marker now records sidecar package version, not desktop manifest version** ([#403](https://github.com/machinefi/rapid-desktop/pull/403), closes [#400](https://github.com/machinefi/rapid-desktop/issues/400)). After first install the file at `~/Library/Application Support/Rapid/runtime-override/VERSION` previously held the desktop version (e.g. `0.8.5`), while the bundled marker held the sidecar package version (e.g. `0.8.18`) — two semantically different versions sharing the same `VERSION` filename. v0.8.5 dogfood Agent A flagged this as confusing for users who screenshot the status bar to file a bug, and breaking for any future desktop release that ships a newer sidecar but same desktop version (marker would compare equal → no re-install). Fixed via Option B (`sidecar_version` threaded through `BootstrapManifest` + `latest.json`) rather than Option A (rename). Decoded manifest now exposes `sidecar_version` explicitly; coordinator re-validates the marker against the fetched manifest on every detect pass. Four codex rounds caught a stale-marker re-install bypass + a manifest-validation-bypass on the re-install path. Dormant for v0.8.x users (`.installed(.bundled)` short-circuit).
* **Bootstrapper archives stale `sessions.json` on genuinely-first install** ([#402](https://github.com/machinefi/rapid-desktop/pull/402), closes [#401](https://github.com/machinefi/rapid-desktop/issues/401)). `~/Library/Application Support/Rapid/sessions.json` lives at the *parent* of `runtime-override/`, so it survives sidecar reinstalls. v0.8.5 dogfood Agent A observed a previously-deleted user's chat resurrected into the very first ChatView a future bootstrapper-DMG user would see — wrong first impression + privacy concern. Fixed via predicate captured BEFORE install starts (`isFirstEverInstall = !runtimeOverrideExisted && !bundledMarkerExisted`); at `.installing → .installed` transition, if the captured flag is true, sessions.json is renamed to `sessions.<ISO8601-timestamp>.bak.json` (archive, not delete, for forensic recovery). Six codex rounds covered: ordering of capture, backfill if predicate raced, manifest-validate gate on re-install, fixture + sanity gate, residual-risk acceptance with test pins, fileExists predicate hardening for corrupt `VERSION` files. 27 new tests across `FirstInstallSessionReset` + `BootstrapCoordinator` wire-up.

### Internal

* **Bootstrapper P3 plumbing landed end-to-end (five slices, all DORMANT).** Together these wire the slim-bootstrapper-DMG production + Quickstart model tarball + concurrent install + R2 publish pipeline. **None are reachable for v0.8.x users** (bundled-VERSION short-circuit means splash never fires); they are reachable only by the bootstrapper splash path that today's release does not ship. The `dmg_url` in `latest.json` continues to resolve to the full bundled DMG.
  * **P3 slice α — slim DMG as CI workflow artifact** ([#404](https://github.com/machinefi/rapid-desktop/pull/404)). New `scripts/build-bootstrapper-dmg.sh` (373 lines, strict-mode bash) strips `Contents/Resources/rapid-mlx/` from a `BUNDLE_MODEL=0` build, re-ad-hoc-codesigns, wraps in DMG. Size sanity gates: fails if output > 50 MB (target: 5–8 MB) or < 1 MB. Local-build size: **6 MB**. Mach-O sanity: `codesign -v --deep --strict` exit 0. New `release.yml` step `Build bootstrapper DMG (artifact-only)` runs after the existing DMG build, uploads the slim DMG as a 90-day GHA workflow artifact, `continue-on-error: true` so it cannot block the main release. 3 codex rounds. 15 new `BootstrapperDMGShape` tests.
  * **P3 slice β — Quickstart model tarball as CI workflow artifact** ([#405](https://github.com/machinefi/rapid-desktop/pull/405)). New `scripts/build-model-tarball.sh` packages the `qwen3-0.6b-4bit` MLX snapshot (~293 MB compressed / ~351 MB uncompressed, 11 files) into a deterministic gzip-tar. SOURCE_DATE_EPOCH-pinned mtime, uid/gid 0, sorted entries, gzip mtime=0, `tarfile` + `gzip` via Python (not bsdtar — `pax:exthdr.mtime` unsupported). **Byte-identical re-runs verified across two back-to-back local builds AND across different `--version` arguments** (content-derived `created_at_epoch` → same SHA regardless of when packaged). Manifest sidecar JSON carries `tarball_sha256`/`tarball_size`/`uncompressed_size`/`file_count`. New `release.yml` step `Build Quickstart model tarball (artifact-only)`, again `continue-on-error: true`, uploads the tarball + manifest as a 90-day GHA workflow artifact. Reads from the HF cache, **does not write** to it. 5 codex rounds. 25 new `ModelTarballShape` tests.
  * **P3 slice γ — `BootstrapCoordinator` concurrent sidecar + Quickstart model download** ([#407](https://github.com/machinefi/rapid-desktop/pull/407)). `BootstrapManifest` extended with four OPTIONAL fields (`model_url` / `model_sha256` / `model_size` / `model_alias`); partial sets rejected via `validateModelFields()`. New `ModelInstaller` actor mirrors `BootstrapInstaller` (stage / commit / rollbackStaging) for the model artifact. Install pipeline rewritten as `withThrowingTaskGroup` with `(URL, URL?)` return shape; combined progress weighted by byte total; cancellation propagates to both child tasks; atomic-or-nothing on commit (sidecar success + model failure → both rolled back; vice-versa). `SplashView` gains `CombinedProgress` with weighted byte-count aggregation + dual-line detail string. **Graceful fallback**: when manifest lacks model fields (today's production shape), the coordinator behaves bit-for-bit identically to today's sidecar-only path. 6 codex rounds caught: atomic-or-nothing on commit; model staging leak on revert; revert-must-verify-disk-truth; introduce `ModelInstallerProtocol` seam to deterministically reach the post-publish revert path. 46 new tests across coordinator + installer + splash view-model.
  * **P3 slice δ — R2 mirror publishes model tarball + `latest.json` schema +4 optional `model_*` fields** ([#408](https://github.com/machinefi/rapid-desktop/pull/408)). `release.yml` R2 mirror block extended to push `quickstart-qwen3-0.6b-4bit-X.Y.Z.tar.gz` + `.manifest.json` after the existing sidecar-tarball R2 puts. `latest.json` composition gains four additive optional fields (`model_url`, `model_sha256`, `model_size`, `model_alias`); `schema_version` STAYS at `1` (v0.8.x `UpdateChecker.swift` Codable tolerates the additive fields via Optional-default-nil — verified by `releaseDecodesWithoutBootstrapperFields` test). Atomicity invariant preserved (latest.json composition is the last R2 put; never references a key that didn't upload). 3 codex rounds. 16 new schema/Codable tests.
  * **P3 slice ε.1 — slim DMG notarized + R2-published + GH Release preview asset** ([#409](https://github.com/machinefi/rapid-desktop/pull/409)). `release.yml` invokes `scripts/notarize.sh` against the slim DMG, staples, then R2-publishes as `rapid-mlx-desktop-bootstrapper-X.Y.Z.dmg` (versioned) + `rapid-mlx-desktop-bootstrapper.dmg` (unversioned alias). GH Release gains preview asset of the same name. Every R2 put gated behind `xcrun stapler validate` so an unstapled DMG can never reach `dl.rapidmlx.com`. **Dormant**: `latest.json.dmg_url` composition UNCHANGED (still resolves to the full bundled DMG); the slim DMG is published-and-discoverable but not the primary release asset. 5 codex rounds. 12 new `BootstrapperNotarizeIntegrationShape` tests + 5 net extended `BootstrapperDMGShape` tests (two old "does NOT notarise/mirror" negative-assertions replaced by ε.1-positive assertions).

### Known issues

* **ε.2 cutover flip pending.** The 1-line `release.yml` change that flips `latest.json.dmg_url` from the full DMG to the slim DMG ships in the next release once this v0.8.6 rehearsal validates the slim-DMG end-to-end. Until then, every release continues to ship both DMGs, with the full DMG as the primary asset.
* **P4 old-Quickstart cleanup pending.** `Sources/Rapid/UI/QuickstartView.swift`, `QuickstartCoordinator`, the `ModelPickerBar` Quickstart section, and the first-chat model-download path in `ChatView` all remain in main. They are still load-bearing for current v0.8.x bundled-DMG users; they get cleaned up only after ε.2 ships and the bootstrapper splash becomes the default first-launch path.
* **P5 Sparkle migration pending.** v0.8.x bundled-DMG users currently rely on in-app `UpdateChecker` polling `latest.json` and offering the DMG as a manual download. Sparkle integration (or a hand-rolled atomic-app-replace) ships in a future patch so existing users get a seamless one-click upgrade to the bootstrapper DMG once ε.2 ships.

## [0.8.5] — 2026-06-24

Patch release. Bundled-sidecar bump from rapid-mlx v0.8.14 → **v0.8.18** carrying 26 user-facing PRs across three sidecar release waves (v0.8.15, v0.8.16, v0.8.18 — v0.8.17 was skipped upstream), plus seven user-visible desktop polish/reliability fixes (sticker label differentiation, disk banner MB rendering, `mailto:` links in chat content, AutoStartDecision cardinality contract, test temp-dir leak teardown, two doc-comment audits), the byte-count `Int64 → UInt64` polish on the dormant bootstrapper module, and five merged-but-dormant bootstrapper P2 slices that ship the future bootstrapper-DMG download/install/extract/launch path without altering existing-user behavior (bundled-VERSION short-circuit means the splash path never fires on this release).

### Fixed

* **Quality sticker — `.small` bucket now renders `· small` instead of `· tiny`** ([#391](https://github.com/machinefi/rapid-desktop/pull/391), closes [#348](https://github.com/machinefi/rapid-desktop/issues/348)). `QualityBucket.tiny` and `QualityBucket.small` exist as distinct entries in the data model (different empirical failure modes — sub-1B first-impression risk vs 1-3B cycle-9 schema-leak) but the picker sticker collapsed them under a single `· tiny` suffix (visual theatre). Option A (split): `.small` now renders `· small`. Symmetrical split applied to `aliasRowAccessibilityLabel` so VoiceOver users get the same distinction.
* **Quickstart disk-low banner — render MB for free space < 1 GB instead of `0.1 GB`** ([#393](https://github.com/machinefi/rapid-desktop/pull/393), closes [#357](https://github.com/machinefi/rapid-desktop/issues/357)). Hand-rolled byte-formatter (locale-independent, deterministic across CI locales) with `%.0f` MB / floored GB branches. Round-1 codex caught a pathology where the MB branch's `%.0f` would round `1 GiB - 1` byte to `1024 MB` (visually crossing the 1 GB cutoff); fixed by integer division (floor) + added `gb - 1 == "1023 MB"` boundary pin.
* **Allow `mailto:` links in chat markdown content** ([#394](https://github.com/machinefi/rapid-desktop/pull/394), closes [#349](https://github.com/machinefi/rapid-desktop/issues/349)). `ChatLinkSafety` allowlist widened by exactly one scheme: `{http, https, mailto}`. `mailto:` opens the system compose window — no FS read, no auto-send — strictly less dangerous than `file://` or `javascript:` which remain rejected, along with `tel:`, `sms:`, `facetime:`, `slack:`, `vscode:`, `zoomus:`, `obsidian:`, `raycast:`, scheme-less.

### Changed

* **Bundled sidecar bumped to rapid-mlx v0.8.18** (submodule `4b253a4 → 26ac5b4`). 29 commits / 26 user-facing PRs spanning three rapid-mlx release waves now ship in the bundled sidecar (chronologically, by wave):
  * **v0.8.15** (raullenchai/Rapid-MLX#876, 2026-06-24) — IDE bootstrap + reasoning/tool-call hardening: `rapid-mlx launch <client>` one-shot IDE bootstrap subcommand ([rapid-mlx #870](https://github.com/raullenchai/Rapid-MLX/pull/870), closes upstream #566); chat path always emits the `content` key on assistant messages, even when the model produced `reasoning_content` only — D-MISSING-CONTENT-KEY ([rapid-mlx #872](https://github.com/raullenchai/Rapid-MLX/pull/872)); strict JSON-schema mode enforced via post-generate validation + repair retry — H-06 ([rapid-mlx #873](https://github.com/raullenchai/Rapid-MLX/pull/873), closes upstream #423); `deepseek_v3` tool-parser variant for the R1-0528 family — D-DSV31 ([rapid-mlx #874](https://github.com/raullenchai/Rapid-MLX/pull/874)); reasoning rescue tail to content on `finish_reason=length` — H-01 ([rapid-mlx #875](https://github.com/raullenchai/Rapid-MLX/pull/875), closes 8-round D-carry upstream #259).
  * **v0.8.16** (raullenchai/Rapid-MLX#883, 2026-06-24) — responses/models/anthropic surfaces: `chat_template_kwargs` plumbed through `ResponsesRequest` so `enable_thinking=true/false` flows end-to-end on `/v1/responses` — M-2 ([rapid-mlx #877](https://github.com/raullenchai/Rapid-MLX/pull/877)); `/v1/models` now surfaces effective `tool_call_parser` + `reasoning_parser` per alias — V-1, S-2 ([rapid-mlx #878](https://github.com/raullenchai/Rapid-MLX/pull/878)); prefix-cache atomic SIGTERM save commit — T-1 ([rapid-mlx #880](https://github.com/raullenchai/Rapid-MLX/pull/880)); `reasoning_content` sanitized on `tool_choice=required` path — V-2 ([rapid-mlx #881](https://github.com/raullenchai/Rapid-MLX/pull/881)); `/v1/messages` Anthropic surface sanitizes `<think>` + dedupes the rescue tail — M-1b ([rapid-mlx #882](https://github.com/raullenchai/Rapid-MLX/pull/882)).
  * **v0.8.18** (raullenchai/Rapid-MLX#900, 2026-06-24 — v0.8.17 was skipped upstream) — responses + tools + reasoning hardening + new aliases: `RAPID_MLX_STRICT_JSON_SCHEMA=off` env override honored on the non-guided path — H-06 ([rapid-mlx #884](https://github.com/raullenchai/Rapid-MLX/pull/884)); context-length re-check before strict_json_schema repair retry — H-06 ([rapid-mlx #886](https://github.com/raullenchai/Rapid-MLX/pull/886)); `deepseek_v3` parser misbind guard — refuses to bind to non-V3 Qwen2 distills, emits warning on misbind — S-1 ([rapid-mlx #887](https://github.com/raullenchai/Rapid-MLX/pull/887)); `managed_tempfile` helper applied to leak sites ([rapid-mlx #888](https://github.com/raullenchai/Rapid-MLX/pull/888), closes upstream #719); `/v1/responses` stream emits leading items before message — M-3 ordering ([rapid-mlx #889](https://github.com/raullenchai/Rapid-MLX/pull/889)); auto-disable thinking when tools provided ([rapid-mlx #891](https://github.com/raullenchai/Rapid-MLX/pull/891)); `_serve_audio_mode` honors `--embedding-model` + `--served-model-name` ([rapid-mlx #894](https://github.com/raullenchai/Rapid-MLX/pull/894), closes R11-K #258); auto-disable thinking on casual chat completions ([rapid-mlx #895](https://github.com/raullenchai/Rapid-MLX/pull/895)); tool_call promotion ported to refactored think_parser ([rapid-mlx #896](https://github.com/raullenchai/Rapid-MLX/pull/896), closes upstream #344) plus 5-BLOCKING codex follow-up r2-r5 ([rapid-mlx #898](https://github.com/raullenchai/Rapid-MLX/pull/898)); **Tmax-9B + Tmax-27B MLX aliases added** (mlx-community first-mover) ([rapid-mlx #899](https://github.com/raullenchai/Rapid-MLX/pull/899)).
  * Additional internal CI/quality PRs in this bump that are not user-visible: rapid-mlx #885 / #890 / #892 (pr_validate hardening), #893 (codex follow-ups for #588), #897 (audio AST contract tests).

### Internal

* **Bootstrapper P2 architecture closed — five merged-but-dormant slices.** The bootstrapper-DMG download/install/extract/launch path is now end-to-end functional in code, gated behind the bundled-VERSION marker so current users with the bundled sidecar take the `.installed(.bundled)` short-circuit straight to ChatView and never see the splash. P3 cutover (which will ship the bootstrapper DMG as the primary release asset) builds on this.
  * **SplashView + SplashViewModel** ([#371](https://github.com/machinefi/rapid-desktop/pull/371)). SwiftUI splash window: CheetahLogo + headline + linear progress + middle-truncated detail + Cancel. `@Observable` view model with `progress`/`headline`/`detail`/`cancellable` fields.
  * **ResumableDownloader** ([#372](https://github.com/machinefi/rapid-desktop/pull/372)). `URLSession`-based actor with HTTP Range header / `.partial` staging / atomic rename. 3 codex rounds caught: HTTP 206 without `Content-Range` validation (CDN could ack 206 then ship offset-0 bytes — strict `bytes <start>-…` parsing + `DownloadError.invalidContentRange`); continuation-attach race (settled-before-attach hung forever — lock-shared `attach()` resumes from `pendingResult`); swallowed write/seek/fsync errors (now propagate via `DownloadError.diskWriteFailed`); cancellation mapped from `NSURLErrorCancelled` to `CancellationError`; flush-order bug (`firstWriteError` sampled before fsync missed close failures).
  * **SHA256Verifier + BootstrapInstaller** ([#388](https://github.com/machinefi/rapid-desktop/pull/388)). Streaming SHA256 (1 MiB chunks, no full-file load) + atomic install via `FileManager.replaceItem` (APFS rename). 3 codex rounds caught: actor reentrancy on same destination (`inFlightDestinations` Set + `InstallError.alreadyInstalling`); disk-error context loss (`DiskFailureInfo` carries operation/path/domain/code/message); cancellation gate missing between verify + publish; NSError synthesis losing POSIX context.
  * **SidecarExtractor — tar + Mach-O sanity** ([#389](https://github.com/machinefi/rapid-desktop/pull/389)). `/usr/bin/bsdtar -xz` extraction + magic-bytes Mach-O detection (6 magics: 32/64-bit native+swapped + fat both) + `codesign -v` per-file sanity (chose `codesign -v` over `spctl -a -t exec` — the latter only validates `.app` bundles and fails on the ~76 standalone `.dylib`/`.so` files in the sidecar tree) + quarantine xattr strip with `XATTR_NOFOLLOW`. 4 codex rounds caught: cancellation theatre (`process.terminate()` not actually wired through `Task.cancellationHandler`); `removexattr` without `XATTR_NOFOLLOW` could clear quarantine from symlink targets; `codesign` had no timeout or stderr drain (deadlock risk).
  * **BootstrapCoordinator + root-view swap** ([#390](https://github.com/machinefi/rapid-desktop/pull/390)). `@Observable` actor composing `UpdateChecker → BootstrapInstaller → SidecarExtractor` with state machine `.checking → .installed (short-circuit) / .installing → .installed / .failed → retry`. Install destination = `~/Library/Application Support/Rapid/runtime-override/` (NOT `<App>/Contents/Resources/rapid-mlx/` — writing to bundle Resources would invalidate the codesign seal on a signed+notarized `.app`). Detect: runtime-override marker FIRST, bundled marker SECOND. `RapidApp.init()` continues to honor v0.8.2's `InitMustNotTouchNSAppTests` invariant (no `NSApp.*` in `App.init()`).
  * Verified end-to-end against the live `dl.rapidmlx.com` manifest by stripping the bundled sidecar from a dev `.app` build and watching splash → byte-counter progress → SHA256 verify → tar extract → root-view swap → ChatView. Quit + relaunch confirmed cached-path (runtime-override marker present → direct ChatView, no splash flash).
* **Byte-count types tightened to `UInt64` across the bootstrapper module** ([#398](https://github.com/machinefi/rapid-desktop/pull/398)). 6 files (3 source + 3 test), +58/-48 lines. Apple framework boundaries (`URLSession` delegates use `Int64`) cast at the boundary with `UInt64(max(0, n.int64Value))`. No behavior change; `Codable` validation now fail-closes on negative `sidecar_size` at decode time rather than passing decode and tripping the `> 0` guard later.
* **`AutoStartDecision.SkipReason` — `CaseIterable` + cardinality contract test** ([#392](https://github.com/machinefi/rapid-desktop/pull/392), closes [#356](https://github.com/machinefi/rapid-desktop/issues/356)). 4 cases (`userOptedOut`, `serverNotIdle`, `binaryMissing`, `noResolvableAlias`). Test pins both `allCases.count == 4` AND the exact raw-value `Set` so adding/removing/renaming a case (and forgetting to update the corresponding handler in the rest of the codebase) breaks CI. No `switch` over `SkipReason` exists today — the contract test is pure inoculation.
* **Per-test teardown for `SessionStoreAttachmentMigrationTests` temp dirs** ([#397](https://github.com/machinefi/rapid-desktop/pull/397), closes [#294](https://github.com/machinefi/rapid-desktop/issues/294)). 1332 `/tmp/rapid-#22-<UUID>` stragglers accumulated on disk; same class as previously-fixed #139. Mirror the PR #159 `TestDefaultsScope` RAII pattern: `struct → final class @MainActor` + `nonisolated(unsafe) var createdTempDirs: [URL]` tracker, `deinit` recursively unlinks. Before: +4 stragglers per run. After: 0 across 3 back-to-back filtered runs + full suite.

### Documentation

* **Reconcile `SettingsViewCmdCommaPerfTests` doc header with shipped guard count** ([#395](https://github.com/machinefi/rapid-desktop/pull/395), closes [#346](https://github.com/machinefi/rapid-desktop/issues/346)). PR #327's body advertised *three* structural perf guards in the Cmd+, suite; only the first two shipped (Category invariants + `formatSource(for:)` — the third was intentionally deferred because `SettingsWebSearchKeyDraftTests` already covers the helper contract end-to-end). Suite doc-comment now states "**two** guard groups" and links #346 + the existing NOTE block. Docs-only.
* **Warn that `SessionStore.delete(id:)` is the unguarded primitive** ([#396](https://github.com/machinefi/rapid-desktop/pull/396), closes [#347](https://github.com/machinefi/rapid-desktop/issues/347)). Adds a DocC `> Warning:` block directing UI-side callers (context menu, swipe, future Cmd+Delete, palette) to route through `SessionsSidebar/requestDelete(of:)` so the pinned-bypass + "Don't ask again" policy in `DeleteSessionConfirmation/decide(...)` is applied uniformly. Direct UI calls are only appropriate after that gate has resolved.

### Known issues (carried)

* **#364** — Main window jumps to X=1908 (off-screen-right) after dismissing OnboardingTour on a 1920-wide single display. P3; user can drag back. Not fixed in this release.
* **#365** — `cliclick` / System Events cannot open ModelPickerBar SwiftUI Menu. P3 testing-tooling, not a product bug.
* **#173** — Investigate AX bridge degradation on macOS 15+. Investigation, not yet a fix.

## [0.8.4] — 2026-06-24

Patch release. Bundled-sidecar bump from rapid-mlx v0.8.11 → **v0.8.14** carrying 16 user-facing PRs across three sidecar release waves (v0.8.12, v0.8.13, v0.8.14), plus one desktop launch-hardening mirror-fix and one CHANGELOG amendment closing the v0.8.3 dogfood follow-up issue.

### Fixed

* **Launch hardening — safe-unwrap `NSApp` in `applicationDidFinishLaunching`** ([#386](https://github.com/machinefi/rapid-desktop/pull/386)). v0.8.3 PR #376 fixed the `NSApp.setActivationPolicy(.accessory)` force-unwrap on the `applicationWillFinishLaunching` hook; the v0.8.3 dogfood agent flagged three more bare `NSApp.*` references at `RapidApp.swift:977/979/988` in the parallel `applicationDidFinishLaunching` hook (same bug-class, theoretical-only because no current test exercises that hook). v0.8.4 mirrors PR #376's `NSApp?` safe-unwrap to all three sites + adds a doc-block citing the precedent so future readers don't reintroduce the pattern. Pure no-op in production under `NSApplicationMain`; the fix only matters if a future `--filter` companion ever invokes the method directly.

### Changed

* **Bundled sidecar bumped to rapid-mlx v0.8.14** ([#384](https://github.com/machinefi/rapid-desktop/pull/384), submodule `e60f497 → 4b253a4`). 19 commits / 16 user-facing PRs spanning three rapid-mlx release waves now ship in the bundled sidecar (chronologically, by wave):
  * **v0.8.12** (raullenchai/Rapid-MLX#857, 2026-06-23) — Wave R10 cross-route hardening: reasoning/ui-tars `enable_thinking=false` honoured + accumulator-anchor ([rapid-mlx #850](https://github.com/raullenchai/Rapid-MLX/pull/850)); `/v1/responses` streaming `output_text.delta` events + `response.completed.output` restored ([rapid-mlx #851](https://github.com/raullenchai/Rapid-MLX/pull/851)); duplicate `delta.reasoning` key dropped ([rapid-mlx #852](https://github.com/raullenchai/Rapid-MLX/pull/852)); cache persist format pinned with schema-version + per-entry magic ([rapid-mlx #853](https://github.com/raullenchai/Rapid-MLX/pull/853)); audio serve-mode + comprehensive alias registry ([rapid-mlx #854](https://github.com/raullenchai/Rapid-MLX/pull/854)); api/tools wire-scrub completion + validation bundle ([rapid-mlx #855](https://github.com/raullenchai/Rapid-MLX/pull/855)); audio aliases.json shipped in wheel — was the 0.8.12 blocker ([rapid-mlx #856](https://github.com/raullenchai/Rapid-MLX/pull/856)).
  * **v0.8.13** (raullenchai/Rapid-MLX#867, 2026-06-24) — Wave R11 audio + tool-call + embeddings: `tool_choice="required"` streaming finalize race fix ([rapid-mlx #859](https://github.com/raullenchai/Rapid-MLX/pull/859), CRIT: 10/20 streams pre-fix shipped `finish_reason:"tool_calls"` with zero `delta.tool_calls` chunks; 0/20 post-fix); audio DX `format` → `response_format` alias + `voice:"default"` fallback + `/v1/models` audio capability ([rapid-mlx #863](https://github.com/raullenchai/Rapid-MLX/pull/863)); MLLM frequency/presence/repetition penalty pass-through to the VLM sampler ([rapid-mlx #864](https://github.com/raullenchai/Rapid-MLX/pull/864), closes upstream #512); vibevoice dynamic voice enumeration + correct default voice ([rapid-mlx #862](https://github.com/raullenchai/Rapid-MLX/pull/862)); embeddings UX hardening — 503 envelope on missing model + `[embeddings]` extra in pyproject + `/v1/models` visibility ([rapid-mlx #861](https://github.com/raullenchai/Rapid-MLX/pull/861)); `/v1/responses` streaming reasoning emit on `max_output_tokens` cutoff ([rapid-mlx #866](https://github.com/raullenchai/Rapid-MLX/pull/866)).
  * **v0.8.14** (raullenchai/Rapid-MLX#871, 2026-06-24) — `/v1/chat/completions` length-stop rescue stub restored for reasoning models ([rapid-mlx #860](https://github.com/raullenchai/Rapid-MLX/pull/860), closes upstream #858 — PR #815 had flipped `_cutoff_notice_enabled()` from default-ON to default-OPT-IN, so GUI clients started rendering empty bubbles on length-stopped reasoning generations). Cross-path parity follow-up: `/v1/responses` non-stream surface mistreated the rescue-stub sentinel as "real downstream output" and flipped `reasoning.status` from `incomplete` to `completed`; fixed by moving the sentinel literal to `vllm_mlx/api/constants.py` (preserves api/service layering) + 4 adapter-boundary regression pins ([rapid-mlx #869](https://github.com/raullenchai/Rapid-MLX/pull/869), 3 codex rounds → 0/0/0 converged). README catches up to the audio surfaces shipped in v0.8.12/v0.8.13 ([rapid-mlx #868](https://github.com/raullenchai/Rapid-MLX/pull/868)).

### Documentation

* **CHANGELOG amend — v0.8.3 sidecar bullet covers 5 missing surfaces** ([#385](https://github.com/machinefi/rapid-desktop/pull/385), closes [#383](https://github.com/machinefi/rapid-desktop/issues/383)). v0.8.3 dogfood Agent 5 surfaced 5 user-observable, non-breaking changes that the original v0.8.3 CHANGELOG bullet omitted (R6-H6 prefix-cache eviction in rapid-mlx v0.8.8; `/v1/audio/translations` in v0.8.6; `computer_use_preview` tools-type synonym in v0.8.9; `probe_fastpath.py` middleware in v0.8.10; `[audio]` pin shifts). All 5 validated against rapid-mlx `git log v0.8.4..v0.8.11`. Two false claims in the issue body were caught by codex review and explicitly NOT written (context-window enforcement existed pre-v0.8.7; `computer_use_preview` is not a `/v1/models` id).

## [0.8.3] — 2026-06-23

Patch release. Three desktop fixes/refactors plus a substantial sidecar bump (rapid-mlx v0.8.4 → v0.8.11) carrying loopback hardening, env-only API-key SSOT, and seven sidecar release waves (R-01 through R8) of cross-route reasoning/tool-call/streaming/audio/runtime hardening.

### Fixed

* **Launch hardening — safe-unwrap `NSApp` in `applicationWillFinishLaunching`** ([#376](https://github.com/machinefi/rapid-desktop/pull/376)). v0.8.2 moved the `.accessory` activation-policy flip out of `RapidApp.init()` and into `AppDelegate.applicationWillFinishLaunching`, where `NSApp` is normally alive. "Normally" — there is still a narrow window during test-runner cold start where `NSApp` is `nil` when the delegate hook fires, and the force-unwrap turned that race into a SIGTRAP on CI. Fix: replace `NSApp.setActivationPolicy(...)` with a `NSApplication.shared` access and a defensive optional binding, so the flip is a no-op (rather than a crash) when `NSApp` hasn't initialised yet. Companion tests now hold an explicit `NSApplication.shared` reference so `swift test --filter` runs trigger the AppKit bootstrap path and exercise the real code, not a degenerate fast-path. This closes the second-order test-isolation flake that v0.8.2 only half-fixed.

### Changed

* **Release notes are now CHANGELOG-driven** ([#375](https://github.com/machinefi/rapid-desktop/pull/375)). `release.yml` extracts the `## [X.Y.Z]` section from this file and uses it as both (a) the GitHub Release body and (b) the `notes` field in `latest.json` consumed by the in-app UpdateChecker. The sidecar tarball + manifest are now also mirrored to the GitHub Release as fallback assets, so an outage of the primary distribution channel doesn't break the auto-updater. **v0.8.3 is the first release that will exercise this end-to-end** — users opening this release on GitHub or seeing the "Update available" banner in-app will read the narrative below verbatim instead of the previous bare commit list.

### Internal

* **Document/enforce lazy main-window init in `.hideAlways` mode** ([#378](https://github.com/machinefi/rapid-desktop/pull/378), [#380](https://github.com/machinefi/rapid-desktop/pull/380)). Doc comments on `RapidApp.swift` `WindowGroup` and `AutoStartDecision.swift` codify the orthogonality contract — autoStart and hideAlways are independent dimensions, and main-window construction must remain lazy in `.hideAlways` so the menu-bar-only path never blocks on a windowed render pass. PR #378's initial `HideAlwaysOrthogonalToAutoStart` test landed at 1093 lines with source-introspection assertions; PR #380 trimmed it to ~115 lines of behavior-level checks, dropping the brittle file-reading scaffolding while retaining the regression backstop.
* **Prevent `Int32(1 << 31)` overflow in `ProcessKillChaosTests.fdSetAdd`** ([#377](https://github.com/machinefi/rapid-desktop/pull/377)). Pre-existing latent bug — `1 << 31` overflows signed `Int32`, which trapped under `-Onone`. Mirrors the BSD `__DARWIN_FD_SET` macro by using unsigned shifts and a bit-pattern cast back to `Int32`. Test-only change; no production impact.

### Sidecar — bundled rapid-mlx v0.8.4 → v0.8.11

* **Submodule** e60f497 (v0.8.11), bringing seven cumulative releases of server-side hardening on top of v0.8.4. Highlights from a desktop-user perspective:
  - **Security (defense in depth): upstream serve path now defaults `--host` to `127.0.0.1`** (rapid-mlx [#848](https://github.com/raullenchai/Rapid-MLX/pull/848)). Closes the PortSweep loopback-bypass surface on the standalone `rapid-mlx serve` path. The desktop has always pinned `--host 127.0.0.1` explicitly via `ServerManager.serveArguments` (the bundled sidecar was never LAN-reachable from the desktop), so this is a defense-in-depth upstream alignment — a future regression that dropped our explicit pin would now still land on a loopback default instead of `0.0.0.0`.
  - **Security: `RAPID_MLX_API_KEY` env-var SSOT** (rapid-mlx [#847](https://github.com/raullenchai/Rapid-MLX/pull/847)). The bundled standalone CLI path previously accepted the bearer token on the command line, leaking it into `ps -ef`. New env-var SSOT closes that leak. The desktop already passed the per-launch bearer via env (PR #145), so this aligns the standalone CLI surface with the desktop's already-hardened path.
  - **UI-TARS / Computer-Use family + EmbeddingGemma aliases added** (rapid-mlx [#812](https://github.com/raullenchai/Rapid-MLX/pull/812), [#818](https://github.com/raullenchai/Rapid-MLX/pull/818), [#823](https://github.com/raullenchai/Rapid-MLX/pull/823), [#829](https://github.com/raullenchai/Rapid-MLX/pull/829), [#833](https://github.com/raullenchai/Rapid-MLX/pull/833), [#834](https://github.com/raullenchai/Rapid-MLX/pull/834)). 9 new UI-TARS aliases + action parser + Anthropic `tool_use` mapping for the UI-TARS / computer-use-preview models, with streaming reasoning hold-back so `Thought:` prefixes don't leak as content tokens. Plus 2 new EmbeddingGemma aliases on the embeddings route.
  - **Reasoning / tool-call / streaming hardening across 7 release waves (R-01 through R8)**. Touches every public route: Anthropic `/v1/messages` stream finalize, `/v1/responses` reasoning items + cross-lane parity, streaming reasoning split, `tool_choice=auto` think-leak fix, finalize-on-truncation no-dup guard, GLM-4 / DeepSeek-V3 fullwidth-pipe tool-call recovery, Harmony analysis-channel routing on cut-short finalize, GPT-2 byte-fallback detokeniser mojibake fix, `stream_options.include_usage=false` honoured at the SSE final chunk.
  - **Audio bundle** (rapid-mlx [#819](https://github.com/raullenchai/Rapid-MLX/pull/819), [#828](https://github.com/raullenchai/Rapid-MLX/pull/828), [#835](https://github.com/raullenchai/Rapid-MLX/pull/835), [#839](https://github.com/raullenchai/Rapid-MLX/pull/839)). Whisper STT 500 fix, Kokoro phonemizer dep + full alias + format encoding + voice validation, `/v1/audio/translations` route, deeper-than-import capability probe, boot guard, `mlx-audio<0.4.4` pin.
  - **Runtime guardrails** (rapid-mlx [#820](https://github.com/raullenchai/Rapid-MLX/pull/820), [#830](https://github.com/raullenchai/Rapid-MLX/pull/830), [#836](https://github.com/raullenchai/Rapid-MLX/pull/836), [#837](https://github.com/raullenchai/Rapid-MLX/pull/837), [#840](https://github.com/raullenchai/Rapid-MLX/pull/840), [#843](https://github.com/raullenchai/Rapid-MLX/pull/843), [#846](https://github.com/raullenchai/Rapid-MLX/pull/846)). Context-window enforcement + prefix-cache eviction, SIGHUP stay-alive, healthz fast-path, cache cap admission + gauges, signal observability + faulthandler, mllm_scheduler executor cancel-gate, cross-route validation hardening, case-insensitive Origin handling, stress-validation attribution.

#### Sidecar — user-observable surfaces worth calling out ([#383](https://github.com/machinefi/rapid-desktop/issues/383))

The compound bullets above bundle a lot of waves into one line each. The following five surfaces are non-breaking but **user-observable** — users upgrading from the v0.8.2-bundled v0.8.4 sidecar may notice a behavior change on each one. None of them block the v0.8.3 tag (no removed fields, no breaking flag flips for the desktop's loopback-default path), but they deserve a dedicated mention so a release-day reader is not surprised.

1. **Prefix-cache eviction actually fires now (R6-H6 in v0.8.8, rapid-mlx [#830](https://github.com/raullenchai/Rapid-MLX/pull/830)).** v0.8.7 had a latent bug where the memory-aware prefix cache could balloon to ~31 GB on a 256 GB host before the LRU-on-cap path tripped, so `rapid_mlx_prefix_cache_evictions_total` stayed at 0. v0.8.8 adds a `RAPID_MLX_PREFIX_CACHE_MAX_BYTES` env-var override and a cache-self-pressure trigger inside `Scheduler.evict_prefix_cache_under_pressure` (independent of the Metal soft cap). Net effect: long desktop chat sessions with diverse system prompts now see steady evictions instead of unbounded cache growth. (Sibling fix in the same PR, R6-H5, adds cross-route test coverage for the pre-existing context-window 400 envelope — no behavior change there.)
2. **New `/v1/audio/translations` endpoint (v0.8.6, rapid-mlx [#819](https://github.com/raullenchai/Rapid-MLX/pull/819)).** Net-new OpenAI-spec endpoint (not a bug-fix on an existing surface). Custom clients wired directly to the sidecar can now call audio translation in addition to transcription. The desktop chat surface does not consume this endpoint today; called out for users with their own integrations.
3. **`computer_use_preview` Responses-API tool-type synonym (R7-A in v0.8.9, rapid-mlx [#834](https://github.com/raullenchai/Rapid-MLX/pull/834)).** The Responses adapter now accepts `tools[].type = "computer_use_preview"` as a synonym for the dated `computer_20251022` spec name (the OpenAI Python SDK default). This is a `tools[].type` alias at the request-validation boundary, **not** a `/v1/models` entry — clients that hard-code the SDK default no longer 400 on the tool-type discriminator. No model-picker impact.
4. **`probe_fastpath.py` middleware on `/healthz` (R8-B in v0.8.10, rapid-mlx [#840](https://github.com/raullenchai/Rapid-MLX/pull/840), [#843](https://github.com/raullenchai/Rapid-MLX/pull/843)).** New 248-line middleware short-circuits `/healthz` before hitting the main request stack, so health-probe latency stays bounded under heavy concurrent inference load. The desktop heartbeat poll (`ServerManager` `/healthz` ping) consumes only the 200 status and is compatible with the fast-path; this is a latency-shape change, not a contract change.
5. **`[audio]` extra dependency pin shifts (R6-C in v0.8.8 + R7-C in v0.8.9, rapid-mlx [#828](https://github.com/raullenchai/Rapid-MLX/pull/828), [#835](https://github.com/raullenchai/Rapid-MLX/pull/835)).** `mlx-audio` capped at `<0.4.4`; new `espeakng-loader>=0.2.0` dependency; `phonemizer>=3.2.0` swapped for `phonemizer-fork>=3.3.0`. The bundled sidecar in this `.app` ships these pins baked in — no user-visible effect on the desktop. Users with a separate `pip install rapid-mlx[audio]` venv may see resolver constraints differ when upgrading that venv to v0.8.11.

`.app` envelope unchanged at ~445 MB; alias catalog grows from 92 to 103 entries (UI-TARS + EmbeddingGemma additions); all desktop-pinned aliases still present; slim mechanism (build-sidecar.sh steps 3.5 + 3.6) intact.

### Known issues (carried)

* **#364** — Main window jumps to X=1908 (off-screen-right) after dismissing OnboardingTour on a 1920-wide single display. P3; user can drag back.
* **#365** — `cliclick` / System Events cannot open ModelPickerBar SwiftUI Menu. P3 testing-tooling, not a product bug.

## [0.8.2] — 2026-06-22

Patch release. Single P0 hotfix — no other changes from v0.8.0.

### Fixed

* **Launch crash for users with "Hide Dock icon + Don't ask again" persisted** (raullenchai/Rapid-MLX#845, [#373](https://github.com/machinefi/rapid-desktop/pull/373)). v0.8.0 (build 117) introduced a "no Dock-icon flash on launch" optimisation inside `RapidApp.init()` that called `NSApp.setActivationPolicy(.accessory)` directly when the persisted `HideDockChoice` was `.hideAlways`. SwiftUI's `App.init()` runs before `NSApplicationMain` initialises `NSApp`, so the implicitly-unwrapped global force-unwrapped `nil` → `EXC_BREAKPOINT (SIGTRAP)` on every launch. New installs were unaffected; the bug only fired for users who had previously chosen "Hide Dock icon + Don't ask again". Workaround for affected users: `defaults delete com.rapidmlx.rapid rapid.window.hideDockChoice` and relaunch. Fix: move the activation-policy flip into `AppDelegate.applicationWillFinishLaunching` where `NSApp` is alive but the Dock icon hasn't rendered yet — preserves the original UX intent without the crash. Added `InitMustNotTouchNSAppTests` as a mechanical regression backstop: reads `RapidApp.swift` via `#filePath` and asserts no `NSApp.*` token appears between `init() {` and its matching `}`, so any future "let's just touch NSApp from init for X" attempt fails CI immediately. Reported by xiaogwu (Sean Wu @ Apple).

### Sidecar

* Bundled rapid-mlx unchanged from v0.8.0 (still v0.8.4, cbc773f). v0.8.2 is desktop-only.

## [0.8.0] — 2026-06-21

Minor version bump reflecting two systematic security/correctness fixes (#361 + #363) and the bundled rapid-mlx jump to v0.8.4. No user-facing UX change versus v0.7.21 — this is a defense-in-depth release.

### Security

* **Closed bundled-sidecar cwd module-hijack** ([#361](https://github.com/machinefi/rapid-desktop/issues/361), [#366](https://github.com/machinefi/rapid-desktop/pull/366)). The bundled rapid-mlx shim invoked `python3.12 -u -s -m vllm_mlx.cli`, and `-m` mode prepends cwd to `sys.path[0]`. A sibling `vllm_mlx/` directory in the caller's cwd could hijack the bundled import path even though existing `-s` / `PYTHONNOUSERSITE` / `PYTHONHOME` / `PYTHONPATH` hardening blocked user-site and host-Python contamination. Pre-existing on every desktop release prior to v0.8.0; the v0.7.20→v0.7.21 sidecar bump did not introduce it. Fix: add `-P` flag (Python 3.11+'s `PYTHONSAFEPATH=1` arg) to the python invocation AND export `PYTHONSAFEPATH=1` in the env block alongside the rest of the hardening (belt+suspenders). 5 new tests: 3 source-grep tripwires guarding both the `-P` flag and the env var, 1 live poison-cwd reproducer against a fresh-built bundle, 1 happy-path regression guard for the `/tmp` clean cwd case.

### Correctness

* **`/v1/models` response now exposes `context_window`** ([#363](https://github.com/machinefi/rapid-desktop/issues/363), [#367](https://github.com/machinefi/rapid-desktop/pull/367) + upstream rapid-mlx [v0.8.4](https://github.com/raullenchai/Rapid-MLX/releases/tag/v0.8.4)). The desktop's max-token auto-scale (PR #318 sibling consumer) needs `context_window` to size the reasoning budget cap. Bundled rapid-mlx versions prior to v0.8.4 never emitted the field at all — desktop silently fell through to a too-conservative default, so large-context aliases got capped at the floor. Pre-existing on every release prior to v0.8.0 (A/B-verified on v0.7.20 bundled v0.7.41 sidecar). Fix lands cross-repo: rapid-mlx v0.8.4 emits the field from `AliasProfile`/HF config, and desktop's `ServerModelProfile` parses it with a per-family fallback table in `ModelInfoCatalog.familyAndContext(for:)` — Qwen 2/2.5/3/3.5/3.6/QwQ → 32_768, Llama 3.1/3.2/3.3/4/4.5 → 131_072 (note `llama-3` without minor → 8_192), Gemma 2/3/3n/4 → 8_192, Mistral → 32_768, GLM 4/4.7/5 → 131_072, DeepSeek V3/V4 → 131_072 (plain `deepseek` → 32_768), Phi 4 → 16_384, Phi 3 → 4_096, SmolLM 3 → 8_192 (older SmolLM → 2_048), Hermes 3 → 131_072 (older Hermes → 32_768), Bonsai → 4_096; unmatched aliases return `nil` (not a generic numeric floor) so callers branch on `Optional<Int>` rather than trust a fabricated default. 13 new tests across `ServerModelProfileTests` + `ModelInfoCatalogTests`.

### Sidecar

* **Bundled rapid-mlx v0.8.2 → v0.8.4** (submodule b25e3af → cbc773f). The v0.8.3 release (188cd91) batches a wave of server-side hardening landed on main between v0.8.2 and v0.8.4: H-01 reasoning rescue on length-cut mid-think (#802), H-06 outlines `json_schema` strict-mode enforcement (#801), H-08/H-09 embeddings guard + 400 on `/v1/embeddings` without `--embedding-model` (#800), D-METAL-CAP/D-METAL-PFX scheduler `--gpu-memory-utilization` admission + pressure-triggered prefix-cache eviction (#797), D-TOOL-RECUR/D-DEEP-JSON iterative tool-schema walk + body-depth guard (#798), D-DSV31 DeepSeek-V3 fullwidth-pipe tool-call recovery (#795), D-M01-DEAD/D-M01-2X metrics sub-counter wiring + 2× over-count race fix (#796), D-HARMONY-LEAK analysis-channel routing on cut-short finalize (#794), D-SSE-USAGE honor `stream_options.include_usage=false` at the SSE final-chunk (#792 — partial; see Known issues), and D-DETOK-BPE GPT2 byte-fallback mojibake fix (#793). The v0.8.4 release (cbc773f) is the targeted chore bump that ships the `/v1/models` `context_window` emit fix (a2f2539 / rapid-mlx PR #808) for #363 as a tagged version the desktop submodule can pin. `.app` envelope unchanged at ~445 MB, 92-alias catalog unchanged, all desktop-pinned aliases still present.

### Known issues (carried)

* **#364** — Main window jumps to X=1908 (off-screen-right) after dismissing OnboardingTour on a 1920-wide single display. P3; user can drag back. Not fixed in this release.
* **#365** — `cliclick` / System Events cannot open ModelPickerBar SwiftUI Menu. P3 testing-tooling, not a product bug.

## [0.7.21] — 2026-06-21

Quickstart's cold install drops from ~11 min to ~1.5 min on a typical
home connection, plus the picker grows a dedicated "Quickstart" section
and a UI race fix. Sidecar moves up to rapid-mlx v0.8.2.

### Quickstart — 4B → 0.6B (~2.9 GB → ~400 MB)

* **Quickstart alias swap**: `qwen3.5-4b-4bit` → `qwen3-0.6b-4bit`. At
  4.4 MB/s home bandwidth this is ~11 min → ~1.5 min cold install,
  i.e. the difference between "watching a progress bar past the
  patience threshold" and "first response in under two minutes". This
  re-litigates the issue #308 receipt (which originally pushed
  Quickstart from 1B → 4B for tool-call demo reliability): the new
  tradeoff explicitly accepts losing the tool-calls demo on the
  Quickstart model in exchange for first-impression latency. Users
  wanting tool-calls trade up via the Recommended Default in the
  picker, which is RAM-tuned and pinned to `.known` tool-capable
  aliases. (#359)
* **Example prompts are now pure text.** The three chips in the chat
  empty state used to include calculator-style tool-using questions
  (e.g. "What is 15% of 2,650 plus the square root of 781?") — these
  would silently fail on the 0.6B Quickstart model, which does not
  reliably emit `tool_calls`. Replaced with three pure-text prompts
  matching the existing ~52-char `.lineLimit(1)` envelope of the
  picker column. (#358, #359)

### Picker — dedicated Quickstart section + race fix

* **New "Quickstart" section** above "Recommended for your <RAM> GB
  Mac". One row, RAM-blind, contains exactly the Quickstart alias
  with a "Smallest model — fastest first install" subtitle. Persists after Quickstart
  is dismissed, so users who skipped the welcome flow can still
  one-click install the demo model from the picker. De-duplicated
  against the "All models" long-tail list. (#359)
* **Picker race fix during Quickstart download.** While Quickstart was
  in flight, the picker auto-pinned the RAMBucketedDefault Default
  alias (e.g. `qwen3.5-9b-4bit` on an 18 GB Mac) and the top-right
  "Download & start" CTA stayed enabled — clicking it would race a
  second concurrent download against the Quickstart card. Now the
  picker selection mirrors the Quickstart alias while Quickstart is
  in `.lowDiskWarning`/`.downloading`/`.starting` phases, and the top-right CTA is
  disabled with a "Quickstart download in progress" tooltip. The
  duplicate top mini-progress strip is also hidden while the
  Quickstart modal is visible. (#359)

### Sidecar — bundled rapid-mlx v0.8.2

* **Bundled rapid-mlx 0.7.41 → v0.8.2.** `.app` envelope unchanged
  at 445 MB (0 MB delta vs prior release), 92-alias catalog
  unchanged, all 14 desktop-pinned aliases still present, slim
  mechanism (build-sidecar.sh steps 3.5 + 3.6) intact. Bundled
  `--version` reads `0.8.2` under cwd-isolated probe per issue
  #355. End-to-end chat-completion smoke green at 84.9 tok/s on
  `qwen3-0.6b-4bit`. (#360)

### Known issues / deferred

* **#361 — sidecar-shim.sh cwd module-hijack hardening.** A sibling
  `vllm_mlx/` directory in the caller's cwd can hijack the bundled
  rapid-mlx import path under `python -m vllm_mlx.cli`. Pre-existing
  (not v0.8.2 introduced); recommended fix is the `-P` /
  `PYTHONSAFEPATH=1` flag in the shim. Filed for a separate PR.

## [0.7.15] — 2026-06-19

Codex-converge post-merge audit of the v0.7.14 release surfaced three real
bugs (1 BLOCKING + 1 MAJOR + 1 P2) and the sidecar shipped on rapid-mlx
v0.7.37 which closes four more (F-007/F-008/F-009/F-010). Both halves
landed under the new "codex review until convergence" SOP — no fixed
round cap.

### Desktop fixes

* **Auto-respawn budget actually bounds now.** v0.7.14's 3-retry budget
  was non-binding for "child reaches `.ready`, then crashes" loops: every
  `.ready` transition reset the counter to 0, so a child that briefly
  answered `/healthz` then died (OOM-on-inference, segfault-on-prompt,
  ...) looped forever at 2 s intervals with the UI showing
  ready→respawning→ready→... thrash. Fixed by requiring `≥ 60 s` of
  `.ready` uptime before the next exit refreshes the budget. (#278, #280)
* **Stop now actually stops mid-respawn.** A user clicking Stop while a
  watchdog-scheduled auto-respawn was suspended at the pre-spawn
  download-settlement await could end up with a brand-new child anyway —
  `cancelAutoRespawn()` cancelled the Task, but `start()` resumed past
  the cancel and slipped past the post-await `!isOperating` /
  `child == nil` guards because Stop had cleared both. Fixed by adding a
  `Task.isCancelled` check the moment the await returns. (#278, #280)
* **`HF_HOME` / `HF_HUB_CACHE` / `XDG_CACHE_HOME` now reach the
  sidecar.** v0.7.14's env allowlist dropped them, so a user with
  `HF_HOME=/Volumes/External/hf` saw the launcher monitor the external
  SSD while the child wrote to `~/.cache/huggingface/hub` — cache split,
  duplicate downloads, 0% bytes-on-disk progress. Allowlist now passes
  them through. Plus four HF behaviour knobs (`HF_ENDPOINT`,
  `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`,
  `HF_HUB_ENABLE_HF_TRANSFER`) the `pull` path already honoured but the
  `serve` path was stripping; the two paths now agree. (#277, #279)

### Sidecar (rapid-mlx 0.7.20-77 → 0.7.37)

* **F-007 oversize body DoS.** `--max-request-bytes 8 MiB` default +
  per-request token-budget pre-check across all four chat surfaces
  (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
  `/v1/messages`). 10–100 MB bodies that previously hung the worker for
  60–90 s now reject with 413 in milliseconds.
* **F-008 llama-3.1-8b raw-JSON tool leak.** The `llama` tool parser
  now accepts both Llama 3.1's bare `{"name": "X", "parameters": {...}}`
  shape and the `<|python_tag|>` variant, so tool calls land in
  `tool_calls` instead of leaking as raw JSON in `content`.
* **F-009 `/metrics` Prometheus endpoint.** 14 metrics mapped to existing
  engine + scheduler + prefix-cache stats; sticky-counter accumulator
  guarantees `_total` monotonicity across cache-clear.
* **F-010 `/v1/messages` duplicate `thinking` block.** Anthropic-format
  responses no longer emit a `thinking` block on non-thinking models,
  matching Anthropic's real API contract.

## [0.7.14] — 2026-06-18

Stress-round hardening: spawn-shape uniformity, idle-crash watchdog,
and an env allowlist that no longer leaks the launcher's third-party
secrets into the `rapid-mlx` sidecar.

* **Idle-state crash now surfaces and respawns.** Previously a
  `rapid-mlx` SIGKILL while no chat window was visible left the
  desktop alive but inert — only `Cmd+N` triggered respawn. The
  desktop now watchdog-respawns the child up to 3 times when the
  prior spawn cycle had reached `.ready`, with a 2 s gap; the
  user's manual `Stop` cancels the queue from any state including
  `.crashed` (no live child). (#270, #271)
* **One spawn shape, everywhere.** Cold start and any future respawn
  trigger build their argv + env via a single static helper, pinned
  by `SpawnArgumentsTests`. The bearer travels via `RAPID_MLX_API_KEY`
  env var only; nothing reaches the cmdline where `ps -ax` could
  read it. (#271)
* **Spawn env allowlist.** The bundled `rapid-mlx` sidecar now sees
  only an explicit 15-var allowlist (POSIX baseline + Python launcher
  + TLS cert pointers + macOS bookkeeping) of the launcher's ambient
  env, plus the desktop-injected vars. A user who launched the
  desktop from a Terminal with `ANTHROPIC_API_KEY` / `BRAVE_API_KEY`
  / etc. exported will no longer leak those into the sidecar's env
  (where they showed up in `ps eww` and could be snapshot by a
  third-party crash reporter). (#272)

## [0.7.13] — 2026-06-18

Multi-hour downloads on slow user links no longer crash mid-pull.

`ServerManager` polls `rapid-mlx`'s `/healthz` after spawning the
child. The old shape gave that polling loop a fixed **30 min
wall-clock budget from launch**, then SIGKILL'd the child. On a
slow link that's the wrong policy: a 10 GB model at 683 KB/s takes
~4 hours; the deadline fired at the 30 min mark, the partial
download was orphaned, the next launch restarted from zero. Reported
on v0.7.12 dogfooding.

The 30 min budget is now a **stall window** measured from the most
recently observed forward-progress signal — heartbeats, R2
completions, tqdm ticks, disk observations from the cache monitor,
or phase transitions. A download that's actively moving never hits
the cap regardless of duration. A genuinely-wedged child still
surfaces as `.crashed` within 30 min of no activity.

Safety invariant preserved: `downloadProgress.reset()` sets
`lastTickAt = .distantPast` before the polling loop runs, so a child
that never emits ANY recognised signal AND never answers `/healthz`
still hits the original 30 min hard cap measured from launch — the
"truly silent wedge" failure mode is unchanged.

Terminate reason text reframed from "did not answer /healthz within
30 minutes" to "made no progress for 30 minutes — treating as
crashed" so the failure mode reads correctly in the crash banner.

6 new tests pin the boundary at the stall-window equality, idle
below/at/over, the 4-hour download survival path, and the
heartbeats-stop-mid-flight tolerance.

## [0.7.12] — 2026-06-18

The download progress bar feels alive on slow user links.

v0.7.11 smoothed the bar by feeding rapid-mlx's `[bytes] D/T`
heartbeat into the desktop's byte counter. But the displayed
subtitle stayed at coarse precision — `1.7 GB / 17%`, with the GB
quantum at 0.1 (~100 MB) and the percent quantum at 1. On a real
683 KB/s user link the percent quantum takes **~3.7 minutes** to
cross. The bar inches forward but the *number* doesn't. Users
wait, decide the app's dead, leave.

This release adds a per-tick download-speed readout + a
Chrome-style ETA so the subtitle has a high-frequency motion
signal regardless of bucket boundaries.

**Before:** `1.7 GB / 10.3 GB · 17%`
**After:** `1.7 GB / 10.3 GB · 17% · 683 KB/s · 5 min left`

- **Rolling-window byte-rate calc** (4 s window over the existing
  500 ms heartbeat cadence). Suppressed on staleness, on the
  shrinking-buffer tick HF emits when it hardlinks blobs into the
  snapshot dir, and on the first observation (before two samples
  exist). `< 1 KB/s` is the displayed floor — the readout never
  reads `0 B/s` next to a still-moving GB counter, because that
  reads as a stall.
- **ETA uses Chrome's idiom:** `< 1 min left` / `5 min left` /
  `1 h 23 min left`, capped at `> 24 h left`. Without that cap, a
  momentary trickle on a 100 GB target could quote "47 h" and
  trash user trust in every subsequent estimate. The minute-rounding
  carries into the hour bucket so `seconds = 7170` reads `2 h left`,
  not the wrong-arithmetic `1 h 60 min left`.
- **Toolbar pill pinned to one line.** The pill now reads
  `Downloading 2.0 GB / 6.6 GB · 30% · 103 MB/s · < 1 min left`
  (~75 chars), which would have wrapped to two lines in narrow
  toolbars with the old layout. Tail truncation lops the elapsed-time
  clock first — that signal is redundant now that ETA exists.

The state model splits the rate calc from `applyDiskObservation`
via a `(bytes:at:)` test-seam overload so the buffer + window
arithmetic is unit-testable with synthesised `Date`s. The
no-arg production overload is unchanged.

11 new tests pin the rate suppression rules, the format helpers
at every cliff, the minute-carry boundary, and the single-sample
"no speed tail" contract.

## [0.7.11] — 2026-06-18

Smooths the download progress bar inside a single big-shard fetch.
v0.7.10 fixed the buffer-stall so the desktop saw rapid-mlx's
`[N/M] file R2 (X MB)` completion lines — but those lines only fire
once per file. On a 6-shard model the bar walks 0 → 17 → 33 → 50 → 67
→ 83% briskly, then sits at **83% for 60–120 seconds** while the
final 3 GB shard streams silently. That is the "stuck at 83%" bug
the bonsai / gemma-4-12b downloads exhibited in v0.7.10.

- **rapid-mlx now emits an aggregate `[bytes] D/T` heartbeat every
  500 ms while any worker is streaming.** ``D`` is cumulative bytes
  across all files (cached + R2 + HF-fallback) and ``T`` is the
  planned snapshot size (sum of HF-advertised sizes from
  ``model_info``). Thread-safe — one print per heartbeat window even
  with 4 concurrent R2 workers. A final flush at the end of the pull
  guarantees the last sub-500 ms tail lands at 100% so the bar reads
  full before the next phase banner appears.
- **The desktop's ``DownloadProgress`` parser feeds the heartbeat
  directly into ``applyDiskObservation(bytes:)`` + ``setTotalBytes``.**
  These were the existing rails the byte-on-disk monitor
  (``HFCacheByteMonitor``) used; the heartbeat is just a more direct
  source — bypasses filesystem polling, includes pre-rename ``.part``
  bytes, and uses the EXACT byte counts rapid-mlx wrote. The
  ``progressFraction``/``progressSubtitle`` priority order is
  unchanged: bytes win whenever they're observed.

  Verified locally: replay of the v0.7.10 "stuck at 83%" sequence
  (5 small completion lines + 1 mid-shard heartbeat at 50% of total)
  moves the progress bar to 50% even though file-count percent is
  still 83 — that is exactly the regression test the new
  ``r2BytesHeartbeatBeatsFileCountForProgress`` case enforces. The
  HFCacheByteMonitor (added in v0.7.9 PR #258) remains as a
  belt-and-braces fallback for environments where stdout is dropped.

  Round-1 review tightening (no user-visible behaviour change):
  - ``matchR2BytesHeartbeat`` skips ``setTotalBytes`` when the value
    is unchanged (~240 redundant ``@Observable`` notifications saved
    per pull) AND refuses a heartbeat whose ``total`` shrunk — a
    buggy mirror replaying an older ``D/T`` pair no longer pins the
    subtitle at "X / smaller-Y · 100%" mid-pull.
  - ``ServerManager.appendLogLines`` drops ``[bytes] D/T`` lines
    from the user-visible 200-line log tail. At ~2 Hz × 60-120 s
    the heartbeat would otherwise evict every legitimate startup
    log / warning before the user opened the drawer.
  - 4 additional unit tests cover the new guards plus a
    ``DownloadProgress.isHeartbeatLogLine`` classifier check.

## [0.7.10] — 2026-06-18

Hot-fix on top of v0.7.9. The R2-puller progress-bar parser shipped
in v0.7.9 was correct, but its lines never reached the parser:
CPython block-buffers stdout/stderr when they are piped (non-TTY),
and rapid-mlx's per-file completion lines (`[N/M] file R2 (X MB)`)
total only a few hundred bytes for a 3 GB download — far below the
~4 KB flush threshold. The desktop saw zero output for the entire
download.

- **Force the bundled sidecar's stdout/stderr to flush line-by-line.**
  ``sidecar-shim.sh`` now sets ``PYTHONUNBUFFERED=1`` AND passes
  ``-u`` to the python interpreter (belt + braces — a future shim
  edit that drops the env-var still has the flag, and vice-versa).
  Every print from `_mirror.py` / `huggingface_hub` / uvicorn now
  reaches the desktop the moment Python writes it, which is the
  precondition the v0.7.9 R2-matchers were silently waiting on.

  Verified live: ``rapid-mlx pull bonsai-1.7b-unpacked`` piped through
  the patched shim now emits ``[1/14]`` … ``[13/14]`` within the first
  second of process start, where the un-patched shim flushed nothing
  until the entire 3.5 GB pull finished (~70 s). The 30 unit-test
  parser suite from v0.7.9 covers what happens after a line arrives.

## [0.7.9] — 2026-06-18

A progress-bar visibility fix for the cold-download experience. v0.7.8
users who tried to pull a brand-new model (e.g. `gemma-4-12b-qat-4bit`)
stared at "Spinning up rapid-mlx…" with no progress bar for many
minutes while multi-GB shards streamed in the background. The bar
eventually appeared only if the rapid-mlx mirror puller fell back to
HuggingFace — now it shows up immediately for the R2-mirror phase too.

- **Download overlay now shows progress during the rapid-mlx R2
  mirror phase.** rapid-mlx ≥ 0.7.6 ships its own per-file puller
  (`vllm_mlx/_mirror.py`) that fetches from the project's R2 mirror
  before falling back to HuggingFace. Its progress output looks
  nothing like HuggingFace's `tqdm` bar (which is what the desktop's
  parser knew about), so the overlay sat at "Spinning up rapid-mlx…"
  for the entire R2 phase — on an 11 GB Gemma-4 pull that's 10+
  minutes of silent staring. The parser now also recognises the R2
  puller's three line shapes: `Pulling <repo> (R2 mirror, fallback:
  HF)` flips the spinner caption to a preparing state; `Found N
  files (~X.X GB total)` primes the linear progress bar; `[N/M]
  <file> R2 (X MB)` / `HF (X MB, fallback)` / `cached (X MB)` /
  `miss (...)` advances the bar one file at a time. The bar jumps
  per-file rather than being smooth (the R2 puller does not stream
  mid-shard progress); the existing bytes-on-disk monitor continues
  to smooth the mid-shard wait.

## [0.7.8] — 2026-06-18

Sidecar packaging migrated home + a gemma-4 load fix that was hiding
inside it. Users who tried to start any `gemma-4-*` alias on v0.7.7
hit an `ImportError` and the server failed to come up — that is now
fixed end-to-end.

- **`gemma-4-*` aliases now load.** v0.7.7's bundled sidecar shipped
  without `mlx_vlm` in `site-packages/`, so every gemma-4 alias
  crashed at load with `ImportError: Gemma 4 models require the
  optional 'mlx-vlm' dependency for the model architecture classes`.
  rapid-mlx's loader needs `mlx_vlm.models.gemma4.{config,language}`
  even on the text-only path because the architecture classes only
  live inside mlx-vlm's tree (rapid-mlx doesn't have its own copy).
  The sidecar build now installs `mlx-vlm>=0.6.3 --no-deps` plus
  `Pillow>=10.0` (Pillow is the only eager-import dep not already in
  the bundle — `mlx_vlm/__init__.py` chains `.convert` → `.utils` →
  `PIL`). Skipping the `[vision]` extras keeps the install ~22 MB
  instead of the ~322 MB cascade those extras would pull (torch +
  torchvision + cv2).
- **`scripts/build-sidecar.sh` migrated from the rapid-mlx submodule
  into this repo.** Sidecar packaging is a rapid-desktop concern —
  tunables like `MACHO_BASELINE_COUNT`, the extras list, the
  transformers trim step, the `.pyc` hoist, and signing entitlements
  all gate a rapid-desktop artifact, so they live next to it. The
  old arrangement (script under `third_party/rapid-mlx/scripts/`,
  called by indirection from `scripts/build.sh`) was the v0.7.4 slim
  mechanism's "hidden landmine" — easy to revert silently on a
  submodule bump. The submodule is still required (its source tree
  is what `pip install` reads to produce the bundle), but the
  packaging logic now sits at `scripts/build-sidecar.sh` next to
  `scripts/build.sh`, `scripts/sidecar-shim.sh`, and
  `scripts/sidecar-entitlements.plist`. No user-visible behavior
  change beyond the gemma-4 fix above.
- **Sidecar Mach-O baseline 51 → 77.** Pillow vendors 8
  `.cpython-312-darwin.so` modules + 18 `.dylibs/` libraries
  (libavif, libbrotli{common,dec}, libfreetype, libharfbuzz, libjpeg,
  liblcms2, liblzma, libopenjp2, libpng16, libsharpyuv, libtiff,
  libwebp{,demux,mux}, libXau, libxcb, libz). All 26 new Mach-Os
  are standard well-formed Pillow-wheel binaries (same mechanism as
  mlx_metal / numpy / safetensors already vendor) and codesign
  cleanly with the existing identity loop.

Size impact: raw `.app` 384 → 420 MB (still 80 MB under the 500 MB
CI cap); DMG growth ≤ 8 MB (well under the 50 MB delta gate).

## [0.7.7] — 2026-06-18

Hygiene release bundling two robustness fixes on the new R2-mirror /
slim-DMG foundation. No API or settings surface change.

- **Server stagger behind in-flight pull (#253).** When a model was
  downloading from the picker's background-pull control and the user
  hit Run on the same alias, `ServerManager.start(...)` could spawn a
  serve child concurrently with the still-running pull. That left
  two processes touching the same Hugging Face cache and an orphaned
  rapid-mlx process holding a port. The serve path now awaits
  `DownloadManager.awaitDownloadSettlement(alias:)` before claiming
  ownership, with post-await guards (`!isOperating` + `child == nil`)
  to bail safely if a sibling `start(...)` won the race during the
  wait. Cancellation exits the wait cleanly without busy-looping the
  MainActor (#256).
- **Failed-Finder-Replace detection (#251).** macOS Finder silently
  drops the "Replace" action when the user is dragging a new
  `Rapid-MLX Desktop.app` onto `/Applications` while the previous
  build is still running — the new bundle ends up on the user's
  desk, the running app keeps quietly running an older version, and
  the next "Up to date" check is technically truthful but
  user-hostile. `InstallTracker` now compares the bundle's Info.plist
  mtime + `CFBundleShortVersionString` against the last-seen
  baseline, with a 0.5 s APFS jitter slack and a both-halves-required
  rule. When detection fires (post-1-hop only — fresh installs are
  silent), a `FailedReplaceBanner` surfaces the "Open update dialog"
  CTA above the chat surface. VoiceOver gets the headline + body
  combined for accessibility, with the dismiss + CTA controls left
  as independently-focusable siblings (#257).
- **Sidecar: rapid-mlx 0.7.28 → 0.7.29.** Test-only fix to
  `test_langchain.py` harness path (rapid-mlx #669); production
  runtime behavior unchanged from 0.7.28.

## [0.7.4] — 2026-06-17

The "slim the DMG and stop relying on HuggingFace Hub" release. Two
load-bearing changes ship together because they only make sense as a
pair: we drop the 320 MB bundled `qwen3-0.6b-4bit` to crush DMG size,
and we point first-run downloads at our own R2 mirror so the new
first-launch flow is fast and rate-limit-free instead of slow and
HF-Hub-throttled.

- **R2 model mirror, default-on.** First-run model pulls now default
  to `RAPID_MLX_MODEL_MIRROR=https://models.rapidmlx.com` (rapid-mlx
  #647). Power users who set the env var in their parent shell keep
  full override (including the empty-string opt-out — we explicitly
  don't clobber it). On a mirror miss the sidecar falls through to
  the HuggingFace Hub exactly like before, so this is purely additive:
  faster + less rate-limited for everyone, with HF Hub as the
  guaranteed fallback.
- **App bundle slimmed ~440 MB.** We dropped the 320 MB bundled
  `qwen3-0.6b-4bit` (first launch now pulls it from the R2 mirror
  above; instant on a normal connection), and the sidecar's
  transformers modeling tree was trimmed + the `.pyc` cache hoisted
  out of the wheel for a ~101 MB sidecar win (rapid-mlx #646). Net
  DMG savings show up as the gates below.
- **CI size gates.** Release now blocks if the unzipped `.app` exceeds
  500 MB or the DMG grows more than 50 MB vs the previous tag.
  Mechanically prevents the slow drift that landed us at a 462 MB
  v0.7.3 DMG.
- **Bonus from rapid-mlx 0.7.20 → 0.7.26.** Anthropic / Claude Code
  passthrough on `/v1/messages` and friends (rapid-mlx #619), Gemma 4
  altup bare-fp quant autodetect fix (#641), diffusion image worker
  `worker_stuck` self-heal (#645), plus assorted agent + bench +
  release-tooling polish.

No API surface changes. No setting changes. If the mirror is down,
you get the v0.7.2 behavior (HF Hub) automatically.

## [0.7.3] — 2026-06-17

Hygiene release on top of v0.7.2 — three follow-up fixes, no new
features, no API surface changes.

- **About panel.** Version + build now render in separate AppKit
  dictionary slots so the About box stops showing
  `Version 0.7.2 (97) (97)` (#238 → #237).
- **BundledModel symlink.** Re-link the HF-cache symlink when the
  `.app` has moved between disks. Previously left a dangling link
  that bricked first-chat (#240).
- **Snapshot rebaseline.** 5 panel snapshots (ModelPickerBar +
  SessionsSidebar empty state) rebaselined against the post-v0.7.2 UI
  drift (#239).

Backend / sidecar / bundled-model behavior is identical to v0.7.2.

## [0.7.2] — 2026-06-17

Follow-up to v0.7.1's bundled-tiny-model first-paint UX. The v0.7.1
ship made the first chat instant + zero-network by bundling a 320 MB
`qwen3-0.6b-4bit` model inside the DMG, but 0.6B is intentionally a
smoke-test-grade model — users will hit its quality ceiling within a
few turns. Without a UI nudge the failure mode was "this app is dumb"
followed by uninstall.

v0.7.2 adds an explicit, hardware-aware upgrade nudge:

- **Upgrade banner after 5 turns with the bundled model.** A subtle
  banner appears above the chat composer once you've sent 5 messages
  to the bundled smoke-test model, suggesting a hardware-appropriate
  upgrade target. Copy is dynamic — on a 32 GB Mac it pitches
  `qwen3.6-27b-4bit`; on an Ultra it pitches the 35B; on a 16 GB Mac
  it pitches `qwen3.5-9b-4bit`. The target is sourced from
  `RAMBucketedDefault.recommendations(forPhysicalRAMGB:)[.default]`,
  the same table that drives the picker's "Recommended for your N GB
  Mac" row.
- **One-click background download.** Clicking "Download in background"
  reuses the existing `DownloadManager` plumbing — the job shows up
  inline in the banner AND in the top-bar `DownloadStrip` so you
  always see progress, regardless of which surface is in view.
- **Three dismissal modes.** "Maybe later" hides the banner for the
  current app launch — it re-appears next time you open Rapid if the
  trigger conditions still hold. "Don't show again" persists the
  decision permanently to `UserDefaults` under the
  `rapid.banner.upgradeFromBundled.suppressed` key; once set the
  banner never fires again on this install. The banner also
  suppresses itself automatically once you've upgraded to a non-
  bundled model — no extra clicks required.

## [0.7.1] — 2026-06-16

First-paint UX fix. v0.7.0 shipped with `gemma-4-12b-4bit` (or whichever
`RAMBucketedDefault` slot matched your Mac's RAM bucket) as the
first-launch model, which meant the very first thing a brand-new user
saw was a "Downloading 0/9 files" progress bar that either stalled on
`cas-bridge.xethub.hf.co` or hit the `huggingface_hub` 10s read-timeout
and never recovered (#229 / "stuck on download" reports). The chat
composer was inactive the entire time. By the time the user had typed
"hi" the app had already failed for them.

v0.7.1 bundles a 320 MB `qwen3-0.6b-4bit` model inside the DMG so the
first chat happens with **zero HuggingFace round-trips**:

- **Bundled instant-on model.** `mlx-community/Qwen3-0.6B-4bit` (~320 MB
  on-disk) is staged into `Contents/Resources/models/hf-cache/hub/` at
  build time. On first launch the desktop symlinks it into the user's
  `~/.cache/huggingface/hub/` so the sidecar resolves it like any
  cached model — no download, no network. Subsequent launches preserve
  the user's chosen model.
- **First-launch alias = bundled.** Both the auto-restart task in
  `ContentView` and the picker's `recommendedDefault()` consult
  `BundledModel.firstLaunchAlias()` before falling through to the
  `RAMBucketedDefault` bucketed alias. The bundled choice only wins on
  fresh installs (no `lastServedAlias` UserDefaults key); a user who
  has already traded up to qwen3.6-35b on a previous launch isn't yanked
  back to the 0.6B model.
- **Hybrid thinking already OFF by default.** The bundled Qwen3-0.6B is
  a hybrid model that defaults `<think>` ON; with a 256-token max it
  would burn the entire budget on chain-of-thought and emit an empty
  visible answer. `SamplingConfig.enableThinkingDefault` is already
  `false` desktop-wide (LM Studio / ChatGPT shape), so the bundled
  model gets `chat_template_kwargs: {enable_thinking: false}` on every
  turn out of the box — no per-alias gating needed.

Sidecar version unchanged: still ships `rapid-mlx` 0.7.25 (was 0.7.20
in v0.7.0). The only sidecar-side change is the new `qwen3-0.6b-4bit`
alias entry the bundled weights resolve to (raullenchai/Rapid-MLX#624).

DMG size grows from ~175 MB to ~495 MB — that's the cost of
making "open the app → instant chat" a real promise instead of an
aspiration. Fixes #229.

## [0.7.0] — 2026-06-16

Launch release. Bundles `rapid-mlx` v0.7.20 (was v0.7.15 in v0.6.14)
which closes two production bugs surfaced during the v0.6.13/v0.6.14
fresh-MBP smoke walks:

- **rapid-mlx #611 — `ignore_eos` leak between requests** (Raullen-MLX
  scheduler bug). A benchmark probe with `ignore_eos=true` followed by
  a normal chat call silently shared the previous request's
  BatchGenerator and inherited its empty `stop_tokens` set — the chat
  ran to `max_tokens` instead of stopping at EOS. End-user impact: any
  app that mixed `rapid-mlx bench` against a long-running `rapid-mlx
  serve` could see chat responses run on indefinitely. Fix folds
  `ignore_eos` into the generator reuse key AND refuses admission to a
  running generator with mismatched params so overlapping batches stay
  correct too.

- **rapid-desktop #230 — sidecar `.pyc` writes broke codesign seal**.
  Confirmed in the v0.6.14 bundle: `codesign --verify --deep` reported
  1008 stray `.pyc` files under
  `Contents/Resources/rapid-mlx/python/lib/python3.12/.../__pycache__/`
  from CPython's runtime bytecode cache. The fresh DMG launched fine,
  but Migration Assistant copy / macOS major upgrade / `mv` of the
  bundle would re-trigger Gatekeeper, which now sees a seal-mismatch
  and refuses with "App is damaged, move to Trash". Fix pre-compiles
  the bundled stdlib + site-packages with `compileall` BEFORE
  codesigning (frozen `SOURCE_DATE_EPOCH` for reproducible bytes), and
  sets `PYTHONDONTWRITEBYTECODE=1` in the shim as belt-and-braces.

- **rapid-desktop #229 — Settings tab list outdated**. `docs/userflows.md`
  F4 spec still listed 8 tabs; v0.6.13 has 11. Docs-only fix; ensures
  the cliclick walkthru tree-walk maps to current UI.

Also bundles **8 new model aliases** in rapid-mlx (Kimi-K2.6,
Qwen3-8B-4bit, Qwen2.5-14B-4bit, gemma-3 QAT 1B + 4B + 27B, gemma-4
e2b + e4b) and drops the over-aligned `gemma-3n-e4b-4bit`.

This is the first 0.7.x marquee release. The 0.6.x line is feature-
complete; subsequent 0.7.x ships continue the cliclick walkthru ⇄
production-hardening cycle.

## [0.6.14] — 2026-06-16

Launchable replacement for v0.6.12 / v0.6.13. Those two tags were
published direct to R2 without a git tag, so `release.yml`
(`push: tags: v*`) never fired and the DMG shipped ad-hoc signed
without an Apple notary ticket — Safari/Chrome downloads triggered
the Gatekeeper "Apple could not verify … Move to Trash" dialog on
first launch. The in-app auto-update path was unaffected (the
`Installer` doesn't go through Gatekeeper) but a fresh download from
the landing page was dead in the water.

v0.6.14 reaches R2 via the standard tag → `release.yml` path that
v0.6.6 – v0.6.11 used: build → Developer ID sign → notarize +
staple .app + .dmg → upload to GitHub Releases + R2. The R2 alias
`rapid-mlx-desktop.dmg` is the public download CTA on
`rapidmlx.com/desktop` and is overwritten on every tag.

This release also bumps the bundled `rapid-mlx` sidecar from v0.7.11
to v0.7.15 (codex-CLI fixes, logger namespace rebrand from
`vllm_mlx.*` to `rapid_mlx.*`, openai-harmony 0.0.6→0.0.8 +
mlx-embeddings 0.0.5→0.1.0 deps refresh, community-bench novice
friction fixes).

## [0.6.13] — 2026-06-16

End-to-end smoke release for the v0.6.12 R2 auto-updater. No
user-visible code changes — only the version bump. Confirms a
v0.6.12 client polls `dl.rapidmlx.com/latest.json`, sees v0.6.13
available, and the in-app Installer drives download → mount →
codesign verify → swap → relaunch without manual intervention.

## [0.6.12] — 2026-06-16

A plumbing release with no user-visible feature changes. The in-app
auto-updater that had been silently broken on v0.6.9–v0.6.11 ("Last
check failed: update server returned HTTP 530") now works again on
a simpler architecture: a static JSON file on the same Cloudflare
CDN that already serves the DMG. No proxy worker, no PAT, no GH
API rate-limit. Once you're on v0.6.12, the next release picks up
automatically.

### Changed

- **Auto-update channel moved from `update.rapidmlx.com` (Cloudflare
  Worker proxying GitHub Releases) to a static manifest at
  `https://dl.rapidmlx.com/latest.json` on R2.** The worker was
  never finished provisioning (HTTP 530 in production); v0.6.11
  users saw "Last check failed: HTTP 530" on every poll. From
  v0.6.12 on the app polls a public, no-auth JSON file on the
  same Cloudflare CDN that already serves the DMG. No GitHub
  PAT, no GH API rate-limit, no proxy worker. The host allowlist
  picked up `rapidmlx.com` + `www.rapidmlx.com` (the manifest's
  release-notes link points at the public landing page; the
  rapid-desktop repo is private, so a github.com URL would 404
  for most users). The in-app installer (`Installer.swift`) was
  already R2-ready since v0.5.22. Companion PR:
  [rapidmlx.com#8](https://github.com/raullenchai/rapidmlx.com/pull/8).
  Closes [#218](https://github.com/machinefi/rapid-desktop/issues/218).

## [0.6.11] — 2026-06-16

A follow-on download polish. v0.6.10 capped how big the xet client
can grow (no more 512-range floods on home routers), but the
adaptive controller still starts at concurrency=1 and ramps up over
the first 30–60 seconds — so the very beginning of every fresh
download looked slow. Our target user is on a laptop on home WiFi,
not a multi-gigabit backbone; the ramp was designed for a network
shape they'll never have. v0.6.11 pins the per-file stream count to
8 from t=0, which matches the "fixed N parts per file" pattern that
makes Ollama's downloads feel instant.

### Changed

- **First-second-of-download is no longer artificially slow.** Set
  `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8` on every `rapid-mlx pull`
  we spawn. The Hugging Face docs surface this as the bypass for
  the adaptive controller; with 2 parallel files × 8 fixed streams
  the total stays at 16 ranges (same upper bound as v0.6.10) but
  the per-file ramp-up tail is gone. Single-stream HF throughput
  caps around 30–100 Mbps, so 8 streams ≈ 240–800 Mbps total —
  saturates a WiFi 6 / 1 Gbps fibre link without overshooting it.
  Power users on multi-gig downlink can override either knob via
  shell export.

## [0.6.10] — 2026-06-16

A first-run polish release. v0.6.9 shipped two regressions the cliclick
sweep caught the same evening: the picker-v2 dropdown dropped its
recommended-row aliases and the cached green dot, and a fresh model
download could peg the user's home network. v0.6.10 fixes both so the
first 60 seconds with a clean install feel calm instead of confusing.

### Fixed

- **Picker dropdown renders the role-row alias and the cached
  cue again.** SwiftUI `Menu` wraps each Button as an `NSMenuItem`,
  and `NSMenuItem` honours only the first `Text` inside the Button's
  label — `HStack` siblings, `Spacer`, the trailing `Circle`, and
  background fills were all silently dropped. The recommended
  section therefore showed only "Default" / "Speed" / "Quality" /
  "Coding" / "Vision" with the alias gone, and the "All models" list
  lost the on-disk green dot. Each row now folds into a single
  `Text` with the state cue on a leading SF Symbol via `Label`
  (which `NSMenuItem` does render at the gutter): leading checkmark
  marks the currently-selected role row, filled circle marks
  on-disk aliases, dashed circle marks not-yet-downloaded ones.
  VoiceOver gets an explicit composed accessibility label so the
  cached/uncached cue isn't a sighted-only signal. (#219)

### Changed

- **Model downloads no longer saturate the home network.** Stock
  `huggingface_hub.snapshot_download` fans out to 8 parallel files
  with each xet-backed file ramping its own range-stream concurrency
  from 1 → 64 — worst case ~512 simultaneous TCP ranges with zero
  global cap. Field reports of "Slack and Zoom freeze while I'm
  pulling a model" matched the bufferbloat symptom exactly. We now
  cap the xet client at 2 files × 8 streams (16 ranges) by default,
  set as env vars on every `rapid-mlx pull` we spawn. Single-stream
  throughput is unchanged (xet chunking already fills a 1 Gbps pipe
  on one stream); the 64-way ramp was designed for multi-gigabit
  backbones, not consumer routers. Power users can override either
  knob by exporting `HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=…`
  or `HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY=…` from the parent
  shell — the inherited environment wins. Stall-detection +
  auto-retry (Option 2 from the research report) follows in the
  next sprint.

## [0.6.9] — 2026-06-16

The "drop the CLI, surface the engine through plain UI" release. v0.6.6
bundled `rapid-mlx` as a sidecar inside the .app, but the chrome was
still in transition — the bottom-bar chip named the engine version,
a "rapid-mlx" Settings tab walked users through PyPI installs that no
longer existed, and a launch-time upgrade banner nagged about an engine
they couldn't independently upgrade anyway. v0.6.9 closes that chapter
and replaces it with a calmer, more honest UX: the app names ITS OWN
version, the Settings sidebar names what users can actually configure,
and a new Model Management surface gives the cache the file-manager
shape users expected.

- **Bottom-bar chip names Rapid Desktop, not rapid-mlx.** Three states,
  three colours: green dot + "Rapid Desktop X.Y.Z · up to date" when
  current, amber dot + "update A.B.C available" when behind, no dot +
  bare version when the first check is in flight or briefly failed
  (we never paint red — a flaky-network blip shouldn't masquerade as
  a fault). Click deep-links to **Settings → App**, the existing
  in-app installer surface. The chip no longer routes to GitHub; the
  source repo is private and a `github.com` nav would 404 for end
  users. (Closes #216.)
- **Settings → Model Management.** New top-level tab gives the
  HF cache a real file-manager treatment: search + filter (All /
  Cached / Not cached) + sort, per-alias status badges (Cached /
  Downloading / Not cached) with size in GB, Download + Cancel +
  Delete actions, and an aggregate "X cached · Y.Z GB" footer.
  Clean replacement for the side-quest "Models" tab's deletion menu.
  (Closes #210 via #213.)
- **Model picker shows the actual alias next to each role.** Top of
  the picker now reads `Default qwen3.5-9b-4bit`, `Speed
  llama-3.1-8b-4bit`, `Quality gemma-4-12b-4bit`, … — selected role
  paints amber so it's scannable at a glance. The previous "Cached"
  section is gone; an "All models" alphabetical list with a green dot
  next to on-disk aliases replaces it (disk state belongs in Model
  Management, not the picker). (#214.)
- **"This won't fit your Mac" confirmation before reload.** Picking
  a model whose memory estimate exceeds usable RAM now triggers a
  destructive-style alert front-loading both the alias and the Mac's
  RAM ("`qwen3.6-122b-a10b-4bit` likely won't fit your 18 GB Mac"),
  the estimated need vs. available, and a clear "Start anyway"
  button. Cancel (default Return) reverts the picker. Same alert
  fires from the Start CTA AND the Switch-and-reload flow so neither
  path can lead a user into an OOM by accident. (#214.)
- **Sidebar toggle icon removed from Settings.** The chevron in the
  Settings titlebar used to suggest you could collapse the category
  list, but the click did nothing — a Mac 14 SwiftUI limitation. The
  Settings panel is now plain `HStack(sidebar, detail)` so the icon
  doesn't appear in the titlebar at all. (Closes #209.)
- **Removed: the entire CLI surface.** `Settings → rapid-mlx` tab,
  the inline "upgrade your CLI" banner, the "your CLI didn't
  respond" probe-failure banner, and the launch-time upgrade modal
  all retired. With `rapid-mlx` bundled as a sidecar, none of these
  pointed at an action a user could take. Power users still get
  `RAPID_BIN` env var + the in-app updater's runtime-override slot
  for explicit overrides. (Net code change: -2.2k LOC.)

### Repo move
- **`raullenchai/rapid-desktop` → `machinefi/rapid-desktop`.** The
  app lives at the org now. In-source links updated; user-facing
  copy already pointed at `machinefi`. (Closes #212.)

## [0.6.8] — 2026-06-16

The "polish pass" release. v0.6.7 shipped four launch-blocker fixes;
v0.6.8 is the overnight follow-up that exercised the resulting build
across Qwen 3.5, Qwen 3.6, and 3.6-A3B-35B on real chat / coding /
multi-turn flows, then closed every paper-cut that surfaced. Two
Mac-convention fixes lead the list; the rest are readable-error /
correct-PATH / consistent-copy work.

- **Persisted Light/Dark choice survives relaunch.** Picking Light
  in Settings used to read back as "Light" on next launch but the
  chrome would still render dark until the toggle was re-clicked.
  `AppearanceConfig.apply()` was losing a race with `NSApp.activate`
  on cold launch — the persisted preference didn't drive the first
  paint. Now applied from `applicationDidFinishLaunching` after the
  activate step. (Closes #204.)
- **Dock-click reopens the chat window.** After `Cmd+W` closed the
  chat surface, the Dock icon click did nothing — the app was alive
  in the menu bar but there was no way back to the conversation
  without Quit+Relaunch. Now wires `applicationShouldHandleReopen`
  → `openWindow`, matching standard Mac behaviour. (Closes #200.)
- **In-app upgrade works on Intel Macs.** The "Upgrade now" button
  used to hard-code `/opt/homebrew/bin/brew` (and `pipx`), which
  silently exited 127 on Intel Macs where the binary lives at
  `/usr/local/bin/brew`. Now walks `PATH` plus a curated fallback
  set (Apple Silicon brew, Intel brew, MacPorts, `~/.local/bin`),
  so the upgrade button actually fires on either architecture.
  (Closes #199.)
- **Settings → App error banner is readable.** The Releases-API
  self-check used to fail with `Rapid.UpdateError error 1` —
  Swift's auto-synthesized message from a raw enum. Now reads
  `Transport error: …` etc. via a `LocalizedError` conformance.
  Same fix applied to `CLIVersionError` in Settings → rapid-mlx,
  which used to read `Rapid.CLIVersionError error 0` and now
  reads `PyPI returned HTTP 503` etc. (Closes #201, #202.)
- **`web_search` tool description is provider-agnostic.** The tool
  description advertised "Search the web via **DuckDuckGo** …" even
  when the user had configured Brave or Tavily. Models would then
  refer to "DuckDuckGo results" in their explanations regardless of
  what actually ran. Now reads "Search the web and get the top
  results …" — the provider stays an implementation detail.
  (Closes #205.)
- **Stale `v0.4.12` caption removed.** Sampling panel still bragged
  "Defaults match v0.4.12. Changes persist …" — load-bearing half
  is the persist note. v0.4.12-pin reference dropped. (Closes #203.)
- **Tooltips renamed `Settings → CLI` → `Settings → rapid-mlx`.**
  v0.5.x renamed the sidebar tab but 17 tooltips, comments, and
  test strings still referred to "Settings → CLI". Swept all six
  files. (Closes #206.)
- **System-prompt sheet copy uses plain prose.** The sheet
  description used DocC ` ``role: system`` ` syntax inside a SwiftUI
  `Text`, which CommonMark-parses to monospaced "role: system" —
  not literal backticks, but still wire-format jargon the end user
  shouldn't need to know. Rewritten as plain prose. (Closes #207.)

## [0.6.7] — 2026-06-15

The "first-launch polish" release. Same day as v0.6.6 — these are
the four launch-blocker fixes that surfaced from the first round of
fresh-MBP install verification. Picker, web search, and updater
surfaces all rework themselves around the bundled-`rapid-mlx`
reality.

- **Role-based model recommendations in the picker.** The model
  picker now opens with a "Recommended for your N GB Mac" header
  followed by five concrete suggestions — Default, Speed, Quality,
  Coding, Vision — picked from a hand-curated table that fits your
  RAM. Diverse families (Qwen, Gemma, OpenAI gpt-oss, Llama,
  DeepSeek) rather than five Qwens. No more 🔴/🟡/🟢 fit warnings on
  recommended models — if it's in the list, it fits. (Closes #197.)
- **Updater retargets the bundled CLI.** Now that every install ships
  with `rapid-mlx` inside the `.app`, the old "PyPI has 0.7.12, you
  should run `pip install --upgrade rapid-mlx`" banner was wrong —
  the bundled CLI moves when `Rapid-MLX Desktop` ships a new DMG, not
  on PyPI cadence. The CLI banner is suppressed for bundled installs;
  Settings → App now hosts the version/recheck UI for the desktop
  build itself. (Closes #191.)
- **First-use web-search onboarding.** Fresh installs get a new
  onboarding-tour page explaining DuckDuckGo's flakiness on hot
  topics (World Cup scores, breaking news) and pointing at Brave /
  Tavily key signup. Existing Settings → Web Search picks up an
  inline "Want better results?" banner when the active provider is
  DDG and no upgrade key is stored. (Closes #193.)
- **Settings sidebar locked + WebSearch Save button.** The Settings
  sidebar no longer collapses on accidental drag (was a common
  "where did everything go?" report). The Web Search API-key row
  now has an explicit Save button + transient "Saved ✓ Stored in
  macOS Keychain" feedback so users know the key actually persisted
  — the old click-out-to-save flow left people unsure whether
  they'd lost their key. (Closes #176, #183.)

## [0.6.6] — 2026-06-15

The "no brew install required" release. v0.6.6 is the first version
that ships `rapid-mlx` itself inside the `.app` — drag to
Applications and launch, no terminal step. Plus a handful of
tool-call + web-search reliability fixes layered on top.

- **Bundled `rapid-mlx`.** The DMG now ships a self-contained
  `rapid-mlx` (embedded Python 3.12 + signed venv + codesigned
  binary, ~110 MB compressed) at
  `Contents/Resources/rapid-mlx/`. `ServerLocator` prefers this
  over Homebrew / pipx / uv when both are present. The About
  panel shows **Bundled v0.7.11** so bug-report screenshots
  identify exactly which CLI version is embedded. Power users with
  their own `rapid-mlx` install can still point Rapid at it via
  the `RAPID_BIN` environment variable. (Closes #171.)
- **Cleaner cap-hit guidance.** When the tool-call retry budget
  runs out mid-conversation, the chat surfaces a model-aware
  explanation ("gpt-oss tends to loop on browser builtins — try
  Qwen 3.5 / 3.6 for this task") instead of the previous generic
  "limit reached" string. (Closes #185.)
- **Web-search auto-promote on first key paste.** Paste a Brave or
  Tavily key into Settings → Web Search while the provider is
  still DuckDuckGo and the provider switches over automatically —
  no extra dropdown click. Once promoted, the choice stays sticky;
  Rapid never silently demotes a paid provider back to DDG.
- **DuckDuckGo anti-bot detection.** When DDG's HTML scrape
  returns their anti-bot modal (increasingly common), the
  `web_search` tool now returns a clear error pointing at Brave /
  Tavily instead of pretending zero results came back. Bug-report
  friendly: you see the block, not a vague "no results".
- **Orphan `rapid-mlx` subprocess cleanup.** When the app crashes
  or is force-quit mid-launch, the next launch's port sweep now
  correctly identifies and reaps the orphan even when it was
  spawned via a `python` shebang wrapper (e.g. brew installs) or
  through a path containing spaces. (Closes #170.)
- **Docs + Cask refresh.** README, Cask manifest, and the
  rapidmlx.com landing-page references all moved from "Rapid" to
  "Rapid-MLX Desktop" so the rename from v0.5.22 reads
  consistently everywhere. (Closes #167.)

## [0.6.5] — 2026-06-15

The "uv users can upgrade too" patch. v0.6.3 fixed the Source classifier
for Homebrew and pipx installs, but `uv tool install rapid-mlx` —
which puts the binary at
`~/.local/share/uv/tools/rapid-mlx/bin/rapid-mlx` — was still
mis-handled in two places:

- Settings → Source read **Unknown location** because `ServerLocator.classify`
  only recognised the pipx variants of `~/.local`, not uv's strict
  subpath under `~/.local/share/uv/tools/...`.
- The **Upgrade now** banner ran `pipx upgrade rapid-mlx` — because
  `CLIVersionChecker.classify` matched the broad `~/.local/` prefix
  first and called it pipx. On a uv-only Mac that's ENOENT (no pipx
  binary), and if pipx happens to be installed it says "rapid-mlx
  not managed by pipx" — either way the button does nothing useful.

- **Distinguish uv from pipx before the prefix fallback.** Both
  classifiers now match `~/.local/share/uv/tools/rapid-mlx/`
  (and `$XDG_DATA_HOME/uv/tools/...` for users who set it)
  BEFORE walking the generic `~/.local/` rules. Source reads
  "uv" and the Upgrade button runs `uv tool upgrade rapid-mlx`
  via the first `uv` executable it finds on PATH
  (`~/.local/bin/uv`, then Homebrew prefixes).
- **Tests pin both paths.** ServerLocatorTests covers default
  + XDG_DATA_HOME override + that pipx territory isn't
  accidentally swallowed; CLIVersionCheckerTests covers the
  classifier branch ordering plus the suggested-command
  string and `canRunUpgradeInProcess` gate.

## [0.6.4] — 2026-06-15

The "stop making up match results" patch. User-reported earlier the
same day: asking qwen3.5-9b-4bit about the 2026 World Cup with
`web_search` enabled fired the tool, got faithful DDG snippets back
(hosts, dates, 48 teams, opening match Mexico vs South Africa), then
the model **fabricated 8 group-stage match results** — Italy 2-0
Poland, Russia 1-0 Hungary, Sweden 2-2 Chile — none of which are in
the snippets and none of which are even playing in the tournament.

The unit-isolated /v1/chat/completions probe with the same DDG
fixture answered correctly. So the bug was the chat-surface flow,
not the model in isolation: when the user hasn't set a per-session
system prompt and tools are advertised, nothing nudges the model
away from confabulation, and small models latch onto a single real
fact in the snippets and auto-complete the rest from training-data
priors.

- **Ambient tool-guidance system message.** When tools are
  advertised on the wire (web search / weather / file read /
  anything in the registry the user hasn't disabled), prepend a
  fixed system message that names every rule we need:
  ground in tool output verbatim, say "the snippets don't cover
  that" instead of inventing, never fabricate scoreboards or
  group-stage tables, ask before answering an ambiguous question.
  The user's per-session system prompt (if set) still goes onto
  the wire AFTER the guidance, so a power user who wants the
  model to roleplay or brainstorm freely can override.
- **Probe matrix release SOP.** New `docs/release/probe-matrix.md`
  defines a 7-category behavioural gate (current-events RAG, code
  generation, multi-step reasoning, tool routing, multi-turn
  coherence, refusal calibration, bilingual code) the release
  engineer runs against every first-touch default alias before
  tagging. Runner at `scripts/probe-matrix.py`; results land in
  `reports/probe-matrix/YYYY-MM-DD-<alias>.json` plus a markdown
  scorecard. The CE-01 probe in the matrix replays the exact
  failure shape this release fixes — so the regression can't
  re-emerge silently.

## [0.6.3] — 2026-06-15

The "where's my rapid-mlx coming from?" patch. Settings → CLI was
reporting **Source: Unknown location** for every Homebrew-installed
``rapid-mlx`` user. Root cause: ``ServerLocator.find`` resolves
symlinks before returning, so a brew install at
``/opt/homebrew/bin/rapid-mlx`` arrives at the classifier as
``/opt/homebrew/Cellar/rapid-mlx/<version>/bin/rapid-mlx``. The
classifier's exact-path match against ``/opt/homebrew/bin/rapid-mlx``
silently fell through to ``.unknown``. Same trap on Intel-brew
(``/usr/local``) and pipx (``~/.local/pipx/venvs/...``).

- **Brew & pipx classified correctly post-symlink-resolution.**
  Accept both the symlink and the resolved Cellar/venv target. The
  Source row now reads "Homebrew (Apple Silicon)" / "Homebrew
  (Intel)" / "pipx" for users on those installs, matching what
  Install kind already reported. No behaviour change for users who
  saw the right label before.

## [0.6.2] — 2026-06-15

The "wait, why didn't that click work?" patch. The orange **New
chat** button at the top of the sidebar had a hit-area bug where
only the left half (where the "✏️ New chat" label is drawn)
actually fired the click handler — the right half of the amber
rectangle looked clickable but taps there fell through to the
session row beneath. User-reported during the v0.6.1 N2N walk.

- **Entire "New chat" button is now tap-active.** Adds an
  explicit ``.contentShape(Rectangle())`` so SwiftUI's ``.plain``
  button style hit-tests the full sidebar width instead of just
  the leading-aligned label content. No visible change (the
  amber rectangle, hover brighten, ⌘N shortcut all unchanged) —
  taps on the right half now register as a New chat action like
  taps on the left half always did.

## [0.6.1] — 2026-06-15

The first-touch UX patch. v0.6.0's N2N walk-through surfaced one
dominant failure mode: on a 128 GB+ Mac (M2/M3 Ultra cohort), the
RAM-bucketed first-launch default model was ``qwen3.5-122b-mxfp4``
(~65 GB download). A fresh user opening the app was handed a
30+ minute wait before the first prompt could fire — far past the
point where most people give the app a chance.

- **Conservative first-touch default for 128 GB+ Macs.** The
  ≥ 49 GB bucket now collapses into a single open-ended range and
  defaults to ``qwen3.6-35b-4bit`` (the A3B MoE — 3 B active
  params, ~18 GB download, 3-5 min on most home connections, runs
  at near-122B quality on Apple Silicon). The 122B alias stays
  in the model picker — power users who explicitly want the
  bigger model can still pick it — but it is no longer the first-
  touch default for the cohort with enough RAM to run it.

## [0.6.0] — 2026-06-15

The capability-chip honesty release. When you tap a Search the web
chip on an empty conversation, the next turn now actually invokes
``web_search`` — not whatever else the model felt like routing
through. Fixes a ~50% misroute observed on qwen3.6-35b-4bit during
the 2026-06-14 cross-model probe where "Search the web for X" would
silently fire ``current_datetime`` and answer about today's date.

- **Capability chips pin their promised tool on the first turn.**
  The four empty-state CTA chips (Search the web → ``web_search``,
  Calculate → ``calculator``, Weather → ``weather``, Read files →
  ``read_file``) now bias the next chat send's ``tool_choice`` to
  the chip-named tool instead of the default ``"auto"``. The bias
  applies to the FIRST tool-loop round only; rounds 2+ fall back
  to ``auto`` so multi-step compositions (search → summarise) still
  work. Free-typed prompts are unaffected — they keep
  ``tool_choice=auto`` end-to-end. The bias drops automatically
  when you edit the chip's seed prefix out of the draft, tap an
  example prompt, or switch sessions. Closes #141.

## [0.5.22] — 2026-06-15

The naming-cleanup release: the desktop app is now officially
**Rapid-MLX Desktop** everywhere a user reads it, and the CLI
stays **rapid-mlx**. The bare word "Rapid" was always an internal
codename; this release retires it from every user-visible surface.

Plus a smarter first-launch model pick: instead of greying out
the Start button until you read the model picker, the app now
proposes a sensible default for your specific Mac's RAM bracket
the moment you open the picker, with explicit safety against
edge cases like an unparseable custom alias or an old big-model
cache from a previous install.

- **First-launch default model based on your Mac's RAM.** The
  model picker now seeds itself with a sensible default chosen
  from a hand-tuned RAM bracket table — 16 GB Air → 4B, 17–36 GB
  M-Pro → 9B, 37–48 GB M-Max → 27B, 49–127 GB Ultra → 35B-A3B,
  128+ GB Ultra → 122B — every bracket vetted against
  ``ModelSizing.classify`` so the proposed default is never
  ``.tooBig``. On an unusual hardware-floor case (8 GB Mac, or a
  catalog of mostly unparseable custom aliases), a safe-fallback
  ranking still hands back the smallest known-fit alias rather
  than a cached too-big or unestimable entry. Closes #163.

- **Renamed: "Rapid" → "Rapid-MLX Desktop".** Every user-visible
  product-name string — window title, About panel, MenuBarExtra
  Open/Quit/About entries, Update window, sidebar wordmark,
  empty-state hero, accessibility descriptions — now reads
  "Rapid-MLX Desktop". The CLI is "rapid-mlx" (unchanged). The
  internal Swift module + bundle identifier (`com.rapidmlx.rapid`)
  + Application Support directory (`~/Library/Application
  Support/Rapid/`) are deliberately stable so existing chat
  history, settings, and auto-update keep working seamlessly.
  Closes #164.

- **DMG renamed to `rapid-mlx-desktop.dmg`** with the legacy
  `Rapid.dmg` URL kept as a permanent R2 alias — older landing-
  page caches, blog posts, and browser bookmarks pointing at
  `https://dl.rapidmlx.com/Rapid.dmg` continue to resolve.
  Existing installs auto-update in place via the Updater helper,
  preserving the on-disk app-bundle filename — only fresh DMG
  downloads land at `/Applications/Rapid-MLX Desktop.app`.

## [0.5.21] — 2026-06-15

A focused hardening drop closing three P2/P3 polish items from the v1
prod-readiness audit. No new features, no API or wire-format changes —
upgrade is drop-in.

- **Destructive-pattern warning on tool calls.** When the model proposes
  a shell command that matches a hostile shape — `rm -rf /`, `dd
  if=/dev/...`, `curl … | sh`, env-var exfiltration like
  `${HF_TOKEN}`, `chmod 777`, `chown root:`, `base64 -d | sh`,
  `/dev/tcp/`, `sudo` — the tool-call chip turns red and pins the
  matched pattern in a tooltip + VoiceOver caption. The matches stay
  visible for the whole stream (no flicker as the model adds more
  arguments) and update only when the union of detected patterns
  actually changes. Closes #140.

- **Settings → Storage pane.** A read-only gauge for how much disk
  your chat history is using — session count, message count,
  `sessions.json` size, attachment-blob size, total — plus a
  Reveal in Finder shortcut and an opt-in archival window picker
  (never / 30d / 90d / 180d / 1y). The picker persists your
  preference but does **not** auto-delete anything in this release;
  silent deletion of chat history is a hostile failure mode and the
  v1 audit explicitly rejected it. The setting is captured now so a
  future build can act on it. Closes #118.

- **Test hygiene: no more leftover plist files in `Preferences/`.**
  Five test suites were minting `UserDefaults(suiteName:)` plists
  with UUID names and never cleaning them up — every CI / local
  `swift test` run quietly leaked a handful of `.plist` files into
  `~/Library/Preferences`. A shared `TestDefaultsScope` helper now
  guarantees teardown for every suite that touches namespaced
  defaults. Closes #139. (Test-only — no runtime change.)

Full Swift test suite: **1242 / 1242** passing.

## [0.5.20] — 2026-06-15

A focused follow-up to v0.5.19 closing the small-model "wrong answer
on follow-up" pathology end-to-end. Two wires meet here:

- **Server-side curated sampling now flows to the client.** Rapid-MLX
  0.7.6+ exposes per-alias `recommended_sampling`, hybrid-thinking +
  MoE flags, parser hints, and modality on `/v1/models/{id}`. The
  desktop fetches that profile when the server transitions to
  `ready(alias)` and, if you haven't overridden the Sampling sliders
  yourself, applies the server's curated values automatically. For
  Gemma-4 12B that's `temperature=1.0`, `top_p=0.95`, `top_k=64` —
  the values that consistently beat the model's bare
  `generation_config.json` on the canonical eval suite. No hand-tuning
  required; if you've already moved the sliders, your overrides win.

- **Old Rapid-MLX servers degrade gracefully.** A 0.7.5 or earlier
  server (no vendor-extension fields on `/v1/models`) returns the
  baseline OpenAI shape; the desktop falls back to the v0.4.12
  hard-coded defaults exactly as before. There's no toast, no
  error, no required version bump on the server side. Upgrading to
  Rapid-MLX 0.7.6 simply makes things better.

### Companion server fix bundled

This release is paired with Rapid-MLX 0.7.6 which also lands the
parser fix for #575 (Qwen3 thinking-ON non-stream leak — the
"wrong-date confabulation, flipped-answer on follow-up" pathology
reported on small hybrid models). With both halves shipped, the
follow-up-question flip mode that prompted the 2026-06-14
autoresearch sweep should be retired.

## [0.5.19] — 2026-06-14

A same-day follow-up to v0.5.18 closing three first-time-UX issues
surfaced by hands-on testing on the 18 GB MacBook Pro: prompts
silently returning empty bubbles on small hybrid-thinking models,
the menu-bar icon reading as a Gemini lookalike, and no way to
download / delete models from inside the app.

### Added
- **Settings → Models tab.** Browse the full model catalog, see
  what's in use / what's already on disk / what's available, kick
  off a download for a different model while the current one is
  loaded, and free disk space by deleting cached models. (#160)
- **"Show reasoning" toggle in Settings → Sampling.** Lets advanced
  users opt hybrid models (Qwen 3.x, GLM 4.7, Qwopus) back into a
  chain-of-thought trace. Off by default — see Fixed. (#161)

### Fixed
- **"Prompts don't work" on small hybrid-thinking models.** On a
  4 B / 9 B-class hybrid model with the default 4 K `max_tokens`
  budget, the `<think>...</think>` reasoning trace alone routinely
  consumed the entire budget, leaving the stream to terminate with
  zero answer tokens and the UI showing a 25-30 s spinner followed
  by an empty bubble. Every chat request now sends
  `chat_template_kwargs: {enable_thinking: false}` by default so
  the hybrid template skips its reasoning lane and emits the answer
  directly. Matches what ChatGPT / Claude Desktop ship. The
  zero-content safety-net error message also surfaces the new
  toggle directly when thinking was the cause. (#161)

### Changed
- **Menu-bar icon swap: sparkles → cheetah.** The v0.5.18 sparkles
  glyph read as a Gemini lookalike on tooltip preview. Replaced
  with the brand cheetah-sm.png — colored when the asset bundle is
  reachable, falling back to a monochrome `hare.fill` SF Symbol on
  the rare missing-bundle path so the menu bar entry never
  silently vanishes. (#153)

## [0.5.18] — 2026-06-14

A same-day follow-up to v0.5.17. Three first-time-UX bugs that were
launch-blockers for new users: the download overlay couldn't show
download progress on `huggingface_hub` ≥0.20 (your 5 GB pull just
sat there reading "Resolving 1/12 files"), there was no menu-bar
entry to reach Rapid from anywhere on the system, and the empty
state's third example prompt could trigger a tool-call loop on
small (~9B) models.

### Added
- **Persistent menu-bar item.** A sparkles glyph next to the system
  clock with `Open Rapid` / `New Chat ⌘N` / `<alias> · <state>` /
  `Settings… ⌘,` / `Quit Rapid ⌘Q`. Reachable from anywhere on
  macOS; dock icon stays visible. (#151)
- **"Still downloading — Ns since last update" caption.** When the
  download overlay sits on the same `N of M files` counter for >8 s
  (the gap between large shard boundaries) the body adds a live
  caption telling the user it isn't deadlocked. (#150)

### Changed
- **Download overlay copy** says "Downloading" everywhere
  ("Resolving" was reading to first-time users as "DNS / network
  handshake is stalled"). The pill collapses the previous two-word
  split from #130 — see issue #150 for why both directions of the
  copy decision matter. (#150)
- **Footer status pill renames "CLI" → "rapid-mlx".** "CLI 0.7.3 ·
  up to date" was confusing for users who'd never typed `rapid-mlx`
  in a terminal; "rapid-mlx 0.7.3 · up to date" maps directly to
  the PyPI / Homebrew package identifier. Same goes for
  `checking…` / `check failed` / `not installed` / `<version>` /
  `<local> → <latest>` shapes.
- **First-run example prompt #3** swapped from a `web_search` call
  ("Search the web for the latest mlx-lm release notes") to a
  reliable plain-text prompt ("Write a haiku about Apple Silicon's
  unified memory"). Small models (~9B at 4-bit) often re-query
  `web_search` five or six times because they can't decide they're
  done — the first-run user saw a chain of green checkmarks and no
  answer. Drive-by from #151.

### Fixed
- **Per-file download progress now reaches the UI.** Root cause:
  `huggingface_hub` ≥0.20 emits its per-file tqdm bytes/speed with
  IEC suffixes (`2.10GiB/5.13GiB`, `23.4MiB/s`); our `isByteToken`
  only matched SI (`K`/`M`/`G`/`T`), so every per-file line silently
  failed `matchPerFile` and the overlay stayed pinned to the outer
  file-count tqdm for the entire multi-minute first-time download.
  4-phase parser state machine now accepts SI / SI+B / IEC marker /
  IEC+B shapes. (#150)
- **`rapid-mlx` subprocess env now sets `PYTHONUNBUFFERED=1` and
  `HF_HUB_DISABLE_PROGRESS_BARS=0`** so Python flushes tqdm
  line-by-line and an ambient user-shell export can't mute the
  download UX. (#150)

## [0.5.17] — 2026-06-14

The "fix the things that kept v1 from feeling done" release. Three
performance / correctness wins in the chat persistence layer (no more
Quit-hang on image-heavy sessions, no more sync disk read blocking the
first frame on cold start, math expressions actually render as math).
Plus a long list of UX rough edges sanded off across the chat surface,
empty state, picker pills, Settings, and onboarding tour.

### Added
- **LaTeX math rendering in chat replies.** Display math (`$$...$$`)
  emitted by a model now renders through native CoreText glyphs via
  `SwiftMath` — no `WKWebView`, no JavaScript, no network call.
  Inline `$...$` math stays as literal source for v1 to preserve
  table / list / heading layout (a follow-up will iterate). (#147)
- **Inline model-switch warning.** When the model picker shows a
  different model than the conversation was first sent to, the chat
  surface displays an orange caption above the compose bar flagging
  that chat template, context formatting, and tool support may
  differ. Suppressed for quant-only swaps on the same family+size
  (`4bit` ↔ `8bit`, Gemma QAT pairs, Bonsai unpacked variants).
  Per-session in-memory dismiss. (#58)
- **Pre-stream transient retry.** The chat client retries once on
  transient `URLError` codes (`.cannotConnectToHost`,
  `.networkConnectionLost`, `.dnsLookupFailed`) before surfacing a
  send failure — eliminates the common "send → bounce → fail"
  pattern when the inference server is spinning up. (#55)
- **Sidecar Phase 1 — multi-slot ServerLocator.** Rapid now resolves
  `rapid-mlx` through a classified slot list (bundled / Homebrew /
  pipx / PATH) with a clear precedence, surfaces the winning slot
  in Settings, and adds a Phase 3b "Use my own rapid-mlx install"
  override toggle so power users can pin a custom build. (#36, #38,
  #39, #43)
- **Settings — Keyboard shortcut cheatsheet pane.** New tab in
  Settings listing every chord (chat compose, picker, sidebar,
  Quick Ask, find-next/prev) so users don't have to spelunk the
  menu bar to remember `⌘G` or `⌥Space`. (#46)
- **Quick Ask launch-at-login toggle.** Settings → Quick Ask now
  has an opt-in to register Rapid with `SMAppService` so the
  `⌥Space` chord works even when the app isn't already running. (#64)
- **Runtime `/healthz` monitor for silent-crash detection.**
  Rapid now polls the inference server's `/healthz` endpoint on a
  short cadence; three consecutive failures flip the model picker
  pill to "Crashed" with the offending alias so the user notices
  before their next send 400s. (#65)
- **About panel surfaces ServerLocator origin.** "Powered by
  rapid-mlx" line in About now shows which slot (Bundled / Brew /
  pipx / PATH) and absolute path resolved at startup. (#89)
- **Onboarding tour — Quick Ask page.** A fourth onboarding card
  surfaces the `⌥Space` hotkey so new users discover it without
  hunting through Settings. (#92)
- **Inline `Bearer` auth on chat requests (server #17 desktop half).**
  Rapid now generates a per-launch bearer secret, hands it to the
  spawned server via a private env, and includes it on every chat
  / models / healthz request — closes a local-network spoofing
  vector where another loopback process could have served arbitrary
  completions to the user. (#145)

### Changed
- **`sessions.json` no longer inlines image-attachment bytes.**
  Image attachments now live in a content-addressed sibling blob
  store at `~/Library/Application Support/Rapid/attachments/<sha256>`;
  the on-disk envelope drops from ~5 MB-per-image down to a one-line
  hash reference. Migration is automatic on first launch after
  upgrade — your existing screenshots are rewritten in place, and
  Quit on heavy users no longer freezes for 1-2 s on `flushSync`.
  Closes #22. (#148)
- **Cold start no longer blocks the first frame.** `SessionStore`
  reads + decodes `sessions.json` on a detached `Task` and surfaces
  a loading state to the UI, so a heavy user with hundreds of
  sessions sees the sidebar spinner instead of a blank window.
  Closes #117. (#146)
- **SSE delta coalescer.** Streamed tokens batch on the read side
  before hopping to `@MainActor`, cutting per-token actor traffic
  from 1× to ~0.1× on a typical 60 tok/s stream — visible as
  smoother scroll-to-bottom on long replies. (#70)
- **WebSearch secrets cached in memory.** Decrypted Brave / Tavily
  API keys are cached for the lifetime of the config object instead
  of running the KeyChain unlock dance on every search call. (#109)
- **`Cmd+Space` / `Cmd+Comma` denied for Quick Ask chord.**
  Defensive validation at the model layer + an explicit UI
  rejection sheet so users don't trap themselves out of Spotlight
  / Settings. (#25, #108, #143)
- **Launch-time CLI upgrade modal dropped.** The "your rapid-mlx is
  out of date" launch-modal moved into a non-blocking banner — new
  installs land straight in chat instead of being interrupted by an
  upgrade prompt. (#134)
- **Window default height 880 → 820.** Default size now fits a
  13" MacBook Air M1 without clipping the compose bar. (#86)
- **Chat surface dynamic type clamped at xxxLarge.** Larger
  accessibility text sizes no longer overflow the bubble layout. (#68)
- **`PRIVACY.md`** documents the trust boundary between the signed
  Rapid Desktop client and the separately-installed `rapid-mlx`
  inference server. (#59)
- **`ChatStreamClient` default base URL** derives from
  `PortSweep.defaultPort` (single source of truth). (#52)
- **Telemetry PII redaction tightened.** Error event fields now go
  through a closed-set `error_type` whitelist + path/alias
  scrubbing before they leave the device. (#51)

### Fixed
- **`Cmd+,` no longer crashes the app.** Settings was missing a
  `ServerManager` injection; opening Settings on a fresh launch
  hit a fatal envEditor unwrap. (#122)
- **Crash reporter is signal-safe.** Crash handlers now use
  `sigaction` + a pre-`malloc`'d arena instead of touching the
  Swift heap on the signal path; a crash on top of a crash no
  longer wedges the process. Closes #24. (#144)
- **PortSweep no longer kills unrelated processes.** Rapid persists
  an owned-server record (PID / PGID / start time) and only
  signals processes whose record matches — closes a window where a
  port collision could have terminated an innocent `python3` on
  port 8000. Closes #20. (#142)
- **Quit during stream no longer leaks the connection.** In-flight
  chat stream is cancelled before `flushSync`. (#54)
- **`FilesystemTools.readFile` is TOCTOU-safe.** `O_NOFOLLOW` on
  the leaf rejects a symlink swap between the sandbox permission
  re-check and the `FileHandle` open. (#56)
- **`SSE` reader caps per-line size at 8 MB** so a malformed
  unbounded `data:` line from rapid-mlx can't OOM the renderer. (#48)
- **`list_directory` tool surfaces hidden files** with a per-entry
  flag and reports truncated + hidden counts back to the caller. (#49)
- **`ServerManager` log drawer scrubs `Bearer` / `api_key` /
  `token` patterns** from streamed stderr before display, so a
  log screenshot can't leak credentials. (#50)
- **Popped conversation window gets a Close button** when the
  underlying session has been deleted from the main sidebar. (#47)
- **Empty state — "Start chatting" CTA + `Cmd+N` hint.** The
  sidebar's empty state now invites the first action instead of
  silently waiting. (#44)
- **System prompt sheet — unsaved-changes guard.** Closing the
  System Prompt editor with edits in flight asks before discarding. (#40)
- **Model delete uses native confirmationDialog with Cancel as
  the default button** instead of a custom Alert that defaulted to
  Delete. (#41)
- **Download overlay shows tqdm ETA.** Starting / Downloading
  overlays surface the inference-server-reported ETA so a 4 GB
  pull doesn't look frozen. (#42)
- **Find-next / Find-prev hotkeys.** `⌘G` and `⇧⌘G` are wired in
  the chat search bar — previously the chord did nothing. (#79)
- **Model picker "Resolving" copy** on a cache-hit relaunch (was
  showing "Idle" for the first 200 ms). (#130, #135)
- **Model picker "Idle/Stopped" pill collapse.** Two off-states
  merged into one less-confusing pill. (#129, #136)
- **Empty-state CapabilityChip click prefills the compose draft.**
  Previously the chip was visually clickable but did nothing. (#123, #128)
- **Empty-state "Powered by" line names the runtime** (e.g.
  "Powered by rapid-mlx · qwen3.6-27b") instead of a static stub. (#125, #132)
- **Upgrade banner hides the raw shell command** for in-process
  install kinds (bundled / pipx) where the user can't run a
  Homebrew upgrade anyway. (#126, #137)
- **Picker "Fetching model list…"** loading copy replaces a
  generic spinner so the user knows what's happening. (#83)
- **VoiceOver accessibility.** Mascot a11y label dropped
  "placeholder"; pill HStacks collapsed so VoiceOver doesn't
  double-read each chip. (#84, #85)

## [0.5.16] — 2026-06-12

### Added
- **Non-technical onboarding copy.** Welcome tour and Settings
  copy rewritten to land for a reader who has never set up a
  local-LLM stack before. (#35)
- **Chat compose accessibility labels.** The compose `NSTextView`
  is now wrapped in an `NSAccessibilityElement` so VoiceOver
  announces it as "Message rapid-mlx, edit text." (#35)

## [0.5.15] — 2026-06-11

### Added
- **TPS pill.** Live tokens-per-second readout on the chat pop-out
  reasoning surface. (#34)

### Changed
- **Unified pop-out reasoning rendering.** Reasoning steps in the
  popped conversation window now share the same Markdown
  pipeline as the main chat view. (#33)

### Fixed
- **Footer overlap regression** on narrow windows. (#34)

## [0.5.14] — 2026-06-10

### Added
- **Pop-out conversation window.** Right-click a session in the
  sidebar → "Open in Window" to detach a conversation into its
  own SwiftUI window that stays in sync with the main app. (#31)
- **`docs/userflows.md`** — 8 documented user flows + a
  release-day smoke checklist. (#29)

### Fixed
- **Activation policy pinned at launch.** The app now reliably
  takes focus on launch when invoked from Spotlight / Finder. (#30)

## [0.5.13] — 2026-06-09

### Added
- **First production-readiness pass.** README, LICENSE, SECURITY,
  Cask formula, EULA, PRIVACY, third-party attribution. (#28, #14)
- **Telemetry + i18n bootstrap.** Anonymous opt-in telemetry to
  `telemetry.rapidmlx.com`; localization scaffolding. (#15)
- **Developer ID signing + notarisation** in the release CI. (#13)

### Changed
- **LoriZ brand polish.** Amber design tokens, RapidTheme,
  sidebar CTA, brand mascots. (#12)

_Notes: v0.5.13 also re-incorporates the v0.5.10–v0.5.12 fixes via
a backfill squash (commit `1497f7f`) — readers cross-referencing
tag dates will see the same fixes listed under both 0.5.10/11/12
and 0.5.13 commit ranges._

## [0.5.12] — 2026-06-09

### Fixed
- **Compose pill no longer balloons** when the draft contains a
  long single line — the height clamps to the documented
  `composeMaxHeight`. (`3772324`)
- **Sidebar rows breathe.** Per-row padding restored. (`3772324`)

## [0.5.11] — 2026-06-09

### Added
- **Silent context-window trim.** Dropping the token-meter chip;
  the chat client now silently trims the oldest turns to fit
  the active model's window. (`4d429e4`)

## [0.5.10] — 2026-06-08

### Fixed
- **`Bundle.module` lookup crash.** SPM's auto-generated
  `Bundle.module` accessor probed
  `<App>.app/<Target>_<Target>.bundle` (sibling to `Contents/`)
  which codesign rejects on the wrapped SwiftUI app. Switched
  to `Bundle.main` + graceful `BundleFinder` probe so a
  cold-launch under Launch Services no longer fatal-errors.
  (`8bec5dc`)

---

Older versions: see the
[GitHub Releases page](https://github.com/machinefi/rapid-desktop/releases)
for auto-generated notes against earlier tags.

[Unreleased]: https://github.com/raullenchai/Rapid-MLX/compare/rapid-mac-v0.12.16...HEAD
[0.5.16]: https://github.com/machinefi/rapid-desktop/compare/v0.5.15...v0.5.16
[0.5.15]: https://github.com/machinefi/rapid-desktop/compare/v0.5.14...v0.5.15
[0.5.14]: https://github.com/machinefi/rapid-desktop/compare/v0.5.13...v0.5.14
[0.5.13]: https://github.com/machinefi/rapid-desktop/compare/v0.5.12...v0.5.13
[0.5.12]: https://github.com/machinefi/rapid-desktop/compare/v0.5.11...v0.5.12
[0.5.11]: https://github.com/machinefi/rapid-desktop/compare/v0.5.10...v0.5.11
[0.5.10]: https://github.com/machinefi/rapid-desktop/compare/v0.5.9...v0.5.10
