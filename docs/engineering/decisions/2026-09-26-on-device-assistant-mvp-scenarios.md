# On-device assistant MVP: scenarios and technology stack

- **Status:** Proposed direction; evidence-based POC in flight
- **Date:** 2026-09-26
- **Owner:** Atlas
- **Consumers:** gui_verifier_cua_poc, Rapid Desktop (Personal Intelligence), always-on service, model catalog
- **Evidence base:** public reporting on the consumer personal-AI agent category (CNBC, LA Times, The Verge, Tom's Hardware, Ars Technica, Music Ally, Business Insider) plus three instrumented dogfood runs in
  `tools/gui_verifier_cua_poc/` (branch `atlas/gui-verifier-cua-poc`)

## Context

A cloud-hosted consumer personal-AI agent launched in September 2026 (its assistant surface,
model family, led by Nat Friedman). It reached 730k downloads in five days and
2.5M in thirteen (CNBC), was rated the top AI agent by J.P. Morgan, and forced
OpenAI to rush an always-on competitor. Public reporting establishes its shape:

- **Execution substrate:** per-user cloud Linux VMs (AMD EPYC, 2 cores, 8 GB)
  with terminal and browser access (Tom's Hardware). It is cloud-hosted and
  Meta collects a transaction fee (Zuckerberg, Yahoo Finance).
- **Headline scenarios:** shopping and travel booking ("can now book travel and
  shop for you", LA Times); money management (Yahoo Finance); an official
  Spotify connector (Music Ally, Spotify Newsroom); first-class filesystem
  access (The Verge); always-on proactive behavior.
- **Friction:** Amazon blocked that agent from shopping (GeekWire, Engadget) using
  standards it ignores for its own agent (TechTimes); a serious 0-day followed
  within weeks (Ars Technica) because an "extraordinarily privileged" agent is
  a high-value attack surface.

Goal: distill the category's proven user flows into 5-6 MVP scenarios for a **local,
Apple-Silicon-native equivalent on Rapid-MLX**, and name the technology stack
they force us to build. Walking every flow fully is not required; stopping at a
human gate (for example Amazon's final order button) is an accepted, expected
outcome whenever the operator declines permission.

## Top scenarios for the MVP

| # | Scenario | Category evidence | Our MVP status | Acceptance for "done" |
|---|---|---|---|---|
| 1 | **Agentic shopping**: NL goal → search → compare (rating × review count × sponsorship) → product page → cart → checkout → order | Category flagship; blocked by Amazon | Proven end-to-end to verified add-to-cart (cart-count postcondition 0→1, 1→2); place-order path implemented behind `CONFIRM_ORDER` gate | One operator-authorized run places a real, cancellable order; trace captures the whole chain |
| 2 | **Travel booking**: NL trip → search flights/hotels across sites → compare → fill passenger/guest forms → stop at payment | Announced capability of the category; high value | **Dogfooded on Google Flights**: the `press` action (added after clicks proved inert on Material UI comboboxes) unlocked trip-type, suggestion, and calendar widgets; search SFO→HND 2026-10-12 completed and GLM produced a three-option comparison (Cathay $658 1-stop vs United $961 nonstop vs JAL $961 nonstop, balanced pick United). Booking/payment intentionally not exercised | Search→select→form-fill completes; run stops at payment with a typed order summary awaiting approval — **search/compare acceptance met** |
| 3 | **Money assistant**: read-only account/spend digest; bill pay strictly behind a human gate | Category money flows; bank-partner coverage | Not started; browser lane + gate machinery already exist | Read-only digest of one bank dashboard; a payment flow demonstrably halts at the gate unexecuted |
| 4 | **Music/media**: create/curate playlists and queues from NL; discovery | Spotify is the category reference connector; operator declined account login, so dogfood ran on YouTube Music | **Dogfooded logged-out on YouTube Music**: search → play first instrumental track → add two more to the queue via each track's action menu (one menu needed the `press` Enter fallback). Trace shows player active (3:53/1:19:58) and `Song added to queue` toasts; no sign-in encountered | NL-driven search + playback + queue curation on the user's own machine, trace-verified; account-scoped actions (saved playlists) remain gated on `--human-login` |
| 5 | **Local files & desktop**: organize folders, find documents, cross-app desktop flows | Filesystem access is the category's highest-privilege surface | **Dogfooded end-to-end** (`tools/flow_tools/file_organizer.py`): NL rule → planner proposal → validation caught a hallucinated file → repair retry → APPROVE gate → 12 files moved with undo log | One rule-based Downloads organization with dry-run summary and per-batch confirm — **met** |
| 6 | **Always-on proactive agent**: price/deal watching, inbox digest, reminders | Competitors copy this; it is the category's retention hook | **Digest flow dogfooded** (`tools/flow_tools/digest.py`): 5 repo docs → urgency-sorted digest with action items (long-doc token fix applied); watcher scheduling pending | One watcher produces a daily human digest for a product price threshold |

Scenarios 1, 2, 3 share one pipeline (browser lane + money gate). Scenario 4 is
the low-risk quick win with an official integration path. Scenario 5 is our home
turf and does not depend on platform goodwill. Scenario 6 composes any of the
above on a schedule.

## Technology stack the scenarios force

All components are already exercised in `tools/gui_verifier_cua_poc/run_poc.py`
unless marked next:

1. **Planner lane** — OpenAI-compatible local vision model through Rapid serve
   with strict JSON schema and low reasoning effort. Proven: GLM-5.3-Flash-EXL3
   at 9-11 s/step; local Qwen 27B qualification is the replacement track.
2. **Fast decision lane** — laya on Rapid System One `/v1/rank`: typed outcome
   classification and list pre-ranking in ~0.3 s. This is our structural
   differentiator: cloud agents pay cloud tokens for every decision; we route most
   decisions to a 2B-class local model.
3. **Grounding verifier** — a 2B vision verifier checkpoint (mlx-vlm) confirms that a
   click's landing point matches intent. Proven value: rejected three
   ad-banner misclicks the planner insisted on during dogfood.
4. **Semantic action protocol** — typed `click/fill/submit(target_id)` actions,
   atomic fill+submit, typed postconditions (URL, DOM hash, focus, input value,
   cart count), deterministic bootstraps for known choreography, and
   `trace.json` audit of every plan, score, and fallback.
5. **Trust and safety layer** — page text is untrusted data; hard regex guards
   split research vs purchase modes (credentials/card data always blocked);
   human gates as filesystem sentinels (`RESUME`, `CONFIRM_ORDER`) with timeout
   fallback; least-privilege is the answer to the category's 0-day lesson.
6. **Execution lanes** — browser lane today (Playwright-controlled Chrome with a
   persistent user profile, human owns all sign-ins); native macOS Accessibility
   lane next (handoff item); local terminal lane with allowlists instead of
   a remote VM.
7. **Host integration** — Rapid Desktop surface plus the always-on service
   (ADR 2026-09-05) as the scheduler for scenario 6.

## Why this beats cloud agents for our users

- **Local-first privacy:** screenshots and page context never leave the Mac;
  Cloud agents process them on remote VMs.
- **No transaction tax:** Meta skims a fee from purchases; we have no such
  incentive and our money gate is opt-in by design.
- **Platform-block resilience:** Amazon locked a leading cloud agent out while running its own
  shopping agent. A human-paced local agent acting through the user's own
  session is a different trust model, and scenarios 4-6 do not depend on
  hostile platforms at all.

## Risks

- Platform ToS/anti-bot measures can still block browser-lane scenarios 1-3;
  mitigation: human-paced frequency, domain allowlists, scenario
  diversification, and honest labeling of blocked sites.
- Local planners are slower than cloud models per step (9-11 s vs
  seconds); the fast lane and deterministic bootstraps absorb most steps.
- Prompt injection via page content remains the top security risk; guards,
  verifier, and gates reduce but do not eliminate it. Every scenario ships with
  its forbidden-action list.

## Next actions

1. Scenario 4 smoke run: sign in to Spotify once at the `--human-login` gate,
   then let the planner create a playlist from an NL prompt.
2. Scenario 2 smoke run: Google Flights search/compare, stopping before
   payment (`--start-url https://www.google.com/travel/flights`, research
   guards, `done` on the comparison summary).
3. Scenario 6 watcher on top of the always-on service (laya as cheap filter).
4. Native macOS Accessibility targets to open non-browser desktop flows.
5. Operator-authorized `CONFIRM_ORDER` run remains optional (per operator:
   add-to-cart is the accepted shopping terminal for the MVP).

## Native-macOS computer-use product settings intel (2026-09-26, operator-supplied)

OCR of a leading commercial computer-use product's desktop settings panel confirms the native-macOS route:

- **Accessibility is the first required permission** — computer use is described as the ability "to click, type and use apps on your computer" (native AX driving, no DOM injection). Followed by File system access, Screen Recording, Dictation.
- **Per-capability consent**: Computer control and Browser automation are
  separate dropdowns (default "Ask every time") — same human-gate philosophy
  as our RESUME/CONFIRM_ORDER file sentinels, generalized per capability.
- **Per-app allowlist** ("Add app") and **Keep screen awake while working**
  (always-on watcher support).

## Local planner qualification (2026-09-26)

`--planner-text-only` now runs the planner without screenshots: text-only
planners judge from page text, targets, and structured deltas while the local
The vision verifier keeps visual grounding. First A/B on the YouTube Music task
(M3 Ultra 256GB, rapid-mlx serve on-device, strict json_schema guided decode):

| planner | steps | success outcomes | median plan latency | notes |
|---|---|---|---|---|
| GLM-5.3-Flash (cloud EXL3, vision) | 13 | 6 | 10.2 s | full task: played 1 + queued 2 via action menus |
| Qwen3.8-27B-4bit-MTP (local, text-only) | 3 | 2 | 17.6 s | played 1, then done'd early claiming queueing "requires login" without attempting it |

Verdict: the local 27B is **protocol-competent** (valid strict-schema plans,
real actions, honest final_summary) but **terminates prematurely** versus
GLM's persistence. Local latency is 1.7x cloud despite MTP speculative decode
(guided-schema generation dominates). Next levers: a done-gate that requires
task milestone evidence before accepting `done` from weak planners,
longer-horizon prompting, and a vision-capable local candidate
(gemma-4-26b-a4b-it-4bit is already in the HF cache) to test whether
screenshots close the persistence gap.

## AX probe result (2026-09-26)

`tools/gui_verifier_cua_poc/ax_driver.py` proved the DOM-free route alongside
the Playwright driver: it enumerated Chrome's AX tree including web content
(92 targets after AXManualAccessibility enablement + lazy-load retry) and
performed a semantic `AXPress` (kAXErrorSuccess) with CGEvent click fallback —
no injected ids, no planner-generated coordinates. Known limits: Finder dump
was empty without an open window; fill/settable-value handling and incremental
AXObserver eventing are not wired yet. Integration plan: expose `--driver ax`
in run_poc so `_collect_targets` and the executor swap to AX targets while
planner/verifier protocols stay unchanged.

## Native AX runner dogfood + architecture validation (2026-09-26 evening)

**ax_runner.py is live**: the full CUA loop (GLM planner → semantic actions →
screenshot-capable capture → laya assessment → domain guards) now runs on the
macOS Accessibility tree with no Playwright and no DOM injection. Wikipedia
smoke passed in 2 steps: fill+submit typed into the search box via CGEvents,
GLM read the Apple Silicon article straight from AX text and terminated with a
correct summary (run /tmp/ax-runs/20260926-183150). Screenshot capture uses
`screencapture -l<window>` (Screen Recording TCC verified on this host);
`--planner-text-only` is the fallback if capture degrades.

**Architecture validation from an established commercial native-AX
computer-use product** ( surveyed for pattern confirmation only; its runtime
is private and peer-locked, so we never drive it — we adopt validated
patterns and implement them independently): browser automation rides a
snapshot daemon (a11y-tree snapshots with stable element refs); desktop
computer-use rides a signed local helper speaking JSON-RPC over a Unix
socket with token auth, exposing app-state/click/scroll/type/press/hotkey/
set-value/secondary-action, a ~120 s snapshot cache keyed by app/window,
stale-element rejection, set-value with read-back verification, typed error
codes with per-code recovery hints for the agent, and per-capability
permission gating. Linux/Windows ride OS script providers (AT-SPI / UIA).
Patterns we adopt:

1. AX element snapshots carry an index + role + label + value; actions accept
   elementIndex (semantic) or x/y (fallback) — matches our target_id scheme.
2. Fill should try settable-value with read-back verification before
   synthetic typing (our fill currently types via CGEvents only).
3. Typed errors with recovery hints belong in the planner context.
4. Per-capability consent prompts are the right generalization of our
   RESUME/CONFIRM_ORDER sentinels.

Adopted now: error taxonomy + recovery hints land with the ax driver; setValue
read-back and AXObserver eventing are queued as follow-ups.

## Tool/agent decoupling shipped: `rapid-mlx computer` CLI (2026-09-26 night)

Per operator decision, the computer-use layer is now a standalone model-free
tool suite: `python -m rapid_mlx.computer_use ...`
(capabilities, permissions, list-apps, list-windows, get-app-state, click,
set-value, type-text, press-key, hotkey, scroll, perform-secondary-action).
Single JSON output shape with typed error codes + recovery hints; snapshot
cache (TTL 120 s, front-window-only) with stale-index rejection; activation
before synthetic input; fill = AX write → read-back verify → synthetic
keycode typing → verify by value.

Hand-driven end-to-end smoke on Wikipedia (agent protocol: observe → act →
re-observe → act): set-value verified `"Alan Turing"` exactly (element 113),
tree shifted (Search button 122→159), fresh observe + AXPress clicked the new
index, results page confirmed. Four real bugs found and fixed along the way:
1) CGEvent modifiers must be set per-event or shift sticks (ALLCAPS);
2) Chromium drops CGEventKeyboardSetUnicodeString on web content — real HID
   keycodes are required (unicode fallback kept for non-keymap chars);
3) multi-window apps need front-window-only element collection (max_windows=1)
   or coordinates hit the wrong window;
4) cached snapshots are unsafe for coordinate actions — observe fresh before
   acting (a runtimeId-based design has the same re-resolution constraint).

What we deliberately did NOT adopt yet: per-action confirmation policy UI
(our RESUME/CONFIRM_ORDER sentinels stay loop-level for now), AXObserver
evented updates (we poll), runtimeId element identity (we re-index per
snapshot), right/middle click and drag (queued).

## Local planner A/B on the decoupled AX tool layer (2026-09-26 night)

Operator directive: skip cloud-vs-local GLM (identical behavior expected);
qualify local models on the same Wikipedia task through `ax_runner` +
text-only planning + strict-schema guided decode, then escalate size only if
the larger fails.

| planner | steps | bad plans | median plan latency | outcome |
|---|---|---|---|---|
| GLM-5.3-Flash (tunnel) | 2 | 0 | 7.1 s | ✅ article reached, summary cites page sections |
| Qwen3.8-27B-4bit-MTP (local) | 2 | 0 | 21.7 s | ✅ same; summary correctly lists A/M/R/S/T-series sections |
| Qwen3.5-9B-8bit (local) | 3 | 0 | 7.8 s | ✅ same; properly re-observed after submit and clicked the article link from results |

Verdict: **both local sizes qualify** on short-horizon tasks; the 9B is 2.8x
faster than the 27B and showed the more disciplined observe-act cadence.
Caveat: this task is 2-3 steps. The earlier music task (13 steps for GLM,
where the 27B terminated prematurely) remains the harder qualification; a
done-gate that requires milestone evidence before accepting `done` is the
next lever before drawing conclusions for long-horizon work.

## 9B across all five flows (2026-09-26 night) — split verdict

Operator asked whether the 9B can do all five flows. Measured:

| flow | result | notes |
|---|---|---|
| 3 file organizer | ✅ (11/12) | classification perfect (incl. csv→Documents); one file omitted by the plan, validator only rejects hallucinations, not omissions |
| 4 digest | ✅ (3/3) | urgency ranking correct (P1 > OKR deadline > weekly), actions extracted |
| 5 music (AX, 20 steps) | ❌ | execution got to search results (site combobox filled, URL /search?q=…), but planner fixated: re-issued "click search button" across tree rebuilds, never played/queued |
| 2 flights, 1 shopping | not attempted | strictly harder than music; ceiling already established |

Diagnosis: 9B's *execution* through the tool layer is fine; its *observation
update* fails under long horizons — outcomes read "success" (tree changed)
while the plan fixates on stale intent. GLM re-plans from changed context.

Levers before retrying local-small on long flows: (a) done/no-progress gate —
N consecutive identical step_instructions force a re-observe with a "you are
here: <domain/page signature>" hint; (b) feed the executor's structured delta
into the next prompt more loudly; (c) vision for small models (Qwen3-VL-8B in
cache) so page identity is unambiguous; (d) grant Automation TCC so the URL
guard has real teeth (AppleScript blocked: -1743).

## Productization: rapid_mlx.cua replaces the POC loop (2026-09-27)

Decision: ship the native-AX pipeline as `rapid_mlx/cua/` and wire it into
the main CLI as `rapid-mlx cua`. Architecture separates the concerns:
- execution layer = existing `rapid_mlx.computer_use` (unchanged, reused)
- fast thinking = always on-device (laya /v1/rank outcome routing + fixation
  gate with recovery hints; two interventions then stall out honestly)
- slow thinking = user-configured preset or loopback URL
  (`~/.rapid-mlx/cua-config.json`; cloud-glm / local-27b / local-9b)
- consent = hard credential/commerce stops + optional sign-in file sentinel
- traces in ~/.rapid-mlx/cua-runs/<ts>/trace.json

The planner protocol keeps the proven strict-schema + one-repair-retry shape,
retargeted from DOM target_ids to AX element indexes. `planner` is injectable
for SDK/testing. 21 new tests (config/validation/fixation/gates/loop paths);
full loop verified end-to-end with local-9b on the Wikipedia task (3 steps,
grounded final summary quoting the article's first paragraph).

The POC tools stay under tools/gui_verifier_cua_poc/ as reference; new work
targets rapid_mlx.cua.
