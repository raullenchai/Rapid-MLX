# Local Meta Muse MVP: scenarios and technology stack

- **Status:** Proposed direction; evidence-based POC in flight
- **Date:** 2026-09-26
- **Owner:** Atlas
- **Consumers:** gui_verifier_cua_poc, Rapid Desktop (Personal Intelligence), always-on service, model catalog
- **Evidence base:** public Muse reporting (CNBC, LA Times, The Verge, Tom's Hardware, Ars Technica, Music Ally, Business Insider) plus three instrumented dogfood runs in
  `tools/gui_verifier_cua_poc/` (branch `atlas/gui-verifier-cua-poc`)

## Context

Meta Muse launched in September 2026 as a consumer personal AI agent (Muse Spark
model family, led by Nat Friedman). It reached 730k downloads in five days and
2.5M in thirteen (CNBC), was rated the top AI agent by J.P. Morgan, and forced
OpenAI to rush an always-on competitor. Public reporting establishes its shape:

- **Execution substrate:** per-user cloud Linux VMs (AMD EPYC, 2 cores, 8 GB)
  with terminal and browser access (Tom's Hardware). Muse is cloud-hosted and
  Meta collects a transaction fee (Zuckerberg, Yahoo Finance).
- **Headline scenarios:** shopping and travel booking ("can now book travel and
  shop for you", LA Times); money management (Yahoo Finance); an official
  Spotify connector (Music Ally, Spotify Newsroom); first-class filesystem
  access (The Verge); always-on proactive behavior.
- **Friction:** Amazon blocked Muse from shopping (GeekWire, Engadget) using
  standards it ignores for its own agent (TechTimes); a serious 0-day followed
  within weeks (Ars Technica) because an "extraordinarily privileged" agent is
  a high-value attack surface.

Goal: distill Muse's proven user flows into 5-6 MVP scenarios for a **local,
Apple-Silicon-native equivalent on Rapid-MLX**, and name the technology stack
they force us to build. Walking every flow fully is not required; stopping at a
human gate (for example Amazon's final order button) is an accepted, expected
outcome whenever the operator declines permission.

## Top scenarios for the MVP

| # | Scenario | Muse evidence | Our MVP status | Acceptance for "done" |
|---|---|---|---|---|
| 1 | **Agentic shopping**: NL goal → search → compare (rating × review count × sponsorship) → product page → cart → checkout → order | Muse's flagship; blocked by Amazon | Proven end-to-end to verified add-to-cart (cart-count postcondition 0→1, 1→2); place-order path implemented behind `CONFIRM_ORDER` gate | One operator-authorized run places a real, cancellable order; trace captures the whole chain |
| 2 | **Travel booking**: NL trip → search flights/hotels across sites → compare → fill passenger/guest forms → stop at payment | Announced Muse capability; high value | **Dogfooded on Google Flights**: the `press` action (added after clicks proved inert on Material UI comboboxes) unlocked trip-type, suggestion, and calendar widgets; search SFO→HND 2026-10-12 completed and GLM produced a three-option comparison (Cathay $658 1-stop vs United $961 nonstop vs JAL $961 nonstop, balanced pick United). Booking/payment intentionally not exercised | Search→select→form-fill completes; run stops at payment with a typed order summary awaiting approval — **search/compare acceptance met** |
| 3 | **Money assistant**: read-only account/spend digest; bill pay strictly behind a human gate | Muse money flows; Wells Fargo partner coverage | Not started; browser lane + gate machinery already exist | Read-only digest of one bank dashboard; a payment flow demonstrably halts at the gate unexecuted |
| 4 | **Music/media**: create/curate playlists and queues from NL; discovery | Spotify is Muse's first official connector; operator declined account login, so dogfood ran on YouTube Music | **Dogfooded logged-out on YouTube Music**: search → play first instrumental track → add two more to the queue via each track's action menu (one menu needed the `press` Enter fallback). Trace shows player active (3:53/1:19:58) and `Song added to queue` toasts; no sign-in encountered | NL-driven search + playback + queue curation on the user's own machine, trace-verified; account-scoped actions (saved playlists) remain gated on `--human-login` |
| 5 | **Local files & desktop**: organize folders, find documents, cross-app desktop flows | Muse expanded filesystem access; highest-privilege surface | **Dogfooded end-to-end** (`tools/local_muse/file_organizer.py`): NL rule → planner proposal → validation caught a hallucinated file → repair retry → APPROVE gate → 12 files moved with undo log | One rule-based Downloads organization with dry-run summary and per-batch confirm — **met** |
| 6 | **Always-on proactive agent**: price/deal watching, inbox digest, reminders | OpenAI's "o" copies this; Muse's retention hook | **Digest flow dogfooded** (`tools/local_muse/digest.py`): 5 repo docs → urgency-sorted digest with action items (long-doc token fix applied); watcher scheduling pending | One watcher produces a daily human digest for a product price threshold |

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
   differentiator: Muse pays cloud tokens for every decision; we route most
   decisions to a 2B-class local model.
3. **Grounding verifier** — GUI-Actor-Verifier-2B (mlx-vlm) confirms that a
   click's landing point matches intent. Proven value: rejected three
   ad-banner misclicks the planner insisted on during dogfood.
4. **Semantic action protocol** — typed `click/fill/submit(target_id)` actions,
   atomic fill+submit, typed postconditions (URL, DOM hash, focus, input value,
   cart count), deterministic bootstraps for known choreography, and
   `trace.json` audit of every plan, score, and fallback.
5. **Trust and safety layer** — page text is untrusted data; hard regex guards
   split research vs purchase modes (credentials/card data always blocked);
   human gates as filesystem sentinels (`RESUME`, `CONFIRM_ORDER`) with timeout
   fallback; least-privilege is the answer to Muse's 0-day lesson.
6. **Execution lanes** — browser lane today (Playwright-controlled Chrome with a
   persistent user profile, human owns all sign-ins); native macOS Accessibility
   lane next (handoff item); local terminal lane with allowlists instead of
   Muse's cloud VM.
7. **Host integration** — Rapid Desktop surface plus the always-on service
   (ADR 2026-09-05) as the scheduler for scenario 6.

## Why this beats Muse for our users

- **Local-first privacy:** screenshots and page context never leave the Mac;
  Muse processes them on cloud VMs.
- **No transaction tax:** Meta skims a fee from purchases; we have no such
  incentive and our money gate is opt-in by design.
- **Platform-block resilience:** Amazon locked Muse out while running its own
  shopping agent. A human-paced local agent acting through the user's own
  session is a different trust model, and scenarios 4-6 do not depend on
  hostile platforms at all.

## Risks

- Platform ToS/anti-bot measures can still block browser-lane scenarios 1-3;
  mitigation: human-paced frequency, domain allowlists, scenario
  diversification, and honest labeling of blocked sites.
- Local planners are slower than Muse's cloud models per step (9-11 s vs
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
