# Atlas handoff: GUI verifier Computer Use POC

- Receiving role: Atlas
- Branch: `atlas/gui-verifier-cua-poc`
- Host: Studio
- Status: POC implemented and dogfooded; product integration remains open
- Landed: semantic action protocol committed as `fe6624989` and pushed to
  `origin/atlas/gui-verifier-cua-poc` (14 POC tests green, ruff clean).
- Scenario research: Muse top flows distilled into
  `docs/engineering/decisions/2026-09-26-local-muse-mvp-scenarios.md`
  (shopping, travel, money, music, local files, always-on) with the MVP stack
  mapping and next actions. Architecture note: laya (not GUI-Actor) is the
  fast-thinking lane; GUI-Actor-Verifier-2B is grounding-only.
- New flows dogfooded 2026-09-26: `tools/local_muse/file_organizer.py`
  (NL rule → validated plan with repair retry → APPROVE gate → reversible
  moves) and `tools/local_muse/digest.py` (5 docs → urgency-sorted digest).
  Generic `--human-login` RESUME gate added to the browser runner for Spotify
  and other sign-in-first sites. Per operator: shopping MVP terminal is
  verified add-to-cart; CONFIRM_ORDER stays optional.

## Verified facts

- Qwen3.5-9B 4bit can plan from a screenshot plus compact DOM context when
  Rapid strict JSON schema is enabled.
- GUI-Actor-Verifier-2B loads unmodified through mlx-vlm 0.7.2 on Apple Silicon.
- The verifier correctly separated oracle search/reviews targets from obvious
  wrong points, but misranked an Amazon carousel arrow above the correct product.
- The main end-to-end failures were Qwen planning, coordinate candidate
  generation, goal retention, and repeated recovery actions.
- GLM-5.3-Flash with low reasoning completed the same task both with the search
  bootstrap (4 model steps) and fully unguided (11 model steps). Unguided mean
  planning latency was 7.65 seconds versus Qwen's 38.68 seconds in the
  controlled run.
- GLM's four redundant search-field clicks exposed an action/state-contract
  failure: a screenshot cannot reliably report keyboard focus, `type` carries
  no semantic target, and reflection therefore reinforced an unnecessary retry.
- The semantic protocol now exposes current target IDs, atomic fill+submit,
  structured postconditions, target-derived click candidates, and adaptive
  verifier scoring. Two clean protocol runs completed in 75-82 seconds without
  a GLM reflection or repeated action.
- Purchase mode (`--purchase`) with human gates (RESUME after sign-in,
  CONFIRM_ORDER before placing the order) reached a verified add-to-cart
  end-to-end without human input; the operator chose to stop before checkout,
  so no order was placed. Verifier correctly rejected ad-banner misclicks on
  the cart page; DOM ids in target context plus a cart checkout bootstrap
  removed that retry loop. One residual finding: a background run died
  silently between gates once (no traceback), restarts were clean.
- The Laya + GLM shopping fast path completed in 37.37 seconds with three visible
  actions and two GLM calls. Laya pre-ranked four organic cards in 0.253 seconds;
  its low top probability (0.3024) means the ranking remains advisory.
- No cart, account, checkout, payment, or purchase action was executed.

## Risks

- GUI-Actor-Verifier is a local coordinate verifier, not a goal-level policy or
  safety model.
- Amazon's dynamic and sponsored layout makes results non-repeatable.
- Raw screenshots and temporary Chrome profiles stay under `/private/tmp` and
  must not be committed.
- Product claims require a repeated benchmark with stable fixtures and an
  independent result oracle.

## Next concrete action

Run the first human-authorized CONFIRM_ORDER pass to certify the complete
purchase path (operator approval required for the real order), then port
semantic targets from browser DOM nodes to native macOS Accessibility, add
typed drag/select operations, and run a held-out multi-application task suite.
Qualify a local Qwen 27B planner against the same traces before replacing
GLM-5.3.

## 2026-09-26 evening — four-flow dogfood status

- Machine rebooted mid-session; /tmp worktree wiped. Rebuilt at durable path
  `/Users/raullenstudio/work/rapid-mlx-gui-verifier-poc` (venv: python 3.12 via
  uv; deps pinned mlx 0.32.2 / mlx-vlm 0.7.2 / laya-mlx 0.2). Run artifacts from
  earlier today are gone; distilled results live in the decisions doc.
- **press action added** to the semantic protocol (click/fill/submit/press/
  scroll/wait/done; keys: Enter, Escape, Tab, ArrowDown/Up, Space; success =
  dom/url changed; key allowlisted in _validate_plan). Unlocked Google Flights
  Material UI widgets that ignore clicks.
- **--human-login now pauses up front** before any planning (in-page login
  modals like Spotify's never change the URL, so the URL-pattern gate alone
  misses them).
- **done path now copies plan.final_summary into trace** (was step-record only).
- Scenario results: S2 travel SEARCH+COMPARE dogfooded end-to-end on Google
  Flights (SFO→HND 2026-10-12 one-way; GLM compared Cathay $658 1-stop vs
  United $961 nonstop vs JAL $961 nonstop, balanced pick United nonstop; no
  booking). S3/S4 file organizer + digest previously green. S4 Spotify playlist
  run waiting on operator sign-in at the upfront gate (1h window from 17:23).
- Failed approach worth remembering: starting a run from a truncated results
  URL redirects to the flights homepage and burns the budget on re-search
  (20260926-170756); always copy full final_url from trace.json.

- Operator declined Spotify account login; music dogfood moved to YouTube Music
  (logged out) and PASSED: search → play first instrumental → 2 tracks queued
  via action menus (press Enter fallback needed once). Trace 20260926-174123.
  Same run surfaced a real bug fixed in code: custom elements (YouTube paper
  slider) expose numeric `.value`, so text collection now String()-coerces
  before .trim(). All four Muse scenarios (shop / travel / files / digest /
  music) have now each been dogfooded end-to-end at least once.

## 2026-09-26 night — computer-use tool layer decoupled (Orca pattern)

- Shipped `rapid_mlx/computer_use/` (CLI: capabilities/permissions/list-apps/
  list-windows/get-app-state/click/set-value/type-text/press-key/hotkey/scroll/
  perform-secondary-action). Model-free; any agent (cloud GLM, local Qwen 27B,
  Claude Code, Codex) can drive the Mac through it. 46 tests green.
- Hand-driven Wikipedia smoke closed the loop: set-value verified exactly,
  stale-index recovery worked (re-observe → new index → AXPress → results).
- Four fixes worth remembering: per-event CGEvent flags (sticky shift),
  keycode typing required for Chromium web fields (unicode-string events
  filtered), front-window-only collection for multi-window apps, observe-fresh
  before coordinate actions.
- Queued: right/middle+drag clicks, AXObserver eventing, per-action consent
  policy (Orca ACTION_POLICY analog), wire ax_runner onto this backend.
