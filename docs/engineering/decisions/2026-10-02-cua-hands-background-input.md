# CUA hands: background-first input delivery

Status: accepted (PR 1 of the hands roadmap below)
Owner: Atlas
Date: 2026-10-02

## Context

The goal for the computer-use agent is the Meta Muse experience: an agent that
operates apps while the user watches and can take over at any time. The
difference is that ours runs entirely on the user's Mac. That requires
"hands" that can drive a window *without* taking the user's cursor, keyboard
focus or front app away.

Before this change, every non-semantic action went through the global HID tap
(`CGEventPost(kCGHIDEventTap, …)`):

- pixel clicks, keystrokes, text and wheel scrolls;
- each one required the target app to be frontmost and its window to be
  topmost at the point;
- each one warped the user's cursor.

`AXPress` / AX value writes were the only actions that worked without
foreground.

We surveyed the mature "hands" implementations:

- trycua/cua `cua-driver` (MIT, the most complete);
- steipete/Peekaboo + AXorcist;
- Hammerspoon and yabai;
- Microsoft UFO2;
- browser-use / macOS-use;
- Hermes Agent, Agent-S, Open Interpreter, and the Anthropic reference
  executor.

The two strongest, cua-driver and Peekaboo, converge on the same design:

1. **Route ladder.** A semantic AX action comes first, then an event routed
   to the exact pid/window, then explicit foreground. Global HID is never a
   silent fallback.
2. **Honest results.** Results separate *dispatched* from *verified*.
   `confirmed` requires read-back evidence; pixel, key and wheel delivery is
   `unverifiable`.
3. **Exact targets.** Every action is pinned to an exact `(pid, CGWindowID)`
   and refused on drift or ambiguity.

## Decision

Add `rapid_mlx/computer_use/background_input.py`, a ctypes binding to the
private SkyLight SPI. It is ported from cua-driver's
`platform-macos/src/input`, with the focus-without-raise record from yabai.
No new dependency and no new Mach-O are introduced. The backend routes
through it whenever it is available.

| Gesture | Background route | Notes |
|---|---|---|
| Left click | focus-without-raise record → stamped `mouseMoved` primer → `(-1,-1)` decoy down/up → target down/up via `SLEventPostToPid` | Stamped fields: f40 target pid, f51/f91/f92 window, f58 click group |
| Right / middle click | primer → down/up with the matching button number, posted via SkyLight only (cua also posts `CGEventPostToPid`; measured on Chrome and TextEdit, each route alone works and both together deliver every event twice) | Right-down stamped as button 0 arrives as a left click; window-local point stamped |
| Double click | clickState 1 → 2 pairs | `AXOpen` first when the element advertises it |
| Wheel scroll | primer + ≤10-line notches, SkyLight only (both routes double the distance: 3 lines → 240px vs 120px in Chrome, 2× in TextEdit) | Window-local point stamped (a screen point does nothing once the window is off the origin — measured); nested scrollers work |
| Text | keycode-0 + `CGEventKeyboardSetUnicodeString` per scalar, flags forced to 0 | Bypasses layout and IME; CJK and emoji arrive literally |
| Keys / non-Cmd chords | keycode + exact flags to the pid, with `SLSEventAuthenticationMessage` | Envelope only on macOS 15+ |
| Cmd chords | **foreground (HID)** | See "Measured limits" |

Element clicks try the semantic action that matches the gesture first, but
only if the element advertises it:

- left → `AXPress`
- double → `AXOpen`
- right → `AXShowMenu`

Failure is not silent at any rung. If a background primitive cannot be
synthesized, the action raises `action_failed`; it never re-posts through HID.
Validation is per route:

- **Background:** keeps the snapshot-age, exact-window and in-bounds checks,
  and requires the exact window to be the app's focused AX window for
  keyboard input. It drops the frontmost-app and topmost-at-point checks,
  because routed events are not hit-tested against other apps' windows.
- **Foreground:** keeps every historical check and call shape.

Focus: after a background click, `_restore_user_focus` hands keyboard focus
back to the user's front window, including another window of the same app.
Capture, gesture and restoration form one transaction under the reentrant
`GESTURE_LOCK`, and restoration runs in a `finally` so a failed synthesis
never leaves the user's focus displaced. It uses the reverse focus record, or
re-activates the user's app if the target activated itself on click.

Results now carry:

- `route`: `accessibility` | `pid_events` | `global_hid`
- `effect`: `confirmed` | `suspected_noop` | `unverifiable`, derived from
  `verified`

Delivery mode: `RAPID_MLX_CUA_INPUT_DELIVERY=auto|background|foreground`.
`auto` is the default and means background when SkyLight resolves.
`foreground` restores the old behaviour exactly.

Finder keeps the foreground route. Its inline-rename safety checks are built
around activation, and its editor rebinds focus asynchronously.

## Measured limits (macOS 26.5.2 Tahoe, SIP on, user's Chrome kept frontmost)

Verified working in the background, with the front app unchanged and the
cursor unmoved:

- Calculator button clicks;
- TextEdit text with CJK and emoji, Delete, wheel scroll, and a right-click
  context menu;
- an isolated Chromium instance: web-input focus click, typing (incl. CJK), a
  left-click `onclick`, and a right-click `contextmenu`.

Known limits:

- **Menu key equivalents** (Cmd+A/S/N…) do **not** fire in a background app
  via any route we tried: with the auth envelope, without it,
  after focus-without-raise, a brief `SLPSSetFrontProcessWithOptions` assist,
  or AXPress on the menu item. AppKit reports those items `AXEnabled=false`
  while the app is inactive, which matches Hammerspoon's documented gotcha.
  Cmd chords therefore stay foreground. Semantic substitutes (e.g. select-all
  via `AXSelectedTextRange`, paste via pasteboard + `AXSelectedText`) are a
  later PR.
- **The auth-envelope probe in the upstream port misses on macOS 26.**
  `messageWithEventRecord:pid:version:` is a *class* method, so
  `class_respondsToSelector` must be asked about the metaclass. Probing the
  class object returns false on macOS 26 and silently drops the envelope.
- **No measurement shows the envelope is required in this setup.** In the
  Chromium test, typing also landed without it, because the preceding
  background click had already made the page key. It is kept for
  Electron/VS Code targets, where cua reports it is required.

Reproduce:

```bash
# unit + routing tests (any platform)
pytest tests/test_computer_use_background.py -q
# real Calculator, opt-in (needs Accessibility for the test host)
RAPID_MLX_LIVE_GUI=1 pytest tests/test_computer_use_background.py -q -k live
```

## Roadmap (serial PRs)

1. **This PR — background transport.** Routed click, right/middle/double
   click, scroll, text and keys; the semantic-first click ladder;
   `route`/`effect` in results; focus restoration.
2. **Observe without stealing focus.** The loop observes with
   `activate=False` whenever background delivery serves the target
   (`observation_activates`): the planner's key vocabulary has no Cmd chords,
   so nothing it can plan needs the foreground. Finder, foreground delivery
   and an unknown identity keep activating. Measured: TextEdit observed in
   0.56 s with the user's app still front, versus 1.06 s and a stolen focus
   when activating. Safety fixes:
   - `focus_only` (focusing before Tab/arrow/Escape) never AXPresses or
     clicks. It sets `AXFocused` and refuses when that fails; the planner
     then needs a consent-gated click. A pixel fallback was tried and
     dropped: AX roles, actions, ancestors and a hit-test still cannot prove
     a click is non-committing (a web input may submit on click while
     reading as a plain `AXTextField`).
     The old path pressed the button under a key plan, bypassing click
     consent;
   - sign-in detection reads `subrole` (password fields are `AXTextField` +
     `AXSecureTextField`);
   - the loop's window identity read `bundle_id`, which never exists;
   - `_app_element` raises `AppNotFoundError` (a `LookupError`) instead of
     `SystemExit`, which escaped every `except Exception` in the server.
   Moving backend calls off the event loop moves to PR 3. The loop's gate and
   stop-event plumbing is bound to the server loop, so it needs its own
   design.
3. **Robust observation and identity.**
   - Visit-budgeted walk;
   - `AXUIElementSetMessagingTimeout`;
   - batched attribute reads;
   - Chromium enablement cached by pid + start time;
   - re-resolvable locators (role path, `AXIdentifier`, `AXDOMIdentifier`)
     instead of DFS index + exact label/center;
   - an explicit truncation marker for the planner;
   - AX read-backs under `AXWebArea` treated as untrusted.
4. **Action vocabulary.**
   - `AXScrollToVisible` before acting;
   - menu-path invocation (`File > Export…`);
   - `AXShowMenu` + menu-item pick;
   - text via `AXSelectedText` with keyboard fallback;
   - pasteboard transaction for long/multiline text;
   - foreground drag;
   - `NSWorkspace` launch with `activates=false` plus a focus-steal guard;
   - the actions exposed in `PLAN_SCHEMA`.
5. **Verification and settling.**
   - `verify_state` predicates (tri-state, stable samples, never claim
     absence);
   - AXObserver/poll settle instead of fixed sleeps;
   - ScreenCaptureKit window capture with identity revalidation.
6. **Finder/TextEdit adapters.** Extract the ~900 lines of app-specific
   rename/save code behind adapter hooks.

Brain and eyes are secondary to the hands:

- **Brain:** GLM-5.3-Flash.
- **Eyes:** the AX tree first; Laya (a text/AX encoder, not vision) for
  re-ranking ≤10 AX candidates; a ~2B GUI grounder (e.g. GUI-Actor-2B class)
  only for content without an AX tree.
