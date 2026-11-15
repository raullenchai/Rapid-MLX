# Screenshots and browser verification checklist

Use synthetic data, hide API keys, and capture the published extension version.
Provide store screenshots at 1280×800 or 640×400.

- [x] Capture method: `test/e2e/store-shots.mjs` runs five real-engine tasks on local
  fixtures and stitches concurrent 880×800 page / 400×800 panel captures into
  1280×800 PNGs. See the README command; supports `LOCALE=en|zh_CN` and
  `STORE_SHOTS_OUT`. Pending Send is captured before denial; financial handover
  stays pending. Endpoint/key controls are hidden. Generated images still need
  human review; this tick records the method, not completed browser verification.

- [ ] English and Simplified Chinese first-run three-step onboarding and example chips.
- [ ] Local engine connection on ports 8000 and 7659; model list and tool probe.
- [ ] Custom HTTPS endpoint; exact-host Save grant and denial handling.
- [ ] Private LAN / 100.64.0.0/10 HTTP warning; public HTTP rejection.
- [ ] Missing permission, rejected key, unreachable/CORS, empty models, no tools.
- [ ] Shared shopping page: main price versus other offers, final cart subtotal.
- [ ] Approval with keyboard-accessible Allow once / Deny and screen reader announcement.
- [ ] Working tool steps, overlay Stop, paused Continue, cancellation and reconnecting.
- [ ] Empty and loading states; restored panel after closing and reopening.
- [ ] Light/dark contrast, focus rings, keyboard-only navigation at narrow widths.
- [ ] Archive installed cleanly; no tests, keys, traces, or local artifacts included.
