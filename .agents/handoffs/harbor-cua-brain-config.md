# Harbor handoff: CUA brain endpoint consent

- Owner: Harbor (Robert), coordinating Pixel UI and Vector server contracts
- Branch/worktree: `harbor/cua-brain-config`, `/private/tmp/harbor-desk-cua-brain`
- Base: `origin/main` at `6be91bab`
- FYI recipients: Atlas, Pixel, Vector, Echo, ds0731. Orca role mailbox was
  unavailable (`terminal_not_found`), so the required start/completion FYI is
  recorded here for later delivery.

## PR intention

Keep computer actuation on the user's Mac while allowing a user-selected
OpenAI-compatible planner on loopback or at a consented HTTPS endpoint. Make
the goal, Accessibility snapshot, and optional screenshot disclosure visible
and store remote-data consent independently from an optional API key.

Scope is limited to CUA planner config/validation/routes, the Rapid Mac CUA
settings surface, and focused tests. Non-goals: computer-use backend changes,
remote VM support, browser automation changes, plaintext non-loopback endpoints,
and LAN HTTP policy. Verification covers config and route behavior, URL
classification, compatibility, consent binding, Swift decoding/view-model
copy, and build integration.

## Reference check (private)

- Orca/native Mac settings patterns: adopted progressive disclosure at the
  brain picker and a destination-bound consent toggle in the add sheet.
- Jan custom endpoints: reviewed explicit same-machine/LAN endpoint setup and
  manual capability selection; adapted explicit image/text choice because an
  OpenAI-compatible URL does not prove vision support.
- LM Studio local server: reviewed localhost/LAN OpenAI-compatible serving;
  retained loopback as the only automatically on-Mac classification.
- Open WebUI and Cherry Studio: existing provider configuration patterns were
  checked at a high level; no code or branded assets were copied.

## Verified behavior and compatibility

- Loopback HTTP(S), including `127.0.0.1`, `localhost`, `localhost.`, and IPv6
  loopback, remains keyless and does not require remote consent.
- All non-loopback endpoints, including LAN addresses, require explicit consent
  and HTTPS. LAN HTTP remains a product-policy follow-up.
- Keyless HTTPS endpoints run once consented; the API key no longer acts as a
  permission bit.
- Changing the draft endpoint clears consent; remote-enabled or credentialed
  saved presets cannot be redirected with a URL override.
- Legacy keyed remote presets preserve their former effective permission.
  Legacy keyless remote presets stay denied until deleted/re-saved with explicit
  consent. This is intentional because they never ran under the previous code.
- Planner list responses expose only `has_api_key`, never key material.

## Verification evidence

- `python3.12 -m pytest tests/test_cua.py -q`: 75 passed.
- `python3.12 -m pytest tests/test_cua_server.py -q -k 'not real_server'`:
  19 passed, 2 deselected. The two real-server import tests abort in the local
  MLX native import path; all route/config tests completed.
- `swift test --disable-sandbox --filter 'CUA(AddBrain|PlannerDecode|ViewModel)'`:
  16 passed; full Rapid target compiled.
- Ruff on changed Python files: passed.
- `git diff --check`: passed.

## Remaining risk / next action

The config file's pre-existing API-key storage remains mode 0600 and is outside
this scoped consent fix; no secret is returned by the planner API. A future
credential migration can move keys to Keychain. Atlas should decide whether
encrypted LAN HTTP is ever supported through a separate trust/pinning design;
this PR fails closed for it.
