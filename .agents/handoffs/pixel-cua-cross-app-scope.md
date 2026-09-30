# Pixel handoff: cross-app scope binding

- Receiving roles: Atlas for integration order; Harbor for the next signed candidate.
- Branch: `pixel/cua-cross-app-scope`
- Base: `vector/cua-finder-transaction-corrected` at `90b5163d4`.
- Scope: Desktop validation of the multi-target create response. Finder transaction
  behavior and server execution are unchanged.

## Verified cause

The Desktop sends `bundle_id` and `process_start_time` on each target so the
server can reject a replaced process. The create endpoint intentionally returns
the narrower committed run-target shape, without those request-only validation
fields. Synthesized Swift equality included the two optional fields, rejected
the valid response, and immediately cancelled the run. Signed run
`621d39eee622` exhibited this exact shape for Safari and TextEdit.

## Resolution

Compare the echoed fields that constitute the committed run binding: target ID,
PID selector, PID, window ID, domain, array order, and active target ID. Continue
to fail closed and cancel when any of those fields differ. A wire-level Swift
regression reproduces the signed Safari/TextEdit response.

## Reference check

Orca's established approval flow keeps consented scope separate from runtime
identity validation and presents failed verification as a stopped task. The
locally available Open WebUI, Jan, Cherry Studio, and LM Studio materials expose
no comparable native multi-app target echo contract. The change therefore keeps
Rapid's current fail-closed flow and fixes only the request/response schema
boundary.

## Next action

After the Finder transaction branch lands, Atlas should integrate this commit
and Harbor should rebuild/sign the candidate, then repeat Safari Example Domain
to TextEdit `Example Domain Notes.txt`. Acceptance is a create response that is
not cancelled and proceeds to at least the first plan event.
