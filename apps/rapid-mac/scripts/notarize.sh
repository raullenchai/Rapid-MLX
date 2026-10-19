#!/usr/bin/env bash
# notarize.sh — submit an artifact to Apple's notary service, wait for
# the verdict, then staple the ticket onto a target so the result is
# verifiable offline.
#
#   notarize.sh <submit-file> <staple-target>
#
#     submit-file    file uploaded to notarytool. notarytool only accepts
#                    .zip / .dmg / .pkg — to notarise a bare .app, zip it
#                    first (`ditto -c -k --keepParent Rapid.app Rapid.zip`)
#                    and submit the zip.
#     staple-target  what the ticket is stapled to. `stapler` can staple a
#                    .app or a .dmg (NOT a .zip). So:
#                      app:  submit Rapid.zip, staple Rapid.app
#                      dmg:  submit Rapid.dmg, staple Rapid.dmg
#
# Why staple the .app AND the .dmg in the release flow: stapling the .app
# (before it goes into the dmg) is what lets the installed app launch on a
# machine that is offline on first run — Gatekeeper reads the embedded
# ticket instead of phoning home. Stapling the .dmg covers the download
# itself. See scripts/dmg.sh + the release workflow for the ordering.
#
# Credentials come from the environment (App Store Connect API key — the
# CI-friendly path, no personal Apple ID / app-specific password):
#   AC_API_KEY_ID      App Store Connect key id
#   AC_API_ISSUER_ID   issuer id (top of the App Store Connect Keys page)
#   AC_API_KEY_PATH    path to the AuthKey_XXX.p8 file
#
# If any are unset we SKIP (exit 0) with a notice — so a local
# `build.sh && dmg.sh` without Apple creds still succeeds; only a real
# release run (which exports these) actually notarises.
set -euo pipefail

SUBMIT_FILE="${1:?usage: notarize.sh <submit-file> <staple-target>}"
STAPLE_TARGET="${2:?usage: notarize.sh <submit-file> <staple-target>}"

if [[ -z "${AC_API_KEY_ID:-}" || -z "${AC_API_ISSUER_ID:-}" || -z "${AC_API_KEY_PATH:-}" ]]; then
    echo "==> notarize: AC_API_* not set — skipping notarisation for $SUBMIT_FILE" >&2
    echo "    (set AC_API_KEY_ID / AC_API_ISSUER_ID / AC_API_KEY_PATH to enable)" >&2
    exit 0
fi

if [[ ! -f "$SUBMIT_FILE" ]]; then
    echo "notarize: submit-file not found: $SUBMIT_FILE" >&2
    exit 1
fi
if [[ ! -f "$AC_API_KEY_PATH" ]]; then
    echo "notarize: AC_API_KEY_PATH not found: $AC_API_KEY_PATH" >&2
    exit 1
fi

echo "==> notarytool submit $SUBMIT_FILE (waiting for Apple verdict)"
# --wait blocks until the verdict; notarytool exits non-zero on
# Invalid/Rejected, so `set -e` aborts the release before we try to
# staple a ticket that was never issued.
#
# NOTARYTOOL_FORCE=1 appends ``--force`` to skip notarytool's local
# pre-submission file-format validation. The canonical full DMG
# leaves this unset (default: client-side validation on). The slim
# bootstrapper DMG sets it because notarytool's UDIF detector
# rejects the slim DMG ("must be a zip archive (.zip), flat
# installer package (.pkg), or UDIF disk image (.dmg)") even though
# ``hdiutil verify`` accepts the same bytes as a valid UDIF image
# (suspected interaction between envelope codesign and notarytool's
# end-of-file ``koly`` trailer detector — see rapid-desktop#427).
# --force only skips the LOCAL validator; Apple's server still runs
# the full notarisation pipeline and would reject a truly malformed
# artifact.
FORCE_FLAG=()
if [[ "${NOTARYTOOL_FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
    echo "    (NOTARYTOOL_FORCE=1 — skipping client-side pre-submission validation; server still validates)"
fi
# Note: macOS ships Bash 3.2.57; under ``set -u`` the bare expansion
# ``"${FORCE_FLAG[@]}"`` errors as "unbound variable" when the array is
# empty. The ``${name[@]+...}`` guard expands to nothing when the array
# is unset/empty and to the protected expansion otherwise — safe on
# Bash 3.2 and forward-compatible with Bash 4+/5+. Codex r1 BLOCKING.
# Capture submission output so we can extract the id and pull the
# per-issue log on Invalid. ``--output-format json`` makes the parse
# robust without depending on the human-readable layout (which Apple
# has shipped breaking changes to in the past).
#
# IMPORTANT (Apple quirk): under ``--output-format json``, notarytool
# returns exit 0 even when the final status is ``Invalid`` (the exit
# code reflects whether the SUBMISSION succeeded, not the notarization
# verdict — the JSON ``status`` field is the source of truth). Without
# this awareness the script would proceed to ``stapler staple`` on an
# un-notarised image and the slim DMG R2 publish gate would then catch
# the missing staple, but the operator would never see Apple's actual
# reason. We always parse the status and treat anything other than
# ``Accepted`` as failure.
NOTARY_OUT="$(mktemp -t notarize-submit-XXXXXX.json)"
trap 'rm -f "$NOTARY_OUT"' EXIT
submit_exit=0
xcrun notarytool submit "$SUBMIT_FILE" \
    --key "$AC_API_KEY_PATH" \
    --key-id "$AC_API_KEY_ID" \
    --issuer "$AC_API_ISSUER_ID" \
    --wait \
    --output-format json \
    ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    > "$NOTARY_OUT" 2>&1 || submit_exit=$?

cat "$NOTARY_OUT"

NOTARY_STATUS="$(/usr/bin/python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get("status",""))
except Exception:
    pass' "$NOTARY_OUT" 2>/dev/null || true)"
SUBMIT_ID="$(/usr/bin/python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get("id",""))
except Exception:
    pass' "$NOTARY_OUT" 2>/dev/null || true)"

if [[ "$NOTARY_STATUS" != "Accepted" || "$submit_exit" -ne 0 ]]; then
    echo "==> notarisation FAILED (status='${NOTARY_STATUS:-unknown}', submit_exit=${submit_exit})"
    if [[ -n "$SUBMIT_ID" ]]; then
        echo "==> notarytool log $SUBMIT_ID (Apple Notary detail)"
        xcrun notarytool log "$SUBMIT_ID" \
            --key "$AC_API_KEY_PATH" \
            --key-id "$AC_API_KEY_ID" \
            --issuer "$AC_API_ISSUER_ID" \
            || echo "(unable to fetch notarytool log — see submit output above)"
    else
        echo "(no submission id — notarytool exited before upload; see submit output above)"
    fi
    exit 1
fi

# notarytool reports Accepted before the ticket is always reachable from
# Apple's ticket-delivery service; stapling immediately afterwards can fail
# with exit 68 (NSURLError -1005 "network connection was lost"). Retry the
# staple with backoff. `stapler validate` below stays mandatory; it queries
# the same service and can hit the same transient error right after a
# successful staple, so it retries with the same backoff.
echo "==> stapler staple $STAPLE_TARGET"
staple_delays=(15 30 60 120)
staple_attempt=1
until xcrun stapler staple "$STAPLE_TARGET"; do
    if (( staple_attempt > ${#staple_delays[@]} )); then
        echo "==> stapler staple FAILED after ${staple_attempt} attempts"
        exit 1
    fi
    delay="${staple_delays[$((staple_attempt - 1))]}"
    echo "==> stapler staple attempt ${staple_attempt} failed; retrying in ${delay}s"
    sleep "$delay"
    staple_attempt=$((staple_attempt + 1))
done
validate_attempt=1
until xcrun stapler validate "$STAPLE_TARGET"; do
    if (( validate_attempt > ${#staple_delays[@]} )); then
        echo "==> stapler validate FAILED after ${validate_attempt} attempts"
        exit 1
    fi
    delay="${staple_delays[$((validate_attempt - 1))]}"
    echo "==> stapler validate attempt ${validate_attempt} failed; retrying in ${delay}s"
    sleep "$delay"
    validate_attempt=$((validate_attempt + 1))
done

echo "==> Gatekeeper assessment"
# stapler validate (above) is the authoritative staple check and
# notarytool --wait already gated the notarisation verdict. spctl is
# supplementary:
#   - .app: `spctl --assess --type exec` is reliable — keep it strict.
#   - .dmg: `spctl --assess` frequently reports "rejected: source=
#     Insufficient Context" even for a correctly notarised + stapled
#     image, so treat it as advisory (don't fail the release on it).
case "$STAPLE_TARGET" in
    *.dmg)
        spctl --assess --type open -vv "$STAPLE_TARGET" \
            || echo "note: spctl .dmg assessment is non-authoritative here; staple already validated"
        ;;
    *)
        spctl --assess --type exec -vv "$STAPLE_TARGET"
        ;;
esac

echo "notarize: done for $STAPLE_TARGET"
