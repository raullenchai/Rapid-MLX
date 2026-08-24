#!/usr/bin/env bash
# build-sidecar-tarball.sh — carve the bundled rapid-mlx sidecar out
# of a built (and ideally already-notarised) Rapid-MLX Desktop.app and
# repackage it as a standalone tarball for the bootstrapper architecture
# (see .claude/loop/bootstrapper-plan.md P1).
#
# The bootstrapper DMG that ships post-P3 will be ~5-8 MB and will pull
# this tarball + the Quickstart model tarball from dl.rapidmlx.com on
# first launch. P1 introduces the tarball as an additional CI artifact
# without changing the existing self-contained DMG flow, so existing
# users are unaffected.
#
# Notarisation note: the Mach-O binaries inside Contents/Resources/rapid-mlx
# are already codesigned with the Developer ID identity from build.sh
# and (in the CI path) participate in the .app's notarytool submission.
# Codesign signatures live in a __LC_CODE_SIGNATURE load command inside
# each Mach-O — not in xattrs — so they survive tar+untar round-trips.
# An extracted binary on a target Mac will validate via Gatekeeper's
# online notarisation lookup (TeamID + signature against Apple's notary
# service) provided the device is online (which it must be anyway to
# have downloaded the tarball). We do NOT staple the tarball: stapler
# binds only to bundle-shaped artefacts (.app/.dmg/.pkg). If offline
# friction shows up in the wild, the follow-up is to wrap the tarball
# in a notarised .zip — out of scope for P1.
#
# Output:
#   build/rapid-mlx-sidecar-X.Y.Z.tar.gz
#   build/rapid-mlx-sidecar-X.Y.Z.manifest.json
#
# Where X.Y.Z is CFBundleShortVersionString from the .app's Info.plist
# (the desktop release version), NOT the bundled rapid-mlx VERSION
# (which is a submodule SHA). Users and the bootstrapper key off the
# desktop version.
#
# Determinism:
#   The archive is produced via Python's tarfile module so we can pin
#   every mtime/uid/gid/uname/gname and force USTAR format. The gzip
#   wrapper is written via Python's gzip module with mtime=0 so the
#   gzip header is also content-only. Repeated runs over the same .app
#   tree produce byte-identical archives (same SHA-256) — useful for
#   R2 dedup and for PR reviewers to spot real content changes.
#
#   The pinned mtime defaults to SOURCE_DATE_EPOCH if set, falling back
#   to the latest git commit timestamp, falling back to a fixed
#   sentinel. Same source → same epoch → same SHA across CI rebuilds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
APP="${1:-$BUILD/Rapid-MLX Desktop.app}"

if [[ ! -d "$APP" ]]; then
  echo "::error::App bundle not found: $APP" >&2
  echo "Usage: $0 [/path/to/Rapid-MLX Desktop.app]" >&2
  exit 1
fi

SIDECAR_DIR="$APP/Contents/Resources/rapid-mlx"
INFO_PLIST="$APP/Contents/Info.plist"

if [[ ! -d "$SIDECAR_DIR" ]]; then
  echo "::error::Sidecar directory not found inside app: $SIDECAR_DIR" >&2
  echo "Did you run scripts/build.sh + scripts/build-sidecar.sh first?" >&2
  exit 1
fi

if [[ ! -f "$INFO_PLIST" ]]; then
  echo "::error::Info.plist not found: $INFO_PLIST" >&2
  exit 1
fi

# /usr/libexec/PlistBuddy non-zero exits abort under `set -e`. Catch
# explicitly so the error rendered to CI logs is a clean ::error::
# notice rather than PlistBuddy's raw "Print: Entry, … Does Not Exist".
if ! APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$INFO_PLIST" 2>/dev/null)"; then
  echo "::error::CFBundleShortVersionString missing or unreadable in $INFO_PLIST" >&2
  exit 1
fi
if [[ -z "$APP_VERSION" ]]; then
  echo "::error::CFBundleShortVersionString is empty in $INFO_PLIST" >&2
  exit 1
fi
# Defensive: filename will be constructed from this and downstream
# consumers (R2 keys, latest.json fields) treat it as opaque. Constrain
# to a SemVer-shaped string so a malformed plist can't slip a path
# separator or whitespace through.
if [[ ! "$APP_VERSION" =~ ^[0-9]+(\.[0-9]+)+([-+][0-9A-Za-z.-]+)?$ ]]; then
  echo "::error::CFBundleShortVersionString '$APP_VERSION' is not a SemVer-shaped string" >&2
  exit 1
fi

SIDECAR_VERSION_FILE="$SIDECAR_DIR/VERSION"
SIDECAR_VERSION="(unknown)"
if [[ -f "$SIDECAR_VERSION_FILE" ]]; then
  SIDECAR_VERSION="$(tr -d '[:space:]' < "$SIDECAR_VERSION_FILE")"
fi
# Defense-in-depth (#411): the bootstrapper's manifest validator at
# Sources/Rapid/Bootstrapper/BootstrapCoordinator.swift's
# ``isValidVersionString`` enforces a strict dotted-digit or ``-rcN`` grammar on
# ``sidecar_version`` (the optional leading ``v`` is stripped). v0.8.6
# shipped ``sidecar_version: "26ac5b4"`` because the upstream VERSION
# file carried a short SHA — that bricked 100% of slim-DMG installs.
# Refuse to emit a manifest with a value the bootstrapper would
# reject. scripts/build.sh has the upstream tag-based derivation; this
# regex gate is the floor.
# Grammar MUST match Sources/Rapid/Bootstrapper/BootstrapCoordinator
# .swift's isValidVersionString: pure dotted-digit with an optional ``-rcN``;
# no arbitrary pre-release or build suffix. A looser ``[-+][0-9A-Za-z.-]+``
# tail would accept ``0.8.19-rc.1`` here but the bootstrapper would
# reject the manifest at runtime — re-creating exactly the #411
# bricking bug for a different upstream value.
if [[ ! "$SIDECAR_VERSION" =~ ^[0-9]+(\.[0-9]+)+(-rc[1-9][0-9]*)?$ ]]; then
  echo "::error::sidecar VERSION '$SIDECAR_VERSION' is not dotted-digit or an -rcN candidate (expected e.g. '0.8.18' or '0.13.0-rc1'). Bootstrapper validator (BootstrapCoordinator.isValidVersionString) would reject this manifest — refusing to emit. See $SIDECAR_VERSION_FILE; fix scripts/build.sh's submodule derivation upstream (#411)." >&2
  exit 1
fi

TARBALL="$BUILD/rapid-mlx-sidecar-${APP_VERSION}.tar.gz"
MANIFEST="$BUILD/rapid-mlx-sidecar-${APP_VERSION}.manifest.json"

mkdir -p "$BUILD"

# Stable epoch for deterministic mtimes inside the archive. CI passes
# SOURCE_DATE_EPOCH; locally we fall back to the latest commit time.
# The fixed sentinel ensures the script still runs in a shallow tree
# with no git history (e.g. a contributor running `bash scripts/…`
# from a release tarball checkout).
if [[ -z "${SOURCE_DATE_EPOCH:-}" ]]; then
  if SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null)" && [[ -n "$SOURCE_DATE_EPOCH" ]]; then
    : # picked up commit time
  else
    SOURCE_DATE_EPOCH=1700000000  # 2023-11-14T22:13:20Z, arbitrary stable fallback
  fi
fi
STABLE_DATE="$(date -u -r "$SOURCE_DATE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ')"

echo "Carving sidecar from: $APP"
echo "  desktop version:   $APP_VERSION"
echo "  sidecar VERSION:   $SIDECAR_VERSION"
echo "  pinned epoch:      $SOURCE_DATE_EPOCH ($STABLE_DATE)"
echo "  output tarball:    $TARBALL"

# Mach-O sanity gate. `file` is the Apple-native classifier; we look
# for the Mach-O magic regardless of file extension. Guards against
# a stub build slipping into the release pipeline (e.g. one that ran
# with BUNDLE_SIDECAR=0). Threshold is loose (≥ 10) because the real
# sidecar contains hundreds — the bundled python3 + dozens of .so
# extensions from mlx-lm/mlx-vlm/transformers.
MACHO_COUNT="$(find "$SIDECAR_DIR" -type f -print0 \
  | xargs -0 file 2>/dev/null \
  | grep -c 'Mach-O' || true)"
if [[ "$MACHO_COUNT" -lt 10 ]]; then
  echo "::error::Sidecar appears truncated — only $MACHO_COUNT Mach-O binaries under $SIDECAR_DIR" >&2
  echo "(expected hundreds — bundled python3 + .so extensions for mlx-lm/mlx-vlm/transformers)" >&2
  exit 1
fi

# Pack via Python tarfile so we can pin metadata that bsdtar can't.
#
# Why not bsdtar (macOS-native): no portable way to override mtime on
# captured entries; pax exthdr option syntax not supported; ownership
# normalisation flags don't compose cleanly with --null -T -.
#
# What Python gives us:
#   - mtime/uid/gid/uname/gname pinned via TarInfo overrides
#   - USTAR_FORMAT (no pax exthdrs, simpler binary layout)
#   - gzip mtime=0 (gzip header is content-only — no wall-clock stamp)
#   - lexicographic sort of dirs and files via sorted() — stable
#     across machines (find's filesystem order is not)
#   - streaming write (mode="w|") keeps memory bounded for the 280 MB
#     sidecar tree
python3 - "$SIDECAR_DIR" "$TARBALL" "$SOURCE_DATE_EPOCH" <<'PY'
import gzip
import os
import sys
import tarfile

src_dir = sys.argv[1]
out_path = sys.argv[2]
epoch = int(sys.argv[3])

# Walk lexicographically — deterministic order independent of FS layout.
entries = []
for root, dirs, files in os.walk(src_dir, followlinks=False):
    dirs.sort()
    for d in dirs:
        entries.append(os.path.join(root, d))
    for f in sorted(files):
        entries.append(os.path.join(root, f))
# Re-sort the flat list for full lexicographic determinism.
entries.sort()

# arcnames are relative to the parent of the sidecar dir so the
# archive's top-level entry is "rapid-mlx/" (mirrors the on-disk
# layout the bootstrapper will extract to Contents/Resources/).
prefix = os.path.dirname(src_dir)

def normalise(ti):
    ti.mtime = epoch
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    return ti

# Two-stage: tarfile streams into a gzip writer with mtime=0 so even
# the gzip header is content-only. Use mode="w|" (streaming) so we
# don't buffer the entire 280 MB tarball in RAM.
with gzip.GzipFile(out_path, mode="wb", compresslevel=9, mtime=0) as gz:
    with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
        # Include the leaf dir itself first so extractors see the
        # parent directory entry before its children.
        leaf_ti = tar.gettarinfo(src_dir, arcname=os.path.relpath(src_dir, prefix))
        tar.addfile(normalise(leaf_ti))

        for path in entries:
            arcname = os.path.relpath(path, prefix)
            ti = tar.gettarinfo(path, arcname=arcname)
            if ti is None:
                continue
            normalise(ti)
            if ti.isfile():
                with open(path, "rb") as f:
                    tar.addfile(ti, f)
            else:
                # Directories, symlinks, hard links — body-less.
                tar.addfile(ti)
PY

if [[ ! -s "$TARBALL" ]]; then
  echo "::error::tarfile output empty: $TARBALL" >&2
  exit 1
fi

SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
SIZE="$(stat -f '%z' "$TARBALL")"
SIZE_MB="$(awk -v b="$SIZE" 'BEGIN { printf "%.1f", b/1024/1024 }')"

# Manifest emitted via Python's json module so version strings and any
# future fields are safely encoded (escapes quotes, control chars,
# Unicode) without hand-rolled string interpolation.
python3 - \
    "$MANIFEST" \
    "$APP_VERSION" \
    "$SIDECAR_VERSION" \
    "$(basename "$TARBALL")" \
    "$SHA" \
    "$SIZE" \
    "$MACHO_COUNT" \
    "$STABLE_DATE" \
<<'PY'
import json, sys
out_path, desktop, sidecar, name, sha, size, macho, built = sys.argv[1:9]
with open(out_path, "w") as f:
    json.dump({
        "schema_version": 1,
        "artifact": "rapid-mlx-sidecar",
        "desktop_version": desktop,
        "sidecar_version": sidecar,
        "tarball_name": name,
        "tarball_sha256": sha,
        "tarball_size_bytes": int(size),
        "macho_file_count": int(macho),
        "built_at_utc": built,
    }, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo ""
echo "Sidecar tarball:"
echo "  path:     $TARBALL"
echo "  size:     ${SIZE_MB} MB (${SIZE} bytes)"
echo "  sha256:   ${SHA}"
echo "  files:    ${MACHO_COUNT} Mach-O binaries"
echo "  manifest: $MANIFEST"

# CI-friendly outputs when running under GitHub Actions. Write via
# stdout-quote-safe heredoc to GITHUB_OUTPUT to avoid output-injection
# from version strings (the regex above already constrains APP_VERSION,
# but stick to the safe pattern out of habit).
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "tarball_path=${TARBALL}"
    echo "tarball_name=$(basename "$TARBALL")"
    echo "tarball_sha256=${SHA}"
    echo "tarball_size_bytes=${SIZE}"
    echo "manifest_path=${MANIFEST}"
    echo "desktop_version=${APP_VERSION}"
  } >> "$GITHUB_OUTPUT"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Sidecar tarball"
    echo ""
    echo "| field | value |"
    echo "|-------|-------|"
    echo "| desktop_version | \`${APP_VERSION}\` |"
    echo "| sidecar_version | \`${SIDECAR_VERSION}\` |"
    echo "| tarball | \`$(basename "$TARBALL")\` |"
    echo "| size | ${SIZE_MB} MB |"
    echo "| sha256 | \`${SHA}\` |"
    echo "| macho_count | ${MACHO_COUNT} |"
    echo "| epoch (UTC) | ${STABLE_DATE} |"
  } >> "$GITHUB_STEP_SUMMARY"
fi
