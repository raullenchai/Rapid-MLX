#!/usr/bin/env bash
# build.sh — produce a launchable "Rapid-MLX Desktop.app" from the SPM executable.
#
# Naming (see issue #164): the .app bundle on disk is
# ``Rapid-MLX Desktop.app`` for fresh installs. Existing 0.5.21 installs
# auto-update in place — the updater replaces the running bundle at its
# existing path, so a user with ``/Applications/Rapid.app``
# keeps that path forever (or until they manually re-install from the
# DMG). The display name everywhere (Dock, About, menus) reads
# "Rapid-MLX Desktop" via CFBundleDisplayName.
#
# SwiftUI executables built by `swift build` are bare Mach-O binaries; macOS
# needs them wrapped in a versioned bundle (Contents/MacOS + Info.plist +
# Resources) before LaunchServices will treat them as an app. We assemble
# that bundle manually and codesign it so Gatekeeper recognises a
# well-formed signature (the previous Tauri build hit the "damaged" trap
# whenever the seal was incomplete — see archive/tauri-v0.1).
#
# Signing is env-driven: ad-hoc by default (local dev, no Apple account
# needed), or a real Developer ID identity when CODESIGN_IDENTITY is set
# (CI release path — adds hardened runtime + secure timestamp, which
# Apple notarisation requires). See scripts/notarize.sh + the release
# GitHub Actions workflow.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.build/release"
APP="$ROOT/build/Rapid-MLX Desktop.app"
CONTENTS="$APP/Contents"

CONFIG="${RAPID_BUILD_CONFIG:-release}"

cd "$ROOT"
echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

echo "==> assembling Rapid-MLX Desktop.app"
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources" "$CONTENTS/Frameworks"
cp "$ROOT/.build/$CONFIG/Rapid" "$CONTENTS/MacOS/Rapid"
cp "$ROOT/Resources/Info.plist" "$CONTENTS/Info.plist"
cp "$ROOT/Resources/AppIcon.icns" "$CONTENTS/Resources/AppIcon.icns"

# SwiftPM links Sparkle dynamically but this app is assembled by hand rather
# than by Xcode, so its normal "Embed & Sign" phase does not run for us. Copy
# the complete framework (including symlinks and installer helpers), then add
# the conventional app-framework search path to the executable before any
# code signing seals it.
SPARKLE_FRAMEWORK_SRC="$ROOT/.build/$CONFIG/Sparkle.framework"
SPARKLE_FRAMEWORK_DST="$CONTENTS/Frameworks/Sparkle.framework"
if [[ ! -d "$SPARKLE_FRAMEWORK_SRC" ]]; then
    echo "ERR: SwiftPM Sparkle.framework missing: $SPARKLE_FRAMEWORK_SRC" >&2
    exit 1
fi
ditto "$SPARKLE_FRAMEWORK_SRC" "$SPARKLE_FRAMEWORK_DST"
if ! otool -l "$CONTENTS/MacOS/Rapid" | grep -Fq '@executable_path/../Frameworks'; then
    install_name_tool -add_rpath '@executable_path/../Frameworks' "$CONTENTS/MacOS/Rapid"
fi

# The EdDSA public key is release configuration, injected from an Actions
# variable. A local build without it deliberately omits SUPublicEDKey and the
# app keeps using the legacy updater fallback. Fail closed on a malformed key
# instead of shipping a Sparkle client that can never validate an update.
if [[ -n "${SPARKLE_PUBLIC_ED_KEY:-}" ]]; then
    if ! printf '%s' "$SPARKLE_PUBLIC_ED_KEY" \
        | openssl base64 -d -A 2>/dev/null \
        | wc -c | tr -d ' ' | grep -qx '32'; then
        echo "ERR: SPARKLE_PUBLIC_ED_KEY must be base64 for a 32-byte Ed25519 public key" >&2
        exit 1
    fi
    plutil -replace SUPublicEDKey -string "$SPARKLE_PUBLIC_ED_KEY" "$CONTENTS/Info.plist"
fi

# Sentry is being removed from the monorepo app (SENTRY_DSN is no longer
# a required/plumbed secret). This block is retained as an inert no-op:
# if SENTRY_DSN is ever set it is still validated + injected, but the
# release workflow does not pass it and the default build leaves Sentry
# disabled with the in-app GitHub-issue route as the reporting fallback.
if [[ -n "${SENTRY_DSN:-}" ]]; then
    if [[ "$SENTRY_DSN" != https://* ]]; then
        echo "error: SENTRY_DSN must use HTTPS" >&2
        exit 1
    fi
    plutil -replace SentryDSN -string "$SENTRY_DSN" "$CONTENTS/Info.plist"
fi

# Privacy manifest: copy it into the bundle only if the Swift sources
# still ship one. Guarded (was unconditional in the desktop repo) so the
# build does not hard-fail once Sentry — and its statically-linked
# PrivacyInfo.xcprivacy — is removed from Sources/ by the app rebrand.
if [[ -f "$ROOT/Sources/Rapid/Resources/PrivacyInfo.xcprivacy" ]]; then
    cp "$ROOT/Sources/Rapid/Resources/PrivacyInfo.xcprivacy" \
       "$CONTENTS/Resources/PrivacyInfo.xcprivacy"
fi

# v0.5.10: SHIP-BLOCKER fix. v0.5.9 attempted to satisfy
# ``Bundle.module`` by copying SPM's ``Rapid_Rapid.bundle`` into
# ``Contents/Resources/``. That looked plausible but doesn't work:
# SPM's auto-generated ``resource_bundle_accessor.swift`` for
# executable targets only probes
# ``Bundle.main.bundleURL/Rapid_Rapid.bundle`` (i.e. the .app's
# TOP level, sibling to ``Contents/``) — not anywhere inside
# ``Contents/``. On miss it ``fatalError``s inside
# ``dispatch_once``, which crashed the v0.5.9 DMG on first
# render of ``CheetahLogo``. Putting the bundle at the .app
# top-level satisfies the accessor but trips
# ``codesign --deep --strict`` (the SPM-emitted directory has
# no ``Info.plist`` and Apple's .app structure expects nothing
# at the top besides ``Contents/``).
#
# The correct fix:
#   * ``CheetahLogo`` no longer uses ``Bundle.module``; it uses
#     ``Bundle.main.url(forResource:)``.
#   * We copy the PNGs into ``Contents/Resources/`` as flat
#     files, so ``Bundle.main`` resolves them the way macOS
#     resource lookup expects.
# Source-of-truth for the assets stays under
# ``Sources/Rapid/Resources/`` so SPM keeps them on the test
# target's ``Bundle.module`` (used by the snapshot suite via
# the ``BundleFinder`` walk in ``CheetahLogo``).
for asset in cheetah.png cheetah-sm.png; do
    if [[ -f "$ROOT/Sources/Rapid/Resources/$asset" ]]; then
        cp "$ROOT/Sources/Rapid/Resources/$asset" "$CONTENTS/Resources/$asset"
    else
        echo "warning: Sources/Rapid/Resources/$asset missing — CheetahLogo will fall back to SF Symbol" >&2
    fi
done

# Localizable.xcstrings: same Bundle.main vs Bundle.module story as
# the PNGs above. SPM compiles the catalog into Rapid_Rapid.bundle
# for tests, but the production .app uses Bundle.main and ships
# without the SPM bundle wrapper (codesign --deep --strict rejects
# the SPM bundle directory at the .app root). Copy the catalog as
# a flat file into Contents/Resources/ so macOS 14+'s
# NSLocalizedString / SwiftUI String catalog reader resolves it.
# Without this, zh-Hans users see English strings in the shipped
# DMG even though `swift test` reports a fully translated catalog.
if [[ -f "$ROOT/Sources/Rapid/Resources/Localizable.xcstrings" ]]; then
    cp "$ROOT/Sources/Rapid/Resources/Localizable.xcstrings" "$CONTENTS/Resources/Localizable.xcstrings"
else
    echo "warning: Localizable.xcstrings missing — shipped .app will fall back to English only" >&2
fi

# v0.7.16: benchmark-scores.json drives the picker hover tooltip.
# Same Bundle.main vs Bundle.module story as the PNGs / xcstrings
# above — ``BenchScoresCatalog`` falls back to
# ``Bundle.main.url(forResource:)`` in the shipped .app, so the JSON
# must live as a flat file under Contents/Resources/. SPM also embeds
# it via Package.swift's ``.process`` block so tests resolve it via
# the SPM resource bundle; the production .app uses this flat copy.
if [[ -f "$ROOT/Sources/Rapid/Resources/benchmark-scores.json" ]]; then
    cp "$ROOT/Sources/Rapid/Resources/benchmark-scores.json" "$CONTENTS/Resources/benchmark-scores.json"
else
    echo "warning: benchmark-scores.json missing — picker hover tooltip will show dashed bars only" >&2
fi

# The recommendation catalog is owned by the Python package so the CLI and
# desktop app consume one physical source file. Copy that SSOT into the shipped
# app; SwiftPM source-checkout tests load it directly from ../../vllm_mlx.
RECOMMENDATIONS_SRC="$ROOT/../../vllm_mlx/model_recommendations.json"
if [[ -f "$RECOMMENDATIONS_SRC" ]]; then
    cp "$RECOMMENDATIONS_SRC" "$CONTENTS/Resources/model_recommendations.json"
else
    echo "ERR: shared model_recommendations.json missing" >&2
    exit 1
fi

# SwiftMath's upstream resource accessor looks beside `Contents/`, a location
# that makes a strict app signature invalid. Our vendored resolver loads this
# nested bundle through Bundle.main instead. Keep the full upstream bundle so
# every public MathFont case remains usable and its font licences travel with
# the corresponding binaries.
SWIFTMATH_FONTS="$ROOT/Vendor/SwiftMath/Sources/SwiftMath/mathFonts.bundle"
if [[ ! -d "$SWIFTMATH_FONTS" ]]; then
    echo "ERR: vendored SwiftMath font bundle missing: $SWIFTMATH_FONTS" >&2
    exit 1
fi
cp -R "$SWIFTMATH_FONTS" "$CONTENTS/Resources/mathFonts.bundle"

# Third-party license texts (#1596). BSD-2-Clause (swift-cmark) and the MIT
# licenses of the other linked Swift packages ask their notice to travel "with
# the distribution" — and a downloaded .app IS that distribution. Assembling
# only the repo's THIRD_PARTY.md does not satisfy that; the notice has to be
# inside the bundle. `stage-licenses.sh` copies each into
# Contents/Resources/Licenses/, driven off the resolved pins in
# Package.resolved (so stale checkouts are ignored) plus the in-tree vendored
# SwiftMath notice, and FAILS the build if any linked package has no license
# file — in the same spirit as the offline link-target test from #1595. It is a
# separate script so the test suite can exercise it against fixtures. The Python
# payload's licenses already travel via pip's `*.dist-info/licenses/` under the
# staged sidecar and are not re-copied.
echo "==> staging third-party license texts"
"$ROOT/scripts/stage-licenses.sh" \
    "$ROOT/Package.resolved" \
    "$ROOT/.build/checkouts" \
    "$ROOT/Vendor/SwiftMath/LICENSE" \
    "$CONTENTS/Resources/Licenses"

# App accent colour. AppKit paints NSMenu highlights, checkboxes and focus
# rings with the app's accent colour, and nothing in SwiftUI reaches those:
# ``.tint()`` styles SwiftUI's own views only, so without this the ··· row
# menu highlighted in stock macOS system blue — the one colour the v0.6
# palette deliberately avoids.
#
# The lookup is Info.plist ``NSAccentColorName`` → a NAMED COLOR inside a
# COMPILED Assets.car. A raw .xcassets directory is not readable at
# runtime, so it has to go through actool. actool ships with Xcode, not
# the Command Line Tools: a machine with only the CLT builds a .app whose
# menus fall back to system blue rather than failing the build, which is a
# cosmetic degradation and not worth blocking a local build over.
ASSETS_SRC="$ROOT/Sources/Rapid/Resources/Assets.xcassets"
if [[ -d "$ASSETS_SRC" ]]; then
    if ACTOOL="$(xcrun --find actool 2>/dev/null)"; then
        "$ACTOOL" "$ASSETS_SRC" \
            --compile "$CONTENTS/Resources" \
            --platform macosx \
            --minimum-deployment-target 14.0 \
            --output-format human-readable-text \
            >/dev/null
    else
        echo "warning: actool not found (Xcode not selected) — menus will use the system accent colour" >&2
    fi
fi

# v0.6.6: embed rapid-mlx sidecar (issue #171). MONOREPO: the sidecar is
# built from the rapid-mlx engine that lives at the repository ROOT (two
# levels up from apps/rapid-mac). There is NO git submodule here — the
# engine checked out at this commit IS the version contract, reviewable
# in the same PR diff. (The desktop repo built the sidecar from a
# ``third_party/rapid-mlx`` submodule; build-sidecar.sh now points
# RAPID_MLX_SOURCE at the engine root instead — see ENGINE_ROOT below.)
#
# Why staging happens BEFORE codesign below: the outer codesign step
# seals every file under Contents/ into CodeResources. The sidecar's
# 77 Mach-Os are individually signed inside build-sidecar.sh; the
# outer (non-deep) codesign then hashes their bytes into the resource
# envelope so any post-build tampering invalidates the seal.
#
# SKIP_SIDECAR=1 short-circuits the 5-10 min pip install for local
# iteration. The resulting .app has no bundled rapid-mlx, so
# ServerLocator falls through to .userInstalled / .path / brew slots
# — fine for dev, not for shipping.
#
# Normal local builds retain the last complete sidecar stage and reuse it when
# the rapid-mlx commit and all packaging inputs are unchanged. Set
# FORCE_SIDECAR_REBUILD=1 to refresh floating Python dependencies explicitly.
# Developer ID release builds always rebuild and verify from scratch.
#
# scripts/build-sidecar.sh lives alongside this script; the packaging
# logic is the app's concern. In the desktop repo the pip-install source
# was a third_party/rapid-mlx submodule — MONOREPO removes that
# indirection: the pip-install source is the engine at the repo root.
# MONOREPO: the engine source is the repository root, two levels up from
# apps/rapid-mac (this script lives at apps/rapid-mac/scripts/). Override
# RAPID_MLX_ENGINE_ROOT to point the sidecar build at a different engine
# checkout. build-sidecar.sh independently defaults RAPID_MLX_SOURCE to
# the same location; we compute it here too so the preflight check and
# the cache-key/version derivation below agree with what the sidecar
# build will actually pip-install.
ENGINE_ROOT="${RAPID_MLX_ENGINE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
SKIP_SIDECAR="${SKIP_SIDECAR:-0}"
SIDECAR_SCRIPT="$ROOT/scripts/build-sidecar.sh"
if [[ "$SKIP_SIDECAR" == "1" ]]; then
    echo "==> SKIPPING sidecar bundling (SKIP_SIDECAR=1)"
elif [[ ! -f "$SIDECAR_SCRIPT" ]]; then
    echo "ERR: scripts/build-sidecar.sh missing at:" >&2
    echo "     $SIDECAR_SCRIPT" >&2
    echo "     (Check git status / re-clone if the file is gone.)" >&2
    exit 1
elif [[ ! -d "$ENGINE_ROOT" || ! -f "$ENGINE_ROOT/pyproject.toml" ]]; then
    echo "ERR: rapid-mlx engine source not found at the monorepo root:" >&2
    echo "     $ENGINE_ROOT" >&2
    echo "     The sidecar build pip-installs the engine into the bundle." >&2
    echo "     Expected apps/rapid-mac to be nested inside the rapid-mlx" >&2
    echo "     engine repo. Set RAPID_MLX_ENGINE_ROOT to an engine checkout," >&2
    echo "     or set SKIP_SIDECAR=1 for a dev build without a bundled engine." >&2
    exit 1
else
    SIDECAR_STAGE="$ROOT/build/sidecar-stage"
    SIDECAR_CACHE_STAMP="$SIDECAR_STAGE/.rapid-sidecar-cache-key"
    FORCE_SIDECAR_REBUILD="${FORCE_SIDECAR_REBUILD:-0}"
    SIDECAR_CACHE_KEY=""
    SIDECAR_CACHE_HIT=0

    # Cache only clean, default-source local builds. A release identity must
    # sign and verify every Mach-O afresh; a dirty source tree has no stable
    # commit identity and must not be hidden behind an old bundle.
    if [[ "${CODESIGN_IDENTITY:--}" == "-" && -z "${RAPID_MLX_SOURCE:-}" ]]; then
        SIDECAR_SOURCE_SHA="$(git -C "$ENGINE_ROOT" rev-parse HEAD 2>/dev/null || true)"
        SIDECAR_SOURCE_STATUS="$(git -C "$ENGINE_ROOT" status --porcelain --untracked-files=normal 2>/dev/null || echo unavailable)"
        if [[ -n "$SIDECAR_SOURCE_SHA" && -z "$SIDECAR_SOURCE_STATUS" ]]; then
            SIDECAR_RECIPE_HASH="$(
                shasum -a 256 "$SIDECAR_SCRIPT" \
                    "$ROOT/scripts/sidecar-shim.sh" \
                    "$ROOT/scripts/sidecar-entitlements.plist" \
                | awk '{print $1}' \
                | shasum -a 256 \
                | awk '{print $1}'
            )"
            SIDECAR_CACHE_KEY="v1:${SIDECAR_SOURCE_SHA}:${SIDECAR_RECIPE_HASH}"
            if [[ "$FORCE_SIDECAR_REBUILD" != "1" \
                && -f "$SIDECAR_CACHE_STAMP" \
                && "$(cat "$SIDECAR_CACHE_STAMP")" == "$SIDECAR_CACHE_KEY" \
                && -x "$SIDECAR_STAGE/rapid-mlx/python/bin/python3.12" \
                && -x "$SIDECAR_STAGE/rapid-mlx/bin/rapid-mlx" \
                && -d "$SIDECAR_STAGE/rapid-mlx/site-packages/vllm_mlx" ]]; then
                SIDECAR_CACHE_HIT=1
            fi
        fi
    fi

    SIDECAR_ARGS=(--out "$SIDECAR_STAGE")
    if [[ "${CODESIGN_IDENTITY:--}" != "-" ]]; then
        # CI release path — use the Developer ID identity already
        # imported into the temp keychain. build-sidecar.sh codesigns
        # the 77 Mach-Os; the outer step below seals them into the
        # resource envelope without re-signing (no --deep).
        SIDECAR_ARGS+=(--developer-id "$CODESIGN_IDENTITY")
    else
        # Local dev — adhoc codesign + skip smoke (smoke needs a
        # Python that can ``import mlx``, which a dev-machine global
        # Python can't reliably do).
        SIDECAR_ARGS+=(--skip-codesign --skip-verify)
    fi
    if [[ "$SIDECAR_CACHE_HIT" == "1" ]]; then
        echo "==> reusing cached rapid-mlx sidecar (${SIDECAR_SOURCE_SHA:0:12})"
        echo "    set FORCE_SIDECAR_REBUILD=1 to rebuild Python dependencies"
    else
        echo "==> building rapid-mlx sidecar"
        rm -rf "$SIDECAR_STAGE"
        bash "$SIDECAR_SCRIPT" "${SIDECAR_ARGS[@]}"
        if [[ -n "$SIDECAR_CACHE_KEY" ]]; then
            printf '%s\n' "$SIDECAR_CACHE_KEY" > "$SIDECAR_CACHE_STAMP"
        fi
    fi

    # Stage into the .app at the exact path ServerLocator's .bundled
    # case expects (Sources/Rapid/Server/ServerLocator.swift:211-215).
    rm -rf "$CONTENTS/Resources/rapid-mlx"
    cp -R "$SIDECAR_STAGE/rapid-mlx" "$CONTENTS/Resources/rapid-mlx"

    # Stamp VERSION from the engine's own version. MONOREPO: the sidecar
    # IS the engine, so its VERSION is the engine version. Downstream
    # consumers (AboutPanel, bootstrapper manifest emit at
    # scripts/build-sidecar-tarball.sh, BootstrapCoordinator validator)
    # all require a strict SemVer-shaped string.
    #
    # SSOT precedence:
    #   1. the engine's pyproject.toml ``[project] version = "X.Y.Z"`` at
    #      the repo root (the canonical engine version in this monorepo);
    #   2. an engine ``v*`` git tag exactly at HEAD (--points-at), then
    #   3. the nearest preceding engine ``v*`` tag (git describe).
    #
    # Hard-fail if the final value isn't dotted-digit SemVer — the
    # alternative is shipping a SHA into latest.json, which the
    # bootstrapper validator rejects and which bricked 100% of slim-DMG
    # installs on v0.8.6 (issue #411). Grammar MUST stay in lockstep with
    # ``Sources/Rapid/Bootstrapper/BootstrapCoordinator.swift``'s
    # ``isValidVersionString`` (strip optional ``v``/``V`` then every
    # dot-separated segment must be pure decimal digits). No pre-release
    # or build suffixes (codex #412 r1 BLOCKING).
    SIDECAR_SEMVER_RE='^[0-9]+(\.[0-9]+)+$'
    SIDECAR_VERSION=""
    if [[ -f "$ENGINE_ROOT/pyproject.toml" ]]; then
        # Scope to the [project] table so a version key in another table
        # (e.g. [tool.*]) can't be picked by mistake. Strip surrounding
        # quotes from the RHS.
        SIDECAR_VERSION="$(awk '
            /^\[project\]/ {inproj=1; next}
            /^\[/ {inproj=0}
            inproj && /^[[:space:]]*version[[:space:]]*=/ {
                sub(/^[^=]*=[[:space:]]*/, "")
                gsub(/["'"'"']/, "")
                print
                exit
            }' "$ENGINE_ROOT/pyproject.toml" 2>/dev/null || true)"
    fi
    if [[ -z "$SIDECAR_VERSION" || ! "$SIDECAR_VERSION" =~ $SIDECAR_SEMVER_RE ]]; then
        SIDECAR_EXACT_TAG="$(cd "$ENGINE_ROOT" && git tag --points-at HEAD --list 'v[0-9]*' 2>/dev/null | sort -V | tail -1 || true)"
        if [[ -n "$SIDECAR_EXACT_TAG" ]]; then
            SIDECAR_VERSION="${SIDECAR_EXACT_TAG#v}"
        fi
    fi
    if [[ -z "$SIDECAR_VERSION" || ! "$SIDECAR_VERSION" =~ $SIDECAR_SEMVER_RE ]]; then
        SIDECAR_DESCRIBE="$(cd "$ENGINE_ROOT" && git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)"
        if [[ -n "$SIDECAR_DESCRIBE" ]]; then
            SIDECAR_VERSION="${SIDECAR_DESCRIBE#v}"
        fi
    fi
    if [[ -z "$SIDECAR_VERSION" || ! "$SIDECAR_VERSION" =~ $SIDECAR_SEMVER_RE ]]; then
        echo "::error::Could not derive a SemVer-shaped engine/sidecar version (got '${SIDECAR_VERSION:-<empty>}'). Tried $ENGINE_ROOT/pyproject.toml [project].version and engine 'v*' git tags. Ensure the engine pyproject.toml carries a dotted-digit version, or tag a 'v*' release at the engine root (#411)." >&2
        exit 1
    fi
    printf '%s\n' "$SIDECAR_VERSION" > "$CONTENTS/Resources/rapid-mlx/VERSION"
    echo "==> sidecar staged ($SIDECAR_VERSION)"
fi

# v0.7.1 #229: bundle the lfm2.5-1b-4bit weights inside the DMG so
# first-launch chat happens with zero HuggingFace round-trips. The HF
# Hub snapshot layout (``models--<owner>--<name>``) lives under
# ``Contents/Resources/models/hf-cache/hub/``; ``BundledModel.swift``
# symlinks it into the user's ``~/.cache/huggingface/hub/`` at first
# launch so the sidecar resolves it like any other cached model.
#
# post-v0.7.3: DEFAULT-OFF since the .app size cap from #242 (500 MB)
# forces us to drop the ~340 MB of weights. First launch instead pulls
# from the R2 mirror (BundledModel falls through to the existing
# DownloadManager path; ~90 s on home Wi-Fi at ~4 MB/s, instant on
# wired). Set ``BUNDLE_MODEL=1`` locally for offline / airgapped builds
# where the user can't reach the R2 mirror on first launch.
#
# ``SKIP_BUNDLED_MODEL=1`` is preserved as a backward-compatible
# negative gate so existing CI invocations don't blow up — it forces
# the skip path even if a future maintainer flips the default back to
# 1.
BUNDLE_MODEL="${BUNDLE_MODEL:-0}"
SKIP_BUNDLED_MODEL="${SKIP_BUNDLED_MODEL:-0}"
BUNDLED_MODEL_REPO="${BUNDLED_MODEL_REPO:-mlx-community/LFM2.5-1.2B-Instruct-4bit}"
# HF Hub cache encodes ``owner/name`` as ``models--owner--name`` —
# each ``/`` becomes a literal ``--`` (double dash). Matches the
# derivation in Sources/Rapid/Server/BundledModel.swift's
# ``bundledCacheDirName`` so the symlink target the desktop lays
# down at first launch points at exactly what we stage here.
BUNDLED_MODEL_DIRNAME="models--$(echo "$BUNDLED_MODEL_REPO" | sed 's|/|--|g')"
MODEL_CACHE_ROOT="$CONTENTS/Resources/models/hf-cache"
if [[ "$BUNDLE_MODEL" != "1" ]] || [[ "$SKIP_BUNDLED_MODEL" == "1" ]]; then
    echo "==> SKIPPING bundled-model staging (BUNDLE_MODEL=$BUNDLE_MODEL, SKIP_BUNDLED_MODEL=$SKIP_BUNDLED_MODEL)"
    echo "    first launch will pull $BUNDLED_MODEL_REPO from the R2 mirror"
elif [[ -d "$MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME" ]]; then
    # Already staged (re-running build.sh without `rm -rf build/`).
    # Re-stamp the path so the log line confirms the bundle survived.
    echo "==> bundled model already staged at $MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME"
else
    echo "==> downloading bundled model $BUNDLED_MODEL_REPO (~500 MB)"
    mkdir -p "$MODEL_CACHE_ROOT"
    # Use huggingface_hub.snapshot_download — the same path the sidecar
    # consults at load time. ``HF_HOME`` (NOT ``HF_HUB_CACHE``) points at
    # our staging directory: HF_HOME implies the ``hub/`` subdirectory
    # so the resulting tree at
    # ``$HF_HOME/hub/models--<owner>--<name>/`` mirrors the shape of the
    # user's default ``~/.cache/huggingface/hub/`` cache. HF_HUB_CACHE
    # would have produced a FLAT
    # ``$HF_HUB_CACHE/models--<owner>--<name>/`` layout WITHOUT the
    # ``hub/`` segment — the symlink ``BundledModel.swift`` lays down
    # would then point at a non-existent path. (Verified with a direct
    # snapshot_download probe on the host before wiring this in.)
    # ``HF_HUB_DISABLE_XET=1`` matches the sidecar default added in the
    # submodule's "disable hf_xet by default" fix — cas-bridge.xethub.hf.co
    # has been the dominant download stall surface on Apple Silicon.
    #
    # Pulls from the host's Python (NOT the bundled python — that hasn't
    # been signed yet at this point in the build). The host needs
    # huggingface_hub on PATH; ``pip install --user huggingface_hub`` is
    # listed in the Contributing guide. If it's missing we surface a
    # clear error rather than a Python ImportError stack.
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "ERR: huggingface_hub not installed on host Python." >&2
        echo "     Install with: pip install --user huggingface_hub" >&2
        echo "     Or unset BUNDLE_MODEL to skip the bundled-model staging" >&2
        echo "     (first launch then pulls from the R2 mirror)." >&2
        exit 1
    fi
    HF_HOME="$MODEL_CACHE_ROOT" HF_HUB_DISABLE_XET=1 \
        python3 -c "
import sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download('$BUNDLED_MODEL_REPO')
    print('snapshot:', path)
except Exception as e:
    print('ERR: snapshot_download failed:', e, file=sys.stderr)
    sys.exit(1)
"
    # snapshot_download writes to ``$HF_HOME/hub/...`` — verify the
    # directory we expect ended up where BundledModel.swift will look
    # for it.
    if [[ ! -d "$MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME" ]]; then
        echo "ERR: expected snapshot directory missing:" >&2
        echo "     $MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME" >&2
        ls -la "$MODEL_CACHE_ROOT/hub/" >&2 || true
        exit 1
    fi
    BUNDLED_SIZE="$(du -sh "$MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME" | cut -f1)"
    echo "==> bundled model staged ($BUNDLED_SIZE) at"
    echo "    $MODEL_CACHE_ROOT/hub/$BUNDLED_MODEL_DIRNAME"
fi

# Signing: ad-hoc by default (local dev), real Developer ID when
# CODESIGN_IDENTITY is set (CI release). CODESIGN_IDENTITY can be the
# identity's common name ("Developer ID Application: … (TEAMID)") or its
# SHA-1 hash from `security find-identity -v -p codesigning`.
SIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
if [[ "$SIGN_IDENTITY" == "-" ]]; then
    echo "==> ad-hoc codesign"
    codesign --force --deep --sign - "$APP"
    codesign --verify --deep --strict "$APP"
else
    echo "==> Developer ID codesign ($SIGN_IDENTITY)"
    # Xcode's CodeSignOnCopy phase normally re-signs Sparkle's nested code
    # with the host application's Team ID. Reproduce that from the inside out
    # so Sparkle's IPC team checks accept its helpers in this hand-built app.
    SPARKLE_VERSION="$SPARKLE_FRAMEWORK_DST/Versions/B"
    for nested in \
        "$SPARKLE_VERSION/Autoupdate" \
        "$SPARKLE_VERSION/Updater.app" \
        "$SPARKLE_VERSION/XPCServices/Downloader.xpc" \
        "$SPARKLE_VERSION/XPCServices/Installer.xpc"; do
        codesign --force --options runtime --timestamp \
            --preserve-metadata=identifier,entitlements,flags \
            --sign "$SIGN_IDENTITY" "$nested"
    done
    codesign --force --options runtime --timestamp \
        --preserve-metadata=identifier,entitlements,flags \
        --sign "$SIGN_IDENTITY" "$SPARKLE_FRAMEWORK_DST"
    codesign --verify --deep --strict "$SPARKLE_FRAMEWORK_DST"
    # No --deep: the sidecar's Mach-Os under Contents/Resources/rapid-mlx/
    # are already individually signed by build-sidecar.sh; this outer
    # (non-deep) codesign hashes their bytes into the .app's resource
    # envelope without re-signing them, which is what Apple recommends for
    # distribution signing (and --deep would clobber the per-Mach-O
    # sidecar entitlements).
    # --options runtime = hardened runtime; --timestamp = secure
    # timestamp; both are prerequisites for notarisation.
    # Resources/Rapid.entitlements carries the 3 hardened-runtime JIT keys
    # (allow-jit, disable-library-validation, allow-unsigned-executable-
    # memory) the bundled Python/MLX sidecar needs — see the file's own
    # comments and scripts/sidecar-entitlements.plist (the per-Mach-O
    # counterpart) — plus device.audio-input, without which dictation's
    # microphone request is silently refused in hardened builds (#2134).
    # No app-sandbox: Rapid is non-sandboxed. The keys are flagged
    # informationally (not errors) in the notary report.
    codesign --force --options runtime --timestamp \
        --entitlements "$ROOT/Resources/Rapid.entitlements" \
        --sign "$SIGN_IDENTITY" "$APP"
    codesign --verify --strict "$APP"
    # Dictation is dead without the audio-input entitlement in the SEALED
    # signature (not just the source plist) — 0.12.16 shipped that way
    # (#2134). Fail the build rather than notarize another silent brick.
    # Parse the value, don't grep the name: a sealed <false/> must fail too.
    audio_input=$(codesign -d --entitlements :- "$APP" 2>/dev/null \
        | plutil -extract 'com\.apple\.security\.device\.audio-input' raw -o - - 2>/dev/null || true)
    if [[ "$audio_input" != "true" ]]; then
        echo "ERROR: sealed entitlements lack com.apple.security.device.audio-input=true (got: '${audio_input:-absent}') — dictation would ship broken (#2134)" >&2
        exit 1
    fi
    codesign -dv --verbose=4 "$APP" 2>&1 | grep -E 'Authority|TeamIdentifier|flags=' || true
fi

echo
echo "Rapid-MLX Desktop.app ready at: $APP"
echo "Launch with: open '$APP'"
