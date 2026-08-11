#!/usr/bin/env bash
# Run the first native XCUITest journey against the production-shaped app.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT/Tests/RapidUITests/RapidUITests.xcodeproj"
APP="$ROOT/build/Rapid-MLX Desktop.app"
RESULT_BUNDLE="${RAPID_XCUI_RESULT_BUNDLE:-$ROOT/build/RapidUITests-$(date +%s)-$$.xcresult}"

[[ -d "$APP" ]] || { echo "error: build the app first: $APP" >&2; exit 1; }
xcodebuild -version >/dev/null 2>&1 || {
    echo "error: full Xcode is required for XCUITest (Command Line Tools are insufficient)" >&2
    exit 1
}
[[ -d "$PROJECT" ]] || {
    echo "error: generated Xcode project missing: $PROJECT" >&2
    exit 1
}

# XCUIApplication(bundleIdentifier:) resolves through LaunchServices.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
"$LSREGISTER" -f "$APP"

xcodebuild test \
    -project "$PROJECT" \
    -scheme RapidUITests \
    -destination 'platform=macOS' \
    -resultBundlePath "$RESULT_BUNDLE" \
    CODE_SIGNING_ALLOWED=NO
