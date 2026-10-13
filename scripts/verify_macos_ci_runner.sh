#!/bin/bash
set -euo pipefail

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required macOS CI tool is unavailable: $command_name" >&2
    exit 1
  fi
}

require_major_at_least() {
  local component="$1"
  local minimum="$2"
  local version="$3"
  local major="${version%%.*}"

  if [[ ! "$major" =~ ^[0-9]+$ ]] || (( major < minimum )); then
    echo "$component $version does not satisfy the required major version >= $minimum" >&2
    exit 1
  fi
}

for command_name in codesign python3 swift swiftc xcodebuild xcrun; do
  require_command "$command_name"
done

architecture="$(uname -m)"
if [[ "$architecture" != "arm64" ]]; then
  echo "Desktop CI requires Apple Silicon arm64; runner reported $architecture" >&2
  exit 1
fi

xcode_version="$(xcodebuild -version | awk 'NR == 1 { print $2 }')"
swift_version="$(swiftc --version | awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "version") { print $(i + 1); exit } }')"
sdk_version="$(xcrun --sdk macosx --show-sdk-version)"

require_major_at_least Xcode 16 "$xcode_version"
require_major_at_least Swift 6 "$swift_version"
require_major_at_least macOS_SDK 15 "$sdk_version"

echo "Desktop runner contract PASS: arch=$architecture xcode=$xcode_version swift=$swift_version macos_sdk=$sdk_version"
