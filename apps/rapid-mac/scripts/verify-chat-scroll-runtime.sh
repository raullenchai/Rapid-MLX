#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CACHE_ROOT=$(mktemp -d /tmp/rapid-chat-scroll-check.XXXXXX)
CACHE="$CACHE_ROOT/modules"
BINARY="$CACHE_ROOT/chat-scroll-runtime-check"

trap 'rm -rf -- "$CACHE_ROOT"' EXIT

mkdir -p "$CACHE"
CLANG_MODULE_CACHE_PATH="$CACHE" \
SWIFT_MODULECACHE_PATH="$CACHE" \
swiftc -parse-as-library \
    "$ROOT/Sources/Rapid/UI/TranscriptScrollPositionProbe.swift" \
    "$ROOT/scripts/verify-chat-scroll-runtime.swift" \
    -o "$BINARY" \
    -framework AppKit \
    -framework SwiftUI

"$BINARY"
