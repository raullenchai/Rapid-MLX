#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CACHE="$ROOT/.build/codex-module-cache"
BINARY="$ROOT/.build/chat-scroll-runtime-check"

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
