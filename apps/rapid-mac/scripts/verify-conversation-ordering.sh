#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CACHE="$ROOT/.build/conversation-ordering-module-cache"
BINARY="$ROOT/.build/conversation-ordering-check"

mkdir -p "$CACHE"
CLANG_MODULE_CACHE_PATH="$CACHE" \
SWIFT_MODULECACHE_PATH="$CACHE" \
swiftc -parse-as-library \
    "$ROOT/Sources/Rapid/Chat/ConversationOrdering.swift" \
    "$ROOT/scripts/verify-conversation-ordering.swift" \
    -o "$BINARY"

"$BINARY"
