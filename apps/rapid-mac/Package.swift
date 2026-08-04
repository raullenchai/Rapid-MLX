// swift-tools-version:6.0
import PackageDescription

// Minimal menu-bar app ("Rapid-MLX" v1.0). Source-level target is
// macOS 14 (the MLX floor). Two SPM dependencies remain — block-level
// markdown rendering and LaTeX — both on the chat streaming path.
let package = Package(
    name: "Rapid",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Block-level markdown rendering for assistant turns. Apple's
        // ``AttributedString(markdown:)`` only does inline formatting and
        // silently flattens headings, lists, fenced code, and tables.
        // ``MarkdownUI`` renders each block as a real SwiftUI view, à la
        // ChatGPT Desktop. Pinned to the maintenance-line 2.4 series.
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
        // Issue #131: LaTeX rendering for math/STEM model responses.
        // ``MarkdownUI`` ships no math engine, so ``$``/``\frac``/``\sqrt``
        // would render as visible tokens. ``SwiftMath`` is the macOS-
        // friendly Swift port of iosMath (pure-Swift, no WKWebView/JS),
        // embedded via ``NSViewRepresentable`` and stitched into the
        // render path by ``LaTeXSegmenter``.
        .package(url: "https://github.com/mgriebling/SwiftMath", from: "1.7.0")
    ],
    targets: [
        // Issue #24: signal-safe arena + handler in pure C. Swift
        // static-property reads compile to ``_swift_beginAccess``
        // runtime calls (Swift 6 exclusivity tracking) — async-
        // signal-unsafe. The C target exposes the arena as a plain
        // extern struct so the signal handler's reads lower to
        // direct memory loads with no runtime re-entry.
        .target(
            name: "RapidCrashHandler",
            path: "Sources/RapidCrashHandler",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "Rapid",
            dependencies: [
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "SwiftMath", package: "SwiftMath"),
                "RapidCrashHandler"
            ],
            path: "Sources/Rapid",
            // The release assembler compiles the app icon catalog with
            // actool. SwiftPM does not consume it, so exclude it from target
            // discovery while leaving it available to scripts/build.sh.
            exclude: ["Resources/Assets.xcassets"],
            // Brand cheetah PNGs + the per-alias benchmark scores + the
            // localizable strings table. Loaded at runtime via
            // ``Bundle.main`` (flat files in the production .app).
            resources: [
                .process("Resources/cheetah.png"),
                .process("Resources/cheetah-sm.png"),
                .process("Resources/Localizable.xcstrings"),
                .process("Resources/benchmark-scores.json")
            ]
        )
        // NOTE: the Tests/RapidTests target was removed from the manifest.
        // The strip deleted the subsystems (Sessions, Presets, Tools, MCP,
        // QuickAsk, Bootstrapper, attachments, feedback) that the vast
        // majority of the test suite exercised, so the target no longer
        // compiles. Per the migration plan, the test target is excluded
        // rather than rewritten; a fresh minimal suite lands with v1.0.
        // (Command-line `swift test` also can't resolve the `import Testing`
        // module in this toolchain, so re-enabling even a minimal subset is
        // a separate toolchain/CI task.) The RAM-tier recommendation tests
        // added here are kept in sync as the v1.0-suite seed; the contract
        // itself is verified during development by a standalone `swift`
        // script (no XCTest/Testing dependency).
    ]
)
