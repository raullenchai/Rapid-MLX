// swift-tools-version:6.0
import PackageDescription

// Minimal menu-bar app ("Rapid-MLX" v1.0). Source-level target is
// macOS 14 (the MLX floor). Two SPM dependencies remain — block-level
// markdown rendering and LaTeX — both on the chat streaming path.
let package = Package(
    name: "Rapid",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Sparkle owns signed desktop updates: background checks/downloads,
        // install-on-quit, authorization when /Applications is not writable,
        // and atomic replacement. Keep the in-tree updater as a migration
        // fallback for builds that do not carry a Sparkle public key.
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.9.5"),
        // Block-level markdown rendering for assistant turns. Apple's
        // ``AttributedString(markdown:)`` only does inline formatting and
        // silently flattens headings, lists, fenced code, and tables.
        // ``MarkdownUI`` renders each block as a real SwiftUI view, à la
        // ChatGPT Desktop. Pinned to the maintenance-line 2.4 series.
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
        // Parsing only (swift-cmark underneath, GFM tables + strikethrough).
        // The TextKit 2 render layer needs a parsed block list, which
        // MarkdownUI's string-only entry point cannot provide — see #1843.
        .package(url: "https://github.com/swiftlang/swift-markdown", from: "0.6.0"),
        // Issue #131: LaTeX rendering for math/STEM model responses.
        // ``MarkdownUI`` ships no math engine, so ``$``/``\frac``/``\sqrt``
        // would render as visible tokens. ``SwiftMath`` is the macOS-
        // friendly Swift port of iosMath (pure-Swift, no WKWebView/JS),
        // embedded via ``NSViewRepresentable`` and stitched into the
        // render path by ``LaTeXSegmenter``.
        // SwiftMath is vendored below because its upstream `Bundle.module`
        // lookups cannot resolve resources from a manually assembled `.app`.
        // The tiny local patch first resolves the signed app resource bundle,
        // while retaining `Bundle.module` for `swift run` and unit tests.
    ],
    targets: [
        .target(
            name: "SwiftMath",
            path: "Vendor/SwiftMath/Sources/SwiftMath",
            resources: [.copy("mathFonts.bundle")],
            // Match upstream's Swift tools 5.7 compilation mode. Rapid's
            // Swift 6 isolation rules apply at the UI boundary instead of
            // rewriting a vendored renderer in the resource-fix PR.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
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
                .product(name: "Markdown", package: "swift-markdown"),
                .product(name: "Sparkle", package: "Sparkle"),
                "SwiftMath",
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
        ),
        .testTarget(
            name: "RapidTests",
            dependencies: ["Rapid"],
            path: "Tests/RapidTests"
        ),
        // Phase 1 model-readiness regression tests: the pure
        // ``ModelReadiness`` state machine that gates Send and drives the
        // readiness banner shared by Chat and Launch.
        .testTarget(
            name: "RapidUXTests",
            dependencies: ["Rapid"],
            path: "Tests/RapidUXTests"
        ),
        .testTarget(
            name: "SwiftMathVendorTests",
            dependencies: ["SwiftMath"],
            path: "Vendor/SwiftMath/Tests"
        )
        // NOTE (2026-08-05): the RapidTests target is BACK in the manifest,
        // above. It had been excluded on the reasoning that the strip
        // deleted the subsystems most of the suite exercised, so the target
        // no longer compiled and a fresh suite would land with v1.0.
        //
        // Measuring that claim rather than inheriting it changed the answer.
        // Two things were conflated:
        //
        //   1. The FIRST compile error was a missing test-only dependency
        //      (ViewInspector, used by 9 files). "No such module" aborts the
        //      build before type-checking, so it masked everything behind it
        //      and made the damage look total. Adding it back was a dead end:
        //      7 of the 9 referenced stripped subsystems anyway, and the
        //      surviving 2 deadlocked EVERY @MainActor test on a headless CI
        //      runner (1,413 started, 0 finished, 45 minutes). SwiftUI view
        //      introspection wants a GUI session; the runner has none. Those
        //      2 files are deleted and the dependency is gone — do not
        //      reintroduce it without a headless-verified alternative.
        //   2. Behind it, 137 of 254 files genuinely did not compile against
        //      the stripped Sources. Those are deleted. The remaining 117
        //      compile and run: 1,485 tests, green.
        //
        // The cost of the exclusion was not hypothetical. The suite pinned
        // `BundledModel.bundledAlias`, `QuickstartCoordinator.defaultChoice`,
        // `storageKey`, and `AutoStartDecision.SkipReason`'s case set — every
        // value the retired-starter swap touched. Dormant, it caught none of
        // them, and a starter model that degenerates on an ordinary chat
        // question shipped for months.
        //
        // `import Testing` resolves fine on Swift 6.1; the toolchain note
        // that said otherwise predates it.
        //
        // Deleting a test is a real decision. If one fails, read it first —
        // several of these were correct about code that had moved, and two
        // were reporting live defects: a $HOME-bypassing Application Support
        // lookup in ConversationStore, and an 8-15 GB tier pick with no
        // benchmark row. Both are fixed here rather than deleted.
    ]
)
