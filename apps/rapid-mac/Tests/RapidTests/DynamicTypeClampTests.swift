import Testing
import SwiftUI
@testable import Rapid

/// Audit P1 (ChatView — no dynamic-type testing; >extraLarge may
/// overflow compose / crush bubbles): the chat-surface clamp lives
/// in ``Sources/Rapid/UI/Modifiers/DynamicTypeClamp.swift``. These
/// tests pin the contract of ``chatDynamicTypeRange`` so a future
/// rename / wider-cap edit can't silently re-open the bug.
///
/// We can't introspect the SwiftUI view tree to check that ChatView
/// has actually applied the modifier (SwiftUI's ``Modifier`` types
/// aren't reflectable), so the wiring side is enforced by code-
/// review + the call-site `// Audit P1` comments. The constant
/// itself is fully unit-testable here.
@Suite("DynamicTypeClamp")
struct DynamicTypeClampTests {

    @Test("upper bound is xxxLarge — last non-accessibility size")
    func upper_bound_is_xxxLarge() {
        #expect(chatDynamicTypeRange.upperBound == .xxxLarge)
    }

    @Test("clamp admits the seven non-accessibility sizes")
    func clamp_admits_non_accessibility_sizes() {
        let allowed: [DynamicTypeSize] = [
            .xSmall, .small, .medium, .large, .xLarge, .xxLarge, .xxxLarge
        ]
        for size in allowed {
            #expect(
                chatDynamicTypeRange.contains(size),
                "\(size) should be inside the chat-surface range"
            )
        }
    }

    @Test("clamp excludes all five accessibility sizes (AX1–AX5)")
    func clamp_excludes_accessibility_sizes() {
        let blocked: [DynamicTypeSize] = [
            .accessibility1,
            .accessibility2,
            .accessibility3,
            .accessibility4,
            .accessibility5,
        ]
        for size in blocked {
            #expect(
                !chatDynamicTypeRange.contains(size),
                "\(size) should be outside the chat-surface range"
            )
        }
    }

    @Test("range lower bound is xSmall — system minimum")
    func range_lower_bound_is_xSmall() {
        // ClosedRange built from `...xxxLarge` starts at the type's
        // ``min``. DynamicTypeSize.allCases is ordered xSmall first,
        // so the lower bound must match that minimum — and never
        // accidentally float above it (e.g. someone editing the
        // range to `.large ... .xxxLarge` would break users on
        // smaller defaults).
        #expect(chatDynamicTypeRange.lowerBound == .xSmall)
    }

    @Test(".rapidChatDynamicTypeClamp() builds without crashing")
    @MainActor
    func modifier_applies_without_crash() {
        // SwiftUI environment modifiers can't be introspected, but we
        // can at least prove the helper is syntactically valid and
        // type-checks against an arbitrary view. A regression that
        // removed the extension on `View` would fail this build —
        // which is exactly the breakage we want surfaced loudly.
        // `View` is MainActor-isolated under Swift 6, so the wrapper
        // call has to run on the main actor too.
        let _ = Text("test").rapidChatDynamicTypeClamp()
    }

    /// Walk parent directories until we find `Package.swift`. Throws
    /// if we hit the filesystem root without finding one.
    static func findPackageRoot() throws -> URL {
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        for _ in 0..<10 {
            if fm.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
            if dir.path == "/" { break }
        }
        struct PackageRootNotFound: Error {}
        throw PackageRootNotFound()
    }
}
