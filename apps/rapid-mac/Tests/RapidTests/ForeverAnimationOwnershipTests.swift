import Foundation
import Testing
@testable import Rapid

/// Source-level tripwire for the idle-CPU leak fixed on 2026-09-18.
///
/// A `repeatForever` animation attached through `.animation(_:value:)` and
/// later "switched off" by passing that modifier a different curve does not
/// stop — SwiftUI keeps the loop alive, the window commits a transaction every
/// frame, and the Desktop idled at 23 % of a core (M3 Ultra) to a full core
/// (M2 Pro with an accessibility pointer). The rule this pins: a repeating
/// animation is owned by a view whose lifetime IS the loop's lifetime —
/// ``BreathingLoop`` — or is started imperatively with `withAnimation` by a
/// view that leaves the hierarchy when it stops signalling.
@Suite("repeatForever animations are owned by a removable view")
struct ForeverAnimationOwnershipTests {
    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("Sources/Rapid")
    }

    private static func swiftFiles() throws -> [URL] {
        var out: [URL] = []
        let enumerator = FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "swift" { out.append(url) }
        }
        return out
    }

    /// Files that may spell `repeatForever`, and why.
    private static let allowed: [String: String] = [
        // The sanctioned owner: a view that exists only while breathing.
        "RapidMotion.swift": "BreathingLoop",
        // Imperative withAnimation inside views that leave the hierarchy
        // when they stop (jump-to-bottom hides at the bottom; onboarding
        // step is dismissed). Migrate to BreathingLoop if either grows a
        // resting state that stays on screen.
        "JumpToBottomButton.swift": "withAnimation on a view hidden when idle",
        "OnboardingDirectionD.swift": "withAnimation on a dismissed step",
    ]

    @Test("No new file spells repeatForever outside the allowlist")
    func repeatForeverIsConfinedToOwners() throws {
        var offenders: [String] = []
        for url in try Self.swiftFiles() {
            let text = try String(contentsOf: url, encoding: .utf8)
            let code = text.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")
            guard code.contains("repeatForever") else { continue }
            if Self.allowed[url.lastPathComponent] == nil {
                offenders.append("\(url.lastPathComponent): not an allowlisted owner")
            }
            // Inside an allowlisted file the loop must still be started
            // imperatively (`withAnimation`), never attached declaratively:
            // `.animation(.x.repeatForever(), value:)` is exactly the shape
            // that leaked, and swapping the curve for `.default` later does
            // not retire it.
            for line in code.split(separator: "\n") where line.contains("repeatForever") {
                if line.contains(".animation(") || !line.contains("withAnimation(") {
                    offenders.append("\(url.lastPathComponent): \(line.trimmingCharacters(in: .whitespaces))")
                }
            }
        }
        #expect(offenders.isEmpty, """
            repeatForever misuse: \(offenders). Wrap the animated content in
            BreathingLoop inside an `if` that is false at rest, started with
            withAnimation — a repeating animation attached via
            .animation(_:value:) keeps running after the value flips and pins
            the display cycle (see BreathingLoop).
            """)
    }

    @Test("BreathingLoop never attaches its loop through .animation(_:value:)")
    func breathingLoopUsesWithAnimation() throws {
        let url = Self.sourcesRoot.appendingPathComponent("UI/Modifiers/RapidMotion.swift")
        let text = try String(contentsOf: url, encoding: .utf8)
        guard let start = text.range(of: "struct BreathingLoop") else {
            Issue.record("BreathingLoop moved — update this test")
            return
        }
        let body = String(text[start.lowerBound...])
        let code = body.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") && !$0.trimmingCharacters(in: .whitespaces).hasPrefix("///") }
            .joined(separator: "\n")
        #expect(code.contains("withAnimation("))
        #expect(!code.contains(".animation("))
    }
}
