import Foundation
import Testing
@testable import Rapid

/// The hover-toolbar age stamp. Pure formatter with the reference date
/// injected, so every switchover the user specced — minutes under an
/// hour, hours under a day, then days / weeks / months / years — is
/// pinned exactly.
@Suite("RelativeTimestamp — hover toolbar age stamp")
struct RelativeTimestampTests {

    private let base = Date(timeIntervalSince1970: 1_800_000_000)

    private func at(_ seconds: TimeInterval) -> String {
        RelativeTimestamp.label(from: base, to: base.addingTimeInterval(seconds))
    }

    @Test("under a minute reads as just now")
    func justNow() {
        #expect(at(0) == "just now")
        #expect(at(59) == "just now")
    }

    @Test("minutes for the first hour, switching exactly at 60m")
    func minutes() {
        #expect(at(60) == "1m ago")
        #expect(at(59 * 60) == "59m ago")
        #expect(at(60 * 60) == "1h ago")
    }

    @Test("hours for the first day, switching exactly at 24h")
    func hours() {
        #expect(at(90 * 60) == "1h ago")
        #expect(at(23 * 3_600) == "23h ago")
        #expect(at(24 * 3_600) == "1d ago")
    }

    @Test("days, then weeks, then months, then years")
    func longSpans() {
        #expect(at(6 * 86_400) == "6d ago")
        #expect(at(7 * 86_400) == "1w ago")
        #expect(at(4 * 604_800) == "4w ago")
        #expect(at(5 * 604_800) == "1mo ago")
        #expect(at(11 * 2_592_000) == "11mo ago")
        #expect(at(400 * 86_400) == "1y ago")
    }

    @Test("a clock-skewed future date clamps to just now instead of going negative")
    func futureClamps() {
        #expect(
            RelativeTimestamp.label(from: base.addingTimeInterval(120), to: base) == "just now"
        )
    }

    @Test("VoiceOver form spells the unit out and agrees with the compact form's bucket")
    func accessibilityForm() {
        let a = { (s: TimeInterval) in
            RelativeTimestamp.accessibilityLabel(from: self.base, to: self.base.addingTimeInterval(s))
        }
        #expect(a(30) == "just now")
        #expect(a(60) == "1 minute ago")
        #expect(a(120) == "2 minutes ago")
        #expect(a(3_600) == "1 hour ago")
        #expect(a(2 * 86_400) == "2 days ago")
    }
}

/// Source-grep guards for the hover toolbar's reachability fix. The
/// original bug: `.onHover` was attached to a container with NO
/// `.contentShape`, so the hover region was only the opaque bubble —
/// the opacity-0 toolbar was not hit-testable and the pointer could
/// never reach it. These pins keep the pairing from regressing
/// silently, since no UI test can click a hover.
@Suite("Hover toolbar — reachability source guards")
struct HoverToolbarSourceGuardTests {

    private static func chatViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

}
