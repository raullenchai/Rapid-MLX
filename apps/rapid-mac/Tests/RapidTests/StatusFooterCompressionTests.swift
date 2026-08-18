import Foundation
import Testing
@testable import Rapid

/// The status footer must shed readouts, never controls.
///
/// This is the rule `EnvironmentValues.settingsContentIsCompact` already
/// states for Settings — "use this to DROP something optional …, never to hide
/// a control — an unreachable control at a supported window size is the defect
/// this exists to prevent" — applied to the one other row in the app that runs
/// out of horizontal room.
///
/// Before this, a narrow window squeezed all six chips until they read
/// "CPU…", "GPU 9…", "13.2 G…", and the version pill wrapped into a five-line
/// block. ViewInspector is not in this target (#1492), so the wiring is pinned
/// by a source guard in the same shape as
/// ``AccessibilityIdentifierInventoryTests``.
@Suite("Status footer compression")
struct StatusFooterCompressionTests {

    private func contentViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The droppable group, sliced out by brace balance.
    private func readoutGroup(_ source: String) -> String {
        guard let anchor = source.range(of: "private var footerReadouts: some View {") else {
            Issue.record("ContentView no longer declares footerReadouts")
            return ""
        }
        var depth = 0
        var index = source.index(before: anchor.upperBound)
        while index < source.endIndex {
            if source[index] == "{" { depth += 1 }
            if source[index] == "}" {
                depth -= 1
                if depth == 0 { return String(source[anchor.upperBound...index]) }
            }
            index = source.index(after: index)
        }
        return ""
    }

    /// A control inside the group would simply vanish at narrow widths with
    /// no other way to reach it. Readouts are the only thing allowed to go.
    @Test("Only readouts live in the droppable group")
    func droppableGroupHoldsNoControls() throws {
        let group = readoutGroup(try contentViewSource())
        #expect(!group.isEmpty)
        for control in ["SettingsGearButton", "DesktopVersionPill", "Button"] {
            #expect(
                !group.contains(control),
                """
                \(control) is inside footerReadouts, which ViewThatFits drops \
                at narrow widths. A control that disappears at a supported \
                window size is unreachable — move it out of the group.
                """
            )
        }
    }

    /// The ambient system probes are the first thing a reader can spare, and
    /// the server state is the last: the other numbers mean nothing without
    /// knowing whether a model is running.
    @Test("The widest arrangement is the only one carrying the system probes")
    func systemProbesAreShedFirst() throws {
        let group = readoutGroup(try contentViewSource())
        for probe in ["CPUPill", "GPUPill", "MemoryPill"] {
            #expect(
                group.components(separatedBy: probe).count - 1 == 1,
                "\(probe) should appear in exactly one arrangement — the widest"
            )
        }
        // Server state survives into arrangements the probes do not.
        let statusCount = group.components(separatedBy: "ServerStatusPill").count - 1
        let cpuCount = group.components(separatedBy: "CPUPill").count - 1
        #expect(statusCount > cpuCount)
    }

    /// A view that wraps reports that it fits at any width, so an unbounded
    /// version label would keep ViewThatFits from ever choosing a shorter
    /// arrangement — the pill's own bug and the row's, in one line of code.
    @Test("The version pill is pinned to one line")
    func versionPillCannotWrap() throws {
        let source = try contentViewSource()
        guard let anchor = source.range(of: "struct DesktopVersionPill") else {
            Issue.record("DesktopVersionPill was renamed")
            return
        }
        let body = String(source[anchor.lowerBound...])
        #expect(body.contains(".lineLimit(1)"))
        #expect(body.contains(".fixedSize(horizontal: true, vertical: false)"))
    }
}
