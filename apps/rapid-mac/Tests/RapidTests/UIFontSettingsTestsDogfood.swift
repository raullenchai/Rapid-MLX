import AppKit
import Observation
import SwiftUI
import Testing
@testable import Rapid

@Suite("UIFontSettings GUI dogfood regressions", .serialized)
@MainActor
struct UIFontSettingsTestsDogfood {
    private func withScale(_ scale: Double, _ check: () throws -> Void) rethrows {
        let previous = UIFontSettings.shared.scale
        defer { UIFontSettings.shared.setScale(previous) }
        UIFontSettings.shared.setScale(scale)
        try check()
    }

    @Test("Table fonts retain styling and measure larger at 1.5×", arguments: [false, true])
    func tableFonts(isHeader: Bool) {
        let options = MarkdownOptions.assistantTranscript()
        var plainWidth: CGFloat = 0
        for scale in [1.0, 1.5] {
            withScale(scale) {
                var code = InlineRun(text: "print(2)")
                code.isInlineCode = true
                let text = MarkdownTableView.styled(
                    [InlineRun(text: "Item Count "), code], isHeader: isHeader, options: options
                )
                let weight: Font.Weight = isHeader ? .semibold : .regular
                let size = UIFontSettings.shared.pointSize(options.tablePointSize)
                #expect(size == (scale == 1 ? options.tablePointSize : 21))
                let runs = Array(text.runs)
                #expect(runs[0].font == Font.system(size: size, weight: weight, design: .default))
                #expect(runs[1].font == Font.system(size: size, weight: weight, design: .monospaced))
                let host = NSHostingView(rootView: Text(text).fixedSize())
                if scale == 1 { plainWidth = host.fittingSize.width }
                else { #expect(host.fittingSize.width > plainWidth * 1.4) }
            }
        }
    }

    @Test("Table styling observes scale for live layout invalidation")
    func tableObservesScale() async {
        await confirmation("Scale changes invalidate table styling") { changed in
            withScale(1) {
                withObservationTracking {
                    _ = MarkdownTableView.styled(
                        [InlineRun(text: "Item")], isHeader: false, options: .assistantTranscript()
                    )
                } onChange: { changed() }
                UIFontSettings.shared.setScale(1.5)
            }
        }
    }

    @Test("Sidebar budget wraps without losing the limit or units", arguments: [1.0, 1.5])
    func sidebarBudget(scale: Double) {
        withScale(scale) {
            let summary = SidebarView.memorySummary(
                usedBytes: 3_532_610_601, limitBytes: 219_902_325_555
            )
            let row = NSHostingView(rootView:
                SidebarView.residencySummaryRow(summary).frame(width: 200))
            let short = NSHostingView(rootView:
                SidebarView.residencySummaryRow("3 GB").frame(width: 200))
            #expect(row.fittingSize.height >= short.fittingSize.height)
            if scale == 1.5 { #expect(row.fittingSize.height > short.fittingSize.height) }
            #expect(row.fittingSize.width == 200)
            let label = NSHostingView(rootView: Text("Resident").font(RapidFont.caption).fixedSize())
            let icon = NSHostingView(rootView:
                Image(systemName: "memorychip").font(.system(size: 11, weight: .semibold)))
            let available = 200 - label.fittingSize.width - icon.fittingSize.width
                - 3 * RapidTheme.Space.xs - 4
            let full = NSHostingView(rootView:
                Text(summary).font(.rapid(size: 10, weight: .medium, design: .monospaced))
                    .fixedSize(horizontal: false, vertical: true).frame(width: available))
            #expect(row.fittingSize.height >= full.fittingSize.height)
        }
    }

    @Test("Onboarding chip identity wraps in the full rail", arguments: [1.0, 1.5])
    func onboardingChip(scale: Double) {
        withScale(scale) {
            // Wide rail: 300pt minus 24pt padding on each side.
            let row = NSHostingView(rootView:
                OnboardingRailFacts.row("Memory", "256 GB · Apple M3 Ultra").frame(width: 252))
            let short = NSHostingView(rootView:
                OnboardingRailFacts.row("Memory", "256 GB").frame(width: 252))
            #expect(row.fittingSize.height >= short.fittingSize.height)
            if scale == 1.5 { #expect(row.fittingSize.height > short.fittingSize.height) }
            #expect(row.fittingSize.width == 252)
            let label = NSHostingView(rootView: Text("Memory").scaledSystemFont(12).fixedSize())
            let available = 252 - label.fittingSize.width - 16
            let full = NSHostingView(rootView:
                Text("256 GB · Apple M3 Ultra").scaledSystemFont(12, design: .monospaced)
                    .fixedSize(horizontal: false, vertical: true).frame(width: available))
            #expect(row.fittingSize.height >= full.fittingSize.height)
        }
    }

    @Test("Both What's New buttons render a 12pt scaled font through Rapid styles")
    func bannerButtonFonts() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/Rapid/UI/WhatsNewBanner.swift"))
        // Native small bordered styles render a fixed AppKit label size and
        // ignore .font, so both buttons must use the Rapid styles, which apply
        // the supplied font and scale the control height.
        for (identifier, style) in [
            ("WhatsNew.OpenNotes", "RapidPrimaryButtonStyle("),
            ("WhatsNew.Dismiss", "RapidSecondaryButtonStyle("),
        ] {
            let end = try #require(source.range(of: ".accessibilityIdentifier(\"\(identifier)\")"))
            let start = try #require(source[..<end.lowerBound].range(of: ".buttonStyle(", options: .backwards))
            let chain = source[start.lowerBound..<end.lowerBound]
            #expect(chain.contains(style))
            #expect(chain.contains("font: .rapid(size: 12)"))
            #expect(!chain.contains(".controlSize(.small)"))
        }
        for scale in [1.0, 1.5] {
            withScale(scale) { #expect(UIFontSettings.shared.pointSize(12) == CGFloat(12 * scale)) }
        }
    }
}
