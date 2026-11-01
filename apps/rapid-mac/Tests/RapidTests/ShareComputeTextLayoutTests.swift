import AppKit
import SwiftUI
import Testing
@testable import Rapid

@Suite("ShareCompute and CommunityBenchmark enlarged text", .serialized)
@MainActor
struct ShareComputeTextLayoutTests {
    @Test("Ledger headers and long values fit without shrinking", arguments: [1.0, 1.5])
    func ledgerColumns(scale: Double) throws {
        let name = "ShareComputeTextLayoutTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let fonts = UIFontSettings(defaults: defaults)
        fonts.setScale(scale)
        let headers: [(ShareComputeLedgerLayout.Column, String)] = [
            (.window, "WINDOW"), (.model, "NODE · MODEL"), (.requests, "REQUESTS"),
            (.tokens, "TOKENS IN/OUT"), (.credit, "CREDIT"), (.status, "STATUS"),
        ]
        for (column, header) in headers {
            let width = NSAttributedString(string: header, attributes: [
                .font: NSFont.systemFont(ofSize: fonts.pointSize(8), weight: .bold),
                .kern: 0.5,
            ]).size().width
            #expect(width <= column.width(fonts: fonts), "\(scale)× header \(header)")
        }
        let values: [(ShareComputeLedgerLayout.Column, String, CGFloat, NSFont.Weight)] = [
            (.window, "Sep 30 · 11:59 PM", 11, .regular),
            (.model, "Qwen 3.8 · 27B · TensorFold", 12, .medium),
            (.model, "Model unavailable", 12, .medium),
            (.requests, "9999999", 12, .regular),
            (.tokens, "999.9k / 999.9k", 11, .regular),
            (.tokens, "999.99M / 999.99M", 11, .regular),
            (.credit, "$0.000000003", 12, .semibold),
            (.status, "CREDITED", 9, .bold),
        ]
        for (column, value, base, weight) in values {
            let font = NSFont.monospacedDigitSystemFont(ofSize: fonts.pointSize(base), weight: weight)
            let width = NSAttributedString(string: value, attributes: [
                .font: font, .kern: column == .status ? 0.4 : 0,
            ]).size().width
                + (column == .status ? 20 : 0)
            #expect(width <= column.width(fonts: fonts), "\(scale)× value \(value): \(width)")
        }
    }

    @Test("Every step state uses the same scaled circle", arguments: [1.0, 1.5])
    func stepCircles(scale: Double) {
        let fonts = UIFontSettings.shared
        let previous = fonts.scale
        defer { fonts.setScale(previous) }
        fonts.setScale(scale)
        let rail = ShareComputeStepRail(rows: [])
        for status: ShareComputePreparationStatus in [
            .waiting, .inProgress, .complete, .alreadyRegistered, .failed,
        ] {
            let row = ShareComputePreparationRow(step: .pauseCurrentModel, status: status)
            let host = NSHostingView(rootView: rail.circle(for: row))
            #expect(host.fittingSize == CGSize(width: 26 * scale, height: 26 * scale))
        }
    }

    @Test("Benchmark provenance wraps to show the full date", arguments: [1.0, 1.5])
    func benchmarkDate(scale: Double) {
        let fonts = UIFontSettings.shared
        let previous = fonts.scale
        defer { fonts.setScale(previous) }
        fonts.setScale(scale)
        let text = "Mac Studio · Sep 30, 2026 at 11:59 PM"
        let wrapped = NSHostingView(rootView:
            CommunityBenchmarkResultView.provenanceText(text).frame(width: 160))
        let reference = NSHostingView(rootView:
            Text(text).font(RapidFont.metric)
                .fixedSize(horizontal: false, vertical: true).frame(width: 160))
        let singleLine = NSHostingView(rootView:
            CommunityBenchmarkResultView.provenanceText("Sep 30"))
        #expect(wrapped.fittingSize.height > singleLine.fittingSize.height)
        #expect(wrapped.fittingSize.height == reference.fittingSize.height)
    }
}
