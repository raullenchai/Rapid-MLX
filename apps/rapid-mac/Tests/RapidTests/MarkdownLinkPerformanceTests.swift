import AppKit
import Testing
@testable import Rapid

/// Link lookup must scale with the number of LINKS, not the number of
/// characters.
///
/// The shape this pins: `resetCursorRects` used to walk every character and
/// ask `link(at:)` whether it was a link, and that call was itself a linear
/// scan over the same characters with an `ensureLayout` inside. Opening a
/// 6 000-character answer put 86% of the main thread in those two methods
/// (sampled on a real transcript) — the chat surface was unusable before a
/// single token streamed.
@Suite("Markdown link hit-testing cost")
@MainActor
struct MarkdownLinkPerformanceTests {

    private func renderer(_ blocks: [MarkdownItem.TextBlock]) -> MarkdownTextRenderer {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let renderer = MarkdownTextRenderer(options: options)
        renderer.setBlocks(blocks)
        _ = renderer.measureHeight(width: 600)
        return renderer
    }

    @Test("A long link-free answer reports no link rects")
    func longProseHasNoLinkRects() {
        let paragraph = String(repeating: "这是一段没有任何链接的长文本。", count: 200)
        let renderer = renderer([.init(runs: [InlineRun(text: paragraph)], kind: .paragraph)])

        #expect(renderer.proseLength > 2_000, "fixture must be long enough to matter")
        #expect(renderer.linkRects().isEmpty)
    }

    @Test("Link rects are per run, not per character")
    func linkRectsAreMerged() {
        // One link inside ordinary prose. A per-character implementation would
        // return one rect per character of the link text.
        var linked = InlineRun(text: "Rapid MLX")
        linked.link = URL(string: "https://example.com")
        let renderer = renderer([
            .init(
                runs: [InlineRun(text: "See "), linked, InlineRun(text: " for details.")],
                kind: .paragraph
            )
        ])

        let rects = renderer.linkRects()
        #expect(!rects.isEmpty, "the link must still be tracked")
        #expect(
            rects.count <= 2,
            "\(rects.count) rects for a 9-character link — merging by line regressed"
        )
    }

    @Test("A wrapped link exposes one clickable rect per rendered line")
    func wrappedLinkSegmentsRemainClickable() throws {
        let url = try #require(URL(string: "https://example.com/wrapped"))
        var linked = InlineRun(text: String(repeating: "wrapped link ", count: 20))
        linked.link = url
        let renderer = renderer([.init(runs: [linked], kind: .paragraph)])

        let rects = renderer.linkRects()
        #expect(rects.count > 1, "fixture must wrap across lines")
        for rect in rects {
            #expect(renderer.link(at: CGPoint(x: rect.midX, y: rect.midY)) == url)
        }
    }

    @Test("Adjacent links keep their own hit targets")
    func adjacentLinksDoNotMerge() throws {
        let firstURL = try #require(URL(string: "https://one.example"))
        let secondURL = try #require(URL(string: "https://two.example"))
        var first = InlineRun(text: "one")
        first.link = firstURL
        var second = InlineRun(text: "two")
        second.link = secondURL
        let renderer = renderer([.init(runs: [first, second], kind: .paragraph)])

        let rects = renderer.linkRects()
        #expect(rects.count == 2)
        #expect(renderer.link(at: CGPoint(x: rects[0].midX, y: rects[0].midY)) == firstURL)
        #expect(renderer.link(at: CGPoint(x: rects[1].midX, y: rects[1].midY)) == secondURL)
    }

    @Test("Hit testing a link-free document is cheap")
    func hitTestingSkipsUnlinkedRuns() {
        let paragraph = String(repeating: "纯文本内容，没有链接。", count: 300)
        let renderer = renderer([.init(runs: [InlineRun(text: paragraph)], kind: .paragraph)])

        let started = Date()
        for _ in 0..<50 {
            _ = renderer.link(at: CGPoint(x: 100, y: 100))
        }
        let elapsed = Date().timeIntervalSince(started)

        // Generous by design: this is a complexity guard, not a benchmark. The
        // O(n²) shape took multiple seconds for a single pass at this length.
        #expect(
            elapsed < 1.0,
            "50 hit tests took \(String(format: "%.2f", elapsed))s — link lookup is scanning characters again"
        )
    }
}
