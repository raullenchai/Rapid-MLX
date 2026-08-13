import AppKit
import Testing
@testable import Rapid

/// Gates from the #1843 maintainer triage that can be checked without a
/// window. The GUI golden flows cover the rest (AX baselines, visual parity),
/// and are not runnable headless.
@Suite("TextKit markdown parity")
@MainActor
struct TextKitMarkdownParityTests {

    private func items(_ source: String) -> [MarkdownItem] {
        TextKitMarkdownView.compile(source).items
    }

    @Test("Custom TextKit prose remains readable through accessibility")
    func prosePublishesAccessibilityValue() {
        let options = MarkdownOptions.assistantTranscript()
        let view = MarkdownTextBlockView(options: options)
        view.configure(
            blocks: [.init(runs: [InlineRun(text: "Hello from TextKit")], kind: .paragraph)],
            options: options,
            streaming: true,
            fadeState: TextFadeAnimationState(),
            fadeConfiguration: .off
        )
        #expect(view.isAccessibilityElement())
        #expect(view.accessibilityRole() == .staticText)
        #expect(view.accessibilityValue() as? String == "Hello from TextKit")
    }

    @Test("Custom TextKit prose resolves rendered links for click handling")
    @MainActor
    func customTextKitLinkHitTesting() throws {
        let result = MarkdownCompiler().compile("Read [Rapid](https://rapidmlx.ai) now")
        let blocks = result.items.compactMap { item -> MarkdownItem.TextBlock? in
            guard case .text(let block) = item else { return nil }
            return block
        }
        let renderer = MarkdownTextRenderer(options: .assistantTranscript())
        renderer.setBlocks(blocks)
        _ = renderer.measureHeight(width: 400)
        let linkedOffset = (renderer.accessibleText as NSString).range(of: "Rapid").location
        let rect = try #require(renderer.rect(forCharacterAt: linkedOffset))
        #expect(renderer.link(at: CGPoint(x: rect.midX, y: rect.midY))?.absoluteString == "https://rapidmlx.ai")
    }

    @Test("A long transcript does not release follow mode for a short new answer")
    func followModeUsesCurrentAnswerGrowth() {
        #expect(!TranscriptScrollPositionProbe.Coordinator.answerOutgrewViewport(
            documentHeight: 5_300,
            documentHeightAtStreamStart: 5_000,
            viewportHeight: 800
        ))
        #expect(TranscriptScrollPositionProbe.Coordinator.answerOutgrewViewport(
            documentHeight: 5_900,
            documentHeightAtStreamStart: 5_000,
            viewportHeight: 800
        ))
    }

    // MARK: - Structural parity with the MarkdownUI path

    @Test("Prose, code and tables compile to distinct blocks")
    func blocksAreSeparate() {
        let result = items("""
        Intro paragraph.

        ```swift
        let x = 1
        ```

        | model | size |
        |-------|------|
        | a     | 1 GB |
        """)
        #expect(result.contains { if case .text = $0 { return true } else { return false } })
        #expect(result.contains { if case .code = $0 { return true } else { return false } })
        #expect(result.contains { if case .table = $0 { return true } else { return false } })
    }

    /// `gui-golden-flows.sh:645` asserts no AX value contains a fence marker.
    /// If a code block ever compiled as prose, that flow would fail on a built
    /// app; this catches it at the compile step instead.
    @Test("Fenced code does not leak its markers into text")
    func noFenceMarkersInProse() {
        for item in items("```python\nprint(1)\n```") {
            if case let .text(block) = item {
                let joined = block.runs.map(\.text).joined()
                #expect(!joined.contains("```"), "fence marker leaked into prose")
            }
        }
    }

    /// `assert_no_literal_list_markers` (gui-golden-flows.sh:524) requires that
    /// no rendered node is a bare list marker.
    @Test("List items render as content, not as literal markers")
    func listMarkersAreNotLiteral() {
        let result = items("- first\n- second")
        let texts = result.compactMap { item -> String? in
            if case let .text(block) = item { return block.runs.map(\.text).joined() }
            return nil
        }
        #expect(texts.contains { $0.contains("first") })
        #expect(!texts.contains { $0.trimmingCharacters(in: .whitespaces) == "-" })
    }

    // MARK: - #1824 table accessibility

    @Test("A table still yields an accessibility model")
    func tableHasAccessibilityModel() throws {
        guard case let .table(block)? = items("""
        | model | size | speed |
        |-------|------|-------|
        | a     | 1 GB | 9 t/s |
        """).first(where: { if case .table = $0 { return true } else { return false } }) else {
            Issue.record("no table compiled")
            return
        }
        let model = try #require(block.accessibilityModel)
        #expect(model.headers == ["model", "size", "speed"])
        #expect(model.rows == [["a", "1 GB", "9 t/s"]])
    }

    /// The old parser rejected >8 columns because macOS 14's
    /// `TableColumnBuilder` has no dynamic-column primitive. Same limit here,
    /// or `AccessibleMarkdownTable` would silently drop columns.
    @Test("More than eight columns yields no accessibility model")
    func nineColumnsRejected() {
        let header = "| " + (1...9).map(String.init).joined(separator: " | ") + " |"
        let sep = "|" + String(repeating: "---|", count: 9)
        let row = "| " + (1...9).map { _ in "x" }.joined(separator: " | ") + " |"
        guard case let .table(block)? = items("\(header)\n\(sep)\n\(row)")
            .first(where: { if case .table = $0 { return true } else { return false } }) else {
            return  // not compiling as a table at all is also acceptable
        }
        #expect(block.accessibilityModel == nil)
    }

    @Test("Short rows are padded so the column count stays rectangular")
    func shortRowsPadded() throws {
        guard case let .table(block)? = items("""
        | a | b | c |
        |---|---|---|
        | 1 | 2 |
        """).first(where: { if case .table = $0 { return true } else { return false } }) else {
            Issue.record("no table compiled")
            return
        }
        let model = try #require(block.accessibilityModel)
        #expect(model.rows.first?.count == 3)
    }

    // MARK: - #131 math

    @Test("Display math becomes its own block")
    func displayMathSplit() {
        #expect(items("$$E = mc^2$$").contains {
            if case .math = $0 { return true } else { return false }
        })
    }

    /// `displayMathOnly` folds inline math back into prose so a table row or
    /// list item containing `$x$` is not torn in half. That behaviour predates
    /// this change and must survive it.
    @Test("Inline math stays inside its sentence")
    func inlineMathFolded() {
        let result = items("The value $x$ is small.")
        guard case let .text(block) = result.first else {
            Issue.record("expected prose, got \(String(describing: result.first))")
            return
        }
        let joined = block.runs.map(\.text).joined()
        #expect(joined.contains("The value"))
        #expect(joined.contains("is small"))
    }

    /// Rapid's segmenter skips fenced blocks — a shell variable is not a
    /// formula.
    @Test("Dollars inside a fence are not math")
    func dollarsInFenceAreNotMath() {
        let result = items("```sh\necho $PATH\n```")
        #expect(!result.contains { if case .math = $0 { return true } else { return false } })
    }

    /// The split must happen BEFORE markdown parsing.
    ///
    /// `$x_1$` handed to a CommonMark parser becomes `x`, emphasis, `1` — the
    /// underscore is markdown syntax and the subscript is destroyed before any
    /// math code sees it. Sabotaging the segmenter call did not fail any other
    /// test in this suite, because the segmenter is a no-op on markdown that
    /// contains no `$`. This is the one that notices.
    @Test("Underscores inside display math are not markdown emphasis")
    func underscoresInMathSurvive() {
        guard case let .math(block)? = items("$$x_1 + x_2 = y_3$$")
            .first(where: { if case .math = $0 { return true } else { return false } }) else {
            Issue.record("display math did not compile to a math block")
            return
        }
        #expect(block.latex.contains("x_1"), "subscript was eaten by the markdown parser")
        #expect(block.latex.contains("y_3"))
    }

    // MARK: - #304/#349 link safety

    /// The allowlist is enforced by `ChatLinkSafety.decide`, which the new
    /// view inherits by applying the same `.chatLinkSafetyFilter()` at the
    /// container level. This pins that the compiler does not somehow produce
    /// links that bypass it — every link run carries a URL the filter will see.
    @Test("Compiled links are ordinary URLs the safety filter can judge")
    func linksGoThroughTheFilter() {
        let result = items("[safe](https://example.com) and [bad](file:///etc/hosts)")
        let links = result.compactMap { item -> [URL] in
            if case let .text(block) = item { return block.runs.compactMap(\.link) }
            return []
        }.flatMap { $0 }

        #expect(links.count == 2, "both links should compile")
        for url in links {
            let decision = ChatLinkSafety.decide(url)
            if url.scheme == "file" {
                #expect(decision == .rejected, "file:// must be rejected")
            } else {
                #expect(decision == .allowed(url))
            }
        }
    }

    @Test("Auto-linked bare URLs are also subject to the allowlist")
    func autoLinkedURLsAreFiltered() {
        let result = items("visit https://example.com now")
        let links = result.compactMap { item -> [URL] in
            if case let .text(block) = item { return block.runs.compactMap(\.link) }
            return []
        }.flatMap { $0 }
        #expect(!links.isEmpty, "bare URL should auto-link")
        for url in links {
            #expect(ChatLinkSafety.decide(url) == .allowed(url))
        }
    }

    // MARK: - #546 Dynamic Type

    /// TextKit needs a resolved point size, so scaling happens once at the
    /// view boundary. This pins that the size actually reaches the options —
    /// if it stopped being threaded through, every message would render at the
    /// default size regardless of the system setting, silently.
    @Test("The caller's point size reaches the render options")
    func pointSizeIsThreaded() {
        var options = MarkdownOptions.assistantTranscript()
        options.textPointSize = 22
        #expect(options.textPointSize == 22)

        let renderer = MarkdownTextRenderer(options: options)
        renderer.setBlocks([
            MarkdownItem.TextBlock(runs: [InlineRun(text: "sized")], kind: .paragraph)
        ])
        let tall = renderer.measureHeight(width: 400)

        options.textPointSize = 11
        let small = MarkdownTextRenderer(options: options)
        small.setBlocks([
            MarkdownItem.TextBlock(runs: [InlineRun(text: "sized")], kind: .paragraph)
        ])
        #expect(tall > small.measureHeight(width: 400), "point size did not affect layout")
    }
}
