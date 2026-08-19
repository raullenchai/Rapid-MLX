import Foundation
import Testing
@testable import Rapid

/// A fenced code block must not change shape while it streams.
///
/// The parser used to be shown a fence that was one keystroke old, which
/// produced three separate flickers from one cause. Compiling a growing prefix
/// of `"Here is code:\n\n```swift\nlet x = 1\nprint(x)\n```\n\nDone."` before
/// the fix:
///
///     n=16   T(13) T(1)         a lone backtick as its own text block
///     n=20   T(13) C(0|sw)      that slot turns from text into code
///     n=24   T(13) C(0|swift)   the language changes, re-running highlighting
///     n=44   T(13) C(21|swift)
///     n=48   T(13) C(19|swift)  content SHRINKS — the closing backticks had
///                               been rendered as code until they closed it
///
/// This was a regression, not a gap. The pre-TextKit path
/// (``ChatStreamInlineFormatter``) documented "never style-then-unstyle
/// flicker" and carried a cross-chunk fence state machine; #1906 replaced the
/// streaming path and left that file without callers.
@Suite("Streaming fence stability")
@MainActor
struct StreamingFenceStabilityTests {

    private let sample = "Here is code:\n\n```swift\nlet x = 1\nprint(x)\n```\n\nDone."

    private func shapes(of full: String, step: Int = 1) -> [[MarkdownItem]] {
        let compiler = MarkdownCompiler()
        return stride(from: 1, through: full.count, by: step).map {
            compiler.compile(String(full.prefix($0)), revision: $0, isComplete: false).items
        }
    }

    private func codeBlocks(_ items: [MarkdownItem]) -> [MarkdownItem.CodeBlock] {
        items.compactMap { if case .code(let b) = $0 { return b } else { return nil } }
    }

    private func proseLength(_ items: [MarkdownItem]) -> Int {
        items.reduce(0) { total, item in
            if case .text(let b) = item { return total + b.runs.map(\.text).joined().count }
            return total
        }
    }

    /// The one a reader actually sees: a code block that grows and then
    /// briefly gets shorter reads as the app losing characters.
    @Test("Code content never shrinks while streaming")
    func codeGrowsMonotonically() {
        var lastByIndex: [Int: Int] = [:]
        for items in shapes(of: sample) {
            for (index, block) in codeBlocks(items).enumerated() {
                if let previous = lastByIndex[index] {
                    #expect(
                        block.code.count >= previous,
                        "code block \(index) shrank \(previous) → \(block.code.count) mid-stream"
                    )
                }
                lastByIndex[index] = block.code.count
            }
        }
    }

    /// A half-typed info string re-runs syntax highlighting for a language the
    /// author never wrote.
    @Test("A code block arrives already knowing its language")
    func languageIsNeverPartial() {
        var seen: Set<String> = []
        for items in shapes(of: sample) {
            for block in codeBlocks(items) where block.language != nil {
                seen.insert(block.language!)
            }
        }
        #expect(seen == ["swift"], "saw partial languages: \(seen.sorted())")
    }

    /// The block sequence used to gain a one-character text block for the
    /// first backtick, which `MarkdownBlockStack`'s index-keyed `ForEach`
    /// turns into a full view swap at that slot.
    @Test("No stray backtick block appears before the fence opens")
    func noLoneBacktickBlock() {
        for items in shapes(of: sample) {
            for case .text(let block) in items {
                let text = block.runs.map(\.text).joined()
                #expect(
                    text.trimmingCharacters(in: .whitespacesAndNewlines) != "`",
                    "a lone backtick was rendered as its own block"
                )
            }
        }
    }

    /// Holding the marker back must not lose it. The settled row re-compiles
    /// with `isComplete` true, and that render has to carry everything.
    @Test("Nothing is held back once the message settles")
    func settledRenderKeepsEverything() {
        let settled = MarkdownCompiler().compile(sample).items
        #expect(codeBlocks(settled).first?.code.contains("print(x)") == true)
        #expect(codeBlocks(settled).first?.language == "swift")
        #expect(proseLength(settled) >= "Here is code:".count + "Done.".count)
    }

    /// The trim is for fences only. An inline span being typed keeps its
    /// characters — hiding those would make the composer feel laggy.
    @Test("An inline code span being typed is left alone")
    func inlineSpanIsUntouched() {
        for source in ["Use `foo", "Call `bar to", "a `b` and `c"] {
            #expect(
                MarkdownCompiler.withoutFormingFence(source) == source,
                "\(source) was trimmed as if it were a fence"
            )
        }
    }

    /// Exactly what does get held back, spelled out so the rule is reviewable
    /// without running a stream.
    @Test("Only a forming fence marker is held back")
    func trimsOnlyFenceMarkers() {
        #expect(MarkdownCompiler.withoutFormingFence("text\n`") == "text\n")
        #expect(MarkdownCompiler.withoutFormingFence("text\n``") == "text\n")
        #expect(MarkdownCompiler.withoutFormingFence("text\n```") == "text\n")
        #expect(MarkdownCompiler.withoutFormingFence("text\n```swi") == "text\n")
        // Finished lines are the parser's business again.
        #expect(MarkdownCompiler.withoutFormingFence("text\n```swift\n") == "text\n```swift\n")
        #expect(MarkdownCompiler.withoutFormingFence("plain text") == "plain text")
        #expect(MarkdownCompiler.withoutFormingFence("") == "")
    }
}
