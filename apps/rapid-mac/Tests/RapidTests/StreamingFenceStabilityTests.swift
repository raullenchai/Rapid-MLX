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

    /// The floor for this whole suite.
    ///
    /// Every other test here is one-sided: it forbids a shape from appearing.
    /// A `withoutFormingFence` that returned `""` satisfies all of them — no
    /// code block can shrink and no stray backtick can be rendered if nothing
    /// is rendered at all. (Measured: with the body replaced by `return ""`,
    /// the rest of the suite passed.) This test is the other side, so the
    /// suite as a whole distinguishes "correctly trimmed" from "deleted".
    ///
    /// The sample's last line is `"Done."`, not a fence, so the final
    /// streaming render must already carry the entire message.
    @Test("The streaming render still carries the whole message")
    func streamingRenderCarriesEverything() {
        let streaming = MarkdownCompiler().compile(sample, isComplete: false).items
        #expect(codeBlocks(streaming).first?.code.contains("print(x)") == true)
        #expect(codeBlocks(streaming).first?.language == "swift")
        #expect(proseLength(streaming) >= "Here is code:".count + "Done.".count)

        // And the settled path is byte-identical to the pre-fix behaviour,
        // because `compile` short-circuits the trim on `isComplete`.
        let settled = MarkdownCompiler().compile(sample).items
        #expect(codeBlocks(settled).first?.code == codeBlocks(streaming).first?.code)
    }

    /// A source that ends mid-fence is where the two paths must disagree:
    /// streaming holds the marker back, settled renders it as the parser sees
    /// it. Without this, nothing checks that `isComplete` still gates the trim.
    @Test("The settled compile is never trimmed")
    func settledCompileIsNeverTrimmed() {
        let midFence = "Here is code:\n\n```swi"
        let streaming = MarkdownCompiler().compile(midFence, isComplete: false).items
        let settled = MarkdownCompiler().compile(midFence, isComplete: true).items
        #expect(codeBlocks(streaming).isEmpty, "the forming fence was not held back")
        #expect(codeBlocks(settled).first?.language == "swi", "the settled row was trimmed")
    }

    /// The trim is for fences only. An inline span being typed keeps its
    /// characters — hiding those would make the composer feel laggy.
    @Test("An inline code span being typed is left alone")
    func inlineSpanIsUntouched() {
        for source in [
            "Use `foo",
            "Call `bar to",
            "a `b` and `c",
            // The three above all begin with a non-backtick, so they bail at
            // the `ticks.isEmpty` guard and never reach the one that tells a
            // span from a fence. Measured: deleting that guard left all six
            // tests in this suite passing. These four are the ones that reach
            // it — a line that IS a span in the making.
            "text\n`foo",
            "text\n``x",
            "\n`variable",
            "text\n`a b",
        ] {
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
        // A backtick fence's info string may contain spaces — only backticks
        // are forbidden. Both of these are fences and flicker without the trim.
        #expect(MarkdownCompiler.withoutFormingFence("text\n```swift {highlight}") == "text\n")
        #expect(MarkdownCompiler.withoutFormingFence("text\n```py file=a.py") == "text\n")
        // Backticks in the tail make it a paragraph, not a fence.
        #expect(MarkdownCompiler.withoutFormingFence("text\n``` a ```") == "text\n``` a ```")
        // Swift reads CRLF as a single Character, so a newline search that
        // looks for "\n" alone never matches it and the fix silently opts out.
        #expect(MarkdownCompiler.withoutFormingFence("text\r\n```swi") == "text\r\n")
        // Finished lines are the parser's business again.
        #expect(MarkdownCompiler.withoutFormingFence("text\n```swift\n") == "text\n```swift\n")
        #expect(MarkdownCompiler.withoutFormingFence("plain text") == "plain text")
        #expect(MarkdownCompiler.withoutFormingFence("") == "")
    }
}
