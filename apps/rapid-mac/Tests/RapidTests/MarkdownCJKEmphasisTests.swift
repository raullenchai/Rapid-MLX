import Foundation
import Markdown
import Testing
@testable import Rapid

@Suite("CJK-friendly markdown emphasis")
@MainActor
struct MarkdownCJKEmphasisTests {
    private let compiler = MarkdownCompiler(postProcessors: [])

    private func runs(_ source: String, isComplete: Bool = true) -> [InlineRun] {
        compiler.compile(source, isComplete: isComplete).items.flatMap {
            if case .text(let block) = $0 { return block.runs }
            return []
        }
    }

    private func strings(_ item: MarkdownItem) -> [String] {
        func inline(_ runs: [InlineRun]) -> [String] {
            runs.flatMap { [$0.text] + [$0.link?.absoluteString, $0.math].compactMap { $0 } }
        }
        switch item {
        case .text(let block): return inline(block.runs)
        case .code(let block): return [block.code] + [block.language].compactMap { $0 }
        case .table(let block): return (block.header + block.rows.flatMap { $0 }).flatMap(inline)
        case .images(let block): return block.altTexts + block.urls.map(\.absoluteString)
        case .math(let block): return [block.latex]
        }
    }

    @Test(arguments: [
        ("由**瑞利散射（Rayleigh scattering）**现象", "瑞利散射（Rayleigh scattering）", true),
        ("**注意：**请", "注意：", true),
        ("これは**「重要」**です", "「重要」", true),
        ("*（注）*を参照", "（注）", false),
        ("中文**（括号开头）**中文", "（括号开头）", true),
        ("**한국어**입니다", "한국어", true)
    ])
    func cjkEmphasis(_ example: (String, String, Bool)) {
        for complete in [true, false] {
            let output = runs(example.0, isComplete: complete)
            #expect(output.contains { $0.text == example.1 && (example.2 ? $0.isStrong : $0.isEmphasis) })
            #expect(output.map(\.text).joined() == example.0.replacingOccurrences(of: "*", with: ""))
            #expect(!output.contains { $0.text.contains(CJKEmphasisFlanking.sentinel) })
        }
    }

    @Test(arguments: ["**bold**", "*it*", "snake_case_name", "a * b * c", "2**3", "**bold**.", "(**x**)"])
    func englishMatchesCmark(_ source: String) {
        // Explicit snapshots of cmark's existing runs (including literal 2**3).
        let expected: [InlineRun]
        switch source {
        case "**bold**": expected = [.init(text: "bold", isStrong: true)]
        case "*it*": expected = [.init(text: "it", isEmphasis: true)]
        case "**bold**.": expected = [.init(text: "bold", isStrong: true), .init(text: ".")]
        case "(**x**)": expected = [.init(text: "("), .init(text: "x", isStrong: true), .init(text: ")")]
        default: expected = [.init(text: source)]
        }
        #expect(CJKEmphasisFlanking.prepare(source) == source)
        #expect(runs(source) == expected)
        // Exercise the scanner as well as its no-CJK fast path.
        #expect(runs(source + "\n\n中文").dropLast() == expected[...])
    }

    @Test(arguments: ["由**（x）**现", "2**3"])
    func codeIsVerbatim(_ code: String) {
        let inline = runs("中文 `\(code)`")
        #expect(inline.last == InlineRun(text: code, isInlineCode: true))
        for fence in ["```", "~~~", "````"] {
            let source = "中文\n\n\(fence)python\n\(code)\n\(fence)"
            let items = compiler.compile(source).items
            #expect(items.last == .code(.init(code: code + "\n", language: "python")))
            #expect(items.flatMap(strings).allSatisfy { !$0.contains(CJKEmphasisFlanking.sentinel) })
        }
    }

    @Test func mathAndUnderscores() {
        #expect(runs("中文_强调_中文") == [.init(text: "中文_强调_中文")])
        #expect(runs("中文 $x_1$").map(\.text).joined() == "中文 $x_1$")
        #expect(runs("中文 $x_1$").last?.math == "x_1")
        #expect(CJKEmphasisFlanking.prepare("中文无标记") == "中文无标记")
        #expect(CJKEmphasisFlanking.prepare("\n\n中文\n") == "\n\n中文\n")
    }

    @Test func flankingRules() {
        typealias F = CJKEmphasisFlanking
        #expect(F.flanking(before: "）", after: "现", delimiter: "*") == .init(canOpen: true, canClose: true))
        #expect(F.flanking(before: "）", after: "现", delimiter: "*", cjkFriendly: false) == .init(canOpen: true, canClose: false))
        #expect(F.flanking(before: "中", after: "强", delimiter: "_") == .init(canOpen: false, canClose: false))
        #expect(F.flanking(before: nil, after: "中", delimiter: "_") == .init(canOpen: true, canClose: false))
        #expect(F.flanking(before: "中", after: nil, delimiter: "_") == .init(canOpen: false, canClose: true))
        #expect(F.flanking(before: "中", after: "!", delimiter: "*") == .init(canOpen: true, canClose: true))
        #expect(F.flanking(before: "!", after: "中", delimiter: "*") == .init(canOpen: true, canClose: true))
        for scalar: Unicode.Scalar in [" ", "\t", "\n", "\u{c}", "\r", "\u{a0}"] {
            #expect(F.isWhitespace(scalar))
        }
        #expect(!F.isWhitespace("\u{b}"))
        #expect(!F.isPunctuation(nil))
        for scalar: Unicode.Scalar in ["!", ":", "[", "{", "©", "—", "。", "😀"] {
            #expect(F.isPunctuation(scalar))
        }
        #expect(!F.isPunctuation("a"))
        #expect(F.isCJK("中"))
        #expect(F.isCJK("\u{20000}"))
        #expect(!F.isCJK("😀"))
        #expect(!F.isCJK("\u{2e9a}"))
    }

    @Test func parserAndDelimiterBoundaries() {
        let unchanged = [
            "中文 \\_强调_中文", "中文 `**（x）**`",
            "中文 `` `**（x）**` ``", "中文 `unmatched", "中文 `a``b`",
            "```中文\n由**（x）**现\n~~~\n``\n```   ",
            "~~~中文\n由**（x）**现\n~~~~\t", "中文\r\n```\r\n由**（x）**现\r\n```\r\n"
        ]
        for source in unchanged {
            #expect(CJKEmphasisFlanking.prepare(source) == source)
        }
        #expect(CJKEmphasisFlanking.prepare("中文 \\**（x）**现").hasPrefix("中文 \\**"))
        #expect(CJKEmphasisFlanking.prepare("* 中文\\") == "* 中文\\")
        #expect(CJKEmphasisFlanking.prepare("\n\n中文**bold**\n") == "\n\n中文**bold**\n")
        #expect(CJKEmphasisFlanking.prepare("~~~\n~~~bad\n由**（x）**现\n~~~") == "~~~\n~~~bad\n由**（x）**现\n~~~")
        #expect(CJKEmphasisFlanking.prepare("中文 `unclosed **（x）**现").contains(CJKEmphasisFlanking.sentinel))
        #expect(CJKEmphasisFlanking.prepare("````\n中文**（x）**现\n```\n~~~~\n````") == "````\n中文**（x）**现\n```\n~~~~\n````")
        #expect(CJKEmphasisFlanking.prepare("```bad`info\n中文**（x）**现").contains(CJKEmphasisFlanking.sentinel))
        #expect(CJKEmphasisFlanking.prepare("    中文**（x）**现") == "    中文**（x）**现")
        // Exercise all neighbour pairs and delimiter types.
        let neighbours = ["中", "）", "a", "!", " ", "😀", "\t", ""]
        for before in neighbours {
            for after in neighbours {
                for delimiter in ["*", "_", "**", "___"] {
                    let source = "中文 " + before + delimiter + after
                    #expect(CJKEmphasisFlanking.strip(CJKEmphasisFlanking.prepare(source)) == source)
                }
            }
        }
    }

    @Test(arguments: ["- ```\n  x\n  ```", "> ```\n> x\n> ```",
                      "- ```\n  由**（x）**现\n  ```", "> ```\n> 由**（x）**现\n> ```"])
    func nestedFences(_ code: String) {
        let source = code + "\n\n**注意：**请"
        let prepared = CJKEmphasisFlanking.prepare(source)
        #expect(prepared.hasPrefix(code + "\n\n"))
        #expect(runs(source).contains { $0.text == "注意：" && $0.isStrong })
        let output = compiler.compile(source).items.flatMap(strings)
        #expect(output.contains(code.contains("由") ? "由**（x）**现\n" : "x\n"))
        #expect(output.allSatisfy { !$0.contains(CJKEmphasisFlanking.sentinel) })
    }

    @Test(arguments: [
        "中文\n\n    由**（x）**现\n", "中文 `first\n由**（x）**现\nlast`",
        "中文 ``first\r\n由**（x）**现\r\nlast``",
        "<div>\n由**（x）**现\n</div>", "中文 <span title=\"由**（x）**现\">尾</span>",
        "| 中文 |\n| --- |\n| `由**（x）**现` |",
        "- > ```\n  > 由**（x）**现\n  > ```", "中**<br>"
    ])
    func opaqueNodesAreUnchanged(_ source: String) {
        #expect(CJKEmphasisFlanking.prepare(source) == source)
        let prepared = CJKEmphasisFlanking.prepare(source + "\n\n**注意：**请")
        #expect(prepared.hasPrefix(source + "\n\n"))
    }

    @Test(arguments: ["\r", "\r\n", "\n"])
    func lineEndings(_ newline: String) {
        let source = "引言" + newline + "由**瑞利散射（R）**现象" + newline + "**注意：**请" + newline
        #expect(runs(source).contains { $0.text == "瑞利散射（R）" && $0.isStrong })
        #expect(runs(source).contains { $0.text == "注意：" && $0.isStrong })
        #expect(CJKEmphasisFlanking.strip(CJKEmphasisFlanking.prepare(source)) == source)
        let boundaries = "中文**" + newline + "**中文"
        #expect(CJKEmphasisFlanking.prepare(boundaries) == boundaries)
    }

    @Test func mixedLanguageWithoutBridgesIsUnchanged() {
        let source = "中文 **bold** text"
        #expect(CJKEmphasisFlanking.prepare(source) == source)
    }

    @Test func missingOpaqueRangeUsesEnclosingRange() {
        let source = "由**（x）**现"
        let fullRange = source.startIndex..<source.endIndex
        let document = Document(Paragraph(InlineCode(source), InlineHTML("<b>")), CodeBlock(source), HTMLBlock("<hr>"))
        #expect(CJKEmphasisFlanking.opaqueRanges(in: document, source: source, enclosing: fullRange)
                == Array(repeating: fullRange, count: 4))
    }

    @Test func bridgeSelection() {
        typealias F = CJKEmphasisFlanking
        let both = F.bridgePlacement(before: "!", after: "!", delimiter: "*",
                                     desired: .init(canOpen: true, canClose: true))
        #expect(both?.before == true && both?.after == true)
        #expect(F.bridgePlacement(before: "!", after: "!", delimiter: "*",
                                  desired: .init(canOpen: false, canClose: false)) == nil)
    }

    @Test func allOutputSurfacesAreClean() {
        let s = CJKEmphasisFlanking.sentinel
        let sources = [
            "# 中文**（标题）**中", "- 中文**（列表）**中", "> 中文**（引用）**中",
            "| 中文**（表头）**中 |\n| --- |\n| 中文**（单元）**中 |",
            "[中文**（链接）**中](https://example.com/中文**（x）**中)",
            "![中文**（图片）**中](https://example.com/中文**（x）**中)",
            "![一](https://example.com/1)\n\n![二](https://example.com/2)",
            "中文\(s) [链接](https://example.com/\(s)x) `代码\(s)`",
            "```语\(s)言\n代码\(s)\n```", "$$x\(s)$$", "---"
        ]
        for source in sources {
            let output = compiler.compile(source).items.flatMap(strings)
            #expect(output.allSatisfy { !$0.contains(s) && !$0.contains("%F3%B0%80%82") })
        }
        let destination = "https://example.com/中文**（x）**中"
        #expect(runs("[链接](\(destination))").first?.link == URL(string: destination))
        #expect(compiler.compile("![中文**（图片）**中](\(destination))").items == [
            .images(.init(urls: [URL(string: destination)!], altTexts: ["中文（图片）中"]))
        ])
        let dirty = MarkdownItem.text(.init(runs: [.init(text: "a"), .init(text: s), .init(text: "b")], kind: .paragraph))
        #expect(compiler.strippingCJKSentinels(dirty) == .text(.init(runs: [.init(text: "ab")], kind: .paragraph)))
        #expect(compiler.strippingCJKSentinels(.math(.init(latex: "x"))) == .math(.init(latex: "x")))
        #expect(compiler.strippingCJKSentinels(.code(.init(code: s, language: s))) == .code(.init(code: "", language: "")))
    }
}
