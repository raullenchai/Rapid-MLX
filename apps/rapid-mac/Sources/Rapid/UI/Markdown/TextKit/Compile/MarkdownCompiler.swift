import Foundation
import Markdown

/// Compiles markdown source into renderable blocks.
///
/// Parsing is outsourced to Apple's `swift-markdown` (swift-cmark underneath,
/// with GFM tables and strikethrough). Writing a markdown parser would be the
/// boring 60% of this work and would not make the result any better; the part
/// that has to be ours is the renderer, because the fade animator needs
/// `NSTextRange` addressing that no SwiftUI text view exposes.
///
/// `nonisolated` throughout: compilation is pure and can run off the main
/// actor, which matters because it happens on a debounce timer during
/// streaming.
struct MarkdownCompiler: Sendable {

    /// A conservative streaming split: every top-level node before the last
    /// one is structurally closed, while the last node remains mutable until a
    /// following sibling proves its boundary.
    struct TopLevelStreamingSplit: Equatable, Sendable {
        let stablePrefix: String
        let mutableTail: String
    }

    /// Passes run over the compiled blocks, in order.
    ///
    /// Auto-linking by default because a bare URL rendering as dead text is a
    /// bug in a chat transcript, not a stylistic choice. Pass an empty array
    /// for a strict-CommonMark compile.
    public var postProcessors: [MarkdownPostProcessor]

    /// Parse options shared by the streaming split and the block compiler.
    ///
    /// `.disableSmartOpts` matters: swift-markdown turns on cmark's "smart"
    /// typography by default, which rewrites `"` to `“ ”`, `'` to `‘ ’`,
    /// `--` to en dashes and `...` to an ellipsis in PROSE. Chat with a
    /// model is not prose typesetting — a bare `{"city": "Tokyo"}` the model
    /// returned outside a code fence rendered as `{ “city”: “Tokyo” }`, so
    /// copying the visible answer produced invalid JSON (0.14.3 dogfood,
    /// 2026-09-18). The characters the model emitted are the characters the
    /// user gets; a model that wants typographic quotes emits them itself.
    nonisolated static let parseOptions: ParseOptions = [.parseBlockDirectives, .disableSmartOpts]

    public init(postProcessors: [MarkdownPostProcessor] = [AutoLinkPostProcessor()]) {
        self.postProcessors = postProcessors
    }

    /// Split a growing Markdown fragment into a stable prefix and mutable tail.
    ///
    /// `swift-markdown` owns the block grammar and source ranges. Keeping the
    /// final top-level node mutable is deliberately conservative: an unfinished
    /// paragraph may still become emphasis, a link, or inline code; a list or
    /// fence may still grow. Once another top-level sibling appears, nodes
    /// before it can no longer be reinterpreted by later input.
    func topLevelStreamingSplit(_ source: String) -> TopLevelStreamingSplit? {
        guard !source.isEmpty else { return nil }
        let document = Document(parsing: source, options: Self.parseOptions)
        let children = Array(document.children)
        guard children.count > 1,
              let tailLocation = children.last?.range?.lowerBound,
              let tailStart = Self.index(at: tailLocation, in: source),
              tailStart > source.startIndex else {
            return nil
        }
        let stablePrefix = source[..<tailStart]
        guard Self.hasBlankLineBoundary(in: stablePrefix) else {
            return nil
        }

        // Reparse only the candidate prefix so reference links count as
        // resolved only when their definition is actually inside the chunk we
        // are about to freeze. If one unresolved reference remains, commit the
        // safe blocks before its top-level owner instead of pinning the whole
        // growing answer behind it.
        let stableDocument = Document(
            parsing: String(stablePrefix),
            options: [.parseBlockDirectives]
        )
        if let definitionOffset = Self.firstReferenceDefinitionUTF8Offset(
            in: stableDocument,
            source: String(stablePrefix)
        ),
           let definitionUTF8 = source.utf8.index(
            source.utf8.startIndex,
            offsetBy: definitionOffset,
            limitedBy: source.utf8.endIndex
           ),
           let definitionStart = definitionUTF8.samePosition(in: source) {
            let safePrefix = source[..<definitionStart]
            let safeSource = String(safePrefix)
            let safeDocument = Document(
                parsing: safeSource,
                options: [.parseBlockDirectives]
            )
            if let blockedLocation = Self.firstPotentialReferenceBlock(
                in: safeDocument,
                source: safeSource
            ),
               let blockedStart = Self.index(at: blockedLocation, in: source) {
                let prefixBeforeReference = source[..<blockedStart]
                guard blockedStart > source.startIndex,
                      Self.hasBlankLineBoundary(in: prefixBeforeReference) else {
                    return nil
                }
                return TopLevelStreamingSplit(
                    stablePrefix: String(prefixBeforeReference),
                    mutableTail: String(source[blockedStart...])
                )
            }
            guard definitionStart > source.startIndex,
                  Self.hasBlankLineBoundary(in: safePrefix) else {
                return nil
            }
            return TopLevelStreamingSplit(
                stablePrefix: String(safePrefix),
                mutableTail: String(source[definitionStart...])
            )
        }
        if let blockedLocation = Self.firstPotentialReferenceBlock(
            in: stableDocument,
            source: String(stablePrefix)
        ),
           let blockedStart = Self.index(at: blockedLocation, in: source) {
            let safePrefix = source[..<blockedStart]
            guard blockedStart > source.startIndex,
                  Self.hasBlankLineBoundary(in: safePrefix) else {
                return nil
            }
            return TopLevelStreamingSplit(
                stablePrefix: String(safePrefix),
                mutableTail: String(source[blockedStart...])
            )
        }

        return TopLevelStreamingSplit(
            stablePrefix: String(stablePrefix),
            mutableTail: String(source[tailStart...])
        )
    }

    /// Keep a definition and everything after it in one compiler document.
    /// A later independently-compiled segment may reference it, while the
    /// parser intentionally emits definitions as no visible AST node. Code
    /// and HTML ranges are excluded so a literal `[name]: value` does not
    /// disable incremental splitting for the rest of a code-heavy answer.
    private static func firstReferenceDefinitionUTF8Offset(
        in document: Document,
        source: String
    ) -> Int? {
        var opaqueRanges: [Range<String.Index>] = []
        collectReferenceOpaqueRanges(in: document, source: source, into: &opaqueRanges)

        var searchStart = source.startIndex
        while searchStart < source.endIndex,
              let match = source.range(
                // Labels may contain escaped closing brackets and one or
                // more soft line endings. This deliberately follows the
                // parser's broad label surface; opaque AST ranges still keep
                // definition-shaped code and HTML out of the result.
                of: #"(?m)^[ ]{0,3}\[(?:\\.|[^\]])+\]:"#,
                options: .regularExpression,
                range: searchStart..<source.endIndex
            ) {
            if !opaqueRanges.contains(where: { $0.overlaps(match) }),
               let utf8Index = match.lowerBound.samePosition(in: source.utf8) {
                return source.utf8.distance(
                    from: source.utf8.startIndex,
                    to: utf8Index
                )
            }
            searchStart = match.upperBound
        }
        return nil
    }

    /// Convert swift-markdown's one-based UTF-8 line/column location into a
    /// native String index without assuming ASCII input or LF-only newlines.
    private static func index(at location: SourceLocation, in source: String) -> String.Index? {
        guard location.line >= 1, location.column >= 1 else { return nil }

        let utf8 = source.utf8
        var line = 1
        var lineStart = utf8.startIndex
        while line < location.line {
            guard lineStart < utf8.endIndex else { return nil }
            var cursor = lineStart
            while cursor < utf8.endIndex, utf8[cursor] != 0x0A, utf8[cursor] != 0x0D {
                cursor = utf8.index(after: cursor)
            }
            guard cursor < utf8.endIndex else { return nil }

            // SourceLocation counts CRLF as one line ending. Consume the pair
            // explicitly instead of relying on String's grapheme clustering.
            if utf8[cursor] == 0x0D {
                cursor = utf8.index(after: cursor)
                if cursor < utf8.endIndex, utf8[cursor] == 0x0A {
                    cursor = utf8.index(after: cursor)
                }
            } else {
                cursor = utf8.index(after: cursor)
            }
            lineStart = cursor
            line += 1
        }

        var lineEnd = lineStart
        while lineEnd < utf8.endIndex, utf8[lineEnd] != 0x0A, utf8[lineEnd] != 0x0D {
            lineEnd = utf8.index(after: lineEnd)
        }
        guard let utf8Target = utf8.index(
                lineStart,
                offsetBy: location.column - 1,
                limitedBy: lineEnd
              ) else {
            return nil
        }
        return utf8Target.samePosition(in: source)
    }

    /// A new AST sibling is not by itself proof of a stable boundary. While a
    /// GFM table row is still being typed, cmark temporarily reports an empty
    /// table followed by a paragraph, then folds that paragraph into the table
    /// once the row is complete. A blank line is the conservative CommonMark
    /// boundary that prevents that cross-node reinterpretation.
    private static func hasBlankLineBoundary(in prefix: Substring) -> Bool {
        var newlineCount = 0
        for character in prefix.reversed() {
            if character.isNewline {
                newlineCount += 1
                if newlineCount >= 2 { return true }
            } else if character == " " || character == "\t" {
                continue
            } else {
                return false
            }
        }
        return false
    }

    /// Return the first top-level block that still contains unresolved
    /// reference-like syntax. swift-markdown turns a resolved reference into a
    /// `Link`; code and HTML are opaque, so brackets in those nodes cannot hold
    /// later blocks mutable.
    private static func firstPotentialReferenceBlock(
        in document: Document,
        source: String
    ) -> SourceLocation? {
        for child in document.children
        where containsPotentialReferenceLink(in: child, source: source) {
            if let location = child.range?.lowerBound { return location }
        }
        return nil
    }

    private static func containsPotentialReferenceLink(
        in markup: Markup,
        source: String
    ) -> Bool {
        guard let scanRange = sourceRange(of: markup, in: source) else { return false }
        var opaqueRanges: [Range<String.Index>] = []
        collectReferenceOpaqueRanges(in: markup, source: source, into: &opaqueRanges)

        var searchStart = scanRange.lowerBound
        while searchStart < scanRange.upperBound,
              let match = source.range(
                // CommonMark reference labels may contain a soft line
                // ending. The scan is already bounded to one parsed block,
                // so allowing newlines here cannot consume later blocks.
                of: #"(?<!\\)\[[^\]]+\](?!\()"#,
                options: .regularExpression,
                range: searchStart..<scanRange.upperBound
              ) {
            if !opaqueRanges.contains(where: { $0.overlaps(match) }) { return true }
            searchStart = match.upperBound
        }
        return false
    }

    private static func collectReferenceOpaqueRanges(
        in markup: Markup,
        source: String,
        into ranges: inout [Range<String.Index>]
    ) {
        switch markup {
        case is CodeBlock, is InlineCode, is Link, is Image, is HTMLBlock, is InlineHTML:
            if let range = sourceRange(of: markup, in: source) { ranges.append(range) }
        default:
            for child in markup.children {
                collectReferenceOpaqueRanges(in: child, source: source, into: &ranges)
            }
        }
    }

    private static func sourceRange(
        of markup: Markup,
        in source: String
    ) -> Range<String.Index>? {
        guard let range = markup.range,
              let lowerBound = index(at: range.lowerBound, in: source),
              let upperBound = index(at: range.upperBound, in: source),
              lowerBound <= upperBound else {
            return nil
        }
        return lowerBound..<upperBound
    }

    /// Drop a trailing fence marker that is still being typed.
    ///
    /// Three separate flickers, all from the same cause — the parser is shown
    /// a fence that is one keystroke old. Measured by compiling a growing
    /// prefix of "Here is code:\n\n```swift\nlet x = 1\nprint(x)\n```\n\nDone.":
    ///
    ///     n=16   T(13) T(1)          a lone backtick renders as its own text block
    ///     n=20   T(13) C(0|sw)       index 1 turns from text into code — SwiftUI
    ///                                swaps the whole representable at that slot
    ///     n=24   T(13) C(0|swift)    the language changes, re-running highlighting
    ///     n=44   T(13) C(21|swift)
    ///     n=48   T(13) C(19|swift)   the closing backticks were being displayed
    ///                                as code content until they closed the fence
    ///
    /// Holding the marker back until its line is finished removes all three:
    /// the code card appears once, already knowing its language, and no stray
    /// backticks are ever shown inside it.
    ///
    /// Only the LAST line is considered, and only while streaming. Nothing
    /// stays hidden because the settled row is rendered by a different view:
    /// ``ChatView`` swaps `StreamingTextKitMarkdownView` for
    /// `TextKitMarkdownView(content:)` the moment the message leaves
    /// `.streaming`, and that one compiles the raw `message.content` with
    /// ``isComplete`` defaulting to true. (``StreamingMarkdownStore/finish()``
    /// is *not* the guarantor — it flushes and then throws the result away.)
    /// Copy and selection read `message.content` directly and never see this
    /// at all.
    ///
    /// Deliberately narrow. A line like ``\`foo`` — an inline code span being
    /// typed — is left alone: only a run of three or more backticks (a real
    /// fence, with or without its info string) or a line that is nothing but
    /// one or two backticks (a fence in the making) qualifies.
    ///
    /// Backtick fences only. A `~~~` fence, and a fence inside a blockquote,
    /// still flicker — both are valid CommonMark and both are out of scope
    /// here, because a tilde info string may itself contain backticks and
    /// would need its own validation rather than a shared one.
    static func withoutFormingFence(_ source: String) -> String {
        // `lastIndex(where: \.isNewline)` rather than `lastIndex(of: "\n")`:
        // Swift reads CRLF as one `Character`, which the latter never matches,
        // and a CRLF stream would silently opt out of the whole fix.
        guard let lineStart = source.lastIndex(where: \.isNewline).map(source.index(after:))
                ?? (source.isEmpty ? nil : source.startIndex) else { return source }
        let lastLine = source[lineStart...]
        let trimmed = lastLine.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return source }

        let ticks = trimmed.prefix { $0 == "`" }
        guard !ticks.isEmpty else { return source }
        let rest = trimmed.dropFirst(ticks.count)
        // A backtick fence's info string may not contain a backtick — that is
        // what separates ```` ``` a ``` ```` (a paragraph) from a fence. It
        // may contain spaces, though, so ```` ```swift {highlight} ```` and
        // ```` ```py file=a.py ```` are fences and must be held back too.
        guard !rest.contains("`") else { return source }
        guard ticks.count >= 3 || rest.isEmpty else { return source }

        return String(source[..<lineStart])
    }

    /// Compile `source`.
    ///
    /// `isComplete` reaches the post-processors: mid-stream they should leave
    /// half-arrived constructs alone rather than commit to a reading they will
    /// have to undo on the next flush.
    public func compile(
        _ source: String, revision: Int = 0, isComplete: Bool = true
    ) -> MarkdownResult {
        let source = isComplete ? source : Self.withoutFormingFence(source)
        var items: [MarkdownItem] = []

        // Split math out BEFORE parsing. `$x_1$` handed to a markdown parser
        // becomes `x`, emphasis, `1` — the underscore is markdown syntax and
        // the subscript is gone before any math code sees it. The segmenter
        // already skips fenced blocks, indented code and inline spans, so a
        // `$` shown inside backticks stays text.
        //
        // Only DISPLAY math (`$$…$$`) becomes its own block. Inline `$…$` is
        // stitched back into the surrounding prose and parsed with it: the
        // segments around it are sentence fragments, and compiling each one
        // separately would emit a paragraph per fragment, breaking one
        // sentence into three stacked blocks.
        //
        // Re-wrapping restores the delimiters the segmenter stripped, so the
        // markdown parser sees the original text. That leaves inline math
        // rendering as literal `$x$` for now — correct-but-unstyled, rather
        // than a shattered sentence. Rendering it properly needs the view
        // attachment path (ChatGPT's `_viewAttachments`), which is a larger
        // change than this one.
        var pending = ""
        var inlineMath: [String] = []
        for segment in LaTeXSegmenter.segment(Self.withoutStraySentinels(source)) {
            switch segment {
            case let .math(latex, displayMode) where displayMode:
                appendMarkdown(pending, depth: 0, into: &items)
                pending = ""
                items.append(.math(.init(latex: latex)))
            case let .math(latex, _):
                // A sentinel, not the source spelling. Handing `$x_1$` back to
                // the markdown parser is what the segmenter ran first to
                // avoid: the underscore is emphasis syntax, and the subscript
                // is gone before any math code sees it. The sentinel carries
                // no markdown-significant character, so it survives parsing as
                // one contiguous piece of a single run and can be swapped back
                // afterwards — inside **bold**, inside a list item, inside a
                // table cell, wherever the sentence happened to put it.
                pending += Self.mathSentinel(inlineMath.count)
                inlineMath.append(latex)
            case let .markdown(body):
                pending += body
            }
        }
        appendMarkdown(pending, depth: 0, into: &items)

        if !inlineMath.isEmpty {
            items = items.map { Self.restoringInlineMath($0, from: inlineMath) }
        }

        return MarkdownResult(items: items, revision: revision)
            .postProcessed(
                with: postProcessors,
                context: MarkdownPostProcessContext(isComplete: isComplete)
            )
    }

    private func appendMarkdown(
        _ source: String, depth: Int, into items: inout [MarkdownItem]
    ) {
        guard !source.isEmpty else { return }
        let document = Document(parsing: source, options: Self.parseOptions)
        for child in document.children {
            appendBlocks(from: child, depth: depth, into: &items)
        }
    }

    // MARK: - Block walk

    private func appendBlocks(from markup: Markup, depth: Int, into items: inout [MarkdownItem]) {
        switch markup {
        case let paragraph as Paragraph:
            // A paragraph carrying nothing but images is an image block, not
            // prose. Without this an `![alt](url)` fell through to the inline
            // walk, which kept only the alt text — the picture never appeared,
            // and `MarkdownImagesView` sat in the codebase with no caller.
            if let images = imagesBlock(from: paragraph) {
                appendImages(images, into: &items)
            } else {
                items.append(.text(.init(runs: inlineRuns(of: paragraph), kind: .paragraph, depth: depth)))
            }

        case let heading as Heading:
            items.append(.text(.init(
                runs: inlineRuns(of: heading),
                kind: .heading(level: heading.level),
                depth: depth
            )))

        case let code as CodeBlock:
            items.append(.code(.init(code: code.code, language: code.language)))

        case let list as UnorderedList:
            for item in list.listItems {
                appendListItem(item, kind: .unorderedListItem, index: nil, depth: depth, into: &items)
            }

        case let list as OrderedList:
            // GFM lists can start at an arbitrary number.
            var index = Int(list.startIndex)
            for item in list.listItems {
                appendListItem(item, kind: .orderedListItem, index: index, depth: depth, into: &items)
                index += 1
            }

        case let quote as BlockQuote:
            for child in quote.children {
                // Render quoted paragraphs as quote-kind text so the bar and
                // indent apply, rather than nesting a whole sub-document.
                if let paragraph = child as? Paragraph {
                    items.append(.text(.init(
                        runs: inlineRuns(of: paragraph), kind: .blockQuote, depth: depth
                    )))
                } else {
                    appendBlocks(from: child, depth: depth + 1, into: &items)
                }
            }

        case is ThematicBreak:
            items.append(.text(.init(runs: [], kind: .horizontalRule, depth: depth)))

        case let table as Markdown.Table:
            items.append(.table(tableBlock(from: table)))

        default:
            // Unknown block: descend rather than drop, so a construct we do
            // not model still surfaces its text instead of vanishing.
            for child in markup.children {
                appendBlocks(from: child, depth: depth, into: &items)
            }
        }
    }

    private func appendListItem(
        _ item: ListItem,
        kind: MarkdownItem.TextBlock.Kind,
        index: Int?,
        depth: Int,
        into items: inout [MarkdownItem]
    ) {
        var isFirstParagraph = true
        for child in item.children {
            if let paragraph = child as? Paragraph, isFirstParagraph {
                items.append(.text(.init(
                    runs: inlineRuns(of: paragraph),
                    kind: kind,
                    depth: depth,
                    listIndex: index
                )))
                isFirstParagraph = false
            } else {
                // Nested lists and follow-on paragraphs indent one level.
                appendBlocks(from: child, depth: depth + 1, into: &items)
            }
        }
    }

    private func tableBlock(from table: Markdown.Table) -> MarkdownItem.TableBlock {
        let alignments = table.columnAlignments.map { alignment -> MarkdownItem.TableBlock.Alignment in
            switch alignment {
            case .center: .center
            case .right: .trailing
            default: .leading
            }
        }
        let header = Array(table.head.cells.map { inlineRuns(of: $0) })
        let rows = Array(table.body.rows.map { row in Array(row.cells.map { inlineRuns(of: $0) }) })
        return .init(header: header, rows: rows, alignments: alignments)
    }

    // MARK: - Images

    /// An images block, when `paragraph` holds images and nothing else.
    ///
    /// Returns nil when any non-image content is present — a sentence with an
    /// inline icon in it is prose, and pulling the icon out into a grid would
    /// break the sentence apart. ChatGPT draws the same line: its `.images`
    /// block is a separate `Item.Kind`, while an image inside running text
    /// arrives as a view attachment on the text instead.
    ///
    /// Whitespace-only text between images is ignored, since `![a](x) ![b](y)`
    /// parses as image, text(" "), image.
    private func imagesBlock(from paragraph: Paragraph) -> MarkdownItem.ImagesBlock? {
        var urls: [URL] = []
        var altTexts: [String] = []

        for child in paragraph.children {
            switch child {
            case let image as Markdown.Image:
                // An image whose source will not parse is dropped rather than
                // rendered as a broken tile.
                guard let source = image.source, let url = URL(string: source) else {
                    return nil
                }
                urls.append(url)
                altTexts.append(image.plainText)
            case let text as Markdown.Text
                where text.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
                continue
            case is SoftBreak, is LineBreak:
                continue
            default:
                return nil
            }
        }

        return urls.isEmpty ? nil : .init(urls: urls, altTexts: altTexts)
    }

    /// Append images, merging into a preceding images block.
    ///
    /// This is ChatGPT's `GroupAdjacentImagesMarkdownPlugin` inlined: images on
    /// consecutive lines belong in one grid, not in a column of one-image
    /// grids. Merging here rather than in a post-pass keeps `MarkdownItem`
    /// free of an intermediate "single image" state.
    private func appendImages(_ block: MarkdownItem.ImagesBlock, into items: inout [MarkdownItem]) {
        if case let .images(previous) = items.last {
            items[items.count - 1] = .images(.init(
                urls: previous.urls + block.urls,
                altTexts: previous.altTexts + block.altTexts
            ))
        } else {
            items.append(.images(block))
        }
    }

    // MARK: - Inline walk

    // MARK: - Inline math

    /// Private-use bracket around a decimal index. Nothing in it is markdown
    /// syntax, so it survives parsing as one contiguous piece of a run.
    ///
    /// Plane 15, not the U+E000 block: that block is where Nerd Fonts put
    /// their glyphs — U+E000 itself is the first Pomicons codepoint — so it
    /// arrives in pasted terminal output and in model text about fonts.
    /// A literal `U+E000 0 U+E001` in a message was enough to have the
    /// reader's own text replaced by somebody else's formula, and a literal
    /// `U+E000 -1 U+E001` crashed the renderer outright. Nothing ships glyphs
    /// in the supplementary private-use planes.
    private static let sentinelOpen: Character = "\u{F0000}"
    private static let sentinelClose: Character = "\u{F0001}"

    static func mathSentinel(_ index: Int) -> String {
        "\(sentinelOpen)\(index)\(sentinelClose)"
    }

    /// Belt and braces for the above: a sentinel that reaches the compiler in
    /// the source text is not ours, and is dropped before segmentation so it
    /// can never be read as an index.
    static func withoutStraySentinels(_ source: String) -> String {
        guard source.contains(sentinelOpen) || source.contains(sentinelClose) else {
            return source
        }
        return source.filter { $0 != sentinelOpen && $0 != sentinelClose }
    }

    /// Swap sentinels back for math runs, everywhere runs can appear.
    ///
    /// Tables and lists carry text blocks of their own, so this walks the
    /// whole item rather than only top-level paragraphs — inline math in a
    /// table cell is the case the original #131 review called out.
    static func restoringInlineMath(
        _ item: MarkdownItem, from latex: [String]
    ) -> MarkdownItem {
        switch item {
        case .text(let block):
            return .text(.init(
                runs: expandingSentinels(block.runs, from: latex),
                kind: block.kind,
                depth: block.depth,
                listIndex: block.listIndex
            ))
        case .table(let block):
            return .table(.init(
                header: block.header.map { expandingSentinels($0, from: latex) },
                rows: block.rows.map { $0.map { expandingSentinels($0, from: latex) } },
                alignments: block.alignments
            ))
        case .code(let block):
            // The segmenter skips code so `$x$` inside a block stays literal,
            // but it and swift-markdown do not agree on every spelling of a
            // block — a `~~~` fence is one swift-markdown recognises and the
            // segmenter does not. The sentinel then lands in code the reader
            // sees and the copy button copies. Putting the source spelling
            // back restores exactly what `main` renders, whichever construct
            // the two disagreed about.
            return .code(.init(
                code: restoringSourceSpelling(in: block.code, from: latex),
                language: block.language
            ))
        case .images, .math:
            // Neither has an inline layer to walk.
            return item
        }
    }

    /// Rewrite sentinels back to `$…$` in plain text that never became runs.
    static func restoringSourceSpelling(
        in text: String, from latex: [String]
    ) -> String {
        guard text.contains(sentinelOpen) else { return text }
        var output = ""
        var index = text.startIndex
        while index < text.endIndex {
            guard text[index] == sentinelOpen,
                  let close = text[index...].firstIndex(of: sentinelClose),
                  let slot = Int(text[text.index(after: index)..<close]),
                  latex.indices.contains(slot)
            else {
                output.append(text[index])
                index = text.index(after: index)
                continue
            }
            output += "$\(latex[slot])$"
            index = text.index(after: close)
        }
        return output
    }

    /// Split each run on sentinels, keeping the surrounding inline styling.
    ///
    /// A math run inherits `isStrong`/`isEmphasis`/`link` from the run it came
    /// out of, so `**$x$**` stays bold and a formula inside a link stays part
    /// of the link.
    static func expandingSentinels(
        _ runs: [InlineRun], from latex: [String]
    ) -> [InlineRun] {
        guard runs.contains(where: { $0.text.contains(sentinelOpen) }) else { return runs }
        var expanded: [InlineRun] = []
        for run in runs {
            guard run.text.contains(sentinelOpen) else { expanded.append(run); continue }
            var buffer = ""
            var index = run.text.startIndex
            while index < run.text.endIndex {
                guard run.text[index] == sentinelOpen,
                      let close = run.text[index...].firstIndex(of: sentinelClose),
                      let slot = Int(run.text[run.text.index(after: index)..<close]),
                      // `indices.contains`, not `slot < latex.count`:
                      // `Int("-1")` parses, and a negative index traps.
                      latex.indices.contains(slot)
                else {
                    buffer.append(run.text[index])
                    index = run.text.index(after: index)
                    continue
                }
                if !buffer.isEmpty {
                    var prose = run; prose.text = buffer; expanded.append(prose)
                    buffer = ""
                }
                var math = run
                // The source spelling, so copy and VoiceOver read `$x$` rather
                // than a private-use character nothing can render.
                math.text = "$\(latex[slot])$"
                math.math = latex[slot]
                expanded.append(math)
                index = run.text.index(after: close)
            }
            if !buffer.isEmpty {
                var prose = run; prose.text = buffer; expanded.append(prose)
            }
        }
        return expanded
    }

    private func inlineRuns(of markup: Markup) -> [InlineRun] {
        var runs: [InlineRun] = []
        collectInline(markup, style: InlineStyle(), into: &runs)
        return merge(runs)
    }

    private struct InlineStyle {
        var strong = false
        var emphasis = false
        var strikethrough = false
        var link: URL?
    }

    private func collectInline(_ markup: Markup, style: InlineStyle, into runs: inout [InlineRun]) {
        switch markup {
        case let text as Markdown.Text:
            runs.append(InlineRun(
                text: text.string,
                isStrong: style.strong,
                isEmphasis: style.emphasis,
                isStrikethrough: style.strikethrough,
                link: style.link
            ))

        case let code as InlineCode:
            runs.append(InlineRun(
                text: code.code,
                isStrong: style.strong,
                isEmphasis: style.emphasis,
                isStrikethrough: style.strikethrough,
                isInlineCode: true,
                link: style.link
            ))

        case is SoftBreak:
            // A single newline in the source is a space when rendered, per
            // CommonMark. Rendering it as a break would double-space prose
            // that was hard-wrapped at 80 columns.
            runs.append(InlineRun(text: " ", isStrong: style.strong, isEmphasis: style.emphasis))

        case is LineBreak:
            runs.append(InlineRun(text: "\n"))

        case let strong as Strong:
            var nested = style; nested.strong = true
            for child in strong.children { collectInline(child, style: nested, into: &runs) }

        case let emphasis as Emphasis:
            var nested = style; nested.emphasis = true
            for child in emphasis.children { collectInline(child, style: nested, into: &runs) }

        case let strike as Strikethrough:
            var nested = style; nested.strikethrough = true
            for child in strike.children { collectInline(child, style: nested, into: &runs) }

        case let link as Markdown.Link:
            var nested = style
            nested.link = link.destination.flatMap(URL.init(string:))
            for child in link.children { collectInline(child, style: nested, into: &runs) }

        default:
            for child in markup.children { collectInline(child, style: style, into: &runs) }
        }
    }

    /// Coalesce adjacent runs with identical styling.
    ///
    /// The AST produces one run per text node, so `**bold** text` arrives as
    /// several fragments. Merging keeps the attributed string's run count — and
    /// therefore the fade animator's per-run work — proportional to the
    /// *styling* rather than to the parse tree's shape.
    private func merge(_ runs: [InlineRun]) -> [InlineRun] {
        var merged: [InlineRun] = []
        for run in runs where !run.text.isEmpty {
            if var last = merged.last,
               last.isStrong == run.isStrong,
               last.isEmphasis == run.isEmphasis,
               last.isStrikethrough == run.isStrikethrough,
               last.isInlineCode == run.isInlineCode,
               last.link == run.link {
                last.text += run.text
                merged[merged.count - 1] = last
            } else {
                merged.append(run)
            }
        }
        return merged
    }
}
