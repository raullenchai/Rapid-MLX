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

    /// Passes run over the compiled blocks, in order.
    ///
    /// Auto-linking by default because a bare URL rendering as dead text is a
    /// bug in a chat transcript, not a stylistic choice. Pass an empty array
    /// for a strict-CommonMark compile.
    public var postProcessors: [MarkdownPostProcessor]

    public init(postProcessors: [MarkdownPostProcessor] = [AutoLinkPostProcessor()]) {
        self.postProcessors = postProcessors
    }

    /// Compile `source`.
    ///
    /// `isComplete` reaches the post-processors: mid-stream they should leave
    /// half-arrived constructs alone rather than commit to a reading they will
    /// have to undo on the next flush.
    public func compile(
        _ source: String, revision: Int = 0, isComplete: Bool = true
    ) -> MarkdownResult {
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
        for segment in LaTeXSegmenter.segment(source) {
            switch segment {
            case let .math(latex, displayMode) where displayMode:
                appendMarkdown(pending, depth: 0, into: &items)
                pending = ""
                items.append(.math(.init(latex: latex)))
            case let .math(latex, _):
                pending += "$\(latex)$"
            case let .markdown(body):
                pending += body
            }
        }
        appendMarkdown(pending, depth: 0, into: &items)

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
        let document = Document(parsing: source, options: [.parseBlockDirectives])
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
