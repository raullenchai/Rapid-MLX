import AppKit
import SwiftUI

/// Renders a compiled document by dispatching each block to the renderer that
/// suits it.
///
/// The split mirrors ChatGPT's `MarkdownBlockStack`, and it is not arbitrary:
///
///   * **text and code → TextKit 2.** The fade animator addresses glyphs by
///     `NSTextRange`, which only the text system provides. Lists and quotes
///     live inside the text block as paragraph styles rather than as stacked
///     views, so a whole list stays one addressable range.
///   * **tables and images → SwiftUI.** Both are layout problems, not text
///     problems. `MarkdownTableBlock` is a SwiftUI struct in the original too,
///     and `nonTableMaxWidth` shows tables are allowed to exceed the prose
///     column — easier to express with a `Grid` in a `ScrollView`.
///
/// Consecutive text blocks are merged into one view so that inter-paragraph
/// spacing comes from `NSParagraphStyle` rather than from stacked view
/// padding. Two adjacent paragraphs in one text view lay out exactly as the
/// text system intends; the same two in separate views need spacing rules that
/// have to be kept in sync by hand.
struct MarkdownBlockStack: View {
    let result: MarkdownResult
    let options: MarkdownOptions
    let isStreaming: Bool
    let fadeState: TextFadeAnimationState?
    let fadeConfiguration: TextFadeConfiguration

    public init(
        result: MarkdownResult,
        options: MarkdownOptions,
        isStreaming: Bool = false,
        fadeState: TextFadeAnimationState? = nil,
        fadeConfiguration: TextFadeConfiguration = TextFadeConfiguration()
    ) {
        self.result = result
        self.options = options
        self.isStreaming = isStreaming
        self.fadeState = fadeState
        self.fadeConfiguration = fadeConfiguration
    }

    public var body: some View {
        let groups = self.groups
        let lastTextIndex = groups.lastIndex { if case .text = $0 { return true } else { return false } }

        VStack(alignment: .leading, spacing: options.interContentSpacing) {
            ForEach(Array(groups.enumerated()), id: \.offset) { index, group in
                switch group {
                case .text(let blocks):
                    MarkdownTextBlockRepresentable(
                        blocks: blocks,
                        options: options,
                        // Only the trailing text group grows during a stream —
                        // earlier ones are already stable, and fading them
                        // would replay text the reader has read.
                        streaming: isStreaming && index == lastTextIndex,
                        fadeState: fadeState,
                        fadeConfiguration: fadeConfiguration
                    )
                case .code(let block):
                    MarkdownCodeBlockRepresentable(block: block, options: options)
                case .table(let block):
                    MarkdownTableView(block: block, options: options)
                        // #1824: VoiceOver reads tables today and that cannot
                        // regress. The visible grid stays as-is; a native
                        // `Table` stands in for it in the AX tree only.
                        .accessibilityRepresentation {
                            if let model = block.accessibilityModel {
                                AccessibleMarkdownTable(model: model)
                            }
                        }
                case .images(let block):
                    MarkdownImagesView(block: block, options: options)
                case .math(let block):
                    // Rapid's own SwiftMath wrapper (#131), not the port's.
                    // It already handles the macOS `fittingSize` trap and
                    // scales its base size off the same @ScaledMetric curve
                    // as the theme, so display math tracks Dynamic Type with
                    // the prose around it.
                    MathView(latex: block.latex, displayMode: true)
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private enum Group {
        case text([MarkdownItem.TextBlock])
        case code(MarkdownItem.CodeBlock)
        case table(MarkdownItem.TableBlock)
        case images(MarkdownItem.ImagesBlock)
        case math(MarkdownItem.MathBlock)
    }

    private var groups: [Group] {
        var groups: [Group] = []
        var pendingText: [MarkdownItem.TextBlock] = []

        func flushText() {
            guard !pendingText.isEmpty else { return }
            groups.append(.text(pendingText))
            pendingText = []
        }

        for item in result.items {
            switch item {
            case .text(let block):
                pendingText.append(block)
            case .code(let block):
                flushText(); groups.append(.code(block))
            case .table(let block):
                flushText(); groups.append(.table(block))
            case .images(let block):
                flushText(); groups.append(.images(block))
            case .math(let block):
                flushText(); groups.append(.math(block))
            }
        }
        flushText()
        return groups
    }
}
// MARK: - AppKit bridges

private struct MarkdownTextBlockRepresentable: NSViewRepresentable {
    let blocks: [MarkdownItem.TextBlock]
    let options: MarkdownOptions
    var streaming: Bool = false
    var fadeState: TextFadeAnimationState?
    var fadeConfiguration: TextFadeConfiguration = TextFadeConfiguration()

    func makeNSView(context: Context) -> MarkdownTextBlockView {
        let view = MarkdownTextBlockView(options: options)
        apply(to: view)
        // Let the view's intrinsic height drive layout while SwiftUI decides
        // the width.
        view.setContentHuggingPriority(.defaultHigh, for: .vertical)
        view.setContentCompressionResistancePriority(.required, for: .vertical)
        return view
    }

    func updateNSView(_ view: MarkdownTextBlockView, context: Context) {
        apply(to: view)
    }

    private func apply(to view: MarkdownTextBlockView) {
        view.configure(
            blocks: blocks,
            options: options,
            streaming: streaming,
            fadeState: fadeState,
            fadeConfiguration: fadeConfiguration
        )
    }

    func sizeThatFits(
        _ proposal: ProposedViewSize, nsView: MarkdownTextBlockView, context: Context
    ) -> CGSize? {
        guard let width = proposal.width, width > 0 else { return nil }
        // Report the width the text actually needs, not the whole proposal.
        //
        // Returning the proposed width unconditionally makes every text block
        // fill its container — which is right for assistant prose but wrong
        // inside a user bubble, where a three-word message would stretch to
        // the full proportional cap. `hugsTextHorizontally` is ChatGPT's own
        // switch for this distinction.
        let height = nsView.height(forWidth: width)
        let resolved = options.hugsTextHorizontally
            ? min(width, nsView.naturalWidth(maxWidth: width))
            : width
        return CGSize(width: resolved, height: height)
    }
}

private struct MarkdownCodeBlockRepresentable: NSViewRepresentable {
    let block: MarkdownItem.CodeBlock
    let options: MarkdownOptions

    func makeNSView(context: Context) -> MarkdownCodeBlockView {
        let view = MarkdownCodeBlockView(options: options)
        view.configure(code: block.code, language: block.language, options: options)
        return view
    }

    func updateNSView(_ view: MarkdownCodeBlockView, context: Context) {
        view.configure(code: block.code, language: block.language, options: options)
    }

    func sizeThatFits(
        _ proposal: ProposedViewSize, nsView: MarkdownCodeBlockView, context: Context
    ) -> CGSize? {
        guard let width = proposal.width, width > 0 else { return nil }
        return CGSize(width: width, height: nsView.height(forWidth: width))
    }
}

// MARK: - SwiftUI blocks

/// Table rendering.
///
/// Note what is absent: grid lines. ChatGPT's option table has
/// `tableCellInsets`, `tableBorderCornerRadius`, `tableHeaderBackgroundColor`
/// and `tableTextStyle` — and no field for cell borders. That absence is the
/// evidence: rows are separated by rhythm and a header fill, not by rules.
struct MarkdownTableView: View {
    let block: MarkdownItem.TableBlock
    let options: MarkdownOptions

    var body: some View {
        // Horizontal scrolling because `nonTableMaxWidth` in the original says
        // tables may exceed the prose column rather than compress into it.
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                row(block.header, isHeader: true)
                ForEach(Array(block.rows.enumerated()), id: \.offset) { _, cells in
                    row(cells, isHeader: false)
                }
            }
            .clipShape(RoundedRectangle(
                cornerRadius: options.tableBorderCornerRadius, style: .continuous
            ))
        }
    }

    private func row(_ cells: [[InlineRun]], isHeader: Bool) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(cells.enumerated()), id: \.offset) { index, runs in
                Text(styled(runs, isHeader: isHeader))
                    .frame(
                        minWidth: 80,
                        alignment: alignment(for: index)
                    )
                    .padding(.horizontal, options.tableCellInsets.leading)
                    .padding(.vertical, options.tableCellInsets.top)
            }
        }
        .frame(minHeight: options.tableRowHeight, alignment: .leading)
        .background(isHeader
            ? Color(nsColor: options.tableHeaderBackgroundColor ?? .clear)
            : Color.clear)
    }

    private func alignment(for column: Int) -> SwiftUI.Alignment {
        guard column < block.alignments.count else { return .leading }
        switch block.alignments[column] {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    /// Build a cell's text with its inline styling intact.
    ///
    /// The compiler already captures bold / italic / strikethrough / code /
    /// links per run; the previous version joined `runs.map(\.text)` and threw
    /// all of it away, so `**bold**` inside a cell rendered as plain text.
    ///
    /// `AttributedString` rather than TextKit 2 here because a table cell is a
    /// SwiftUI layout problem — the fade animator never addresses table text,
    /// so nothing needs an `NSTextRange` into it.
    static func styled(
        _ runs: [InlineRun], isHeader: Bool, options: MarkdownOptions
    ) -> AttributedString {
        var result = AttributedString()
        for run in runs {
            var piece = AttributedString(run.text)
            // The header's semibold is a row-level decision, so a run that is
            // *also* bold cannot go heavier — it simply stays semibold.
            let weight: Font.Weight = (isHeader || run.isStrong) ? .semibold : .regular
            var font = Font.system(
                size: options.tablePointSize,
                weight: weight,
                design: run.isInlineCode ? .monospaced : .default
            )
            if run.isEmphasis { font = font.italic() }
            piece.font = font
            if run.isStrikethrough { piece.strikethroughStyle = .single }
            if let link = run.link {
                piece.link = link
                piece.foregroundColor = Color(nsColor: options.linkColor)
            }
            result.append(piece)
        }
        return result
    }

    private func styled(_ runs: [InlineRun], isHeader: Bool) -> AttributedString {
        Self.styled(runs, isHeader: isHeader, options: options)
    }
}

struct MarkdownImagesView: View {
    let block: MarkdownItem.ImagesBlock
    let options: MarkdownOptions

    var body: some View {
        LazyVGrid(
            columns: Array(
                repeating: GridItem(.flexible(), spacing: options.imageGridSpacing),
                count: min(block.urls.count, options.maxImagesPerGridRowRegular)
            ),
            spacing: options.imageGridSpacing
        ) {
            ForEach(Array(block.urls.enumerated()), id: \.offset) { index, url in
                AsyncImage(url: url) { image in
                    image.resizable().aspectRatio(contentMode: .fit)
                } placeholder: {
                    Rectangle().fill(.quaternary)
                }
                .frame(maxWidth: options.maxImageWidth, maxHeight: options.maxImageHeight)
                .clipShape(RoundedRectangle(
                    cornerRadius: block.urls.count > 1
                        ? options.gridImageCornerRadius
                        : options.imageCornerRadius,
                    style: .continuous
                ))
                .accessibilityLabel(index < block.altTexts.count ? block.altTexts[index] : "")
            }
        }
        .frame(maxWidth: options.maxImageGridWidth)
    }
}
