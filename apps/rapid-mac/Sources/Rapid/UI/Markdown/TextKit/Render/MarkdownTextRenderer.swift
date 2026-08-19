import AppKit

/// Turns compiled markdown blocks into an `NSAttributedString`, and owns the
/// TextKit 2 objects that lay it out.
///
/// TextKit 2 rather than SwiftUI `Text` for one decisive reason: the fade
/// animator has to address glyph runs by `NSTextRange` to vary their opacity
/// and colour mid-stream, and SwiftUI exposes no such handle. Everything else
/// follows from that choice — and pays off twice, because
/// `NSTextLayoutManager` can also measure text *without a view*, which is
/// 20-50× cheaper than instantiating an `NSHostingController` per row.
///
/// ChatGPT made the same call: its `HorizontallyScrollingMarkdownTextBlock`
/// has `typealias Body = Never`, the signature of an `NSViewRepresentable`.
@MainActor
final class MarkdownTextRenderer {

    public let textContentStorage = NSTextContentStorage()
    public let textLayoutManager = NSTextLayoutManager()
    public let textContainer: NSTextContainer

    private var options: MarkdownOptions

    public init(options: MarkdownOptions) {
        self.options = options
        textContainer = NSTextContainer(size: CGSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        textContainer.lineFragmentPadding = 0
        textLayoutManager.textContainer = textContainer
        textContentStorage.addTextLayoutManager(textLayoutManager)
    }

    public func update(options: MarkdownOptions) {
        self.options = options
    }

    // MARK: - Content

    /// Replace the content.
    ///
    /// `showsTypingDot` appends the pulsing indicator as an attachment on the
    /// last line. It is part of the text, not an overlay, so it flows and wraps
    /// with the final glyph and needs no frame maintenance.
    public func setBlocks(_ blocks: [MarkdownItem.TextBlock], showsTypingDot: Bool = false) {
        let string = NSMutableAttributedString(attributedString: attributedString(for: blocks))
        proseLength = string.length
        typingDotLocation = nil
        if showsTypingDot {
            string.append(typingDotString(trailing: string))
        }
        textContentStorage.performEditingTransaction {
            textContentStorage.textStorage?.setAttributedString(string)
        }
    }

    /// Length of the prose, excluding any typing dot.
    ///
    /// The fade animator schedules against this: the dot is a character that
    /// appears and vanishes at the tail on every flush, and treating it as text
    /// would have the animator schedule a unit that is gone by the next frame.
    public private(set) var proseLength: Int = 0

    /// Character offset of the typing dot, or nil when it is absent.
    public private(set) var typingDotLocation: Int?

    /// Visible prose without the private attachment character used by the
    /// streaming typing dot. Custom-drawn text views publish this through AX.
    var accessibleText: String {
        guard let storage = textContentStorage.textStorage else { return "" }
        let range = NSRange(location: 0, length: min(proseLength, storage.length))
        let source = NSMutableString(string: (storage.string as NSString).substring(with: range))

        // NSTextAttachment occupies one U+FFFC character in the backing
        // string. That is correct for layout, but this custom-drawn view uses
        // this value as its entire accessibility surface: leaving the object
        // replacement character here makes VoiceOver announce a hole where
        // the formula is. Replace math attachments from the tail so earlier
        // ranges stay valid while the string grows.
        var replacements: [(NSRange, String)] = []
        storage.enumerateAttribute(.attachment, in: range) { value, attachmentRange, _ in
            guard let math = value as? InlineMathAttachment else { return }
            replacements.append((attachmentRange, "$\(math.latex)$"))
        }
        for (attachmentRange, latex) in replacements.reversed() {
            source.replaceCharacters(in: attachmentRange, with: latex)
        }
        return source as String
    }

    private func typingDotString(trailing prose: NSAttributedString) -> NSAttributedString {
        let font = NSFont.systemFont(ofSize: options.textPointSize)
        let attachment = TypingDotAttachment(
            color: options.textColor, pointSize: font.pointSize
        )
        let string = NSMutableAttributedString(attachment: attachment)
        // A hair of space so the dot does not touch the final glyph.
        string.insert(NSAttributedString(string: "\u{2009}"), at: 0)
        typingDotLocation = prose.length + string.length - 1

        // Inherit the last paragraph's style, or the dot starts its own line
        // with default spacing.
        if prose.length > 0 {
            let inherited = prose.attributes(at: prose.length - 1, effectiveRange: nil)
            if let style = inherited[.paragraphStyle] {
                string.addAttribute(.paragraphStyle, value: style,
                                    range: NSRange(location: 0, length: string.length))
            }
        }
        string.addAttribute(.font, value: font,
                            range: NSRange(location: 0, length: string.length))
        return string
    }

    /// Laid-out rect of a single character, in the container's coordinates.
    ///
    /// Used to place the typing dot. Returns nil if the character has no
    /// segment yet — during the frame between an edit and the next layout pass.
    public func rect(forCharacterAt offset: Int) -> CGRect? {
        guard let start = textContentStorage.location(
                textContentStorage.documentRange.location, offsetBy: offset),
              let end = textContentStorage.location(start, offsetBy: 1),
              let range = NSTextRange(location: start, end: end) else { return nil }

        var result: CGRect?
        textLayoutManager.enumerateTextSegments(
            in: range, type: .standard, options: []
        ) { _, frame, _, _ in
            result = frame
            return false
        }
        return result
    }

    /// Resolve a rendered link at a view-local point. TextKit's drawing-only
    /// host has no NSTextView delegate to do this on our behalf.
    func link(at point: CGPoint) -> URL? {
        linkRegions().first { $0.rect.contains(point) }?.url
    }

    /// Bounding rects of every link run, for cursor tracking.
    ///
    /// One rect per run rather than per character: adjacent character rects on
    /// the same line are merged, so a link is a single tracking area instead of
    /// dozens.
    func linkRects() -> [CGRect] {
        linkRegions().map(\.rect)
    }

    /// Geometry for each laid-out line segment carrying a link.
    ///
    /// TextKit already splits an attributed range at line boundaries. Asking
    /// it for those segments avoids both per-character geometry calls and the
    /// fragile `minY` tolerance previously used to merge character rects.
    private func linkRegions() -> [(url: URL, rect: CGRect)] {
        guard let storage = textContentStorage.textStorage else { return [] }
        textLayoutManager.ensureLayout(for: textContentStorage.documentRange)
        var regions: [(url: URL, rect: CGRect)] = []
        storage.enumerateAttribute(
            .link,
            in: NSRange(location: 0, length: min(proseLength, storage.length))
        ) { value, range, _ in
            guard let url = value as? URL,
                  let start = textContentStorage.location(
                    textContentStorage.documentRange.location,
                    offsetBy: range.location
                  ),
                  let end = textContentStorage.location(start, offsetBy: range.length),
                  let textRange = NSTextRange(location: start, end: end)
            else { return }

            textLayoutManager.enumerateTextSegments(
                in: textRange, type: .standard, options: []
            ) { _, frame, _, _ in
                regions.append((url, frame))
                return true
            }
        }
        return regions
    }

    /// Lay out at a given width and report the height the text needs.
    ///
    /// Deliberately usable with no view attached: this is the measurement path
    /// the collection view calls for offscreen rows, and it is the second
    /// dividend of owning the renderer.
    public func measureHeight(width: CGFloat) -> CGFloat {
        guard width > 0 else { return 0 }
        textContainer.size = CGSize(width: width, height: CGFloat.greatestFiniteMagnitude)
        textLayoutManager.textViewportLayoutController.layoutViewport()
        textLayoutManager.ensureLayout(for: textContentStorage.documentRange)
        return ceil(textLayoutManager.usageBoundsForTextContainer.height)
    }

    /// Width the laid-out text actually occupies, up to `maxWidth`.
    ///
    /// `usageBoundsForTextContainer` reports the used rect rather than the
    /// container, so a short line yields its own width rather than the
    /// container's. This is what lets a user bubble hug its text instead of
    /// stretching to the proportional cap.
    public func measureNaturalWidth(maxWidth: CGFloat) -> CGFloat {
        guard maxWidth > 0 else { return 0 }
        textContainer.size = CGSize(width: maxWidth, height: CGFloat.greatestFiniteMagnitude)
        textLayoutManager.ensureLayout(for: textContentStorage.documentRange)
        return ceil(textLayoutManager.usageBoundsForTextContainer.width)
    }

    // MARK: - Attributed string construction

    /// Replace the content with a fenced code block.
    ///
    /// A separate entry point from ``setBlocks`` because code is not prose and
    /// routing it through the prose path was wrong twice over:
    ///
    ///   * inline-code runs carry a per-run `.backgroundColor`, which painted a
    ///     grey bar behind each line whose length tracked the text — instead of
    ///     one card behind the whole block;
    ///   * the prose paragraph style collapsed blank lines, so consecutive
    ///     statements ran together. Code that will not run when pasted is worse
    ///     than code that looks wrong.
    public func setCode(_ code: String, language: String?) {
        let font = NSFont.monospacedSystemFont(ofSize: options.codePointSize, weight: .regular)

        let paragraph = NSMutableParagraphStyle()
        // Exact line height, not a multiple: mixing CJK comments with Latin
        // code otherwise gives lines of two different heights.
        paragraph.minimumLineHeight = options.codeLineHeight
        paragraph.maximumLineHeight = options.codeLineHeight
        paragraph.paragraphSpacing = 0
        // Wrap long lines rather than clip them.
        //
        // `.byClipping` was tried first, on the theory that a wrapped line of
        // code reads as a new statement. It does the opposite: the clipped
        // remainder vanishes, so `pivot = …`, `left = […]` and `middle = […]`
        // appeared merged onto one line and the right-hand side of the block
        // was simply gone. Losing code is worse than wrapping it. Real
        // horizontal scrolling is the eventual answer; wrapping is the correct
        // fallback until then.
        paragraph.lineBreakMode = .byWordWrapping

        let string = NSMutableAttributedString(
            string: code,
            attributes: [
                .font: font,
                .foregroundColor: options.textColor,
                .paragraphStyle: paragraph,
            ]
        )

        let theme = isDarkAppearance ? CodeHighlighter.Theme.darkDefault : .default
        for (range, colour) in CodeHighlighter.ranges(in: code, language: language, theme: theme) {
            string.addAttribute(NSAttributedString.Key.foregroundColor,
                                value: colour, range: range)
        }

        proseLength = string.length
        typingDotLocation = nil
        textContentStorage.performEditingTransaction {
            textContentStorage.textStorage?.setAttributedString(string)
        }
    }

    private var isDarkAppearance: Bool {
        NSApp?.effectiveAppearance
            .bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
    }

    public func attributedString(for blocks: [MarkdownItem.TextBlock]) -> NSAttributedString {
        let output = NSMutableAttributedString()
        for (index, block) in blocks.enumerated() {
            if index > 0 { output.append(NSAttributedString(string: "\n")) }
            output.append(attributedString(for: block))
        }
        return output
    }

    public func attributedString(for block: MarkdownItem.TextBlock) -> NSAttributedString {
        if case .horizontalRule = block.kind {
            // A rule has no text. It is drawn as a decoration by the view; the
            // string carries one newline so it still occupies a line.
            return NSAttributedString(string: "\n")
        }

        let output = NSMutableAttributedString()
        let paragraphStyle = paragraphStyle(for: block)
        let baseFont = font(for: block)

        // Ordered/unordered list markers are part of the string rather than a
        // separate view, so the hanging indent in the paragraph style lines up
        // wrapped text under the first character — which is what ChatGPT's
        // nine list metrics describe.
        if let marker = listMarker(for: block) {
            output.append(NSAttributedString(string: marker, attributes: [
                .font: baseFont,
                .foregroundColor: options.textColor,
                .paragraphStyle: paragraphStyle,
            ]))
        }

        for run in block.runs {
            output.append(attributedString(for: run, block: block,
                                           baseFont: baseFont, paragraphStyle: paragraphStyle))
        }
        return output
    }

    private func attributedString(
        for run: InlineRun,
        block: MarkdownItem.TextBlock,
        baseFont: NSFont,
        paragraphStyle: NSParagraphStyle
    ) -> NSAttributedString {
        var attributes: [NSAttributedString.Key: Any] = [
            .paragraphStyle: paragraphStyle,
            .foregroundColor: options.textColor,
        ]

        // Inline math becomes one attachment character. Falling through to the
        // prose path when rasterising fails is deliberate: an unparseable body
        // renders as the `$…$` the author typed, which is worse than a formula
        // and much better than a blank.
        if let latex = run.math,
           let image = InlineMathImage.image(
               latex: latex,
               pointSize: options.textPointSize,
               color: options.textColor
           ) {
            let attachment = InlineMathAttachment(
                latex: latex, image: image, pointSize: options.textPointSize
            )
            let string = NSMutableAttributedString(attachment: attachment)
            string.addAttributes(
                [.paragraphStyle: paragraphStyle],
                range: NSRange(location: 0, length: string.length)
            )
            if let link = run.link {
                string.addAttribute(
                    .link, value: link,
                    range: NSRange(location: 0, length: string.length)
                )
            }
            return string
        }

        if run.isInlineCode {
            attributes[.font] = NSFont.monospacedSystemFont(
                ofSize: options.textPointSize - 1, weight: .regular
            )
            if options.inlineCodeBackgroundEnabled, let bg = options.inlineCodeBackgroundColor {
                attributes[.backgroundColor] = bg
            }
            if let color = options.inlineCodeTextColor {
                attributes[.foregroundColor] = color
            }
        } else {
            var font = baseFont
            if run.isStrong {
                font = NSFontManager.shared.convert(font, toHaveTrait: .boldFontMask)
                if let strongColor = options.strongTextColor {
                    attributes[.foregroundColor] = strongColor
                }
            }
            if run.isEmphasis {
                font = NSFontManager.shared.convert(font, toHaveTrait: .italicFontMask)
            }
            attributes[.font] = font
        }

        if run.isStrikethrough {
            attributes[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if let link = run.link {
            attributes[.link] = link
            attributes[.foregroundColor] = options.linkColor
            if let style = options.linkUnderlineStyle {
                attributes[.underlineStyle] = style.rawValue
                attributes[.underlineColor] = options.linkUnderlineColor ?? options.linkColor
            }
        }
        if let kern = options.kern {
            attributes[.kern] = kern
        }

        return NSAttributedString(string: run.text, attributes: attributes)
    }

    // MARK: - Styling

    private func font(for block: MarkdownItem.TextBlock) -> NSFont {
        switch block.kind {
        case .heading(let level):
            // Six levels compressed onto a ramp that keeps h1 readable next to
            // 15pt body without shouting.
            let scale: [CGFloat] = [1.55, 1.35, 1.2, 1.1, 1.0, 0.95]
            let size = options.textPointSize * scale[min(max(level, 1), 6) - 1]
            return .systemFont(ofSize: size, weight: .semibold)
        default:
            return .systemFont(ofSize: options.textPointSize)
        }
    }

    private func paragraphStyle(for block: MarkdownItem.TextBlock) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = options.lineHeightMultiple

        switch block.kind {
        case .paragraph, .heading:
            style.paragraphSpacing = options.paragraphSpacing

        case .unorderedListItem, .orderedListItem:
            let depthIndent = options.listDepthIndent * CGFloat(block.depth)
            let base = block.kind == .orderedListItem
                ? options.orderedListBaseLeftMargin
                : options.unorderedListBaseLeftMargin
            let markerGap = block.kind == .orderedListItem
                ? options.listSpacingFromIndex
                : options.unorderedListSpacingFromMarker
            style.firstLineHeadIndent = base + depthIndent
            // Wrapped lines align under the text, not under the bullet.
            style.headIndent = base + depthIndent + markerGap + 8
            style.paragraphSpacing = options.listInterItemSpacing

        case .blockQuote:
            let indent = options.blockQuoteLeadingInset + options.blockQuoteIndentation
                * CGFloat(block.depth)
            style.firstLineHeadIndent = indent
            style.headIndent = indent
            style.paragraphSpacing = options.paragraphSpacing

        case .horizontalRule:
            style.paragraphSpacing = 0
        }
        return style
    }

    private func listMarker(for block: MarkdownItem.TextBlock) -> String? {
        switch block.kind {
        case .unorderedListItem:
            // Depth-varied bullets, matching the usual macOS convention.
            let bullets = ["•", "◦", "▪"]
            return bullets[min(block.depth, bullets.count - 1)] + "\t"
        case .orderedListItem:
            guard let index = block.listIndex else { return nil }
            return "\(index).\t"
        default:
            return nil
        }
    }
}
