import AppKit

/// Renders a fenced code block: monospaced body on a rounded card, with a
/// header carrying the language and a copy button.
///
/// TextKit 2 like the prose renderer, for the same reason — `codeTextStyle`
/// and `codeTextContainerInset` are text-system parameters in ChatGPT's field
/// table, and the fade animator needs to reach code as well as prose.
final class MarkdownCodeBlockView: NSView {

    private let renderer: MarkdownTextRenderer
    private var options: MarkdownOptions
    private var code: String = ""
    private var language: String?

    private let headerLabel = NSTextField(labelWithString: "")
    private let copyButton = NSButton()
    private let previewButton = NSButton()
    private var labelBeforePreviewConstraint: NSLayoutConstraint!
    private var labelBeforeCopyConstraint: NSLayoutConstraint!
    private var didCopyResetWork: DispatchWorkItem?

    /// The parsed document, kept so `draw(_:)` and `height(forWidth:)` do not
    /// re-parse on every pass. Nil means "not an SVG, or not yet valid" —
    /// which is what half a streamed document looks like.
    private var previewImage: NSImage?
    private var previewSource: String?
    private var isShowingPreview = false

    public override var isFlipped: Bool { true }

    public init(options: MarkdownOptions) {
        self.options = options
        self.renderer = MarkdownTextRenderer(options: options)
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = options.codeCornerRadius
        layer?.masksToBounds = true
        setAccessibilityElement(true)
        setAccessibilityRole(.staticText)
        setAccessibilityEnabled(true)
        setUpHeader()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    private func setUpHeader() {
        headerLabel.font = .systemFont(ofSize: 11, weight: .medium)
        headerLabel.textColor = .secondaryLabelColor
        headerLabel.lineBreakMode = .byTruncatingTail
        headerLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        headerLabel.translatesAutoresizingMaskIntoConstraints = false
        addSubview(headerLabel)

        copyButton.title = "复制"
        copyButton.bezelStyle = .inline
        copyButton.isBordered = false
        copyButton.font = .systemFont(ofSize: 11, weight: .medium)
        copyButton.contentTintColor = .secondaryLabelColor
        copyButton.target = self
        copyButton.action = #selector(copyCode)
        copyButton.translatesAutoresizingMaskIntoConstraints = false
        addSubview(copyButton)

        previewButton.title = "Preview"
        previewButton.bezelStyle = .inline
        previewButton.isBordered = false
        previewButton.font = .systemFont(ofSize: 11, weight: .medium)
        previewButton.contentTintColor = .secondaryLabelColor
        previewButton.target = self
        previewButton.action = #selector(togglePreview)
        previewButton.isHidden = true
        previewButton.setAccessibilityIdentifier("CodeBlock.Preview")
        previewButton.translatesAutoresizingMaskIntoConstraints = false
        addSubview(previewButton)

        labelBeforePreviewConstraint = headerLabel.trailingAnchor.constraint(
            lessThanOrEqualTo: previewButton.leadingAnchor,
            constant: -RapidTheme.Space.md)
        labelBeforeCopyConstraint = headerLabel.trailingAnchor.constraint(
            lessThanOrEqualTo: copyButton.leadingAnchor,
            constant: -RapidTheme.Space.md)
        labelBeforeCopyConstraint.isActive = true

        NSLayoutConstraint.activate([
            previewButton.trailingAnchor.constraint(
                equalTo: copyButton.leadingAnchor, constant: -RapidTheme.Space.md),
            previewButton.centerYAnchor.constraint(equalTo: headerLabel.centerYAnchor),
            headerLabel.leadingAnchor.constraint(
                equalTo: leadingAnchor, constant: options.codeHeaderInsets.leading),
            headerLabel.topAnchor.constraint(
                equalTo: topAnchor, constant: options.codeHeaderInsets.top),
            copyButton.trailingAnchor.constraint(
                equalTo: trailingAnchor, constant: -options.codeHeaderInsets.trailing),
            copyButton.centerYAnchor.constraint(equalTo: headerLabel.centerYAnchor),
        ])
    }

    public func configure(code: String, language: String?, options: MarkdownOptions) {
        self.code = code
        self.language = language
        self.options = options

        var codeOptions = options
        // The code body has its own type scale; reusing the prose renderer
        // with substituted metrics keeps one text pipeline rather than two.
        codeOptions.textPointSize = options.codePointSize
        codeOptions.lineHeightMultiple = options.codeLineHeight / options.codePointSize
        codeOptions.paragraphSpacing = 0
        renderer.update(options: codeOptions)
        renderer.setCode(code, language: language)
        setAccessibilityValue(code)

        headerLabel.stringValue = language?.capitalized ?? ""
        headerLabel.isHidden = (language?.isEmpty ?? true)
        updatePreviewAvailability()
        layer?.cornerRadius = options.codeCornerRadius
        applyAppearanceDependentColors()
        needsDisplay = true
        // Both source text and an already-open preview can change height on a
        // streaming reconfiguration (including a new SVG aspect ratio). The
        // SwiftUI representable will not necessarily remeasure an AppKit row
        // merely because its header height stayed the same.
        invalidateLayoutChain()
    }

    /// Paint the card fill and border for the CURRENT appearance.
    ///
    /// A `CALayer` takes a `CGColor`, which is a resolved value — asking a
    /// dynamic `NSColor` for `.cgColor` snapshots it against whatever
    /// appearance happens to be current and never updates again. So the
    /// resolve is wrapped in `performAsCurrentDrawingAppearance` (which makes
    /// this view's effective appearance the one the provider sees) and re-run
    /// from ``viewDidChangeEffectiveAppearance`` on every light/dark flip.
    /// Without the second half, switching to dark left a near-white card
    /// behind dark-palette syntax colours until the block was rebuilt.
    private func applyAppearanceDependentColors() {
        let fill = options.codeBlockBackground
        let border = options.codeBlockBorder
        effectiveAppearance.performAsCurrentDrawingAppearance {
            layer?.backgroundColor = fill.cgColor
            layer?.borderWidth = border == nil ? 0 : 1
            layer?.borderColor = border?.cgColor
        }
    }

    public override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        applyAppearanceDependentColors()
        // The syntax palette is resolved into the attributed string at build
        // time, so unlike the fill it cannot re-resolve itself — the text has
        // to be rebuilt against the new appearance.
        renderer.setCode(code, language: language)
        needsDisplay = true
    }

    /// Room for the header row.
    ///
    /// Reserved for a language tag OR for a preview button, and the second
    /// half was missing. ``SVGPreview/looksLikeSVG(code:language:)`` ignores
    /// the tag, so an untagged ``` fence holding an SVG got a button laid out
    /// with no room reserved — and once the picture is wide enough to reach
    /// the top right, it paints underneath Preview and Copy with nothing
    /// behind them. Source text hid this because it is left-aligned and
    /// rarely reaches that corner; a full-width image reaches it every time.
    private var headerHeight: CGFloat {
        guard !(language?.isEmpty ?? true) || previewImage != nil else { return 0 }
        return options.codeHeaderInsets.top + 16 + options.codeHeaderInsets.bottom
    }

    public func height(forWidth width: CGFloat) -> CGFloat {
        let contentWidth = max(0, width - options.codeInsets.leading - options.codeInsets.trailing)
        return headerHeight + options.codeInsets.top
            + bodyHeight(forWidth: contentWidth) + options.codeInsets.bottom
    }

    /// The card shows the source or the picture, never both.
    ///
    /// Preview is a mode, not an addition. Stacking the two doubles the height
    /// of every previewed block, and for anything but a toy document the
    /// source dominates the card while the thing the reader asked to see is
    /// pushed off the bottom of the window.
    private func bodyHeight(forWidth contentWidth: CGFloat) -> CGFloat {
        if isShowingPreview, let previewImage {
            let size = SVGPreview.drawSize(for: previewImage.size, inWidth: contentWidth)
            if size.height > 0 { return size.height }
        }
        return renderer.measureHeight(width: contentWidth)
    }

    public override var intrinsicContentSize: NSSize {
        guard bounds.width > 0 else { return NSSize(width: NSView.noIntrinsicMetric, height: 0) }
        return NSSize(width: NSView.noIntrinsicMetric, height: height(forWidth: bounds.width))
    }

    public override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard bounds.width > 0, let context = NSGraphicsContext.current?.cgContext else { return }

        let textWidth = bounds.width - options.codeInsets.leading - options.codeInsets.trailing

        if drawPreview(contentWidth: max(0, textWidth)) { return }

        renderer.textContainer.size = CGSize(
            width: max(0, textWidth), height: CGFloat.greatestFiniteMagnitude
        )
        renderer.textLayoutManager.ensureLayout(for: renderer.textContentStorage.documentRange)

        context.saveGState()
        context.translateBy(
            x: options.codeInsets.leading,
            y: headerHeight + options.codeInsets.top
        )
        renderer.textLayoutManager.enumerateTextLayoutFragments(
            from: renderer.textLayoutManager.documentRange.location,
            options: [.ensuresLayout, .ensuresExtraLineFragment]
        ) { fragment in
            fragment.draw(at: fragment.layoutFragmentFrame.origin, in: context)
            return true
        }
        context.restoreGState()
    }

    /// Draw the SVG in place of the source, and report whether it did.
    ///
    /// No backing plate: the reader asked for the document to be shown, not
    /// for it to be shown on a canvas of our choosing. That does mean an SVG
    /// drawn entirely in black strokes is invisible against a dark card — a
    /// real cost, accepted because the alternative is a white rectangle
    /// punched into every dark-mode transcript, and because the source is one
    /// press away.
    @discardableResult
    private func drawPreview(contentWidth: CGFloat) -> Bool {
        guard isShowingPreview, let previewImage else { return false }
        let size = SVGPreview.drawSize(for: previewImage.size, inWidth: contentWidth)
        guard size.width > 0, size.height > 0 else { return false }
        let origin = CGPoint(
            x: options.codeInsets.leading,
            y: headerHeight + options.codeInsets.top
        )
        // `respectFlipped: true` is the whole of it. This view is `isFlipped`,
        // and the short `draw(in:)` overload does not compensate — it paints
        // the image bottom-up into a top-down coordinate space. Nothing else
        // is needed, and mirroring the destination as well flips it back.
        previewImage.draw(
            in: CGRect(origin: origin, size: size),
            from: .zero,
            operation: .sourceOver,
            fraction: 1,
            respectFlipped: true,
            hints: nil
        )
        return true
    }

    /// Re-decide whether this block can be previewed, and re-parse if so.
    ///
    /// Runs on every `configure`, which during a stream is every flush. The
    /// cheap `looksLikeSVG` check short-circuits before the parse, so a plain
    /// Swift block costs one substring search per flush and nothing else.
    private func updatePreviewAvailability() {
        guard SVGPreview.looksLikeSVG(code: code, language: language) else {
            setPreviewHidden(true)
            previewImage = nil
            previewSource = nil
            isShowingPreview = false
            return
        }
        if previewSource != code {
            previewSource = code
            previewImage = SVGPreview.image(from: code)
        }
        // A document that does not parse yet — the usual case mid-stream —
        // offers no button. It appears when the last tag closes.
        setPreviewHidden(previewImage == nil)
        if previewImage == nil { isShowingPreview = false }
        previewButton.title = isShowingPreview ? "Code" : "Preview"
    }

    private func setPreviewHidden(_ hidden: Bool) {
        previewButton.isHidden = hidden
        labelBeforePreviewConstraint.isActive = !hidden
        labelBeforeCopyConstraint.isActive = hidden
    }

    @objc private func togglePreview() {
        isShowingPreview.toggle()
        previewButton.title = isShowingPreview ? "Code" : "Preview"
        needsDisplay = true
        // The block stack sizes rows from `height(forWidth:)`, so the row has
        // to be re-measured rather than merely redrawn.
        invalidateLayoutChain()
    }

    private func invalidateLayoutChain() {
        invalidateIntrinsicContentSize()
        var ancestor = superview
        while let view = ancestor {
            view.invalidateIntrinsicContentSize()
            view.needsLayout = true
            ancestor = view.superview
        }
    }

    @objc private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)

        // Momentary confirmation, matching ChatGPT's
        // `MarkdownCodeBlockHeaderCopyButton` which tracks a
        // `recentlyPerformed` state.
        copyButton.title = "已复制"
        didCopyResetWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.copyButton.title = "复制" }
        didCopyResetWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6, execute: work)
    }
}
