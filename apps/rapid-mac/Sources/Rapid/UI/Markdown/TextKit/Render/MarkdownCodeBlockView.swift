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
    private var didCopyResetWork: DispatchWorkItem?

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

        NSLayoutConstraint.activate([
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
        layer?.cornerRadius = options.codeCornerRadius
        layer?.backgroundColor = options.codeBlockBackground.cgColor
        if let border = options.codeBlockBorder {
            layer?.borderWidth = 1
            layer?.borderColor = border.cgColor
        }
        needsDisplay = true
        invalidateIntrinsicContentSize()
    }

    private var headerHeight: CGFloat {
        guard !(language?.isEmpty ?? true) else { return 0 }
        return options.codeHeaderInsets.top + 16 + options.codeHeaderInsets.bottom
    }

    public func height(forWidth width: CGFloat) -> CGFloat {
        let textWidth = width - options.codeInsets.leading - options.codeInsets.trailing
        let textHeight = renderer.measureHeight(width: max(0, textWidth))
        return headerHeight + options.codeInsets.top + textHeight + options.codeInsets.bottom
    }

    public override var intrinsicContentSize: NSSize {
        guard bounds.width > 0 else { return NSSize(width: NSView.noIntrinsicMetric, height: 0) }
        return NSSize(width: NSView.noIntrinsicMetric, height: height(forWidth: bounds.width))
    }

    public override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard bounds.width > 0, let context = NSGraphicsContext.current?.cgContext else { return }

        let textWidth = bounds.width - options.codeInsets.leading - options.codeInsets.trailing
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
