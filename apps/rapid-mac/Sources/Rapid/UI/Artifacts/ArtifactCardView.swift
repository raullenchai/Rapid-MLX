import AppKit

/// The inline stand-in for an HTML artifact: title, size and a button that
/// opens the panel.
///
/// Deliberately static. A live page lives in the panel, one at a time, so a
/// long transcript never spawns a web process per block.
final class ArtifactCardView: NSView {

    static let height: CGFloat = 64

    /// Nil hides the button: outside a chat transcript there is no panel.
    var onOpen: (@MainActor () -> Void)? {
        didSet { openButton.isHidden = onOpen == nil }
    }

    private let iconView = NSImageView()
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(labelWithString: "")
    private let openButton = NSButton()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.borderWidth = 1

        iconView.image = NSImage(systemSymbolName: "globe", accessibilityDescription: nil)
        iconView.contentTintColor = .secondaryLabelColor
        iconView.translatesAutoresizingMaskIntoConstraints = false

        titleLabel.lineBreakMode = .byTruncatingTail
        titleLabel.textColor = .labelColor
        titleLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        detailLabel.lineBreakMode = .byTruncatingTail
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.translatesAutoresizingMaskIntoConstraints = false

        openButton.title = String(localized: "Open in panel")
        openButton.bezelStyle = .rounded
        openButton.controlSize = .small
        openButton.target = self
        openButton.action = #selector(open)
        openButton.isHidden = true
        openButton.setAccessibilityIdentifier("Artifact.Card.Open")
        openButton.translatesAutoresizingMaskIntoConstraints = false
        openButton.setContentHuggingPriority(.required, for: .horizontal)
        openButton.setContentCompressionResistancePriority(.required, for: .horizontal)

        for view in [iconView, titleLabel, detailLabel, openButton] { addSubview(view) }

        NSLayoutConstraint.activate([
            iconView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            iconView.centerYAnchor.constraint(equalTo: centerYAnchor),
            iconView.widthAnchor.constraint(equalToConstant: 20),
            iconView.heightAnchor.constraint(equalToConstant: 20),

            titleLabel.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: 10),
            titleLabel.bottomAnchor.constraint(equalTo: centerYAnchor, constant: -1),
            titleLabel.trailingAnchor.constraint(
                lessThanOrEqualTo: openButton.leadingAnchor, constant: -10)
                .withPriority(.defaultHigh),

            detailLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            detailLabel.topAnchor.constraint(equalTo: centerYAnchor, constant: 1),
            detailLabel.trailingAnchor.constraint(
                lessThanOrEqualTo: openButton.leadingAnchor, constant: -10)
                .withPriority(.defaultHigh),

            openButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            openButton.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
        applyAppearanceDependentColors()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func configure(item: ArtifactItem) {
        titleLabel.stringValue = item.title
        detailLabel.stringValue = String(localized: "HTML · \(item.sizeDescription)")
        titleLabel.font = .systemFont(
            ofSize: UIFontSettings.shared.pointSize(13), weight: .semibold)
        detailLabel.font = MarkdownCodeBlockView.headerFont
        openButton.setAccessibilityLabel(
            String(localized: "Open \(item.title) in panel"))
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        applyAppearanceDependentColors()
    }

    // A layer takes a resolved CGColor, so re-resolve on every appearance flip.
    private func applyAppearanceDependentColors() {
        effectiveAppearance.performAsCurrentDrawingAppearance {
            layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            layer?.borderColor = NSColor.separatorColor.cgColor
        }
    }

    @objc private func open() { onOpen?() }
}

private extension NSLayoutConstraint {
    /// Below required, so a not-yet-sized card does not log a conflict.
    func withPriority(_ priority: NSLayoutConstraint.Priority) -> NSLayoutConstraint {
        self.priority = priority
        return self
    }
}
