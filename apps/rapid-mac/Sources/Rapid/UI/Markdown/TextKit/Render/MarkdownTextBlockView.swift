import AppKit

/// Draws markdown text through TextKit 2.
///
/// A plain `NSView` rather than `NSTextView`: we need neither editing nor the
/// text system's own scrolling, and going one layer lower is what lets the
/// fade animator drive rendering attributes directly on the layout manager.
///
/// Height is a pure function of `(blocks, width, options)`, which is the
/// property the collection view relies on — an ambiguous intrinsic size here
/// shows up as rows jittering by a point during scroll.
final class MarkdownTextBlockView: NSView {

    private let renderer: MarkdownTextRenderer
    private var blocks: [MarkdownItem.TextBlock] = []
    private var options: MarkdownOptions

    /// Reveals newly-streamed text. Created lazily — a static transcript never
    /// needs one, and a display link that exists is a display link that can
    /// leak.
    private var fadeAnimator: TextFadeAnimator?

    /// Drives the typing dot's pulse. Separate from the fade animator's link
    /// because the dot keeps breathing while the model is thinking and the
    /// fade queue is empty.
    private var showsTypingDot = false
    /// Persistent layer for the typing dot. Moved rather than redrawn — see
    /// ``updateTypingDotLayer``.
    private var typingDotLayer: CAShapeLayer?

    public override var isFlipped: Bool { true }

    public init(options: MarkdownOptions) {
        self.options = options
        self.renderer = MarkdownTextRenderer(options: options)
        super.init(frame: .zero)
        wantsLayer = true
        setAccessibilityElement(true)
        setAccessibilityRole(.staticText)
        setAccessibilityEnabled(true)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    public func configure(blocks: [MarkdownItem.TextBlock], options: MarkdownOptions) {
        configure(blocks: blocks, options: options, streaming: false, fadeState: nil)
    }

    /// Configure, optionally animating text that is arriving now.
    ///
    /// `fadeState` is shared across every block of one message so the reveal
    /// runs on a single timeline — without it each block would restart the
    /// animation at its own boundary.
    public func configure(
        blocks: [MarkdownItem.TextBlock],
        options: MarkdownOptions,
        streaming: Bool,
        fadeState: TextFadeAnimationState?,
        fadeConfiguration: TextFadeConfiguration = TextFadeConfiguration()
    ) {
        let contentGrew = streaming && blocks != self.blocks
        // The dot answers "is anything happening?" — a question only worth
        // answering while there is nothing to read. Once the first words
        // arrive, the fade-in of each new word says the same thing better,
        // and a dot trailing the text is one more moving part competing with
        // the words for attention. ChatGPT gates it the same way: `streaming`
        // and `showsTypingDotWhenStreaming` are separate fields, so a stream
        // can run without one.
        let wantsDot = Self.showsTypingDot(streaming: streaming, blocks: blocks)
        let dotChanged = wantsDot != showsTypingDot
        self.blocks = blocks
        self.options = options
        self.showsTypingDot = wantsDot

        renderer.update(options: options)
        renderer.setBlocks(blocks, showsTypingDot: wantsDot)
        // A custom-drawn NSView has no automatic text semantics. Mirror the
        // backing text into AX so VoiceOver and GUI automation can read the
        // answer just as they could through MarkdownUI's native Text views.
        setAccessibilityValue(renderer.accessibleText)

        if streaming, fadeConfiguration.isEnabled, let fadeState {
            let animator: TextFadeAnimator
            if let existing = fadeAnimator {
                animator = existing
            } else {
                animator = TextFadeAnimator(
                    textLayoutManager: renderer.textLayoutManager,
                    textContentStorage: renderer.textContentStorage,
                    animationState: fadeState,
                    configuration: fadeConfiguration
                )
                animator.contentLengthProvider = { [weak self] in
                    self?.renderer.proseLength ?? 0
                }
                animator.attach(to: self)
                fadeAnimator = animator
            }
            animator.configuration = fadeConfiguration
            animator.textColor = options.textColor
            if contentGrew || dotChanged {
                animator.contentDidGrow()
            } else {
                // `setBlocks` above wiped the rendering attributes even though
                // nothing changed. Without this the fade never comes back.
                animator.storageDidReset()
            }
        } else if let animator = fadeAnimator {
            // Streaming ended: leave the text fully visible rather than
            // replaying the reveal on the next render pass.
            animator.markAllRevealed()
            animator.reset()
            fadeAnimator = nil
        }

        updateTypingDotAnimation()
        needsDisplay = true
        invalidateIntrinsicContentSize()
    }

    /// Whether the typing dot should be shown.
    ///
    /// Only while streaming AND before any words exist. Once text arrives its
    /// fade-in carries the "still generating" signal, and a dot trailing the
    /// words is a second moving thing competing with them for attention.
    /// ChatGPT gates it the same way — `streaming` and
    /// `showsTypingDotWhenStreaming` are separate fields on
    /// `MarkdownViewParameters`, so a stream can run without one.
    static func showsTypingDot(
        streaming: Bool, blocks: [MarkdownItem.TextBlock]
    ) -> Bool {
        guard streaming else { return false }
        let hasVisibleText = blocks.contains { block in
            block.runs.contains {
                !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
        }
        return !hasVisibleText
    }

    /// Place the dot after layout settles.
    ///
    /// No display link any more: the pulse lives in a `CAShapeLayer`
    /// animation, which Core Animation drives on its own thread. The previous
    /// version ticked a display link purely to call `needsDisplay` 30×/second
    /// so a hand-drawn circle could change opacity — that repainted the whole
    /// text view for one dot, and its rhythm wobbled with the stream's own
    /// redraws.
    private func updateTypingDotAnimation() {
        updateTypingDotLayer()
    }

    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        updateTypingDotAnimation()
        // The fade's display link can only bind once the view is on a screen,
        // and content routinely arrives before SwiftUI mounts the
        // representable. Re-arm here or the queue never drains.
        fadeAnimator?.hostViewDidMoveToWindow()
    }

    /// Measure without rendering. Cheap enough to call for every offscreen row.
    public func height(forWidth width: CGFloat) -> CGFloat {
        renderer.measureHeight(width: width)
    }

    /// The width the text actually occupies when laid out at `maxWidth`.
    ///
    /// Used by user bubbles, which must hug their content rather than fill the
    /// column. TextKit reports this directly, so no binary search is needed.
    public func naturalWidth(maxWidth: CGFloat) -> CGFloat {
        renderer.measureNaturalWidth(maxWidth: maxWidth)
    }

    public override var intrinsicContentSize: NSSize {
        let width = bounds.width
        guard width > 0 else { return NSSize(width: NSView.noIntrinsicMetric, height: 0) }
        return NSSize(width: NSView.noIntrinsicMetric, height: renderer.measureHeight(width: width))
    }

    public override func setFrameSize(_ newSize: NSSize) {
        let widthChanged = newSize.width != frame.width
        super.setFrameSize(newSize)
        if widthChanged {
            invalidateIntrinsicContentSize()
            needsDisplay = true
        }
    }

    public override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard bounds.width > 0 else { return }

        renderer.textContainer.size = CGSize(
            width: bounds.width, height: CGFloat.greatestFiniteMagnitude
        )
        renderer.textLayoutManager.ensureLayout(
            for: renderer.textContentStorage.documentRange
        )

        guard let context = NSGraphicsContext.current?.cgContext else { return }
        renderer.textLayoutManager.enumerateTextLayoutFragments(
            from: renderer.textLayoutManager.documentRange.location,
            options: [.ensuresLayout, .ensuresExtraLineFragment]
        ) { fragment in
            fragment.draw(at: fragment.layoutFragmentFrame.origin, in: context)
            return true
        }

        drawTextDecorations(in: context)
        drawBlockDecorations(in: context)

        // Reposition after layout: `rect(forCharacterAt:)` is only meaningful
        // once the fragments above have been laid out at this width.
        updateTypingDotLayer()
    }

    public override func resetCursorRects() {
        super.resetCursorRects()
        // Ask the storage where the links ARE, rather than asking every
        // character whether it is one.
        //
        // The previous shape walked all `proseLength` offsets and called
        // `renderer.link(at:)` on each, and that call is itself a linear scan
        // over the same offsets — O(n²) with an `ensureLayout` inside the inner
        // loop. AppKit re-runs `resetCursorRects` after every layout and every
        // scroll, so opening a 6 000-character answer put 86% of the main
        // thread in this one method (sampled). Enumerating the `.link`
        // attribute visits only the ranges that actually carry a link.
        for rect in renderer.linkRects() {
            addCursorRect(rect, cursor: .pointingHand)
        }
    }

    public override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        guard let url = renderer.link(at: point) else {
            super.mouseDown(with: event)
            return
        }
        // This AppKit leaf cannot consume SwiftUI's OpenURLAction environment,
        // so apply the same central policy before handing off to the system.
        if case .allowed(let safeURL) = ChatLinkSafety.decide(url) {
            NSWorkspace.shared.open(safeURL)
        }
    }

    /// Position the typing dot's layer at the spot the text system reserved.
    ///
    /// A persistent `CAShapeLayer` that is *moved*, not a circle redrawn every
    /// frame. Redrawing it from `draw(_:)` meant re-reading the last glyph's
    /// rect on every token, so the dot jittered with each measurement and
    /// jumped whenever the text rewrapped. ChatGPT does the same thing —
    /// `TypingDotPostProcessState` holds a `TypingDotView` with a
    /// `CAShapeLayer`, so the pulse animates in its own layer and never
    /// participates in text layout.
    ///
    /// Movement is wrapped in a disabled implicit-animation transaction: a
    /// layer's `position` animates by default, and a 0.25s slide toward every
    /// new glyph is exactly the drifting the dot is meant to avoid.
    private func updateTypingDotLayer() {
        guard showsTypingDot,
              let location = renderer.typingDotLocation,
              let rect = renderer.rect(forCharacterAt: location) else {
            typingDotLayer?.removeFromSuperlayer()
            typingDotLayer = nil
            return
        }

        let size = TypingDotAttachment.diameter
        let layer: CAShapeLayer
        if let existing = typingDotLayer {
            layer = existing
        } else {
            layer = CAShapeLayer()
            layer.path = CGPath(
                ellipseIn: CGRect(x: 0, y: 0, width: size, height: size),
                transform: nil
            )
            layer.bounds = CGRect(x: 0, y: 0, width: size, height: size)
            // Pulse in the layer, not in `draw(_:)`. Core Animation runs this
            // off the main thread, so it keeps a steady rhythm no matter how
            // busy the text pipeline is — the old version's opacity was
            // sampled per draw, so its "pulse" ran at whatever irregular rate
            // the stream happened to redraw at.
            let pulse = CABasicAnimation(keyPath: "opacity")
            pulse.fromValue = 1.0
            pulse.toValue = TypingDotAttachment.minimumOpacity
            pulse.duration = TypingDotAttachment.pulseDuration / 2
            pulse.autoreverses = true
            pulse.repeatCount = .infinity
            pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            layer.add(pulse, forKey: "pulse")
            self.layer?.addSublayer(layer)
            typingDotLayer = layer
        }

        layer.fillColor = options.textColor.cgColor
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        layer.position = CGPoint(x: rect.minX + size / 2, y: rect.midY)
        CATransaction.commit()
    }

    /// Draw strikethrough and underline.
    ///
    /// `NSTextLayoutFragment.draw(at:in:)` renders glyphs but **not** the
    /// decoration attributes — strikethrough and underline are absent from its
    /// output. That is easy to miss: the text looks right, and only a specific
    /// span is silently missing a line through it.
    ///
    /// Rather than abandon fragment drawing (which is what gives the fade
    /// animator its per-range control), we walk the decorated ranges and
    /// stroke them ourselves, using the layout manager's own segment geometry
    /// so the lines follow wrapping correctly.
    private func drawTextDecorations(in context: CGContext) {
        guard let storage = renderer.textContentStorage.textStorage else { return }
        let full = NSRange(location: 0, length: storage.length)

        for key in [NSAttributedString.Key.strikethroughStyle, .underlineStyle] {
            storage.enumerateAttribute(key, in: full) { value, nsRange, _ in
                guard let raw = value as? Int, raw != 0 else { return }
                guard let textRange = self.textRange(from: nsRange) else { return }

                let color = (storage.attribute(.foregroundColor, at: nsRange.location,
                                               effectiveRange: nil) as? NSColor)
                    ?? self.options.textColor
                let font = storage.attribute(.font, at: nsRange.location,
                                             effectiveRange: nil) as? NSFont

                context.setFillColor(color.cgColor)
                renderer.textLayoutManager.enumerateTextSegments(
                    in: textRange, type: .standard, options: []
                ) { _, frame, baselineOffset, _ in
                    let resolved = font ?? .systemFont(ofSize: self.options.textPointSize)
                    let thickness = max(1, resolved.pointSize / 14)
                    // KNOWN ISSUE: vertical placement is approximate.
                    //
                    // `frame` includes leading (lineHeightMultiple is 1.35), so
                    // frame.midY draws across the top of the glyphs. Deriving
                    // from `baselineOffset` instead put the line outside the
                    // visible band entirely. Neither reading of the segment
                    // geometry has been right yet, and this is a minor
                    // decoration, so it is parked at a value that is visible
                    // and close rather than chased further.
                    //
                    // Worth revisiting alongside the fade animator, which needs
                    // exact per-range geometry anyway.
                    let y: CGFloat = key == .strikethroughStyle
                        ? frame.minY + frame.height * 0.55
                        : frame.minY + frame.height * 0.82
                    context.fill(CGRect(x: frame.minX, y: y,
                                        width: frame.width, height: thickness))
                    return true
                }
            }
        }
    }

    /// Convert an `NSRange` in the backing storage to a TextKit 2 range.
    private func textRange(from nsRange: NSRange) -> NSTextRange? {
        let storage = renderer.textContentStorage
        guard let start = storage.location(storage.documentRange.location,
                                           offsetBy: nsRange.location),
              let end = storage.location(start, offsetBy: nsRange.length) else { return nil }
        return NSTextRange(location: start, end: end)
    }

    /// Block-quote bars and horizontal rules — decorations the text system
    /// does not draw. ChatGPT keeps the same split: its renderer carries
    /// explicit `blockQuotes` and `customUnderlines` arrays alongside the text.
    ///
    /// Positions come from the layout manager's own segment geometry rather
    /// than from re-measuring each block in isolation. The earlier version did
    /// the latter and put the quote bar in the wrong place: measuring a block
    /// alone gives a different height than it occupies in context, because
    /// paragraph spacing collapses between neighbours.
    private func drawBlockDecorations(in context: CGContext) {
        guard blocks.contains(where: { $0.kind == .horizontalRule || $0.kind == .blockQuote })
        else { return }
        guard let storage = renderer.textContentStorage.textStorage else { return }

        // Walk the blocks alongside their ranges in the backing string, so a
        // decoration lands on the text it belongs to.
        var location = 0
        for (index, block) in blocks.enumerated() {
            if index > 0 { location += 1 }  // the "\n" joining blocks
            let text = renderer.attributedString(for: block).string
            let length = (text as NSString).length
            let nsRange = NSRange(location: location, length: length)
            location += length

            guard nsRange.location + nsRange.length <= storage.length,
                  block.kind == .horizontalRule || block.kind == .blockQuote,
                  let textRange = self.textRange(from: nsRange) else { continue }

            var bounds = CGRect.null
            renderer.textLayoutManager.enumerateTextSegments(
                in: textRange, type: .standard, options: []
            ) { _, frame, _, _ in
                bounds = bounds.isNull ? frame : bounds.union(frame)
                return true
            }
            guard !bounds.isNull else { continue }

            switch block.kind {
            case .horizontalRule:
                let color = options.horizontalRuleColor ?? NSColor.separatorColor
                context.setFillColor(color.cgColor)
                context.fill(CGRect(
                    x: options.horizontalRuleInsets.leading,
                    y: bounds.midY,
                    width: self.bounds.width - options.horizontalRuleInsets.leading
                        - options.horizontalRuleInsets.trailing,
                    height: options.horizontalRuleHeight
                ))
            case .blockQuote:
                let color = options.blockQuoteBarColor ?? NSColor.separatorColor
                context.setFillColor(color.cgColor)
                context.fill(CGRect(
                    x: max(0, options.blockQuoteLeadingInset - options.blockQuoteBarWidth - 6),
                    y: bounds.minY,
                    width: options.blockQuoteBarWidth,
                    height: bounds.height
                ))
            default:
                break
            }
        }
    }

}
