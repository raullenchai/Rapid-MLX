import AppKit
import QuartzCore

/// Fades newly-streamed text in, word by word.
///
/// The entire design rests on one API: `NSTextLayoutManager.setRenderingAttributes(_:for:)`
/// changes how glyphs are *drawn* without invalidating layout. A fade
/// therefore never changes a measured height, never reflows, and never
/// disturbs the collection view. That is why this approach is cheap and why
/// the SwiftUI equivalent — re-rendering `Text` with a changing opacity — is
/// not: there, every frame is a layout pass.
///
/// It is also why the renderer had to be ours. `NSTextRange` addressing is the
/// requirement that ruled out swift-markdown-ui at the start of phase 3.
@MainActor
final class TextFadeAnimator {

    /// One unit of text being revealed.
    ///
    /// Held as a plain `NSRange`, not an `NSTextRange`. The renderer replaces
    /// the entire text storage on every markdown flush (~100ms), which
    /// invalidates every live `NSTextLocation` and drops all rendering
    /// attributes. Integer offsets survive that; they are re-resolved into
    /// `NSTextRange` each frame.
    struct TextPart {
        let range: NSRange
        var startTime: CFTimeInterval
        /// Last alpha bucket written, so we can skip redundant attribute
        /// updates. See `optimizedRenderingUpdatesEnabled`.
        var lastAppliedBucket: Int?
    }

    public var configuration: TextFadeConfiguration
    public let animationState: TextFadeAnimationState

    private let textLayoutManager: NSTextLayoutManager
    private let textContentStorage: NSTextContentStorage
    private let displayLink = ClosureDisplayLink()

    private var fadingParts: [TextPart] = []
    /// Character offset up to which text has been scheduled — new content
    /// starts here.
    private var scheduledLength = 0
    private var lastGrowthTime: CFTimeInterval?
    /// Injectable monotonic clock. Production uses Core Animation's clock;
    /// tests pin it so rate estimation does not depend on scheduler latency.
    private var now: () -> CFTimeInterval = CACurrentMediaTime

    /// Colour newly-arrived text is tinted toward before decaying.
    public var accentColor: NSColor = .controlAccentColor
    /// Resting colour, used when a fade completes.
    public var textColor: NSColor = .textColor

    private weak var hostView: NSView?

    public init(
        textLayoutManager: NSTextLayoutManager,
        textContentStorage: NSTextContentStorage,
        animationState: TextFadeAnimationState,
        configuration: TextFadeConfiguration = TextFadeConfiguration()
    ) {
        self.textLayoutManager = textLayoutManager
        self.textContentStorage = textContentStorage
        self.animationState = animationState
        self.configuration = configuration
    }

    public func attach(to view: NSView) {
        hostView = view
    }

    /// Reset to "everything is already visible".
    ///
    /// Called when a message finishes or the transcript changes, so a
    /// re-render does not replay the whole reveal.
    public func reset() {
        displayLink.stop()
        fadingParts.removeAll()
        scheduledLength = 0
        lastGrowthTime = nil
        animationState.reset()
        clearRenderingAttributes()
    }

    /// Mark everything currently in the document as already revealed, without
    /// animating it. Used when restoring history: replaying a fade over an old
    /// conversation would be absurd.
    public func markAllRevealed() {
        fadingParts.removeAll()
        scheduledLength = documentLength
        clearRenderingAttributes()
    }

    /// Note that the document grew, and schedule the new text to fade in.
    /// The renderer replaced the text storage without the document growing.
    ///
    /// `setBlocks` unconditionally calls `setAttributedString`, which drops
    /// every rendering attribute — including the alphas of parts still mid-
    /// fade. `contentDidGrow` already re-applies them from scratch for that
    /// reason, but it only runs when the content actually changed. A render
    /// pass that re-configures with identical blocks wipes the alphas and
    /// nothing puts them back: the bucket cache still holds the value it
    /// believes is on screen, so the next display-link tick sees no change and
    /// skips the write. Mid-fade text then snaps to full opacity and stays
    /// there.
    ///
    /// That is the common case here, not the rare one. `MessageRow` takes the
    /// raw `ChatMessage`, so every SSE delta (~16 ms) re-renders it, while the
    /// blocks only change on the compiler's 100 ms beat — five out of six
    /// passes wipe without restoring. native-chat avoids it structurally by
    /// handing its row a compiled, `Equatable` view model keyed on the
    /// compile revision, so SwiftUI skips the pass entirely.
    func storageDidReset() {
        guard configuration.isEnabled, !fadingParts.isEmpty else { return }
        for index in fadingParts.indices {
            fadingParts[index].lastAppliedBucket = nil
        }
        applyPendingAlphaZero()
        startDisplayLinkIfNeeded()
    }

    public func contentDidGrow() {
        guard configuration.isEnabled else {
            markAllRevealed()
            return
        }

        // The renderer replaced the text storage, so every rendering attribute
        // we wrote is gone. Parts still in flight must be re-applied from
        // scratch — otherwise the bucket cache would suppress the write and
        // half-faded text would snap to full opacity.
        for index in fadingParts.indices {
            fadingParts[index].lastAppliedBucket = nil
        }

        let length = documentLength
        // Recompilation can shorten the document (a fence closing rewrites the
        // block). Re-schedule from wherever it now ends.
        guard length > scheduledLength else {
            scheduledLength = min(scheduledLength, length)
            return
        }

        let newRange = NSRange(location: scheduledLength, length: length - scheduledLength)
        let now = now()

        let units = enumerateUnits(in: newRange)
        scheduledLength = length
        guard !units.isEmpty else { return }
        // After enumeration: the rate is units per second, so it needs the
        // count this batch actually carried.
        updateWordRate(at: now, unitCount: units.count)

        // Anchor the stagger after whatever is already queued, so a flush
        // carrying twenty words does not start them all at once.
        let queuedTail = fadingParts.map(\.startTime).max() ?? now
        var cursor = max(now, queuedTail)
        let advance = adaptiveAdvanceDuration
        for unit in units {
            cursor += advance
            fadingParts.append(TextPart(range: unit, startTime: cursor))
        }

        // If the stagger tail already exceeds what a reader will tolerate,
        // compress the whole queue rather than dumping it — a hard flush is a
        // visible jolt.
        if cursor - now > configuration.flushDurationThreshold {
            compressBacklog(now: now)
        }

        if animationState.accentDecayStartTime == nil {
            animationState.accentDecayStartTime = now
        }

        // Hide the new text immediately. Without this it renders at full
        // opacity until its start time arrives — a pop-in, which is the exact
        // artefact the fade exists to remove.
        applyPendingAlphaZero()
        startDisplayLinkIfNeeded()
    }

    private var documentLength: Int {
        // Excludes the typing dot when one is present. The dot is appended and
        // removed on every flush; scheduling it as a fade unit would queue a
        // range that no longer exists a frame later.
        contentLengthProvider?() ?? (textContentStorage.textStorage?.length ?? 0)
    }

    /// Supplies the length of the fadeable prose, excluding trailing
    /// decoration. Set by the renderer's owner.
    public var contentLengthProvider: (() -> Int)?

    // MARK: - Rate adaptation

    /// `advanceDuration`, adjusted to the observed word rate.
    ///
    /// ChatGPT ships a constant because its server rate is predictable. A
    /// local model's is not: 8 tok/s on a large model starves the animation,
    /// 200 tok/s on a tiny one keeps it permanently flushing. Deviating from
    /// the original is correct here.
    private var adaptiveAdvanceDuration: CFTimeInterval {
        let rate = animationState.smoothedWordsPerSecond
        guard rate > 0.5 else { return configuration.advanceDuration }
        // Floor at 4 ms (250 units/second).
        //
        // This was 15 ms — 67 units/second — which was set against an English
        // estimate of ~120 tokens/second where a "unit" is a whole word. CJK
        // has no spaces, so the segmenter emits roughly one unit per
        // character, and the same engine produces several hundred units per
        // second. The reveal then ran 5× slower than the text arrived and fell
        // permanently behind: by the time the fifth heading was on screen the
        // fourth was still half-grey.
        //
        // The clamp exists to stop a rate spike collapsing the animation to
        // nothing, not to cap throughput. 4 ms is under one frame at 120 Hz,
        // so each unit remains individually visible.
        return min(max(1.0 / rate, 0.004), 0.08)
    }

    private func updateWordRate(at now: CFTimeInterval, unitCount: Int) {
        defer { lastGrowthTime = now }
        guard let last = lastGrowthTime else { return }
        let elapsed = now - last
        guard elapsed > 0.001, unitCount > 0 else { return }
        // Units per second, not calls per second.
        //
        // This counted 1 per `contentDidGrow`, which measures how often the
        // markdown compiler flushes — a fixed ~10 Hz — rather than how much
        // text those flushes carry. A stream delivering 375 units/second in
        // 10 batches was therefore measured as 10/second and smoothed to ~6,
        // so `adaptiveAdvanceDuration` clamped to its 0.08 s CEILING and
        // revealed 12 units/second against 375 arriving. The queue fell
        // behind by a factor of thirty and stayed there: on screen, the fifth
        // heading of an answer was fully drawn while the fourth was still
        // half-grey.
        //
        // Dividing the batch across the interval that produced it measures the
        // arrival rate the reveal actually has to match.
        let instantaneous = Double(unitCount) / elapsed
        // EWMA over roughly a 2-second window.
        let alpha = 0.15
        animationState.smoothedWordsPerSecond =
            animationState.smoothedWordsPerSecond == 0
            ? instantaneous
            : animationState.smoothedWordsPerSecond * (1 - alpha) + instantaneous * alpha
    }

    /// Pull the queue's tail back inside `flushDurationThreshold`.
    ///
    /// `flushMultiplier` is the *minimum* compression, not the only one. A
    /// fixed divide is fine for a small overrun but useless for a large one: a
    /// 200-word paste schedules a 7s tail, and dividing by 2.5 still leaves
    /// 2.8s — four times the lag the threshold exists to bound. Scaling to the
    /// threshold keeps the invariant at any backlog size.
    ///
    /// Compression shortens the gaps *between* starts, never the fade itself,
    /// so a heavy burst reads as text arriving quickly rather than as text
    /// appearing all at once.
    private func compressBacklog(now: CFTimeInterval) {
        guard let tail = fadingParts.map(\.startTime).max() else { return }
        let overrun = tail - now
        guard overrun > configuration.flushDurationThreshold else { return }

        let required = overrun / configuration.flushDurationThreshold
        let factor = max(configuration.flushMultiplier, required)
        for index in fadingParts.indices {
            let delay = fadingParts[index].startTime - now
            guard delay > 0 else { continue }
            fadingParts[index].startTime = now + delay / factor
        }
    }

    // MARK: - Unit enumeration

    private func enumerateUnits(in nsRange: NSRange) -> [NSRange] {
        guard let storage = textContentStorage.textStorage else { return [nsRange] }

        switch configuration.advanceBy {
        case .line:
            return [nsRange]
        case .character, .word:
            let options: NSString.EnumerationOptions =
                configuration.advanceBy == .word ? .byWords : .byComposedCharacterSequences
            var units: [NSRange] = []
            let text = storage.string as NSString
            // Enumerate substrings, but emit ranges that include the trailing
            // whitespace: revealing words while their separators appear
            // instantly produces a visible shimmer between them.
            var cursor = nsRange.location
            text.enumerateSubstrings(in: nsRange, options: options) { _, sub, _, _ in
                let end = sub.location + sub.length
                guard end > cursor else { return }
                units.append(NSRange(location: cursor, length: end - cursor))
                cursor = end
            }
            let upperBound = nsRange.location + nsRange.length
            if cursor < upperBound {
                units.append(NSRange(location: cursor, length: upperBound - cursor))
            }
            return units.isEmpty ? [nsRange] : units
        }
    }

    // MARK: - Frame loop

    /// Retry starting the frame loop after the host view joins a window.
    ///
    /// Content usually arrives before SwiftUI mounts the representable, so the
    /// first `startDisplayLinkIfNeeded()` runs off-window and declines. Without
    /// this the queue never drains and every fade lands fully opaque.
    func hostViewDidMoveToWindow() {
        guard !fadingParts.isEmpty else { return }
        startDisplayLinkIfNeeded()
    }

    private func startDisplayLinkIfNeeded() {
        guard !displayLink.isRunning, let view = hostView else { return }
        displayLink.start(
            in: view,
            preferredFrameRate: configuration.frameRate,
            minimumFrameRate: configuration.minimumFrameRate
        ) { [weak self] timestamp in
            self?.tick(at: timestamp)
        }
    }

    private func tick(at now: CFTimeInterval) {

        guard !fadingParts.isEmpty else {
            displayLink.stop()
            return
        }

        var needsRedraw = false
        for index in fadingParts.indices {
            let part = fadingParts[index]
            let elapsed = now - part.startTime

            // Not started yet: hold it invisible. Re-asserted every frame
            // because a markdown flush wipes rendering attributes.
            guard elapsed >= 0 else {
                if fadingParts[index].lastAppliedBucket != 0 {
                    fadingParts[index].lastAppliedBucket = 0
                    apply(alpha: 0, to: part.range, now: now)
                    needsRedraw = true
                }
                continue
            }

            let raw = min(1, elapsed / configuration.animationDuration)
            let alpha = configuration.animationType.ease(raw)

            if configuration.optimizedRenderingUpdatesEnabled {
                let bucket = renderingBucket(for: alpha)
                if fadingParts[index].lastAppliedBucket == bucket { continue }
                fadingParts[index].lastAppliedBucket = bucket
            }
            apply(alpha: alpha, to: part.range, now: now)
            needsRedraw = true
        }

        // Drop finished parts. Their final write left them at full opacity,
        // which is also the default, so nothing has to be restored.
        fadingParts.removeAll { now - $0.startTime > configuration.animationDuration }

        if fadingParts.isEmpty {
            displayLink.stop()
        }
        if needsRedraw { hostView?.needsDisplay = true }
    }

    /// Paint everything not yet started as fully transparent.
    ///
    /// Called the moment new text is scheduled, before the next display-link
    /// frame — text inserted by the compiler is otherwise visible at full
    /// opacity for up to one frame.
    private func applyPendingAlphaZero() {
        let now = CACurrentMediaTime()
        for index in fadingParts.indices where fadingParts[index].startTime > now {
            fadingParts[index].lastAppliedBucket = 0
            apply(alpha: 0, to: fadingParts[index].range, now: now)
        }
    }

    /// Quantise alpha so a frame that moved it imperceptibly does not cost an
    /// attribute write. 32 buckets is below the eye's threshold at these
    /// durations.
    private func renderingBucket(for alpha: Double) -> Int {
        Int(alpha * 32)
    }

    private func apply(alpha: Double, to range: NSRange, now: CFTimeInterval) {
        guard let textRange = textRange(from: range) else { return }
        let colour = resolvedColor(alpha: alpha, now: now)
        textLayoutManager.setRenderingAttributes(
            [.foregroundColor: colour], for: textRange
        )
    }

    private func resolvedColor(alpha: Double, now: CFTimeInterval) -> NSColor {
        guard configuration.textFadeAccentColorEnabled,
              let decayStart = animationState.accentDecayStartTime else {
            return textColor.withAlphaComponent(alpha)
        }

        let decayProgress = min(1, (now - decayStart) / configuration.accentDecayDuration)
        let accentFraction = (1 - decayProgress) * 0.18
        guard accentFraction > 0.001 else { return textColor.withAlphaComponent(alpha) }

        // Blend the *hue* at full opacity, then apply the fade alpha.
        //
        // `blended(withFraction:of:)` mixes alpha as well as colour, so
        // blending a transparent base with an opaque accent hands back a
        // partially opaque result — a word scheduled to be invisible rendered
        // at 18%, which is a pop-in at the leading edge of every stream. Tint
        // first, fade second.
        let tinted = textColor.withAlphaComponent(1)
            .blended(withFraction: accentFraction, of: accentColor) ?? textColor
        return tinted.withAlphaComponent(alpha)
    }

    private func clearRenderingAttributes() {
        textLayoutManager.setRenderingAttributes([:], for: textContentStorage.documentRange)
    }

    // MARK: - Range conversion

    /// Resolve a character range against the *current* text storage.
    ///
    /// Returns nil if the range no longer fits — recompilation can shorten the
    /// document, and addressing past its end traps.
    private func textRange(from nsRange: NSRange) -> NSTextRange? {
        let storage = textContentStorage
        guard nsRange.length > 0,
              nsRange.location + nsRange.length <= documentLength,
              let start = storage.location(storage.documentRange.location,
                                           offsetBy: nsRange.location),
              let end = storage.location(start, offsetBy: nsRange.length) else { return nil }
        return NSTextRange(location: start, end: end)
    }

    // MARK: - Test hooks

    var testing_partCount: Int { fadingParts.count }
    var testing_scheduledLength: Int { scheduledLength }
    var testing_partRanges: [NSRange] { fadingParts.map(\.range) }
    var testing_startTimes: [CFTimeInterval] { fadingParts.map(\.startTime) }

    /// Advance the frame loop with an explicit clock, bypassing the display
    /// link. Lets the timeline be tested without racing a real stream.
    func testing_tick(at time: CFTimeInterval) { tick(at: time) }
    func testing_setClock(_ clock: @escaping () -> CFTimeInterval) { now = clock }
}
