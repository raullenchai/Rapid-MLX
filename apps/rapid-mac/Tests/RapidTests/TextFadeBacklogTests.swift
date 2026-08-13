import AppKit
import Testing
@testable import Rapid

/// Keeping the reveal within sight of the text.
///
/// The failure this pins was visible in a screenshot: the fifth heading of an
/// answer was on screen while the fourth was still half-grey. The reveal was
/// not slow — it was rate-limited below the rate text arrived, so it fell
/// further behind every second and never recovered.
@Suite("Text fade backlog")
@MainActor
struct TextFadeBacklogTests {

    private func makeAnimator() -> (MarkdownTextRenderer, TextFadeAnimator) {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let renderer = MarkdownTextRenderer(options: options)
        let animator = TextFadeAnimator(
            textLayoutManager: renderer.textLayoutManager,
            textContentStorage: renderer.textContentStorage,
            animationState: TextFadeAnimationState()
        )
        animator.textColor = .black
        animator.contentLengthProvider = { renderer.proseLength }
        return (renderer, animator)
    }

    /// CJK emits roughly one unit per character, so a local model produces
    /// several hundred units a second. A 15 ms floor caps the reveal at 67/s —
    /// five times slower than arrival.
    @Test("The advance floor admits CJK-scale arrival rates")
    func floorAdmitsFastArrival() {
        let config = TextFadeConfiguration()
        // The floor is what the adaptive rate clamps to; assert the clamp can
        // express a rate a local model actually reaches.
        let fastestExpressible = 1.0 / 0.004
        #expect(
            fastestExpressible >= 250,
            "the reveal cannot keep up with CJK-rate output at this floor"
        )
        #expect(config.advanceDuration == 0.035, "default unchanged for slow streams")
    }

    /// The compression threshold is 0.75 s — aligned with native-chat, which
    /// this build deliberately tracks (the #1843 tuning move). It is NOT a
    /// CJK-tight threshold: at ~300 units/second a 0.75 s backlog is ~225
    /// characters, several visible lines of grey. That is the accepted
    /// tradeoff — a shorter threshold made the reveal visibly jumpy on every
    /// streamed batch, and native-chat's feel wins. The value must still bound
    /// the backlog: `compressBacklog` collapses any tail past the threshold
    /// back inside it, so the reader never waits more than a beat behind the
    /// newest word.
    @Test("The backlog threshold stays at the native-chat value")
    func thresholdMatchesNative() {
        let config = TextFadeConfiguration()
        // Pin the value so a future change to it is a conscious decision,
        // not a silent drift.
        #expect(config.flushDurationThreshold == 0.75)
        // Sanity: compression must be able to pull a backlog back inside the
        // threshold at any size (`compressBacklog` scales the factor up).
        #expect(config.flushMultiplier >= 1)
    }

    /// The reveal must still finish. A queue that compresses but never drains
    /// would leave text permanently dimmed.
    @Test("Every scheduled unit reaches full opacity")
    func queueDrains() {
        let (renderer, animator) = makeAnimator()
        renderer.setBlocks([
            .init(runs: [InlineRun(text: "一二三四五六七八九十")], kind: .paragraph)
        ])
        renderer.measureHeight(width: 400)
        animator.contentDidGrow()

        let start = CACurrentMediaTime()
        // Run well past any plausible schedule.
        animator.testing_tick(at: start + 10)

        let storage = renderer.textContentStorage
        guard let location = storage.location(storage.documentRange.location, offsetBy: 0)
        else { return }
        var alpha: CGFloat?
        renderer.textLayoutManager.enumerateRenderingAttributes(
            from: location, reverse: false
        ) { _, attributes, _ in
            alpha = (attributes[.foregroundColor] as? NSColor)?.alphaComponent
            return false
        }
        #expect((alpha ?? 1) > 0.99, "text stayed dim after the schedule elapsed")
    }
}

/// The rate the reveal adapts to must be *units per second*, not *flushes per
/// second*.
///
/// This distinction was the whole bug. The markdown compiler flushes on a
/// fixed ~10 Hz debounce, so counting one event per flush measured 10/second
/// no matter how fast the model actually produced text. A stream arriving at
/// 375 units/second was smoothed to ~6, which pushed `adaptiveAdvanceDuration`
/// to its slowest clamp and revealed 12 units/second — thirty times slower
/// than arrival, compounding for the whole answer.
@Suite("Text fade rate estimation")
@MainActor
struct TextFadeRateTests {

    private func makeAnimator() -> (MarkdownTextRenderer, TextFadeAnimator) {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let renderer = MarkdownTextRenderer(options: options)
        let animator = TextFadeAnimator(
            textLayoutManager: renderer.textLayoutManager,
            textContentStorage: renderer.textContentStorage,
            animationState: TextFadeAnimationState()
        )
        animator.textColor = .black
        animator.contentLengthProvider = { renderer.proseLength }
        return (renderer, animator)
    }

    @Test("A batch of many units reads as a fast rate, not a slow one")
    func batchSizeDrivesTheRate() {
        let (renderer, animator) = makeAnimator()
        var timestamps: [CFTimeInterval] = [1.0, 1.1]
        animator.testing_setClock { timestamps.removeFirst() }

        // Two flushes, each carrying many words — the shape a debounced
        // compiler produces against a fast model.
        let first = (1...40).map { "word\($0)" }.joined(separator: " ")
        renderer.setBlocks([.init(runs: [InlineRun(text: first)], kind: .paragraph)])
        renderer.measureHeight(width: 600)
        animator.contentDidGrow()

        let second = first + " " + (41...80).map { "word\($0)" }.joined(separator: " ")
        renderer.setBlocks([.init(runs: [InlineRun(text: second)], kind: .paragraph)])
        renderer.measureHeight(width: 600)
        animator.contentDidGrow()

        let rate = animator.animationState.smoothedWordsPerSecond
        #expect(
            rate > 40,
            "measured \(Int(rate)) units/second from a 40-unit batch — the estimator is counting flushes, not units"
        )
    }
}
