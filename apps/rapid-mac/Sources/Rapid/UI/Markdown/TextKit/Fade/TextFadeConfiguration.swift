import AppKit
import CoreGraphics

/// Tuning for the streaming text fade.
///
/// Field names and types come from ChatGPT's `TextFadeAnimator.Configuration`,
/// recovered with byte offsets. The **values** were all optimised away, so
/// each one below is a calibration with its reasoning attached — none of them
/// is a recovered constant, and none should be read as one.
struct TextFadeConfiguration: Sendable {

    /// Master switch. Off means text appears instantly, with byte-identical
    /// layout — the animator only ever touches rendering attributes.
    public var isEnabled: Bool = true

    /// Tint newly-arrived text toward an accent, then decay it back.
    ///
    /// Subtle enough to be easy to dismiss, but it is what makes the leading
    /// edge of a stream feel alive rather than merely faded.
    public var textFadeAccentColorEnabled: Bool = true

    /// Granularity of reveal.
    ///
    /// `.character` reads as a typewriter, which is a different (and dated)
    /// effect. `.line` reads as blocks popping in. Word-level is what reads as
    /// "text materialising", and it is what ChatGPT ships.
    public var advanceBy: AdvanceUnit = .word

    /// Skip re-issuing rendering attributes when alpha has not visibly moved.
    /// See `renderingBucket` in the animator.
    public var optimizedRenderingUpdatesEnabled: Bool = true

    /// How long one unit takes to go from invisible to fully opaque.
    ///
    /// Calibration method: screen-record ChatGPT at 60fps, extract frames,
    /// sample one word's pixel alpha across them, count frames from
    /// first-visible to fully-shown, divide by 60. 0.28s is the starting
    /// estimate.
    public var animationDuration: CFTimeInterval = 0.28

    /// Gap between successive units starting their fade.
    ///
    /// ≈28 words/second. This has to slightly exceed the model's word rate or
    /// the animation is permanently flushing; too far above and it lags
    /// visibly behind the text that has already arrived. Healthy steady state
    /// is 6-12 parts in flight — worth logging while tuning.
    public var advanceDuration: CFTimeInterval = 0.035

    /// Maximum acceptable visual lag behind the model. Past this the UI reads
    /// as slow rather than as animated.
    public var flushDurationThreshold: CFTimeInterval = 0.75

    /// When the backlog exceeds the threshold, compress `advanceDuration` by
    /// this factor rather than dumping the queue. A hard flush is a visible
    /// jolt; speeding up is not.
    public var flushMultiplier: Double = 2.5

    public var frameRate: Float = 120
    public var minimumFrameRate: Float = 30

    /// Easing. `.standard` is easeOutQuad, `.snappy` is easeOutCubic — the two
    /// cases ChatGPT's `AnimationType` enum carries.
    public var animationType: AnimationType = .standard

    /// Haptic feedback on the first N units.
    ///
    /// Zero by design. macOS haptics only fire on a Force Touch trackpad, and
    /// per-word feedback would be obnoxious where it does. The field exists so
    /// the shape matches the original.
    public var hapticFadeInCount: Int = 0
    public var hapticFadeOutIntensity: ClosedRange<Float> = 0.0...0.35

    /// How long the accent tint takes to decay back to the text colour.
    public var accentDecayDuration: CFTimeInterval = 0.6

    public init() {}

    public enum AdvanceUnit: Sendable {
        case character, word, line
    }

    public enum AnimationType: Sendable {
        case standard, snappy

        /// Progress curve, t in 0...1.
        func ease(_ t: Double) -> Double {
            switch self {
            case .standard: 1 - pow(1 - t, 2)
            case .snappy: 1 - pow(1 - t, 3)
            }
        }
    }

    /// Disabled preset — text appears instantly.
    public static let off: TextFadeConfiguration = {
        var c = TextFadeConfiguration()
        c.isEnabled = false
        return c
    }()
}

/// Shared fade state for one message.
///
/// Text and code blocks in the same message have to advance on one timeline,
/// or the reveal restarts at every block boundary. ChatGPT threads a
/// `TextFadeAnimationState` through `MarkdownBlockStack` for the same reason.
@MainActor
final class TextFadeAnimationState {
    /// Word-arrival rate, exponentially smoothed. Drives the adaptive
    /// `advanceDuration` — see ``TextFadeAnimator/adaptiveAdvanceDuration``.
    public internal(set) var smoothedWordsPerSecond: Double = 0
    /// When the accent tint began decaying.
    public internal(set) var accentDecayStartTime: CFTimeInterval?

    public init() {}

    public func reset() {
        smoothedWordsPerSecond = 0
        accentDecayStartTime = nil
    }
}
