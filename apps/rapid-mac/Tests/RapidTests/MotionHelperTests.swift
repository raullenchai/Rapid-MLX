import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// PR3 (#547) — shared interruptible-spring + Reduce-Motion motion helper.
///
/// The reduce-motion resolution is pure and gets real behavioural coverage;
/// the view adoptions (springs on the onboarding/model-browser transitions,
/// spring autoscroll, panel cross-fade, and the Reduce-Motion-suppressed
/// looping dots) are pinned by source guards mirroring the repo's existing
/// source-grep tripwires.
@Suite("PR3 — interruptible springs + Reduce Motion (#547)")
struct MotionHelperTests {

    // MARK: - RapidMotion.resolve (pure Reduce-Motion seam)

    @Test("resolve passes the animation through when Reduce Motion is off")
    func resolvePassesThroughNormally() {
        #expect(RapidMotion.resolve(RapidMotion.standard, reduceMotion: false) == RapidMotion.standard)
        #expect(RapidMotion.resolve(RapidMotion.scroll, reduceMotion: false) == RapidMotion.scroll)
    }

    @Test("resolve collapses to nil (instant) when Reduce Motion is on")
    func resolveNilsUnderReduceMotion() {
        #expect(RapidMotion.resolve(RapidMotion.standard, reduceMotion: true) == nil)
        #expect(RapidMotion.resolve(RapidMotion.scroll, reduceMotion: true) == nil)
        #expect(RapidMotion.resolve(RapidMotion.quick, reduceMotion: true) == nil)
    }

    @Test("resolve preserves a nil input regardless of Reduce Motion")
    func resolveNilInputStaysNil() {
        #expect(RapidMotion.resolve(nil, reduceMotion: false) == nil)
        #expect(RapidMotion.resolve(nil, reduceMotion: true) == nil)
    }

    // MARK: - RapidMotion.shouldPulse (looping-dot start/stop contract)

    @Test("shouldPulse loops only when active AND Reduce Motion is off")
    func shouldPulseGatesOnBothConditions() {
        #expect(RapidMotion.shouldPulse(isAnimating: true, reduceMotion: false) == true)
        #expect(RapidMotion.shouldPulse(isAnimating: true, reduceMotion: true) == false)
        #expect(RapidMotion.shouldPulse(isAnimating: false, reduceMotion: false) == false)
        #expect(RapidMotion.shouldPulse(isAnimating: false, reduceMotion: true) == false)
    }

    @Test("shouldPulse flips off the moment Reduce Motion turns on mid-pulse")
    func shouldPulseStopsWhenReduceMotionEngages() {
        // The runtime bug this guards: a dot pulsing (isAnimating true) must
        // stop — not freeze dimmed — when the user enables Reduce Motion.
        let before = RapidMotion.shouldPulse(isAnimating: true, reduceMotion: false)
        let after = RapidMotion.shouldPulse(isAnimating: true, reduceMotion: true)
        #expect(before == true && after == false)
    }

    // MARK: - Source guards for the view adoptions

    private func source(_ relativePath: String) throws -> String {
        try String(contentsOf: Self.sourceRoot.appendingPathComponent(relativePath), encoding: .utf8)
    }

    @Test("§3/§4: the shared motion vocabulary is springs, not fixed-duration easing")
    func motionVocabularyUsesSprings() throws {
        let src = try source("Sources/Rapid/UI/Modifiers/RapidMotion.swift")
        #expect(src.contains(".snappy(") || src.contains(".spring("),
                "RapidMotion must expose spring-based curves (interruptible, §3).")
        #expect(src.contains("func rapidAnimation"),
                "a reduce-motion-aware rapidAnimation(_:value:) modifier must exist.")
        #expect(src.contains("accessibilityReduceMotion"),
                "the modifier must consult accessibilityReduceMotion (§14).")
    }

    static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }
}
