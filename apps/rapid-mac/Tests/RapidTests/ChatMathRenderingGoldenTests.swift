import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `math-rendering` golden journey, sunk from the AX bash harness:
/// display and inline math are segmented out of prose (neither raw `$$`
/// source nor the `Unrenderable math:` fallback reaches the transcript),
/// and code/table chrome survives a live appearance transition. The
/// appearance leg drives ``AppearanceConfig`` directly — the same object
/// the Settings picker mutates — instead of mounting the full settings
/// surface, whose environment (updater, telemetry, dictation) is out of
/// scope for a chat-transcript journey.
@MainActor
@Suite("Golden journey: math-rendering", .serialized)
struct ChatMathRenderingGoldenTests {

    @Test("Math is segmented out of prose; code and tables survive theme flips")
    func mathCodeAndTableThroughAppearanceTransition() async throws {
        let surface = GoldenChatSurface.mount()
        let stage = surface.stage

        // Turn 1: math. #1504/#1576/#2107 — MathView exposes rendered math
        // only after SwiftMath parsed and hosted it; the safe fallback
        // exposes "Unrenderable math:". This catches both a missing font
        // resource and a parser regression.
        try await surface.sendPrompt("shape:math show me the Gaussian integral")
        try await stage.waitForText("A bridged alignment is")
        try await surface.waitForSendIdle()
        // Send idle marks network settle, not render settle: the streaming
        // markdown reveal can still be mid-flight on the trailing formula
        // (on a slow runner the raw "$$\begin{align}…" tail is briefly on
        // screen after idle). Wait for the LAST display formula's rendered
        // node before sampling the tree for the raw-source negatives.
        try await stage.wait(for: "the bridged alignment's rendered math node") {
            stage.treeText().contains("Math: \\begin{align}")
        }
        let mathText = stage.treeText()
        #expect(mathText.contains("The Gaussian integral is"))
        #expect(mathText.contains("and inline it reads $e^{i\\pi} + 1 = 0$."))
        #expect(mathText.contains("A bridged congruence is"))
        #expect(
            !mathText.contains("$$\\int_"),
            "display math reached the transcript as literal source"
        )
        #expect(
            !mathText.contains("Unrenderable math:"),
            "SwiftMath took the literal-source fallback"
        )
        // Positive artifact per display formula: the rendered SwiftMath
        // host exposes "Math: <latex>" as its accessibility label
        // (`MathView`). Without these, dropping display-math views
        // entirely would still satisfy every negative check above.
        #expect(
            mathText.contains("Math: \\int_{-\\infty}^{\\infty}"),
            "the Gaussian integral was not rendered as a math node"
        )
        #expect(
            mathText.contains("Math: a^{p-1}"),
            "the bridged congruence was not rendered as a math node"
        )
        #expect(
            mathText.contains("Math: \\begin{align}"),
            "the bridged alignment was not rendered as a math node"
        )

        // Turns 2 and 3: the code and table fixtures whose chrome the
        // appearance transition must preserve (#2056).
        try await surface.sendPrompt("shape:code show me fibonacci in python")
        try await stage.waitForText("It runs in O(n) time and constant space.")
        try await surface.waitForSendIdle()
        try await surface.sendPrompt("shape:table compare those two models for me")
        try await stage.waitForText("Both fit comfortably in 16 GB.")
        try await surface.waitForSendIdle()
        GoldenMarkdownAssertions.assertMarkdownCodeAndTable(on: stage)

        // Live Light -> Dark -> Light via the app's own appearance object.
        // NSApp.appearance is process-global, so restore whatever this test
        // process had; the config gets a throwaway defaults suite so the
        // developer's persisted appearance choice is never touched.
        let previousAppearance = NSApp.appearance
        defer { NSApp.appearance = previousAppearance }
        let defaultsSuite = "golden-appearance-\(UUID().uuidString)"
        defer { UserDefaults.standard.removePersistentDomain(forName: defaultsSuite) }
        let appearance = AppearanceConfig(
            defaults: UserDefaults(suiteName: defaultsSuite)!
        )

        appearance.mode = .dark
        #expect(NSApp.appearance?.name == .darkAqua, "Dark appearance did not apply")
        // One suspension lets AppKit propagate the appearance through the
        // stage's view hierarchy before the structure is re-asserted.
        try await Task.sleep(nanoseconds: 100_000_000)
        GoldenMarkdownAssertions.assertMarkdownCodeAndTable(on: stage)

        appearance.mode = .light
        #expect(NSApp.appearance?.name == .aqua, "Light appearance did not apply")
        try await Task.sleep(nanoseconds: 100_000_000)
        GoldenMarkdownAssertions.assertMarkdownCodeAndTable(on: stage)
    }
}
