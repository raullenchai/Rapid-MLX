import AppKit
import Testing
@testable import Rapid

/// The fade must not paint dark-mode text with the light-mode colour.
///
/// `withAlphaComponent` and `blended(withFraction:of:)` flatten a dynamic
/// `NSColor`: both read components, which resolves the colour against the
/// drawing appearance current on the calling thread. A display-link callback
/// has none, so `NSColor.textColor` resolved to its light-mode value in *both*
/// appearances. Light mode wanted near-black anyway and looked correct; dark
/// mode painted black-on-dark, so a streaming reply stayed invisible for the
/// whole fade and then appeared all at once when the animator cleared its
/// rendering attributes.
///
/// The asymmetry is why this needs a test: the bug is invisible in light mode,
/// which is where it was written and reviewed.
@Suite("Text fade appearance")
@MainActor
struct TextFadeAppearanceTests {

    /// Relative luminance of `color` as resolved under `appearance`.
    private func luminance(_ color: NSColor, under appearance: NSAppearance.Name) -> CGFloat {
        var out: CGFloat = 0
        NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
            let c = color.usingColorSpace(.sRGB) ?? .black
            out = 0.2126 * c.redComponent
                + 0.7152 * c.greenComponent
                + 0.0722 * c.blueComponent
        }
        return out
    }

    /// The colour the animator writes for a half-faded word, for a view whose
    /// window is pinned to `appearance`.
    private func fadedColor(under appearance: NSAppearance.Name) -> NSColor? {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .textColor

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 400),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        window.appearance = NSAppearance(named: appearance)

        let view = MarkdownTextBlockView(options: options)
        view.appearance = NSAppearance(named: appearance)
        view.frame = NSRect(x: 0, y: 0, width: 600, height: 400)
        window.contentView?.addSubview(view)

        view.configure(
            blocks: [.init(runs: [InlineRun(text: "alpha beta gamma delta")],
                           kind: .paragraph)],
            options: options,
            streaming: true,
            fadeState: TextFadeAnimationState()
        )

        // Draw once so the animator writes its rendering attributes.
        let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)!
        view.cacheDisplay(in: view.bounds, to: rep)

        return view.testing_renderingForegroundColor()
    }

    @Test("Dark mode fades toward a light colour")
    func darkModeFadeIsVisible() throws {
        let color = try #require(fadedColor(under: .darkAqua))
        let lum = luminance(color, under: .darkAqua)
        #expect(
            lum > 0.5,
            """
            the dark-mode fade resolved to luminance \(lum); text painted this \
            way is invisible against the dark transcript background until the \
            fade clears its rendering attributes
            """
        )
    }

    @Test("Light mode still fades toward a dark colour")
    func lightModeFadeStaysDark() throws {
        let color = try #require(fadedColor(under: .aqua))
        let lum = luminance(color, under: .aqua)
        #expect(lum < 0.5, "light mode must keep dark text; got luminance \(lum)")
    }
}
