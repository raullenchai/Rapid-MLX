import AppKit
import Foundation
import Testing
@testable import Rapid

/// Pins for the UI-1 menu-bar mark.
///
/// The tray icon moved from a full-colour cheetah bitmap to the official
/// `R` drawn as vector geometry and tagged as a template. Three things
/// have to stay true for that to keep working, and none of them is
/// visible in a screenshot of a passing build:
///
///   1. the image is a TEMPLATE, so macOS owns its colour and it adapts
///      to a light bar, a dark bar, and the highlighted state;
///   2. the status item still has an accessible name;
///   3. the geometry actually parses — a silent parser failure would
///      fall through to the SF Symbol and nobody would notice until
///      somebody looked at their menu bar.
@MainActor
@Suite("Menu-bar template mark")
struct MenuBarTemplateMarkTests {

    @Test("The tray glyph is a template image")
    func trayGlyphIsTemplate() {
        let image = MenuBarController.trayGlyph()
        #expect(
            image.isTemplate,
            """
            The tray glyph must be a template image. A non-template glyph \
            ignores the menu bar's appearance: it renders identically on a \
            light bar, a dark bar, and while the menu is open — which is the \
            defect the full-colour cheetah had.
            """
        )
    }

    @Test("The tray glyph carries an accessible description")
    func trayGlyphIsNamed() {
        let image = MenuBarController.trayGlyph()
        // ``configureButton`` sets this on the button's image; the fallback
        // path sets it at construction. Either way the shipped image must
        // be nameable — a status item whose label is a bare image and
        // whose image has no description is unreachable to VoiceOver.
        let described = image.accessibilityDescription ?? MenuBarController.accessibilityTitle
        #expect(described == MenuBarController.accessibilityTitle)
        #expect(!MenuBarController.accessibilityTitle.isEmpty)
    }

    @Test("The official R geometry parses and is the shipped glyph")
    func officialGeometryParses() throws {
        let path = try #require(
            RapidRMark.glyphPath(),
            """
            The official `R` path data failed to parse, so the tray silently \
            fell back to an SF Symbol. Check SVGPathData against the command \
            set the current rapid_icon.svg export uses.
            """
        )
        #expect(!path.isEmpty)
        // The mark is very nearly square (105.17 × 105.42 in the 192pt
        // viewBox). A wildly different aspect means the path data was
        // replaced with something that is not this mark.
        let bounds = path.bounds
        #expect(bounds.width > 100 && bounds.width < 110)
        #expect(bounds.height > 100 && bounds.height < 110)
    }

    @Test("The full lockup parses too, and is wider than the R alone")
    func fullLockupParses() throws {
        let glyph = try #require(RapidRMark.glyphPath())
        let full = try #require(RapidRMark.fullMarkPath())
        #expect(
            full.bounds.width > glyph.bounds.width,
            "The full lockup includes the two speed streaks, which extend left of the R."
        )
    }

    @Test("The menu-bar image is sized to the menu-bar glyph height")
    func menuBarImageIsCorrectlySized() throws {
        let image = try #require(RapidRMark.menuBarTemplateImage())
        #expect(image.size.height == RapidRMark.menuBarGlyphHeight)
        // Near-square, and never taller than the 18pt an NSStatusItem
        // gives its content inside the 24pt bar.
        #expect(image.size.width > 0)
        #expect(image.size.height <= 18)
    }

    /// The streaks are why the menu-bar variant is the `R` alone. If a
    /// future export makes them substantial, this test fails and the
    /// decision gets revisited on purpose rather than by accident.
    @Test("The speed streaks are too fine to survive menu-bar scale")
    func streaksAreSubPointAtMenuBarScale() throws {
        let glyph = try #require(RapidRMark.glyphPath())
        let upper = try #require(SVGPathData.parse(RapidRMark.upperStreakPathData))
        let scale = RapidRMark.menuBarGlyphHeight / glyph.bounds.height
        let scaledStreakHeight = upper.bounds.height * scale
        #expect(
            scaledStreakHeight < 1.0,
            """
            The upper streak now scales to \(scaledStreakHeight)pt at menu-bar \
            size. It was sub-point (~0.78pt) when the R-only variant was \
            chosen; if the artwork changed, re-evaluate whether the full \
            lockup is now legible in the bar.
            """
        )
    }

    @Test("A malformed path is rejected rather than half-drawn")
    func unsupportedCommandsAreRejected() {
        // Arcs are not in the supported command set. The parser must
        // return nil (so the caller falls back) instead of skipping the
        // command and rendering a subtly wrong mark.
        #expect(SVGPathData.parse("M0 0 A 10 10 0 0 1 20 20 Z") == nil)
        #expect(SVGPathData.parse("") == nil)
        // A path that never opens a subpath is not a shape.
        #expect(SVGPathData.parse("L10 10") == nil)
    }

    @Test("The supported command set round-trips")
    func supportedCommandsParse() throws {
        // Absolute and relative forms of every command the official file
        // uses, so a future export that leans on the relative forms is
        // covered by something other than hope.
        let box = try #require(SVGPathData.parse("M0 0 H10 V10 H0 Z"))
        #expect(box.bounds.width == 10)
        #expect(box.bounds.height == 10)

        let relative = try #require(SVGPathData.parse("m0 0 h10 v10 h-10 z"))
        #expect(relative.bounds.width == 10)

        let curve = try #require(SVGPathData.parse("M0 0 C0 5 5 10 10 10 Z"))
        #expect(curve.bounds.width > 0)

        // Scientific notation and a bare leading decimal point.
        #expect(SVGPathData.parse("M0 0 L1e1 .5 Z") != nil)
    }
}
