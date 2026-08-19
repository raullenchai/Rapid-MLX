import SwiftUI
import AppKit
import SwiftMath

/// Issue #131: SwiftUI wrapper around SwiftMath's ``MTMathUILabel``
/// (the macOS-friendly Swift port of iosMath). Renders a LaTeX
/// expression natively — no ``WKWebView``, no JS.
///
/// The label is an ``NSView`` subclass; we bridge it via
/// ``NSViewRepresentable`` and recompute the intrinsic size on
/// every update so the SwiftUI layout system gives us the right
/// frame.
///
/// Why a fresh ``label.sizeToFit()`` on every update:
/// ``MTMathUILabel`` is a CoreText-backed view. When the latex
/// changes, the internal mathlist re-parses and the natural size
/// can shift dramatically (a ``\frac`` row is much taller than a
/// plain ``x^2`` baseline). SwiftUI caches our frame between
/// ``updateNSView`` calls, so we must publish the new size or the
/// glyph either clips or floats inside an oversize bounding box.
///
/// Error states: an unparseable LaTeX body falls back to rendering
/// the raw source in a monospaced ``Text``. Better than an empty
/// hole — gives the user a hint that the model emitted something
/// the renderer didn't understand.
struct MathView: View {
    let latex: String
    let displayMode: Bool

    @Environment(\.colorScheme) private var colorScheme

    /// #546: match the transcript's Dynamic-Type scaling. ``MarkdownUI``
    /// scales the surrounding prose by wrapping the ``.rapidChat`` theme's
    /// fixed root size in its own `@ScaledMetric(relativeTo: .body)`
    /// (`Markdown.swift` `ScaledFontSizeModifier`). Mirror that exact
    /// curve here so a rendered formula tracks the prose instead of
    /// staying pinned while the text around it grows. 15 matches the
    /// theme root (2026-07 typography sweep) — this literal moves in
    /// lockstep with `.rapidChat`'s `FontSize` and the streaming Text,
    /// or formulas render one size off from the prose around them.
    @ScaledMetric(relativeTo: .body) private var baseFontSize: CGFloat = 15

    var body: some View {
        // Bridge, then probe-parse; ``renderable`` documents both, and why
        // ``mathFontsAvailable`` has to be the first thing checked.
        if Self.mathFontsAvailable, let source = Self.renderable(latex) {
            MathHost(
                latex: source,
                displayMode: displayMode,
                colorScheme: colorScheme,
                baseFontSize: baseFontSize
            )
                .accessibilityLabel("Math: \(latex)")
        } else {
            Text(displayMode ? "$$\(latex)$$" : "$\(latex)$")
                .scaledSystemFont(14, design: .monospaced)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .accessibilityLabel("Unrenderable math: \(latex)")
        }
    }

    /// The body to hand ``MathHost``, or nil when SwiftMath cannot parse it.
    ///
    /// ``LaTeXCompatibility`` runs first: a model writing `\mod` or
    /// `\begin{align}` is writing valid LaTeX that SwiftMath's table happens
    /// not to cover, and without the bridge every such formula reaches the
    /// raw-source fallback below. It runs *here*, inside the
    /// ``mathFontsAvailable`` branch, for the reason that guard exists — no
    /// SwiftMath type may be touched until the resources behind it are known
    /// to be complete.
    ///
    /// The probe parse is cheap: ``MTMathListBuilder/build`` goes `String` →
    /// `MTMathList` without laying anything out. The two fallbacks keep
    /// `latex` rather than the bridged source, so what a reader sees when
    /// rendering fails is what the model actually wrote.
    @MainActor
    private static func renderable(_ latex: String) -> String? {
        let source = LaTeXCompatibility.normalized(latex)
        var error: NSError?
        let list = MTMathListBuilder.build(fromString: source, error: &error)
        guard list != nil, error == nil else { return nil }
        return source
    }

    /// Is the signed app's SwiftMath resource bundle usable? This deliberately
    /// checks `Bundle.main`, not SwiftPM's development-only resource accessor:
    /// the assembled app must stand on its own after the build checkout is
    /// removed. The vendored SwiftMath resolver uses this same app-bundle path
    /// first and retains `Bundle.module` only for `swift run` and tests.
    ///
    /// The check walks SwiftMath's resource chain rather than merely
    /// testing that something exists at the path. `fileExists(atPath:)` is
    /// true for a regular file, a broken symlink target, or a directory that
    /// was copied incompletely — and in every one of those cases SwiftMath
    /// would load `Bundle.module` successfully and then die a step later, in
    /// `registerCGFont` / `registerMathTable` / `MTFont.fontBundle`, all of
    /// which force-unwrap or `fatalError` on the resources below. A guard that
    /// stops one link short of the thing that actually crashes is not a guard.
    ///
    /// Mirrors, in order, what SwiftMath does for its default font
    /// (`MathFont.latinModernFont` → `"latinmodern-math"`): open the outer
    /// `mathFonts.bundle`, then require both
    /// the `.otf` (`registerCGFont`) and the `.plist` (`registerMathTable`).
    /// All of it is plain Foundation — no SwiftMath type is touched.
    /// Resource *contents* are validated, not just their names. A truncated
    /// `.otf` or a malformed `.plist` satisfies a name lookup and then dies one
    /// step further in, where SwiftMath turns its own `FontError` back into a
    /// `fatalError` — so every step SwiftMath can fail at is performed here
    /// first, in the same order, using the same APIs:
    ///
    /// * `registerCGFont`   — `NSData(contentsOfFile:)` → `CGDataProvider` → `CGFont`
    /// * `registerMathTable` — `NSDictionary(contentsOf:)` and `version == "1.3"`
    ///
    /// Cost is one ~1 MB font read, once per process (`static let`), on the
    /// first message containing math — not per render.
    private static let mathFontsAvailable: Bool = {
        // An assembled app must prove its own signed resources are complete;
        // never let the build checkout hide a packaging regression. Bare
        // SwiftPM binaries have no app wrapper and intentionally share
        // SwiftMath's Bundle.module development fallback.
        let fontsURL = Bundle.main.bundleURL.pathExtension == "app"
            ? Bundle.main.url(forResource: "mathFonts", withExtension: "bundle")
            : SwiftMathResources.fontsBundleURL
        guard let fontsURL,
              let fonts = Bundle(url: fontsURL),
              let fontPath = fonts.path(forResource: "latinmodern-math", ofType: "otf"),
              let tableURL = fonts.url(forResource: "latinmodern-math", withExtension: "plist"),
              let fontData = NSData(contentsOfFile: fontPath),
              let provider = CGDataProvider(data: fontData),
              CGFont(provider) != nil,
              let table = NSDictionary(contentsOf: tableURL),
              (table["version"] as? String) == "1.3"
        else { return false }
        return true
    }()
}

private struct MathHost: NSViewRepresentable {
    let latex: String
    let displayMode: Bool
    let colorScheme: ColorScheme
    /// Dynamic-Type-scaled base point size, resolved by ``MathView``'s
    /// `@ScaledMetric` and threaded in so the label tracks the prose.
    let baseFontSize: CGFloat

    func makeNSView(context: Context) -> MTMathUILabel {
        let label = MTMathUILabel()
        configure(label)
        return label
    }

    func updateNSView(_ label: MTMathUILabel, context: Context) {
        configure(label)
    }

    private func configure(_ label: MTMathUILabel) {
        label.latex = latex
        label.labelMode = displayMode ? .display : .text
        // Use the Dynamic-Type-scaled body point size so inline math
        // sits on the same baseline as surrounding ``MarkdownUI`` prose
        // AND grows with it (#546). Display math gets a small bump for
        // visual prominence, matching MathJax / KaTeX default conventions.
        let baseSize = baseFontSize
        label.fontSize = displayMode ? baseSize + 2 : baseSize
        label.textAlignment = displayMode ? .center : .left
        // Tint glyphs to match the current colour scheme so dark
        // mode doesn't render math as black-on-black.
        let glyphColor: NSColor = (colorScheme == .dark) ? .white : .black
        label.textColor = MTColor(cgColor: glyphColor.cgColor) ?? MTColor.black
        // Intrinsic-size recompute — see file header. SwiftMath's
        // ``MTMathUILabel`` overrides ``intrinsicContentSize`` so
        // ``invalidateIntrinsicContentSize`` is the right trigger;
        // there's no ``sizeToFit`` API to call on this NSView.
        label.invalidateIntrinsicContentSize()
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MTMathUILabel, context: Context) -> CGSize? {
        // Codex r1 P1 (#131): on macOS, ``MTMathUILabel`` overrides
        // ``NSView.fittingSize``, NOT ``intrinsicContentSize`` — the
        // latter returns AppKit's no-intrinsic-metric sentinel and
        // SwiftUI lays the view out as zero-size, hiding every
        // rendered formula. ``fittingSize`` re-runs the underlying
        // ``_sizeThatFits`` math and returns the real glyph bounds.
        nsView.fittingSize
    }
}
