import SwiftUI

/// The rapidmlx.com brand cheetah — the cute illustrated mascot the
/// website ships, vendored from `landing/public/cheetah*.png`. Replaces
/// the hand-rolled SwiftUI ``CheetahMark`` silhouette at every
/// brand-mark site (sidebar header, QuickAsk splash, chat empty
/// state) because users repeatedly called the silhouette ugly and
/// asked specifically for the site mascot.
///
/// Two physical assets are vendored so SwiftUI can pick the closer
/// match for its rendering context:
///   * ``cheetah.png`` — 440 × 390 master, used at hero sizes
///     (QuickAsk splash, About panel, large empty state).
///   * ``cheetah-sm.png`` — 56 × 50 down-rendered crop, used at
///     sidebar / chip sizes where the master's detail would alias.
///
/// macOS picks @2x automatically for Retina displays — a 28 pt
/// destination renders from cheetah-sm.png at 56 px, perfect 1:1.
/// A 120 pt destination renders from cheetah.png at 240 px, well
/// inside the 440 × 390 source.
struct CheetahLogo: View {
    /// Pixel-side hint that drives which vendored asset gets used.
    /// Numbers under 64 use the small crop; larger sizes use the
    /// master. SwiftUI then ``resizable().scaledToFit()`` to the
    /// caller's frame.
    var size: CGFloat

    var body: some View {
        if let nsImage = Self.load(forSize: size) {
            Image(nsImage: nsImage)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .frame(width: size, height: size)
                .accessibilityHidden(true)
        } else {
            // Defensive fallback — the asset is bundled at build
            // time, so this branch is "the .app got corrupted"
            // territory. A tinted Image symbol keeps the chrome
            // intact so the sidebar / splash don't blank out.
            Image(systemName: "hare.fill")
                .resizable()
                .scaledToFit()
                .frame(width: size, height: size)
                .foregroundStyle(RapidTheme.brandAmber)
                .accessibilityHidden(true)
        }
    }

    /// Loads the right vendored asset for ``size``.
    ///
    /// v0.5.9 used SPM's synthesised ``Bundle.module`` accessor.
    /// That was a SHIP-BLOCKER on real Macs: for executable
    /// targets SPM generates a ``resource_bundle_accessor.swift``
    /// that looks for ``Rapid.app/Rapid_Rapid.bundle`` (top of
    /// the .app, sibling to ``Contents/``) — not
    /// ``Contents/Resources/Rapid_Rapid.bundle/`` where macOS
    /// codesign requires it to live. On miss the accessor
    /// ``fatalError``s inside ``dispatch_once`` the FIRST time
    /// any view's body reads the static. Result: app abort with
    /// trace ``CheetahLogo.module → __assertionFailure`` the
    /// instant SwiftUI tried to render the sidebar header.
    ///
    /// The fix avoids ``Bundle.module`` entirely:
    ///   * Production .app: ``scripts/build.sh`` copies the PNGs
    ///     as flat resources into ``Contents/Resources/`` so
    ///     ``Bundle.main.url(forResource:)`` resolves them.
    ///   * Dev / tests: ``Bundle(for: BundleFinder.self)`` walks
    ///     to ``Rapid_Rapid.bundle`` next to the test runner /
    ///     ``swift run`` executable. This is the same path
    ///     ``Bundle.module`` would compute, but probed
    ///     gracefully via ``Bundle(url:)`` so a miss returns
    ///     nil instead of crashing.
    static func load(forSize size: CGFloat) -> NSImage? {
        let name = size < 64 ? "cheetah-sm" : "cheetah"

        if let url = Bundle.main.url(forResource: name, withExtension: "png"),
           let image = NSImage(contentsOf: url) {
            return image
        }

        let anchor = Bundle(for: BundleFinder.self).bundleURL.deletingLastPathComponent()
        let bundleURL = anchor.appendingPathComponent("Rapid_Rapid.bundle")
        if let bundle = Bundle(url: bundleURL),
           let url = bundle.url(forResource: name, withExtension: "png"),
           let image = NSImage(contentsOf: url) {
            return image
        }

        return nil
    }
}

private final class BundleFinder {}
