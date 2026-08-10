import Foundation

/// Rapid's assembled app keeps third-party resources under
/// `Contents/Resources`, the only code-signing-safe location. SwiftPM's
/// generated accessor instead probes the app wrapper root. Prefer the real
/// application bundle and retain the generated bundle as the development and
/// test fallback.
public enum SwiftMathResources {
    public static var fontsBundleURL: URL? {
        Bundle.main.url(forResource: "mathFonts", withExtension: "bundle")
            ?? Bundle.module.url(forResource: "mathFonts", withExtension: "bundle")
    }
}
