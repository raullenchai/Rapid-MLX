import CommonCrypto
import Foundation

/// Finds the vendored Mermaid bundle, and proves it is the one we shipped.
///
/// Shaped after ``SwiftMathResources``: the assembled `.app` must stand on
/// its own after the build checkout is gone, so a signed build resolves only
/// through `Bundle.main`. The development branch exists for `swift test` and
/// `swift run`, which have no app wrapper.
///
/// The digest check is the ``MathView`` discipline applied to a second
/// resource: a name lookup is true for a truncated file, and a truncated
/// `mermaid.min.js` loads, sets no `mermaid` global, and fails inside the
/// first render — one link past where a name check would have stopped. The
/// digest catches truncation and tampering exactly, where a size band catches
/// neither reliably.
enum MermaidLibrary {

    /// Kept beside the file it describes so the two move together.
    static let vendorDirectoryName = "mermaid"

    /// The bytes, or nil when anything about them is wrong.
    static func load() -> Data? {
        guard let url = locate(), let data = try? Data(contentsOf: url) else { return nil }
        guard let expected = expectedDigest() else { return nil }
        return validated(data: data, expectedDigest: expected)
    }

    /// Pure integrity boundary used by both the loader and its tamper test.
    static func validated(data: Data, expectedDigest: String) -> Data? {
        digest(of: data) == expectedDigest ? data : nil
    }

    private static func locate() -> URL? {
        if Bundle.main.bundleURL.pathExtension == "app" {
            return Bundle.main.url(forResource: "mermaid.min", withExtension: "js")
        }
        return developmentVendorURL?.appendingPathComponent("mermaid.min.js")
    }

    /// `Vendor/mermaid/`, found by walking up from this source file. Only
    /// reachable in a non-`.app` build — a shipped app that fell back here
    /// would be depending on a directory the reader does not have, which is
    /// the packaging regression ``MathView`` refuses to let hide.
    static var developmentVendorURL: URL? {
        var directory = URL(fileURLWithPath: #filePath)
        for _ in 0..<8 {
            directory.deleteLastPathComponent()
            let candidate = directory
                .appendingPathComponent("Vendor", isDirectory: true)
                .appendingPathComponent(vendorDirectoryName, isDirectory: true)
            if FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("mermaid.min.js").path
            ) { return candidate }
        }
        return nil
    }

    /// The digest committed beside the library, in `shasum -a 256` format.
    private static func expectedDigest() -> String? {
        let url: URL?
        if Bundle.main.bundleURL.pathExtension == "app" {
            url = Bundle.main.url(forResource: "mermaid.min.js", withExtension: "sha256")
        } else {
            url = developmentVendorURL?.appendingPathComponent("mermaid.min.js.sha256")
        }
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        return text.split(separator: " ").first.map(String.init)
    }

    static func digest(of data: Data) -> String {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBytes { buffer in
            _ = CC_SHA256(buffer.baseAddress, CC_LONG(buffer.count), &hash)
        }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}
