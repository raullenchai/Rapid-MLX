import AppKit
import Foundation
import Testing

/// Pin that the shipped `Resources/AppIcon.icns` actually contains a
/// representation at every size macOS will request. The cmd-key
/// chase down the audit P2 ("App icon — Untested at all sizes
/// (16/32/128/256/512)"): a designer can ship an updated icon that
/// passes basic Finder review at 1024×1024 but is missing one of the
/// smaller representations, which silently lets LaunchServices fall
/// back to a blurry upscale at e.g. 16 pt in the menu bar or 32 pt
/// in Spotlight.
///
/// macOS reads 5 base sizes off an icns (16, 32, 128, 256, 512) and
/// each has a `@2x` retina pair, so 10 representations total. We
/// assert all 10 are present and decode to a non-empty PNG payload.
@Suite("AppIcon.icns subimage coverage", TestTimeouts.hangProne)
struct AppIconBundleTests {
    /// Locate the shipped icns by walking up from the test bundle
    /// to the repo root, since `swift test` doesn't expose Resources/
    /// as a bundle path. The walk stops at the first directory
    /// containing `Package.swift` — that's our repo root regardless
    /// of where the test binary lives under `.build/`.
    private static func icnsURL() -> URL? {
        var dir = URL(fileURLWithPath: #file)
        for _ in 0..<10 {
            dir = dir.deletingLastPathComponent()
            let manifest = dir.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: manifest.path) {
                return dir.appendingPathComponent("Resources/AppIcon.icns")
            }
        }
        return nil
    }

    /// The five base sizes Apple's icns spec requires for a complete
    /// icon. Each pairs with a `@2x` retina sibling — so 10 PNG
    /// representations total inside the .icns container.
    private static let baseSizes: [Int] = [16, 32, 128, 256, 512]

    @Test("icns container ships all 5 base sizes + their @2x retina siblings")
    func icnsContainsEveryRequiredSize() async throws {
        let url = try #require(Self.icnsURL(), "AppIcon.icns not found under Resources/")
        try #require(FileManager.default.fileExists(atPath: url.path))

        // `iconutil --convert iconset` is the same tool the build
        // pipeline uses; it's the ground-truth way to enumerate
        // an icns. Run it against a tmpdir so we don't pollute the
        // repo with extracted PNGs that could confuse a clean git
        // working tree.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-iconset-\(UUID().uuidString).iconset")
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/iconutil"),
            arguments: ["--convert", "iconset", url.path, "-o", tmpDir.path]
        )
        try #require(
            result.terminationStatus == 0,
            "iconutil failed: \(String(decoding: result.standardError, as: UTF8.self))"
        )

        for size in Self.baseSizes {
            let base = tmpDir.appendingPathComponent("icon_\(size)x\(size).png")
            let retina = tmpDir.appendingPathComponent("icon_\(size)x\(size)@2x.png")
            #expect(
                FileManager.default.fileExists(atPath: base.path),
                "missing icon_\(size)x\(size).png — designer dropped a base size"
            )
            #expect(
                FileManager.default.fileExists(atPath: retina.path),
                "missing icon_\(size)x\(size)@2x.png — designer dropped a retina pair"
            )

            // Each PNG must decode to an image of the size advertised
            // in its filename — guards against a designer dropping a
            // wrong-size PNG into the iconset (e.g. shipping a 256×256
            // labelled as `icon_128x128@2x.png` would still be 256×256
            // visually, but `icon_32x32.png` mislabelled is a hard
            // regression).
            for path in [base.path, retina.path] {
                guard let img = NSImage(contentsOfFile: path),
                      let rep = img.representations.first as? NSBitmapImageRep else {
                    Issue.record("\(path) failed to decode as bitmap image")
                    continue
                }
                let advertised = path.hasSuffix("@2x.png") ? size * 2 : size
                #expect(
                    rep.pixelsWide == advertised && rep.pixelsHigh == advertised,
                    "\(path) decoded as \(rep.pixelsWide)×\(rep.pixelsHigh), expected \(advertised)×\(advertised)"
                )
            }
        }
    }
}
