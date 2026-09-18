import AppKit
import Foundation
import SwiftUI
import Testing

/// Tiny hand-written snapshot assert that fits Swift Testing's
/// ``#expect`` shape and doesn't drag in XCTest (which isn't on the
/// CommandLineTools toolchain we build with).
///
/// What it does:
///   1. Render the view through ``NSHostingView`` at the requested
///      size (no NSWindow wrap — avoids the headless title-bar
///      height bug pointfreeco's default strategy trips on).
///   2. Snapshot the bitmap and write a PNG to
///      ``Tests/RapidTests/__Snapshots__/<name>.png``.
///   3. First run: record the PNG (no baseline exists yet) and pass
///      with a "recorded" note in the test output.
///   4. Subsequent runs: load the baseline, compare pixel-for-pixel.
///      On mismatch: write a ``.diff.png`` alongside the baseline
///      with the new bitmap, fail with both paths in the message.
///
/// To rebaseline intentionally:
///   * Delete the relevant ``.png`` under ``__Snapshots__/`` and
///     re-run.
///   * OR set ``SNAPSHOT_RECORD=1`` in the env (rebases all).
///
/// Pixel comparison strategy: equality on the raw ``NSBitmapImageRep``
/// bitmap data (TIFF representation), not PNG byte equality —
/// PNG metadata can include timestamps and zlib state that differ
/// between runs even when the visual content is identical.

@MainActor
func assertSnapshot<V: View>(
    of view: V,
    size: CGSize,
    name: String,
    appearance: NSAppearance.Name = .aqua,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    // Build the bitmap via NSHostingView (no window) so the host
    // chrome doesn't pollute the snapshot.
    let host = NSHostingView(rootView: view)
    // Pin appearance to the requested name so baseline PNGs render
    // identically regardless of whichever Appearance the developer's
    // OS happens to be running. Default is Aqua (Light Mode); pass
    // ``.darkAqua`` to pin a dark-mode regression baseline (used by
    // #932 dark-mode sweep coverage).
    host.appearance = NSAppearance(named: appearance)
    host.frame = CGRect(origin: .zero, size: size)
    host.layoutSubtreeIfNeeded()

    // Keep snapshots deterministic when tests run without an attached Retina
    // window. `bitmapImageRepForCachingDisplay` otherwise inherits an implicit
    // 1x scale in headless shells, while the same view renders at 2x from a
    // normal developer session. All committed baselines use the documented
    // 2x scale, so construct that backing store explicitly.
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: Int(size.width * 2),
        pixelsHigh: Int(size.height * 2),
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .calibratedRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        Issue.record(
            "snapshot: could not allocate a 2x bitmap rep for \(name) at \(size)",
            sourceLocation: sourceLocation
        )
        return
    }
    rep.size = size
    host.cacheDisplay(in: host.bounds, to: rep)

    let snapshotDir = snapshotsDirectory()
    try? FileManager.default.createDirectory(
        at: snapshotDir,
        withIntermediateDirectories: true
    )

    let baselineURL = snapshotDir.appendingPathComponent("\(name).png")
    let diffURL = snapshotDir.appendingPathComponent("\(name).diff.png")

    guard let newPNG = rep.representation(using: .png, properties: [:]) else {
        Issue.record(
            "snapshot: failed to encode \(name) as PNG",
            sourceLocation: sourceLocation
        )
        return
    }

    let forceRecord = ProcessInfo.processInfo.environment["SNAPSHOT_RECORD"] == "1"

    if forceRecord || !FileManager.default.fileExists(atPath: baselineURL.path) {
        do {
            try newPNG.write(to: baselineURL)
        } catch {
            Issue.record(
                "snapshot: could not write baseline at \(baselineURL.path): \(error)",
                sourceLocation: sourceLocation
            )
            return
        }
        // First-run path: signal the recording but don't fail. CI
        // gets a once-only "recording new snapshot" message; the
        // next run will actually compare.
        print("snapshot: recorded baseline \(name).png (\(newPNG.count) bytes)")
        return
    }

    // Pixel-compare against the recorded baseline. We canonicalise
    // both images by rendering through ``CGBitmapContext`` with the
    // same pixel format, then byte-compare the resulting buffers.
    //
    // Why not just compare PNG/TIFF bytes: ``NSBitmapImageRep``
    // bakes color profile + DPI + orientation metadata into both
    // formats, and the metadata is regenerated per render — so two
    // identical pixel buffers can produce non-equal serialised
    // bytes. The canonical CGBitmapContext path strips all of that.
    guard let baselineData = try? Data(contentsOf: baselineURL),
          let baselineRep = NSBitmapImageRep(data: baselineData) else {
        Issue.record(
            "snapshot: could not load baseline \(baselineURL.path)",
            sourceLocation: sourceLocation
        )
        return
    }

    let newPixels = canonicalPixelBuffer(of: rep)
    let baselinePixels = canonicalPixelBuffer(of: baselineRep)
    if newPixels == baselinePixels, !newPixels.isEmpty {
        return // OK
    }

    // Mismatch — write the new bitmap as a .diff.png so the user
    // can eyeball what changed without re-running with RECORD=1.
    try? newPNG.write(to: diffURL)
    Issue.record(
        """
        snapshot mismatch: \(name)
          baseline: \(baselineURL.path)
          new:      \(diffURL.path)
        Eyeball the new image; if it's an intended change, delete the
        baseline (or set SNAPSHOT_RECORD=1) and re-run.
        """,
        sourceLocation: sourceLocation
    )
}

/// Render ``rep`` into a fresh ``CGBitmapContext`` with a fixed
/// pixel format (premultiplied RGBA, sRGB) and return the raw byte
/// buffer. The output is metadata-free and deterministic, so two
/// equal pixel buffers compare ``Data`` equal.
private func canonicalPixelBuffer(of rep: NSBitmapImageRep) -> Data {
    let width = rep.pixelsWide
    let height = rep.pixelsHigh
    guard width > 0, height > 0,
          let cgImage = rep.cgImage,
          let space = CGColorSpace(name: CGColorSpace.sRGB) else {
        return Data()
    }
    let bytesPerRow = width * 4
    var buffer = Data(count: bytesPerRow * height)
    let success: Bool = buffer.withUnsafeMutableBytes { rawBytes -> Bool in
        guard let baseAddress = rawBytes.baseAddress,
              let ctx = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            return false
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    return success ? buffer : Data()
}

private func snapshotsDirectory() -> URL {
    // ``#file`` resolves to the absolute path of THIS source file.
    // Walk up to the ``RapidTests`` test target dir and append
    // ``__Snapshots__/``. SPM's ``Bundle.module`` is the canonical
    // way to reach test resources, but it copies them at build
    // time — that's wrong here, we want to write back into the
    // source tree so the baselines get committed.
    let here = URL(fileURLWithPath: #filePath)
    return here.deletingLastPathComponent().appendingPathComponent("__Snapshots__")
}
