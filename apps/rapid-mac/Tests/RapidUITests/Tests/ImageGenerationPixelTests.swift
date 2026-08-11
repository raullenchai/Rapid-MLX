import AppKit
import XCTest

@MainActor
final class ImageGenerationPixelTests: XCTestCase {
    func testTwoImageRendersDrawDistinctThumbnailPixels() throws {
        continueAfterFailure = false
        let testHome = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-xcui-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: testHome, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: testHome) }

        // The CI runner is disposable, but clearing the app domain makes local
        // full-Xcode runs deterministic too. HOME isolates sessions and owned
        // server records; CFPreferences needs the explicit domain reset.
        let defaults = Process()
        defaults.executableURL = URL(fileURLWithPath: "/usr/bin/defaults")
        defaults.arguments = ["delete", "com.rapidmlx.rapid"]
        defaults.environment = ProcessInfo.processInfo.environment.merging([
            "CFFIXED_USER_HOME": testHome.path,
        ]) { _, isolated in isolated }
        try defaults.run()
        defaults.waitUntilExit()

        let rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // RapidUITests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
        let fakeSidecar = rapidMacRoot.appendingPathComponent("scripts/fake-rapid-mlx.sh").path
        let appURL = rapidMacRoot.appendingPathComponent("build/Rapid-MLX Desktop.app")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: fakeSidecar))
        XCTAssertTrue(FileManager.default.fileExists(atPath: appURL.path))
        let app = XCUIApplication(url: appURL)
        app.launchEnvironment = [
            "HOME": testHome.path,
            "CFFIXED_USER_HOME": testHome.path,
            "RAPID_BIN": fakeSidecar,
            "FAKE_EVENT_LOG": testHome.appendingPathComponent("fake-events.jsonl").path,
            "FAKE_IMAGE_STEPS": "4",
            "FAKE_IMAGE_STEP_MS": "100",
        ]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.windows["Rapid-MLX"].waitForExistence(timeout: 20))
        dismissFirstRunIfNeeded(in: app)
        let images = element("Sidebar.Images", in: app)
        XCTAssertTrue(images.waitForExistence(timeout: 10))
        images.click()

        let readiness = element("Readiness.Action", in: app)
        XCTAssertTrue(readiness.waitForExistence(timeout: 20))
        readiness.click()

        let prompt = element("Images.Prompt", in: app)
        XCTAssertTrue(prompt.waitForExistence(timeout: 20))
        prompt.click()
        prompt.typeText("a cheetah on a red couch")
        let generate = element("Images.Generate", in: app)
        XCTAssertTrue(waitUntil(timeout: 30) { generate.isEnabled })
        generate.click()

        let save = element("Images.Result.Save", in: app)
        XCTAssertTrue(save.waitForExistence(timeout: 30))
        let first = element("Images.Gallery.Thumb.1", in: app)
        XCTAssertTrue(first.waitForExistence(timeout: 30))

        prompt.click()
        prompt.typeKey("a", modifierFlags: .command)
        prompt.typeText("the same cheetah, at night")
        XCTAssertTrue(waitUntil(timeout: 10) { generate.isEnabled })
        generate.click()
        XCTAssertTrue(save.waitForNonExistence(timeout: 5))
        XCTAssertTrue(save.waitForExistence(timeout: 30))

        let newest = element("Images.Gallery.Thumb.1", in: app)
        let older = element("Images.Gallery.Thumb.2", in: app)
        XCTAssertTrue(older.waitForExistence(timeout: 30))

        let newestShot = newest.screenshot()
        let olderShot = older.screenshot()
        add(XCTAttachment(screenshot: newestShot))
        add(XCTAttachment(screenshot: olderShot))

        let newestPixels = try centerRGBSamples(newestShot.pngRepresentation)
        let olderPixels = try centerRGBSamples(olderShot.pngRepresentation)
        XCTAssertEqual(newestPixels.count, olderPixels.count)
        let meanSquaredDistance = zip(newestPixels, olderPixels)
            .map { Double($0.0) - Double($0.1) }
            .map { $0 * $0 }
            .reduce(0, +) / Double(newestPixels.count)
        XCTAssertGreaterThan(
            meanSquaredDistance.squareRoot(), 10,
            "The two records exist but their rendered thumbnail interiors are indistinguishable"
        )
    }

    private func dismissFirstRunIfNeeded(in app: XCUIApplication) {
        let decline = element("TelemetryConsent.DontShare", in: app)
        if decline.waitForExistence(timeout: 5) { decline.click() }
        let skip = element("Quickstart.Skip", in: app)
        if skip.waitForExistence(timeout: 10) { skip.click() }
    }

    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func waitUntil(timeout: TimeInterval, condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline
        return condition()
    }

    /// Compare only the central 60% of each element screenshot. This removes
    /// the selected/unselected stroke and button chrome, leaving the pixels
    /// the user perceives as the generated image.
    private func centerRGBSamples(_ png: Data) throws -> [CGFloat] {
        let image = try XCTUnwrap(NSImage(data: png), "XCTest returned an undecodable screenshot")
        let source = try XCTUnwrap(
            image.cgImage(forProposedRect: nil, context: nil, hints: nil),
            "XCTest returned a screenshot without a CGImage"
        )
        let insetX = source.width / 5
        let insetY = source.height / 5
        let rect = CGRect(
            x: CGFloat(insetX), y: CGFloat(insetY),
            width: CGFloat(source.width - 2 * insetX),
            height: CGFloat(source.height - 2 * insetY)
        )
        let cropped = try XCTUnwrap(
            source.cropping(to: rect),
            "thumbnail screenshot was too small to crop"
        )
        let rep = NSBitmapImageRep(cgImage: cropped)
        var samples: [CGFloat] = []
        samples.reserveCapacity((rep.pixelsWide / 2) * (rep.pixelsHigh / 2) * 3)
        for y in stride(from: 0, to: rep.pixelsHigh, by: 2) {
            for x in stride(from: 0, to: rep.pixelsWide, by: 2) {
                guard let color = rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                samples.append(color.redComponent * 255)
                samples.append(color.greenComponent * 255)
                samples.append(color.blueComponent * 255)
            }
        }
        XCTAssertFalse(samples.isEmpty)
        return samples
    }
}
