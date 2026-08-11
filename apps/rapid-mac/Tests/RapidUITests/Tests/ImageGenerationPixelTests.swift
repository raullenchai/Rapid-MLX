import AppKit
import XCTest

@MainActor
final class ImageGenerationPixelTests: XCTestCase {
    private var app: XCUIApplication!
    private var testHome: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false
        testHome = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-xcui-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: testHome, withIntermediateDirectories: true)

        // The CI runner is disposable, but clearing the app domain makes local
        // full-Xcode runs deterministic too. HOME isolates sessions and owned
        // server records; CFPreferences needs the explicit domain reset.
        let defaults = Process()
        defaults.executableURL = URL(fileURLWithPath: "/usr/bin/defaults")
        defaults.arguments = ["delete", "com.rapidmlx.rapid"]
        try defaults.run()
        defaults.waitUntilExit()

        app = XCUIApplication(bundleIdentifier: "com.rapidmlx.rapid")
        app.launchEnvironment = [
            "HOME": testHome.path,
            "RAPID_BIN": ProcessInfo.processInfo.environment["RAPID_XCUI_FAKE_BIN"]!,
            "FAKE_EVENT_LOG": testHome.appendingPathComponent("fake-events.jsonl").path,
            "FAKE_IMAGE_STEPS": "4",
            "FAKE_IMAGE_STEP_MS": "100",
        ]
        app.launch()
        XCTAssertTrue(app.windows["Rapid-MLX"].waitForExistence(timeout: 20))
    }

    override func tearDownWithError() throws {
        app?.terminate()
        if let testHome { try? FileManager.default.removeItem(at: testHome) }
    }

    func testTwoImageRendersDrawDistinctThumbnailPixels() throws {
        dismissFirstRunIfNeeded()
        let images = element("Sidebar.Images")
        XCTAssertTrue(images.waitForExistence(timeout: 10))
        images.click()

        let readiness = element("Readiness.Action")
        XCTAssertTrue(readiness.waitForExistence(timeout: 20))
        readiness.click()

        let prompt = element("Images.Prompt")
        XCTAssertTrue(prompt.waitForExistence(timeout: 20))
        prompt.click()
        prompt.typeText("a cheetah on a red couch")
        let generate = element("Images.Generate")
        XCTAssertTrue(waitUntil(timeout: 30) { generate.isEnabled })
        generate.click()

        let first = element("Images.Gallery.Thumb.1")
        XCTAssertTrue(first.waitForExistence(timeout: 30))

        prompt.click()
        prompt.typeKey("a", modifierFlags: .command)
        prompt.typeText("the same cheetah, at night")
        XCTAssertTrue(waitUntil(timeout: 10) { generate.isEnabled })
        generate.click()

        let newest = element("Images.Gallery.Thumb.1")
        let older = element("Images.Gallery.Thumb.2")
        XCTAssertTrue(older.waitForExistence(timeout: 30))

        let newestShot = newest.screenshot()
        let olderShot = older.screenshot()
        add(XCTAttachment(screenshot: newestShot))
        add(XCTAttachment(screenshot: olderShot))

        let newestRGB = try centerMeanRGB(newestShot.pngRepresentation)
        let olderRGB = try centerMeanRGB(olderShot.pngRepresentation)
        let distance = zip(newestRGB, olderRGB)
            .map { Double($0.0) - Double($0.1) }
            .map { $0 * $0 }
            .reduce(0, +).squareRoot()
        XCTAssertGreaterThan(
            distance, 20,
            "The two records exist but their rendered thumbnail interiors are indistinguishable"
        )
    }

    private func dismissFirstRunIfNeeded() {
        let decline = element("TelemetryConsent.DontShare")
        if decline.waitForExistence(timeout: 5) { decline.click() }
        let skip = element("Quickstart.Skip")
        if skip.waitForExistence(timeout: 10) { skip.click() }
    }

    private func element(_ identifier: String) -> XCUIElement {
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
    private func centerMeanRGB(_ png: Data) throws -> [CGFloat] {
        guard let image = NSImage(data: png),
              let source = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else { throw XCTSkip("XCTest returned an undecodable screenshot") }
        let insetX = source.width / 5
        let insetY = source.height / 5
        let rect = CGRect(
            x: insetX, y: insetY,
            width: source.width - 2 * insetX,
            height: source.height - 2 * insetY
        )
        guard let cropped = source.cropping(to: rect) else {
            throw XCTSkip("thumbnail screenshot was too small to crop")
        }
        let rep = NSBitmapImageRep(cgImage: cropped)
        var totals = [CGFloat](repeating: 0, count: 3)
        var count: CGFloat = 0
        for y in stride(from: 0, to: rep.pixelsHigh, by: 2) {
            for x in stride(from: 0, to: rep.pixelsWide, by: 2) {
                guard let color = rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                totals[0] += color.redComponent
                totals[1] += color.greenComponent
                totals[2] += color.blueComponent
                count += 1
            }
        }
        XCTAssertGreaterThan(count, 0)
        return totals.map { $0 / count * 255 }
    }
}
