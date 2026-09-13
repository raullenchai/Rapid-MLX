import AppKit
import XCTest

@MainActor
final class ImageGenerationPixelTests: XCTestCase {
    private var activeHarness: RapidUITestHarness?

    override func tearDown() {
        activeHarness?.shutDown()
        activeHarness = nil
        super.tearDown()
    }

    func testMemoryConfirmationRetriesAreSpacedBoundedAndRearmed() {
        var policy = MemoryConfirmationRetryPolicy()

        XCTAssertTrue(policy.shouldClick(signature: "load", isEnabled: true))
        for _ in 1..<MemoryConfirmationRetryPolicy.retryPollInterval {
            XCTAssertFalse(policy.shouldClick(signature: "load", isEnabled: true))
        }
        XCTAssertTrue(policy.shouldClick(signature: "load", isEnabled: true))
        for _ in 1..<MemoryConfirmationRetryPolicy.retryPollInterval {
            XCTAssertFalse(policy.shouldClick(signature: "load", isEnabled: true))
        }
        XCTAssertTrue(policy.shouldClick(signature: "load", isEnabled: true))
        for _ in 0..<(MemoryConfirmationRetryPolicy.retryPollInterval * 2) {
            XCTAssertFalse(policy.shouldClick(signature: "load", isEnabled: false))
            XCTAssertFalse(policy.shouldClick(signature: "load", isEnabled: true))
        }

        XCTAssertTrue(policy.shouldClick(signature: "load-anyway", isEnabled: true))
        XCTAssertFalse(policy.shouldClick(signature: nil, isEnabled: false))
        XCTAssertTrue(policy.shouldClick(signature: "load-anyway", isEnabled: true))
    }

    func testTwoImageRendersDrawDistinctThumbnailPixels() throws {
        continueAfterFailure = false
        let harness = try RapidUITestHarness(
            testName: "image-generation-pixels",
            fakeSettings: [
                "FAKE_IMAGE_STEPS": "8",
                "FAKE_IMAGE_STEP_MS": "300",
            ],
            sidecarAlias: "fake-image-alias"
        )
        activeHarness = harness
        harness.launch()
        let app = harness.app
        let eventLog = harness.eventLog
        let images = element("Sidebar.Images", in: app)
        XCTAssertTrue(images.waitForExistence(timeout: 10))
        images.click()

        // Catalog discovery is asynchronous. Wait for the Images picker to
        // resolve before pressing the shared readiness control; otherwise the
        // click can still target the previously selected chat model.
        let picker = element("Images.ModelPicker", in: app)
        XCTAssertTrue(picker.waitForExistence(timeout: 20))
        XCTAssertTrue(waitUntil(timeout: 20) {
            picker.label.contains("fake-image-alias")
        })

        harness.startModel()

        let prompt = element("Images.Prompt", in: app)
        XCTAssertTrue(prompt.waitForExistence(timeout: 20))
        prompt.click()
        prompt.typeText("a cheetah on a red couch")
        let generate = element("Images.Generate", in: app)
        XCTAssertTrue(waitUntil(timeout: 30) { generate.isEnabled })
        generate.click()

        XCTAssertTrue(waitUntil(timeout: 30) { imageResponseCount(in: eventLog) == 1 })
        let first = element("Images.Gallery.Thumb.1", in: app)
        XCTAssertTrue(first.waitForExistence(timeout: 30))

        prompt.click()
        prompt.typeKey("a", modifierFlags: .command)
        prompt.typeText("the same cheetah, at night")
        XCTAssertTrue(waitUntil(timeout: 10) { generate.isEnabled })
        generate.click()
        XCTAssertTrue(waitUntil(timeout: 30) { imageResponseCount(in: eventLog) == 2 })

        let newest = element("Images.Gallery.Thumb.1", in: app)
        let older = element("Images.Gallery.Thumb.2", in: app)
        XCTAssertTrue(older.waitForExistence(timeout: 30))

        // Capture each record while it has the same selected styling. The
        // center crop already removes the stroke, and equalizing selection
        // also prevents any future interior selection treatment from being
        // mistaken for different generated pixels.
        older.click()
        let olderShot = older.screenshot()
        newest.click()
        let newestShot = newest.screenshot()
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

    private func imageResponseCount(in eventLog: URL) -> Int {
        guard let events = try? String(contentsOf: eventLog, encoding: .utf8) else { return 0 }
        return events.split(separator: "\n").count { $0.contains(#""event": "image_response""#) }
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
