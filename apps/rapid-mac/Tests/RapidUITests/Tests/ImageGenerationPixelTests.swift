import AppKit
import Darwin
import XCTest

@MainActor
final class ImageGenerationPixelTests: XCTestCase {
    func testTwoImageRendersDrawDistinctThumbnailPixels() throws {
        continueAfterFailure = false
        let testHome = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-xcui-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: testHome, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: testHome) }

        let rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // RapidUITests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
        let fakeSidecar = rapidMacRoot.appendingPathComponent("scripts/fake-rapid-mlx.sh").path
        let appURL = rapidMacRoot.appendingPathComponent("build/Rapid-MLX Desktop.app")
        let eventLog = testHome.appendingPathComponent("fake-events.jsonl")
        // ServerManager deliberately sanitizes arbitrary FAKE_* variables
        // before spawning a sidecar. The fake's checked-in config file is the
        // durable channel shared with the AX GoldenFlow harness.
        let fakeConfig: [String: String] = [
            "FAKE_EVENT_LOG": eventLog.path,
            "FAKE_IMAGE_STEPS": "8",
            "FAKE_IMAGE_STEP_MS": "300",
        ]
        let fakeConfigData = try JSONSerialization.data(withJSONObject: fakeConfig)
        try fakeConfigData.write(to: testHome.appendingPathComponent(".rapid-golden-fake.json"))
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: fakeSidecar))
        XCTAssertTrue(FileManager.default.fileExists(atPath: appURL.path))
        let app = XCUIApplication(url: appURL)
        app.launchEnvironment = [
            "HOME": testHome.path,
            "CFFIXED_USER_HOME": testHome.path,
            "RAPID_BIN": fakeSidecar,
            "FAKE_EVENT_LOG": eventLog.path,
            "FAKE_IMAGE_STEPS": "8",
            "FAKE_IMAGE_STEP_MS": "300",
            // This XCUITest runs immediately before the AX GoldenFlows in CI.
            // Keep its fake away from the product's canonical :8000 and from
            // the operator-ownership regression fixture exercised there.
            "RAPID_DESKTOP_PORT": "65000",
            "RAPID_DESKTOP_NO_PORT_SWEEP": "1",
        ]
        app.launch()
        defer {
            app.terminate()
            terminateFakeSidecars(recordedIn: eventLog, alias: "fake-image-alias")
        }
        XCTAssertTrue(app.windows["Rapid-MLX"].waitForExistence(timeout: 20))
        dismissFirstRunIfNeeded(in: app)
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

        let readiness = element("Readiness.Action", in: app)
        XCTAssertTrue(readiness.waitForExistence(timeout: 20))
        readiness.click()
        XCTAssertTrue(waitUntil(timeout: 30) {
            guard let events = try? String(contentsOf: eventLog, encoding: .utf8) else { return false }
            return events.contains(#""event": "server_started""#)
                && events.contains(#""alias": "fake-image-alias""#)
        })

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

    /// XCUITest termination does not guarantee that an app-owned child has
    /// exited before the next workflow step starts. Reap only the exact fake
    /// PIDs recorded by this test, after verifying their command still names
    /// this fixture alias; this is both deterministic and PID-reuse safe.
    private func terminateFakeSidecars(recordedIn eventLog: URL, alias: String) {
        guard let text = try? String(contentsOf: eventLog, encoding: .utf8) else { return }
        let pids: Set<Int32> = Set(text.split(separator: "\n").compactMap { line in
            guard let data = line.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["event"] as? String == "server_started",
                  object["alias"] as? String == alias,
                  let pid = object["pid"] as? NSNumber else { return nil }
            return pid.int32Value
        })

        for pid in pids where processCommand(pid: pid).contains("serve \(alias)") {
            Darwin.kill(pid, SIGTERM)
            for _ in 0..<20 where Darwin.kill(pid, 0) == 0 {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if Darwin.kill(pid, 0) == 0,
               processCommand(pid: pid).contains("serve \(alias)") {
                Darwin.kill(pid, SIGKILL)
            }
        }
    }

    private func processCommand(pid: Int32) -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "command="]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        process.waitUntilExit()
        return String(
            data: output.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
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
