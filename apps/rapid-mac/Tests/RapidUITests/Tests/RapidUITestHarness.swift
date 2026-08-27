import AppKit
import Darwin
import XCTest

@MainActor
final class RapidUITestHarness {
    let app: XCUIApplication
    let eventLog: URL
    let rapidMacRoot: URL

    private let testHome: URL
    private let conversationStore: URL
    private let sidecarAlias: String
    private let sidecarPIDFile: URL
    private var portReservation: Int32?
    private var originalPasteboardItems: [[NSPasteboard.PasteboardType: Data]]?
    private var ownedPasteboardChangeCount: Int?

    private static func reserveLoopbackPort() throws -> (descriptor: Int32, port: Int) {
        let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw POSIXError(.ENOTSOCK) }

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            Darwin.close(descriptor)
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EADDRINUSE)
        }

        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let resolved = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(descriptor, $0, &length)
            }
        }
        guard resolved == 0 else {
            Darwin.close(descriptor)
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EINVAL)
        }
        return (descriptor, Int(UInt16(bigEndian: address.sin_port)))
    }

    init(testName: String, fakeSettings: [String: String]) throws {
        let reservedPort = try Self.reserveLoopbackPort()
        var reservationTransferred = false
        defer {
            if !reservationTransferred { Darwin.close(reservedPort.descriptor) }
        }
        portReservation = reservedPort.descriptor
        testHome = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-xcui-\(testName)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: testHome, withIntermediateDirectories: true)
        conversationStore = testHome
            .appendingPathComponent("Library/Application Support/com.rapidmlx.rapid")
            .appendingPathComponent("conversations.json")

        rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // RapidUITests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
        let fakeSidecar = rapidMacRoot.appendingPathComponent("scripts/fake-rapid-mlx.sh").path
        let appURL = rapidMacRoot.appendingPathComponent("build/Rapid-MLX Desktop.app")
        eventLog = testHome.appendingPathComponent("fake-events.jsonl")
        sidecarPIDFile = testHome.appendingPathComponent("fake-sidecar.pid")
        sidecarAlias = fakeSettings["FAKE_VISION_CHAT"] == "1"
            ? "qwen3-vl-2b-4bit"
            : "fake-alias"

        var config = fakeSettings
        config["FAKE_EVENT_LOG"] = eventLog.path
        config["FAKE_PID_FILE"] = sidecarPIDFile.path
        let configData = try JSONSerialization.data(withJSONObject: config)
        try configData.write(to: testHome.appendingPathComponent(".rapid-golden-fake.json"))

        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: fakeSidecar))
        XCTAssertTrue(FileManager.default.fileExists(atPath: appURL.path))
        app = XCUIApplication(url: appURL)
        app.launchArguments += [
            "-com.rapidmlx.rapid.telemetry.enabled", "false",
        ]
        app.launchEnvironment = [
            "HOME": testHome.path,
            "CFFIXED_USER_HOME": testHome.path,
            "RAPID_BIN": fakeSidecar,
            "FAKE_EVENT_LOG": eventLog.path,
            "RAPID_DESKTOP_PORT": String(reservedPort.port),
            "RAPID_DESKTOP_NO_PORT_SWEEP": "1",
        ].merging(fakeSettings) { _, fixture in fixture }
        reservationTransferred = true
    }

    func launch() {
        app.launch()
        XCTAssertTrue(app.windows["Rapid-MLX"].waitForExistence(timeout: 20))
        dismissFirstRunIfNeeded()
    }

    func relaunch() {
        app.terminate()
        terminateFakeSidecars()
        releasePortReservation()
        do {
            let reservedPort = try Self.reserveLoopbackPort()
            portReservation = reservedPort.descriptor
            app.launchEnvironment["RAPID_DESKTOP_PORT"] = String(reservedPort.port)
        } catch {
            XCTFail("Could not reserve a fresh loopback port for relaunch: \(error)")
            return
        }
        app.launch()
        XCTAssertTrue(app.windows["Rapid-MLX"].waitForExistence(timeout: 20))
        dismissFirstRunIfNeeded()
    }

    func shutDown() {
        app.terminate()
        releasePortReservation()
        terminateFakeSidecars()
        restorePasteboardIfOwned()
        try? FileManager.default.removeItem(at: testHome)
    }

    func startModel() {
        let readiness = element("Readiness.Action")
        XCTAssertTrue(readiness.waitForExistence(timeout: 20))
        XCTAssertTrue(waitUntil(timeout: 20) { readiness.isEnabled })
        let priorServerStartCount = serverStartCount()
        // Hold the OS-selected port until the app is ready to spawn its fake
        // sidecar, reducing the bind race to the click-to-process-launch edge.
        releasePortReservation()
        readiness.click()
        let memoryConfirmation = element("MemoryWarning.Confirm")
        var confirmedMemoryWarning = false
        XCTAssertTrue(waitUntil(timeout: 60) {
            if !confirmedMemoryWarning,
               memoryConfirmation.exists,
               memoryConfirmation.isEnabled {
                memoryConfirmation.click()
                confirmedMemoryWarning = true
            }
            return self.serverStartCount() > priorServerStartCount
        })
    }

    func waitForConversationPersistence(containing markers: [String]) {
        XCTAssertTrue(waitUntil(timeout: 20) {
            guard let persisted = try? String(
                contentsOf: self.conversationStore,
                encoding: .utf8
            ) else { return false }
            return markers.allSatisfy(persisted.contains)
        })
    }

    func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    func element(label: String) -> XCUIElement {
        app.descendants(matching: .any).matching(
            NSPredicate(format: "label == %@", label)
        ).firstMatch
    }

    func staticText(valuePrefix prefix: String) -> XCUIElement {
        // SwiftUI exposes a combined, line-limited accessibility label as the
        // AX value of a StaticText on hosted macOS. Constraining the query to
        // that element type also avoids an expensive value predicate across
        // the entire application hierarchy.
        app.staticTexts.matching(
            NSPredicate(format: "value BEGINSWITH %@", prefix)
        ).firstMatch
    }

    func messageAction(_ action: String) -> XCUIElement {
        app.descendants(matching: .any).matching(
            NSPredicate(
                format: "identifier MATCHES %@",
                "^ChatView\\.Message\\.\(action)\\.[0-9A-Fa-f-]{36}$"
            )
        ).firstMatch
    }

    func conversationRows() -> XCUIElementQuery {
        app.descendants(matching: .any).matching(
            NSPredicate(
                format: "identifier MATCHES %@",
                #"^Sidebar\.Conversation\.[0-9A-Fa-f-]{36}$"#
            )
        )
    }

    func chooseFile(_ url: URL, actionIdentifier: String) {
        let add = element("ChatView.AddAttachments")
        XCTAssertTrue(add.waitForExistence(timeout: 10))
        add.click()
        let action = element(actionIdentifier)
        XCTAssertTrue(action.waitForExistence(timeout: 10))
        XCTAssertTrue(action.isEnabled)
        action.click()

        // NSOpenPanel has no stable product-owned identifiers. “Go to Folder”
        // is the native keyboard path and avoids coordinate clicks entirely.
        app.typeKey("g", modifierFlags: [.command, .shift])
        app.typeText(url.path)
        app.typeKey(.return, modifierFlags: [])
        let open = app.dialogs["open-panel"].buttons["OKButton"]
        XCTAssertTrue(waitUntil(timeout: 10) { open.isHittable })
        open.click()
    }

    func dragFile(_ url: URL) {
        let dragSource = XCUIApplication(bundleIdentifier: "com.rapidmlx.rapid-uitest-host")
        dragSource.launchEnvironment = ["RAPID_XCUI_DRAG_FILE": url.path]
        dragSource.launch()
        defer { dragSource.terminate() }
        let source = dragSource.descendants(matching: .any)
            .matching(identifier: "RapidUITests.FileDragSource").firstMatch
        XCTAssertTrue(source.waitForExistence(timeout: 15))
        // Exercise the native text editor itself. The editor must explicitly
        // register for file URLs; otherwise AppKit inserts the path as text
        // before SwiftUI's enclosing drop destination can handle the event.
        let dropTarget = element("rapid.chat.compose")
        XCTAssertTrue(dropTarget.waitForExistence(timeout: 10))
        source.click(forDuration: 1, thenDragTo: dropTarget)
    }

    func pasteImage(_ url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let image = NSImage(data: data) else {
            XCTFail("Could not decode image for native pasteboard journey")
            return
        }
        let pasteboard = NSPasteboard.general
        let stillOwnsPasteboard = ownedPasteboardChangeCount != nil
            && pasteboard.changeCount == ownedPasteboardChangeCount
        if originalPasteboardItems == nil || !stillOwnsPasteboard {
            originalPasteboardItems = pasteboard.pasteboardItems?.map { item in
                Dictionary(uniqueKeysWithValues: item.types.compactMap { type in
                    item.data(forType: type).map { (type, $0) }
                })
            } ?? []
        }
        pasteboard.clearContents()
        // Use AppKit's image pasteboard writer instead of publishing only a
        // bare PNG representation. This matches a native image copy and makes
        // NSImage(pasteboard:) portable across hosted macOS image versions.
        XCTAssertTrue(pasteboard.writeObjects([image]))
        ownedPasteboardChangeCount = pasteboard.changeCount
        XCTAssertNotNil(NSImage(pasteboard: pasteboard))
        let composer = element("rapid.chat.compose")
        XCTAssertTrue(composer.waitForExistence(timeout: 10))
        composer.click()
        composer.typeKey("v", modifierFlags: .command)
    }

    private func restorePasteboardIfOwned() {
        let pasteboard = NSPasteboard.general
        guard let originalPasteboardItems,
              pasteboard.changeCount == ownedPasteboardChangeCount else { return }
        let items = originalPasteboardItems.map { representations in
            let item = NSPasteboardItem()
            for (type, data) in representations {
                item.setData(data, forType: type)
            }
            return item
        }
        pasteboard.clearContents()
        if !items.isEmpty { pasteboard.writeObjects(items) }
    }

    private func releasePortReservation() {
        guard let portReservation else { return }
        Darwin.close(portReservation)
        self.portReservation = nil
    }

    func send(_ text: String, expectedRequestCount: Int) {
        let composer = element("rapid.chat.compose")
        XCTAssertTrue(composer.waitForExistence(timeout: 10))
        composer.click()
        composer.typeText(text)
        let send = element("ChatView.SendOrStopButton")
        XCTAssertTrue(waitUntil(timeout: 10) { send.isEnabled })
        send.click()
        XCTAssertTrue(waitUntil(timeout: 30) { self.chatRequests().count == expectedRequestCount })
        XCTAssertTrue(waitUntil(timeout: 30) {
            self.element("ChatView.SendOrStopButton").label == "Send message"
        })
    }

    func retryResponse(expectedRequestCount: Int) {
        let retry = messageAction("Retry")
        XCTAssertTrue(retry.waitForExistence(timeout: 10))
        XCTAssertTrue(waitUntil(timeout: 60) { retry.isEnabled })
        retry.click()
        XCTAssertTrue(waitUntil(timeout: 30) { self.chatRequests().count == expectedRequestCount })
        XCTAssertTrue(waitUntil(timeout: 30) {
            self.element("ChatView.SendOrStopButton").label == "Send message"
        })
    }

    func chatRequests() -> [[String: Any]] {
        events().filter { $0["event"] as? String == "chat_request" }
    }

    @discardableResult
    func waitUntil(timeout: TimeInterval, condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline
        return condition()
    }

    private func dismissFirstRunIfNeeded() {
        let skip = element("Quickstart.Skip")
        if skip.waitForExistence(timeout: 10) { skip.click() }
    }

    private func events() -> [[String: Any]] {
        guard let text = try? String(contentsOf: eventLog, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            guard let data = line.data(using: .utf8) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
    }

    private func serverStartCount() -> Int {
        events().count { $0["event"] as? String == "server_started" }
    }

    private func terminateFakeSidecars() {
        var pids: Set<Int32> = Set(events().compactMap { event in
            guard event["event"] as? String == "server_started",
                  event["alias"] as? String == sidecarAlias,
                  let pid = event["pid"] as? NSNumber else { return nil }
            return pid.int32Value
        })
        if let text = try? String(contentsOf: sidecarPIDFile, encoding: .utf8),
           let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            pids.insert(pid)
        }
        for pid in pids where processCommand(pid: pid).contains("serve \(sidecarAlias)") {
            Darwin.kill(pid, SIGTERM)
            for _ in 0..<20 where Darwin.kill(pid, 0) == 0 {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if Darwin.kill(pid, 0) == 0,
               processCommand(pid: pid).contains("serve \(sidecarAlias)") {
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
}
