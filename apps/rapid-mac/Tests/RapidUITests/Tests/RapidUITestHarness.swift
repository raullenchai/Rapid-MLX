import AppKit
import Darwin
import XCTest

/// Bounded edge follower for the production memory confirmation presented to
/// native GUI tests on a busy hosted Mac. A posted click is not proof SwiftUI
/// consumed it, so an unchanged presentation gets spaced retries. The cap
/// prevents a stuck alert from being hammered for the whole readiness timeout.
struct MemoryConfirmationRetryPolicy {
    static let maximumAttempts = 3
    static let retryPollInterval = 10

    private(set) var attempts = 0
    private var pollsSinceAttempt = 0
    private var presentationSignature: String?

    mutating func shouldClick(signature: String?, isEnabled: Bool) -> Bool {
        guard let signature else {
            attempts = 0
            pollsSinceAttempt = 0
            presentationSignature = nil
            return false
        }
        if signature != presentationSignature {
            attempts = 0
            pollsSinceAttempt = 0
            presentationSignature = signature
        }
        pollsSinceAttempt += 1
        guard isEnabled,
              attempts < Self.maximumAttempts,
              attempts == 0 || pollsSinceAttempt >= Self.retryPollInterval else {
            return false
        }
        attempts += 1
        pollsSinceAttempt = 0
        return true
    }

    @MainActor
    mutating func follow(_ confirmation: XCUIElement) {
        let isPresent = confirmation.exists
        let signature = isPresent
            ? [
                confirmation.identifier,
                confirmation.label,
                String(describing: confirmation.value),
            ].joined(separator: "\u{1F}")
            : nil
        if shouldClick(
            signature: signature,
            isEnabled: isPresent && confirmation.isEnabled
        ) {
            confirmation.click()
        }
    }
}

enum FileDropRetryPolicy {
    // Long enough to observe the product-owned completion marker on the slow
    // hosted runner, but bounded so a transport miss reaches its one allowed
    // fresh-session retry promptly.
    static let completionObservationTimeout: TimeInterval = 4.5
    static func observationTimeout(settleTimeout: TimeInterval) -> TimeInterval {
        min(max(0, settleTimeout), completionObservationTimeout)
    }

    static func shouldRetry(
        completedDrop: Bool,
        transportFailed: Bool,
        attempt: Int,
        maximumAttempts: Int
    ) -> Bool {
        !completedDrop
            && transportFailed
            && attempt < maximumAttempts
    }
}

enum DragTransportFile {
    enum Result: String {
        case notStarted = "not-started"
        case none
        case copy
        case other

        var isAuthoritativeFailure: Bool { self == .notStarted || self == .none }
    }

    enum ResultError: Error, Equatable {
        case invalidResult(String)
    }

    static func result(at url: URL, fileManager: FileManager = .default) throws -> Result? {
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        let value = try String(contentsOf: url, encoding: .utf8)
        guard let result = Result(rawValue: value) else {
            throw ResultError.invalidResult(value)
        }
        return result
    }
}

enum DropEventFile {
    enum EventError: Error, Equatable {
        case remainedAfterRemoval
        case invalidPhase(String)
    }

    static func clear(at url: URL, fileManager: FileManager = .default) throws {
        do {
            try fileManager.removeItem(at: url)
        } catch let error as CocoaError where error.code == .fileNoSuchFile {
            // An absent marker is the required pre-drag state.
        }
        guard !fileManager.fileExists(atPath: url.path) else {
            throw EventError.remainedAfterRemoval
        }
    }

    static func completedPhase(
        at url: URL,
        fileManager: FileManager = .default
    ) throws -> String? {
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        let phase = try String(contentsOf: url, encoding: .utf8)
        guard phase == "performed" else { throw EventError.invalidPhase(phase) }
        return phase
    }
}

@MainActor
final class RapidUITestHarness {
    let app: XCUIApplication
    let eventLog: URL
    let rapidMacRoot: URL

    private let testHome: URL
    private let conversationStore: URL
    private let sidecarAlias: String
    private let sidecarPIDFile: URL
    private let dropEventFile: URL
    private var portReservation: Int32?
    private var originalPasteboardItems: [[NSPasteboard.PasteboardType: Data]]?
    private var ownedPasteboardChangeCount: Int?
    private var activeFileDragSource: XCUIApplication?

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

    init(
        testName: String,
        fakeSettings: [String: String],
        sidecarAlias explicitSidecarAlias: String? = nil
    ) throws {
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
        dropEventFile = testHome.appendingPathComponent("xcui-drop-event.txt")
        sidecarAlias = explicitSidecarAlias ?? (
            fakeSettings["FAKE_VISION_CHAT"] == "1"
                ? "qwen3-vl-2b-4bit"
                : "fake-alias"
        )

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
            "RAPID_XCUI_DROP_EVENT_FILE": dropEventFile.path,
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
        if let activeFileDragSource {
            _ = terminateFileDragSource(activeFileDragSource)
        }
        app.terminate()
        _ = app.wait(for: .notRunning, timeout: 5)
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
        var memoryConfirmationPolicy = MemoryConfirmationRetryPolicy()
        XCTAssertTrue(waitUntil(timeout: 60) {
            if self.serverStartCount() > priorServerStartCount { return true }
            memoryConfirmationPolicy.follow(memoryConfirmation)
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

    /// Drag ``url`` from the helper host app onto the compose field.
    ///
    /// ``expectedChip`` is the remove control the drop must produce, fetched at
    /// the call site via ``element(_:)`` (which keeps the query literal in the
    /// test source for the xcui workflow contract). A landed drop is treated as
    /// one whose chip settles (exists and is hittable). The product's compose
    /// destination emits a test-only marker after `performDragOperation`
    /// consumes the drop. The drag-source helper separately records AppKit's
    /// final transport result. A gesture is retried only when that result is
    /// authoritatively `none`/`not-started`; temporary absence of a product
    /// marker is never treated as failure. A consumed drop is never retried:
    /// if its chip does not appear, the test still exposes the product/AX
    /// regression. The
    /// chip is never dereferenced before it exists, so a not-yet-matched
    /// ``firstMatch`` cannot throw (#2481).
    /// Callers without an expected chip (the unsupported-file negative case)
    /// keep the original single-drop behaviour.
    @discardableResult
    func dragFile(
        _ url: URL,
        expectedChip chip: XCUIElement? = nil,
        dropSettleTimeout: TimeInterval = 14,
        simulateMissedFirstGesture: Bool = false,
        simulateChipVisibilityDelay: TimeInterval = 0,
        simulateCompletionVisibilityDelay: TimeInterval = 0
    ) -> Int {
        guard dropSettleTimeout > 0 else {
            XCTFail("file-drop settle timeout must be positive")
            return 0
        }
        guard let chip = chip else {
            let (dragSource, source, dropTarget, _) = launchFileDragSource(
                url: url,
                dropFirstGesture: simulateMissedFirstGesture
            )
            // Preserve scope-based cleanup for the negative unsupported-file
            // journey as well as requiring an orderly shutdown on its normal
            // path. An interrupted gesture must not leak a helper into a later
            // XCUITest.
            defer {
                if dragSource.state != .notRunning {
                    _ = terminateFileDragSource(dragSource)
                }
            }
            source.click(forDuration: 1, thenDragTo: dropTarget)
            guard terminateFileDragSource(dragSource) else { return 1 }
            return 1
        }
        let maximumAttempts = 2
        do {
            // Clear once for the whole logical drop. Never clear between
            // attempts: an acknowledgement that arrives while the helper is
            // being recycled must prevent replay of a consumed product drop.
            try DropEventFile.clear(at: dropEventFile)
        } catch {
            XCTFail("could not clear UI-test drop marker before gesture: \(error)")
            return 1
        }
        for attempt in 1...maximumAttempts {
            // Each bounded attempt owns a fresh helper process. A missed
            // gesture is a transport failure only when AppKit's source-side
            // session result says it never started or ended with no accepted
            // operation. Recycling the helper clears the stale AppKit
            // drag/mouse session without guessing from marker latency.
            let (dragSource, source, dropTarget, transportResultFile) = launchFileDragSource(
                url: url,
                dropFirstGesture: simulateMissedFirstGesture && attempt == 1
            )
            source.click(forDuration: 1, thenDragTo: dropTarget)
            // Startup, termination and the blocking synthetic gesture are
            // bounded separately. Each completed gesture gets the full,
            // honest post-gesture settle window.
            let settleDeadline = Date().addingTimeInterval(dropSettleTimeout)
            // Simulation delays model post-gesture observation latency. Anchor
            // them after the blocking drag returns so its duration cannot
            // accidentally satisfy the delay before the probe begins.
            let chipObservationStart = Date().addingTimeInterval(
                simulateChipVisibilityDelay
            )
            let completionObservationStart = Date().addingTimeInterval(
                simulateCompletionVisibilityDelay
            )
            let chipIsSettled = {
                Date() >= chipObservationStart && chip.exists && chip.isHittable
            }
            let completionIsVisible = {
                Date() >= completionObservationStart
                    && FileManager.default.fileExists(atPath: self.dropEventFile.path)
            }
            let transportResultIsVisible = {
                FileManager.default.fileExists(atPath: transportResultFile.path)
            }

            // The drop-completion marker and the product render arrive
            // independently. First wait briefly for either authoritative
            // signal, then spend the rest of the original budget on an
            // observed drop's chip.
            let observationTimeout = FileDropRetryPolicy.observationTimeout(
                settleTimeout: settleDeadline.timeIntervalSinceNow
            )
            _ = waitUntil(timeout: observationTimeout) {
                chipIsSettled()
                    || completionIsVisible()
                    || transportResultIsVisible()
            }
            if chipIsSettled() {
                guard terminateFileDragSource(dragSource) else { return attempt }
                return attempt
            }

            var observedPhase: String?
            let transportResult: DragTransportFile.Result?
            do {
                observedPhase = completionIsVisible()
                    ? try DropEventFile.completedPhase(at: dropEventFile)
                    : nil
                transportResult = try DragTransportFile.result(at: transportResultFile)
            } catch {
                _ = terminateFileDragSource(dragSource)
                XCTFail("could not read valid UI-test drag result after gesture: \(error)")
                return attempt
            }
            guard terminateFileDragSource(dragSource) else { return attempt }
            // Helper shutdown is the final source-session boundary. Re-read
            // the destination acknowledgement after that bounded wait so a
            // completion that arrived during termination always vetoes replay.
            if observedPhase == nil {
                do {
                    observedPhase = try DropEventFile.completedPhase(at: dropEventFile)
                } catch {
                    XCTFail("could not read final UI-test drop marker: \(error)")
                    return attempt
                }
            }
            if FileDropRetryPolicy.shouldRetry(
                completedDrop: observedPhase != nil,
                transportFailed: transportResult?.isAuthoritativeFailure == true,
                attempt: attempt,
                maximumAttempts: maximumAttempts
            ) {
                continue
            }

            // A source-side `.copy` is authoritative acceptance even if the
            // destination marker has not become visible yet. Keep the same
            // post-gesture deadline alive for the independent marker/render
            // signals; never turn accepted transport into an immediate fail.
            let acceptedDrop = observedPhase != nil || transportResult == .copy
            if acceptedDrop {
                let acceptedRemaining = max(0, settleDeadline.timeIntervalSinceNow)
                _ = waitUntil(timeout: acceptedRemaining) {
                    chipIsSettled() || completionIsVisible()
                }
            }
            let chipRemaining = max(0, settleDeadline.timeIntervalSinceNow)
            if waitUntil(timeout: chipRemaining, condition: chipIsSettled) {
                return attempt
            }
            XCTFail(
                "dropped attachment chip did not settle within \(dropSettleTimeout)s "
                    + "(drop phase: \(observedPhase ?? "not performed"), "
                    + "transport: \(transportResult?.rawValue ?? "missing"), "
                    + "attempts: \(attempt))"
            )
            return attempt
        }
        return maximumAttempts
    }

    private func launchFileDragSource(
        url: URL,
        dropFirstGesture: Bool
    ) -> (app: XCUIApplication, source: XCUIElement, target: XCUIElement, result: URL) {
        let dragSource = XCUIApplication(bundleIdentifier: "com.rapidmlx.rapid-uitest-host")
        let resultFile = testHome.appendingPathComponent(
            "drag-transport-\(UUID().uuidString).txt"
        )
        dragSource.launchEnvironment = [
            "RAPID_XCUI_DRAG_FILE": url.path,
            "RAPID_XCUI_DROP_FIRST_GESTURE": dropFirstGesture ? "1" : "0",
            "RAPID_XCUI_DRAG_RESULT_FILE": resultFile.path,
        ]
        dragSource.launch()
        // Track the helper immediately after launch. If any subsequent setup
        // assertion aborts this journey, harness shutdown still owns and
        // terminates the process before the next test starts.
        activeFileDragSource = dragSource
        let source = dragSource.descendants(matching: .any)
            .matching(identifier: "RapidUITests.FileDragSource").firstMatch
        XCTAssertTrue(source.waitForExistence(timeout: 15))

        // Exercise the native text editor itself. The editor must explicitly
        // register for file URLs; otherwise AppKit inserts the path as text
        // before SwiftUI's enclosing drop destination can handle the event.
        let dropTarget = element("rapid.chat.compose")
        XCTAssertTrue(dropTarget.waitForExistence(timeout: 10))
        // The synthetic drop must land on a laid-out target. The compose field
        // can exist in the AX tree before its final frame after model start or
        // after the helper is relaunched for a bounded retry.
        XCTAssertTrue(
            waitUntil(timeout: 10) { dropTarget.isHittable },
            "compose drop target never became hittable before drag"
        )
        return (dragSource, source, dropTarget, resultFile)
    }

    private func terminateFileDragSource(_ dragSource: XCUIApplication) -> Bool {
        if dragSource.state == .notRunning {
            activeFileDragSource = nil
            return true
        }
        dragSource.terminate()
        let terminated = dragSource.wait(for: .notRunning, timeout: 5)
        if !terminated {
            XCTFail("file-drag helper did not terminate; retry suppressed")
        } else {
            activeFileDragSource = nil
        }
        return terminated
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
        events().filter {
            $0["event"] as? String == "chat_request"
                && $0["request_origin"] as? String != "background_assist"
        }
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

    /// Take the chip fetched via ``element(_:)`` at the call site (which
    /// keeps the query literal in the test source for the xcui workflow
    /// contract) and wait until the remove control it names has settled —
    /// ``exists`` and ``isHittable`` — so it is fully rendered on-screen.
    /// Returns the settled element. Reuses ``waitUntil`` (XCUIElement's
    /// ``exists`` and ``isHittable`` re-query the AX tree on every poll, so the
    /// stale-capture and mid-animation races a one-shot ``waitForExistence``
    /// can miss are covered) and the chip is never dereferenced before it
    /// exists, so a not-yet-matched ``firstMatch`` cannot throw (#2481).
    @discardableResult
    func waitForAttachmentRemove(
        _ chip: XCUIElement,
        timeout: TimeInterval = 15
    ) -> XCUIElement {
        if !waitUntil(timeout: timeout, condition: { chip.exists && chip.isHittable }) {
            if !chip.exists {
                XCTFail("Attachment remove control never appeared within \(timeout)s")
            } else {
                XCTFail("Attachment remove control never became hittable within \(timeout)s")
            }
        }
        return chip
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
