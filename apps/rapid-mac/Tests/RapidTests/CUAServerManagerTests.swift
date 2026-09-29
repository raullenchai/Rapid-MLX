import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use model-free server")
struct CUAServerManagerTests {
    @Test("Cold panel entry starts the sidecar before a session model exists")
    @MainActor
    func coldPanelEntryStartsSession() async {
        var launches = 0
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: { 7_658 },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in true },
            launcher: { _, _, _, _, _, _ in
                launches += 1
                return ProcessGroupChild.testStub()
            }
        )
        #expect(manager.viewModel == nil)

        manager.startIfNeeded()
        for _ in 0..<20 where manager.state != .ready {
            await Task.yield()
        }

        #expect(launches == 1)
        #expect(manager.state == .ready)
        #expect(manager.viewModel != nil)
    }

    @Test("Launch arguments select CUA-only mode without a model")
    func argumentsAreModelFree() {
        let arguments = CUAServerManager.serveArguments(host: "127.0.0.1", port: 7_659)
        #expect(arguments.prefix(2) == ["serve", "--cua-only"])
        #expect(arguments.contains("127.0.0.1"))
        #expect(arguments.contains("7659"))
        #expect(!arguments.contains(where: { $0 == "--model" }))
        #expect(!arguments.contains(where: { $0.contains("qwen") }))
        #expect(!Set(CUAServerManager.candidatePorts).isEmpty)
    }

    @Test("Packaged permission helper runs Python in its own process identity")
    func permissionHelperLaunchContract() throws {
        let outer = URL(fileURLWithPath: "/Applications/Rapid-MLX Desktop.app")
        let helper = outer.appendingPathComponent(
            "Contents/Helpers/Rapid Computer Use.app/Contents/MacOS/RapidComputerUse"
        )
        let located = CUAServerManager.locateExecutable(
            bundleURL: outer,
            isExecutable: { $0 == helper.path }
        )
        #expect(located == helper)
        #expect(CUAServerManager.isPermissionHelper(helper))

        let arguments = CUAServerManager.serveArguments(
            host: "127.0.0.1", port: 7_659, usesEmbeddedPython: true
        )
        #expect(arguments.prefix(5) == ["-P", "-u", "-s", "-m", "rapid_mlx.cli"])
        #expect(arguments.dropFirst(5).prefix(2) == ["serve", "--cua-only"])

        let environment = CUAServerManager.embeddedPythonEnvironment(
            helperExecutable: helper
        )
        #expect(environment["PYTHONHOME"] == outer.path + "/Contents/Resources/rapid-mlx/python")
        #expect(environment["PYTHONPATH"] == outer.path + "/Contents/Resources/rapid-mlx/site-packages")
        #expect(environment["PYTHONNOUSERSITE"] == "1")
        #expect(environment["PYTHONDONTWRITEBYTECODE"] == "1")
    }

    @Test("An occupied chat port is skipped without sweeping its listener")
    func occupiedPortIsNeverSwept() throws {
        let listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        #expect(listener >= 0)
        defer { close(listener) }
        var reuse: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        try #require(bound == 0)
        try #require(listen(listener, 1) == 0)
        var liveAddress = sockaddr_in()
        var liveAddressLength = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &liveAddress) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(listener, $0, &liveAddressLength)
            }
        }
        try #require(named == 0)
        let occupiedPort = Int(UInt16(bigEndian: liveAddress.sin_port))
        let freePort = try #require((60_000...60_100).first {
            PortAllocator.canBind(port: $0, host: "127.0.0.1")
        })

        let selected = CUAServerManager.selectPort(
            candidates: [occupiedPort, freePort], reserved: []
        ) { PortAllocator.canBind(port: $0, host: "127.0.0.1") }

        #expect(selected == freePort)
        #expect(fcntl(listener, F_GETFD) != -1)
        #expect(!PortAllocator.canBind(port: occupiedPort, host: "127.0.0.1"))
    }

    @Test("Repeated panel appearances share one model-free child")
    @MainActor
    func repeatedEnsureRunsOnce() async {
        var launches = 0
        var capturedArguments: [String] = []
        var capturedEnvironment: [String: String] = [:]
        var outputWasDrained = false
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: {
                try? await Task.sleep(nanoseconds: 20_000_000)
                return 7_659
            },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in true },
            launcher: { _, arguments, environment, output, errors, _ in
                launches += 1
                capturedArguments = arguments
                capturedEnvironment = environment
                outputWasDrained = output.fileHandleForReading.readabilityHandler != nil
                    && errors.fileHandleForReading.readabilityHandler != nil
                return ProcessGroupChild.testStub()
            }
        )

        async let first: Void = manager.ensureRunning()
        async let second: Void = manager.ensureRunning()
        _ = await (first, second)

        #expect(launches == 1)
        #expect(manager.state == .ready)
        #expect(manager.port == 7_659)
        #expect(manager.client != nil)
        let navigationSession = manager.viewModel
        navigationSession?.goal = "Format this document"
        navigationSession?.plannerName = "studio-glm-flash"
        navigationSession?.phase = .awaitingApproval
        navigationSession?.pendingApproval = CUAPendingApproval(
            gateID: "gate-1", app: "TextEdit", action: "click",
            target: "Bold", reason: "Changes formatting"
        )
        #expect(capturedArguments.first == "serve")
        #expect(capturedArguments.dropFirst().first == "--cua-only")
        #expect(capturedEnvironment["RAPID_MLX_API_KEY"] == "secret")
        #expect(capturedEnvironment["RAPID_MLX_WATCHDOG_PPID"] != nil)
        #expect(outputWasDrained)

        await manager.ensureRunning()
        #expect(launches == 1)
        #expect(manager.viewModel === navigationSession)
        #expect(manager.viewModel?.goal == "Format this document")
        #expect(manager.viewModel?.plannerName == "studio-glm-flash")
        #expect(manager.viewModel?.phase == .awaitingApproval)
        #expect(manager.viewModel?.pendingApproval?.gateID == "gate-1")

        await manager.stop()
        #expect(manager.viewModel == nil)

        await manager.retry()
        #expect(launches == 2)
        #expect(manager.viewModel != nil)
        #expect(manager.viewModel !== navigationSession)
        #expect(manager.viewModel?.phase == .idle)
    }

    @Test("Closed sidecar output pipes detach their readers while the session stays ready")
    @MainActor
    func outputReadersDetachAtEOF() async throws {
        var output: Pipe?
        var errors: Pipe?
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: { 7_675 },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in true },
            launcher: { _, _, _, stdout, stderr, _ in
                output = stdout
                errors = stderr
                return ProcessGroupChild.testStub()
            }
        )

        await manager.ensureRunning()
        let stdout = try #require(output)
        let stderr = try #require(errors)
        #expect(manager.state == .ready)
        #expect(stdout.fileHandleForReading.readabilityHandler != nil)
        #expect(stderr.fileHandleForReading.readabilityHandler != nil)

        // An open writer with no buffered data is not EOF. The handler must
        // stay installed until all writers close.
        try stdout.fileHandleForWriting.write(contentsOf: Data("log\n".utf8))
        try stderr.fileHandleForWriting.write(contentsOf: Data("error\n".utf8))
        for _ in 0..<20 { await Task.yield() }
        #expect(stdout.fileHandleForReading.readabilityHandler != nil)
        #expect(stderr.fileHandleForReading.readabilityHandler != nil)

        try stdout.fileHandleForWriting.close()
        try stderr.fileHandleForWriting.close()
        for _ in 0..<100 {
            if stdout.fileHandleForReading.readabilityHandler == nil
                && stderr.fileHandleForReading.readabilityHandler == nil { break }
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(manager.state == .ready)
        #expect(stdout.fileHandleForReading.readabilityHandler == nil)
        #expect(stderr.fileHandleForReading.readabilityHandler == nil)
        await manager.stop()
    }

    @Test("Unexpected sidecar exit preserves context but revokes run authority")
    @MainActor
    func unexpectedExitInterruptsAndRestoresReadOnlyContext() async throws {
        var exits: [@Sendable (ProcessGroupChild) -> Void] = []
        var children: [ProcessGroupChild] = []
        var nextPort = 7_671
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: {
                defer { nextPort += 1 }
                return nextPort
            },
            bearerProvider: { UUID().uuidString },
            readinessProbe: { _, _, _ in true },
            launcher: { _, _, _, _, _, termination in
                let child = ProcessGroupChild.testStub()
                children.append(child)
                exits.append(termination)
                return child
            }
        )
        await manager.ensureRunning()
        let originalSessionID = try #require(manager.sessionID)
        let original = try #require(manager.viewModel)
        original.goal = "Rename the selected item"
        original.plannerName = "remote-brain"
        original.maxSteps = 9
        original.phase = .awaitingApproval
        original.pendingApproval = CUAPendingApproval(
            gateID: "old-gate", app: "Finder", action: "press",
            target: "Rename", reason: "Changes a filename"
        )
        original.events = [
            CUAEvent(
                seq: 1, kind: "executed", step: 1, action: "click",
                stepInstruction: "Select Rename", outcome: nil,
                targetLabel: "Rename", status: nil, finalSummary: nil, reason: nil
            ),
        ]
        original.appOptions = [
            CUAAppOption(name: "Finder", bundleID: "com.apple.finder", pid: 42),
        ]
        original.selectedPID = 42
        original.windowOptions = [
            CUAWindowOption(
                windowID: "cg:77", index: 0, title: "Documents",
                x: 0, y: 0, width: 800, height: 600
            ),
        ]
        original.selectedWindowID = "cg:77"

        exits[0](children[0])
        for _ in 0..<20 where manager.state == .ready { await Task.yield() }

        #expect(manager.failureMessage != nil)
        #expect(manager.viewModel === original)
        #expect(original.isSessionDetached)
        #expect(original.wasSessionInterrupted)
        #expect(original.pendingApproval == nil)
        #expect(!original.canApprove)
        #expect(!original.canStart)
        #expect(original.selectedPID == nil)
        #expect(original.selectedWindowID == nil)
        #expect(original.events.count == 1)
        guard case let .failed(message) = original.phase else {
            Issue.record("active task did not become an interrupted failure")
            return
        }
        #expect(message.contains("cannot resume"))

        await manager.retry()
        let replacement = try #require(manager.viewModel)
        #expect(replacement !== original)
        #expect(manager.sessionID != originalSessionID)
        #expect(!replacement.isSessionDetached)
        #expect(replacement.wasSessionInterrupted)
        #expect(replacement.goal == "Rename the selected item")
        #expect(replacement.plannerName == "remote-brain")
        #expect(replacement.maxSteps == 9)
        #expect(replacement.events.count == 1)
        #expect(replacement.pendingApproval == nil)
        #expect(!replacement.canApprove)
        #expect(replacement.selectedPID == nil)
        #expect(replacement.selectedWindowID == nil)
    }

    @Test("Session rotation preserves idle draft and terminal history without targets")
    @MainActor
    func detachedPresentationContinuityIsLocalOnly() {
        let draft = CUAViewModel(api: nil)
        draft.goal = "Inspect this window"
        draft.plannerName = "brain"
        draft.selectedPID = 7
        draft.selectedWindowID = "cg:9"
        draft.detachFromSession()

        #expect(draft.phase == .idle)
        #expect(draft.goal == "Inspect this window")
        #expect(draft.selectedPID == nil)
        #expect(draft.selectedWindowID == nil)
        #expect(!draft.canStart)

        let terminal = CUAViewModel(api: nil)
        terminal.goal = "Read the title"
        terminal.phase = .finished(summary: "The title is Notes.")
        terminal.events = [
            CUAEvent(
                seq: 1, kind: "done", step: 1, action: nil,
                stepInstruction: nil, outcome: nil, targetLabel: nil,
                status: "completed", finalSummary: "The title is Notes.", reason: nil
            ),
        ]
        terminal.detachFromSession()

        let restored = CUAViewModel(api: nil)
        restored.restoreContinuity(from: terminal)
        #expect(restored.phase == .finished(summary: "The title is Notes."))
        #expect(restored.events == terminal.events)
        #expect(!restored.isSessionDetached)
        #expect(!restored.wasSessionInterrupted)
    }

    @Test("Explicit stop wins when the termination callback runs synchronously")
    @MainActor
    func explicitStopConsumesItsChildExit() async throws {
        var exitHandler: (@Sendable (ProcessGroupChild) -> Void)?
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: { 7_673 },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in true },
            launcher: { _, _, _, _, _, termination in
                exitHandler = termination
                return ProcessGroupChild.testStub()
            },
            stopSignaler: { child in exitHandler?(child) }
        )
        await manager.ensureRunning()
        let viewModel = try #require(manager.viewModel)
        viewModel.goal = "Keep this draft only while the service is alive"

        await manager.stop()

        #expect(manager.state == .idle)
        #expect(manager.viewModel == nil)
        #expect(manager.sessionID == nil)
        #expect(!viewModel.isSessionDetached)
    }

    @Test("Failed readiness is recoverable without duplicate retry children")
    @MainActor
    func failureCanRetry() async {
        var launches = 0
        var probes = 0
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: { 7_660 },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in
                probes += 1
                return probes > 1
            },
            launcher: { _, _, _, _, _, _ in
                launches += 1
                return ProcessGroupChild.testStub()
            }
        )

        await manager.ensureRunning()
        #expect(manager.failureMessage?.contains("didn't become ready") == true)
        #expect(manager.client == nil)

        async let first: Void = manager.retry()
        async let second: Void = manager.retry()
        _ = await (first, second)

        #expect(launches == 2)
        #expect(manager.state == .ready)
        #expect(manager.client != nil)
    }

    @Test("Missing engine fails without selecting or downloading a model")
    @MainActor
    func missingEngineFailsClosed() async {
        var launches = 0
        let manager = CUAServerManager(
            binaryPath: nil,
            binaryRefresh: { nil },
            launcher: { _, _, _, _, _, _ in
                launches += 1
                return ProcessGroupChild.testStub()
            }
        )

        await manager.ensureRunning()

        #expect(launches == 0)
        #expect(manager.client == nil)
        #expect(manager.failureMessage?.contains("unavailable") == true)
    }
}
