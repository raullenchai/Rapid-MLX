import Darwin
import Foundation
import Observation

enum CUAServerState: Equatable {
    case idle
    case starting
    case ready
    case failed(String)
}

/// Owns the model-free sidecar used by Computer Use.
///
/// This lifecycle is deliberately separate from ``ServerManager``. Opening the
/// Computer Use surface must not select, download, or retain a chat model, and
/// chat model replacement must not rotate the endpoint underneath an active
/// desktop automation task.
@MainActor
@Observable
final class CUAServerManager {
    typealias PortProvider = @MainActor () async -> Int?
    typealias BearerProvider = @MainActor () -> String
    typealias ReadinessProbe = @MainActor (String, Int, String) async -> Bool
    typealias StopSignaler = @MainActor (ProcessGroupChild) -> Void
    typealias Launcher = @MainActor (
        URL, [String], [String: String], Pipe, Pipe,
        @escaping @Sendable (ProcessGroupChild) -> Void
    ) throws -> ProcessGroupChild

    private(set) var state: CUAServerState
    private(set) var port: Int?
    private(set) var bearerToken: String?
    private(set) var sessionID: UUID?
    /// The automation session belongs to the sidecar lifecycle, not to the
    /// currently visible navigation destination. Keeping one view model here
    /// preserves its run identity, polling, approvals, and form state while
    /// the user visits another tab.
    private(set) var viewModel: CUAViewModel?

    let host: String
    @ObservationIgnored private var binaryPath: URL?
    @ObservationIgnored private let binaryRefresh: @MainActor () -> URL?
    @ObservationIgnored private let portProvider: PortProvider
    @ObservationIgnored private let bearerProvider: BearerProvider
    @ObservationIgnored private let readinessProbe: ReadinessProbe
    @ObservationIgnored private let launcher: Launcher
    @ObservationIgnored private let stopSignaler: StopSignaler
    @ObservationIgnored private var child: ProcessGroupChild?
    @ObservationIgnored private var stdoutPipe: Pipe?
    @ObservationIgnored private var stderrPipe: Pipe?
    @ObservationIgnored private var startTask: Task<Void, Never>?
    @ObservationIgnored private var shutdownRequested = false
    @ObservationIgnored private var stoppingChild: ProcessGroupChild?
    @ObservationIgnored private var stoppingPreservesContinuity = false

    /// A separate window keeps the CUA sidecar away from the user-selected
    /// chat port. Do not call ``PortAllocator.allocate`` here: that allocator
    /// intentionally sweeps rapid-owned listeners before a model replacement,
    /// which would terminate the live chat server when both services coexist.
    nonisolated static let candidatePorts = Array(7_670...7_679) + Array(8_010...8_019)

    init(
        host: String = "127.0.0.1",
        binaryPath: URL? = ServerLocator.locate()?.binary,
        binaryRefresh: @escaping @MainActor () -> URL? = { ServerLocator.locate()?.binary },
        initialState: CUAServerState = .idle,
        portProvider: @escaping PortProvider = {
            await PortSweep.awaitLaunchSweep()
            return await Task.detached(priority: .userInitiated) {
                let reserved = Set(PortAllocator.candidatePorts)
                return selectPort(candidates: candidatePorts, reserved: reserved) {
                    PortAllocator.canBind(port: $0, host: "127.0.0.1")
                }
            }.value
        },
        bearerProvider: @escaping BearerProvider = { BearerSecret.generate() ?? "" },
        readinessProbe: @escaping ReadinessProbe = CUAServerManager.probeReadiness,
        launcher: @escaping Launcher = CUAServerManager.launch,
        stopSignaler: @escaping StopSignaler = { $0.signalProcessGroup(SIGTERM) }
    ) {
        self.host = host
        self.binaryPath = binaryPath
        self.binaryRefresh = binaryRefresh
        self.state = initialState
        self.portProvider = portProvider
        self.bearerProvider = bearerProvider
        self.readinessProbe = readinessProbe
        self.launcher = launcher
        self.stopSignaler = stopSignaler
    }

    var client: CUAClient? {
        guard state == .ready, let port, let bearerToken else { return nil }
        return CUAClient(host: host, port: port, bearerToken: bearerToken)
    }

    var failureMessage: String? {
        guard case let .failed(message) = state else { return nil }
        return message
    }

    /// Starts lazily when the panel first appears, but the task is owned by
    /// the app-level manager so navigating away cannot cancel startup and
    /// strand a partially launched child.
    func startIfNeeded() {
        guard startTask == nil else { return }
        switch state {
        case .starting, .ready:
            return
        case .idle, .failed:
            break
        }
        startTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.ensureRunning()
            self.startTask = nil
        }
    }

    func ensureRunning() async {
        switch state {
        case .starting, .ready:
            return
        case .idle, .failed:
            break
        }
        guard !shutdownRequested else { return }
        state = .starting

        if binaryPath == nil { binaryPath = binaryRefresh() }
        guard let binaryPath else {
            fail("The Computer Use service is unavailable. Reinstall or update Rapid-MLX, then try again.")
            return
        }
        guard let allocatedPort = await portProvider(), !shutdownRequested else {
            if !shutdownRequested {
                fail("Rapid couldn't reserve a local port for Computer Use. Close another local server and try again.")
            }
            return
        }
        let bearer = bearerProvider()
        guard !bearer.isEmpty else {
            fail("Rapid couldn't create a secure Computer Use session. Restart Rapid-MLX and try again.")
            return
        }

        let output = Pipe()
        let errors = Pipe()
        Self.drain(output)
        Self.drain(errors)
        // Retain the pipes before launch so every failure path can detach the
        // readability handlers. An undrained child pipe eventually fills and
        // can block the sidecar while it writes routine request logs.
        stdoutPipe = output
        stderrPipe = errors
        let environment = ServerManager.serveEnvironmentAdditions(
            bearer: bearer,
            ambient: ProcessInfo.processInfo.environment,
            supervisorPID: ProcessInfo.processInfo.processIdentifier
        )
        let arguments = Self.serveArguments(host: host, port: allocatedPort)
        let launched: ProcessGroupChild
        do {
            launched = try launcher(
                binaryPath, arguments, environment, output, errors
            ) { [weak self] process in
                Task { @MainActor [weak self] in
                    self?.childExited(process)
                }
            }
        } catch {
            fail("Rapid couldn't start the Computer Use service. Try again, or reinstall the engine if this continues.")
            return
        }
        guard !shutdownRequested else {
            launched.signalProcessGroup(SIGTERM)
            return
        }
        child = launched
        port = allocatedPort
        bearerToken = bearer

        guard await readinessProbe(host, allocatedPort, bearer),
              child === launched, !shutdownRequested else {
            if child === launched, !shutdownRequested {
                await stop(preserveContinuity: true)
                fail("The Computer Use service didn't become ready. Check the engine installation and try again.")
            }
            return
        }
        let replacement = CUAViewModel(api: CUAClient(
            host: host, port: allocatedPort, bearerToken: bearer
        ))
        if let previous = viewModel {
            replacement.restoreContinuity(from: previous)
        }
        viewModel = replacement
        sessionID = UUID()
        state = .ready
    }

    func retry() async {
        await ensureRunning()
    }

    func stop() async {
        await stop(preserveContinuity: false)
    }

    private func stop(preserveContinuity: Bool) async {
        guard let child else {
            clearSession(nextState: .idle, preserveContinuity: preserveContinuity)
            return
        }
        // Bind intent to this exact process before SIGTERM. Its asynchronous
        // termination callback may run before this method resumes.
        stoppingChild = child
        stoppingPreservesContinuity = preserveContinuity
        stopSignaler(child)
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline, child.isProcessGroupAlive {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if child.isProcessGroupAlive { child.signalProcessGroup(SIGKILL) }
        if self.child === child {
            clearSession(nextState: .idle, preserveContinuity: preserveContinuity)
        }
    }

    func beginShutdown() {
        shutdownRequested = true
        startTask?.cancel()
        startTask = nil
        child?.signalProcessGroup(SIGTERM)
    }

    func shutdownSync() {
        beginShutdown()
        guard let child else { return }
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline, child.isProcessGroupAlive {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if child.isProcessGroupAlive { child.signalProcessGroup(SIGKILL) }
        clearSession(nextState: .idle, preserveContinuity: false)
    }

    nonisolated static func serveArguments(host: String, port: Int) -> [String] {
        [
            "serve", "--cua-only",
            "--host", host,
            "--port", String(port),
            "--cors-origins", "http://127.0.0.1", "http://localhost",
        ]
    }

    /// Selects without the model server allocator's destructive orphan sweep.
    /// A live chat listener is an occupied candidate to skip, never a process
    /// this auxiliary lifecycle may signal.
    nonisolated static func selectPort(
        candidates: [Int],
        reserved: Set<Int>,
        canBind: (Int) -> Bool
    ) -> Int? {
        candidates.first { !reserved.contains($0) && canBind($0) }
    }

    private func childExited(_ process: ProcessGroupChild) {
        guard child === process else { return }
        if stoppingChild === process {
            clearSession(
                nextState: .idle,
                preserveContinuity: stoppingPreservesContinuity
            )
            return
        }
        if shutdownRequested {
            clearSession(nextState: .idle, preserveContinuity: false)
        } else {
            clearSession(nextState: .failed(
                "The Computer Use service stopped unexpectedly. Try again."
            ), preserveContinuity: true)
        }
    }

    private func fail(_ message: String) {
        clearSession(nextState: .failed(message), preserveContinuity: true)
    }

    private func clearSession(
        nextState: CUAServerState,
        preserveContinuity: Bool
    ) {
        if preserveContinuity {
            viewModel?.detachFromSession()
        } else {
            viewModel?.invalidateSession()
            viewModel = nil
        }
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        child = nil
        stdoutPipe = nil
        stderrPipe = nil
        port = nil
        bearerToken = nil
        sessionID = nil
        stoppingChild = nil
        stoppingPreservesContinuity = false
        state = nextState
    }

    private nonisolated static func drain(_ pipe: Pipe) {
        let drainer = PipeDrainer(pipe.fileHandleForReading)
        pipe.fileHandleForReading.readabilityHandler = { _ in
            _ = drainer.drain()
        }
    }

    private nonisolated static func launch(
        binary: URL,
        arguments: [String],
        environment: [String: String],
        output: Pipe,
        errors: Pipe,
        termination: @escaping @Sendable (ProcessGroupChild) -> Void
    ) throws -> ProcessGroupChild {
        try ProcessGroupChild.spawn(
            executableURL: binary,
            arguments: arguments,
            standardInput: .nullDevice,
            standardOutput: output,
            standardError: errors,
            environmentAdditions: environment,
            replaceEnvironment: true,
            terminationHandler: termination
        )
    }

    private nonisolated static func probeReadiness(
        host: String, port: Int, bearer: String
    ) async -> Bool {
        guard let healthURL = URL(string: "http://\(host):\(port)/health/ready"),
              let capabilitiesURL = URL(string: "http://\(host):\(port)/v1/cua/capabilities")
        else { return false }
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline, !Task.isCancelled {
            do {
                var healthRequest = URLRequest(url: healthURL)
                healthRequest.timeoutInterval = 1
                let (healthData, healthResponse) = try await URLSession.shared.data(for: healthRequest)
                if (healthResponse as? HTTPURLResponse)?.statusCode == 200,
                   let payload = try? JSONSerialization.jsonObject(with: healthData) as? [String: Any],
                   payload["ready"] as? Bool == true,
                   payload["model"] is NSNull,
                   payload["model_loaded"] as? Bool == false {
                    var request = URLRequest(url: capabilitiesURL)
                    request.timeoutInterval = 1
                    request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
                    let (_, response) = try await URLSession.shared.data(for: request)
                    if (response as? HTTPURLResponse)?.statusCode == 200 { return true }
                }
            } catch {
                // The child may still be importing; retry within the bounded window.
            }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        return false
    }
}
