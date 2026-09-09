import Foundation
import Observation

@MainActor
@Observable
final class ShareComputeManager {
    struct SpawnRequest: Equatable {
        let arguments: [String]
        let environment: [String: String]
        let standardInput: Data
    }

    enum State: Equatable {
        case idle
        case preparing
        case registering
        case starting
        case warming
        case connecting
        case online
        case reconnecting
        case stopping
        case failed(String)

        var isActive: Bool {
            switch self {
            case .idle, .failed: false
            default: true
            }
        }
    }

    private(set) var state: State = .idle
    private(set) var snapshot: ShareComputeStatusSnapshot?
    private(set) var activeModel: ShareComputeModel?

    private weak var server: ServerManager?
    private var child: ProcessGroupChild?
    private var reservation: UUID?
    private var statusTask: Task<Void, Never>?
    private var stopEscalationTask: Task<Void, Never>?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var expectedStop = false
    private var restoreAlias: String?
    private var restoreTask: Task<Void, Never>?
    private var restoreAttemptID: UUID?
    private var shuttingDown = false
    private var shutdownSignalledAt: Date?
    private var currentStatusURL: URL?
    private var currentSession: String?
    /// Invalidates a join that is still waiting for the shared residency
    /// lease. Without this, disabling the experiment during preparation could
    /// let the suspended task wake later and start a now-hidden provider.
    private var operationID: UUID?

    init(server: ServerManager) {
        self.server = server
    }

    nonisolated static func runtimeHomeURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fallback: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        guard let raw = environment["HOME"], raw.hasPrefix("/") else { return fallback }
        return URL(fileURLWithPath: raw, isDirectory: true).standardizedFileURL
    }

    nonisolated static func statusURL(session: String, home: URL? = nil) -> URL {
        let home = home ?? runtimeHomeURL()
        return home.appendingPathComponent(".rapid-mlx/quicksilver/desktop-status-\(session).json")
    }

    nonisolated static func registrationURL(catalogID: String, home: URL? = nil) -> URL {
        let home = home ?? runtimeHomeURL()
        return home.appendingPathComponent(
            ".rapid-mlx/quicksilver/\(catalogID).registration.json"
        )
    }

    nonisolated static func credentialCacheURL(
        catalogID: String,
        home: URL? = nil
    ) -> URL {
        let home = home ?? runtimeHomeURL()
        return home.appendingPathComponent(".rapid-mlx/quicksilver/\(catalogID).json")
    }

    func hasRegistration(for model: ShareComputeModel, worker: String) -> Bool {
        Self.registrationMatches(model: model, worker: worker)
    }

    nonisolated static func registrationMatches(
        model: ShareComputeModel,
        worker: String,
        home: URL? = nil
    ) -> Bool {
        let url = registrationURL(catalogID: model.catalogID, home: home)
        let credentialURL = credentialCacheURL(catalogID: model.catalogID, home: home)
        let credentialValues = try? credentialURL.resourceValues(forKeys: [.isRegularFileKey])
        guard credentialValues?.isRegularFile == true,
              FileManager.default.isReadableFile(atPath: credentialURL.path) else {
            return false
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 4_097), data.count <= 4_096,
              let marker = try? JSONDecoder().decode(
                  ShareComputeRegistrationSnapshot.self,
                  from: data
              ) else { return false }
        return marker.schemaVersion == 1
            && marker.model == model.catalogID
            && marker.alias == model.alias
            && marker.worker == ShareComputeModel.sanitizedWorker(worker)
    }

    func join(model: ShareComputeModel, worker: String, providerKey: String? = nil) async {
        // Connect is idempotent while a preparation or provider is already
        // live. A second button event must not replace truthful session state
        // with an unrelated engine-availability error.
        guard child == nil, operationID == nil else { return }
        guard let server, let binary = server.binaryPath else {
            state = .failed("Rapid-MLX's local engine is not available.")
            return
        }
        guard ShareComputeModel.supported.contains(model) else {
            state = .failed("That model is not available in the QuickSilver pool.")
            return
        }
        let trimmedKey = providerKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmedKey, !Self.isValidProviderKey(trimmedKey) {
            state = .failed("That provider key is not valid.")
            return
        }
        if !hasRegistration(for: model, worker: worker), trimmedKey?.isEmpty != false {
            state = .failed("Connect QuickSilver with a provider key first.")
            return
        }

        // A replacement join owns the pending restore alias. Cancel an
        // in-flight restore attempt, but keep the alias so the last session in
        // the chain can still restore it.
        restoreAttemptID = nil
        restoreTask?.cancel()
        restoreTask = nil

        state = .preparing
        activeModel = model
        snapshot = nil
        // Preserve the model captured by the first join in a cancel/rejoin
        // chain. The first preparation may already have stopped it by the
        // time the replacement join reads `servingAlias`.
        restoreAlias = Self.restoreAliasForJoin(
            pending: restoreAlias,
            currentlyServing: server.servingAlias
        )
        let operation = UUID()
        operationID = operation
        let lease: UUID
        do {
            lease = try await server.prepareForCommunityBenchmark()
        } catch is CancellationError {
            guard operationID == operation else {
                restoreIfPreparationHasNoSuccessor()
                return
            }
            operationID = nil
            resetToIdle()
            restorePreviousModelIfNeeded()
            return
        } catch {
            guard operationID == operation else {
                restoreIfPreparationHasNoSuccessor()
                return
            }
            operationID = nil
            state = .failed("Rapid couldn't pause the current model safely.")
            activeModel = nil
            restorePreviousModelIfNeeded()
            return
        }
        guard operationID == operation, !shuttingDown else {
            restoreIfPreparationHasNoSuccessor()
            server.finishCommunityBenchmark(lease)
            // This operation no longer owns manager state. A subsequent join
            // may already have replaced activeModel and restoreAlias, so only
            // release the lease acquired by this stale operation. The pending
            // restore is inherited by a successor, or performed now if the
            // user simply cancelled and no successor exists.
            return
        }
        reservation = lease

        let session = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let statusURL = Self.statusURL(session: session)
        currentStatusURL = statusURL
        currentSession = session
        try? FileManager.default.removeItem(at: statusURL)
        let inputPipe = Pipe()
        let out = Pipe()
        let err = Pipe()
        installDiscardingDrainer(on: out)
        installDiscardingDrainer(on: err)
        let environment = ServerManager.serveEnvironmentAdditions(
            bearer: "",
            ambient: ProcessInfo.processInfo.environment,
            physicalRAMBytes: MacHardware.detect().physicalRAMBytes,
            availableRAMBytes: MemoryProbe.snapshot()?.freeBytes ?? 0,
            supervisorPID: ProcessInfo.processInfo.processIdentifier,
            modelsFolderOverride: ModelsFolderPreference.validatedOverrideURL()?.path
        )
        let request = Self.spawnRequest(
            model: model,
            worker: worker,
            session: session,
            providerKey: trimmedKey,
            environment: environment
        )

        do {
            // Fill the bounded pipe while its read end is unquestionably open.
            // Writing after spawn creates a narrow SIGPIPE race if the child
            // fails before Swift gets scheduled again.
            if !request.standardInput.isEmpty {
                inputPipe.fileHandleForWriting.write(request.standardInput)
            }
            let spawned = try ProcessGroupChild.spawn(
                executableURL: binary,
                arguments: request.arguments,
                standardInput: inputPipe.fileHandleForReading,
                standardOutput: out,
                standardError: err,
                environmentAdditions: request.environment,
                replaceEnvironment: true,
                startMonitorImmediately: false
            ) { [weak self] process in
                Task { @MainActor [weak self] in self?.childExited(process) }
            }
            child = spawned
            operationID = nil
            stdoutPipe = out
            stderrPipe = err
            expectedStop = false
            spawned.startMonitor()
            inputPipe.fileHandleForReading.closeFile()
            inputPipe.fileHandleForWriting.closeFile()
            state = .starting
            statusTask = Task { [weak self] in
                await self?.pollStatus(at: statusURL, session: session)
            }
        } catch {
            inputPipe.fileHandleForReading.closeFile()
            inputPipe.fileHandleForWriting.closeFile()
            detachPipes(out, err)
            removeCurrentStatusFile()
            activeModel = nil
            state = .failed("Rapid couldn't start Share Compute.")
            operationID = nil
            restorePreviousModelIfNeeded()
            releaseReservation()
        }
    }

    func leave() {
        operationID = nil
        guard let child else {
            releaseReservation()
            resetToIdle()
            return
        }
        expectedStop = true
        state = .stopping
        statusTask?.cancel()
        child.signalProcessGroup(SIGTERM)
        scheduleStopEscalation(for: child)
    }

    func beginShutdown() {
        if shutdownSignalledAt == nil { shutdownSignalledAt = Date() }
        shuttingDown = true
        operationID = nil
        restoreAlias = nil
        restoreAttemptID = nil
        restoreTask?.cancel()
        restoreTask = nil
        statusTask?.cancel()
        stopEscalationTask?.cancel()
        expectedStop = true
        child?.signalProcessGroup(SIGTERM)
    }

    func finishShutdown() {
        beginShutdown()
        guard let child else { return }
        // The server and provider are signalled together, then reaped in
        // sequence. Measure this grace from the signal so their shutdown
        // windows overlap instead of making app termination take 10 seconds.
        let deadline = (shutdownSignalledAt ?? Date()).addingTimeInterval(5)
        while Date() < deadline && child.isProcessGroupAlive {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if child.isProcessGroupAlive {
            child.signalProcessGroup(SIGKILL)
            let killDeadline = Date().addingTimeInterval(0.5)
            while Date() < killDeadline && child.isProcessGroupAlive {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        // Never advertise the unified-memory reservation as free while the
        // kernel can still see this process group. App termination is already
        // irreversible; retaining bookkeeping is safer than permitting a
        // later in-process owner to overlap a pathological uninterruptible
        // child.
        guard !child.isProcessGroupAlive else { return }
        removeCurrentStatusFile()
        cleanupPipes()
        self.child = nil
        releaseReservation()
    }

    private func pollStatus(at url: URL, session: String) async {
        while !Task.isCancelled, child != nil {
            if let value = Self.loadStatus(at: url),
               value.schemaVersion == 1, value.session == session {
                snapshot = value
                state = Self.state(for: value)
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    nonisolated static func loadStatus(at url: URL) -> ShareComputeStatusSnapshot? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 65_537),
              data.count <= 65_536 else { return nil }
        return try? JSONDecoder().decode(ShareComputeStatusSnapshot.self, from: data)
    }

    nonisolated static func state(for snapshot: ShareComputeStatusSnapshot) -> State {
        switch snapshot.phase {
        case "preparing": .preparing
        case "registering": .registering
        case "starting": .starting
        case "warming": .warming
        case "connecting": .connecting
        case "online": .online
        case "reconnecting": .reconnecting
        // The provider publishes this before its process group is necessarily
        // gone. Keep controls locked until childExited/finishDeferredExit has
        // confirmed teardown and released unified-memory residency.
        case "stopped": .stopping
        case "error": .failed(snapshot.message ?? "Share Compute stopped unexpectedly.")
        default: .failed("Share Compute reported an unsupported status.")
        }
    }

    nonisolated static func restoreAliasForJoin(
        pending: String?,
        currentlyServing: String?
    ) -> String? {
        pending ?? currentlyServing
    }

    nonisolated static func isValidProviderKey(_ value: String) -> Bool {
        value.utf8.count <= 4_095
            && !value.contains("\n")
            && !value.contains("\r")
    }

    /// Pure representation of every caller-controlled value handed to the
    /// process boundary. Keeping stdin beside argv and environment makes the
    /// credential transport invariant directly testable.
    nonisolated static func spawnRequest(
        model: ShareComputeModel,
        worker: String,
        session: String,
        providerKey: String?,
        environment: [String: String]
    ) -> SpawnRequest {
        var arguments = [
            "share", model.alias,
            "--quicksilver",
            "--quicksilver-model", model.catalogID,
            "--worker", ShareComputeModel.sanitizedWorker(worker),
            "--desktop-session", session,
        ]
        var standardInput = Data()
        if let providerKey, !providerKey.isEmpty {
            arguments += ["--provider-key-stdin", "--reregister"]
            standardInput = Data((providerKey + "\n").utf8)
        }
        return SpawnRequest(
            arguments: arguments,
            environment: environment,
            standardInput: standardInput
        )
    }

    private func childExited(_ process: ProcessGroupChild) {
        guard child === process else { return }
        let wasExpected = expectedStop
        let groupIsAlive = process.isProcessGroupAlive
        // The provider can publish its actionable terminal status and exit
        // between two 250 ms polls. Read once at process-exit before removing
        // the snapshot so a revoked credential does not degrade into the
        // generic "stopped unexpectedly" message.
        let diskSnapshot = currentStatusURL.flatMap(Self.loadStatus(at:))
        let finalSnapshot = if let diskSnapshot, let currentSession,
                               diskSnapshot.schemaVersion == 1,
                               diskSnapshot.session == currentSession {
            diskSnapshot
        } else {
            snapshot
        }
        if let finalSnapshot { snapshot = finalSnapshot }
        statusTask?.cancel()
        removeCurrentStatusFile()
        cleanupPipes()
        if wasExpected || shuttingDown {
            state = groupIsAlive ? .stopping : .idle
        } else if let finalSnapshot, finalSnapshot.phase == "error" {
            state = Self.state(for: finalSnapshot)
        } else if case .failed = state {
            // Keep the actionable status published by the provider.
        } else {
            state = .failed(finalSnapshot?.message ?? "Share Compute stopped unexpectedly.")
        }
        if groupIsAlive {
            // The supervisor may exit a beat before its nested serve child.
            // Keep exclusive residency until the kernel confirms the whole
            // group is gone; retain `child` too so leave/app shutdown can
            // still signal that surviving process group.
            if !wasExpected, !shuttingDown {
                // A model whose pool supervisor died cannot serve useful
                // work. Give it the same bounded TERM→KILL teardown as an
                // explicit Stop rather than monitoring an orphan forever.
                process.signalProcessGroup(SIGTERM)
                scheduleStopEscalation(for: process)
            }
            ProcessGroupChild.monitorProcessGroupUntilExit(
                processGroupID: process.processGroupID
            ) { [weak self] in
                Task { @MainActor [weak self] in
                    self?.finishDeferredExit(process)
                }
            }
        } else {
            finishDeferredExit(process)
        }
    }

    private func finishDeferredExit(_ process: ProcessGroupChild) {
        guard child === process else { return }
        stopEscalationTask?.cancel()
        stopEscalationTask = nil
        child = nil
        activeModel = nil
        if expectedStop, !shuttingDown { state = .idle }
        restorePreviousModelIfNeeded()
        releaseReservation()
    }

    private func scheduleStopEscalation(for child: ProcessGroupChild) {
        stopEscalationTask?.cancel()
        stopEscalationTask = Task { [weak self, weak child] in
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled, let self, let child,
                  self.child === child, child.isProcessGroupAlive else { return }
            child.signalProcessGroup(SIGKILL)
        }
    }

    private func releaseReservation() {
        guard let reservation else { return }
        server?.finishCommunityBenchmark(reservation)
        self.reservation = nil
    }

    private func resetToIdle() {
        removeCurrentStatusFile()
        state = .idle
        activeModel = nil
        snapshot = nil
    }

    private func restorePreviousModelIfNeeded() {
        guard !shuttingDown, let alias = restoreAlias, let server else {
            restoreAlias = nil
            return
        }
        restoreTask?.cancel()
        let attempt = UUID()
        restoreAttemptID = attempt
        restoreTask = Task { @MainActor [weak self, weak server] in
            guard let server else { return }
            await server.start(alias: alias)
            guard let self, self.restoreAttemptID == attempt else { return }
            self.restoreAttemptID = nil
            self.restoreTask = nil
            if self.operationID == nil, self.child == nil {
                self.restoreAlias = nil
            }
        }
    }

    private func restoreIfPreparationHasNoSuccessor() {
        guard operationID == nil else { return }
        restorePreviousModelIfNeeded()
    }

    private func installDiscardingDrainer(on pipe: Pipe) {
        let drainer = PipeDrainer(pipe.fileHandleForReading)
        pipe.fileHandleForReading.readabilityHandler = { _ in _ = drainer.drain() }
    }

    private func detachPipes(_ stdout: Pipe, _ stderr: Pipe) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
    }

    private func cleanupPipes() {
        if let stdoutPipe, let stderrPipe { detachPipes(stdoutPipe, stderrPipe) }
        stdoutPipe = nil
        stderrPipe = nil
    }

    private func removeCurrentStatusFile() {
        if let currentStatusURL {
            try? FileManager.default.removeItem(at: currentStatusURL)
        }
        self.currentStatusURL = nil
        currentSession = nil
    }
}
