import Darwin
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

    private(set) var state: State = .idle {
        didSet { failureNeedsProviderKey = false }
    }
    /// Set only by local key failures, independently of their localized text.
    private(set) var failureNeedsProviderKey = false
    private(set) var snapshot: ShareComputeStatusSnapshot?
    private(set) var activeModel: ShareComputeModel?

    /// Sharing owns a separate, exclusive model process. Stop leaves it and
    /// the displaced local model unloaded; old receipts retain their historical
    /// restore outcome for compatibility.
    private(set) var lastRestoreOutcome: ShareComputeRestoreStatus = .notNeeded

    private weak var server: ServerManager?
    private var child: ProcessGroupChild?
    private var reservation: UUID?
    private var statusTask: Task<Void, Never>?
    private var stopEscalationTask: Task<Void, Never>?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var expectedStop = false
    private let providerStore: ShareComputeProviderKeyStore
    private let home: URL
    private(set) var savedProviderKeyLabel: String?
    private(set) var providerKeyIsStored = false
    private(set) var providerKeyMessage: String?
    /// Bumped on every provider-key WRITE or REMOVAL — never on a mere
    /// refresh. ``savedProviderKeyLabel`` alone cannot carry this: a key
    /// replaced with another key that redacts to the same label (same last
    /// four characters, different secret) leaves the label unchanged, so a
    /// view observing only the label never learns the credential moved and
    /// the ledger coordinator keeps reading a now-stale world. Observers
    /// that need to react to "the credential changed" watch this instead.
    private(set) var providerKeyRevision = 0
    private var shuttingDown = false
    private var shutdownSignalledAt: Date?
    private var currentStatusURL: URL?
    private var currentSession: String?
    /// Invalidates a join that is still waiting for the shared residency
    /// lease. Without this, disabling the experiment during preparation could
    /// let the suspended task wake later and start a now-hidden provider.
    private var operationID: UUID?
    private var preparationTask: Task<UUID, Error>?

    init(
        server: ServerManager,
        providerStore: ShareComputeProviderKeyStore = ShareComputeProviderKeyStore(),
        home: URL? = nil
    ) {
        self.server = server
        self.providerStore = providerStore
        self.home = home ?? Self.runtimeHomeURL()
        refreshProviderKey()
    }

    var canEditProviderKey: Bool {
        child == nil && operationID == nil && preparationTask == nil && reservation == nil && !shuttingDown
    }

    func refreshProviderKey() {
        savedProviderKeyLabel = nil
        providerKeyIsStored = false
        switch providerStore.load() {
        case .found(let key):
            providerKeyIsStored = true
            savedProviderKeyLabel = key.redactedLabel
            providerKeyMessage = nil
        case .missing:
            providerKeyMessage = nil
        case .unavailable:
            providerKeyIsStored = true
            providerKeyMessage = String(localized: "Unlock your Keychain to use the saved provider key.")
        case .corrupted:
            providerKeyIsStored = true
            providerKeyMessage = String(localized: "Replace the saved provider key in Settings → Experimental → Share Compute.")
        }
    }

    @discardableResult
    func saveProviderKey(_ raw: String) -> Bool {
        guard canEditProviderKey else { return false }
        guard let key = ShareComputeProviderKey.validate(raw) else {
            providerKeyMessage = String(localized: "Paste a complete qsppk- provider key as a single line.")
            return false
        }
        if case .found(let existing) = providerStore.load(), existing == key {
            refreshProviderKey()
            return true
        }
        // Reuse the existing registration-proof protocol: invalidate the old
        // account binding BEFORE changing the Keychain secret. A crash between
        // these operations can require another registration, never silently
        // route rewards to the previous account. Node credentials are retained.
        guard invalidateRegistrationProof() else { return false }
        let saved = providerStore.save(key)
        providerKeyRevision += 1
        refreshProviderKey()
        guard saved else {
            providerKeyMessage = String(localized: "This Mac couldn't save the provider key. Check that you have an unlocked login Keychain and try again.")
            return false
        }
        return true
    }

    @discardableResult
    func removeProviderKey() -> Bool {
        guard canEditProviderKey else { return false }
        guard invalidateRegistrationProof() else { return false }
        let removed = providerStore.remove()
        providerKeyRevision += 1
        refreshProviderKey()
        guard removed else {
            providerKeyMessage = String(localized: "This Mac couldn't remove the provider key. Unlock your Keychain and try again.")
            return false
        }
        return true
    }

    func needsRegistration(for model: ShareComputeModel, worker: String) -> Bool {
        if !hasRegistration(for: model, worker: worker) { return true }
        if case .failed(let message) = state { return Self.isRegistrationFailure(message) }
        return false
    }

    func requiresProviderKey(for model: ShareComputeModel, worker: String) -> Bool {
        guard needsRegistration(for: model, worker: worker) else { return false }
        if case .found = providerStore.load() { return false }
        return true
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
        Self.registrationMatches(model: model, worker: worker, home: home)
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
        guard canEditProviderKey else { return }
        guard let server, let binary = server.binaryPath else {
            state = .failed(String(localized: "Rapid-MLX's local engine is not available."))
            return
        }
        guard ShareComputeModel.desktopSharing.contains(model) else {
            state = .failed(String(localized: "That model is not available in the QuickSilver pool."))
            return
        }
        if let providerKey, !providerKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard saveProviderKey(providerKey) else {
                state = .failed(providerKeyMessage ?? String(localized: "The provider key couldn't be saved."))
                failureNeedsProviderKey = true
                return
            }
        }
        let registrationKey: String?
        if needsRegistration(for: model, worker: worker) {
            guard case .found(let key) = providerStore.load() else {
                refreshProviderKey()
                state = .failed(providerKeyMessage ?? String(localized: "Save your QuickSilver provider key to connect this Mac."))
                failureNeedsProviderKey = true
                return
            }
            registrationKey = key.rawValue
        } else {
            // Ordinary off/on reuses the existing node, without the account key
            // and without --reregister. Reading a saved key is unnecessary here.
            registrationKey = nil
        }

        state = .preparing
        activeModel = model
        snapshot = nil
        let operation = UUID()
        operationID = operation
        let preparing = Task { try await server.prepareForCommunityBenchmark() }
        preparationTask = preparing
        let lease: UUID
        do {
            lease = try await preparing.value
            preparationTask = nil
        } catch is CancellationError {
            preparationTask = nil
            guard operationID == operation else {
                if child == nil, operationID == nil { resetToIdle() }
                return
            }
            operationID = nil
            resetToIdle()
            return
        } catch {
            preparationTask = nil
            guard operationID == operation else {
                if child == nil, operationID == nil { resetToIdle() }
                return
            }
            operationID = nil
            state = .failed(String(localized: "Rapid couldn't unload the current model safely."))
            activeModel = nil
            return
        }
        lastRestoreOutcome = .unloaded
        guard operationID == operation, !shuttingDown else {
            server.finishCommunityBenchmark(lease)
            // Release only this stale operation's lease. The previous local
            // model stays unloaded; a replacement join owns its own session.
            if child == nil, operationID == nil { resetToIdle() }
            return
        }
        reservation = lease

        let session = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let statusURL = Self.statusURL(session: session, home: home)
        currentStatusURL = statusURL
        currentSession = session
        try? FileManager.default.removeItem(at: statusURL)
        let inputPipe = Pipe()
        let out = Pipe()
        let err = Pipe()
        installDiscardingDrainer(on: out)
        installDiscardingDrainer(on: err)
        var environment = EngineProcessEnvironment.sidecar(
            ServerManager.serveEnvironmentAdditions(
                bearer: "",
                ambient: ProcessInfo.processInfo.environment,
                physicalRAMBytes: MacHardware.detect().physicalRAMBytes,
                availableRAMBytes: MemoryProbe.snapshot()?.freeBytes ?? 0,
                supervisorPID: ProcessInfo.processInfo.processIdentifier,
                modelsFolderOverride: ModelsFolderPreference.validatedOverrideURL()?.path
            )
        )
        environment["HOME"] = home.path
        let request = Self.spawnRequest(
            model: model,
            worker: worker,
            session: session,
            providerKey: registrationKey,
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
            state = .failed(String(localized: "Rapid couldn't start Share Compute."))
            operationID = nil
            releaseReservation()
        }
    }

    func leave() {
        operationID = nil
        guard let child else {
            if let preparationTask {
                state = .stopping
                preparationTask.cancel()
            } else {
                releaseReservation()
                resetToIdle()
            }
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
        preparationTask?.cancel()
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
                acknowledgeRegistration(value)
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
        case "error": .failed(snapshot.message ?? String(localized: "Share Compute stopped unexpectedly."))
        default: .failed(String(localized: "Share Compute reported an unsupported status."))
        }
    }

    nonisolated static func isValidProviderKey(_ value: String) -> Bool {
        ShareComputeProviderKey.validate(value) != nil
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
        if let finalSnapshot {
            acknowledgeRegistration(finalSnapshot)
            snapshot = finalSnapshot
        }
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
            state = .failed(finalSnapshot?.message ?? String(localized: "Share Compute stopped unexpectedly."))
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

    @discardableResult
    private func invalidateRegistrationProof() -> Bool {
        for model in ShareComputeModel.desktopSharing {
            let url = Self.registrationURL(catalogID: model.catalogID, home: home)
            do {
                // fileExists also returns false for inaccessible ancestors.
                // Only ENOENT proves absence; every other metadata error must
                // prevent changing the account while an old proof may exist.
                var metadata = stat()
                if url.path.withCString({ Darwin.lstat($0, &metadata) }) != 0 {
                    let code = errno
                    if code == ENOENT { continue }
                    throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
                }
                guard metadata.st_mode & S_IFMT == S_IFREG else {
                    providerKeyMessage = String(localized: "The local node registration isn't a regular file. Repair the QuickSilver state directory before changing the saved key.")
                    return false
                }
                try FileManager.default.removeItem(at: url)
            } catch {
                providerKeyMessage = String(localized: "This Mac couldn't reset its node registration. Check access to the local QuickSilver state directory and try again.")
                return false
            }
        }
        return true
    }

    private func acknowledgeRegistration(_ status: ShareComputeStatusSnapshot) {
        if status.phase == "error", let message = status.message,
           Self.isRegistrationFailure(message) {
            invalidateRegistrationProof()
        }
    }

    nonisolated static func isRegistrationFailure(_ message: String) -> Bool {
        ["register again", "re-register", "reregister", "provider key"].contains {
            message.localizedCaseInsensitiveContains($0)
        }
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
