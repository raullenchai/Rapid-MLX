import Foundation

/// Admission is counted before dispatch, including queued work. A timed-out
/// syscall retains its slot until it actually returns, so Retry cannot create
/// an unbounded number of threads waiting for removable-volume consent.
final class DiscoveryBlockingIO: @unchecked Sendable {
    static let shared = DiscoveryBlockingIO()
    static let defaultLimit = 4
    private let queue = DispatchQueue(label: "rapid.discovery.blocking-io", qos: .utility, attributes: .concurrent)
    private let lock = NSLock()
    private let limit: Int
    private var outstanding = 0

    init(limit: Int = defaultLimit) { self.limit = limit }

    @discardableResult
    func submit(_ operation: @escaping @Sendable () -> Void) -> Bool {
        lock.lock()
        guard outstanding < limit else { lock.unlock(); return false }
        outstanding += 1
        lock.unlock()
        queue.async {
            defer {
                self.lock.lock()
                self.outstanding -= 1
                self.lock.unlock()
            }
            operation()
        }
        return true
    }
}

/// A deadline must not await a cancelled task: filesystem reads waiting for
/// removable-volume consent do not necessarily cooperate with cancellation.
enum DiscoveryDeadline {
    struct Expired: Error {}

    static func run<Value: Sendable>(
        seconds: TimeInterval = 10,
        blockingIO: DiscoveryBlockingIO = .shared,
        operation: @escaping @Sendable () throws -> Value
    ) async throws -> Value {
        let race = DiscoveryRace<Value>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                race.install(continuation)
                let timer = Task.detached {
                    do { try await Task.sleep(for: .seconds(seconds)) }
                    catch { return }
                    race.finish(.failure(Expired()))
                }
                race.setTasks([timer])
                guard !race.isFinished else { return }
                if !blockingIO.submit({
                    guard !race.isFinished else { return }
                    do { race.finish(.success(try operation())) }
                    catch { race.finish(.failure(error)) }
                }) {
                    race.finish(.failure(Expired()))
                }
            }
        } onCancel: {
            race.finish(.failure(CancellationError()))
        }
    }

    /// Only for suspending orchestration. Every synchronous filesystem probe
    /// inside this closure must use run (or the bounded blocking-I/O queue).
    static func runAsync<Value: Sendable>(
        seconds: TimeInterval = 10,
        operation: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let race = DiscoveryRace<Value>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                race.install(continuation)
                guard !race.isFinished else { return }
                // Inherit the failure recorder used by catalog subprocesses.
                let worker = Task {
                    do { race.finish(.success(try await operation())) }
                    catch { race.finish(.failure(error)) }
                }
                let timer = Task.detached {
                    do { try await Task.sleep(for: .seconds(seconds)) }
                    catch { return }
                    race.finish(.failure(Expired()))
                }
                race.setTasks([worker, timer])
            }
        } onCancel: {
            race.finish(.failure(CancellationError()))
        }
    }
}

private final class DiscoveryRace<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Value, Error>?
    private var result: Result<Value, Error>?
    private var tasks: [Task<Void, Never>] = []

    var isFinished: Bool {
        lock.lock(); defer { lock.unlock() }
        return result != nil
    }

    func install(_ continuation: CheckedContinuation<Value, Error>) {
        lock.lock()
        let result = self.result
        if result == nil { self.continuation = continuation }
        lock.unlock()
        if let result { continuation.resume(with: result) }
    }

    func setTasks(_ tasks: [Task<Void, Never>]) {
        lock.lock()
        let finished = result != nil
        if !finished { self.tasks = tasks }
        lock.unlock()
        if finished { tasks.forEach { $0.cancel() } }
    }

    func finish(_ result: Result<Value, Error>) {
        lock.lock()
        guard self.result == nil else { lock.unlock(); return }
        self.result = result
        let continuation = self.continuation
        self.continuation = nil
        let tasks = self.tasks
        self.tasks = []
        lock.unlock()
        tasks.forEach { $0.cancel() }
        continuation?.resume(with: result)
    }
}

struct CachedModelDiscoveryFailure: Error, Sendable, Equatable {
    let volumeName: String?
    var message: String {
        if let volumeName {
            return String(localized: "Rapid needs permission to read models on \(volumeName)")
        }
        return String(localized: "Couldn't read cached models")
    }
}

struct CachedModelDiscoveryResult: Sendable {
    let entries: [ModelEntry]
    let failure: CachedModelDiscoveryFailure?
}

/// Records failures across the concurrent catalog probes without changing
/// legacy callers' empty-array contract. Each discovery attempt owns its box.
final class DiscoveryProbeFailure: @unchecked Sendable {
    @TaskLocal static var current: DiscoveryProbeFailure?
    private let lock = NSLock()
    private var error: Error?
    func record(_ error: Error = CocoaError(.fileReadUnknown)) {
        lock.lock(); if self.error == nil { self.error = error }; lock.unlock()
    }
    var recordedError: Error? { lock.lock(); defer { lock.unlock() }; return error }
}

enum CachedModelDiscovery {
    /// Snapshot the preference without stat() on the UI or cache actor. Actual
    /// validation belongs inside the timed worker, since it can prompt too.
    static func requestedOverrideURL(defaults: UserDefaults = .standard) -> URL? {
        guard let path = ModelsFolderPreference.storedPath(defaults: defaults), path.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// Cache and Settings must validate the same captured preference inside
    /// the deadline. A denied override is not evidence of an empty cache.
    static func discoverCatalog(
        timeout: TimeInterval = 10,
        requestedOverride: URL?,
        directoryProbe: @escaping ModelsFolderPreference.DirectoryProbe = ModelsFolderPreference.isDirectory,
        volumeResolver: @escaping @Sendable (URL) -> String? = externalVolumeName,
        provider: @escaping @Sendable (URL?) async throws -> [ModelEntry]
    ) async -> CachedModelDiscoveryResult {
        await discover(
            timeout: timeout,
            cacheURL: BundledModel.userHFCacheURL(
                environment: ProcessInfo.processInfo.environment, preferredOverride: requestedOverride
            ),
            volumeResolver: volumeResolver
        ) {
            let override = try await DiscoveryDeadline.run(seconds: timeout) {
                try ModelsFolderPreference.discoveryOverrideURL(
                    requestedOverride, directoryProbe: directoryProbe
                )
            }
            return try await provider(override)
        }
    }

    static func discover(
        timeout: TimeInterval = 10,
        cacheURL: URL?,
        volumeResolver: @escaping @Sendable (URL) -> String? = externalVolumeName,
        provider: @escaping @Sendable () async throws -> [ModelEntry]
    ) async -> CachedModelDiscoveryResult {
        do {
            let entries = try await DiscoveryDeadline.runAsync(seconds: timeout) {
                let failures = DiscoveryProbeFailure()
                let entries = try await DiscoveryProbeFailure.$current.withValue(failures) {
                    try await provider()
                }
                if let error = failures.recordedError { throw error }
                return entries
            }
            return CachedModelDiscoveryResult(entries: entries, failure: nil)
        } catch {
            // Resolving symlinks must not introduce a second unbounded read.
            let nsError = error as NSError
            let permissionOrTimeout = error is DiscoveryDeadline.Expired ||
                (nsError.domain == NSPOSIXErrorDomain && [Int(EPERM), Int(EACCES)].contains(nsError.code)) ||
                (nsError.domain == NSCocoaErrorDomain && nsError.code == CocoaError.fileReadNoPermission.rawValue)
            let volume: String? = if permissionOrTimeout, let cacheURL {
                (try? await DiscoveryDeadline.run(seconds: 0.25) {
                    volumeResolver(cacheURL)
                }) ?? nil
            } else { nil }
            return CachedModelDiscoveryResult(
                entries: [QuickstartCoordinator.defaultChoice, QuickstartCoordinator.lowMemoryChoice].map {
                    ModelEntry(alias: $0.alias, hfRepo: $0.hfRepo, sizeOnDisk: nil,
                               cached: false, downloadSizeBytes: $0.downloadBytes)
                },
                failure: (error as? CachedModelDiscoveryFailure) ?? CachedModelDiscoveryFailure(volumeName: volume)
            )
        }
    }

    /// Read link metadata one component at a time, including a symlink at
    /// ~/.cache/huggingface itself. Stop BEFORE touching the protected volume.
    /// This works even when access to that volume has just been denied.
    static func externalVolumeName(_ url: URL) -> String? {
        // standardizedFileURL folds /private/var back to /var on macOS.
        // Reapplying it to /var's symlink target would revisit that link
        // until the loop limit, before ever reaching the cache symlink.
        var components = url.pathComponents.filter { $0 != "/" }
        var path = "/"
        var links = 0
        while !components.isEmpty {
            let component = components.removeFirst()
            if component == "." { continue }
            if component == ".." {
                path = (path as NSString).deletingLastPathComponent
                continue
            }
            path = (path as NSString).appendingPathComponent(component)
            let parts = (path as NSString).pathComponents
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: path) {
                links += 1
                guard links <= 40 else { return nil }
                let resolved = target.hasPrefix("/") ? target :
                    ((path as NSString).deletingLastPathComponent as NSString).appendingPathComponent(target)
                components = (resolved as NSString).pathComponents.filter { $0 != "/" } + components
                path = "/"
            } else if parts.count >= 3, parts[1] == "Volumes" {
                return parts[2]
            }
        }
        return nil
    }
}
