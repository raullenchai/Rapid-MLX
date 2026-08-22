import Foundation

/// Process-wide cache in front of ``ModelCatalog/load(binary:hubCacheOverride:)``.
///
/// Every catalog read forks **at least two** `rapid-mlx` subprocesses — `models`
/// and `ls`, plus one `info` per cached sibling — and five separate surfaces
/// call it (both settings panels, the composer's model picker, launch
/// auto-start, and deletion). None of them shared anything, so simply moving
/// between Models and Model Management paid for a fresh round of process
/// spawns and showed a spinner each time.
///
/// ## Staleness is bounded, not merely time-limited
///
/// Entries are keyed on ``DownloadManager/cacheGeneration``, the counter the app
/// already bumps whenever the on-disk model set changes (download finished,
/// alias deleted, models folder repointed). A cached snapshot is therefore
/// invalidated by the *event* that makes it wrong rather than by a timer that
/// might fire too late — an expiring TTL would still hand out a stale
/// "downloaded" badge for however long it had left to run.
///
/// A short TTL is kept **on top of** that as a backstop for changes nothing
/// announces: a model deleted from `~/.cache/huggingface` by another tool, or
/// by `rapid-mlx` from a terminal.
///
/// ## Serving stale while revalidating
///
/// ``entries(binary:generation:)`` returns any usable snapshot immediately and
/// refreshes in the background. The first read of a generation has nothing to
/// serve and awaits the real load; every read after that is instant. Callers
/// distinguish the two via ``cached(binary:generation:)`` so they can show a
/// spinner only when there is genuinely nothing to show.
actor ModelCatalogCache {
    static let shared = ModelCatalogCache()

    /// Backstop for out-of-band changes (see above).
    ///
    /// Five minutes, not seconds: ``cacheGeneration`` already catches every
    /// mutation the app makes, so this only has to bound how long an edit made
    /// *outside* the app can go unnoticed. An earlier 30s value expired while
    /// the user was simply reading the Settings panel — measured MISSes at
    /// age=33s and age=40s during ordinary tab switching — which is exactly
    /// the needless refetch this cache exists to remove.
    private static let ttl: TimeInterval = 300

    private struct Snapshot {
        let entries: [ModelEntry]
        let generation: UInt
        let fetchedAt: Date
        /// The override in force when this was captured. Pointing the models
        /// folder somewhere else changes what "cached" means for every alias,
        /// so a snapshot taken under a different root cannot be reused.
        let overridePath: String?
        /// The rapid-mlx executable that produced this catalog. A dev build and
        /// the shipped binary can enumerate different models, so a snapshot
        /// from one must not be served after the app switches binaries.
        let binaryPath: String
    }

    private var snapshot: Snapshot?

    /// Mirror of ``snapshot`` readable without `await`.
    ///
    /// SwiftUI builds `@State` initial values synchronously, so a view cannot
    /// consult an actor before its first frame. Without this, a panel that
    /// re-appears starts at `catalog = []` + `loading = true`, renders the
    /// spinner, and only replaces it once the `.task` resumes.
    ///
    /// Every access goes through ``mirrorLock``. `nonisolated(unsafe)` tells
    /// the compiler we take responsibility for that synchronisation ourselves
    /// (the actor cannot: ``seed`` must read synchronously from the main
    /// thread). The previous lock-free version was a genuine data race — an
    /// actor-side write could tear against a ``seed`` read on another thread.
    private static let mirrorLock = NSLock()
    nonisolated(unsafe) private static var lastSnapshot: Snapshot?

    private static func readMirror() -> Snapshot? {
        mirrorLock.lock()
        defer { mirrorLock.unlock() }
        return lastSnapshot
    }

    private static func writeMirror(_ snap: Snapshot?) {
        mirrorLock.lock()
        defer { mirrorLock.unlock() }
        lastSnapshot = snap
    }

    /// Synchronous peek for `@State` seeding. Returns entries only when they
    /// still satisfy the generation / override / TTL gate. The binary is not
    /// compared here (a `@State` initializer has no access to it); the async
    /// ``entries(binary:generation:)`` the view calls immediately afterwards
    /// re-validates against the real binary and refetches on a mismatch, so a
    /// wrong-binary seed survives at most one frame.
    nonisolated static func seed(generation: UInt) -> [ModelEntry]? {
        guard let snap = readMirror(),
              snap.generation == generation,
              snap.overridePath == ModelsFolderPreference.validatedOverrideURL()?.path,
              Date().timeIntervalSince(snap.fetchedAt) < ttl
        else { return nil }
        return snap.entries
    }

    /// Synchronous behavioral capability for render and spawn paths that
    /// cannot introduce another catalog-probe suspension. The binary match
    /// prevents metadata from a dev/runtime sidecar leaking across a switch;
    /// missing or stale provenance fails closed.
    nonisolated static func supportsImageInput(forAlias alias: String, binary: URL?) -> Bool {
        guard let binary, let snap = readMirror(), snap.binaryPath == binary.path,
              let entry = snap.entries.first(where: {
                  $0.alias.caseInsensitiveCompare(alias) == .orderedSame
              }) else { return false }
        return ModelBrandStyle.supportsImageInput(
            forAlias: alias,
            isBuiltinProfile: entry.isBuiltinProfile,
            isTextOnly: entry.isTextOnly
        )
    }

    /// An in-flight load together with the inputs it was started for, and a
    /// monotonic id (``Task`` is not `Equatable`, so identity is tracked
    /// explicitly). N simultaneous views join ONE set of subprocesses — but
    /// only when their inputs match. A caller arriving after a download (new
    /// generation), a folder repoint (new override) or a binary switch must
    /// NOT receive the pre-change catalog, so it starts its own load instead.
    private struct InFlight {
        let id: UInt64
        let binaryPath: String
        let generation: UInt
        let overridePath: String?
        let task: Task<[ModelEntry], Never>

        func matches(binaryPath: String, generation: UInt, overridePath: String?) -> Bool {
            self.binaryPath == binaryPath
                && self.generation == generation
                && self.overridePath == overridePath
        }
    }

    private var inFlight: InFlight?
    private var nextInFlightID: UInt64 = 0

    /// A snapshot only if one is currently valid; never triggers a load.
    ///
    /// Views use this to decide whether to show a loading state: `nil` means
    /// "nothing to display yet", not "nothing exists".
    func cached(binary: URL, generation: UInt) -> [ModelEntry]? {
        guard let snapshot, isFresh(snapshot, binary: binary, generation: generation) else {
            return nil
        }
        return snapshot.entries
    }

    /// The catalog, served from cache when possible.
    ///
    /// Returns immediately with a valid snapshot; on a same-inputs TTL expiry
    /// it still returns the stale snapshot and refreshes in the background (the
    /// "serve stale while revalidating" contract). Only a genuine miss — no
    /// snapshot, or one whose generation/override/binary changed — awaits a
    /// real load, joining an in-flight one with matching inputs rather than
    /// starting a second.
    func entries(binary: URL, generation: UInt) async -> [ModelEntry] {
        let override = ModelsFolderPreference.validatedOverrideURL()
        let overridePath = override?.path
        let binaryPath = binary.path

        if let snapshot,
           snapshotMatchesInputs(
               snapshot, binaryPath: binaryPath, generation: generation,
               overridePath: overridePath
           ) {
            if Date().timeIntervalSince(snapshot.fetchedAt) < Self.ttl {
                return snapshot.entries
            }
            // Same inputs, past the TTL backstop: serve the stale entries now
            // and revalidate in the background (deduplicated via inFlight), so
            // a passive out-of-band change is picked up without a spinner.
            let stale = snapshot.entries
            if inFlight == nil {
                startLoad(
                    binary: binary, override: override, binaryPath: binaryPath,
                    generation: generation, overridePath: overridePath
                )
            }
            return stale
        }

        if let inFlight,
           inFlight.matches(
               binaryPath: binaryPath, generation: generation,
               overridePath: overridePath
           ) {
            let loaded = await inFlight.task.value
            // Publish the synchronous mirror before returning to callers that
            // immediately start a model from this catalog result.
            await finish(inFlight)
            return loaded
        }
        let task = startLoad(
            binary: binary, override: override, binaryPath: binaryPath,
            generation: generation, overridePath: overridePath
        )
        let loaded = await task.value
        if let inFlight,
           inFlight.matches(
               binaryPath: binaryPath, generation: generation,
               overridePath: overridePath
           ) {
            await finish(inFlight)
        }
        return loaded
    }

    /// Start (and register) a load, replacing any in-flight one with different
    /// inputs. The follow-up stamps the snapshot when the load lands. Returns
    /// the task so a synchronous caller can await it directly.
    @discardableResult
    private func startLoad(
        binary: URL, override: URL?, binaryPath: String,
        generation: UInt, overridePath: String?
    ) -> Task<[ModelEntry], Never> {
        let task = Task<[ModelEntry], Never> {
            await ModelCatalog.load(binary: binary, hubCacheOverride: override)
        }
        nextInFlightID &+= 1
        let entry = InFlight(
            id: nextInFlightID, binaryPath: binaryPath, generation: generation,
            overridePath: overridePath, task: task
        )
        inFlight = entry
        Task { await self.finish(entry) }
        return task
    }

    /// Await a registered load and stamp its result — unless a newer load has
    /// already replaced it (a mutation landed mid-load; its snapshot is
    /// authoritative and this now-stale result must not clobber it).
    private func finish(_ entry: InFlight) async {
        let loaded = await entry.task.value
        guard let current = inFlight, current.id == entry.id else { return }
        snapshot = Snapshot(
            entries: loaded,
            generation: entry.generation,
            fetchedAt: Date(),
            overridePath: entry.overridePath,
            binaryPath: entry.binaryPath
        )
        Self.writeMirror(snapshot)
        inFlight = nil
    }

    /// Drop the snapshot. For callers that mutate the model set themselves and
    /// want the next read to be authoritative without waiting on the
    /// generation counter to propagate.
    func invalidate() {
        snapshot = nil
        inFlight = nil
        Self.writeMirror(nil)
    }

    private func snapshotMatchesInputs(
        _ snapshot: Snapshot, binaryPath: String, generation: UInt,
        overridePath: String?
    ) -> Bool {
        snapshot.generation == generation
            && snapshot.overridePath == overridePath
            && snapshot.binaryPath == binaryPath
    }

    private func isFresh(_ snapshot: Snapshot, binary: URL, generation: UInt) -> Bool {
        guard snapshotMatchesInputs(
            snapshot, binaryPath: binary.path, generation: generation,
            overridePath: ModelsFolderPreference.validatedOverrideURL()?.path
        ) else { return false }
        return Date().timeIntervalSince(snapshot.fetchedAt) < Self.ttl
    }
}
